//! Fit a stored session transcript into the next model prompt.
//!
//! `session resume` reloads the saved conversation before the first new
//! prompt. When the transcript fits, it is copied verbatim. When it does not,
//! older turns become one summary and the latest messages stay intact,
//! including the last tool actions.

use crate::agent::memory::{
    ApproximateTokenCounter, BufferMemory, BufferMemoryConfig, MemoryStrategy, TokenCounter,
};
use crate::agent::service::AgentService;
use crate::messages::{Message, MessageRole};
use anyhow::{Context, Result};
use std::sync::Arc;

/// Token budget for the reconstituted resume prompt.
pub const RESUME_CONTEXT_TOKEN_BUDGET: usize = 6_000;

const SUMMARY_TOKEN_RESERVE: usize = 700;
const RECENT_MESSAGE_FLOOR: usize = 4;
const MAX_KEPT_MESSAGE_CHARS: usize = 1_200;
const SUMMARY_LINE_CHARS: usize = 180;
const PREFIX_TURN_LINES: usize = 12;
const TOOL_ACTION_LIMIT: usize = 8;
const TOOL_ARG_CHARS: usize = 80;

/// Transcript prepared for the next model request.
#[derive(Debug, Clone)]
pub struct FittedResumeContext {
    /// Messages to place in harness history and service memory.
    pub messages: Vec<Message>,
    /// `true` when older turns were folded into a summary message.
    pub summarized: bool,
}

/// Copy `stored` when it fits in `max_tokens`. Otherwise prepend a summary.
///
/// Stored `metadata.reasoning_content` is removed from every replayed
/// message. The token estimate counts visible text only, so leaving the
/// field in place would resend an unbounded chain-of-thought. The saved
/// transcript still has it for export.
#[must_use]
pub fn fit_resume_messages(stored: &[Message], max_tokens: usize) -> FittedResumeContext {
    if stored.is_empty() {
        return FittedResumeContext {
            messages: Vec::new(),
            summarized: false,
        };
    }
    let counter = ApproximateTokenCounter::default();
    if counter.estimate_messages_tokens(stored) <= max_tokens {
        return FittedResumeContext {
            messages: stored
                .iter()
                .map(Message::without_reasoning_content)
                .collect(),
            summarized: false,
        };
    }
    let tail_budget = max_tokens.saturating_sub(SUMMARY_TOKEN_RESERVE).max(256);
    let tail = select_tail(stored, tail_budget, RECENT_MESSAGE_FLOOR);
    let prefix_len = stored.len().saturating_sub(tail.len());
    let mut messages = Vec::with_capacity(tail.len().saturating_add(1));
    messages.push(Message::system(render_resume_summary(
        &stored[..prefix_len],
        stored,
    )));
    messages.extend(tail);
    FittedResumeContext {
        messages,
        summarized: true,
    }
}

/// Install `messages` as the service memory the next chat reads.
///
/// Replaces any existing strategy with a buffer sized to hold this transcript
/// so a smaller default window cannot drop the restored turns.
///
/// # Errors
///
/// Returns an error when the memory strategy rejects the messages.
pub async fn seed_service_resume_context(
    service: &mut AgentService,
    messages: Vec<Message>,
) -> Result<()> {
    let counter = Arc::new(ApproximateTokenCounter::default());
    let estimated = counter.estimate_messages_tokens(&messages);
    let mut config = BufferMemoryConfig::default()
        .with_max_tokens(estimated.saturating_add(1_024).max(8_192))
        .with_min_messages(messages.len());
    config.summary_placeholder = false;
    let mut memory = BufferMemory::new(config, counter);
    if !messages.is_empty() {
        memory
            .add_messages(messages)
            .await
            .context("seed resume conversation into agent memory")?;
    }
    service.set_memory_strategy(Some(Box::new(memory)));
    Ok(())
}

fn select_tail(messages: &[Message], tail_budget: usize, floor: usize) -> Vec<Message> {
    let counter = ApproximateTokenCounter::default();
    let mut chosen = Vec::new();
    let mut tokens = 0usize;
    for message in messages.iter().rev() {
        let bounded = bounded_message(message);
        let cost = counter.estimate_message_tokens(&bounded);
        let fits = tokens.saturating_add(cost) <= tail_budget;
        let need_floor = chosen.len() < floor;
        if !fits && !need_floor {
            break;
        }
        tokens = tokens.saturating_add(cost);
        chosen.push(bounded);
    }
    chosen.reverse();
    chosen
}

fn bounded_message(message: &Message) -> Message {
    let text = message.extract_text();
    let mut bounded = if text.chars().count() <= MAX_KEPT_MESSAGE_CHARS {
        message.clone()
    } else {
        let clipped = clip_chars(&text, MAX_KEPT_MESSAGE_CHARS);
        let tools = tool_names(message);
        let body = if tools.is_empty() {
            clipped
        } else {
            format!("{clipped} [tools: {tools}]")
        };
        Message {
            role: copy_role(&message.role),
            content: crate::messages::MessageContent::text(body),
            metadata: message.metadata.clone(),
        }
    };
    bounded.clear_reasoning_content();
    bounded
}

fn render_resume_summary(prefix: &[Message], full: &[Message]) -> String {
    let mut lines = Vec::new();
    lines.push(
        "Prior session context restored on resume. Older turns are summarized; the latest messages follow verbatim."
            .to_string(),
    );
    let turns = prefix_turn_lines(prefix);
    if !turns.is_empty() {
        lines.push("Earlier turns:".to_string());
        lines.extend(turns);
    }
    let actions = last_tool_actions(full);
    if !actions.is_empty() {
        lines.push("Last tool actions:".to_string());
        lines.extend(actions);
    }
    lines.join("\n")
}

fn prefix_turn_lines(prefix: &[Message]) -> Vec<String> {
    let start = prefix.len().saturating_sub(PREFIX_TURN_LINES);
    prefix[start..].iter().map(turn_line).collect()
}

fn turn_line(message: &Message) -> String {
    let role = role_label(&message.role);
    let mut text = clip_chars(&message.extract_text(), SUMMARY_LINE_CHARS);
    let tools = tool_names(message);
    if !tools.is_empty() {
        if text.is_empty() {
            text = format!("[tools: {tools}]");
        } else {
            text = format!("{text} [tools: {tools}]");
        }
    }
    format!("- {role}: {text}")
}

fn last_tool_actions(messages: &[Message]) -> Vec<String> {
    let mut lines = Vec::new();
    for message in messages {
        collect_tool_actions(message, &mut lines);
    }
    if lines.len() > TOOL_ACTION_LIMIT {
        let drop = lines.len() - TOOL_ACTION_LIMIT;
        lines.drain(..drop);
    }
    lines
}

fn collect_tool_actions(message: &Message, lines: &mut Vec<String>) {
    if let Some(calls) = message.content.as_tool_calls() {
        for call in calls {
            let args = clip_chars(&call.arguments_json(), TOOL_ARG_CHARS);
            lines.push(format!("- {} {args}", call.name_str()));
        }
        return;
    }
    if message.role == MessageRole::Tool {
        let text = clip_chars(&message.extract_text(), SUMMARY_LINE_CHARS);
        if !text.is_empty() {
            lines.push(format!("- {text}"));
        }
    }
}

fn tool_names(message: &Message) -> String {
    let Some(calls) = message.content.as_tool_calls() else {
        return String::new();
    };
    calls
        .iter()
        .map(|call| call.name_str())
        .collect::<Vec<_>>()
        .join(", ")
}

fn copy_role(role: &MessageRole) -> MessageRole {
    match role {
        MessageRole::System => MessageRole::System,
        MessageRole::User => MessageRole::User,
        MessageRole::Assistant => MessageRole::Assistant,
        MessageRole::Tool => MessageRole::Tool,
    }
}

fn role_label(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "System",
        MessageRole::User => "User",
        MessageRole::Assistant => "Assistant",
        MessageRole::Tool => "Tool",
    }
}

fn clip_chars(text: &str, max_chars: usize) -> String {
    let trimmed = text.trim();
    let count = trimmed.chars().count();
    if count <= max_chars {
        return trimmed.to_string();
    }
    let clipped: String = trimmed.chars().take(max_chars).collect();
    format!("{clipped}…")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{MessageContent, ToolCallInfo};
    use std::sync::Arc;

    fn tool_message(name: &str, body: &str) -> Message {
        Message {
            role: MessageRole::Tool,
            content: MessageContent::Structured {
                text: Some(Arc::new(format!("{name}: {body}"))),
                tool_calls: Some(vec![ToolCallInfo::from_json(
                    "call-1",
                    name,
                    r#"{"path":"parser.rs"}"#,
                )]),
                tool_call_id: Some(beef::lean::Cow::borrowed("call-1")),
            },
            metadata: None,
        }
    }

    #[test]
    fn small_transcript_is_copied_verbatim() {
        let stored = vec![
            Message::user("rename parser.rs"),
            Message::assistant("Renamed parser.rs with the fs tool"),
        ];
        let fitted = fit_resume_messages(&stored, RESUME_CONTEXT_TOKEN_BUDGET);
        assert!(!fitted.summarized);
        assert_eq!(fitted.messages.len(), 2);
        assert_eq!(
            fitted.messages[1].extract_text(),
            "Renamed parser.rs with the fs tool"
        );
    }

    #[test]
    fn oversized_transcript_keeps_last_turn_and_tool_actions() {
        let mut stored = vec![tool_message("fs", "renamed parser.rs")];
        for index in 0..30 {
            stored.push(Message::user(format!(
                "filler question {index} {}",
                "x".repeat(500)
            )));
            stored.push(Message::assistant(format!(
                "filler answer {index} {}",
                "y".repeat(500)
            )));
        }
        stored.push(Message::user("what did you just do"));
        stored.push(Message::assistant("I renamed parser.rs"));

        let fitted = fit_resume_messages(&stored, 900);
        assert!(fitted.summarized);
        let blob = fitted
            .messages
            .iter()
            .map(Message::extract_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(blob.contains("what did you just do"));
        assert!(blob.contains("I renamed parser.rs"));
        assert!(blob.contains("Last tool actions"));
        assert!(blob.contains("fs"));
        assert!(blob.contains("parser.rs"));
    }

    /// 32 visible bytes estimate as 13 tokens. A 25,500-char trace must not ride along.
    #[test]
    fn resume_strips_reasoning_outside_the_visible_budget() {
        use crate::agent::memory::{ApproximateTokenCounter, TokenCounter};

        let visible = "short visible answer text here!!";
        assert_eq!(visible.len(), 32);
        let reasoning = "r".repeat(25_500);
        let stored = vec![
            Message::user("hi"),
            Message::assistant_with_reasoning(visible, &reasoning),
        ];
        let counter = ApproximateTokenCounter::default();
        assert_eq!(
            counter.estimate_message_tokens(&stored[1]),
            13,
            "the estimate ignores reasoning, so a verbatim copy would resend it"
        );
        assert!(counter.estimate_messages_tokens(&stored) < RESUME_CONTEXT_TOKEN_BUDGET);

        let fitted = fit_resume_messages(&stored, RESUME_CONTEXT_TOKEN_BUDGET);
        assert!(
            !fitted.summarized,
            "visible text fits, which is why the unstripped copy used to be sent"
        );
        assert!(
            fitted
                .messages
                .iter()
                .all(|message| message.reasoning_content().is_none())
        );
        let wire = serde_json::to_string(&fitted.messages).expect("fitted json");
        assert!(wire.contains(visible));
        assert!(
            !wire.contains(&reasoning[..64]),
            "thin-client history POST uses this JSON"
        );
        assert!(
            wire.matches('r').count() < 64,
            "fitted replay must not carry the 25,500-char trace"
        );
    }

    #[test]
    fn summarized_resume_tail_also_strips_reasoning() {
        let marker = "Q".repeat(25_500);
        let mut stored = Vec::new();
        for index in 0..30 {
            stored.push(Message::user(format!(
                "filler question {index} {}",
                "x".repeat(500)
            )));
            stored.push(Message::assistant(format!(
                "filler answer {index} {}",
                "y".repeat(500)
            )));
        }
        stored.push(Message::assistant_with_reasoning("kept answer", &marker));

        let fitted = fit_resume_messages(&stored, 900);
        assert!(fitted.summarized);
        assert!(
            fitted
                .messages
                .iter()
                .any(|message| message.extract_text().contains("kept answer"))
        );
        assert!(
            fitted
                .messages
                .iter()
                .all(|message| message.reasoning_content().is_none())
        );
        let wire = serde_json::to_string(&fitted.messages).expect("fitted json");
        assert!(!wire.contains(&marker[..80]));
    }
}
