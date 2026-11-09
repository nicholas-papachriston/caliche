//! Keep provider transcripts free of orphaned tool calls and tool results.
//!
//! Anthropic rejects a `tool_result` whose `tool_use` was dropped, and a
//! `tool_use` that has no following result. Trimming removes whole turns or
//! whole tool-call groups, including parallel calls.

use crate::agent::memory::TokenCounter;
use crate::messages::{Message, MessageContent, MessageRole};
use std::collections::{HashSet, VecDeque};
use std::ops::Range;

/// Drop orphan tool results and tool calls that have no matching partner.
///
/// Leading non-user messages are removed so a request does not open on a
/// tool result. System messages at the front, including compaction summaries,
/// stay in place.
#[must_use]
pub fn repair_tool_history(messages: Vec<Message>) -> Vec<Message> {
    let mut pending: VecDeque<Message> = messages.into();
    let mut output = Vec::new();
    while let Some(message) = pending.pop_front() {
        if is_assistant_with_tools(&message) {
            let mut results = Vec::new();
            while pending.front().is_some_and(is_tool_result) {
                if let Some(result) = pending.pop_front() {
                    results.push(result);
                }
            }
            push_paired_exchange(&mut output, message, results);
            continue;
        }
        if is_tool_result(&message) {
            continue;
        }
        output.push(message);
    }
    drop_leading_non_user(&mut output);
    output
}

/// Drop oldest turns, then oldest tool groups, until `messages` fit `max_tokens`.
///
/// Groups are never split. `min_messages` is a floor only when the history
/// already fits; an over-budget transcript still drops whole groups.
#[must_use]
pub fn trim_history_preserving_tool_pairs(
    mut messages: Vec<Message>,
    counter: &dyn TokenCounter,
    max_tokens: usize,
    _min_messages: usize,
) -> Vec<Message> {
    let mut steps = 0usize;
    let limit = messages.len().saturating_add(1);
    while counter.estimate_messages_tokens(&messages) > max_tokens && steps < limit {
        steps = steps.saturating_add(1);
        if !drop_one_unit(&mut messages) {
            break;
        }
    }
    repair_tool_history(messages)
}

/// True when every tool call has its results and every result has its call.
#[cfg(test)]
#[must_use]
pub(crate) fn tool_history_is_provider_safe(messages: &[Message]) -> bool {
    let mut index = 0usize;
    while index < messages.len() && matches!(messages[index].role, MessageRole::System) {
        index = index.saturating_add(1);
    }
    if index < messages.len() && !matches!(messages[index].role, MessageRole::User) {
        return false;
    }
    while index < messages.len() {
        if is_assistant_with_tools(&messages[index]) {
            if !exchange_is_paired(messages, &mut index) {
                return false;
            }
            continue;
        }
        if is_tool_result(&messages[index]) {
            return false;
        }
        index = index.saturating_add(1);
    }
    true
}

#[cfg(test)]
fn exchange_is_paired(messages: &[Message], index: &mut usize) -> bool {
    let call_ids = assistant_call_ids(&messages[*index]);
    *index = index.saturating_add(1);
    let mut seen = HashSet::new();
    while *index < messages.len() && is_tool_result(&messages[*index]) {
        let Some(id) = tool_result_id(&messages[*index]) else {
            return false;
        };
        if !call_ids.contains(id) || !seen.insert(id.to_string()) {
            return false;
        }
        *index = index.saturating_add(1);
    }
    seen.len() == call_ids.len()
}

fn push_paired_exchange(output: &mut Vec<Message>, mut assistant: Message, results: Vec<Message>) {
    let result_ids = tool_result_id_set(&results);
    retain_calls(&mut assistant, &result_ids);
    let kept_ids = assistant_call_ids(&assistant);
    if kept_ids.is_empty() {
        if assistant_has_text(&assistant) {
            clear_tool_calls(&mut assistant);
            output.push(assistant);
        }
        return;
    }
    output.push(assistant);
    for result in results {
        if tool_result_id(&result).is_some_and(|id| kept_ids.contains(id)) {
            output.push(result);
        }
    }
}

fn drop_one_unit(messages: &mut Vec<Message>) -> bool {
    if let Some(range) = oldest_turn_with_successor(messages) {
        return drain_unit(messages, range);
    }
    if let Some(range) = oldest_tool_group(messages) {
        return drain_unit(messages, range);
    }
    false
}

fn drain_unit(messages: &mut Vec<Message>, range: Range<usize>) -> bool {
    if range.start >= range.end || range.end > messages.len() {
        return false;
    }
    if range.start == 0 && range.end == messages.len() {
        return false;
    }
    let mut index = range.end;
    let mut removed = 0usize;
    while index > range.start {
        index = index.saturating_sub(1);
        if !matches!(messages[index].role, MessageRole::System) {
            messages.remove(index);
            removed = removed.saturating_add(1);
        }
    }
    removed > 0
}

fn oldest_turn_with_successor(messages: &[Message]) -> Option<Range<usize>> {
    let users = user_positions(messages);
    let start = *users.first()?;
    let end = *users.get(1)?;
    Some(start..end)
}

fn oldest_tool_group(messages: &[Message]) -> Option<Range<usize>> {
    let start = messages.iter().position(is_assistant_with_tools)?;
    let mut end = start.saturating_add(1);
    while end < messages.len() && is_tool_result(&messages[end]) {
        end = end.saturating_add(1);
    }
    Some(start..end)
}

fn user_positions(messages: &[Message]) -> Vec<usize> {
    messages
        .iter()
        .enumerate()
        .filter(|(_, message)| matches!(message.role, MessageRole::User))
        .map(|(index, _)| index)
        .collect()
}

fn drop_leading_non_user(messages: &mut Vec<Message>) {
    let mut index = 0usize;
    while index < messages.len() && matches!(messages[index].role, MessageRole::System) {
        index = index.saturating_add(1);
    }
    while index < messages.len() && !matches!(messages[index].role, MessageRole::User) {
        messages.remove(index);
    }
}

fn is_assistant_with_tools(message: &Message) -> bool {
    matches!(message.role, MessageRole::Assistant)
        && message
            .content
            .as_tool_calls()
            .is_some_and(|calls| !calls.is_empty())
}

fn is_tool_result(message: &Message) -> bool {
    matches!(message.role, MessageRole::Tool)
}

fn tool_result_id(message: &Message) -> Option<&str> {
    message.content.as_tool_call_id()
}

fn tool_result_id_set(results: &[Message]) -> HashSet<String> {
    results
        .iter()
        .filter_map(|message| tool_result_id(message).map(str::to_string))
        .collect()
}

fn assistant_call_ids(message: &Message) -> HashSet<String> {
    message
        .content
        .as_tool_calls()
        .map(|calls| {
            calls
                .iter()
                .map(|call| call.id.as_ref().to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn retain_calls(message: &mut Message, keep: &HashSet<String>) {
    let MessageContent::Structured { tool_calls, .. } = &mut message.content else {
        return;
    };
    let Some(calls) = tool_calls else {
        return;
    };
    calls.retain(|call| keep.contains(call.id.as_ref()));
    if calls.is_empty() {
        *tool_calls = None;
    }
}

fn clear_tool_calls(message: &mut Message) {
    if let MessageContent::Structured { tool_calls, .. } = &mut message.content {
        *tool_calls = None;
    }
}

fn assistant_has_text(message: &Message) -> bool {
    message
        .content
        .as_text()
        .is_some_and(|text| !text.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::memory::ApproximateTokenCounter;
    use crate::messages::ToolCallInfo;
    use std::sync::Arc;

    fn assistant_calls(ids: &[(&str, &str)]) -> Message {
        Message {
            role: MessageRole::Assistant,
            content: MessageContent::Structured {
                text: None,
                tool_calls: Some(
                    ids.iter()
                        .map(|(id, name)| {
                            ToolCallInfo::from_json(*id, *name, r#"{"path":"src/lib.rs"}"#)
                        })
                        .collect(),
                ),
                tool_call_id: None,
            },
            metadata: None,
        }
    }

    fn tool_result(id: &str, body: &str) -> Message {
        Message {
            role: MessageRole::Tool,
            content: MessageContent::Structured {
                text: Some(Arc::new(body.to_string())),
                tool_calls: None,
                tool_call_id: Some(beef::lean::Cow::owned(id.to_string())),
            },
            metadata: None,
        }
    }

    fn parallel_turn(user: &str, id_a: &str, id_b: &str, body: &str) -> Vec<Message> {
        vec![
            Message::user(user),
            assistant_calls(&[(id_a, "fs"), (id_b, "fs")]),
            tool_result(id_a, body),
            tool_result(id_b, body),
        ]
    }

    #[test]
    fn repair_drops_orphan_results_and_unmatched_parallel_calls() {
        let messages = vec![
            tool_result("gone", "orphan result"),
            Message::user("keep going"),
            assistant_calls(&[("keep", "fs"), ("drop", "fs")]),
            tool_result("keep", "file contents"),
        ];
        let repaired = repair_tool_history(messages);
        assert!(tool_history_is_provider_safe(&repaired));
        let blob = repaired
            .iter()
            .map(Message::extract_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!blob.contains("orphan result"));
        assert!(blob.contains("file contents"));
        let calls = repaired[1]
            .content
            .as_tool_calls()
            .expect("paired assistant");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id.as_ref(), "keep");
    }

    #[test]
    fn parallel_tool_turns_crossing_8192_stay_paired() {
        let body = "y".repeat(8_000);
        let mut stored = parallel_turn("turn one", "c1", "c2", &body);
        stored.extend(parallel_turn("turn two", "c3", "c4", &body));
        stored.extend(parallel_turn("turn three", "c5", "c6", &body));
        let counter = ApproximateTokenCounter::default();
        assert!(
            counter.estimate_messages_tokens(&stored) > 8_192,
            "fixture must cross the old session cap"
        );

        let trimmed = trim_history_preserving_tool_pairs(stored, &counter, 8_192, 4);
        assert!(counter.estimate_messages_tokens(&trimmed) <= 8_192);
        assert!(tool_history_is_provider_safe(&trimmed));
        assert!(
            trimmed.iter().any(|message| {
                message
                    .content
                    .as_tool_calls()
                    .is_some_and(|calls| calls.len() == 2)
            }),
            "a parallel tool pair should survive as a group"
        );
    }
}
