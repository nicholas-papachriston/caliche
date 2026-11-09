//! Turn outcome quality detection and metrics helpers.

use crate::core::state::ReplQualityMetrics;
use crate::core::traits::OutputRenderer;
use caliche::agent::hooks::CONTINUE_TASK_PROMPT;
use caliche::messages::Message;

/// Prefix-compatible with Caliche steering skip (`STREAM_TIMEOUT_CONTINUATION_PREFIX`).
pub(crate) const TIMEOUT_CONTINUATION_MESSAGE: &str = "The previous response was interrupted by a stream timeout. \
     The conversation history (including all tool results) is still intact. \
     Continue from where you left off. Do not repeat completed work — \
     pick up at the next unfinished step.";

pub(crate) const LIMIT_CONTINUE_STATUS: &str =
    "Iteration limit reached before the task was complete.";
pub(crate) const LIMIT_AUTO_CONTINUE_STATUS: &str = "Continuing remaining work.";
pub(crate) const LIMIT_MANUAL_CONTINUE_STATUS: &str =
    "Type continue or a follow-up message to keep going.";
pub(crate) const NARRATION_ONLY_STATUS: &str =
    "(response ended before acting — try 'continue' or ask for explicit next actions)";

/// What the REPL should do after an incomplete successful turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum IncompleteTurnAction {
    None,
    AutoContinue,
    InjectContinue,
}

/// Return true when the model's response is pure narration with no tool calls —
/// a sentence like "I'll check X and do Y." that announces an intent without
/// acting on it.  Used to surface a helpful nudge to the user.
pub(crate) fn is_narration_only(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    // Only flag short responses (< 300 chars) — long prose almost certainly
    // contains real content, not just an announcement.
    if t.len() >= 300 {
        return false;
    }
    // Must look like a single short passage, not a markdown block or list.
    if t.contains('\n') && t.lines().count() > 3 {
        return false;
    }
    let lower = t.to_lowercase();
    // Check for canonical narration openers the system prompt forbids.
    let narration_prefixes = [
        "i'll ",
        "i will ",
        "let me ",
        "i need to ",
        "i'm going to ",
        "i'll check",
        "i'll look",
        "i'll start",
        "i'll begin",
        "i can see",
        "i'll now",
        "i'll first",
        "i'll use",
    ];
    narration_prefixes.iter().any(|p| lower.starts_with(p))
        // Also flag text that ends with a colon or ellipsis (cut-off intent).
        || t.ends_with(':')
        || t.ends_with("...")
}

/// Accumulate per-turn tool-call counters into session quality metrics.
pub(crate) fn record_tool_call_metrics(
    metrics: &mut ReplQualityMetrics,
    tool_calls_started: u64,
    tool_calls_succeeded: u64,
    tool_calls_failed: u64,
    tool_retries_attempted: u64,
    tool_retries_succeeded: u64,
) {
    metrics.tool_calls_started += tool_calls_started;
    metrics.tool_calls_succeeded += tool_calls_succeeded;
    metrics.tool_calls_failed += tool_calls_failed;
    metrics.tool_retries_attempted += tool_retries_attempted;
    metrics.tool_retries_succeeded += tool_retries_succeeded;
}

/// Surface empty, narration-only, and iteration-limit outcomes on every REPL path.
pub(crate) fn surface_incomplete_turn<R: OutputRenderer>(
    renderer: &mut R,
    metrics: &mut ReplQualityMetrics,
    text: &str,
    tools_used: bool,
    limit_reached: bool,
    max_iterations: Option<usize>,
    allow_auto_continue: bool,
) -> IncompleteTurnAction {
    if limit_reached {
        metrics.iteration_limits_reached += 1;
        match max_iterations {
            Some(limit) => {
                renderer.show_status(&format!("{LIMIT_CONTINUE_STATUS} Cap was {limit}."))
            }
            None => renderer.show_status(LIMIT_CONTINUE_STATUS),
        }
        if allow_auto_continue {
            renderer.show_status(LIMIT_AUTO_CONTINUE_STATUS);
            IncompleteTurnAction::AutoContinue
        } else {
            renderer.show_status(LIMIT_MANUAL_CONTINUE_STATUS);
            IncompleteTurnAction::InjectContinue
        }
    } else if text.trim().is_empty() && !tools_used {
        renderer.show_status("(no response)");
        IncompleteTurnAction::None
    } else if !tools_used && is_narration_only(text) {
        metrics.narration_only_responses += 1;
        renderer.show_status(NARRATION_ONLY_STATUS);
        IncompleteTurnAction::None
    } else {
        IncompleteTurnAction::None
    }
}

/// Persist a continuation user message so the next turn can resume.
pub(crate) fn inject_continue_prompt(messages: &mut Vec<Message>) {
    messages.push(Message::user(CONTINUE_TASK_PROMPT));
}

/// Classification of a failed message turn for notify and metrics.
pub(crate) struct TurnErrorOutcome {
    pub is_cancelled: bool,
    pub is_timeout: bool,
    pub continuation_injected: bool,
}

fn displayed_turn_error(error: &anyhow::Error) -> String {
    let text = format!("{error:#}");
    if let Some(index) = text.find("Turn failed:") {
        return text[index..].trim().to_string();
    }
    format!("Error: {text}")
}

/// Render turn errors, update quality metrics, and inject timeout continuation when needed.
pub(crate) fn handle_turn_error<R: OutputRenderer>(
    renderer: &mut R,
    metrics: &mut ReplQualityMetrics,
    messages: &mut Vec<Message>,
    error: &anyhow::Error,
) -> TurnErrorOutcome {
    renderer.clear_thinking_indicator();
    let err_text = error.to_string();
    let is_cancelled = err_text.contains("cancelled by user");
    let stream_error = error.downcast_ref::<caliche::agent::runtime::StreamError>();
    let is_timeout =
        stream_error.is_some_and(|se| se.is_timeout()) || err_text.contains("Stream timed out");

    let mut continuation_injected = false;

    if is_cancelled {
        metrics.cancellations += 1;
        renderer.show_status("Cancelled.");
    } else if is_timeout {
        renderer.show_error(&format!("Error: {:#}", error));
        renderer.show_status(
            "Stream timed out. Conversation history is preserved. \
             Type 'continue' or a follow-up message to resume.",
        );
        if let Some(caliche::agent::runtime::StreamError::Timeout { partial_text, .. }) =
            stream_error
            && !partial_text.trim().is_empty()
        {
            messages.push(Message::assistant(partial_text));
        }
        messages.push(Message::user(TIMEOUT_CONTINUATION_MESSAGE));
        continuation_injected = true;
    } else {
        if err_text.contains("Loop detected: model repeated") {
            metrics.text_loop_detections += 1;
        }
        if err_text.contains("Tool loop detected:") {
            metrics.tool_loop_detections += 1;
        }
        renderer.show_error(&displayed_turn_error(error));
    }

    TurnErrorOutcome {
        is_cancelled,
        is_timeout,
        continuation_injected,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        IncompleteTurnAction, TIMEOUT_CONTINUATION_MESSAGE, handle_turn_error, is_narration_only,
        surface_incomplete_turn,
    };
    use crate::core::slash_commands::SlashCommandResult;
    use crate::core::state::ReplQualityMetrics;
    use crate::core::traits::OutputRenderer;
    use caliche::agent::hooks::CONTINUE_TASK_PROMPT;
    use caliche::agent::runtime::StreamError;
    use caliche::messages::TokenUsage;

    struct VecRenderer {
        status: Vec<String>,
        errors: Vec<String>,
    }

    impl VecRenderer {
        fn new() -> Self {
            Self {
                status: Vec::new(),
                errors: Vec::new(),
            }
        }
    }

    impl OutputRenderer for VecRenderer {
        fn handle_command_result(&mut self, _result: SlashCommandResult) {}
        fn render_chunk(&mut self, _text: &str) {}
        fn render_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
        fn render_tool_complete(&mut self, _name: &str, _result: &str) {}
        fn render_tool_error(&mut self, _name: &str, _error: &str) {}
        fn render_thinking(&mut self, _content: &str) {}
        fn render_token_usage(&mut self, _usage: &caliche::messages::TokenUsage) {}
        fn show_error(&mut self, message: &str) {
            self.errors.push(message.to_string());
        }
        fn show_status(&mut self, message: &str) {
            self.status.push(message.to_string());
        }
        fn clear(&mut self) {}
    }

    #[test]
    fn turn_failure_is_shown_without_an_error_prefix() {
        let mut renderer = VecRenderer::new();
        let mut metrics = ReplQualityMetrics::default();
        let mut messages = Vec::new();
        let error = anyhow::anyhow!("Turn failed: anthropic 400: tool_result without tool_use");
        handle_turn_error(&mut renderer, &mut metrics, &mut messages, &error);
        assert_eq!(
            renderer.errors,
            vec!["Turn failed: anthropic 400: tool_result without tool_use".to_string()]
        );
    }

    #[test]
    fn timeout_continuation_starts_with_steering_skip_prefix() {
        assert!(
            TIMEOUT_CONTINUATION_MESSAGE
                .starts_with("The previous response was interrupted by a stream timeout.")
        );
    }

    #[test]
    fn narration_only_flags_short_intent_sentences() {
        assert!(is_narration_only("I'll check the auth module next."));
        assert!(!is_narration_only(
            "The auth module already validates tokens in middleware.rs."
        ));
        assert!(!is_narration_only(""));
    }

    #[test]
    fn limit_reached_auto_continues_once() {
        let mut renderer = VecRenderer::new();
        let mut metrics = ReplQualityMetrics::default();
        let action = surface_incomplete_turn(
            &mut renderer,
            &mut metrics,
            "partial",
            true,
            true,
            Some(100),
            true,
        );
        assert_eq!(action, IncompleteTurnAction::AutoContinue);
        assert_eq!(metrics.iteration_limits_reached, 1);
        assert!(renderer.status.iter().any(|s| s.contains("Cap was 100")));
    }

    #[test]
    fn limit_reached_without_budget_asks_the_user_to_continue() {
        let mut renderer = VecRenderer::new();
        let mut metrics = ReplQualityMetrics::default();
        let action = surface_incomplete_turn(
            &mut renderer,
            &mut metrics,
            "partial",
            true,
            true,
            Some(100),
            false,
        );
        assert_eq!(action, IncompleteTurnAction::InjectContinue);
        assert!(CONTINUE_TASK_PROMPT.contains("Continue from where you left off"));
    }

    #[test]
    fn timeout_error_keeps_partial_assistant_text() {
        let mut renderer = VecRenderer::new();
        let mut metrics = ReplQualityMetrics::default();
        let mut messages = Vec::new();
        let error = anyhow::Error::new(StreamError::Timeout {
            elapsed_secs: 900,
            partial_text: "halfway through the edit".to_string(),
            usage: TokenUsage::default(),
            tool_calls_completed: 3,
        });
        let outcome = handle_turn_error(&mut renderer, &mut metrics, &mut messages, &error);
        assert!(outcome.is_timeout);
        assert!(outcome.continuation_injected);
        assert_eq!(messages.len(), 2);
        assert!(matches!(
            messages[0].role,
            caliche::messages::MessageRole::Assistant
        ));
        assert!(matches!(
            messages[1].role,
            caliche::messages::MessageRole::User
        ));
    }
}
