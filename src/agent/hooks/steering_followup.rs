//! Steering and follow-up message provider
//!
//! Allows the runtime to inject user messages at two points:
//! - **Steering**: after a tool execution, before the next LLM call (interrupt mid-run).
//! - **Follow-up**: when the agent would otherwise stop (no more tool calls); messages are appended and the agent continues.
//!
//! Implementations typically use a queue (e.g. CLI input or WebSocket) that is checked by the runtime.

use crate::messages::Message;
use async_trait::async_trait;

/// Outer-turn prompt used when a coding/harness task hits the inner iteration cap.
/// Workspace RAG must skip this string: history already has the original task.
pub const CONTINUE_TASK_PROMPT: &str = "Continue from where you left off. Do not restart the original task. Finish remaining work and return the result.";

/// Prefix of the REPL timeout continuation injected into history.
pub const STREAM_TIMEOUT_CONTINUATION_PREFIX: &str =
    "The previous response was interrupted by a stream timeout.";

/// Queries that should not trigger hybrid workspace search on a follow-up turn.
#[must_use]
pub fn is_steering_only_query(query: &str) -> bool {
    let query = query.trim();
    query.is_empty()
        || query.starts_with(CONTINUE_TASK_PROMPT)
        || query.starts_with(STREAM_TIMEOUT_CONTINUATION_PREFIX)
}

/// Provider for steering and follow-up messages during a chat run.
///
/// - **Steering**: called after each tool execution; if non-empty, remaining tools in the current batch can be skipped and the returned messages are injected before the next LLM call.
/// - **Follow-up**: called when the agent would stop (no pending tool calls); if non-empty, messages are appended to history and the agent runs another turn.
#[async_trait]
pub trait SteeringFollowUpProvider: Send + Sync {
    /// Messages to inject mid-run (after tool execution). If non-empty, the runtime may skip remaining tool calls and inject these before the next LLM call.
    async fn get_steering_messages(&self) -> Vec<Message> {
        Vec::new()
    }

    /// Messages to process after the agent would stop. If non-empty, they are appended to history and the agent runs another turn.
    async fn get_follow_up_messages(&self) -> Vec<Message> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{CONTINUE_TASK_PROMPT, STREAM_TIMEOUT_CONTINUATION_PREFIX, is_steering_only_query};

    #[test]
    fn skips_empty_and_harness_continuation_prompts() {
        assert!(is_steering_only_query(""));
        assert!(is_steering_only_query("   "));
        assert!(is_steering_only_query(CONTINUE_TASK_PROMPT));
        assert!(is_steering_only_query(&format!(
            "{STREAM_TIMEOUT_CONTINUATION_PREFIX} The conversation history is intact."
        )));
        assert!(!is_steering_only_query(
            "fix the failing indexer test in workspace_rag_wiring"
        ));
        assert!(!is_steering_only_query(
            "Verification failed after your edits. Do not restart the original task."
        ));
    }
}
