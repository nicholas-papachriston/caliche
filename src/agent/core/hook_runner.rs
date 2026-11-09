use super::message_history::arc_prepend_message;
use crate::agent::hooks::{
    CompletionContext, CompletionHookResult, PreLlmContext, SessionStartContext, TurnContext,
};
use crate::messages::{Message, MessageRole};
use std::sync::Arc;

impl super::Agent {
    /// Run `on_session_start` hooks and inject any system prepend into position 1 of the history.
    pub(super) async fn run_session_start_hooks(
        &self,
        mut messages: Arc<Vec<Message>>,
        turn: &TurnContext<'_>,
    ) -> Arc<Vec<Message>> {
        let ctx = SessionStartContext {
            session_id: turn.session_id,
            workspace_root: turn.workspace_root,
        };
        for hook in self.hooks.get_matching("*") {
            let result = hook.on_session_start(&ctx).await;
            if let Some(prepend) = result.system_prepend {
                messages =
                    arc_prepend_message(messages, 1, Message::new(MessageRole::System, prepend));
                break;
            }
        }
        messages
    }

    /// Run `on_pre_llm_call` hooks and prepend any injected system messages.
    pub(super) async fn run_pre_llm_hooks(
        &self,
        messages: Arc<Vec<Message>>,
        turn: &TurnContext<'_>,
    ) -> Arc<Vec<Message>> {
        let pre_llm_ctx = PreLlmContext {
            session_id: turn.session_id,
            turn_count: turn.turn_count,
        };
        let mut injections: Vec<String> = Vec::new();
        for hook in self.hooks.get_matching("*") {
            if let Some(injection) = hook.on_pre_llm_call(&pre_llm_ctx).await {
                injections.push(injection);
            }
        }
        if injections.is_empty() {
            messages
        } else {
            let mut vec = (*messages).clone();
            let combined = injections.join("\n\n");
            vec.insert(
                1.min(vec.len()),
                Message::new(MessageRole::System, combined),
            );
            Arc::new(vec)
        }
    }

    /// Run `on_pre_completion` hooks and return the first injection message if any hook
    /// requests `InjectAndContinue`.
    pub(super) async fn run_completion_hooks<'a>(
        &self,
        ctx: &CompletionContext<'a>,
    ) -> Option<String> {
        for hook in self.hooks.completion_hooks() {
            match hook.on_pre_completion(ctx).await {
                CompletionHookResult::Allow => {}
                CompletionHookResult::InjectAndContinue { system_message } => {
                    return Some(system_message);
                }
            }
        }
        None
    }
}
