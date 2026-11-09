//! Completion guard to reduce narration-only assistant turns.
//!
//! If the assistant appears to "announce" work (e.g. "I'll check...") instead of
//! doing it, inject a brief system nudge to execute tools immediately.

use super::lifecycle::{
    CompletionContext, CompletionHookResult, HookContext, HookResult, LifecycleHook,
};
use async_trait::async_trait;

/// Config for narration-action guard behavior.
#[derive(Debug, Clone)]
pub struct ActionNarrationGuardConfig {
    /// Maximum number of injections in a single session.
    pub max_injections_per_session: usize,
}

impl Default for ActionNarrationGuardConfig {
    fn default() -> Self {
        Self {
            max_injections_per_session: 1,
        }
    }
}

#[derive(Debug)]
pub struct ActionNarrationGuardMiddleware {
    config: ActionNarrationGuardConfig,
}

impl ActionNarrationGuardMiddleware {
    #[must_use]
    pub const fn new(config: ActionNarrationGuardConfig) -> Self {
        Self { config }
    }

    fn looks_actionable_user_request(text: &str) -> bool {
        let lower = text.to_lowercase();
        [
            "fix",
            "implement",
            "update",
            "refactor",
            "change",
            "run",
            "debug",
            "investigate",
            "check",
            "build",
            "test",
            "search",
            "find",
            "create",
            "edit",
            "add",
            "remove",
        ]
        .iter()
        .any(|kw| lower.contains(kw))
    }

    fn looks_like_narration_only(text: &str) -> bool {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return false;
        }

        let lower = trimmed.to_lowercase();
        let starts_with_intent = [
            "i'll ",
            "i will ",
            "let me ",
            "now i'll ",
            "next i'll ",
            "i'm going to ",
            "i am going to ",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix));

        if !starts_with_intent {
            return false;
        }

        // Avoid interfering with explicit clarification requests.
        !lower.ends_with('?')
    }

    const NUDGE: &'static str = r#"Do not narrate planned actions. Execute the next concrete step now.
- If tools are needed, call them in this response (prefer parallel independent calls)
- If no tools are needed, provide the concrete answer directly
- Avoid "I'll...", "Let me...", or similar intent-only phrasing"#;
}

#[async_trait]
impl LifecycleHook for ActionNarrationGuardMiddleware {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::allow()
    }

    async fn on_pre_completion(&self, ctx: &CompletionContext<'_>) -> CompletionHookResult {
        if ctx.injections_this_session >= self.config.max_injections_per_session {
            return CompletionHookResult::Allow;
        }

        let Some(last_user_message) = ctx.last_user_message else {
            return CompletionHookResult::Allow;
        };
        if !Self::looks_actionable_user_request(last_user_message) {
            return CompletionHookResult::Allow;
        }

        let Some(last_assistant_response) = ctx.last_assistant_response else {
            return CompletionHookResult::Allow;
        };
        if !Self::looks_like_narration_only(last_assistant_response) {
            return CompletionHookResult::Allow;
        }

        CompletionHookResult::InjectAndContinue {
            system_message: Self::NUDGE.to_string(),
        }
    }
}
