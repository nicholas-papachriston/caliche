//! Lifecycle hooks for intercepting agent operations.

use crate::agent::hooks::permission::PermissionDecision;
use crate::agent::tools::tool::ToolResult;
use async_trait::async_trait;
use beef::Cow;
use serde_json::Value;

/// Summary of a tool call for completion hooks (tool name + input summary).
#[derive(Debug, Clone)]
pub struct ToolCallSummary {
    /// Tool name (e.g. "bash", "fs")
    pub tool_name: String,
    /// Command or path for display (e.g. bash command, file path)
    pub input_summary: String,
}

/// Shared context for the current turn used by hook runners.
#[derive(Debug, Clone)]
pub struct TurnContext<'a> {
    /// Session identifier
    pub session_id: &'a str,
    /// Current turn count
    pub turn_count: usize,
    /// Workspace root if known
    pub workspace_root: Option<&'a std::path::Path>,
}

/// Context passed to completion hooks when the LLM signals intent to stop (no tool calls).
#[derive(Debug, Clone)]
pub struct CompletionContext<'a> {
    /// Session identifier
    pub session_id: &'a str,
    /// Current turn count
    pub turn_count: usize,
    /// Recent tool calls from this session (last N)
    pub recent_tool_calls: &'a [ToolCallSummary],
    /// Message history length
    pub message_history_len: usize,
    /// Number of pre-completion injections already applied this session
    pub injections_this_session: usize,
    /// Last user message text in the current history, if available.
    pub last_user_message: Option<&'a str>,
    /// Last assistant response text from the current step, if available.
    pub last_assistant_response: Option<&'a str>,
}

/// Result of a completion hook: allow return or inject and continue.
#[derive(Debug, Clone)]
pub enum CompletionHookResult {
    /// Proceed with returning the final response.
    Allow,
    /// Inject the given system message and continue the agent loop for another iteration.
    InjectAndContinue {
        /// System message to prepend (verification prompt, etc.)
        system_message: String,
    },
}

/// Context passed to session-start hooks.
#[derive(Debug, Clone)]
pub struct SessionStartContext<'a> {
    /// Session identifier
    pub session_id: &'a str,
    /// Workspace root if known
    pub workspace_root: Option<&'a std::path::Path>,
}

/// Result of a session-start hook: optional content to prepend to system prompt.
#[derive(Debug, Clone)]
pub struct SessionStartHookResult {
    /// Content to prepend to the system prompt (e.g. [Environment] section)
    pub system_prepend: Option<String>,
}

impl SessionStartHookResult {
    /// No injection
    #[must_use]
    pub const fn none() -> Self {
        Self {
            system_prepend: None,
        }
    }

    /// Prepend the given content to the system prompt
    pub fn prepend(s: impl Into<String>) -> Self {
        Self {
            system_prepend: Some(s.into()),
        }
    }
}

impl Default for SessionStartHookResult {
    fn default() -> Self {
        Self::none()
    }
}

/// Context passed to hooks before each LLM call (for context injections).
#[derive(Debug, Clone)]
pub struct PreLlmContext<'a> {
    /// Session identifier
    pub session_id: &'a str,
    /// Current turn count
    pub turn_count: usize,
}

/// Context passed to hooks during tool execution
#[derive(Debug, Clone)]
pub struct HookContext<'a> {
    /// Name of the tool being called
    pub tool_name: Cow<'a, str>,
    /// Input arguments for the tool
    pub tool_input: &'a Value,
    /// Unique ID for this tool use
    pub tool_use_id: Cow<'a, str>,
}

impl<'a> HookContext<'a> {
    /// Create a new hook context
    pub fn new(
        tool_name: impl Into<Cow<'a, str>>,
        tool_input: &'a Value,
        tool_use_id: impl Into<Cow<'a, str>>,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            tool_input,
            tool_use_id: tool_use_id.into(),
        }
    }

    /// Return owned copies of the context fields (for storage or FFI).
    #[must_use]
    pub fn to_owned(&self) -> (String, Value, String) {
        (
            self.tool_name.clone().into_owned(),
            self.tool_input.clone(),
            self.tool_use_id.clone().into_owned(),
        )
    }
}

/// Result returned by lifecycle hooks
#[derive(Debug, Clone)]
pub struct HookResult {
    /// Permission decision for the operation
    pub permission_decision: PermissionDecision,
    /// Optional additional output from the hook
    pub hook_specific_output: Option<Value>,
}

impl HookResult {
    /// Create an allow result
    #[must_use]
    pub const fn allow() -> Self {
        Self {
            permission_decision: PermissionDecision::Allow,
            hook_specific_output: None,
        }
    }

    /// Create a deny result with a reason
    pub fn deny(reason: impl Into<beef::lean::Cow<'static, str>>) -> Self {
        Self {
            permission_decision: PermissionDecision::Deny {
                reason: reason.into(),
            },
            hook_specific_output: None,
        }
    }

    /// Create a result that requires user approval before continuing
    pub fn require_approval(
        reason: impl Into<beef::lean::Cow<'static, str>>,
        proposed_amendment: Option<Vec<String>>,
    ) -> Self {
        Self {
            permission_decision: PermissionDecision::require_approval(
                reason.into(),
                proposed_amendment,
            ),
            hook_specific_output: None,
        }
    }

    /// Create a result with custom output
    #[must_use]
    pub const fn with_output(permission_decision: PermissionDecision, output: Value) -> Self {
        Self {
            permission_decision,
            hook_specific_output: Some(output),
        }
    }

    /// Check if the result allows the operation
    #[must_use]
    pub fn is_allowed(&self) -> bool {
        self.permission_decision.is_allowed()
    }

    /// Check if the result denies the operation
    #[must_use]
    pub fn is_denied(&self) -> bool {
        self.permission_decision.is_denied()
    }

    /// Check if approval is required
    #[must_use]
    pub fn is_approval_required(&self) -> bool {
        self.permission_decision.is_approval_required()
    }
}

/// Trait for lifecycle hooks that can intercept agent operations
#[async_trait]
pub trait LifecycleHook: Send + Sync + std::fmt::Debug {
    /// Called before a tool is executed
    ///
    /// Return a `HookResult` with a permission decision. If denied, the tool
    /// will not be executed and an error will be returned instead.
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult;

    /// Called after a tool is executed
    ///
    /// This hook cannot deny the operation (it already happened), but can
    /// perform side effects like logging, metrics, or modifying the result.
    async fn on_post_tool_use(&self, _ctx: &HookContext<'_>, _result: &ToolResult) -> HookResult {
        // Default implementation does nothing
        HookResult::allow()
    }

    /// Called when the LLM signals intent to stop (no tool calls) but before the
    /// response is returned. Can inject a verification prompt to force another iteration.
    async fn on_pre_completion(&self, _ctx: &CompletionContext<'_>) -> CompletionHookResult {
        CompletionHookResult::Allow
    }

    /// Called once at session start, before the first user message. Can return
    /// content to prepend to the system prompt (e.g. environment discovery).
    async fn on_session_start(&self, _ctx: &SessionStartContext<'_>) -> SessionStartHookResult {
        SessionStartHookResult::none()
    }

    /// Called before each LLM call. Can return content to prepend to the prompt
    /// (e.g. loop detection warning).
    async fn on_pre_llm_call(&self, _ctx: &PreLlmContext<'_>) -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hook_result_allow_deny_and_approval_predicates() {
        let allow = HookResult::allow();
        assert!(allow.is_allowed());
        assert!(!allow.is_denied());
        assert!(!allow.is_approval_required());
        assert!(allow.hook_specific_output.is_none());

        let deny = HookResult::deny("blocked");
        assert!(deny.is_denied());
        assert!(!deny.is_allowed());
        assert!(!deny.is_approval_required());

        let approval = HookResult::require_approval("needs review", Some(vec!["edit".into()]));
        assert!(approval.is_approval_required());
        assert!(!approval.is_allowed());
        assert!(!approval.is_denied());
    }

    #[test]
    fn hook_result_with_output_preserves_decision_and_payload() {
        let result = HookResult::with_output(PermissionDecision::Allow, json!({"note": "ok"}));
        assert!(result.is_allowed());
        assert_eq!(result.hook_specific_output, Some(json!({"note": "ok"})));
    }
}
