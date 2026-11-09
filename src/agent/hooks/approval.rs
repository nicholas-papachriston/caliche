//! Approval resolver for tool execution when hooks return `RequireApproval`.
//!
//! Allows the runtime to prompt the user (CLI) or invoke a callback (API) instead of
//! denying when `ExecPolicy` or other hooks return a Prompt/RequireApproval decision.

use async_trait::async_trait;
use futures_util::future::BoxFuture;
use serde_json::Value;

/// Outcome of resolving an approval request
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalOutcome {
    /// Allow the tool to execute
    Allow,
    /// Allow and persist an allow rule for future executions
    AllowAndPersist {
        /// Command prefix to append as allow rule
        rule_prefix: Vec<String>,
    },
    /// Deny the tool execution
    Deny,
}

/// Resolver for approval requests when hooks return `RequireApproval`.
///
/// Implemented by CLI (stdio prompt) or API (callback/webhook). When no resolver
/// is configured, `RequireApproval` is treated as Deny.
#[async_trait]
pub trait ApprovalResolver: Send + Sync {
    /// Resolve an approval request. Block until the user responds or callback completes.
    fn resolve_approval(
        &self,
        tool_name: &str,
        tool_input: &Value,
        reason: &str,
        proposed_amendment: Option<&[String]>,
    ) -> BoxFuture<'_, anyhow::Result<ApprovalOutcome>>;

    /// Resolve approval for one tool call.
    ///
    /// `call_id` is the id already attached to that call. The default ignores
    /// it and forwards to [`Self::resolve_approval`].
    fn resolve_approval_for_call<'a>(
        &'a self,
        tool_name: &'a str,
        tool_input: &'a Value,
        reason: &'a str,
        proposed_amendment: Option<&'a [String]>,
        call_id: &'a str,
    ) -> BoxFuture<'a, anyhow::Result<ApprovalOutcome>> {
        let _ = call_id;
        self.resolve_approval(tool_name, tool_input, reason, proposed_amendment)
    }
}

/// Persister for appending allow rules when user chooses "always".
///
/// When `ApprovalOutcome` is `AllowAndPersist`, the execution layer calls this to
/// persist the rule to config. If no persister is configured, the rule is not
/// persisted (next session will prompt again).
#[async_trait]
pub trait AmendmentPersister: Send + Sync {
    /// Persist an allow rule for the given command prefix.
    fn persist_allow_rule(&self, rule_prefix: &[String]) -> BoxFuture<'_, anyhow::Result<()>>;
}
