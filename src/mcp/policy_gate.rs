//! MCP calls use the same compiled policy snapshot as agent tools.
//!
//! [`Effect::Deny`](caliche_policy::Effect::Deny) blocks the call.
//! [`Effect::Ask`](caliche_policy::Effect::Ask) does not. There is no
//! approval broker on this path. Network URLs still go through egress.

use crate::agent::hooks::policy_shadow::observe_live_outcome;
use crate::agent::hooks::{HookContext, HookRegistry, HookResult};
use crate::gateway::egress::{EgressClass, EgressPolicy, current_policy};
use beef::lean::Cow;
use caliche_policy::{
    Action, Decision, Effect, evaluate_action, evaluate_shadow, tool_name_as_run,
};
use serde_json::Value;

const URL_WALK_DEPTH: u8 = 8;

/// Hard deny for a tool call. `None` means the call may run.
///
/// The call is checked under the name it runs as (see
/// [`tool_name_as_run`]), so `" fs"` gets the `fs` decision.
#[must_use]
pub fn tool_deny_reason(tool: &str, args: &Value) -> Option<String> {
    let tool = tool_name_as_run(tool);
    let decision = evaluate_shadow(tool, args);
    finish_tool(tool, args, &decision)
}

/// Compiled-policy deny, then pre-tool hooks.
///
/// `Ask` from the compiled policy does not block. A hook deny does. A hook
/// that asks for approval also blocks, because this path has no approval broker.
pub async fn pre_tool_block(hooks: &HookRegistry, tool: &str, args: &Value) -> Option<String> {
    if let Some(reason) = tool_deny_reason(tool, args) {
        return Some(reason);
    }
    hook_block(hooks, tool, args).await
}

/// Hard deny for an MCP connect. `None` means connect may continue.
#[must_use]
pub fn connect_deny_reason(server: &str) -> Option<String> {
    let decision = evaluate_action(
        Action::McpConnect {
            server: server.to_string(),
        },
        &Value::Null,
    );
    (decision.effect == Effect::Deny).then(|| decision.reason.message().to_string())
}

/// First HTTP(S) URL in `value` that egress refuses.
///
/// A policy load error denies the value. Callers must not return that URL.
#[must_use]
pub fn rejected_input_url(value: &Value) -> Option<String> {
    let policy = match current_policy() {
        Ok(policy) => policy,
        Err(err) => return Some(err.to_string()),
    };
    first_denied_url(&policy, value, 0)
}

/// Egress decision for one MCP URL.
#[must_use]
pub fn url_denied_by(policy: &EgressPolicy, url: &str) -> Option<String> {
    policy
        .validate_static(EgressClass::Mcp, url)
        .err()
        .map(|err| err.to_string())
}

fn finish_tool(tool: &str, args: &Value, decision: &Decision) -> Option<String> {
    let denied = decision.effect == Effect::Deny;
    let message = decision.reason.message().to_string();
    let live = if denied {
        Err(Cow::owned(message.clone()))
    } else {
        Ok(())
    };
    observe_live_outcome(tool, args, &live);
    denied.then_some(message)
}

fn first_denied_url(policy: &EgressPolicy, value: &Value, depth: u8) -> Option<String> {
    if depth > URL_WALK_DEPTH {
        return None;
    }
    match value {
        Value::String(text) => denied_http_text(policy, text),
        Value::Array(items) => items
            .iter()
            .find_map(|item| first_denied_url(policy, item, depth.saturating_add(1))),
        Value::Object(map) => map
            .values()
            .find_map(|item| first_denied_url(policy, item, depth.saturating_add(1))),
        _ => None,
    }
}

async fn hook_block(hooks: &HookRegistry, tool: &str, args: &Value) -> Option<String> {
    let tool = tool_name_as_run(tool);
    let matching = hooks.get_matching(tool);
    if matching.is_empty() {
        return None;
    }
    let ctx = HookContext::new(tool, args, "mcp");
    for hook in matching {
        let result = hook.on_pre_tool_use(&ctx).await;
        if let Some(reason) = hook_stop_reason(&result) {
            let live = Err(Cow::owned(reason.clone()));
            observe_live_outcome(tool, args, &live);
            return Some(reason);
        }
    }
    None
}

fn hook_stop_reason(result: &HookResult) -> Option<String> {
    if result.is_denied() {
        return Some(
            result
                .permission_decision
                .denial_reason()
                .unwrap_or("Hook denied tool execution")
                .to_string(),
        );
    }
    if result.is_approval_required() {
        return Some(
            result
                .permission_decision
                .approval_reason()
                .unwrap_or("Approval required")
                .to_string(),
        );
    }
    None
}

fn denied_http_text(policy: &EgressPolicy, text: &str) -> Option<String> {
    if text.starts_with("http://") || text.starts_with("https://") {
        url_denied_by(policy, text)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gateway::egress::EgressPolicy;

    #[test]
    fn vendor_auth_tool_is_blocked() {
        let reason = tool_deny_reason(
            "fs",
            &serde_json::json!({"op": "read", "path": "/tmp/.codex/auth.json"}),
        )
        .expect("deny");
        assert!(reason.contains("vendor auth"));
    }

    #[test]
    fn ask_does_not_block_an_mcp_tool() {
        assert!(tool_deny_reason("acme__search", &Value::Null).is_none());
    }

    #[tokio::test]
    async fn hook_deny_blocks_before_the_tool_runs() {
        use crate::agent::hooks::{HookMatcher, LifecycleHook};
        use async_trait::async_trait;
        use std::sync::Arc;

        #[derive(Debug)]
        struct DenyHook;

        #[async_trait]
        impl LifecycleHook for DenyHook {
            async fn on_pre_tool_use(
                &self,
                _ctx: &crate::agent::hooks::HookContext<'_>,
            ) -> HookResult {
                HookResult::deny("blocked by hook")
            }
        }

        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("acme__search"), Arc::new(DenyHook));
        let reason = pre_tool_block(&hooks, "acme__search", &Value::Null)
            .await
            .expect("hook deny");
        assert!(reason.contains("blocked by hook"));
    }

    #[tokio::test]
    async fn hook_approval_blocks_without_a_broker() {
        use crate::agent::hooks::{HookMatcher, LifecycleHook};
        use async_trait::async_trait;
        use std::sync::Arc;

        #[derive(Debug)]
        struct AskHook;

        #[async_trait]
        impl LifecycleHook for AskHook {
            async fn on_pre_tool_use(
                &self,
                _ctx: &crate::agent::hooks::HookContext<'_>,
            ) -> HookResult {
                HookResult::require_approval("needs review", None)
            }
        }

        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("acme__search"), Arc::new(AskHook));
        let reason = pre_tool_block(&hooks, "acme__search", &Value::Null)
            .await
            .expect("approval deny");
        assert!(reason.contains("needs review"));
    }

    #[test]
    fn ask_does_not_block_connect() {
        assert!(connect_deny_reason("acme").is_none());
    }

    #[test]
    fn metadata_url_is_denied_and_loopback_stays_open() {
        let policy = EgressPolicy::allow_public();
        assert!(url_denied_by(&policy, "http://169.254.169.254/latest/meta-data/").is_some());
        assert!(url_denied_by(&policy, "http://127.0.0.1:9/mcp").is_none());
    }

    #[test]
    fn nested_input_url_is_rejected() {
        let body = serde_json::json!({
            "inputRequests": {
                "1": { "params": { "url": "http://169.254.169.254/latest" } }
            }
        });
        assert!(rejected_input_url(&body).is_some());
    }
}
