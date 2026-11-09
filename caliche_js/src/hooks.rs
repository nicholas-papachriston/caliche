//! JavaScript bindings for agent lifecycle hooks
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]
#![expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "NAPI instance methods require a borrowed self receiver"
)]

use async_trait::async_trait;
use caliche::agent::hooks::lifecycle::{HookContext, HookResult, LifecycleHook};
use caliche::agent::hooks::matcher::HookMatcher;
use caliche::agent::hooks::permission::{PermissionDecision, PermissionMode};
use caliche::agent::tools::tool::ToolResult;
use napi::threadsafe_function::ThreadsafeFunction;
use napi_derive::napi;
use serde_json::Value;
use std::sync::Arc;

use crate::call_js_sync_bridge;

/// JavaScript wrapper for `PermissionMode` enum
#[napi]
#[derive(Clone, Copy)]
pub struct JsPermissionMode {
    inner: PermissionMode,
}

#[napi]
impl JsPermissionMode {
    #[napi(constructor, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn new(mode: String) -> napi::Result<Self> {
        let inner = match mode.to_lowercase().as_str() {
            "default" => PermissionMode::Default,
            "accept-edits" => PermissionMode::AcceptEdits,
            "accept-all" => PermissionMode::AcceptAll,
            "deny-all" => PermissionMode::DenyAll,
            "interactive" => PermissionMode::Interactive,
            _ => {
                return Err(napi::Error::from_reason(format!(
                    "Invalid permission mode: {mode}. Must be one of: default, accept-edits, accept-all, deny-all, interactive"
                )));
            }
        };
        Ok(Self { inner })
    }

    #[napi(factory, catch_unwind)]
    pub const fn default() -> Self {
        Self {
            inner: PermissionMode::Default,
        }
    }

    #[napi(factory, catch_unwind)]
    pub const fn accept_edits() -> Self {
        Self {
            inner: PermissionMode::AcceptEdits,
        }
    }

    #[napi(factory, catch_unwind)]
    pub const fn accept_all() -> Self {
        Self {
            inner: PermissionMode::AcceptAll,
        }
    }

    #[napi(factory, catch_unwind)]
    pub const fn deny_all() -> Self {
        Self {
            inner: PermissionMode::DenyAll,
        }
    }

    #[napi(factory, catch_unwind)]
    pub const fn interactive() -> Self {
        Self {
            inner: PermissionMode::Interactive,
        }
    }

    #[napi(catch_unwind)]
    #[allow(clippy::inherent_to_string)]
    pub const fn to_string(&self) -> &'static str {
        match self.inner {
            PermissionMode::Default => "Default",
            PermissionMode::AcceptEdits => "AcceptEdits",
            PermissionMode::AcceptAll => "AcceptAll",
            PermissionMode::DenyAll => "DenyAll",
            PermissionMode::Interactive => "Interactive",
        }
    }
}

impl From<JsPermissionMode> for PermissionMode {
    fn from(js_mode: JsPermissionMode) -> Self {
        js_mode.inner
    }
}

/// JavaScript wrapper for `HookMatcher` enum
#[napi]
#[derive(Clone)]
pub struct JsHookMatcher {
    inner: HookMatcher,
}

#[napi]
impl JsHookMatcher {
    #[napi(constructor, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn new(matcher_type: String, tool_name: Option<String>) -> napi::Result<Self> {
        let inner = match matcher_type.to_lowercase().as_str() {
            "tool" => {
                let name = tool_name.ok_or_else(|| {
                    napi::Error::from_reason("Tool matcher requires a tool_name parameter")
                })?;
                HookMatcher::Tool { name: name.into() }
            }
            "all" => HookMatcher::All,
            _ => {
                return Err(napi::Error::from_reason(format!(
                    "Invalid matcher type: {matcher_type}. Must be 'tool' or 'all'"
                )));
            }
        };
        Ok(Self { inner })
    }

    #[napi(factory, catch_unwind)]
    pub fn tool(tool_name: String) -> Self {
        Self {
            inner: HookMatcher::Tool {
                name: tool_name.into(),
            },
        }
    }

    #[napi(factory, catch_unwind)]
    pub const fn all() -> Self {
        Self {
            inner: HookMatcher::All,
        }
    }
}

impl From<JsHookMatcher> for HookMatcher {
    fn from(js_matcher: JsHookMatcher) -> Self {
        js_matcher.inner
    }
}

/// JavaScript wrapper for `HookContext`
#[napi(object)]
#[expect(
    clippy::struct_field_names,
    reason = "tool-prefixed fields are part of the public JavaScript hook context API"
)]
pub struct JsHookContext {
    #[napi(js_name = "tool_name")]
    pub tool_name: String,
    #[napi(js_name = "tool_input")]
    pub tool_input: Value,
    #[napi(js_name = "tool_use_id")]
    pub tool_use_id: String,
}

impl From<&HookContext<'_>> for JsHookContext {
    fn from(ctx: &HookContext<'_>) -> Self {
        Self {
            tool_name: ctx.tool_name.clone().into_owned(),
            tool_input: ctx.tool_input.clone(),
            tool_use_id: ctx.tool_use_id.clone().into_owned(),
        }
    }
}

/// JavaScript wrapper for `HookResult`
#[napi]
pub struct JsHookResult {
    inner: HookResult,
}

#[napi]
impl JsHookResult {
    #[napi(factory, catch_unwind)]
    pub fn allow() -> Self {
        Self {
            inner: HookResult::allow(),
        }
    }

    #[napi(factory, catch_unwind)]
    pub fn deny(reason: String) -> Self {
        Self {
            inner: HookResult::deny(reason),
        }
    }

    #[napi(getter, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn permission_decision(&self) -> Value {
        match &self.inner.permission_decision {
            PermissionDecision::Allow => serde_json::json!({"decision": "allow"}),
            PermissionDecision::Deny { reason } => {
                serde_json::json!({"decision": "deny", "reason": reason})
            }
            PermissionDecision::RequireApproval {
                reason,
                proposed_amendment,
            } => {
                let mut obj = serde_json::json!({"decision": "require_approval", "reason": reason});
                if let Some(prefix) = proposed_amendment {
                    obj["proposed_amendment"] = serde_json::json!(prefix);
                }
                obj
            }
        }
    }

    #[napi(getter, catch_unwind)]
    pub fn hook_specific_output(&self) -> Option<Value> {
        self.inner.hook_specific_output.clone()
    }
}

impl From<HookResult> for JsHookResult {
    fn from(result: HookResult) -> Self {
        Self { inner: result }
    }
}

/// JavaScript lifecycle hook wrapper that implements the Rust `LifecycleHook` trait
struct JsLifecycleHook {
    pre_tool_use_fn: ThreadsafeFunction<Value>,
    post_tool_use_fn: Option<ThreadsafeFunction<Value>>,
}

impl std::fmt::Debug for JsLifecycleHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsLifecycleHook")
            .field("pre_tool_use_fn", &"<ThreadsafeFunction>")
            .field("post_tool_use_fn", &"<ThreadsafeFunction>")
            .finish()
    }
}

#[async_trait]
impl LifecycleHook for JsLifecycleHook {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        let js_ctx = JsHookContext::from(ctx);
        let ctx_value = serde_json::json!({
            "tool_name": js_ctx.tool_name,
            "tool_input": js_ctx.tool_input,
            "tool_use_id": js_ctx.tool_use_id
        });

        // Call the JS function using async-aware bridge
        // The JS wrapper handles Promise unwrapping, so this works for both sync and async hooks
        let result_str = match call_js_sync_bridge(&self.pre_tool_use_fn, ctx_value).await {
            Ok(s) => s,
            Err(e) => {
                let reason = format!("JavaScript pre_tool_use hook error: {e}");
                tracing::warn!("{reason}");
                return HookResult::deny(reason);
            }
        };

        // Parse the result
        let result_value: Value = match serde_json::from_str(&result_str) {
            Ok(v) => v,
            Err(e) => {
                let reason = format!("Failed to parse pre_tool_use hook result: {e}");
                tracing::warn!("{reason}");
                return HookResult::deny(reason);
            }
        };

        let Some(decision) = result_value.get("decision").and_then(Value::as_str) else {
            let reason = "pre_tool_use hook returned an unparseable result (missing decision)";
            tracing::warn!("{reason}");
            return HookResult::deny(reason.to_string());
        };
        match decision {
            "allow" => HookResult::allow(),
            "deny" => {
                let reason = result_value
                    .get("reason")
                    .and_then(Value::as_str)
                    .map_or_else(
                        || "Denied by hook".to_string(),
                        std::string::ToString::to_string,
                    );
                HookResult::deny(reason)
            }
            other => {
                let reason = format!("Invalid pre_tool_use hook decision: {other}");
                tracing::warn!("{reason}");
                HookResult::deny(reason)
            }
        }
    }

    async fn on_post_tool_use(&self, _ctx: &HookContext<'_>, _result: &ToolResult) -> HookResult {
        let Some(post_fn) = &self.post_tool_use_fn else {
            return HookResult::allow();
        };
        let js_ctx = JsHookContext::from(_ctx);
        let ctx_value = serde_json::json!({
            "tool_name": js_ctx.tool_name,
            "tool_input": js_ctx.tool_input,
            "tool_use_id": js_ctx.tool_use_id
        });

        let result_value = serde_json::json!({
            "success": _result.success,
            "content": _result.content,
            "error": _result.error
        });

        let combined = serde_json::json!({
            "context": ctx_value,
            "result": result_value
        });

        // Call the JS function using async-aware bridge
        // The JS wrapper handles Promise unwrapping, so this works for both sync and async hooks
        let _ = call_js_sync_bridge(post_fn, combined).await;
        HookResult::allow()
    }
}

/// Create a JavaScript lifecycle hook from callbacks
pub fn create_js_lifecycle_hook(
    pre_tool_use_fn: ThreadsafeFunction<Value>,
    post_tool_use_fn: Option<ThreadsafeFunction<Value>>,
) -> Arc<dyn LifecycleHook> {
    Arc::new(JsLifecycleHook {
        pre_tool_use_fn,
        post_tool_use_fn,
    })
}
