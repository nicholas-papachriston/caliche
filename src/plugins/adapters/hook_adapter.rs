//! Schema-defined lifecycle hook adapter: implements LifecycleHook via the step DSL.
//!
//! Pre-tool-use steps run with args: tool_name, tool_input, tool_use_id. The final step
//! output is parsed as a decision: string "allow", or object with decision/reason/amendment.

use crate::agent::hooks::LifecycleHook;
use crate::agent::hooks::lifecycle::{HookContext, HookResult};
use crate::plugins::runtime::{AgentCapabilities, StepRuntime, resolve_path};
use crate::plugins::schema::{HookPluginSpec, PluginDefinition, PluginSpec};
use async_trait::async_trait;
use std::sync::Arc;

const HOOK_TIMEOUT_SECS: u64 = 30;

#[derive(Debug)]
pub struct SchemaHookAdapter {
    _definition: Arc<PluginDefinition>,
    spec: HookPluginSpec,
    runtime: Arc<StepRuntime>,
    capabilities: Arc<AgentCapabilities>,
}

impl SchemaHookAdapter {
    pub fn new(
        definition: Arc<PluginDefinition>,
        runtime: Arc<StepRuntime>,
        capabilities: Arc<AgentCapabilities>,
    ) -> Option<Self> {
        let spec = match &definition.spec {
            PluginSpec::Hook(s) => s.clone(),
            _ => return None,
        };
        Some(Self {
            _definition: definition,
            spec,
            runtime,
            capabilities,
        })
    }

    fn args_from_context(ctx: &HookContext<'_>) -> serde_json::Value {
        serde_json::json!({
            "tool_name": ctx.tool_name.as_ref(),
            "tool_input": ctx.tool_input,
            "tool_use_id": ctx.tool_use_id.as_ref(),
        })
    }

    fn parse_decision(&self, value: serde_json::Value) -> HookResult {
        let decision_path = self.spec.decision_path.as_deref().unwrap_or("decision");
        let reason_path = self.spec.reason_path.as_deref().unwrap_or("reason");
        let amendment_path = self
            .spec
            .amendment_path
            .as_deref()
            .unwrap_or("proposed_amendment");

        if let Some(s) = value.as_str()
            && s.trim().eq_ignore_ascii_case("allow")
        {
            return HookResult::allow();
        }

        let root = match value.as_object() {
            Some(o) => serde_json::Value::Object(o.clone()),
            None => return HookResult::allow(),
        };

        let decision = resolve_path(&root, decision_path)
            .ok()
            .and_then(|v| v.as_str().map(String::from));
        match decision.as_deref() {
            Some(d) if d.eq_ignore_ascii_case("allow") => return HookResult::allow(),
            Some(d) if d.eq_ignore_ascii_case("deny") => {
                let reason = resolve_path(&root, reason_path)
                    .ok()
                    .and_then(|v| v.as_str().map(String::from))
                    .unwrap_or_else(|| "Hook denied".to_string());
                return HookResult::deny(reason);
            }
            Some(d) if d.eq_ignore_ascii_case("require_approval") => {
                let reason = resolve_path(&root, reason_path)
                    .ok()
                    .and_then(|v| v.as_str().map(String::from))
                    .unwrap_or_else(|| "Approval required".to_string());
                let proposed_amendment = resolve_path(&root, amendment_path).ok().and_then(|v| {
                    v.as_array().map(|arr| {
                        arr.iter()
                            .filter_map(|x| x.as_str().map(String::from))
                            .collect::<Vec<_>>()
                    })
                });
                return HookResult::require_approval(reason, proposed_amendment);
            }
            _ => {}
        }

        HookResult::allow()
    }
}

#[async_trait]
impl LifecycleHook for SchemaHookAdapter {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        let args = Self::args_from_context(ctx);
        match self
            .runtime
            .execute_steps(
                &self.spec.pre_tool_use_steps,
                args,
                &self.capabilities,
                HOOK_TIMEOUT_SECS,
            )
            .await
        {
            Ok(value) => self.parse_decision(value),
            Err(e) => {
                let reason = format!("hook plugin error: {}", e);
                HookResult::deny(reason)
            }
        }
    }
}
