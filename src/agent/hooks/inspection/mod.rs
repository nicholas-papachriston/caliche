//! Goose-inspired tool inspection chain.
//!
//! Inspectors run in registration order before tool execution. Results are
//! merged with Deny > `RequireApproval` > Allow precedence.

mod manager;
mod permission_inspector;
mod security_inspector;

pub use manager::{
    InspectionAction, InspectionResult, ToolInspectionManager, ToolInspector, ToolInspectorRequest,
};
pub use permission_inspector::PermissionModeInspector;
pub use security_inspector::PathSecurityInspector;

use async_trait::async_trait;
use std::fmt;
use std::sync::Arc;

use crate::agent::hooks::lifecycle::{HookContext, HookResult, LifecycleHook};

/// Adapts a [`ToolInspectionManager`] into a [`LifecycleHook`].
pub struct InspectionHookAdapter {
    manager: Arc<ToolInspectionManager>,
}

impl InspectionHookAdapter {
    #[must_use]
    pub const fn new(manager: Arc<ToolInspectionManager>) -> Self {
        Self { manager }
    }

    #[must_use]
    pub fn manager(&self) -> &ToolInspectionManager {
        &self.manager
    }
}

impl fmt::Debug for InspectionHookAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InspectionHookAdapter")
            .field("inspectors", &self.manager.inspector_names())
            .finish()
    }
}

#[async_trait]
impl LifecycleHook for InspectionHookAdapter {
    async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
        let request = ToolInspectorRequest {
            tool_name: ctx.tool_name.as_ref(),
            arguments: ctx.tool_input,
            call_id: ctx.tool_use_id.as_ref(),
        };
        match self.manager.inspect_one(&request).await {
            Ok(action) => match action {
                InspectionAction::Allow => HookResult::allow(),
                InspectionAction::Deny { reason } => HookResult::deny(reason),
                InspectionAction::RequireApproval { reason } => {
                    HookResult::require_approval(reason, None)
                }
            },
            Err(err) => {
                tracing::error!(error = %err, "tool inspection failed; denying");
                HookResult::deny(format!("tool inspection failed: {err}"))
            }
        }
    }
}

/// Build a default inspection chain: path security, then permission mode.
#[must_use]
pub fn default_inspection_manager(
    permission_mode: crate::agent::hooks::PermissionMode,
    workspace_root: Option<std::path::PathBuf>,
) -> ToolInspectionManager {
    let mut manager = ToolInspectionManager::new();
    if let Some(root) = workspace_root {
        manager.add_inspector(Box::new(PathSecurityInspector::new(root)));
    }
    manager.add_inspector(Box::new(PermissionModeInspector::new(permission_mode)));
    manager
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hooks::PermissionMode;
    use serde_json::json;
    use std::path::PathBuf;

    #[tokio::test]
    async fn adapter_denies_when_security_inspector_denies() {
        let manager = Arc::new(default_inspection_manager(
            PermissionMode::AcceptAll,
            Some(PathBuf::from("/tmp/workspace")),
        ));
        let adapter = InspectionHookAdapter::new(manager);
        let args = json!({"op":"read","path":"/etc/passwd"});
        let ctx = HookContext::new("fs", &args, "call-1");
        let result = adapter.on_pre_tool_use(&ctx).await;
        assert!(result.is_denied());
    }

    #[tokio::test]
    async fn adapter_allows_accept_all_for_safe_path() {
        let manager = Arc::new(default_inspection_manager(
            PermissionMode::AcceptAll,
            Some(PathBuf::from("/tmp/workspace")),
        ));
        let adapter = InspectionHookAdapter::new(manager);
        let args = json!({"op":"read","path":"src/main.rs"});
        let ctx = HookContext::new("fs", &args, "call-2");
        let result = adapter.on_pre_tool_use(&ctx).await;
        assert!(result.is_allowed());
    }
}
