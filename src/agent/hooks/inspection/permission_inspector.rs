//! Permission-mode inspector (`AcceptAll` / `DenyAll` / `AcceptEdits` / Interactive).

use super::manager::{InspectionAction, InspectionResult, ToolInspector, ToolInspectorRequest};
use crate::agent::hooks::PermissionMode;
use crate::agent::tools::names::{TOOL_FILESYSTEM, fs_op, is_fs_write_op};
use anyhow::Result;
use async_trait::async_trait;

/// Applies [`PermissionMode`] policy before tool execution.
pub struct PermissionModeInspector {
    mode: PermissionMode,
}

impl PermissionModeInspector {
    #[must_use]
    pub const fn new(mode: PermissionMode) -> Self {
        Self { mode }
    }

    #[must_use]
    pub const fn mode(&self) -> PermissionMode {
        self.mode
    }

    pub const fn set_mode(&mut self, mode: PermissionMode) {
        self.mode = mode;
    }
}

fn is_edit_tool(tool_name: &str, arguments: &serde_json::Value) -> bool {
    if tool_name == "apply_patch" {
        return true;
    }
    if tool_name == TOOL_FILESYSTEM {
        return fs_op(arguments).is_some_and(is_fs_write_op);
    }
    false
}

#[async_trait]
impl ToolInspector for PermissionModeInspector {
    fn name(&self) -> &'static str {
        "permission_mode"
    }

    async fn inspect(&self, request: &ToolInspectorRequest<'_>) -> Result<Vec<InspectionResult>> {
        let action = match self.mode {
            PermissionMode::Default => InspectionAction::Allow,
            PermissionMode::AcceptAll => InspectionAction::Allow,
            PermissionMode::DenyAll => InspectionAction::Deny {
                reason: "PermissionMode::DenyAll blocks all tool calls".to_string(),
            },
            PermissionMode::AcceptEdits => {
                // Auto-allow file mutations and read-only fs; require approval elsewhere.
                if is_edit_tool(request.tool_name, request.arguments)
                    || request.tool_name == TOOL_FILESYSTEM
                {
                    InspectionAction::Allow
                } else {
                    InspectionAction::RequireApproval {
                        reason: format!(
                            "PermissionMode::AcceptEdits requires approval for '{}'",
                            request.tool_name
                        ),
                    }
                }
            }
            PermissionMode::Interactive => {
                if is_edit_tool(request.tool_name, request.arguments) || request.tool_name == "bash"
                {
                    InspectionAction::RequireApproval {
                        reason: format!(
                            "PermissionMode::Interactive requires approval for '{}'",
                            request.tool_name
                        ),
                    }
                } else {
                    InspectionAction::Allow
                }
            }
        };

        let reason = match &action {
            InspectionAction::Allow => "allowed by permission mode".to_string(),
            InspectionAction::Deny { reason } | InspectionAction::RequireApproval { reason } => {
                reason.clone()
            }
        };

        Ok(vec![InspectionResult {
            call_id: request.call_id.to_string(),
            action,
            reason,
            inspector_name: self.name(),
            confidence: 1.0,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn deny_all_blocks() {
        let inspector = PermissionModeInspector::new(PermissionMode::DenyAll);
        let args = json!({"op":"read","path":"."});
        let request = ToolInspectorRequest {
            tool_name: "fs",
            arguments: &args,
            call_id: "1",
        };
        let results = inspector.inspect(&request).await.expect("inspect");
        assert!(matches!(results[0].action, InspectionAction::Deny { .. }));
    }

    #[tokio::test]
    async fn accept_edits_allows_fs_write_and_prompts_for_bash() {
        let inspector = PermissionModeInspector::new(PermissionMode::AcceptEdits);
        let write_args = json!({"op":"write","path":"a","content":"x"});
        let write_req = ToolInspectorRequest {
            tool_name: "fs",
            arguments: &write_args,
            call_id: "w",
        };
        let write = inspector.inspect(&write_req).await.expect("inspect")[0]
            .action
            .clone();
        assert_eq!(write, InspectionAction::Allow);

        let bash_args = json!({"command":"ls"});
        let bash_req = ToolInspectorRequest {
            tool_name: "bash",
            arguments: &bash_args,
            call_id: "b",
        };
        let bash = inspector.inspect(&bash_req).await.expect("inspect")[0]
            .action
            .clone();
        assert!(matches!(bash, InspectionAction::RequireApproval { .. }));
    }

    #[tokio::test]
    async fn interactive_prompts_for_mutations() {
        let inspector = PermissionModeInspector::new(PermissionMode::Interactive);
        let args = json!({"patch":"*** Begin Patch\n*** End Patch\n"});
        let request = ToolInspectorRequest {
            tool_name: "apply_patch",
            arguments: &args,
            call_id: "p",
        };
        let action = inspector.inspect(&request).await.expect("inspect")[0]
            .action
            .clone();
        assert!(matches!(action, InspectionAction::RequireApproval { .. }));
    }
}
