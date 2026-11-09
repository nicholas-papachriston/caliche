//! Path security inspector — blocks absolute paths outside the workspace.

use super::manager::{InspectionAction, InspectionResult, ToolInspector, ToolInspectorRequest};
use crate::agent::tools::names::TOOL_FILESYSTEM;
use anyhow::Result;
use async_trait::async_trait;
use std::path::{Path, PathBuf};

/// Tools that may carry filesystem path arguments.
const PATH_TOOLS: &[&str] = &[TOOL_FILESYSTEM, "apply_patch", "lsp", "lsp_diagnostics"];

/// Denies absolute paths that resolve outside `workspace_root`.
pub struct PathSecurityInspector {
    workspace_root: PathBuf,
}

impl PathSecurityInspector {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
        }
    }

    fn path_allowed(&self, raw: &str) -> bool {
        if Path::new(raw)
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return false;
        }
        let path = Path::new(raw);
        if !path.is_absolute() {
            return true;
        }
        let Ok(root) = self.workspace_root.canonicalize() else {
            return path == self.workspace_root.as_path() || path.starts_with(&self.workspace_root);
        };
        path.starts_with(&root)
            || path
                .canonicalize()
                .map(|canonical| canonical.starts_with(&root))
                .unwrap_or(false)
    }

    fn extract_paths(tool_name: &str, arguments: &serde_json::Value) -> Vec<String> {
        let mut paths = Vec::new();
        if (tool_name == TOOL_FILESYSTEM || tool_name.starts_with("lsp"))
            && let Some(p) = arguments.get("path").and_then(|v| v.as_str())
        {
            paths.push(p.to_string());
        }
        if tool_name == "apply_patch"
            && let Some(patch) = arguments.get("patch").and_then(|v| v.as_str())
        {
            for line in patch.lines() {
                for prefix in [
                    "*** Add File: ",
                    "*** Delete File: ",
                    "*** Update File: ",
                    "*** Move to: ",
                ] {
                    if let Some(rest) = line.trim().strip_prefix(prefix) {
                        paths.push(rest.trim().to_string());
                    }
                }
            }
        }
        paths
    }
}

#[async_trait]
impl ToolInspector for PathSecurityInspector {
    fn name(&self) -> &'static str {
        "path_security"
    }

    async fn inspect(&self, request: &ToolInspectorRequest<'_>) -> Result<Vec<InspectionResult>> {
        if !PATH_TOOLS
            .iter()
            .any(|name| request.tool_name == *name || request.tool_name.starts_with("lsp"))
        {
            return Ok(vec![]);
        }

        for path in Self::extract_paths(request.tool_name, request.arguments) {
            if !self.path_allowed(&path) {
                return Ok(vec![InspectionResult {
                    call_id: request.call_id.to_string(),
                    action: InspectionAction::Deny {
                        reason: format!(
                            "Path '{}' is outside workspace '{}'",
                            path,
                            self.workspace_root.display()
                        ),
                    },
                    reason: format!("path outside workspace: {path}"),
                    inspector_name: self.name(),
                    confidence: 1.0,
                }]);
            }
        }

        Ok(vec![InspectionResult {
            call_id: request.call_id.to_string(),
            action: InspectionAction::Allow,
            reason: "paths within workspace".to_string(),
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
    async fn allows_relative_and_denies_foreign_absolute() {
        let inspector = PathSecurityInspector::new("/tmp/my-workspace");
        let ok_args = json!({"op":"read","path":"src/lib.rs"});
        let ok = ToolInspectorRequest {
            tool_name: "fs",
            arguments: &ok_args,
            call_id: "1",
        };
        let results = inspector.inspect(&ok).await.expect("inspect");
        assert_eq!(results[0].action, InspectionAction::Allow);

        let bad_args = json!({"op":"read","path":"/etc/passwd"});
        let bad = ToolInspectorRequest {
            tool_name: "fs",
            arguments: &bad_args,
            call_id: "2",
        };
        let results = inspector.inspect(&bad).await.expect("inspect");
        assert!(matches!(results[0].action, InspectionAction::Deny { .. }));
    }

    #[tokio::test]
    async fn scans_apply_patch_paths() {
        let inspector = PathSecurityInspector::new("/tmp/my-workspace");
        let args = json!({
            "patch": "*** Begin Patch\n*** Add File: /etc/evil\n+x\n*** End Patch\n"
        });
        let request = ToolInspectorRequest {
            tool_name: "apply_patch",
            arguments: &args,
            call_id: "p",
        };
        let results = inspector.inspect(&request).await.expect("inspect");
        assert!(matches!(results[0].action, InspectionAction::Deny { .. }));
    }

    #[tokio::test]
    async fn allows_dots_in_filename() {
        let inspector = PathSecurityInspector::new("/tmp/my-workspace");
        let args = json!({"op":"read","path":"foo..bar.txt"});
        let request = ToolInspectorRequest {
            tool_name: "fs",
            arguments: &args,
            call_id: "3",
        };
        let results = inspector.inspect(&request).await.expect("inspect");
        assert_eq!(results[0].action, InspectionAction::Allow);
    }
}
