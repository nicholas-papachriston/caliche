//! LSP diagnostics tool - fetch diagnostics for a file.

use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::lsp::{DiagnosticFormatter, LspManager};
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Fetch LSP diagnostics for a file.
#[derive(Clone, Debug)]
pub struct LspDiagnosticsTool {
    manager: Arc<LspManager>,
}

impl LspDiagnosticsTool {
    #[must_use]
    pub const fn new(manager: Arc<LspManager>) -> Self {
        Self { manager }
    }
}

impl ToolMeta for LspDiagnosticsTool {
    fn name(&self) -> &'static str {
        "lsp_diagnostics"
    }

    fn description(&self) -> &'static str {
        "Fetch Language Server Protocol (LSP) diagnostics for a file. Returns errors, warnings, and other issues detected by the language server."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to get diagnostics for"
                }
            },
            "required": ["path"]
        })
    }
}

#[async_trait]
impl Tool for LspDiagnosticsTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(&arguments, "path", None) {
            Ok(path) => path,
            Err(error) => return ToolResult::error(error),
        };
        let resolved = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(path) => path,
            Err(error) => return ToolResult::error(error),
        };
        match self.manager.diagnostics_for_path(&resolved).await {
            Ok(diagnostics) => ToolResult::success(
                json!({
                    "path": resolved,
                    "summary": DiagnosticFormatter::summary(&diagnostics),
                    "formatted": DiagnosticFormatter::format_for_llm(&diagnostics),
                    "diagnostics": diagnostics,
                })
                .to_string(),
            ),
            Err(error) => ToolResult::error(format!("LSP diagnostics failed: {error}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsp_diagnostics_meta() {
        let tool = LspDiagnosticsTool::new(Arc::new(LspManager::default()));
        assert_eq!(tool.name(), "lsp_diagnostics");
        assert!(tool.description().contains("diagnostics"));
    }
}
