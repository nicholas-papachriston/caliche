//! LSP hover tool - get type information at a location.

use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::lsp::{LspManager, Position};
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Get type information and hover details at a specific location.
#[derive(Clone, Debug)]
pub struct LspHoverTool {
    manager: Arc<LspManager>,
}

impl LspHoverTool {
    #[must_use]
    pub const fn new(manager: Arc<LspManager>) -> Self {
        Self { manager }
    }
}

impl ToolMeta for LspHoverTool {
    fn name(&self) -> &'static str {
        "lsp_hover"
    }

    fn description(&self) -> &'static str {
        "Get type information, documentation, and hover details at a specific location in a file using Language Server Protocol. Useful for understanding types and function signatures."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file"
                },
                "line": {
                    "type": "integer",
                    "description": "0-based line number"
                },
                "column": {
                    "type": "integer",
                    "description": "0-based UTF-16/LSP character offset"
                }
            },
            "required": ["path", "line", "column"]
        })
    }
}

#[async_trait]
impl Tool for LspHoverTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(&arguments, "path", None) {
            Ok(path) => path,
            Err(error) => return ToolResult::error(error),
        };

        let line = match params::extract_number(&arguments, "line", None) {
            Ok(line) if line <= u64::from(u32::MAX) => line as u32,
            Ok(_) => return ToolResult::error("Parameter 'line' is too large"),
            Err(error) => return ToolResult::error(error),
        };

        let column = match params::extract_number(&arguments, "column", None) {
            Ok(column) if column <= u64::from(u32::MAX) => column as u32,
            Ok(_) => return ToolResult::error("Parameter 'column' is too large"),
            Err(error) => return ToolResult::error(error),
        };

        let resolved = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(path) => path,
            Err(error) => return ToolResult::error(error),
        };
        match self
            .manager
            .hover(&resolved, Position::new(line, column))
            .await
        {
            Ok(Some(hover)) => ToolResult::success(
                json!({
                    "path": resolved,
                    "line": line,
                    "column": column,
                    "hover": hover,
                })
                .to_string(),
            ),
            Ok(None) => ToolResult::success(
                json!({
                    "path": resolved,
                    "line": line,
                    "column": column,
                    "hover": null,
                })
                .to_string(),
            ),
            Err(error) => ToolResult::error(format!("LSP hover failed: {error}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsp_hover_meta() {
        let tool = LspHoverTool::new(Arc::new(LspManager::default()));
        assert_eq!(tool.name(), "lsp_hover");
        assert!(tool.description().contains("type information"));
    }
}
