//! LSP symbols tool - search for symbols in the workspace.

use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::lsp::LspManager;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;

/// Search for symbols in the workspace using LSP.
#[derive(Clone, Debug)]
pub struct LspSymbolsTool {
    manager: Arc<LspManager>,
}

impl LspSymbolsTool {
    #[must_use]
    pub const fn new(manager: Arc<LspManager>) -> Self {
        Self { manager }
    }
}

impl ToolMeta for LspSymbolsTool {
    fn name(&self) -> &'static str {
        "lsp_symbols"
    }

    fn description(&self) -> &'static str {
        "Search for symbols (functions, classes, variables, etc.) in the workspace using Language Server Protocol. Useful for finding definitions and understanding code structure."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name or pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "Optional: workspace root or file path used to select the language server"
                }
            },
            "required": ["query"]
        })
    }
}

#[async_trait]
impl Tool for LspSymbolsTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let query = match params::extract_string(&arguments, "query", None) {
            Ok(query) => query,
            Err(error) => return ToolResult::error(error),
        };

        let scope = match arguments.get("path").and_then(|v| v.as_str()) {
            Some(path_str) => match path::validate_and_resolve_path(path_str, None).await {
                Ok(path) => Some(path),
                Err(error) => return ToolResult::error(error),
            },
            None => None,
        };

        match self
            .manager
            .workspace_symbols(&query, scope.as_deref())
            .await
        {
            Ok(symbols) => ToolResult::success(
                json!({
                    "query": query,
                    "path": scope,
                    "count": symbols.len(),
                    "symbols": symbols,
                })
                .to_string(),
            ),
            Err(error) => ToolResult::error(format!("LSP symbol search failed: {error}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lsp_symbols_meta() {
        let tool = LspSymbolsTool::new(Arc::new(LspManager::default()));
        assert_eq!(tool.name(), "lsp_symbols");
        assert!(tool.description().contains("symbols"));
    }
}
