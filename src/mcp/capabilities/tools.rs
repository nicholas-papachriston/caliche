//! Tools capability implementation for MCP.

use crate::agent::hooks::HookRegistry;
use crate::agent::tools::ToolRegistry;
use crate::mcp::protocol::*;
use anyhow::{Context, Result};
use caliche_mcp::pagination::{DEFAULT_PAGE_SIZE, cmp_name, paginate_owned};
use serde_json::Value;
use std::sync::Arc;

pub(crate) fn collect_tools(tool_registry: &ToolRegistry) -> Vec<McpTool> {
    tool_registry
        .list()
        .into_iter()
        .map(|(name, description, input_schema)| McpTool::new(name, description, input_schema))
        .collect()
}

/// Handle `tools/list` with deterministic cursor pagination.
pub async fn handle_tools_list(
    tool_registry: &Arc<ToolRegistry>,
    params: Option<ToolsListParams>,
) -> Result<ToolsListResult> {
    let mut mcp_tools = collect_tools(tool_registry);
    mcp_tools.sort_by(|a, b| cmp_name(a.name.as_ref(), b.name.as_ref()));

    let cursor = params.as_ref().and_then(|p| p.cursor.as_deref());
    let page = paginate_owned(mcp_tools, cursor, Some(DEFAULT_PAGE_SIZE))
        .context("invalid tools/list cursor")?;

    Ok(ToolsListResult {
        tools: page.items,
        next_cursor: page.next_cursor,
    })
}

/// Handle `tools/call`, mapping typed payloads to `structuredContent`.
pub async fn handle_tools_call(
    tool_registry: &Arc<ToolRegistry>,
    hooks: &HookRegistry,
    params: ToolsCallParams,
) -> Result<ToolsCallResult> {
    let arguments = params
        .arguments
        .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
    Ok(execute_mcp_tool(tool_registry, hooks, &params.name, arguments).await)
}

pub(crate) async fn execute_mcp_tool(
    tool_registry: &ToolRegistry,
    hooks: &HookRegistry,
    name: &str,
    arguments: Value,
) -> ToolsCallResult {
    if let Some(reason) = crate::mcp::policy_gate::pre_tool_block(hooks, name, &arguments).await {
        return ToolsCallResult::from_text(reason, None, true);
    }

    let tool_result = tool_registry.execute(name, arguments).await;

    if tool_result.success {
        ToolsCallResult::from_text(tool_result.content, tool_result.typed_payload, false)
    } else {
        ToolsCallResult::from_text(
            tool_result
                .error
                .unwrap_or_else(|| "Unknown error".to_string()),
            tool_result.typed_payload,
            true,
        )
    }
}
