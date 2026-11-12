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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::hooks::{HookContext, HookMatcher, HookResult, LifecycleHook};
    use crate::agent::tools::common::set_tool_workspace_root;
    use crate::agent::tools::fs_tool::FsTool;
    use crate::mcp::protocol::ContentPart;
    use async_trait::async_trait;

    const VENDOR_DENY: &str = "floor denies reading vendor auth files";

    #[derive(Debug)]
    struct DenyHook;

    #[async_trait]
    impl LifecycleHook for DenyHook {
        async fn on_pre_tool_use(&self, ctx: &HookContext<'_>) -> HookResult {
            HookResult::deny(format!("hook stopped {}", ctx.tool_name))
        }
    }

    fn text(result: &ToolsCallResult) -> String {
        result
            .content
            .iter()
            .flatten()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn fs_registry() -> ToolRegistry {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(FsTool::new()));
        registry
    }

    /// The MCP server gates a call under the name the registry runs it as,
    /// so a padded name gets the `fs` decision and the `fs` hooks.
    #[tokio::test]
    async fn an_mcp_tool_call_is_gated_as_the_tool_it_runs() {
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        std::fs::create_dir_all(root.join(".codex")).expect("codex dir");
        std::fs::write(root.join(".codex/auth.json"), "VENDOR-TOKEN").expect("auth");
        std::fs::write(root.join("notes.txt"), "notes").expect("notes");
        set_tool_workspace_root(Some(root.clone()));
        let registry = fs_registry();
        let auth = root.join(".codex/auth.json").to_string_lossy().into_owned();

        let mut seen = Vec::new();
        for name in ["fs", " fs", "fs ", "\tfs\n"] {
            let args = serde_json::json!({"op": "read", "path": auth});
            let result = execute_mcp_tool(&registry, &HookRegistry::new(), name, args).await;
            seen.push((name, result.is_error, text(&result)));
        }

        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("fs"), Arc::new(DenyHook));
        let args = serde_json::json!({"op": "read", "path": "notes.txt"});
        let hooked = execute_mcp_tool(&registry, &hooks, " fs", args).await;

        set_tool_workspace_root(None);
        let refused = |name| (name, Some(true), VENDOR_DENY.to_string());
        assert_eq!(
            seen,
            vec![
                refused("fs"),
                refused(" fs"),
                refused("fs "),
                refused("\tfs\n")
            ]
        );
        assert_eq!(
            (hooked.is_error, text(&hooked)),
            (Some(true), "hook stopped fs".to_string())
        );
    }

    /// Every entry that runs a tool by name gives a padded name the decision
    /// of the tool it runs: the agent loop (the CLI and the server's chat),
    /// the MCP server over stdio and HTTP, the in-process MCP server, and
    /// the MCP client's check.
    #[tokio::test]
    async fn every_entry_path_gates_a_padded_name_as_the_tool_it_runs() {
        use crate::agent::tools::execution::{
            HookExecutionMiddleware, ToolExecutionContext, ToolExecutionMiddleware as _,
        };
        let tmp = tempfile::tempdir().expect("tmp");
        let root = tmp.path().canonicalize().expect("root");
        std::fs::create_dir_all(root.join(".codex")).expect("codex dir");
        std::fs::write(root.join(".codex/auth.json"), "VENDOR-TOKEN").expect("auth");
        set_tool_workspace_root(Some(root.clone()));
        let registry = Arc::new(fs_registry());
        let agent = HookExecutionMiddleware::new(HookRegistry::new())
            .with_workspace(root.to_string_lossy());
        let mut handler = crate::mcp::McpHandler::new(crate::mcp::McpServer::new(
            Arc::clone(&registry),
            crate::templates::TemplateRegistry::new(),
        ));
        let in_process = crate::mcp::InProcessMcpServerBuilder::new("entry")
            .add_tools(vec![Arc::new(FsTool::new())])
            .build()
            .expect("in-process server");
        let auth = root.join(".codex/auth.json").to_string_lossy().into_owned();

        let mut seen = Vec::new();
        for name in ["fs", " fs", "fs ", "\tfs\n"] {
            let args = serde_json::json!({"op": "read", "path": auth});
            let ctx = ToolExecutionContext {
                tool_name: name,
                arguments: &args,
                call_id: "entry",
            };
            let from_agent = agent
                .before_execute(&ctx)
                .await
                .err()
                .map(|e| e.into_owned());
            let request = JsonRpcRequest {
                jsonrpc: beef::lean::Cow::borrowed("2.0"),
                method: beef::lean::Cow::borrowed("tools/call"),
                params: Some(serde_json::json!({
                    "name": name,
                    "arguments": args,
                    "_meta": {"io.modelcontextprotocol/protocolVersion": LATEST_PROTOCOL_VERSION},
                })),
                id: Some(serde_json::json!(1)),
            };
            let response = handler.handle_request(request).await.expect("response");
            let result: ToolsCallResult =
                serde_json::from_value(response.result.expect("result")).expect("decode");
            let from_mcp = (result.is_error == Some(true)).then(|| text(&result));
            let local = in_process
                .call_tool(name, args.clone())
                .await
                .expect("call");
            let from_in_process = (local.is_error == Some(true)).then(|| text(&local));
            let from_client = crate::mcp::policy_gate::tool_deny_reason(name, &args);
            seen.push((name, [from_agent, from_mcp, from_in_process, from_client]));
        }
        set_tool_workspace_root(None);

        let refused = || std::array::from_fn(|_| Some(VENDOR_DENY.to_string()));
        assert_eq!(
            seen,
            vec![
                ("fs", refused()),
                (" fs", refused()),
                ("fs ", refused()),
                ("\tfs\n", refused()),
            ]
        );
    }

    /// A hook registered under an alias fires for the tool the alias runs.
    #[tokio::test]
    async fn a_hook_registered_under_an_alias_fires_for_its_tool() {
        let mut hooks = HookRegistry::new();
        hooks.register(HookMatcher::tool("memory"), Arc::new(DenyHook));
        hooks.register(HookMatcher::tool(" Shell "), Arc::new(DenyHook));
        let registry = ToolRegistry::new();

        let mut seen = Vec::new();
        for name in ["workspace", "memory", "bash", "shell"] {
            let result = execute_mcp_tool(&registry, &hooks, name, Value::Null).await;
            seen.push((name, result.is_error, text(&result)));
        }

        assert_eq!(
            seen,
            vec![
                (
                    "workspace",
                    Some(true),
                    "hook stopped workspace".to_string()
                ),
                ("memory", Some(true), "hook stopped workspace".to_string()),
                ("bash", Some(true), "hook stopped bash".to_string()),
                ("shell", Some(true), "hook stopped bash".to_string()),
            ]
        );
    }
}
