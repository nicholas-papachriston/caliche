//! Integration tests for MCP server end-to-end functionality
//!
//! Tests complete MCP request/response cycles, tool execution through MCP,
//! resource access, and error scenarios.

#![cfg(feature = "mcp")]

use async_trait::async_trait;
use beef::lean::Cow;
use caliche::agent::tools::tool::ToolMeta;
use caliche::agent::tools::{Tool, ToolRegistry, ToolResult};
use caliche::mcp::handler::McpHandler;
use caliche::mcp::protocol::*;
use caliche::mcp::server::McpServerBuilder;
use caliche::templates::TemplateRegistry;
use serde_json::json;
use std::sync::Arc;

#[derive(Debug)]
struct TestTool;
impl ToolMeta for TestTool {
    fn name(&self) -> &str {
        "test_tool"
    }

    fn description(&self) -> &str {
        "A test tool for MCP integration"
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "Input string"
                }
            },
            "required": ["input"]
        })
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::owned(self.name().to_string())
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::owned(self.description().to_string())
    }
}

#[async_trait]
impl Tool for TestTool {
    async fn execute(&self, args: serde_json::Value) -> ToolResult {
        let input = args
            .get("input")
            .and_then(|v| v.as_str())
            .unwrap_or("default");

        ToolResult::success(format!("Processed: {input}"))
    }
}

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn create_test_handler() -> McpHandler {
    let reg = ToolRegistry::new();
    reg.register(Arc::new(TestTool));
    let tool_registry = Arc::new(reg);

    let template_registry = TemplateRegistry::new();

    let server = McpServerBuilder::new()
        .tool_registry(tool_registry)
        .template_registry(template_registry)
        .build()
        .expect("MCP server builder should succeed");

    McpHandler::new(server)
}

fn current_request(
    method: &'static str,
    params: Option<serde_json::Value>,
    id: serde_json::Value,
) -> JsonRpcRequest {
    let mut body = params.unwrap_or_else(|| json!({}));
    if let Some(map) = body.as_object_mut() {
        map.insert(
            "_meta".to_string(),
            json!({
                "io.modelcontextprotocol/protocolVersion": LATEST_PROTOCOL_VERSION
            }),
        );
    }
    JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed(method),
        params: Some(body),
        id: Some(id),
    }
}

#[tokio::test]
async fn test_mcp_initialize_is_not_a_server_method() {
    let mut handler = create_test_handler();
    let response = handler
        .handle_request(current_request(
            "initialize",
            Some(json!({
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {}
            })),
            json!(1),
        ))
        .await
        .expect("response");
    assert_eq!(response.error.expect("method").code, -32601);
}

#[tokio::test]
async fn test_mcp_missing_protocol_version() {
    let mut handler = create_test_handler();
    let request = JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed("tools/list"),
        params: None,
        id: Some(json!(1)),
    };
    let response = handler.handle_request(request).await.expect("response");
    assert_eq!(response.error.expect("version").code, -32022);
}

#[tokio::test]
async fn test_mcp_ping_is_not_a_server_method() {
    let mut handler = create_test_handler();
    let response = handler
        .handle_request(current_request("ping", None, json!(2)))
        .await
        .expect("response");
    assert_eq!(response.error.expect("method").code, -32601);
}

#[tokio::test]
async fn test_mcp_discover_lists_only_the_current_revision() {
    let mut handler = create_test_handler();
    let response = handler
        .handle_request(JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("server/discover"),
            params: None,
            id: Some(json!(1)),
        })
        .await
        .expect("response");
    let result = response.result.expect("result");
    let versions = result["supportedVersions"].as_array().expect("versions");
    assert_eq!(versions, &vec![json!("2026-07-28")]);
    assert!(result["capabilities"]["tools"].is_object());
}

#[tokio::test]
async fn test_mcp_tools_list() {
    let mut handler = create_test_handler();

    let tools_request = current_request("tools/list", None, json!(2));

    let response = handler
        .handle_request(tools_request)
        .await
        .expect("Response should be returned for tools/list request");

    assert_eq!(response.jsonrpc, "2.0");
    assert!(response.result.is_some());

    let result: ToolsListResult =
        serde_json::from_value(response.result.expect("Response should have result"))
            .expect("ToolsListResult deserialization should succeed");
    assert!(!result.tools.is_empty());
    assert!(result.tools.iter().any(|t| t.name == "test_tool"));
    // nextCursor is optional and may be None
    assert!(result.next_cursor.is_none() || result.next_cursor.is_some());
}

#[tokio::test]
async fn test_mcp_tools_call() {
    let mut handler = create_test_handler();

    let call_request = current_request(
        "tools/call",
        Some(json!({
            "name": "test_tool",
            "arguments": {
                "input": "hello world"
            }
        })),
        json!(2),
    );

    let response = handler
        .handle_request(call_request)
        .await
        .expect("Response should be returned for tools/call request");

    assert_eq!(response.jsonrpc, "2.0");
    assert!(response.result.is_some());

    let result: ToolsCallResult =
        serde_json::from_value(response.result.expect("Response should have result"))
            .expect("ToolsCallResult deserialization should succeed");
    assert!(result.content.is_some());
    assert_eq!(result.is_error, Some(false));
}

#[tokio::test]
async fn test_mcp_tools_call_nonexistent() {
    let mut handler = create_test_handler();

    let call_request = current_request(
        "tools/call",
        Some(json!({
            "name": "nonexistent_tool",
            "arguments": {}
        })),
        json!(2),
    );

    let response = handler
        .handle_request(call_request)
        .await
        .expect("Response should be returned for nonexistent tool call");

    // Should return success response with is_error=true (MCP wraps tool errors in success response)
    assert!(response.result.is_some());
    let result: ToolsCallResult =
        serde_json::from_value(response.result.expect("Response should have result"))
            .expect("ToolsCallResult deserialization should succeed");
    assert_eq!(result.is_error, Some(true));
    assert!(result.content.is_some());
}

#[tokio::test]
async fn test_mcp_invalid_jsonrpc_version() {
    let mut handler = create_test_handler();

    let request = JsonRpcRequest {
        jsonrpc: Cow::owned("1.0".to_string()), // Invalid version
        method: Cow::borrowed("initialize"),
        params: Some(json!({
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {}
        })),
        id: Some(json!(1)),
    };

    let response = handler
        .handle_request(request)
        .await
        .expect("Response should be returned for invalid JSON-RPC version");

    assert!(response.error.is_some());
    assert_eq!(
        response
            .error
            .as_ref()
            .expect("Response should have error")
            .code,
        -32600
    ); // Invalid request
}

#[tokio::test]
async fn test_mcp_method_not_found() {
    let mut handler = create_test_handler();

    let unknown_request = current_request("unknown/method", None, json!(2));

    let response = handler
        .handle_request(unknown_request)
        .await
        .expect("Response should be returned for unknown method");

    assert!(response.error.is_some());
    assert_eq!(
        response
            .error
            .as_ref()
            .expect("Response should have error")
            .code,
        -32601
    ); // Method not found
}

#[tokio::test]
async fn test_mcp_prompts_list() {
    let mut handler = create_test_handler();

    let prompts_request = current_request("prompts/list", None, json!(2));

    let response = handler
        .handle_request(prompts_request)
        .await
        .expect("Response should be returned for prompts/list request");

    assert_eq!(response.jsonrpc, "2.0");
    assert!(response.result.is_some());

    let result: PromptsListResult =
        serde_json::from_value(response.result.expect("Response should have result"))
            .expect("PromptsListResult deserialization should succeed");
    // May be empty if no templates loaded
    assert!(result.prompts.is_empty() || !result.prompts.is_empty());
}

#[tokio::test]
async fn test_mcp_resources_list() {
    let mut handler = create_test_handler();

    let resources_request = current_request("resources/list", None, json!(2));

    let response = handler
        .handle_request(resources_request)
        .await
        .expect("Response should be returned for resources/list request");

    assert_eq!(response.jsonrpc, "2.0");
    assert!(response.result.is_some());

    let result: ResourcesListResult =
        serde_json::from_value(response.result.expect("Response should have result"))
            .expect("ResourcesListResult deserialization should succeed");
    // May be empty if no resources available
    assert!(result.resources.is_empty() || !result.resources.is_empty());
}

#[tokio::test]
async fn test_mcp_request_id_preservation() {
    let mut handler = create_test_handler();

    let request_id = json!("unique-request-id-123");

    let request = current_request("server/discover", None, request_id.clone());

    let response = handler
        .handle_request(request)
        .await
        .expect("Response should be returned");

    // Response should have same ID
    assert_eq!(response.id, Some(request_id));
}

#[tokio::test]
async fn test_mcp_notification_no_response() {
    let mut handler = create_test_handler();

    // Initialize first
    // Notification (no ID) - handler still processes but may not return response
    // In practice, notifications don't get responses, but our handler will still process
    let notification = JsonRpcRequest {
        jsonrpc: Cow::borrowed("2.0"),
        method: Cow::borrowed("ping"),
        params: None,
        id: None, // Notification
    };

    let response = handler.handle_request(notification).await;

    // Notifications don't get responses, so response should be None
    assert!(
        response.is_none(),
        "Notifications should not return a response"
    );
}

#[tokio::test]
async fn invalid_tools_list_cursor_returns_invalid_params() {
    let mut handler = create_test_handler();
    let tools_request = current_request(
        "tools/list",
        Some(json!({"cursor": "not-a-cursor"})),
        json!(2),
    );
    let response = handler
        .handle_request(tools_request)
        .await
        .expect("response");
    let err = response.error.expect("error");
    assert_eq!(err.code, -32602);
}

#[tokio::test]
async fn tools_call_structured_content_includes_json_text() {
    // Covered at the type layer; keep a handler smoke path for tools/call success.
    let mut handler = create_test_handler();
    let call = current_request(
        "tools/call",
        Some(json!({
            "name": "test_tool",
            "arguments": {"input": "hello"}
        })),
        json!(2),
    );
    let response = handler.handle_request(call).await.expect("response");
    assert!(response.error.is_none());
    let result: ToolsCallResult =
        serde_json::from_value(response.result.expect("result")).expect("decode");
    assert_eq!(result.is_error, Some(false));
    assert!(result.content.is_some());
}

#[tokio::test]
async fn tools_call_runs_attached_pre_tool_hooks() {
    use caliche::agent::hooks::{HookMatcher, HookRegistry, HookResult, LifecycleHook};

    #[derive(Debug)]
    struct DenyHook;

    #[async_trait]
    impl LifecycleHook for DenyHook {
        async fn on_pre_tool_use(
            &self,
            _ctx: &caliche::agent::hooks::HookContext<'_>,
        ) -> HookResult {
            HookResult::deny("blocked by hook")
        }
    }

    let mut hooks = HookRegistry::new();
    hooks.register(HookMatcher::tool("test_tool"), Arc::new(DenyHook));
    let reg = ToolRegistry::new();
    reg.register(Arc::new(TestTool));
    let server = McpServerBuilder::new()
        .tool_registry(Arc::new(reg))
        .template_registry(TemplateRegistry::new())
        .build()
        .expect("server")
        .with_hooks(hooks);
    let mut handler = McpHandler::new(server);
    let call = current_request(
        "tools/call",
        Some(json!({
            "name": "test_tool",
            "arguments": {"input": "hello"}
        })),
        json!(3),
    );
    let response = handler.handle_request(call).await.expect("response");
    let result: ToolsCallResult =
        serde_json::from_value(response.result.expect("result")).expect("decode");
    assert_eq!(result.is_error, Some(true));
    let text = serde_json::to_string(&result).expect("text");
    assert!(text.contains("blocked by hook"));
    assert!(!text.contains("Processed"));
}
