#![cfg(feature = "mcp")]

//! Tests for MCP server implementation
//!
//! Tests server initialization, capability registration, tool/prompt/resource listing,
//! and server lifecycle management.

use beef::lean::Cow;
use caliche::ToolRegistry;
use caliche::mcp::McpHandler;
use caliche::mcp::protocol::JsonRpcRequest;
use caliche::mcp::server::{McpServer, McpServerConfig};
use caliche::templates::TemplateRegistry;
use std::sync::Arc;

fn config(
    name: &'static str,
    version: Option<&'static str>,
    (tools, prompts, resources): (bool, bool, bool),
) -> McpServerConfig {
    McpServerConfig {
        name: Cow::borrowed(name),
        version: version.map(Cow::borrowed),
        enable_tools: tools,
        enable_prompts: prompts,
        enable_resources: resources,
        request_timeout: None,
    }
}

#[tokio::test]
async fn test_mcp_server_creation() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::new(tool_registry, template_registry);

    assert_eq!(server.get_server_info().name, "caliche");
}

#[tokio::test]
async fn test_mcp_server_with_config() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::with_config(
        tool_registry,
        template_registry,
        config("test_server", Some("1.0.0"), (true, false, false)),
    );

    assert_eq!(&*server.get_server_info().name, "test_server");
    assert_eq!(server.get_server_info().version.as_deref(), Some("1.0.0"));
}

#[tokio::test]
async fn test_mcp_server_capabilities_all_enabled() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::new(tool_registry, template_registry);

    let capabilities = server.get_capabilities();

    assert!(capabilities.tools.is_some());
    assert!(capabilities.prompts.is_some());
    assert!(capabilities.resources.is_some());
}

#[tokio::test]
async fn test_mcp_server_capabilities_selective() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::with_config(
        tool_registry,
        template_registry,
        config("test", None, (true, false, false)),
    );

    let capabilities = server.get_capabilities();

    assert!(capabilities.tools.is_some());
    assert!(capabilities.prompts.is_none());
    assert!(capabilities.resources.is_none());
}

#[tokio::test]
async fn test_mcp_server_tool_registry_access() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::new(Arc::clone(&tool_registry), template_registry);

    let retrieved_registry = server.tool_registry();
    assert_eq!(Arc::as_ptr(retrieved_registry), Arc::as_ptr(&tool_registry));
}

#[tokio::test]
async fn test_mcp_server_template_registry_access() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::new(tool_registry, template_registry);

    let retrieved_registry = server.template_registry();
    assert!(retrieved_registry.list().is_empty());
}

#[tokio::test]
async fn test_mcp_server_default_config() {
    let cfg = McpServerConfig::default();

    assert_eq!(&*cfg.name, "caliche");
    assert!(cfg.version.is_some());
    assert!(cfg.enable_tools);
    assert!(cfg.enable_prompts);
    assert!(cfg.enable_resources);
    assert!(cfg.request_timeout.is_none());
}

#[tokio::test]
async fn test_mcp_server_capabilities_tools_only() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::with_config(
        tool_registry,
        template_registry,
        config("tools_only", None, (true, false, false)),
    );

    let capabilities = server.get_capabilities();

    assert!(capabilities.tools.is_some());
    assert_eq!(
        capabilities
            .tools
            .as_ref()
            .expect("Tools capability should exist")
            .list_changed,
        Some(false)
    );
    assert!(capabilities.prompts.is_none());
    assert!(capabilities.resources.is_none());
}

#[tokio::test]
async fn test_mcp_server_capabilities_prompts_only() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::with_config(
        tool_registry,
        template_registry,
        config("prompts_only", None, (false, true, false)),
    );

    let capabilities = server.get_capabilities();

    assert!(capabilities.tools.is_none());
    assert!(capabilities.prompts.is_some());
    assert_eq!(
        capabilities
            .prompts
            .as_ref()
            .expect("Prompts capability should exist")
            .list_changed,
        Some(false)
    );
    assert!(capabilities.resources.is_none());
}

#[tokio::test]
async fn test_mcp_server_capabilities_resources_only() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::with_config(
        tool_registry,
        template_registry,
        config("resources_only", None, (false, false, true)),
    );

    let capabilities = server.get_capabilities();

    assert!(capabilities.tools.is_none());
    assert!(capabilities.prompts.is_none());
    assert!(capabilities.resources.is_some());
    assert_eq!(
        capabilities
            .resources
            .as_ref()
            .expect("Resources capability should exist")
            .subscribe,
        Some(false)
    );
}

#[cfg(feature = "tools-rag")]
#[tokio::test]
async fn test_mcp_server_rag_tool_access() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::new(tool_registry, template_registry).with_rag_tool(None);

    assert!(server.rag_tool().is_none());
}

#[tokio::test]
async fn test_mcp_server_document_collections_access() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::new(tool_registry, template_registry).with_document_collections(None);

    assert!(server.document_collections().is_none());
}

#[tokio::test]
async fn test_mcp_server_info_version() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::with_config(
        tool_registry,
        template_registry,
        config("versioned_server", Some("2.5.1"), (true, true, true)),
    );

    let info = server.get_server_info();
    assert_eq!(&*info.name, "versioned_server");
    assert_eq!(info.version.as_deref(), Some("2.5.1"));
}

#[tokio::test]
async fn test_mcp_server_info_no_version() {
    let tool_registry = Arc::new(ToolRegistry::new());
    let template_registry = TemplateRegistry::new();

    let server = McpServer::with_config(
        tool_registry,
        template_registry,
        config("unversioned", None, (true, true, true)),
    );

    let info = server.get_server_info();
    assert_eq!(&*info.name, "unversioned");
    assert_eq!(info.version, None);
}

#[tokio::test]
async fn modern_request_skips_initialize() {
    let server = McpServer::new(Arc::new(ToolRegistry::new()), TemplateRegistry::new());
    let mut handler = McpHandler::new(server);
    let response = handler
        .handle_request(JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("server/discover"),
            params: Some(serde_json::json!({
                "_meta": { "io.modelcontextprotocol/protocolVersion": "2026-07-28" }
            })),
            id: Some(serde_json::json!(1)),
        })
        .await
        .expect("discover response");
    assert!(response.error.is_none());
    let result = response.result.expect("result");
    assert_eq!(result["resultType"], "complete");
    let versions = result["supportedVersions"].as_array().expect("versions");
    assert!(versions.iter().any(|version| version == "2026-07-28"));

    let blocked = handler
        .handle_request(JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("tools/list"),
            params: None,
            id: Some(serde_json::json!(2)),
        })
        .await
        .expect("legacy response");
    assert_eq!(blocked.error.expect("unsupported").code, -32022);
}

#[tokio::test]
async fn subscriptions_listen_and_cancel_before_start() {
    let server = McpServer::new(Arc::new(ToolRegistry::new()), TemplateRegistry::new());
    let mut handler = McpHandler::new(server);
    let listen = handler
        .handle_request(JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("subscriptions/listen"),
            params: Some(serde_json::json!({
                "_meta": { "io.modelcontextprotocol/protocolVersion": "2026-07-28" }
            })),
            id: Some(serde_json::json!(3)),
        })
        .await
        .expect("listen");
    let result = listen.result.expect("result");
    assert_eq!(result["subscriptionId"], "3");
    assert_eq!(result["resultType"], "complete");

    let note = handler
        .handle_request(JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("notifications/cancelled"),
            params: Some(serde_json::json!({"requestId": 8})),
            id: None,
        })
        .await;
    assert!(note.is_none());
    let cancelled = handler
        .handle_request(JsonRpcRequest {
            jsonrpc: Cow::borrowed("2.0"),
            method: Cow::borrowed("tools/list"),
            params: None,
            id: Some(serde_json::json!(8)),
        })
        .await
        .expect("cancelled");
    assert_eq!(cancelled.error.expect("cancelled").code, -32800);
}
