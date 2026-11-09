use anyhow::Result;
use caliche::AgentService;
use caliche::agent::tools::ToolRegistry;
use caliche::mcp::McpServer;
use caliche::templates::TemplateRegistry;
use std::sync::Arc;

/// Build a fresh [`McpServer`] backed by `tool_registry` and the default
/// on-disk template directory. Extracted so every transport branch below
/// and [`handle_mcp_stdio`] construct the server the same way.
pub(crate) async fn build_mcp_server(tool_registry: Arc<ToolRegistry>) -> McpServer {
    let mut template_registry = TemplateRegistry::new();
    template_registry.add_template_dir("templates");
    let _ = template_registry.load_templates().await;
    McpServer::new(tool_registry, template_registry)
}

pub(crate) async fn attach_mcp_transport_routes(
    router: axum::Router,
    tool_registry: Arc<ToolRegistry>,
    transport_str: &str,
) -> axum::Router {
    use caliche_server::handlers::mcp::{
        StreamableHttpState, streamable_http_delete_handler, streamable_http_get_handler,
        streamable_http_post_handler,
    };

    let enable_http = transport_str
        .split(',')
        .map(str::trim)
        .any(|t| matches!(t, "all" | "streamable-http" | "streamable_http" | "http"));

    if !enable_http {
        return router;
    }

    let mcp_server = build_mcp_server(Arc::clone(&tool_registry)).await;
    let state = StreamableHttpState {
        server: Arc::new(mcp_server),
    };

    router.route(
        "/mcp",
        axum::routing::post(streamable_http_post_handler)
            .get(streamable_http_get_handler)
            .delete(streamable_http_delete_handler)
            .with_state(state),
    )
}

/// Handle MCP stdio mode
pub async fn handle_mcp_stdio(service: AgentService) -> Result<()> {
    use caliche::mcp::transports::stdio::run_stdio_transport;

    let tool_registry = service.tool_registry();
    let mcp_server = build_mcp_server(Arc::clone(tool_registry)).await;

    let handler = caliche::mcp::McpHandler::new(mcp_server);
    run_stdio_transport(handler).await?;

    Ok(())
}
