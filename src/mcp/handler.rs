//! MCP request handler.
//!
//! Protocol admission, cancel tracking, and method dispatch live in
//! [`caliche_mcp::router::ProviderRouter`]. This type supplies the engine
//! registries and the optional request timeout.

use crate::mcp::engine_providers::{EnginePrompts, EngineResources, EngineTools};
use crate::mcp::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::mcp::server::McpServer;
use caliche_mcp::router::{ProviderRouter, ProviderRouterConfig};
use std::sync::Arc;
use std::time::Duration;

const INSTRUCTIONS: &str =
    "This MCP server provides access to caliche tools, prompts, and resources.";

/// MCP request handler
pub struct McpHandler {
    router: ProviderRouter<EngineTools, EnginePrompts, EngineResources>,
    timeout: Option<Duration>,
}

impl McpHandler {
    /// Create a new MCP handler
    pub fn new(server: McpServer) -> Self {
        let timeout = server.request_timeout();
        let mut config = ProviderRouterConfig::new(server.get_server_info());
        config.instructions = Some(INSTRUCTIONS.to_string());
        config.advertise_tools = server.tools_enabled();
        config.advertise_prompts = server.prompts_enabled();
        config.advertise_resources = server.resources_enabled();
        let tools = Arc::new(EngineTools {
            registry: Arc::clone(server.tool_registry()),
            hooks: server.hooks().clone(),
        });
        let prompts = Arc::new(EnginePrompts {
            registry: Arc::clone(server.template_registry()),
        });
        let resources = Arc::new(EngineResources {
            collections: server.document_collections().cloned(),
            #[cfg(feature = "tools-rag")]
            rag: server.rag_tool().cloned(),
        });
        Self {
            router: ProviderRouter::new(tools, prompts, resources, config),
            timeout,
        }
    }

    /// Handle a JSON-RPC request.
    ///
    /// Returns `None` for notifications (which do not receive responses).
    /// When the configured server carries a request timeout, dispatching
    /// runs under [`tokio::time::timeout`]. A slow tool is then reported
    /// via `-32001 Request timeout`.
    pub async fn handle_request(&mut self, request: JsonRpcRequest) -> Option<JsonRpcResponse> {
        let Some(duration) = self.timeout else {
            return self.router.handle(request).await;
        };
        let id_for_timeout = request.id.clone();
        match tokio::time::timeout(duration, self.router.handle(request)).await {
            Ok(result) => result,
            Err(_) => Some(JsonRpcResponse::error(
                id_for_timeout,
                JsonRpcError::server_error(
                    -32001,
                    format!("Request timeout after {}ms", duration.as_millis()),
                    None,
                ),
            )),
        }
    }
}
