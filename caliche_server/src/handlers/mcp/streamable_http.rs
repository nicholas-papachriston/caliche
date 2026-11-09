//! Streamable HTTP handler for MCP.

#[cfg(not(feature = "mcp"))]
compile_error!("Streamable HTTP handler requires the 'mcp' feature");

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use caliche::mcp::modern_header_error;
use caliche::mcp::protocol::{JsonRpcRequest, JsonRpcResponse};
use caliche::mcp::server::McpServer;
use std::sync::Arc;

/// Stateless HTTP MCP state.
///
/// Each POST builds a new handler. Protocol `2026-07-28` has no session id,
/// so cancel state stays on the stdio connection and does not cross HTTP clients.
#[derive(Clone)]
pub struct StreamableHttpState {
    pub server: Arc<McpServer>,
}

/// POST /mcp - Handle one JSON-RPC request.
pub async fn streamable_http_post_handler(
    State(state): State<StreamableHttpState>,
    headers: HeaderMap,
    Json(request): Json<JsonRpcRequest>,
) -> Response {
    if let Some(err) = modern_header_error(
        request.method.as_ref(),
        request.params.as_ref(),
        header_str(&headers, "mcp-method"),
        header_str(&headers, "mcp-protocol-version"),
        header_str(&headers, "mcp-name"),
    ) {
        let body = JsonRpcResponse::error(request.id, err);
        return (StatusCode::BAD_REQUEST, Json(body)).into_response();
    }

    if let Some(response) = dispatch_http_request(&state.server, request).await {
        return Json(response).into_response();
    }
    StatusCode::ACCEPTED.into_response()
}

/// GET /mcp is not part of protocol `2026-07-28`.
///
/// Clients call `subscriptions/listen`.
pub async fn streamable_http_get_handler(State(_state): State<StreamableHttpState>) -> Response {
    (StatusCode::METHOD_NOT_ALLOWED, "Use subscriptions/listen").into_response()
}

/// DELETE /mcp is not part of protocol `2026-07-28`.
pub async fn streamable_http_delete_handler(State(_state): State<StreamableHttpState>) -> Response {
    StatusCode::METHOD_NOT_ALLOWED.into_response()
}

async fn dispatch_http_request(
    server: &McpServer,
    request: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let mut handler = server.clone().create_handler();
    handler.handle_request(request).await
}

fn header_str<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers.get(name).and_then(|value| value.to_str().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliche::ToolRegistry;
    use caliche::mcp::protocol::LATEST_PROTOCOL_VERSION;
    use caliche::templates::TemplateRegistry;
    use serde_json::json;

    fn server() -> McpServer {
        McpServer::new(Arc::new(ToolRegistry::new()), TemplateRegistry::new())
    }

    fn request(method: &str, id: Option<i64>, extra: serde_json::Value) -> JsonRpcRequest {
        let mut body = extra;
        if let Some(map) = body.as_object_mut() {
            map.insert(
                "_meta".to_string(),
                json!({
                    "io.modelcontextprotocol/protocolVersion": LATEST_PROTOCOL_VERSION
                }),
            );
        }
        let mut value = json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "method": method,
            "params": body,
        });
        if let Some(id) = id {
            value["id"] = json!(id);
        }
        serde_json::from_value(value).expect("request")
    }

    #[tokio::test]
    async fn http_posts_do_not_share_cancel_state() {
        let server = server();
        let cancel = request("notifications/cancelled", None, json!({ "requestId": 7 }));
        assert!(dispatch_http_request(&server, cancel).await.is_none());

        let follow = request("tools/list", Some(7), json!({}));
        let response = dispatch_http_request(&server, follow)
            .await
            .expect("response");
        assert!(response.error.is_none());
    }
}
