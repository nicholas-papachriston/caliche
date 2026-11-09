//! Unknown permission modes return 400 before any provider call.
//!
//! These tests build only the agent execute/stream routes. They do not call
//! `create_router` and they do not open an LM Studio or localhost harness.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
use axum::routing::post;
use caliche::harness::HarnessConfig;
use caliche_server::AgentRouterState;
use caliche_server::adapter::ApiAdapter;
use caliche_server::handlers::agent_stream::{execute_task_handler, stream_task_handler};
use serde_json::json;
use tower::ServiceExt;

fn permission_app() -> Router {
    let state = AgentRouterState {
        adapter: Arc::new(ApiAdapter::new(HarnessConfig::default())),
        shared_service: None,
        settings_broadcast: None,
    };
    Router::new()
        .route("/api/agent/execute", post(execute_task_handler))
        .route("/api/agent/stream", post(stream_task_handler))
        .with_state(state)
}

async fn post_json(app: Router, uri: &str, body: serde_json::Value) -> axum::http::Response<Body> {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .expect("request");
    app.oneshot(request).await.expect("response")
}

#[tokio::test]
async fn execute_rejects_unknown_permission_mode() {
    let response = post_json(
        permission_app(),
        "/api/agent/execute",
        json!({"task": "hi", "permission_mode": "yolo"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("yolo"), "{text}");
}

#[tokio::test]
async fn execute_rejects_unknown_mode_when_accept_edits_is_set() {
    let response = post_json(
        permission_app(),
        "/api/agent/execute",
        json!({
            "task": "hi",
            "accept_edits": true,
            "permission_mode": "yolo"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("yolo"), "{text}");
}

#[tokio::test]
async fn stream_rejects_unknown_permission_mode() {
    let response = post_json(
        permission_app(),
        "/api/agent/stream",
        json!({"message": "hi", "permission_mode": "yolo"}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("yolo"), "{text}");
}
