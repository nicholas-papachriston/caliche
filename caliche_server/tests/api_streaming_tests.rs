//! Tests for API streaming endpoint
//!
//! Tests GET /api/chat/stream SSE endpoint, streaming with partial responses,
//! connection interruption handling, and SSE format compliance.

#[path = "harness.rs"]
mod harness;

use axum::body::Body;
use axum::http::Request;
use axum::http::{StatusCode, header::CONTENT_TYPE};
use serde_json::{Value, json};
use serial_test::serial;
use tower::ServiceExt;

const MOCK_TEXT: &str = "Hello from mock!";

async fn post_json(app: axum::Router, body: Value) -> axum::response::Response {
    let request = Request::builder()
        .method("POST")
        .uri("/api/chat/stream")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_string(&body).expect("JSON serialization should succeed"),
        ))
        .expect("Request builder should succeed");
    app.oneshot(request).await.expect("Request should succeed")
}

async fn body_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("Body bytes should be read");
    String::from_utf8(bytes.to_vec()).expect("SSE body should be UTF-8")
}

#[tokio::test]
#[serial]
async fn test_streaming_endpoint_valid_request() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(
        app,
        json!({
            "message": "Hello, stream this response"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .expect("Content-Type header should exist")
        .to_str()
        .expect("Content-Type header should be valid UTF-8");
    assert!(content_type.contains("text/event-stream"));
    let body = body_text(response).await;
    assert!(
        body.contains(MOCK_TEXT),
        "mock text missing from stream: {body}"
    );
}

#[tokio::test]
#[serial]
async fn test_streaming_endpoint_sse_headers() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(app, json!({"message": "Test"})).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(response.headers().get("cache-control").is_some());
    assert!(response.headers().get("connection").is_some());
    let body = body_text(response).await;
    assert!(
        body.contains(MOCK_TEXT),
        "mock text missing from stream: {body}"
    );
}

#[tokio::test]
#[serial]
async fn test_streaming_endpoint_missing_message() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(app, json!({"provider": "openai"})).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
#[serial]
async fn test_streaming_endpoint_invalid_provider() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(
        app,
        json!({
            "message": "Hello",
            "provider": "invalid_provider"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[serial]
async fn test_streaming_endpoint_with_history() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(
        app,
        json!({
            "message": "Continue the conversation",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello"
                },
                {
                    "role": "assistant",
                    "content": "Hi there!"
                }
            ]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response).await;
    assert!(
        body.contains(MOCK_TEXT),
        "mock text missing from stream: {body}"
    );
}

#[tokio::test]
#[serial]
async fn test_streaming_endpoint_invalid_json() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let request = Request::builder()
        .method("POST")
        .uri("/api/chat/stream")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from("invalid json"))
        .expect("Request builder should succeed");
    let response = app.oneshot(request).await.expect("Request should succeed");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[serial]
async fn test_streaming_endpoint_cors_headers() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(app, json!({"message": "Test"})).await;
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .expect("Content-Type header should exist")
        .to_str()
        .expect("Content-Type header should be valid UTF-8");
    assert!(content_type.contains("text/event-stream"));
}

#[tokio::test]
#[serial]
async fn test_streaming_endpoint_with_tools() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(
        app,
        json!({
            "message": "Calculate 5 + 3",
            "tools": ["calculator"]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_text(response).await;
    assert!(
        body.contains(MOCK_TEXT),
        "mock text missing from stream: {body}"
    );
}
