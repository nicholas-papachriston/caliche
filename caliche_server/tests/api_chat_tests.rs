//! Tests for API chat endpoint
//!
//! Tests POST /api/chat endpoint with various scenarios including
//! valid requests, error handling, tool usage, and response validation.

#[path = "harness.rs"]
mod harness;

use axum::body::Body;
use axum::http::Request;
use axum::http::{StatusCode, header::CONTENT_TYPE};
use caliche::llm::ProviderType;
use caliche_server::create_router;
use serde_json::{Value, json};
use serial_test::serial;
use tower::ServiceExt;

const MOCK_TEXT: &str = "Hello from mock!";

async fn post_json(app: axum::Router, body: Value) -> axum::response::Response {
    let request = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::to_string(&body).expect("JSON serialization should succeed"),
        ))
        .expect("Request builder should succeed");
    app.oneshot(request).await.expect("Request should succeed")
}

async fn json_body(response: axum::response::Response) -> Value {
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("Body bytes should be read");
    serde_json::from_slice(&body_bytes).expect("JSON deserialization should succeed")
}

fn assert_mock_text(body: &Value) {
    let text = body
        .get("response")
        .and_then(Value::as_str)
        .expect("response text");
    assert!(text.contains(MOCK_TEXT), "mock text missing from {text}");
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_mocked() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(
        app,
        json!({
            "message": "Hello"
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_mock_text(&body);
}

#[tokio::test]
#[serial]
#[ignore = "requires LM Studio; set CALICHE_RUN_LMSTUDIO_INTEGRATION=true and run with --ignored"]
async fn test_chat_endpoint_valid_request() {
    let _config = harness::hold();
    assert_eq!(
        std::env::var("CALICHE_RUN_LMSTUDIO_INTEGRATION").as_deref(),
        Ok("true"),
        "CALICHE_RUN_LMSTUDIO_INTEGRATION must be true when this ignored test runs"
    );
    let app = create_router(ProviderType::LMStudio, "local-model".to_string(), None)
        .await
        .expect("router should build");
    let response = post_json(
        app,
        json!({
            "message": "Hello, how are you?",
            "provider": "lmstudio",
            "model": "local-model"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_missing_message() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(app, json!({"provider": "openai"})).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_invalid_provider() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(
        app,
        json!({
            "message": "Hello",
            "provider": "invalid_provider_name"
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_with_tools() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(
        app,
        json!({
            "message": "Calculate 2 + 2",
            "tools": ["calculator"]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_mock_text(&json_body(response).await);
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_with_history() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(
        app,
        json!({
            "message": "What did I say before?",
            "messages": [
                {
                    "role": "user",
                    "content": "My name is Alice"
                },
                {
                    "role": "assistant",
                    "content": "Nice to meet you, Alice!"
                }
            ]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_mock_text(&json_body(response).await);
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_response_format() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(app, json!({"message": "Hello"})).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_mock_text(&body);
    assert!(body.get("request_id").is_some());
    assert!(body.get("usage").is_some());
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_invalid_json() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let request = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header(CONTENT_TYPE, "application/json")
        .body(Body::from("invalid json"))
        .expect("Request builder should succeed");
    let response = app.oneshot(request).await.expect("Request should succeed");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_missing_content_type() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let request = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .body(Body::from(
            serde_json::to_string(&json!({"message": "Hello"}))
                .expect("JSON serialization should succeed"),
        ))
        .expect("Request builder should succeed");
    let response = app.oneshot(request).await.expect("Request should succeed");
    assert_eq!(response.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_empty_message() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(app, json!({"message": ""})).await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_mock_text(&json_body(response).await);
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_multiple_providers() {
    let _config = harness::hold();
    let bodies = [
        json!({"message": "Hello", "provider": "openai", "model": "mock-model"}),
        json!({"message": "Hello", "provider": "openai"}),
        json!({"message": "Hello"}),
    ];
    for body in bodies {
        let app = harness::mock_router().await;
        let response = post_json(app, body).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_mock_text(&json_body(response).await);
    }
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_request_id_generation() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(app, json!({"message": "Hello"})).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_mock_text(&body);
    let request_id = body.get("request_id").expect("request_id should exist");
    assert!(request_id.is_string());
    assert!(
        !request_id
            .as_str()
            .expect("request_id should be a string")
            .is_empty()
    );
}

#[tokio::test]
#[serial]
async fn test_chat_endpoint_usage_tracking() {
    let _config = harness::hold();
    let app = harness::mock_router().await;
    let response = post_json(app, json!({"message": "Hello"})).await;
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_mock_text(&body);
    let usage = body.get("usage").expect("usage should exist");
    assert!(usage.is_object());
}
