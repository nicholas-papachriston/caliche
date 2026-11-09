//! HTTP session collection list, cancel/stop, and embedded-UI root behavior.
//!
//! List and cancel are process-wide. These tests run with auth off, which is
//! the open API: a caller who can reach the process can list and cancel every
//! live session id.

#[path = "harness.rs"]
mod harness;

use axum::body::Body;
use axum::http::{Request, StatusCode, header::CONTENT_TYPE};
use caliche::llm::ProviderType;
use caliche_server::create_router;
use serde_json::{Value, json};
use serial_test::serial;
use tower::ServiceExt;

async fn test_app() -> axum::Router {
    let _config = harness::hold();
    create_router(ProviderType::LMStudio, "local-model".to_string(), None)
        .await
        .expect("router should build")
}

async fn body_json(response: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    serde_json::from_slice(&bytes).expect("json body")
}

#[tokio::test]
#[serial]
async fn get_session_collection_is_empty_array() {
    let app = test_app().await;
    let request = Request::builder()
        .method("GET")
        .uri("/api/agent/sessions")
        .body(Body::empty())
        .expect("request");
    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
    let payload = body_json(response).await;
    assert_eq!(payload, json!([]));
}

#[cfg(feature = "embedded-ui")]
async fn body_text(response: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    String::from_utf8(bytes.to_vec()).expect("utf-8 body")
}

#[cfg(not(feature = "embedded-ui"))]
#[tokio::test]
#[serial]
async fn root_without_embedded_ui_is_404_while_health_is_ok() {
    let app = test_app().await;
    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .expect("health request"),
        )
        .await
        .expect("health response");
    assert_eq!(health.status(), StatusCode::OK);

    let root = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/")
                .body(Body::empty())
                .expect("root request"),
        )
        .await
        .expect("root response");
    assert_eq!(root.status(), StatusCode::NOT_FOUND);
}

/// `./caliche rust test-embedded-ui` compiles this crate with `embedded-ui`.
/// `GET /` must be the embedded HTML document. The missing-UI sentence is a
/// startup log on binaries built without the feature, not this response.
#[cfg(feature = "embedded-ui")]
#[tokio::test]
#[serial]
async fn root_with_embedded_ui_serves_html() {
    let app = test_app().await;
    let health = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/health")
                .body(Body::empty())
                .expect("health request"),
        )
        .await
        .expect("health response");
    assert_eq!(health.status(), StatusCode::OK);

    let root = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/")
                .body(Body::empty())
                .expect("root request"),
        )
        .await
        .expect("root response");
    assert_eq!(root.status(), StatusCode::OK);
    let content_type = root
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.starts_with("text/html"),
        "content-type {content_type}"
    );
    let body = body_text(root).await;
    assert!(
        !body.contains("No embedded UI"),
        "GET / returned the missing-UI message: {body}"
    );
    let lowered = body.to_ascii_lowercase();
    assert!(
        lowered.contains("<!doctype") || lowered.contains("<html"),
        "GET / was not an HTML document: {body}"
    );
}

#[tokio::test]
#[serial]
#[expect(
    unsafe_code,
    reason = "tests set OPENROUTER_API_KEY for harness construction"
)]
async fn create_list_cancel_then_turn_is_missing() {
    let previous = std::env::var("OPENROUTER_API_KEY").ok();
    unsafe {
        std::env::set_var("OPENROUTER_API_KEY", "test-key-for-http-sessions");
    }

    let app = test_app().await;
    let create = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/agent/sessions")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(
                    r#"{"workspace":null,"context_files":[],"max_turns":1,"system_prompt":null}"#,
                ))
                .expect("create request"),
        )
        .await
        .expect("create response");
    assert_eq!(create.status(), StatusCode::OK, "create session");
    let created = body_json(create).await;
    let session_id = created["session_id"]
        .as_str()
        .expect("session id")
        .to_string();

    let listed = app
        .clone()
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/api/agent/sessions")
                .body(Body::empty())
                .expect("list request"),
        )
        .await
        .expect("list response");
    assert_eq!(listed.status(), StatusCode::OK);
    let listed_body = body_json(listed).await;
    let ids = listed_body
        .as_array()
        .expect("list array")
        .iter()
        .filter_map(|row| row["session_id"].as_str())
        .collect::<Vec<_>>();
    assert!(
        ids.contains(&session_id.as_str()),
        "process list missing {session_id}: {listed_body}"
    );

    let cancel = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/agent/sessions/{session_id}/cancel"))
                .body(Body::empty())
                .expect("cancel request"),
        )
        .await
        .expect("cancel response");
    assert_eq!(cancel.status(), StatusCode::OK);
    let cancel_body = body_json(cancel).await;
    assert_eq!(cancel_body["cancelled"], json!(true));
    assert_eq!(cancel_body["session_id"], json!(session_id));

    let turn = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/agent/sessions/{session_id}/turn"))
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(r#"{"input":"hello"}"#))
                .expect("turn request"),
        )
        .await
        .expect("turn response");
    assert_eq!(turn.status(), StatusCode::NOT_FOUND);

    let stop_again = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/api/agent/sessions/{session_id}/stop"))
                .body(Body::empty())
                .expect("stop request"),
        )
        .await
        .expect("stop response");
    assert_eq!(stop_again.status(), StatusCode::NOT_FOUND);

    match previous {
        Some(value) => unsafe {
            std::env::set_var("OPENROUTER_API_KEY", value);
        },
        None => unsafe {
            std::env::remove_var("OPENROUTER_API_KEY");
        },
    }
}
