//! Integration tests for middleware stack
//!
//! Tests middleware execution order, short-circuiting, rate limiting integration,
//! and error handling across middleware layers.

// Note: These tests would require setting up a full Axum router with middleware
// For now, we test the individual components. Full integration tests would
// require more complex setup with actual HTTP server.

#[path = "harness.rs"]
mod harness;

use axum::http::{HeaderMap, HeaderValue};
use caliche_server::middleware::jwt_auth::{JwtAuthConfig, JwtAuthState};

#[tokio::test]
#[serial_test::serial]
async fn test_middleware_auth_disabled_allows_request() {
    let config = JwtAuthConfig {
        enabled: false,
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    let headers = HeaderMap::new();
    let context = state
        .extract_user_context(&headers)
        .await
        .expect("Extracting user context should succeed");

    // When disabled, should return None (allows request)
    assert!(context.is_none());
}

#[tokio::test]
#[serial_test::serial]
async fn test_middleware_auth_enabled_requires_token() {
    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some("test-secret".to_string()),
        algorithm: jsonwebtoken::Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    let headers = HeaderMap::new();
    let result = state.extract_user_context(&headers).await;

    // Should fail without token
    assert!(result.is_err());
}

#[tokio::test]
#[serial_test::serial]
async fn test_middleware_context_extraction_success() {
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    let secret = "test-secret-key";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time calculation should succeed")
        .as_secs() as usize;

    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "tenant_id": "tenant1",
        "email": "user@example.com",
        "roles": ["admin"],
        "exp": now + 3600,
        "iat": now
    });

    let header = Header::new(Algorithm::HS256);
    let key = EncodingKey::from_secret(secret.as_bytes());
    let token = encode(&header, &claims, &key).expect("JWT encoding should succeed");

    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some(secret.to_string()),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {}", token))
            .expect("Header value creation should succeed"),
    );

    let context = state
        .extract_user_context(&headers)
        .await
        .expect("Extracting user context should succeed");

    assert!(context.is_some());
    let ctx = context.expect("Context should be present");
    assert_eq!(ctx.user_id, "user123");
    assert_eq!(ctx.tenant_id, Some("tenant1".to_string()));
}

#[tokio::test]
#[serial_test::serial]
async fn test_middleware_error_propagation() {
    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some("test-secret".to_string()),
        algorithm: jsonwebtoken::Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    // Invalid token format
    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        HeaderValue::from_static("Bearer invalid.token.here"),
    );

    let result = state.extract_user_context(&headers).await;

    // Should propagate error
    assert!(result.is_err());
}

#[tokio::test]
#[serial_test::serial]
async fn test_middleware_optional_user_context() {
    // Test that OptionalUserContext extractor would work
    // This is tested through the actual API handlers in integration tests
    // Here we verify the underlying state works correctly

    let config = JwtAuthConfig {
        enabled: false, // Disabled = optional
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");
    let headers = HeaderMap::new();
    let context = state
        .extract_user_context(&headers)
        .await
        .expect("Extracting user context should succeed");

    // Should return None when disabled (allows optional auth)
    assert!(context.is_none());
}

#[tokio::test]
#[serial_test::serial]
async fn json_rpc_only_blocks_rest_when_enabled() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let _config = harness::hold();
    env.set("CALICHE_JSON_RPC_ONLY", "true");

    let app = harness::mock_router().await;

    // A non-allowlisted /api/* route must return 410 GONE.
    let request = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"message":"hi"}"#))
        .expect("request builder should succeed");
    let response = app
        .clone()
        .oneshot(request)
        .await
        .expect("request should succeed");
    assert_eq!(response.status(), StatusCode::GONE);

    // Non-streaming chat multipart is REST, not an SSE exception.
    let multipart_request = Request::builder()
        .method("POST")
        .uri("/api/chat/multipart")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"message":"hi"}"#))
        .expect("request builder should succeed");
    let multipart_response = app
        .clone()
        .oneshot(multipart_request)
        .await
        .expect("request should succeed");
    assert_eq!(multipart_response.status(), StatusCode::GONE);

    // Allowlisted endpoints must still route.
    let health_request = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("request builder should succeed");
    let health_response = app
        .clone()
        .oneshot(health_request)
        .await
        .expect("request should succeed");
    assert_ne!(health_response.status(), StatusCode::GONE);

    let metrics_request = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .expect("request builder should succeed");
    let metrics_response = app
        .clone()
        .oneshot(metrics_request)
        .await
        .expect("request should succeed");
    assert_ne!(metrics_response.status(), StatusCode::GONE);

    // /api/chat/stream is in the allowlist and must not return 410.
    let stream_request = Request::builder()
        .method("POST")
        .uri("/api/chat/stream")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"message":"hi"}"#))
        .expect("request builder should succeed");
    let stream_response = app
        .oneshot(stream_request)
        .await
        .expect("request should succeed");
    assert_eq!(stream_response.status(), StatusCode::OK);

    env.remove("CALICHE_JSON_RPC_ONLY");
}

#[tokio::test]
#[serial_test::serial]
async fn prometheus_exposes_app_metrics() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let _config = harness::hold();
    env.remove("CALICHE_JSON_RPC_ONLY");

    let app = harness::mock_router().await;

    let request = Request::builder()
        .method("GET")
        .uri("/metrics")
        .body(Body::empty())
        .expect("request builder should succeed");
    let response = app.oneshot(request).await.expect("request should succeed");
    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(
        content_type.starts_with("text/plain"),
        "expected text/plain content-type, got {content_type}"
    );
    let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
        .await
        .expect("body read");
    let body_str = std::str::from_utf8(&bytes).expect("utf8");
    // The MetricsRecorder registers these HELP lines on the app-local registry.
    assert!(
        body_str.contains("agent_requests_total"),
        "/metrics output missing agent_requests_total: {body_str}"
    );
    assert!(
        body_str.contains("agent_request_duration_seconds"),
        "/metrics output missing agent_request_duration_seconds: {body_str}"
    );
}

#[tokio::test]
#[serial_test::serial]
async fn chat_stream_mounted_by_default() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    let _config = harness::hold();
    env.remove("CALICHE_JSON_RPC_ONLY");

    let app = harness::mock_router().await;

    let request = Request::builder()
        .method("POST")
        .uri("/api/chat/stream")
        .header("content-type", "application/json")
        .body(Body::from(r#"{"message":"hi"}"#))
        .expect("request builder should succeed");
    let response = app.oneshot(request).await.expect("request should succeed");
    assert_eq!(response.status(), StatusCode::OK);
}
