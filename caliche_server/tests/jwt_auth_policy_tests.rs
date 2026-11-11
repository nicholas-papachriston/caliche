//! JWT auth path policy and middleware routing tests.

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use axum::middleware;
use axum::routing::get;
use axum::{Extension, Router};
use caliche_server::middleware::jwt_auth::{
    AUTH_REQUIRED_MESSAGE, INVALID_TOKEN_MESSAGE, JwtAuthConfig, JwtAuthState,
    ORCHESTRATION_AUTH_REQUIRED_MESSAGE, require_jwt_middleware,
};
use caliche_server::security::{SecurityPolicy, env_bool};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde_json::json;
use serial_test::serial;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;

fn test_policy(require_auth: bool, jwt_enabled: bool) -> Arc<SecurityPolicy> {
    Arc::new(SecurityPolicy {
        jwt_enabled,
        require_auth,
        jwt_state: None,
        trusted_proxy: false,
        trusted_proxy_cidrs: Vec::new(),
        metrics_token: None,
        allow_unauth_orchestration: false,
        public_chat: false,
    })
}

#[test]
fn path_requires_auth_for_sensitive_prefixes() {
    let policy = test_policy(false, true);
    assert!(policy.path_requires_auth("/api/mcp/config", &Method::GET));
    assert!(policy.path_requires_auth("/api/agent/sessions", &Method::GET));
    assert!(policy.path_requires_auth("/api/agent/sessions/abc", &Method::GET));
    assert!(policy.path_requires_auth("/api/agent/sessions/abc/turn", &Method::POST));
    assert!(policy.path_requires_auth("/api/agent/sessions/abc/cancel", &Method::POST));
    assert!(policy.path_requires_auth("/api/agent/sessions/abc/stop", &Method::POST));
    assert!(policy.path_requires_auth("/api/cli/vault", &Method::GET));
}

#[test]
fn path_requires_auth_for_rpc_when_jwt_enabled() {
    let policy = test_policy(false, true);
    assert!(policy.path_requires_auth("/rpc", &Method::POST));
    assert!(policy.path_requires_auth("/ws", &Method::GET));
}

#[test]
fn path_allows_public_chat_stream_post() {
    let policy = test_policy(false, true);
    assert!(!policy.path_requires_auth("/api/chat/stream", &Method::POST));
    assert!(!policy.path_requires_auth("/api/chat/stream/multipart", &Method::POST));
}

#[test]
fn path_allows_gateway_webhook_ingress() {
    let policy = test_policy(true, true);
    assert!(!policy.path_requires_auth("/api/gateway/ingress/slack", &Method::POST));
}

#[test]
fn path_requires_auth_for_gateway_simulate() {
    let policy = test_policy(true, true);
    assert!(policy.path_requires_auth("/api/gateway/ingress/simulate", &Method::POST));
    assert!(policy.path_requires_auth("/api/gateway/ingress/simulate-envelope", &Method::POST));
}

#[test]
fn path_requires_auth_for_chat_stream_when_require_auth() {
    let policy = test_policy(true, true);
    assert!(policy.path_requires_auth("/api/chat/stream", &Method::POST));
    assert!(policy.path_requires_auth("/api/chat/stream/multipart", &Method::POST));
}

#[test]
#[serial]
fn path_requires_auth_for_metrics_when_require_auth_set() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_REQUIRE_AUTH", "true");
    env.set("JWT_ENABLED", "true");
    env.set("JWT_ALGORITHM", "HS256");
    env.set("JWT_PUBLIC_KEY", "metrics-test-secret");
    env.set("JWT_AUDIENCE", "caliche");
    env.set("JWT_ISSUER", "caliche");
    let policy = SecurityPolicy::from_env().expect("policy");
    assert!(policy.path_requires_auth("/metrics", &Method::GET));
}

#[tokio::test]
#[serial]
async fn require_jwt_middleware_rejects_protected_route_without_state() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_REQUIRE_AUTH", "true");
    env.set("JWT_ENABLED", "true");
    env.set("JWT_ALGORITHM", "HS256");
    env.set("JWT_PUBLIC_KEY", "middleware-test-secret");
    env.set("JWT_AUDIENCE", "caliche");
    env.set("JWT_ISSUER", "caliche");

    let policy = Arc::new(SecurityPolicy::from_env().expect("policy"));

    let app = Router::new()
        .route("/api/mcp/config", get(|| async { "ok" }))
        .layer(middleware::from_fn(require_jwt_middleware))
        .layer(Extension(policy));

    let request = Request::builder()
        .uri("/api/mcp/config")
        .body(Body::empty())
        .expect("request");

    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn require_jwt_middleware_allows_public_route() {
    let policy = test_policy(false, true);
    let app = Router::new()
        .route("/api/chat/stream", axum::routing::post(|| async { "ok" }))
        .layer(middleware::from_fn(require_jwt_middleware))
        .layer(Extension(policy));

    let request = Request::builder()
        .method(Method::POST)
        .uri("/api/chat/stream")
        .body(Body::empty())
        .expect("request");

    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn require_jwt_middleware_accepts_valid_bearer_token() {
    use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
    use serde_json::json;
    use std::time::{SystemTime, UNIX_EPOCH};

    let secret = "middleware-test-secret";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_secs() as usize;
    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "exp": now + 3600,
        "iat": now,
    });
    let token = encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .expect("token");

    let auth_state = JwtAuthState::new(JwtAuthConfig {
        enabled: true,
        public_key: Some(secret.to_string()),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    })
    .expect("jwt state");

    let policy = Arc::new(SecurityPolicy {
        jwt_enabled: true,
        require_auth: false,
        jwt_state: Some(auth_state.clone()),
        trusted_proxy: false,
        trusted_proxy_cidrs: Vec::new(),
        metrics_token: None,
        allow_unauth_orchestration: false,
        public_chat: false,
    });

    let app = Router::new()
        .route("/api/mcp/config", get(|| async { "ok" }))
        .layer(middleware::from_fn(require_jwt_middleware))
        .layer(Extension(auth_state))
        .layer(Extension(policy));

    let request = Request::builder()
        .uri("/api/mcp/config")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .expect("request");

    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
}

/// `PUT /api/policy` writes the deployment preset. With JWT on and
/// `CALICHE_REQUIRE_AUTH` unset, that write is authenticated the same way
/// as `PUT /api/settings`.
#[tokio::test]
async fn put_policy_rejects_unauthenticated_when_jwt_on_require_auth_unset() {
    let secret = "policy-route-test-secret";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_secs() as usize;
    let claims = json!({
        "sub": "operator",
        "aud": "caliche",
        "iss": "caliche",
        "exp": now + 3600,
        "iat": now,
    });
    let token = encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .expect("token");

    let auth_state = JwtAuthState::new(JwtAuthConfig {
        enabled: true,
        public_key: Some(secret.to_string()),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    })
    .expect("jwt state");

    let policy = Arc::new(SecurityPolicy {
        jwt_enabled: true,
        require_auth: false,
        jwt_state: Some(auth_state.clone()),
        trusted_proxy: false,
        trusted_proxy_cidrs: Vec::new(),
        metrics_token: None,
        allow_unauth_orchestration: false,
        public_chat: false,
    });
    assert!(policy.path_requires_auth("/api/policy", &Method::PUT));
    assert!(!policy.path_requires_auth("/api/policy", &Method::GET));
    assert!(!policy.path_requires_auth("/api/policy/effective", &Method::GET));

    let app = || {
        Router::new()
            .route("/api/policy", axum::routing::put(|| async { "ok" }))
            .layer(middleware::from_fn(require_jwt_middleware))
            .layer(Extension(auth_state.clone()))
            .layer(Extension(Arc::clone(&policy)))
    };

    let anonymous = Request::builder()
        .method(Method::PUT)
        .uri("/api/policy")
        .body(Body::empty())
        .expect("request");
    let anonymous_response = app().oneshot(anonymous).await.expect("response");
    assert_eq!(anonymous_response.status(), StatusCode::UNAUTHORIZED);
    let body = axum::body::to_bytes(anonymous_response.into_body(), usize::MAX)
        .await
        .expect("body");
    assert_eq!(String::from_utf8_lossy(&body), INVALID_TOKEN_MESSAGE);

    let authenticated = Request::builder()
        .method(Method::PUT)
        .uri("/api/policy")
        .header("authorization", format!("Bearer {token}"))
        .body(Body::empty())
        .expect("request");
    let authenticated_response = app().oneshot(authenticated).await.expect("response");
    assert_eq!(authenticated_response.status(), StatusCode::OK);
}

#[tokio::test]
async fn require_jwt_middleware_rejects_invalid_token_message() {
    let auth_state = JwtAuthState::new(JwtAuthConfig {
        enabled: true,
        public_key: Some("secret".to_string()),
        algorithm: jsonwebtoken::Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    })
    .expect("jwt state");

    let policy = Arc::new(SecurityPolicy {
        jwt_enabled: true,
        require_auth: false,
        jwt_state: Some(auth_state.clone()),
        trusted_proxy: false,
        trusted_proxy_cidrs: Vec::new(),
        metrics_token: None,
        allow_unauth_orchestration: false,
        public_chat: false,
    });

    let app = Router::new()
        .route("/api/mcp/config", get(|| async { "ok" }))
        .layer(middleware::from_fn(require_jwt_middleware))
        .layer(Extension(auth_state))
        .layer(Extension(policy));

    let request = Request::builder()
        .uri("/api/mcp/config")
        .header("authorization", "Bearer not-a-jwt")
        .body(Body::empty())
        .expect("request");

    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    assert_eq!(String::from_utf8_lossy(&body), INVALID_TOKEN_MESSAGE);
}

/// Orchestration must not spawn work for an anonymous caller, even in the development
/// posture where the rest of the API runs without JWT.
#[tokio::test]
async fn orchestration_route_is_refused_without_auth_in_development() {
    let policy = test_policy(false, false);
    let app = Router::new()
        .route(
            "/api/orchestration/multi-agent",
            axum::routing::post(|| async { "ok" }),
        )
        .layer(middleware::from_fn(require_jwt_middleware))
        .layer(Extension(policy));

    let request = Request::builder()
        .method(Method::POST)
        .uri("/api/orchestration/multi-agent")
        .body(Body::empty())
        .expect("request");

    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body");
    assert_eq!(
        String::from_utf8_lossy(&body),
        ORCHESTRATION_AUTH_REQUIRED_MESSAGE
    );
}

#[tokio::test]
async fn orchestration_route_opens_only_with_the_explicit_opt_in() {
    let policy = Arc::new(SecurityPolicy {
        jwt_enabled: false,
        require_auth: false,
        jwt_state: None,
        trusted_proxy: false,
        trusted_proxy_cidrs: Vec::new(),
        metrics_token: None,
        allow_unauth_orchestration: true,
        public_chat: false,
    });
    let app = Router::new()
        .route(
            "/api/workflows/execute",
            axum::routing::post(|| async { "ok" }),
        )
        .layer(middleware::from_fn(require_jwt_middleware))
        .layer(Extension(policy));

    let request = Request::builder()
        .method(Method::POST)
        .uri("/api/workflows/execute")
        .body(Body::empty())
        .expect("request");

    let response = app.oneshot(request).await.expect("response");
    assert_eq!(response.status(), StatusCode::OK);
}

#[test]
fn orchestration_paths_are_recognized() {
    let policy = test_policy(false, false);
    assert!(policy.orchestration_requires_auth("/api/orchestration/coordinator"));
    assert!(policy.orchestration_requires_auth("/api/orchestration/harnesses"));
    assert!(policy.orchestration_requires_auth("/api/workflows"));
    assert!(!policy.orchestration_requires_auth("/api/chat/stream"));
}

#[test]
#[serial]
fn production_profile_ignores_the_unauth_orchestration_opt_in() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_ENV", "production");
    env.set("CALICHE_ALLOW_UNAUTH_ORCHESTRATION", "true");
    env.set("CALICHE_ALLOW_INSECURE_PRODUCTION", "true");

    let policy = SecurityPolicy::from_env().expect("policy");
    assert!(!policy.allow_unauth_orchestration);
    assert!(policy.orchestration_requires_auth("/api/orchestration/multi-agent"));
}

#[test]
#[serial]
fn production_without_jwt_fails_closed() {
    #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
    // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
    let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
    env.set("CALICHE_ENV", "production");
    env.remove("CALICHE_ALLOW_INSECURE_PRODUCTION");
    env.remove("JWT_ENABLED");
    env.remove("CALICHE_REQUIRE_AUTH");

    let err = SecurityPolicy::from_env().expect_err("production without JWT");
    assert!(err.to_string().contains("JWT_ENABLED"));
}

#[test]
fn auth_messages_are_stable() {
    assert_eq!(AUTH_REQUIRED_MESSAGE, "authentication required");
    assert!(!env_bool("CALICHE_TEST_BOOL_MISSING_FOR_JWT"));
}
