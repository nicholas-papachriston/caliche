//! Tests for JWT authentication middleware
//!
//! Tests JWT validation, token extraction, error handling, and user context creation.

use axum::http::{HeaderMap, HeaderValue};
use caliche_server::middleware::context::UserContext;
use caliche_server::middleware::jwt_auth::{JwtAuthConfig, JwtAuthState};
use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    tenant_id: Option<String>,
    email: Option<String>,
    roles: Option<Vec<String>>,
    exp: usize,
    iat: usize,
    aud: String,
    iss: String,
}

fn create_test_token(claims: Claims, secret: &str) -> String {
    let header = Header::new(Algorithm::HS256);
    let key = EncodingKey::from_secret(secret.as_bytes());
    encode(&header, &claims, &key).expect("JWT encoding should succeed")
}

fn create_expired_token(secret: &str) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time calculation should succeed")
        .as_secs() as usize;

    let claims = Claims {
        sub: "user123".to_string(),
        tenant_id: Some("tenant1".to_string()),
        email: Some("user@example.com".to_string()),
        roles: Some(vec!["user".to_string()]),
        exp: now - 3600, // Expired 1 hour ago
        iat: now - 7200,
        aud: "caliche".to_string(),
        iss: "caliche".to_string(),
    };

    create_test_token(claims, secret)
}

fn create_valid_token(secret: &str) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time calculation should succeed")
        .as_secs() as usize;

    let claims = Claims {
        sub: "user123".to_string(),
        tenant_id: Some("tenant1".to_string()),
        email: Some("user@example.com".to_string()),
        roles: Some(vec!["user".to_string(), "admin".to_string()]),
        exp: now + 3600, // Valid for 1 hour
        iat: now,
        aud: "caliche".to_string(),
        iss: "caliche".to_string(),
    };

    create_test_token(claims, secret)
}

#[tokio::test]
async fn test_jwt_auth_config_default() {
    let config = JwtAuthConfig::default();

    assert!(!config.enabled);
    assert_eq!(config.algorithm, Algorithm::RS256);
    assert_eq!(config.tenant_claim, "tenant_id");
    assert!(config.required_claims.contains(&"sub".to_string()));
    assert!(config.validate_exp);
    assert!(config.audience.is_empty());
    assert!(config.issuer.is_empty());
}

#[tokio::test]
async fn test_jwt_auth_state_disabled() {
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

    // Should return None when disabled
    assert!(context.is_none());
}

#[tokio::test]
async fn test_jwt_extraction_missing_header() {
    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some("test-secret-key".to_string()),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    let headers = HeaderMap::new();
    let result = state.extract_user_context(&headers).await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Missing Authorization header")
    );
}

#[tokio::test]
async fn test_jwt_extraction_invalid_format() {
    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some("test-secret-key".to_string()),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        HeaderValue::from_static("InvalidFormat token"),
    );

    let result = state.extract_user_context(&headers).await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Invalid Authorization header format")
    );
}

#[tokio::test]
async fn test_jwt_extraction_valid_token() {
    let secret = "test-secret-key-for-validation";
    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some(secret.to_string()),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    let token = create_valid_token(secret);
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
    assert_eq!(ctx.email, Some("user@example.com".to_string()));
    assert!(ctx.roles.contains(&"user".to_string()));
    assert!(ctx.roles.contains(&"admin".to_string()));
}

#[tokio::test]
async fn test_jwt_extraction_expired_token() {
    let secret = "test-secret-key";
    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some(secret.to_string()),
        algorithm: Algorithm::HS256,
        validate_exp: true,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    let token = create_expired_token(secret);
    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {}", token))
            .expect("Header value creation should succeed"),
    );

    let result = state.extract_user_context(&headers).await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("JWT validation failed")
    );
}

#[tokio::test]
async fn test_jwt_extraction_invalid_signature() {
    let secret = "test-secret-key";
    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some(secret.to_string()),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    // Token signed with different secret
    let wrong_token = create_valid_token("different-secret");
    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        HeaderValue::from_str(&format!("Bearer {}", wrong_token))
            .expect("Header value creation should succeed"),
    );

    let result = state.extract_user_context(&headers).await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("JWT validation failed")
    );
}

#[tokio::test]
async fn test_jwt_extraction_malformed_token() {
    let config = JwtAuthConfig {
        enabled: true,
        public_key: Some("test-secret-key".to_string()),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config).expect("JWT auth state creation should succeed");

    let mut headers = HeaderMap::new();
    headers.insert(
        "Authorization",
        HeaderValue::from_static("Bearer not.a.valid.token"),
    );

    let result = state.extract_user_context(&headers).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_jwt_extraction_missing_sub_claim() {
    let secret = "test-secret-key";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time calculation should succeed")
        .as_secs() as usize;

    // Token without 'sub' claim
    let claims = json!({
        "user_id": "user123", // Wrong claim name
        "aud": "caliche",
        "iss": "caliche",
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

    let result = state.extract_user_context(&headers).await;

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Missing required claim")
    );
}

#[tokio::test]
async fn test_jwt_extraction_with_custom_claims() {
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
        "custom_field": "custom_value",
        "another_field": 42,
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
        .expect("Extracting user context should succeed")
        .expect("Context should be present");

    // Custom claims should be preserved
    assert_eq!(
        context
            .custom_claims
            .get("custom_field")
            .expect("Custom claim should exist")
            .as_str()
            .expect("Custom claim should be a string"),
        "custom_value"
    );
    assert_eq!(
        context
            .custom_claims
            .get("another_field")
            .expect("Custom claim should exist")
            .as_i64()
            .expect("Custom claim should be a number"),
        42
    );
}

#[tokio::test]
async fn test_jwt_extraction_tenant_id_variations() {
    let secret = "test-secret-key";

    // Test with tenant_id
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time calculation should succeed")
        .as_secs() as usize;

    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "org_id": "org1", // Alternative claim name
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
        .expect("Extracting user context should succeed")
        .expect("Context should be present");

    // Should extract org_id as tenant_id
    assert_eq!(context.tenant_id, Some("org1".to_string()));
}

#[tokio::test]
async fn test_jwt_extraction_roles_as_string() {
    let secret = "test-secret-key";
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time calculation should succeed")
        .as_secs() as usize;

    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "role": "admin", // Single role as string
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
        .expect("Extracting user context should succeed")
        .expect("Context should be present");

    // Should convert single role string to array
    assert_eq!(context.roles, vec!["admin"]);
}

#[tokio::test]
async fn test_jwt_config_with_public_key_path() {
    use std::fs;
    use tempfile::NamedTempFile;

    // Create temporary key file
    let key_file = NamedTempFile::new().expect("Temp file creation should succeed");
    let key_path = key_file.path();
    fs::write(key_path, "test-secret-key").expect("Writing key file should succeed");

    let config = JwtAuthConfig {
        enabled: true,
        public_key_path: Some(
            key_path
                .to_str()
                .expect("Path should be valid UTF-8")
                .to_string(),
        ),
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config);

    // Should succeed loading from file
    assert!(state.is_ok());
}

#[tokio::test]
async fn test_jwt_config_validation_no_key_source() {
    let config = JwtAuthConfig {
        enabled: true,
        public_key: None,
        public_key_path: None,
        jwks_url: None,
        algorithm: Algorithm::HS256,
        audience: vec!["caliche".to_string()],
        issuer: vec!["caliche".to_string()],
        ..Default::default()
    };

    let state = JwtAuthState::new(config);

    // Should fail when enabled but no key source
    assert!(state.is_err());
}

#[tokio::test]
async fn test_user_context_from_claims() {
    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "tenant_id": "tenant1",
        "email": "user@example.com",
        "roles": ["user", "admin"],
        "custom": "value"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert_eq!(context.user_id, "user123");
    assert_eq!(context.tenant_id, Some("tenant1".to_string()));
    assert_eq!(context.email, Some("user@example.com".to_string()));
    assert_eq!(context.roles.len(), 2);
    assert!(context.roles.contains(&"user".to_string()));
    assert!(context.roles.contains(&"admin".to_string()));
    assert_eq!(
        context
            .custom_claims
            .get("custom")
            .expect("Custom claim should exist")
            .as_str()
            .expect("Custom claim should be a string"),
        "value"
    );
}

#[tokio::test]
async fn test_user_context_tenant_id_required() {
    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "tenant_id": "tenant1"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert_eq!(
        context
            .tenant_id_required()
            .expect("Tenant ID should be present"),
        "tenant1"
    );
}

#[tokio::test]
async fn test_user_context_tenant_id_required_missing() {
    let claims = json!({
        "sub": "user123"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert!(context.tenant_id_required().is_err());
}
