//! Tests for middleware context extraction
//!
//! Tests user context extraction, tenant context, request ID generation,
//! and context propagation.

use caliche_server::middleware::context::UserContext;
use serde_json::json;

#[tokio::test]
async fn test_user_context_from_claims_basic() {
    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "tenant_id": "tenant1",
        "email": "user@example.com"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert_eq!(context.user_id, "user123");
    assert_eq!(context.tenant_id, Some("tenant1".to_string()));
    assert_eq!(context.email, Some("user@example.com".to_string()));
}

#[tokio::test]
async fn test_user_context_from_claims_with_roles() {
    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "roles": ["admin", "user"]
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert_eq!(context.user_id, "user123");
    assert_eq!(context.roles.len(), 2);
    assert!(context.roles.contains(&"admin".to_string()));
    assert!(context.roles.contains(&"user".to_string()));
}

#[tokio::test]
async fn test_user_context_from_claims_single_role_string() {
    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "role": "admin"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert_eq!(context.roles.len(), 1);
    assert_eq!(context.roles[0], "admin");
}

#[tokio::test]
async fn test_user_context_from_claims_missing_sub() {
    let claims = json!({
        "email": "user@example.com"
    });

    let result = UserContext::from_claims(&claims, Some("tenant_id"));

    assert!(result.is_err());
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("Missing required claim")
    );
}

#[tokio::test]
async fn test_user_context_from_claims_user_id_alternative() {
    let claims = json!({
        "user_id": "user456",
        "tenant_id": "tenant2"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    // Should accept user_id as alternative to sub
    assert_eq!(context.user_id, "user456");
}

#[tokio::test]
async fn test_user_context_tenant_id_alternatives() {
    // Test org_id
    let claims1 = json!({
        "sub": "user1",
        "aud": "caliche",
        "iss": "caliche",
        "org_id": "org1"
    });
    let context1 = UserContext::from_claims(&claims1, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");
    assert_eq!(context1.tenant_id, Some("org1".to_string()));

    // Test organization_id
    let claims2 = json!({
        "sub": "user2",
        "aud": "caliche",
        "iss": "caliche",
        "organization_id": "org2"
    });
    let context2 = UserContext::from_claims(&claims2, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");
    assert_eq!(context2.tenant_id, Some("org2".to_string()));
}

#[tokio::test]
async fn test_user_context_custom_claims() {
    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "custom_field": "custom_value",
        "another_field": 42,
        "nested": {
            "key": "value"
        }
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

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
    assert!(context.custom_claims.get("nested").is_some());
}

#[tokio::test]
async fn test_user_context_excludes_standard_claims() {
    let claims = json!({
        "sub": "user123",
        "aud": "caliche",
        "iss": "caliche",
        "iat": 1234567890,
        "exp": 1234567890,
        "iss": "test-issuer",
        "aud": "test-audience"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    // Standard JWT claims should not be in custom_claims
    assert!(context.custom_claims.get("iat").is_none());
    assert!(context.custom_claims.get("exp").is_none());
    assert!(context.custom_claims.get("iss").is_none());
    assert!(context.custom_claims.get("aud").is_none());
}

#[tokio::test]
async fn test_user_context_tenant_id_required_success() {
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
async fn test_user_context_tenant_id_required_failure() {
    let claims = json!({
        "sub": "user123"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert!(context.tenant_id_required().is_err());
    assert!(
        context
            .tenant_id_required()
            .unwrap_err()
            .to_string()
            .contains("Tenant ID is required")
    );
}

#[tokio::test]
async fn test_user_context_empty_roles() {
    let claims = json!({
        "sub": "user123"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert!(context.roles.is_empty());
}

#[tokio::test]
async fn test_user_context_no_email() {
    let claims = json!({
        "sub": "user123"
    });

    let context = UserContext::from_claims(&claims, Some("tenant_id"))
        .expect("Creating user context from claims should succeed");

    assert_eq!(context.email, None);
}
