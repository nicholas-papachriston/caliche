//! Session ownership / IDOR guard tests.

use axum::http::StatusCode;
use caliche::harness::config::HarnessConfig;
use caliche_server::adapter::SessionOwner;
use caliche_server::middleware::context::UserContext;
use caliche_server::middleware::session_auth::{
    authorize_session_access, verify_session_ownership,
};
use caliche_server::security::SecurityPolicy;
use caliche_server::{ApiAdapter, SecurityPolicy as Policy};
use std::sync::Arc;
use std::time::Duration;

fn user(user_id: &str, tenant_id: Option<&str>) -> UserContext {
    UserContext {
        user_id: user_id.to_string(),
        tenant_id: tenant_id.map(str::to_string),
        email: None,
        roles: Vec::new(),
        custom_claims: Default::default(),
    }
}

fn enforced_policy() -> SecurityPolicy {
    SecurityPolicy {
        jwt_enabled: true,
        require_auth: false,
        jwt_state: None,
        trusted_proxy: false,
        trusted_proxy_cidrs: Vec::new(),
        metrics_token: None,
        allow_unauth_orchestration: false,
        public_chat: false,
    }
}

fn open_policy() -> SecurityPolicy {
    SecurityPolicy {
        jwt_enabled: false,
        require_auth: false,
        jwt_state: None,
        trusted_proxy: false,
        trusted_proxy_cidrs: Vec::new(),
        metrics_token: None,
        allow_unauth_orchestration: false,
        public_chat: false,
    }
}

#[test]
fn verify_session_ownership_accepts_matching_owner() {
    let owner = SessionOwner {
        user_id: Some("alice".to_string()),
        tenant_id: Some("tenant-a".to_string()),
    };
    let caller = user("alice", Some("tenant-a"));
    verify_session_ownership(&owner, &caller).expect("matching owner should pass");
}

#[test]
fn verify_session_ownership_rejects_user_mismatch() {
    let owner = SessionOwner {
        user_id: Some("alice".to_string()),
        tenant_id: None,
    };
    let caller = user("bob", None);
    let err = verify_session_ownership(&owner, &caller).expect_err("user mismatch should fail");
    assert_eq!(err.0, StatusCode::FORBIDDEN);
}

#[test]
fn verify_session_ownership_rejects_tenant_mismatch() {
    let owner = SessionOwner {
        user_id: Some("alice".to_string()),
        tenant_id: Some("tenant-a".to_string()),
    };
    let caller = user("alice", Some("tenant-b"));
    let err = verify_session_ownership(&owner, &caller).expect_err("tenant mismatch should fail");
    assert_eq!(err.0, StatusCode::FORBIDDEN);
}

#[test]
fn verify_session_ownership_rejects_missing_owner_user_id() {
    let owner = SessionOwner::default();
    let caller = user("alice", None);
    let err = verify_session_ownership(&owner, &caller).expect_err("missing owner should fail");
    assert_eq!(err.0, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn authorize_session_skipped_when_auth_not_required() {
    let adapter =
        ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
    let caller = user("alice", None);
    authorize_session_access(&adapter, "missing-session", &caller, &open_policy())
        .await
        .expect("open policy should bypass ownership checks");
}

#[tokio::test]
async fn authorize_session_not_found_when_enforced() {
    let adapter =
        ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None);
    let caller = user("alice", None);
    let err = authorize_session_access(&adapter, "missing-session-id", &caller, &enforced_policy())
        .await
        .expect_err("unknown session should 404");
    assert_eq!(err.0, StatusCode::NOT_FOUND);
}

#[tokio::test]
#[serial_test::serial]
async fn authorize_session_rejects_idor_for_existing_session() {
    let previous = std::env::var("OPENROUTER_API_KEY").ok();
    unsafe {
        std::env::set_var("OPENROUTER_API_KEY", "test-key-for-session-auth");
    }

    let adapter =
        ApiAdapter::with_session_ttl(HarnessConfig::default(), Duration::from_secs(60), None)
            .with_security(Arc::new(Policy {
                jwt_enabled: true,
                require_auth: false,
                jwt_state: None,
                trusted_proxy: false,
                trusted_proxy_cidrs: Vec::new(),
                metrics_token: None,
                allow_unauth_orchestration: false,
                public_chat: false,
            }));
    let owner = SessionOwner {
        user_id: Some("alice".to_string()),
        tenant_id: Some("tenant-a".to_string()),
    };
    let session_id = adapter
        .create_session(None, Some(owner), None)
        .await
        .expect("session should be created");

    let attacker = user("bob", Some("tenant-a"));
    let err = authorize_session_access(&adapter, &session_id, &attacker, &enforced_policy())
        .await
        .expect_err("foreign user should be forbidden");
    assert_eq!(err.0, StatusCode::FORBIDDEN);

    let legitimate = user("alice", Some("tenant-a"));
    authorize_session_access(&adapter, &session_id, &legitimate, &enforced_policy())
        .await
        .expect("owner should be allowed");

    unsafe {
        match previous {
            Some(value) => std::env::set_var("OPENROUTER_API_KEY", value),
            None => std::env::remove_var("OPENROUTER_API_KEY"),
        }
    }
}
