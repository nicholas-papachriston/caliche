/// Security module tests
use caliche::security::{
    ApiKeyValidator, AuthContext, Permission, RateLimitConfig, RateLimiter, RbacEnforcer, Role,
};
use std::time::Duration;

#[tokio::test]
async fn test_rate_limiter_creation() {
    let config = RateLimitConfig {
        requests_per_window: 60,
        window_duration: Duration::from_secs(60),
        burst_size: 10,
    };

    let limiter = RateLimiter::new(config);
    assert!(limiter.check_rate_limit("user1"));
}

#[tokio::test]
async fn test_rate_limiter_enforcement() {
    let config = RateLimitConfig {
        requests_per_window: 2,
        window_duration: Duration::from_secs(60),
        burst_size: 2,
    };

    let limiter = RateLimiter::new(config);

    // First 2 requests should pass (burst)
    assert!(limiter.check_rate_limit("user1"));
    assert!(limiter.check_rate_limit("user1"));

    // Third request should fail
    assert!(!limiter.check_rate_limit("user1"));
}

#[tokio::test]
async fn test_rate_limiter_per_user() {
    let config = RateLimitConfig {
        requests_per_window: 2,
        window_duration: Duration::from_secs(60),
        burst_size: 2,
    };

    let limiter = RateLimiter::new(config);

    // User1 exhausts their limit
    assert!(limiter.check_rate_limit("user1"));
    assert!(limiter.check_rate_limit("user1"));
    assert!(!limiter.check_rate_limit("user1"));

    // User2 should still have their limit
    assert!(limiter.check_rate_limit("user2"));
    assert!(limiter.check_rate_limit("user2"));
}

#[tokio::test]
async fn test_api_key_validator() {
    let validator = ApiKeyValidator::new();

    let context = AuthContext {
        user_id: "user1".to_string(),
        tenant_id: None,
        roles: vec!["user".to_string()],
        permissions: vec![],
    };

    // Add valid key
    validator.add_key("test_key_123".to_string(), context);

    // Validate
    let result = validator.validate("test_key_123");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().user_id, "user1");
}

#[tokio::test]
async fn test_api_key_validator_invalid() {
    let validator = ApiKeyValidator::new();

    let context = AuthContext {
        user_id: "user1".to_string(),
        tenant_id: None,
        roles: vec![],
        permissions: vec![],
    };

    validator.add_key("valid_key".to_string(), context);

    let result = validator.validate("invalid_key");
    assert!(result.is_err());
}

#[tokio::test]
async fn test_api_key_validator_revoke() {
    let validator = ApiKeyValidator::new();

    let context = AuthContext {
        user_id: "user1".to_string(),
        tenant_id: None,
        roles: vec![],
        permissions: vec![],
    };

    validator.add_key("test_key".to_string(), context);

    // Should be valid
    assert!(validator.validate("test_key").is_ok());

    // Revoke
    validator.revoke("test_key");

    // Should now be invalid
    assert!(validator.validate("test_key").is_err());
}

#[tokio::test]
async fn test_rbac_enforcer_creation() {
    let enforcer = RbacEnforcer::new();

    // Should be created successfully
    let has_permission = enforcer.check_permission(&["user".to_string()], "agent", "read");
    assert!(!has_permission); // No roles assigned yet
}

#[tokio::test]
async fn test_rbac_add_role() {
    let enforcer = RbacEnforcer::new();

    let user_role = Role {
        name: "user".to_string(),
        permissions: vec![Permission {
            resource: "agent".to_string(),
            action: "read".to_string(),
        }],
    };

    enforcer.add_role(user_role);

    // Users should have read permissions
    assert!(enforcer.check_permission(&["user".to_string()], "agent", "read"));
}

#[tokio::test]
async fn test_rbac_admin_permissions() {
    let enforcer = RbacEnforcer::new();

    let admin_role = Role {
        name: "admin".to_string(),
        permissions: vec![
            Permission {
                resource: "agent".to_string(),
                action: "read".to_string(),
            },
            Permission {
                resource: "agent".to_string(),
                action: "write".to_string(),
            },
            Permission {
                resource: "agent".to_string(),
                action: "delete".to_string(),
            },
            Permission {
                resource: "users".to_string(),
                action: "manage".to_string(),
            },
        ],
    };

    enforcer.add_role(admin_role);

    // Admins should have all permissions
    assert!(enforcer.check_permission(&["admin".to_string()], "agent", "read"));
    assert!(enforcer.check_permission(&["admin".to_string()], "agent", "write"));
    assert!(enforcer.check_permission(&["admin".to_string()], "agent", "delete"));
    assert!(enforcer.check_permission(&["admin".to_string()], "users", "manage"));
}

#[tokio::test]
async fn test_rbac_user_permissions() {
    let enforcer = RbacEnforcer::new();

    let user_role = Role {
        name: "user".to_string(),
        permissions: vec![Permission {
            resource: "agent".to_string(),
            action: "read".to_string(),
        }],
    };

    enforcer.add_role(user_role);

    // Users should have read permissions
    assert!(enforcer.check_permission(&["user".to_string()], "agent", "read"));

    // But not admin permissions
    assert!(!enforcer.check_permission(&["user".to_string()], "users", "manage"));
    assert!(!enforcer.check_permission(&["user".to_string()], "agent", "delete"));
}

#[tokio::test]
async fn test_rbac_multiple_roles() {
    let enforcer = RbacEnforcer::new();

    let user_role = Role {
        name: "user".to_string(),
        permissions: vec![Permission {
            resource: "agent".to_string(),
            action: "read".to_string(),
        }],
    };

    let admin_role = Role {
        name: "admin".to_string(),
        permissions: vec![Permission {
            resource: "users".to_string(),
            action: "manage".to_string(),
        }],
    };

    enforcer.add_role(user_role);
    enforcer.add_role(admin_role);

    assert!(enforcer.check_permission(&["user".to_string()], "agent", "read"));
    assert!(enforcer.check_permission(&["admin".to_string()], "users", "manage"));

    // User with both roles
    assert!(enforcer.check_permission(&["user".to_string(), "admin".to_string()], "agent", "read"));
    assert!(enforcer.check_permission(
        &["user".to_string(), "admin".to_string()],
        "users",
        "manage"
    ));
}

#[tokio::test]
async fn test_auth_context_creation() {
    let context = AuthContext {
        user_id: "user123".to_string(),
        tenant_id: Some("tenant1".to_string()),
        roles: vec!["user".to_string()],
        permissions: vec!["read".to_string()],
    };

    assert_eq!(context.user_id, "user123");
    assert_eq!(context.tenant_id, Some("tenant1".to_string()));
    assert_eq!(context.roles.len(), 1);
    assert_eq!(context.permissions.len(), 1);
}

#[tokio::test]
async fn test_permission_creation() {
    let permission = Permission {
        resource: "agent".to_string(),
        action: "read".to_string(),
    };

    assert_eq!(permission.resource, "agent");
    assert_eq!(permission.action, "read");
}

#[tokio::test]
async fn test_role_creation() {
    let role = Role {
        name: "test_role".to_string(),
        permissions: vec![Permission {
            resource: "agent".to_string(),
            action: "read".to_string(),
        }],
    };

    assert_eq!(role.name, "test_role");
    assert_eq!(role.permissions.len(), 1);
}

#[tokio::test]
async fn test_rate_limit_config() {
    let config = RateLimitConfig {
        requests_per_window: 100,
        window_duration: Duration::from_secs(60),
        burst_size: 20,
    };

    assert_eq!(config.requests_per_window, 100);
    assert_eq!(config.burst_size, 20);
}

#[tokio::test]
async fn test_rate_limiter_burst() {
    let config = RateLimitConfig {
        requests_per_window: 10,
        window_duration: Duration::from_secs(60),
        burst_size: 5,
    };

    let limiter = RateLimiter::new(config);

    // Should handle initial burst
    for i in 0..5 {
        assert!(
            limiter.check_rate_limit("user1"),
            "Request {} should succeed",
            i + 1
        );
    }

    // Additional requests should still work since we have requests_per_window = 10
    // But the rate will be limited by the refill rate
}

#[tokio::test]
async fn test_api_key_multiple_keys_per_user() {
    let validator = ApiKeyValidator::new();

    let context = AuthContext {
        user_id: "user1".to_string(),
        tenant_id: None,
        roles: vec![],
        permissions: vec![],
    };

    validator.add_key("key1".to_string(), context.clone());
    validator.add_key("key2".to_string(), context.clone());
    validator.add_key("key3".to_string(), context);

    // All keys should validate to same user
    assert_eq!(validator.validate("key1").unwrap().user_id, "user1");
    assert_eq!(validator.validate("key2").unwrap().user_id, "user1");
    assert_eq!(validator.validate("key3").unwrap().user_id, "user1");
}

#[tokio::test]
async fn test_rbac_readonly_role() {
    let enforcer = RbacEnforcer::new();

    let readonly_role = Role {
        name: "readonly".to_string(),
        permissions: vec![
            Permission {
                resource: "agent".to_string(),
                action: "read".to_string(),
            },
            Permission {
                resource: "metrics".to_string(),
                action: "view".to_string(),
            },
        ],
    };

    enforcer.add_role(readonly_role);

    // Should have read permissions
    assert!(enforcer.check_permission(&["readonly".to_string()], "agent", "read"));
    assert!(enforcer.check_permission(&["readonly".to_string()], "metrics", "view"));

    // Should not have write permissions
    assert!(!enforcer.check_permission(&["readonly".to_string()], "agent", "write"));
    assert!(!enforcer.check_permission(&["readonly".to_string()], "agent", "delete"));
    assert!(!enforcer.check_permission(&["readonly".to_string()], "workflow", "execute"));
}
