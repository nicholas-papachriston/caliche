//! Security and authentication module.
//!
//! This module provides security features for the agent framework:
//!
//! - **Authentication**: API key and JWT validation
//! - **Authorization**: Role-based access control (RBAC)
//! - **Rate Limiting**: Token bucket algorithm for request throttling
//! - **Audit Logging**: Security event tracking (see also [`crate::observability::audit_log`] for full audit capabilities)
//!
//! # Examples
//!
//! ## API Key Authentication
//!
//! ```rust,no_run
//! use caliche::security::{ApiKeyValidator, AuthContext};
//!
//! let validator = ApiKeyValidator::new();
//! let context = AuthContext {
//!     user_id: "user-123".to_string(),
//!     tenant_id: None,
//!     roles: vec![],
//!     permissions: vec![],
//! };
//! validator.add_key("secret-api-key".to_string(), context);
//! if validator.validate("secret-api-key").is_ok() {
//!     println!("Authenticated");
//! }
//! ```
//!
//! ## Rate Limiting
//!
//! ```rust,no_run
//! use caliche::security::{RateLimiter, RateLimitConfig};
//! use std::time::Duration;
//!
//! let config = RateLimitConfig {
//!     requests_per_window: 100,
//!     window_duration: Duration::from_secs(60),
//!     burst_size: 10,
//! };
//! let limiter = RateLimiter::new(config);
//!
//! if limiter.check_rate_limit("user-123") {
//!     println!("Request allowed");
//! }
//! ```
//!
//! ## RBAC
//!
//! ```rust,no_run
//! use caliche::security::{RbacEnforcer, Role, Permission};
//!
//! let enforcer = RbacEnforcer::new();
//! let admin_role = Role {
//!     name: "admin".to_string(),
//!     permissions: vec![Permission {
//!         resource: "agents".to_string(),
//!         action: "manage".to_string(),
//!     }],
//! };
//! enforcer.add_role(admin_role);
//!
//! if enforcer.check_permission(&["admin".to_string()], "agents", "manage") {
//!     println!("Access granted");
//! }
//! ```

pub mod audit;
pub mod auth;
pub mod http_policy;
#[cfg(all(feature = "auth-jwt", feature = "reqwest"))]
pub mod jwks;
#[cfg(feature = "auth-jwt")]
pub mod jwt;
pub mod rate_limit;
pub mod rbac;
#[cfg(feature = "router")]
pub mod spend;
#[cfg(unix)]
pub mod system_binary;

pub use audit::{AuditBackend, AuditConfig, AuditConfigBuilder, AuditEvent, AuditLogger};
pub use auth::{ApiKeyValidator, AuthContext};
pub use http_policy::{HttpPolicy, ValidatedHttpUrl};
#[cfg(all(feature = "auth-jwt", feature = "reqwest"))]
pub use jwks::{JwksCache, jwk_to_decoding_key, jwk_x5c_to_decoding_key, pem_wrap_certificate};
#[cfg(feature = "auth-jwt")]
pub use jwt::{JwtConfig, JwtConfigBuilder, JwtValidator};
pub use rate_limit::{RateLimitConfig, RateLimiter};
pub use rbac::{Permission, RbacEnforcer, Role};
#[cfg(feature = "router")]
pub use spend::{BudgetExceeded, BudgetLimitType, InMemorySpendTracker, SpendTracker};
#[cfg(unix)]
pub use system_binary::SystemBinary;

pub fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

/// Whether the process is running with a production environment profile.
///
/// `CALICHE_ENV` values `production` and `prod` are production.
#[must_use]
pub fn is_production_profile() -> bool {
    std::env::var("CALICHE_ENV")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "production" | "prod"
            )
        })
        .unwrap_or(false)
}

/// Explicit opt-out for production fail-closed checks.
///
/// Set `CALICHE_ALLOW_INSECURE_PRODUCTION=1` only for break-glass
/// local debugging of a production-profiled process.
#[must_use]
pub fn allow_insecure_production() -> bool {
    env_truthy("CALICHE_ALLOW_INSECURE_PRODUCTION")
}
