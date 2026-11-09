//! JavaScript bindings for Caliche Security Module
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

use caliche::security::{RbacEnforcer as RustRbacEnforcer, Role as RustRole};
use napi_derive::napi;
use std::sync::Arc;
use std::time::Duration;

/// JavaScript wrapper for `AuthContext`
#[napi(object)]
pub struct AuthContext {
    pub user_id: String,
    pub tenant_id: Option<String>,
    pub roles: Vec<String>,
    pub permissions: Vec<String>,
}

impl From<caliche::security::AuthContext> for AuthContext {
    fn from(ctx: caliche::security::AuthContext) -> Self {
        Self {
            user_id: ctx.user_id,
            tenant_id: ctx.tenant_id,
            roles: ctx.roles,
            permissions: ctx.permissions,
        }
    }
}

impl From<AuthContext> for caliche::security::AuthContext {
    fn from(ctx: AuthContext) -> Self {
        Self {
            user_id: ctx.user_id,
            tenant_id: ctx.tenant_id,
            roles: ctx.roles,
            permissions: ctx.permissions,
        }
    }
}

/// JavaScript wrapper for Permission
#[napi(object)]
pub struct Permission {
    pub resource: String,
    pub action: String,
}

impl From<caliche::security::Permission> for Permission {
    fn from(perm: caliche::security::Permission) -> Self {
        Self {
            resource: perm.resource,
            action: perm.action,
        }
    }
}

impl From<Permission> for caliche::security::Permission {
    fn from(perm: Permission) -> Self {
        Self {
            resource: perm.resource,
            action: perm.action,
        }
    }
}

/// JavaScript wrapper for Role
#[napi]
pub struct Role {
    inner: caliche::security::Role,
}

#[napi]
impl Role {
    /// Create a new Role
    #[napi(constructor, catch_unwind)]
    pub fn new(name: String, permissions: Vec<Permission>) -> Self {
        Self {
            inner: RustRole {
                name,
                permissions: permissions.into_iter().map(Into::into).collect(),
            },
        }
    }

    /// Get name from Role
    #[napi(catch_unwind)]
    pub fn get_name(&self) -> String {
        self.inner.name.clone()
    }

    /// Get permissions from Role
    #[napi(catch_unwind)]
    pub fn get_permissions(&self) -> Vec<Permission> {
        self.inner
            .permissions
            .iter()
            .map(|p| Permission::from(p.clone()))
            .collect()
    }
}

/// JavaScript wrapper for `ApiKeyValidator`
#[napi]
pub struct ApiKeyValidator {
    inner: Arc<caliche::security::ApiKeyValidator>,
}

#[napi]
impl ApiKeyValidator {
    /// Create a new `ApiKeyValidator`
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(caliche::security::ApiKeyValidator::new()),
        }
    }

    /// Add an API key to the validator
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_key(&self, api_key: String, context: AuthContext) {
        self.inner.add_key(api_key, context.into());
    }

    /// Validate an API key
    #[napi(catch_unwind)]
    pub fn validate(&self, api_key: String) -> napi::Result<AuthContext> {
        self.inner
            .validate(&api_key)
            .map(Into::into)
            .map_err(|e| napi::Error::from_reason(format!("Invalid API key: {e}")))
    }

    /// Revoke an API key
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn revoke(&self, api_key: String) {
        self.inner.revoke(&api_key);
    }
}

/// JavaScript wrapper for `JwtValidator`
#[napi]
pub struct JwtValidator {
    inner: Arc<caliche::security::JwtValidator>,
}

#[napi]
impl JwtValidator {
    /// Create a new HS256 `JwtValidator` backed by a shared secret
    #[napi(constructor, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when the validator cannot be constructed from the
    /// provided secret.
    pub fn new(secret: String) -> napi::Result<Self> {
        let config = caliche::security::JwtConfig::builder()
            .algorithm("HS256")
            .public_key(secret)
            .audience(["caliche"])
            .issuer(["caliche"])
            .build();
        let inner = caliche::security::JwtValidator::from_config(&config)
            .map_err(|e| napi::Error::from_reason(format!("Invalid JWT config: {e}")))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Generate a JWT token
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn generate_token(&self, context: AuthContext, ttl_secs: u32) -> napi::Result<String> {
        self.inner
            .generate_token(&context.into(), u64::from(ttl_secs))
            .map_err(|e| napi::Error::from_reason(format!("Failed to generate token: {e}")))
    }

    /// Validate a JWT token
    #[napi(catch_unwind)]
    pub fn validate_token(&self, token: String) -> napi::Result<AuthContext> {
        self.inner
            .validate_token(&token)
            .map(Into::into)
            .map_err(|e| napi::Error::from_reason(format!("Invalid token: {e}")))
    }
}

/// JavaScript wrapper for `RateLimitConfig`
#[napi(object)]
pub struct RateLimitConfig {
    pub requests_per_window: u32,
    pub window_secs: f64,
    pub burst_size: u32,
}

impl TryFrom<caliche::security::RateLimitConfig> for RateLimitConfig {
    type Error = napi::Error;

    fn try_from(config: caliche::security::RateLimitConfig) -> napi::Result<Self> {
        Ok(Self {
            requests_per_window: config.requests_per_window,
            window_secs: u64_to_js_number(config.window_duration.as_secs(), "rate-limit window")?,
            burst_size: config.burst_size,
        })
    }
}

impl TryFrom<RateLimitConfig> for caliche::security::RateLimitConfig {
    type Error = napi::Error;

    fn try_from(config: RateLimitConfig) -> napi::Result<Self> {
        Ok(Self {
            requests_per_window: config.requests_per_window,
            window_duration: Duration::from_secs(f64_to_u64(
                config.window_secs,
                "rate-limit window",
            )?),
            burst_size: config.burst_size,
        })
    }
}

fn u64_to_js_number(value: u64, field: &str) -> napi::Result<f64> {
    const MAX_SAFE_INTEGER: u64 = (1_u64 << 53) - 1;
    if value > MAX_SAFE_INTEGER {
        return Err(napi::Error::from_reason(format!(
            "{field} exceeds JavaScript's maximum safe integer"
        )));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the preceding JavaScript safe-integer bound makes this conversion exact"
    )]
    Ok(value as f64)
}

fn f64_to_u64(value: f64, field: &str) -> napi::Result<u64> {
    let u64_exclusive_upper_bound = 2.0_f64.powi(64);
    if !value.is_finite()
        || value < 0.0
        || value >= u64_exclusive_upper_bound
        || value.fract() != 0.0
    {
        return Err(napi::Error::from_reason(format!(
            "{field} must be a non-negative integer within the 64-bit range"
        )));
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finiteness, sign, integral value, and range are checked above"
    )]
    Ok(value as u64)
}

/// JavaScript wrapper for `RateLimiter`
#[napi]
pub struct RateLimiter {
    inner: Arc<caliche::security::RateLimiter>,
}

#[napi]
impl RateLimiter {
    /// Create a new `RateLimiter`
    #[napi(constructor, catch_unwind)]
    pub fn new(config: RateLimitConfig) -> napi::Result<Self> {
        Ok(Self {
            inner: Arc::new(caliche::security::RateLimiter::new(config.try_into()?)),
        })
    }

    /// Check rate limit
    #[napi(catch_unwind)]
    pub fn check_rate_limit(&self, key: String) -> bool {
        self.inner.check_rate_limit(&key)
    }

    /// Reset rate limit for a key
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn reset(&self, key: String) {
        self.inner.reset(&key);
    }
}

/// JavaScript wrapper for `RbacEnforcer`
#[napi]
pub struct RbacEnforcer {
    inner: Arc<RustRbacEnforcer>,
}

#[napi]
impl RbacEnforcer {
    /// Create a new `RbacEnforcer`
    #[napi(constructor, catch_unwind)]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(caliche::security::RbacEnforcer::new()),
        }
    }

    /// Add a role to the enforcer
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_role(&self, name: String, permissions: Vec<Permission>) {
        let role = RustRole {
            name,
            permissions: permissions.into_iter().map(Into::into).collect(),
        };
        self.inner.add_role(role);
    }

    /// Check if user has permission
    #[napi(catch_unwind)]
    pub fn check_permission(&self, roles: Vec<String>, resource: String, action: String) -> bool {
        self.inner.check_permission(&roles, &resource, &action)
    }
}
