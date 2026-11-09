//! Python bindings for Caliche Security Module

use crate::conversions::{json_to_python_dict, python_to_json_value};
use crate::get_runtime;
use caliche::security::audit::AuditOutcome;
use caliche::security::{
    ApiKeyValidator, AuditEvent, AuditLogger, AuthContext, JwtConfig, JwtValidator, Permission,
    RateLimitConfig, RateLimiter, RbacEnforcer, Role,
};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;
use std::time::Duration;

/// Python wrapper for `AuthContext`
#[pyclass]
pub struct PyAuthContext {
    inner: AuthContext,
}

#[pymethods]
impl PyAuthContext {
    #[new]
    pub const fn new(
        user_id: String,
        tenant_id: Option<String>,
        roles: Vec<String>,
        permissions: Vec<String>,
    ) -> Self {
        Self {
            inner: AuthContext {
                user_id,
                tenant_id,
                roles,
                permissions,
            },
        }
    }

    pub fn user_id(&self) -> String {
        self.inner.user_id.clone()
    }

    pub fn tenant_id(&self) -> Option<String> {
        self.inner.tenant_id.clone()
    }

    pub fn roles(&self) -> Vec<String> {
        self.inner.roles.clone()
    }

    pub fn permissions(&self) -> Vec<String> {
        self.inner.permissions.clone()
    }
}

/// Python wrapper for `ApiKeyValidator`
#[pyclass]
pub struct PyApiKeyValidator {
    inner: ApiKeyValidator,
}

#[pymethods]
impl PyApiKeyValidator {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: ApiKeyValidator::new(),
        }
    }

    pub fn add_key(&self, api_key: String, context: &PyAuthContext) {
        self.inner.add_key(api_key, context.inner.clone());
    }

    pub fn validate(&self, api_key: &str) -> PyResult<PyAuthContext> {
        match self.inner.validate(api_key) {
            Ok(ctx) => Ok(PyAuthContext { inner: ctx }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Invalid API key: {e}"))),
        }
    }

    pub fn revoke(&self, api_key: &str) {
        self.inner.revoke(api_key);
    }
}

/// Python wrapper for `JwtValidator`
#[pyclass]
pub struct PyJwtValidator {
    inner: JwtValidator,
}

#[pymethods]
impl PyJwtValidator {
    #[new]
    pub fn new(secret: String) -> PyResult<Self> {
        let config = JwtConfig::builder()
            .algorithm("HS256")
            .public_key(secret)
            .audience(["caliche"])
            .issuer(["caliche"])
            .build();
        let inner = JwtValidator::from_config(&config)
            .map_err(|e| PyRuntimeError::new_err(format!("Invalid JWT config: {e}")))?;
        Ok(Self { inner })
    }

    pub fn generate_token(&self, context: &PyAuthContext, ttl_secs: u64) -> PyResult<String> {
        match self.inner.generate_token(&context.inner, ttl_secs) {
            Ok(token) => Ok(token),
            Err(e) => Err(PyRuntimeError::new_err(format!(
                "Failed to generate token: {e}"
            ))),
        }
    }

    pub fn validate_token(&self, token: &str) -> PyResult<PyAuthContext> {
        match self.inner.validate_token(token) {
            Ok(ctx) => Ok(PyAuthContext { inner: ctx }),
            Err(e) => Err(PyRuntimeError::new_err(format!("Invalid token: {e}"))),
        }
    }
}

/// Python wrapper for `RateLimitConfig`
#[pyclass]
pub struct PyRateLimitConfig {
    inner: RateLimitConfig,
}

#[pymethods]
impl PyRateLimitConfig {
    #[new]
    pub const fn new(requests_per_window: u32, window_secs: u64, burst_size: u32) -> Self {
        Self {
            inner: RateLimitConfig {
                requests_per_window,
                window_duration: Duration::from_secs(window_secs),
                burst_size,
            },
        }
    }

    pub const fn requests_per_window(&self) -> u32 {
        self.inner.requests_per_window
    }

    pub const fn window_secs(&self) -> u64 {
        self.inner.window_duration.as_secs()
    }

    pub const fn burst_size(&self) -> u32 {
        self.inner.burst_size
    }
}

/// Python wrapper for `RateLimiter`
#[pyclass]
pub struct PyRateLimiter {
    inner: RateLimiter,
}

#[pymethods]
impl PyRateLimiter {
    #[new]
    pub fn new(config: &PyRateLimitConfig) -> Self {
        Self {
            inner: RateLimiter::new(config.inner.clone()),
        }
    }

    pub fn check_rate_limit(&self, key: &str) -> bool {
        self.inner.check_rate_limit(key)
    }

    pub fn reset(&self, key: &str) {
        self.inner.reset(key);
    }
}

/// Python wrapper for Permission
#[pyclass]
pub struct PyPermission {
    inner: Permission,
}

#[pymethods]
impl PyPermission {
    #[new]
    pub const fn new(resource: String, action: String) -> Self {
        Self {
            inner: Permission { resource, action },
        }
    }

    pub fn resource(&self) -> String {
        self.inner.resource.clone()
    }

    pub fn action(&self) -> String {
        self.inner.action.clone()
    }
}

/// Python wrapper for Role
#[pyclass]
pub struct PyRole {
    inner: Role,
}

#[pymethods]
impl PyRole {
    #[new]
    pub fn new(name: String, permissions: &Bound<'_, PyList>) -> PyResult<Self> {
        let mut perms = Vec::new();
        for item in permissions.iter() {
            let perm = item.extract::<PyRef<'_, PyPermission>>()?;
            perms.push(perm.inner.clone());
        }
        Ok(Self {
            inner: Role {
                name,
                permissions: perms,
            },
        })
    }

    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    pub fn permissions(&self) -> Vec<PyPermission> {
        self.inner
            .permissions
            .iter()
            .map(|p| PyPermission { inner: p.clone() })
            .collect()
    }
}

/// Python wrapper for `RbacEnforcer`
#[pyclass]
pub struct PyRbacEnforcer {
    inner: RbacEnforcer,
}

#[pymethods]
impl PyRbacEnforcer {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: RbacEnforcer::new(),
        }
    }

    pub fn add_role(&self, role: &PyRole) {
        self.inner.add_role(role.inner.clone());
    }

    pub fn check_permission(
        &self,
        user_roles: &Bound<'_, PyList>,
        resource: &str,
        action: &str,
    ) -> PyResult<bool> {
        let roles = user_roles.extract::<Vec<String>>()?;
        Ok(self.inner.check_permission(&roles, resource, action))
    }
}

/// Python wrapper for `AuditEvent`
#[pyclass]
pub struct PyAuditEvent {
    inner: AuditEvent,
}

#[pymethods]
impl PyAuditEvent {
    #[new]
    pub fn new(
        user_id: String,
        action: String,
        resource: String,
        outcome: &str,
        tenant_id: Option<String>,
        metadata: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let outcome_enum = match outcome.to_lowercase().as_str() {
            "success" => AuditOutcome::Success,
            "failure" => AuditOutcome::Failure,
            "denied" => AuditOutcome::Denied,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "Invalid outcome: must be success, failure, or denied",
                ));
            }
        };

        let meta_value = metadata.map_or_else(
            || Ok(Value::Object(serde_json::Map::new())),
            |dict| python_to_json_value(dict.as_any()),
        )?;

        Ok(Self {
            inner: AuditEvent {
                timestamp: chrono::Utc::now(),
                user_id,
                tenant_id,
                action,
                resource,
                outcome: outcome_enum,
                metadata: meta_value,
            },
        })
    }

    pub fn to_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("timestamp", self.inner.timestamp.to_rfc3339())?;
        dict.set_item("user_id", &self.inner.user_id)?;
        if let Some(ref tenant_id) = self.inner.tenant_id {
            dict.set_item("tenant_id", tenant_id)?;
        }
        dict.set_item("action", &self.inner.action)?;
        dict.set_item("resource", &self.inner.resource)?;
        let outcome_str = match self.inner.outcome {
            AuditOutcome::Success => "success",
            AuditOutcome::Failure => "failure",
            AuditOutcome::Denied => "denied",
        };
        dict.set_item("outcome", outcome_str)?;
        let meta_dict = json_to_python_dict(py, &self.inner.metadata)?;
        dict.set_item("metadata", meta_dict)?;
        Ok(dict.into())
    }
}

/// Python wrapper for `AuditLogger`
#[pyclass]
pub struct PyAuditLogger {
    inner: AuditLogger,
}

#[pymethods]
impl PyAuditLogger {
    #[new]
    pub fn new(max_events: usize) -> Self {
        Self {
            inner: AuditLogger::new(max_events),
        }
    }

    pub fn log(&self, event: &PyAuditEvent) -> PyResult<()> {
        let logger_clone = &self.inner;
        let event_clone = event.inner.clone();
        let rt = get_runtime()?;

        rt.block_on(async { logger_clone.log(event_clone).await });
        Ok(())
    }

    pub fn get_events(&self, limit: usize) -> PyResult<Vec<PyAuditEvent>> {
        let logger_clone = &self.inner;
        let rt = get_runtime()?;

        let events = rt.block_on(async { logger_clone.get_events(limit).await });
        Ok(events
            .into_iter()
            .map(|e| PyAuditEvent { inner: e })
            .collect())
    }
}
