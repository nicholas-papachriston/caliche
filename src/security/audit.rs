//! Security audit logging.
//!
//! This module re-exports audit functionality from [`crate::observability::audit_log`].
//! Use that module directly for full audit capabilities including PostgreSQL, webhook,
//! and file backends.
//!
//! This module provides a simple in-memory audit logger for basic use cases.
//! For production deployments, use the backends from `observability::audit_log`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub timestamp: DateTime<Utc>,
    pub user_id: String,
    pub tenant_id: Option<String>,
    pub action: String,
    pub resource: String,
    pub outcome: AuditOutcome,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    Failure,
    Denied,
}

/// Audit logging backend type
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AuditBackend {
    /// In-memory backend (default)
    #[default]
    Memory,
    /// PostgreSQL backend
    Postgres,
    /// Webhook backend
    Webhook,
    /// File backend
    File,
}

impl std::str::FromStr for AuditBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "memory" => Ok(Self::Memory),
            "postgres" | "postgresql" => Ok(Self::Postgres),
            "webhook" => Ok(Self::Webhook),
            "file" => Ok(Self::File),
            _ => Err(format!("Unknown audit backend: {}", s)),
        }
    }
}

/// Audit logging configuration
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Backend type for audit logs
    pub backend: AuditBackend,
    /// Retention period in days
    pub retention_days: Option<u32>,
    /// Webhook URL (for webhook backend)
    pub webhook_url: Option<String>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: AuditBackend::Memory,
            retention_days: None,
            webhook_url: None,
        }
    }
}

impl AuditConfig {
    /// Create a builder for audit configuration
    pub fn builder() -> AuditConfigBuilder {
        AuditConfigBuilder::default()
    }

    /// Create configuration from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `AUDIT_LOGGING_ENABLED`: Enable audit logging (default: false)
    /// - `AUDIT_LOG_BACKEND`: Backend type (memory, postgres, webhook, file) (default: "memory")
    /// - `AUDIT_RETENTION_DAYS`: Retention period in days (optional)
    /// - `AUDIT_WEBHOOK_URL`: Webhook URL for webhook backend (optional)
    pub fn from_env() -> Self {
        let enabled = std::env::var("AUDIT_LOGGING_ENABLED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(false);

        if !enabled {
            return Self {
                enabled: false,
                ..Default::default()
            };
        }

        let backend_str =
            std::env::var("AUDIT_LOG_BACKEND").unwrap_or_else(|_| "memory".to_owned());

        let backend = backend_str.parse().unwrap_or_default();

        let retention_days = std::env::var("AUDIT_RETENTION_DAYS")
            .ok()
            .and_then(|s| s.parse().ok());

        let webhook_url = std::env::var("AUDIT_WEBHOOK_URL").ok();

        Self {
            enabled: true,
            backend,
            retention_days,
            webhook_url,
        }
    }
}

/// Builder for audit configuration
#[derive(Debug, Default)]
pub struct AuditConfigBuilder {
    enabled: Option<bool>,
    backend: Option<AuditBackend>,
    retention_days: Option<u32>,
    webhook_url: Option<String>,
}

impl AuditConfigBuilder {
    /// Enable or disable audit logging
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = Some(enabled);
        self
    }

    /// Set the audit backend
    pub fn backend(mut self, backend: AuditBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set retention period in days
    pub fn retention_days(mut self, days: u32) -> Self {
        self.retention_days = Some(days);
        self
    }

    /// Set webhook URL
    pub fn webhook_url(mut self, url: impl Into<String>) -> Self {
        self.webhook_url = Some(url.into());
        self
    }

    /// Build the configuration
    pub fn build(self) -> AuditConfig {
        let defaults = AuditConfig::default();
        AuditConfig {
            enabled: self.enabled.unwrap_or(defaults.enabled),
            backend: self.backend.unwrap_or(defaults.backend),
            retention_days: self.retention_days.or(defaults.retention_days),
            webhook_url: self.webhook_url.or(defaults.webhook_url),
        }
    }
}

pub struct AuditLogger {
    events: Arc<RwLock<Vec<AuditEvent>>>,
    max_events: usize,
}

impl AuditLogger {
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            max_events,
        }
    }

    pub async fn log(&self, event: AuditEvent) {
        let mut events = self.events.write().await;
        events.push(event.clone());

        if events.len() > self.max_events {
            events.remove(0);
        }

        tracing::info!(
            user_id = %event.user_id,
            action = %event.action,
            resource = %event.resource,
            "Audit event logged"
        );
    }

    pub async fn get_events(&self, limit: usize) -> Vec<AuditEvent> {
        let events = self.events.read().await;
        events.iter().rev().take(limit).cloned().collect()
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new(10000)
    }
}
