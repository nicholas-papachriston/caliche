use async_trait::async_trait;

use crate::utils::user_context::UserContext;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Audit log entry for compliance and security tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLog {
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
    pub action: String,
    pub resource: String,
    pub status: String,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl AuditLog {
    pub fn new(
        request_id: impl Into<String>,
        action: impl Into<String>,
        resource: impl Into<String>,
        status: impl Into<String>,
    ) -> Self {
        Self {
            request_id: request_id.into(),
            timestamp: Utc::now(),
            user_id: None,
            tenant_id: None,
            action: action.into(),
            resource: resource.into(),
            status: status.into(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_user_context(mut self, context: Option<&UserContext>) -> Self {
        if let Some(ctx) = context {
            self.user_id = Some(ctx.user_id.clone());
            self.tenant_id = ctx.tenant_id.clone();
        }
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn add_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Audit log backend trait
#[async_trait]
pub trait AuditLogBackend: Send + Sync + std::fmt::Debug {
    async fn log(&self, entry: AuditLog) -> anyhow::Result<()>;
}

/// PostgreSQL audit log backend
#[cfg(feature = "sqlx")]
pub struct PostgresAuditBackend {
    pool: sqlx::PgPool,
    table_name: String,
}

#[cfg(feature = "sqlx")]
impl PostgresAuditBackend {
    pub async fn new(database_url: &str, table_name: Option<String>) -> anyhow::Result<Self> {
        let pool = sqlx::PgPool::connect(database_url).await?;
        let table_name = table_name.unwrap_or_else(|| "audit_logs".to_string());

        // Create table if it doesn't exist
        let create_table_sql = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {} (
                id BIGSERIAL PRIMARY KEY,
                request_id VARCHAR(255) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                user_id VARCHAR(255),
                tenant_id VARCHAR(255),
                action VARCHAR(255) NOT NULL,
                resource TEXT NOT NULL,
                status VARCHAR(50) NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_{}_request_id ON {} (request_id);
            CREATE INDEX IF NOT EXISTS idx_{}_user_id ON {} (user_id);
            CREATE INDEX IF NOT EXISTS idx_{}_tenant_id ON {} (tenant_id);
            CREATE INDEX IF NOT EXISTS idx_{}_timestamp ON {} (timestamp);
            "#,
            table_name, // CREATE TABLE (1)
            table_name, // idx_{}_request_id name (2)
            table_name, // idx_{}_request_id table (3)
            table_name, // idx_{}_user_id name (4)
            table_name, // idx_{}_user_id table (5)
            table_name, // idx_{}_tenant_id name (6)
            table_name, // idx_{}_tenant_id table (7)
            table_name, // idx_{}_timestamp name (8)
            table_name  // idx_{}_timestamp table (9)
        );

        sqlx::query(&create_table_sql)
            .execute(&pool)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create audit log table: {}", e))?;

        Ok(Self { pool, table_name })
    }
}

#[cfg(feature = "sqlx")]
impl std::fmt::Debug for PostgresAuditBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PostgresAuditBackend")
            .field("pool", &"sqlx::PgPool")
            .field("table_name", &self.table_name)
            .finish()
    }
}

#[cfg(feature = "sqlx")]
#[async_trait]
impl AuditLogBackend for PostgresAuditBackend {
    async fn log(&self, entry: AuditLog) -> anyhow::Result<()> {
        sqlx::query(&format!(
            r#"
            INSERT INTO {} (request_id, timestamp, user_id, tenant_id, action, resource, status, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            "#,
            self.table_name
        ))
        .bind(entry.request_id)
        .bind(entry.timestamp)
        .bind(entry.user_id)
        .bind(entry.tenant_id)
        .bind(entry.action)
        .bind(entry.resource)
        .bind(entry.status)
        .bind(serde_json::to_value(entry.metadata)?)
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

/// Webhook audit log backend
pub struct WebhookAuditBackend {
    url: String,
    client: reqwest::Client,
}

impl std::fmt::Debug for WebhookAuditBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebhookAuditBackend")
            .field("url", &self.url)
            .field("client", &"reqwest::Client")
            .finish()
    }
}

impl WebhookAuditBackend {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl AuditLogBackend for WebhookAuditBackend {
    async fn log(&self, entry: AuditLog) -> anyhow::Result<()> {
        self.client.post(&self.url).json(&entry).send().await?;
        Ok(())
    }
}

/// File-based audit log backend (for development/testing)
pub struct FileAuditBackend {
    path: std::path::PathBuf,
}

impl std::fmt::Debug for FileAuditBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileAuditBackend")
            .field("path", &self.path)
            .finish()
    }
}

impl FileAuditBackend {
    pub fn new(path: impl AsRef<std::path::Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

#[async_trait]
impl AuditLogBackend for FileAuditBackend {
    async fn log(&self, entry: AuditLog) -> anyhow::Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        writeln!(file, "{}", serde_json::to_string(&entry)?)?;
        Ok(())
    }
}

/// Audit logger that routes to configured backend
pub struct AuditLogger {
    backend: Option<Arc<dyn AuditLogBackend>>,
    enabled: bool,
}

impl AuditLogger {
    pub fn new(backend: Option<Arc<dyn AuditLogBackend>>) -> Self {
        let enabled = std::env::var("AUDIT_LOGGING_ENABLED")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        Self { backend, enabled }
    }

    #[cfg_attr(
        not(feature = "sqlx"),
        expect(
            clippy::unused_async,
            reason = "same signature as the async feature build"
        )
    )]
    pub async fn from_env() -> anyhow::Result<Self> {
        let enabled = std::env::var("AUDIT_LOGGING_ENABLED")
            .unwrap_or_else(|_| "false".to_string())
            .parse::<bool>()
            .unwrap_or(false);

        if !enabled {
            return Ok(Self {
                backend: None,
                enabled: false,
            });
        }

        let backend_type = std::env::var("AUDIT_LOG_BACKEND")
            .unwrap_or_else(|_| "postgres".to_string())
            .to_lowercase();

        let backend: Arc<dyn AuditLogBackend> = match backend_type.as_str() {
            "postgres" | "postgresql" => {
                #[cfg(feature = "sqlx")]
                {
                    let database_url = std::env::var("POSTGRES_URL")
                        .or_else(|_| std::env::var("AUDIT_POSTGRES_URL"))
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "POSTGRES_URL or AUDIT_POSTGRES_URL required for audit logging"
                            )
                        })?;
                    Arc::new(PostgresAuditBackend::new(&database_url, None).await?)
                }
                #[cfg(not(feature = "sqlx"))]
                {
                    return Err(anyhow::anyhow!(
                        "PostgreSQL audit logging requires enabling the 'sqlx' feature"
                    ));
                }
            }
            "webhook" => {
                let url = std::env::var("AUDIT_WEBHOOK_URL").map_err(|_| {
                    anyhow::anyhow!("AUDIT_WEBHOOK_URL required for webhook audit logging")
                })?;
                Arc::new(WebhookAuditBackend::new(url))
            }
            "file" => {
                let path =
                    std::env::var("AUDIT_LOG_FILE").unwrap_or_else(|_| "audit.log".to_string());
                Arc::new(FileAuditBackend::new(path))
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported audit log backend: {}. Supported: postgres, webhook, file",
                    backend_type
                ));
            }
        };

        Ok(Self {
            backend: Some(backend),
            enabled,
        })
    }

    pub async fn log(&self, entry: AuditLog) {
        if !self.enabled {
            return;
        }

        if let Some(ref backend) = self.backend
            && let Err(e) = backend.log(entry).await
        {
            tracing::warn!("Failed to write audit log: {}", e);
        }
    }
}

use std::sync::Arc;

/// Audit log backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditBackendType {
    Postgres,
    Webhook,
    File,
}

impl AuditBackendType {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "postgres" | "postgresql" => Some(Self::Postgres),
            "webhook" => Some(Self::Webhook),
            "file" => Some(Self::File),
            _ => None,
        }
    }
}

/// Configuration for audit logging
#[derive(Debug, Clone, Default)]
pub struct AuditConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Backend type
    pub backend: Option<AuditBackendType>,
    /// Retention days (for postgres backend)
    pub retention_days: Option<u32>,
    /// Webhook URL (for webhook backend)
    pub webhook_url: Option<String>,
    /// Database URL (for postgres backend)
    pub database_url: Option<String>,
    /// File path (for file backend)
    pub file_path: Option<String>,
}

impl AuditConfig {
    /// Create a builder for AuditConfig
    pub fn builder() -> AuditConfigBuilder {
        AuditConfigBuilder::default()
    }

    /// Create configuration from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `AUDIT_LOGGING_ENABLED`: Enable audit logging (default: false)
    /// - `AUDIT_LOG_BACKEND`: Backend type - "postgres", "webhook", or "file" (default: "postgres")
    /// - `AUDIT_RETENTION_DAYS`: Retention days for postgres backend (default: 90)
    /// - `AUDIT_WEBHOOK_URL`: Webhook URL for webhook backend
    /// - `POSTGRES_URL` or `AUDIT_POSTGRES_URL`: Database URL for postgres backend
    /// - `AUDIT_LOG_FILE`: File path for file backend (default: "audit.log")
    pub fn from_env() -> Self {
        let enabled = std::env::var("AUDIT_LOGGING_ENABLED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(false);

        if !enabled {
            return Self::default();
        }

        let backend_str =
            std::env::var("AUDIT_LOG_BACKEND").unwrap_or_else(|_| "postgres".to_string());
        let backend = AuditBackendType::from_str(&backend_str);

        let retention_days = std::env::var("AUDIT_RETENTION_DAYS")
            .ok()
            .and_then(|s| s.parse().ok());

        let webhook_url = std::env::var("AUDIT_WEBHOOK_URL").ok();

        let database_url = std::env::var("POSTGRES_URL")
            .or_else(|_| std::env::var("AUDIT_POSTGRES_URL"))
            .ok();

        let file_path = std::env::var("AUDIT_LOG_FILE").ok();

        Self {
            enabled,
            backend,
            retention_days,
            webhook_url,
            database_url,
            file_path,
        }
    }
}

/// Builder for AuditConfig
#[derive(Debug, Default)]
pub struct AuditConfigBuilder {
    enabled: Option<bool>,
    backend: Option<AuditBackendType>,
    retention_days: Option<u32>,
    webhook_url: Option<String>,
    database_url: Option<String>,
    file_path: Option<String>,
}

impl AuditConfigBuilder {
    /// Enable or disable audit logging
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = Some(enabled);
        self
    }

    /// Set the backend type
    pub fn backend(mut self, backend: AuditBackendType) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set retention days (for postgres backend)
    pub fn retention_days(mut self, days: u32) -> Self {
        self.retention_days = Some(days);
        self
    }

    /// Set webhook URL (for webhook backend)
    pub fn webhook_url(mut self, url: impl Into<String>) -> Self {
        self.webhook_url = Some(url.into());
        self
    }

    /// Set database URL (for postgres backend)
    pub fn database_url(mut self, url: impl Into<String>) -> Self {
        self.database_url = Some(url.into());
        self
    }

    /// Set file path (for file backend)
    pub fn file_path(mut self, path: impl Into<String>) -> Self {
        self.file_path = Some(path.into());
        self
    }

    /// Build the configuration
    pub fn build(self) -> AuditConfig {
        let defaults = AuditConfig::default();
        AuditConfig {
            enabled: self.enabled.unwrap_or(defaults.enabled),
            backend: self.backend.or(defaults.backend),
            retention_days: self.retention_days.or(defaults.retention_days),
            webhook_url: self.webhook_url.or(defaults.webhook_url),
            database_url: self.database_url.or(defaults.database_url),
            file_path: self.file_path.or(defaults.file_path),
        }
    }
}
