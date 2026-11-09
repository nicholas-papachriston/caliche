use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{
    Row,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous},
};
use std::str::FromStr;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CredentialMetadata {
    pub provider: String,
    pub source: String,
    pub is_default: bool,
    pub default_model: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[async_trait]
pub trait CredentialStoreBackend: Send + Sync {
    async fn init(&self) -> Result<()>;
    async fn set_credential(&self, metadata: &CredentialMetadata) -> Result<()>;
    async fn get_credential(&self, provider: &str) -> Result<Option<CredentialMetadata>>;
    async fn delete_credential(&self, provider: &str) -> Result<bool>;
    async fn list_credentials(&self) -> Result<Vec<CredentialMetadata>>;
    async fn set_default_provider(&self, provider: &str) -> Result<()>;
    async fn get_default_provider(&self) -> Result<Option<String>>;
}

pub struct SqliteCredentialStore {
    pool: SqlitePool,
}

impl SqliteCredentialStore {
    pub async fn new(database_url: &str) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(database_url)?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(Duration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal);

        let pool = crate::data::persistence::sqlite_pool::local_sqlite_pool()
            .connect_with(options)
            .await
            .context("Failed to connect to SQLite database")?;

        let store = Self { pool };
        store.init().await?;
        Ok(store)
    }

    pub async fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }

        let database_url = format!("sqlite:{}", path.display());
        let store = Self::new(&database_url).await?;

        // Set restrictive permissions on the database file (Unix-like systems)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Err(e) = std::fs::metadata(path).and_then(|m| {
                let mut perms = m.permissions();
                perms.set_mode(0o600); // rw------- (owner read/write only)
                std::fs::set_permissions(path, perms)
            }) {
                tracing::warn!(
                    "Failed to set restrictive permissions on credential database: {}",
                    e
                );
            }
        }

        Ok(store)
    }

    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

#[async_trait]
impl CredentialStoreBackend for SqliteCredentialStore {
    async fn init(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS credentials (
                provider TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                is_default INTEGER DEFAULT 0,
                default_model TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create credentials table")?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_credentials_is_default ON credentials(is_default)",
        )
        .execute(&self.pool)
        .await
        .context("Failed to create is_default index")?;

        Ok(())
    }

    async fn set_credential(&self, metadata: &CredentialMetadata) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO credentials (provider, source, is_default, default_model, created_at, updated_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            ON CONFLICT(provider) DO UPDATE SET
                source = excluded.source,
                is_default = excluded.is_default,
                default_model = excluded.default_model,
                updated_at = excluded.updated_at,
                created_at = COALESCE((SELECT created_at FROM credentials WHERE provider = excluded.provider), excluded.created_at)
            "#,
        )
        .bind(&metadata.provider)
        .bind(&metadata.source)
        .bind(if metadata.is_default { 1i64 } else { 0i64 })
        .bind(&metadata.default_model)
        .bind(metadata.created_at.to_rfc3339())
        .bind(metadata.updated_at.to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to set credential")?;

        Ok(())
    }

    async fn get_credential(&self, provider: &str) -> Result<Option<CredentialMetadata>> {
        let row = sqlx::query(
            "SELECT provider, source, is_default, default_model, created_at, updated_at FROM credentials WHERE provider = ?1",
        )
        .bind(provider)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to get credential")?;

        if let Some(row) = row {
            let created_at_str: String = row.try_get("created_at")?;
            let updated_at_str: String = row.try_get("updated_at")?;

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .context("Failed to parse created_at")?
                .with_timezone(&Utc);
            let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                .context("Failed to parse updated_at")?
                .with_timezone(&Utc);

            Ok(Some(CredentialMetadata {
                provider: row.try_get("provider")?,
                source: row.try_get("source")?,
                is_default: row.try_get::<i64, _>("is_default")? != 0,
                default_model: row.try_get("default_model")?,
                created_at,
                updated_at,
            }))
        } else {
            Ok(None)
        }
    }

    async fn delete_credential(&self, provider: &str) -> Result<bool> {
        let result = sqlx::query("DELETE FROM credentials WHERE provider = ?1")
            .bind(provider)
            .execute(&self.pool)
            .await
            .context("Failed to delete credential")?;

        Ok(result.rows_affected() > 0)
    }

    async fn list_credentials(&self) -> Result<Vec<CredentialMetadata>> {
        let rows = sqlx::query(
            "SELECT provider, source, is_default, default_model, created_at, updated_at FROM credentials ORDER BY provider",
        )
        .fetch_all(&self.pool)
        .await
        .context("Failed to list credentials")?;

        let mut credentials = Vec::with_capacity(rows.len());
        for row in rows {
            let created_at_str: String = row.try_get("created_at")?;
            let updated_at_str: String = row.try_get("updated_at")?;

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .context("Failed to parse created_at")?
                .with_timezone(&Utc);
            let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                .context("Failed to parse updated_at")?
                .with_timezone(&Utc);

            credentials.push(CredentialMetadata {
                provider: row.try_get("provider")?,
                source: row.try_get("source")?,
                is_default: row.try_get::<i64, _>("is_default")? != 0,
                default_model: row.try_get("default_model")?,
                created_at,
                updated_at,
            });
        }

        Ok(credentials)
    }

    async fn set_default_provider(&self, provider: &str) -> Result<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .context("Failed to begin default-provider transaction")?;

        sqlx::query("UPDATE credentials SET is_default = 0")
            .execute(&mut *tx)
            .await
            .context("Failed to clear existing default")?;

        let result = sqlx::query("UPDATE credentials SET is_default = 1 WHERE provider = ?1")
            .bind(provider)
            .execute(&mut *tx)
            .await
            .context("Failed to set default provider")?;

        if result.rows_affected() == 0 {
            return Err(anyhow::anyhow!(
                "Cannot set default: provider '{provider}' has no stored credentials"
            ));
        }

        tx.commit()
            .await
            .context("Failed to commit default-provider transaction")?;

        Ok(())
    }

    async fn get_default_provider(&self) -> Result<Option<String>> {
        let provider: Option<String> =
            sqlx::query_scalar("SELECT provider FROM credentials WHERE is_default = 1 LIMIT 1")
                .fetch_optional(&self.pool)
                .await
                .context("Failed to get default provider")?;

        Ok(provider)
    }
}
