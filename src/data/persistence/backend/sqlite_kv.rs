use super::super::PersistenceBackend;
use anyhow::{Context, Result};
use async_trait::async_trait;
use sqlx::{
    Row,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous},
};
use std::str::FromStr;
use std::time::Duration;

#[derive(Debug)]
pub struct SqlitePersistenceBackend {
    pool: SqlitePool,
}

impl SqlitePersistenceBackend {
    /// Create a new SQLite key-value backend and initialize schema.
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

        let backend = Self { pool };
        backend.init().await?;
        Ok(backend)
    }

    /// Create a new SQLite backend from a file path.
    pub async fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }

        let database_url = format!("sqlite:{}", path.display());
        Self::new(&database_url).await
    }

    /// Construct a backend from an existing connection pool.
    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    async fn init(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                expires_at TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create kv_store table")?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_kv_store_expires_at ON kv_store(expires_at)")
            .execute(&self.pool)
            .await
            .context("Failed to create expires_at index")?;

        Ok(())
    }
}

#[async_trait]
impl PersistenceBackend for SqlitePersistenceBackend {
    async fn set(&self, key: &str, value: &[u8], ttl: Option<u64>) -> Result<()> {
        let expires_at =
            ttl.map(|seconds| chrono::Utc::now() + chrono::Duration::seconds(seconds as i64));

        sqlx::query(
            r#"
            INSERT INTO kv_store (key, value, expires_at)
            VALUES (?1, ?2, ?3)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                expires_at = excluded.expires_at
            "#,
        )
        .bind(key)
        .bind(value)
        .bind(expires_at.map(|dt| dt.to_rfc3339()))
        .execute(&self.pool)
        .await
        .context("Failed to set key")?;

        Ok(())
    }

    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let row = sqlx::query(
            r#"
            SELECT value
            FROM kv_store
            WHERE key = ?1
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            "#,
        )
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to get key")?;

        if let Some(row) = row {
            Ok(Some(row.try_get("value")?))
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, key: &str) -> Result<()> {
        sqlx::query("DELETE FROM kv_store WHERE key = ?1")
            .bind(key)
            .execute(&self.pool)
            .await
            .context("Failed to delete key")?;

        Ok(())
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let count: Option<i64> = sqlx::query_scalar(
            r#"
            SELECT COUNT(*)
            FROM kv_store
            WHERE key = ?1
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            "#,
        )
        .bind(key)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to check if key exists")?;

        Ok(count.unwrap_or(0) > 0)
    }

    async fn expire(&self, key: &str, ttl: u64) -> Result<()> {
        let expires_at = chrono::Utc::now() + chrono::Duration::seconds(ttl as i64);

        sqlx::query(
            r#"
            UPDATE kv_store
            SET expires_at = ?2
            WHERE key = ?1
            "#,
        )
        .bind(key)
        .bind(expires_at.to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to set expiration")?;

        Ok(())
    }
}
