//! PostgreSQL backend for durable event streams.
//!
//! Stores events in a single table with one row per event. Uses sqlx with the
//! Postgres driver. Good for multi-node or existing Postgres infrastructure.

use crate::streams::backend_trait::DurableStreamBackend;
use crate::streams::types::{DurableEvent, STREAM_START, TrimPolicy};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::time::Duration;

/// PostgreSQL-backed durable stream backend.
///
/// Uses a single table: (id BIGSERIAL PRIMARY KEY, stream_key TEXT,
/// inserted_at BIGINT, payload JSONB). Event IDs are the row id as string.
#[derive(Clone, Debug)]
pub struct PostgresDurableStreamBackend {
    pool: sqlx::PgPool,
}

impl PostgresDurableStreamBackend {
    /// Create a backend from a PostgreSQL connection string.
    /// Creates the table if it does not exist.
    pub async fn new(connection_string: &str) -> Result<Self> {
        let pool = sqlx::postgres::PgPoolOptions::new()
            .connect(connection_string)
            .await
            .context("Failed to connect Postgres for durable streams")?;
        let backend = Self { pool };
        backend.init().await?;
        Ok(backend)
    }

    async fn init(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS durable_stream_events (
                id BIGSERIAL PRIMARY KEY,
                stream_key TEXT NOT NULL,
                inserted_at BIGINT NOT NULL,
                payload JSONB NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Create durable_stream_events table")?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_durable_stream_events_stream_key_id ON durable_stream_events (stream_key, id)",
        )
        .execute(&self.pool)
        .await
        .context("Create index on durable_stream_events")?;
        Ok(())
    }
}

#[async_trait]
impl DurableStreamBackend for PostgresDurableStreamBackend {
    async fn append(&self, stream_key: &str, event: &DurableEvent) -> Result<String> {
        let json = serde_json::to_value(event).context("Serialize DurableEvent")?;
        let inserted_at = event.timestamp.timestamp_millis();
        let row: (i64,) = sqlx::query_as(
            "INSERT INTO durable_stream_events (stream_key, inserted_at, payload) VALUES ($1, $2, $3) RETURNING id",
        )
        .bind(stream_key)
        .bind(inserted_at)
        .bind(&json)
        .fetch_one(&self.pool)
        .await
        .context("INSERT durable stream event")?;
        Ok(row.0.to_string())
    }

    async fn read_after(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let start_id: i64 = if after_id == STREAM_START {
            0
        } else {
            after_id.parse().context("Parse after_id for Postgres")?
        };
        let rows: Vec<(i64, serde_json::Value)> = sqlx::query_as(
            "SELECT id, payload FROM durable_stream_events WHERE stream_key = $1 AND id > $2 ORDER BY id ASC LIMIT $3",
        )
        .bind(stream_key)
        .bind(start_id)
        .bind(count as i64)
        .fetch_all(&self.pool)
        .await
        .context("SELECT read_after")?;
        let mut out = Vec::with_capacity(rows.len());
        for (id, payload) in rows {
            let ev: DurableEvent =
                serde_json::from_value(payload).context("Deserialize DurableEvent")?;
            out.push((id.to_string(), ev));
        }
        Ok(out)
    }

    async fn read_blocking(
        &self,
        stream_key: &str,
        after_id: &str,
        count: usize,
        timeout: Option<Duration>,
    ) -> Result<Vec<(String, DurableEvent)>> {
        let deadline = timeout.map(|d| std::time::Instant::now() + d);
        let poll_interval = Duration::from_millis(100);
        loop {
            let out = self.read_after(stream_key, after_id, count).await?;
            if !out.is_empty() {
                return Ok(out);
            }
            if let Some(d) = deadline {
                if std::time::Instant::now() >= d {
                    return Ok(Vec::new());
                }
            }
            tokio::time::sleep(poll_interval).await;
        }
    }

    async fn trim(&self, stream_key: &str, policy: TrimPolicy) -> Result<u64> {
        let removed = match policy {
            TrimPolicy::MaxLen(n) => {
                let n = n as i64;
                // $1 = stream_key (reused in subquery), $2 = LIMIT count.
                // sqlx binds positionally: each .bind() maps to the next $N.
                // $1 is reused in the SQL but only needs one .bind() call.
                let r = sqlx::query(
                    r#"
                    DELETE FROM durable_stream_events
                    WHERE stream_key = $1 AND id < (
                        SELECT COALESCE(MIN(id), 0) FROM (
                            SELECT id FROM durable_stream_events WHERE stream_key = $1 ORDER BY id DESC LIMIT $2
                        ) sub
                    )
                    "#,
                )
                .bind(stream_key)
                .bind(n)
                .execute(&self.pool)
                .await
                .context("TRIM MaxLen")?;
                r.rows_affected()
            }
            TrimPolicy::MinId(min_id) => {
                let min_id: i64 = min_id.parse().context("Parse MinId for Postgres")?;
                let r = sqlx::query(
                    "DELETE FROM durable_stream_events WHERE stream_key = $1 AND id < $2",
                )
                .bind(stream_key)
                .bind(min_id)
                .execute(&self.pool)
                .await
                .context("TRIM MinId")?;
                r.rows_affected()
            }
            TrimPolicy::MaxAgeSecs(secs) => {
                let cutoff = chrono::Utc::now().timestamp_millis() - (secs as i64 * 1000);
                let r = sqlx::query(
                    "DELETE FROM durable_stream_events WHERE stream_key = $1 AND inserted_at < $2",
                )
                .bind(stream_key)
                .bind(cutoff)
                .execute(&self.pool)
                .await
                .context("TRIM MaxAgeSecs")?;
                r.rows_affected()
            }
        };
        Ok(removed)
    }

    async fn last_id(&self, stream_key: &str) -> Result<Option<String>> {
        let row: Option<(i64,)> = sqlx::query_as(
            "SELECT id FROM durable_stream_events WHERE stream_key = $1 ORDER BY id DESC LIMIT 1",
        )
        .bind(stream_key)
        .fetch_optional(&self.pool)
        .await
        .context("SELECT last_id")?;
        Ok(row.map(|(id,)| id.to_string()))
    }

    async fn delete_stream(&self, stream_key: &str) -> Result<()> {
        sqlx::query("DELETE FROM durable_stream_events WHERE stream_key = $1")
            .bind(stream_key)
            .execute(&self.pool)
            .await
            .context("DELETE stream")?;
        Ok(())
    }
}
