use super::{
    ConversationListQuery, ConversationRecord, ConversationStats, ConversationStoreBackend,
    MetadataQuery,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::{
    Row,
    sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous},
};
use std::str::FromStr;
use std::time::Duration;
use tracing::instrument;

pub struct SqliteConversationStore {
    pool: SqlitePool,
}

impl SqliteConversationStore {
    /// Create a new SQLite-backed conversation store and initialize schema.
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

    /// Create a new SQLite store from a file path.
    pub async fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }

        let database_url = format!("sqlite:{}", path.display());
        Self::new(&database_url).await
    }

    /// Construct a store from an existing connection pool.
    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    pub async fn save(&self, conversation: &ConversationRecord) -> Result<()> {
        <Self as ConversationStoreBackend>::save(self, conversation).await
    }

    pub async fn get(&self, id: &str) -> Result<Option<ConversationRecord>> {
        <Self as ConversationStoreBackend>::fetch(self, id).await
    }

    pub async fn list_by_agent(
        &self,
        agent_id: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ConversationRecord>> {
        let query = ConversationListQuery::new(agent_id, limit, offset);
        <Self as ConversationStoreBackend>::list(self, &query).await
    }

    pub async fn delete(&self, id: &str) -> Result<()> {
        <Self as ConversationStoreBackend>::delete(self, id).await
    }

    pub async fn cleanup_expired(&self) -> Result<u64> {
        <Self as ConversationStoreBackend>::cleanup_expired(self).await
    }

    pub async fn update_ttl(&self, id: &str, ttl_seconds: i64) -> Result<()> {
        <Self as ConversationStoreBackend>::update_ttl(self, id, ttl_seconds).await
    }

    pub async fn search_by_metadata(
        &self,
        filter: &Value,
        limit: i64,
    ) -> Result<Vec<ConversationRecord>> {
        let query = MetadataQuery::new(filter.clone(), limit);
        <Self as ConversationStoreBackend>::search(self, &query).await
    }

    pub async fn get_stats(&self) -> Result<ConversationStats> {
        <Self as ConversationStoreBackend>::stats(self).await
    }
}

#[async_trait]
impl ConversationStoreBackend for SqliteConversationStore {
    async fn init(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                messages TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create conversations table")?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_conversations_agent_id ON conversations(agent_id)",
        )
        .execute(&self.pool)
        .await
        .context("Failed to create agent_id index")?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC)",
        )
        .execute(&self.pool)
        .await
        .context("Failed to create created_at index")?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_conversations_expires_at ON conversations(expires_at)",
        )
        .execute(&self.pool)
        .await
        .context("Failed to create expires_at index")?;

        Ok(())
    }

    #[instrument(skip(self, conversation), fields(conversation_id = %conversation.id, agent_id = %conversation.agent_id), err)]
    async fn save(&self, conversation: &ConversationRecord) -> Result<()> {
        let messages_json = serde_json::to_string(&conversation.messages)
            .context("Failed to serialize messages")?;
        let metadata_json = serde_json::to_string(&conversation.metadata)
            .context("Failed to serialize metadata")?;

        sqlx::query(
            r#"
            INSERT INTO conversations (id, agent_id, messages, metadata, created_at, updated_at, expires_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(id) DO UPDATE SET
                messages = excluded.messages,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at,
                expires_at = excluded.expires_at
            "#,
        )
        .bind(&conversation.id)
        .bind(&conversation.agent_id)
        .bind(&messages_json)
        .bind(&metadata_json)
        .bind(conversation.created_at.to_rfc3339())
        .bind(conversation.updated_at.to_rfc3339())
        .bind(conversation.expires_at.map(|dt| dt.to_rfc3339()))
        .execute(&self.pool)
        .await
        .context("Failed to save conversation")?;

        Ok(())
    }

    #[instrument(skip(self), fields(conversation_id = %id), err)]
    async fn fetch(&self, id: &str) -> Result<Option<ConversationRecord>> {
        let row = sqlx::query(
            r#"
            SELECT id, agent_id, messages, metadata, created_at, updated_at, expires_at
            FROM conversations
            WHERE id = ?1
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch conversation")?;

        if let Some(row) = row {
            let messages_json: String = row.try_get("messages")?;
            let metadata_json: String = row.try_get("metadata")?;
            let created_at_str: String = row.try_get("created_at")?;
            let updated_at_str: String = row.try_get("updated_at")?;
            let expires_at_str: Option<String> = row.try_get("expires_at")?;

            let messages: Vec<crate::messages::Message> =
                serde_json::from_str(&messages_json).context("Failed to deserialize messages")?;
            let metadata: Value =
                serde_json::from_str(&metadata_json).context("Failed to deserialize metadata")?;

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .context("Failed to parse created_at")?
                .with_timezone(&Utc);
            let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                .context("Failed to parse updated_at")?
                .with_timezone(&Utc);
            let expires_at = expires_at_str
                .map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .context("Failed to parse expires_at")
                        .map(|dt| dt.with_timezone(&Utc))
                })
                .transpose()?;

            Ok(Some(ConversationRecord {
                id: row.try_get("id")?,
                agent_id: row.try_get("agent_id")?,
                messages,
                metadata,
                created_at,
                updated_at,
                expires_at,
            }))
        } else {
            Ok(None)
        }
    }

    #[instrument(skip(self, query), fields(agent_id = %query.agent_id, limit = query.limit, offset = query.offset), err)]
    async fn list(&self, query: &ConversationListQuery) -> Result<Vec<ConversationRecord>> {
        let rows = sqlx::query(
            r#"
            SELECT id, agent_id, messages, metadata, created_at, updated_at, expires_at
            FROM conversations
            WHERE agent_id = ?1
              AND (expires_at IS NULL OR expires_at > datetime('now'))
            ORDER BY created_at DESC
            LIMIT ?2 OFFSET ?3
            "#,
        )
        .bind(&query.agent_id)
        .bind(query.limit)
        .bind(query.offset)
        .fetch_all(&self.pool)
        .await
        .context("Failed to list conversations")?;

        let mut conversations = Vec::with_capacity(rows.len());
        for row in rows {
            let messages_json: String = row.try_get("messages")?;
            let metadata_json: String = row.try_get("metadata")?;
            let created_at_str: String = row.try_get("created_at")?;
            let updated_at_str: String = row.try_get("updated_at")?;
            let expires_at_str: Option<String> = row.try_get("expires_at")?;

            let messages: Vec<crate::messages::Message> =
                serde_json::from_str(&messages_json).context("Failed to deserialize messages")?;
            let metadata: Value =
                serde_json::from_str(&metadata_json).context("Failed to deserialize metadata")?;

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .context("Failed to parse created_at")?
                .with_timezone(&Utc);
            let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                .context("Failed to parse updated_at")?
                .with_timezone(&Utc);
            let expires_at = expires_at_str
                .map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .context("Failed to parse expires_at")
                        .map(|dt| dt.with_timezone(&Utc))
                })
                .transpose()?;

            conversations.push(ConversationRecord {
                id: row.try_get("id")?,
                agent_id: row.try_get("agent_id")?,
                messages,
                metadata,
                created_at,
                updated_at,
                expires_at,
            });
        }

        Ok(conversations)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM conversations WHERE id = ?1")
            .bind(id)
            .execute(&self.pool)
            .await
            .context("Failed to delete conversation")?;

        Ok(())
    }

    async fn cleanup_expired(&self) -> Result<u64> {
        let result = sqlx::query(
            "DELETE FROM conversations WHERE expires_at IS NOT NULL AND expires_at < datetime('now')",
        )
        .execute(&self.pool)
        .await
        .context("Failed to cleanup expired conversations")?;

        Ok(result.rows_affected())
    }

    async fn update_ttl(&self, id: &str, ttl_seconds: i64) -> Result<()> {
        let expires_at = Utc::now() + chrono::Duration::seconds(ttl_seconds);

        sqlx::query(
            r#"
            UPDATE conversations
            SET expires_at = ?2,
                updated_at = ?3
            WHERE id = ?1
            "#,
        )
        .bind(id)
        .bind(expires_at.to_rfc3339())
        .bind(Utc::now().to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to update TTL")?;

        Ok(())
    }

    #[instrument(skip(self, query), fields(limit = query.limit, json_path = query.json_path.as_deref().unwrap_or("")), err)]
    async fn search(&self, query: &MetadataQuery) -> Result<Vec<ConversationRecord>> {
        let metadata_condition = if query
            .json_path
            .as_deref()
            .is_some_and(|path| !path.is_empty())
        {
            "json_extract(metadata, ?1) = json(?2)".to_string()
        } else {
            "json_extract(metadata, '$') = json(?1)".to_string()
        };
        let expiry = "(expires_at IS NULL OR expires_at > datetime('now'))";
        let sql = format!(
            r#"
            SELECT id, agent_id, messages, metadata, created_at, updated_at, expires_at
            FROM conversations
            WHERE {metadata_condition}
              AND {expiry}
            ORDER BY created_at DESC
            LIMIT {{limit_placeholder}}
            "#,
            metadata_condition = metadata_condition,
            expiry = expiry,
        );

        let filter_json =
            serde_json::to_string(&query.filter).context("Failed to serialize filter")?;

        let rows = if let Some(path) = query
            .json_path
            .as_deref()
            .filter(|segment| !segment.is_empty())
            .map(|json_path| {
                format!(
                    "$.{}",
                    json_path
                        .split('.')
                        .filter(|segment| !segment.is_empty())
                        .collect::<Vec<_>>()
                        .join(".")
                )
            }) {
            let sql = sql.replace("{limit_placeholder}", "?3");
            sqlx::query(&sql)
                .bind(path)
                .bind(filter_json)
                .bind(query.limit)
                .fetch_all(&self.pool)
                .await
        } else {
            let sql = sql.replace("{limit_placeholder}", "?2");
            sqlx::query(&sql)
                .bind(filter_json)
                .bind(query.limit)
                .fetch_all(&self.pool)
                .await
        }
        .context("Failed to search conversations")?;

        let mut conversations = Vec::with_capacity(rows.len());
        for row in rows {
            let messages_json: String = row.try_get("messages")?;
            let metadata_json: String = row.try_get("metadata")?;
            let created_at_str: String = row.try_get("created_at")?;
            let updated_at_str: String = row.try_get("updated_at")?;
            let expires_at_str: Option<String> = row.try_get("expires_at")?;

            let messages: Vec<crate::messages::Message> =
                serde_json::from_str(&messages_json).context("Failed to deserialize messages")?;
            let metadata: Value =
                serde_json::from_str(&metadata_json).context("Failed to deserialize metadata")?;

            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .context("Failed to parse created_at")?
                .with_timezone(&Utc);
            let updated_at = DateTime::parse_from_rfc3339(&updated_at_str)
                .context("Failed to parse updated_at")?
                .with_timezone(&Utc);
            let expires_at = expires_at_str
                .map(|s| {
                    DateTime::parse_from_rfc3339(&s)
                        .context("Failed to parse expires_at")
                        .map(|dt| dt.with_timezone(&Utc))
                })
                .transpose()?;

            conversations.push(ConversationRecord {
                id: row.try_get("id")?,
                agent_id: row.try_get("agent_id")?,
                messages,
                metadata,
                created_at,
                updated_at,
                expires_at,
            });
        }

        Ok(conversations)
    }

    #[instrument(skip(self), err)]
    async fn stats(&self) -> Result<ConversationStats> {
        let sql = r#"
            SELECT 
                COUNT(*) as total_conversations,
                SUM(CASE WHEN expires_at IS NULL OR expires_at > datetime('now') THEN 1 ELSE 0 END) as active_conversations,
                SUM(CASE WHEN expires_at IS NOT NULL AND expires_at <= datetime('now') THEN 1 ELSE 0 END) as expired_conversations
            FROM conversations
            "#;
        let row = sqlx::query(sql)
            .fetch_one(&self.pool)
            .await
            .context("Failed to get stats")?;

        let total: i64 = row.try_get("total_conversations")?;
        let active: i64 = row.try_get("active_conversations")?;
        let expired: i64 = row.try_get("expired_conversations")?;

        let avg_message_count = if total > 0 {
            let row = sqlx::query(
                r#"
                SELECT AVG(json_array_length(messages)) as avg_message_count
                FROM conversations
                "#,
            )
            .fetch_one(&self.pool)
            .await
            .context("Failed to get average message count")?;

            row.try_get::<Option<f64>, _>("avg_message_count")?
                .unwrap_or(0.0)
        } else {
            0.0
        };

        Ok(ConversationStats {
            total_conversations: total,
            active_conversations: active,
            expired_conversations: expired,
            avg_message_count,
        })
    }
}
