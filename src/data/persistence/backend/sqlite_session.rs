use crate::data::persistence::backend::sql_session::{
    SessionFilter, collect_filters, new_tracked_session, status_to_db,
};
use crate::data::persistence::session_mapper::sqlite_row_to_session;
use crate::data::persistence::session_registry::{
    SessionQuery, SessionRegistry, SessionStatus, TrackedSession, generate_session_id,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Duration, Utc};
use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous};
use std::str::FromStr;
use std::time::Duration as StdDuration;
use uuid::Uuid;

#[derive(Debug)]
pub struct SqliteSessionRegistry {
    pool: SqlitePool,
}

impl SqliteSessionRegistry {
    pub async fn new(database_url: &str) -> Result<Self> {
        let options = SqliteConnectOptions::from_str(database_url)?
            .create_if_missing(true)
            .journal_mode(SqliteJournalMode::Wal)
            .busy_timeout(StdDuration::from_secs(5))
            .synchronous(SqliteSynchronous::Normal);

        let pool = crate::data::persistence::sqlite_pool::local_sqlite_pool()
            .connect_with(options)
            .await
            .context("Failed to connect to SQLite database")?;

        let registry = Self { pool };
        registry.init().await?;
        Ok(registry)
    }

    pub async fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).context("Failed to create database directory")?;
        }

        let database_url = format!("sqlite:{}", path.display());
        Self::new(&database_url).await
    }

    pub fn from_pool(pool: SqlitePool) -> Self {
        Self { pool }
    }

    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }

    // status mapping and row mapping live in shared modules
}

#[async_trait]
impl SessionRegistry for SqliteSessionRegistry {
    async fn init(&self) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS tracked_sessions (
                id TEXT PRIMARY KEY,
                name_slug TEXT,
                agent_id TEXT NOT NULL,
                user_id TEXT,
                conversation_id TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                metadata TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL
            )
            "#,
        )
        .execute(&self.pool)
        .await
        .context("Failed to create tracked_sessions table")?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_tracked_sessions_slug ON tracked_sessions(name_slug)",
        )
        .execute(&self.pool)
        .await
        .context("Failed to create name_slug index")?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_tracked_sessions_agent ON tracked_sessions(agent_id)",
        )
        .execute(&self.pool)
        .await
        .context("Failed to create agent_id index")?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_tracked_sessions_created ON tracked_sessions(created_at DESC)",
        )
        .execute(&self.pool)
        .await
        .context("Failed to create created_at index")?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_tracked_sessions_status ON tracked_sessions(status)",
        )
        .execute(&self.pool)
        .await
        .context("Failed to create status index")?;

        Ok(())
    }

    async fn create(&self, agent_id: &str, user_id: Option<&str>) -> Result<TrackedSession> {
        let id = generate_session_id();
        let session = new_tracked_session(id, agent_id, user_id);

        let metadata_json =
            serde_json::to_string(&session.metadata).context("Failed to serialize metadata")?;

        sqlx::query(
            r#"
            INSERT INTO tracked_sessions
                (id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata, created_at, updated_at, last_accessed)
            VALUES
                (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            "#,
        )
        .bind(session.id.to_string())
        .bind(session.name_slug.as_deref())
        .bind(&session.agent_id)
        .bind(session.user_id.as_deref())
        .bind(session.conversation_id.as_deref())
        .bind(session.message_count as i64)
        .bind(status_to_db(session.status))
        .bind(&metadata_json)
        .bind(session.created_at.to_rfc3339())
        .bind(session.updated_at.to_rfc3339())
        .bind(session.last_accessed.to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to insert tracked session")?;

        Ok(session)
    }

    async fn create_with_id(
        &self,
        id: Uuid,
        agent_id: &str,
        user_id: Option<&str>,
    ) -> Result<TrackedSession> {
        let session = new_tracked_session(id, agent_id, user_id);

        let metadata_json =
            serde_json::to_string(&session.metadata).context("Failed to serialize metadata")?;

        sqlx::query(
            r#"
            INSERT INTO tracked_sessions
                (id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata, created_at, updated_at, last_accessed)
            VALUES
                (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
            "#,
        )
        .bind(session.id.to_string())
        .bind(session.name_slug.as_deref())
        .bind(&session.agent_id)
        .bind(session.user_id.as_deref())
        .bind(session.conversation_id.as_deref())
        .bind(session.message_count as i64)
        .bind(status_to_db(session.status))
        .bind(&metadata_json)
        .bind(session.created_at.to_rfc3339())
        .bind(session.updated_at.to_rfc3339())
        .bind(session.last_accessed.to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to insert tracked session")?;

        Ok(session)
    }

    async fn get(&self, id: Uuid) -> Result<Option<TrackedSession>> {
        let row = sqlx::query(
            r#"
            SELECT id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata,
                   created_at, updated_at, last_accessed
            FROM tracked_sessions
            WHERE id = ?1
            "#,
        )
        .bind(id.to_string())
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch tracked session")?;

        row.as_ref().map(sqlite_row_to_session).transpose()
    }

    async fn get_by_slug(&self, slug: &str) -> Result<Option<TrackedSession>> {
        let row = sqlx::query(
            r#"
            SELECT id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata,
                   created_at, updated_at, last_accessed
            FROM tracked_sessions
            WHERE name_slug = ?1
            "#,
        )
        .bind(slug)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch tracked session by slug")?;

        row.as_ref().map(sqlite_row_to_session).transpose()
    }

    async fn update(&self, session: &TrackedSession) -> Result<()> {
        let metadata_json =
            serde_json::to_string(&session.metadata).context("Failed to serialize metadata")?;

        sqlx::query(
            r#"
            UPDATE tracked_sessions
            SET name_slug = ?2,
                agent_id = ?3,
                user_id = ?4,
                conversation_id = ?5,
                message_count = ?6,
                status = ?7,
                metadata = ?8,
                updated_at = ?9,
                last_accessed = ?10
            WHERE id = ?1
            "#,
        )
        .bind(session.id.to_string())
        .bind(session.name_slug.as_deref())
        .bind(&session.agent_id)
        .bind(session.user_id.as_deref())
        .bind(session.conversation_id.as_deref())
        .bind(session.message_count as i64)
        .bind(status_to_db(session.status))
        .bind(&metadata_json)
        .bind(session.updated_at.to_rfc3339())
        .bind(session.last_accessed.to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to update tracked session")?;

        Ok(())
    }

    async fn update_slug(&self, id: Uuid, slug: &str) -> Result<()> {
        let now = Utc::now();
        sqlx::query(
            r#"
            UPDATE tracked_sessions
            SET name_slug = ?2,
                updated_at = ?3
            WHERE id = ?1
            "#,
        )
        .bind(id.to_string())
        .bind(slug)
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to update session slug")?;
        Ok(())
    }

    async fn touch(&self, id: Uuid) -> Result<()> {
        let now = Utc::now();
        sqlx::query(
            r#"
            UPDATE tracked_sessions
            SET last_accessed = ?2,
                updated_at = ?3
            WHERE id = ?1
            "#,
        )
        .bind(id.to_string())
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to touch tracked session")?;
        Ok(())
    }

    async fn delete(&self, id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM tracked_sessions WHERE id = ?1")
            .bind(id.to_string())
            .execute(&self.pool)
            .await
            .context("Failed to delete tracked session")?;
        Ok(())
    }

    async fn list(&self, query: &SessionQuery) -> Result<Vec<TrackedSession>> {
        let mut sql = String::from(
            r#"
            SELECT id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata,
                   created_at, updated_at, last_accessed
            FROM tracked_sessions
            "#,
        );

        let filters = collect_filters(query);
        if !filters.is_empty() {
            let mut clauses: Vec<&'static str> = Vec::with_capacity(filters.len());
            for f in &filters {
                match f {
                    SessionFilter::AgentId(_) => clauses.push("agent_id = ?"),
                    SessionFilter::UserId(_) => clauses.push("user_id = ?"),
                    SessionFilter::Status(_) => clauses.push("status = ?"),
                }
            }
            sql.push_str(" WHERE ");
            sql.push_str(&clauses.join(" AND "));
        }

        sql.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");

        let mut q = sqlx::query(&sql);
        for f in filters {
            match f {
                SessionFilter::AgentId(agent_id) => {
                    q = q.bind(agent_id);
                }
                SessionFilter::UserId(user_id) => {
                    q = q.bind(user_id);
                }
                SessionFilter::Status(status) => {
                    q = q.bind(status_to_db(status));
                }
            }
        }

        q = q.bind(query.limit).bind(query.offset);

        let rows = q
            .fetch_all(&self.pool)
            .await
            .context("Failed to list sessions")?;

        let mut sessions = Vec::with_capacity(rows.len());
        for row in rows {
            sessions.push(sqlite_row_to_session(&row)?);
        }
        Ok(sessions)
    }

    async fn archive(&self, id: Uuid) -> Result<()> {
        let now = Utc::now();
        sqlx::query(
            r#"
            UPDATE tracked_sessions
            SET status = ?2,
                updated_at = ?3
            WHERE id = ?1
            "#,
        )
        .bind(id.to_string())
        .bind(status_to_db(SessionStatus::Archived))
        .bind(now.to_rfc3339())
        .execute(&self.pool)
        .await
        .context("Failed to archive session")?;
        Ok(())
    }

    async fn cleanup_expired(&self, older_than: Duration) -> Result<u64> {
        let threshold = Utc::now() - older_than;
        let result = sqlx::query("DELETE FROM tracked_sessions WHERE last_accessed < ?1")
            .bind(threshold.to_rfc3339())
            .execute(&self.pool)
            .await
            .context("Failed to cleanup expired sessions")?;
        Ok(result.rows_affected())
    }
}
