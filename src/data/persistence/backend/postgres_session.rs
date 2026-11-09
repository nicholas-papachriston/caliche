use crate::data::persistence::backend::sql_session::{
    SessionFilter, collect_filters, new_tracked_session, status_to_db,
};
use crate::data::persistence::session_mapper::pg_row_to_session;
use crate::data::persistence::session_registry::{
    SessionQuery, SessionRegistry, SessionStatus, TrackedSession, generate_session_id,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Duration, Utc};
use sqlx::{PgPool, postgres::PgPoolOptions, types::Json};
use uuid::Uuid;

fn message_count_to_db(message_count: u32) -> Result<i32> {
    i32::try_from(message_count).context("Session message count exceeds PostgreSQL i32 range")
}

#[derive(Debug)]
pub struct PostgresSessionRegistry {
    pool: PgPool,
}

impl PostgresSessionRegistry {
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(10)
            .connect(database_url)
            .await
            .context("Failed to connect to PostgreSQL database")?;

        let registry = Self { pool };
        registry.init().await?;
        Ok(registry)
    }

    #[must_use]
    pub const fn from_pool(pool: PgPool) -> Self {
        Self { pool }
    }

    #[must_use]
    pub const fn pool(&self) -> &PgPool {
        &self.pool
    }

    // status mapping lives in backend::sql_session
}

#[async_trait]
impl SessionRegistry for PostgresSessionRegistry {
    async fn init(&self) -> Result<()> {
        sqlx::query(
            r"
            CREATE TABLE IF NOT EXISTS tracked_sessions (
                id UUID PRIMARY KEY,
                name_slug TEXT,
                agent_id TEXT NOT NULL,
                user_id TEXT,
                conversation_id TEXT,
                message_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                last_accessed TIMESTAMPTZ NOT NULL
            )
            ",
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

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tracked_sessions_created ON tracked_sessions(created_at DESC)")
            .execute(&self.pool)
            .await
            .context("Failed to create created_at index")?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tracked_sessions_metadata ON tracked_sessions USING GIN(metadata)")
            .execute(&self.pool)
            .await
            .context("Failed to create metadata GIN index")?;

        Ok(())
    }

    async fn create(&self, agent_id: &str, user_id: Option<&str>) -> Result<TrackedSession> {
        let id = generate_session_id();
        let session = new_tracked_session(id, agent_id, user_id);
        let message_count = message_count_to_db(session.message_count)?;

        sqlx::query(
            r"
            INSERT INTO tracked_sessions
                (id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata, created_at, updated_at, last_accessed)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ",
        )
        .bind(session.id)
        .bind(session.name_slug.as_deref())
        .bind(&session.agent_id)
        .bind(session.user_id.as_deref())
        .bind(session.conversation_id.as_deref())
        .bind(message_count)
        .bind(status_to_db(session.status))
        .bind(Json(session.metadata.clone()))
        .bind(session.created_at)
        .bind(session.updated_at)
        .bind(session.last_accessed)
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
        let message_count = message_count_to_db(session.message_count)?;

        sqlx::query(
            r"
            INSERT INTO tracked_sessions
                (id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata, created_at, updated_at, last_accessed)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ",
        )
        .bind(session.id)
        .bind(session.name_slug.as_deref())
        .bind(&session.agent_id)
        .bind(session.user_id.as_deref())
        .bind(session.conversation_id.as_deref())
        .bind(message_count)
        .bind(status_to_db(session.status))
        .bind(Json(session.metadata.clone()))
        .bind(session.created_at)
        .bind(session.updated_at)
        .bind(session.last_accessed)
        .execute(&self.pool)
        .await
        .context("Failed to insert tracked session")?;

        Ok(session)
    }

    async fn get(&self, id: Uuid) -> Result<Option<TrackedSession>> {
        let row = sqlx::query(
            r"
            SELECT id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata,
                   created_at, updated_at, last_accessed
            FROM tracked_sessions
            WHERE id = $1
            ",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch tracked session")?;

        let Some(row) = row else {
            return Ok(None);
        };
        Ok(Some(pg_row_to_session(&row)?))
    }

    async fn get_by_slug(&self, slug: &str) -> Result<Option<TrackedSession>> {
        let row = sqlx::query(
            r"
            SELECT id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata,
                   created_at, updated_at, last_accessed
            FROM tracked_sessions
            WHERE name_slug = $1
            ",
        )
        .bind(slug)
        .fetch_optional(&self.pool)
        .await
        .context("Failed to fetch tracked session by slug")?;

        let Some(row) = row else {
            return Ok(None);
        };
        Ok(Some(pg_row_to_session(&row)?))
    }

    async fn update(&self, session: &TrackedSession) -> Result<()> {
        let message_count = message_count_to_db(session.message_count)?;
        sqlx::query(
            r"
            UPDATE tracked_sessions
            SET name_slug = $2,
                agent_id = $3,
                user_id = $4,
                conversation_id = $5,
                message_count = $6,
                status = $7,
                metadata = $8,
                updated_at = $9,
                last_accessed = $10
            WHERE id = $1
            ",
        )
        .bind(session.id)
        .bind(session.name_slug.as_deref())
        .bind(&session.agent_id)
        .bind(session.user_id.as_deref())
        .bind(session.conversation_id.as_deref())
        .bind(message_count)
        .bind(status_to_db(session.status))
        .bind(Json(session.metadata.clone()))
        .bind(session.updated_at)
        .bind(session.last_accessed)
        .execute(&self.pool)
        .await
        .context("Failed to update tracked session")?;
        Ok(())
    }

    async fn update_slug(&self, id: Uuid, slug: &str) -> Result<()> {
        let now = Utc::now();
        sqlx::query(
            r"
            UPDATE tracked_sessions
            SET name_slug = $2,
                updated_at = $3
            WHERE id = $1
            ",
        )
        .bind(id)
        .bind(slug)
        .bind(now)
        .execute(&self.pool)
        .await
        .context("Failed to update session slug")?;
        Ok(())
    }

    async fn touch(&self, id: Uuid) -> Result<()> {
        let now = Utc::now();
        sqlx::query(
            r"
            UPDATE tracked_sessions
            SET last_accessed = $2,
                updated_at = $3
            WHERE id = $1
            ",
        )
        .bind(id)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await
        .context("Failed to touch tracked session")?;
        Ok(())
    }

    async fn delete(&self, id: Uuid) -> Result<()> {
        sqlx::query("DELETE FROM tracked_sessions WHERE id = $1")
            .bind(id)
            .execute(&self.pool)
            .await
            .context("Failed to delete tracked session")?;
        Ok(())
    }

    async fn list(&self, query: &SessionQuery) -> Result<Vec<TrackedSession>> {
        let mut qb = sqlx::QueryBuilder::new(
            r"
            SELECT id, name_slug, agent_id, user_id, conversation_id, message_count, status, metadata,
                   created_at, updated_at, last_accessed
            FROM tracked_sessions
            ",
        );

        let filters = collect_filters(query);
        let mut has_where = false;
        for f in filters {
            qb.push(if has_where { " AND " } else { " WHERE " });
            has_where = true;
            match f {
                SessionFilter::AgentId(agent_id) => {
                    qb.push("agent_id = ").push_bind(agent_id);
                }
                SessionFilter::UserId(user_id) => {
                    qb.push("user_id = ").push_bind(user_id);
                }
                SessionFilter::Status(status) => {
                    qb.push("status = ").push_bind(status_to_db(status));
                }
            }
        }

        qb.push(" ORDER BY created_at DESC LIMIT ")
            .push_bind(query.limit)
            .push(" OFFSET ")
            .push_bind(query.offset);

        let rows = qb
            .build()
            .fetch_all(&self.pool)
            .await
            .context("Failed to list sessions")?;

        let mut sessions = Vec::with_capacity(rows.len());
        for row in rows {
            sessions.push(pg_row_to_session(&row)?);
        }

        Ok(sessions)
    }

    async fn archive(&self, id: Uuid) -> Result<()> {
        let now = Utc::now();
        sqlx::query(
            r"
            UPDATE tracked_sessions
            SET status = $2,
                updated_at = $3
            WHERE id = $1
            ",
        )
        .bind(id)
        .bind(status_to_db(SessionStatus::Archived))
        .bind(now)
        .execute(&self.pool)
        .await
        .context("Failed to archive session")?;
        Ok(())
    }

    async fn cleanup_expired(&self, older_than: Duration) -> Result<u64> {
        let threshold = Utc::now() - older_than;
        let result = sqlx::query("DELETE FROM tracked_sessions WHERE last_accessed < $1")
            .bind(threshold)
            .execute(&self.pool)
            .await
            .context("Failed to cleanup expired sessions")?;
        Ok(result.rows_affected())
    }
}

#[cfg(test)]
mod tests {
    use super::message_count_to_db;

    #[test]
    fn message_count_conversion_rejects_database_overflow() {
        assert_eq!(
            message_count_to_db(u32::try_from(i32::MAX).unwrap()).unwrap(),
            i32::MAX
        );
        assert!(message_count_to_db(u32::MAX).is_err());
    }
}
