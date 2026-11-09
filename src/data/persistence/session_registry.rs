use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Status of a persisted/tracked session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Archived,
    Expired,
}

impl SessionStatus {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Archived => "archived",
            Self::Expired => "expired",
        }
    }
}

/// Query options for listing sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<SessionStatus>,
    pub limit: i64,
    pub offset: i64,
}

impl SessionQuery {
    #[must_use]
    pub const fn new(limit: i64, offset: i64) -> Self {
        Self {
            agent_id: None,
            user_id: None,
            status: None,
            limit,
            offset,
        }
    }

    #[must_use]
    pub fn for_agent(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    #[must_use]
    pub fn for_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    #[must_use]
    pub const fn with_status(mut self, status: SessionStatus) -> Self {
        self.status = Some(status);
        self
    }
}

/// Persisted metadata about a session, optimized for fast retrieval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedSession {
    pub id: Uuid,
    pub name_slug: Option<String>,
    pub agent_id: String,
    pub user_id: Option<String>,
    pub conversation_id: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub metadata: serde_json::Value,
    pub message_count: u32,
    pub status: SessionStatus,
}

impl TrackedSession {
    pub fn new(id: Uuid, agent_id: impl Into<String>, user_id: Option<String>) -> Self {
        let now = Utc::now();
        Self {
            id,
            name_slug: None,
            agent_id: agent_id.into(),
            user_id,
            conversation_id: None,
            created_at: now,
            updated_at: now,
            last_accessed: now,
            metadata: serde_json::json!({}),
            message_count: 0,
            status: SessionStatus::Active,
        }
    }
}

pub use crate::utils::time_id::generate_time_id;

/// Generate a `UUIDv7` session id (time-sortable).
///
/// Sessions use the same clock-ordered id as tool actions and thinking segments.
#[must_use]
pub fn generate_session_id() -> Uuid {
    generate_time_id()
}

#[async_trait]
pub trait SessionRegistry: Send + Sync + std::fmt::Debug {
    async fn init(&self) -> Result<()>;

    async fn create(&self, agent_id: &str, user_id: Option<&str>) -> Result<TrackedSession>;

    /// Create a tracked session with a caller-provided `UUIDv7`.
    ///
    /// This is useful when the session id is generated outside the registry (e.g., by an agent
    /// harness) but we still want a unified tracking record.
    async fn create_with_id(
        &self,
        id: Uuid,
        agent_id: &str,
        user_id: Option<&str>,
    ) -> Result<TrackedSession>;

    async fn get(&self, id: Uuid) -> Result<Option<TrackedSession>>;

    async fn get_by_slug(&self, slug: &str) -> Result<Option<TrackedSession>>;

    async fn update(&self, session: &TrackedSession) -> Result<()>;

    async fn update_slug(&self, id: Uuid, slug: &str) -> Result<()>;

    /// Update `last_accessed` (and optionally `updated_at`) for a session.
    async fn touch(&self, id: Uuid) -> Result<()>;

    async fn delete(&self, id: Uuid) -> Result<()>;

    async fn list(&self, query: &SessionQuery) -> Result<Vec<TrackedSession>>;

    async fn archive(&self, id: Uuid) -> Result<()>;

    /// Cleanup sessions whose last access time is older than `older_than`.
    async fn cleanup_expired(&self, older_than: Duration) -> Result<u64>;
}
