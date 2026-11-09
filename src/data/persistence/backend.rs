#[cfg(feature = "agentfs")]
pub mod agentfs_kv;
#[cfg(feature = "sqlx")]
pub mod postgres;
#[cfg(feature = "sqlx")]
pub mod postgres_session;
#[cfg(any(feature = "sqlx", feature = "sqlite-store"))]
pub(crate) mod sql_session;
#[cfg(feature = "sqlite-store")]
pub mod sqlite;
#[cfg(feature = "sqlite-store")]
pub mod sqlite_kv;
#[cfg(feature = "sqlite-store")]
pub mod sqlite_session;

use super::ConversationRecord;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Parameters for listing conversations belonging to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationListQuery {
    pub agent_id: String,
    pub limit: i64,
    pub offset: i64,
}

impl ConversationListQuery {
    pub fn new(agent_id: impl Into<String>, limit: i64, offset: i64) -> Self {
        Self {
            agent_id: agent_id.into(),
            limit,
            offset,
        }
    }
}

/// Metadata-based search request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataQuery {
    pub filter: Value,
    pub limit: i64,
    pub json_path: Option<String>,
}

impl MetadataQuery {
    #[must_use]
    pub const fn new(filter: Value, limit: i64) -> Self {
        Self {
            filter,
            limit,
            json_path: None,
        }
    }

    #[must_use]
    pub fn with_json_path(mut self, path: impl Into<String>) -> Self {
        self.json_path = Some(path.into());
        self
    }
}

/// Summary statistics for stored conversations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationStats {
    pub total_conversations: i64,
    pub active_conversations: i64,
    pub expired_conversations: i64,
    pub avg_message_count: f64,
}

/// Abstraction over conversation persistence backends.
#[async_trait]
pub trait ConversationStoreBackend: Send + Sync {
    /// Perform any required initialization (schema creation, index warmup, etc.).
    async fn init(&self) -> Result<()>;

    /// Persist or update a conversation.
    async fn save(&self, conversation: &ConversationRecord) -> Result<()>;

    /// Fetch a conversation by identifier.
    async fn fetch(&self, id: &str) -> Result<Option<ConversationRecord>>;

    /// List conversations for a given agent.
    async fn list(&self, query: &ConversationListQuery) -> Result<Vec<ConversationRecord>>;

    /// Delete a conversation by identifier.
    async fn delete(&self, id: &str) -> Result<()>;

    /// Remove any expired conversations, returning the number of rows affected.
    async fn cleanup_expired(&self) -> Result<u64>;

    /// Update the TTL for a conversation.
    async fn update_ttl(&self, id: &str, ttl_seconds: i64) -> Result<()>;

    /// Search conversations by metadata filter.
    async fn search(&self, query: &MetadataQuery) -> Result<Vec<ConversationRecord>>;

    /// Retrieve high-level statistics about stored conversations.
    async fn stats(&self) -> Result<ConversationStats>;
}
