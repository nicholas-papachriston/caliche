pub mod adapters;
pub mod backend;
#[cfg(feature = "sqlx")]
pub mod connection_manager;
pub mod conversation_store;
#[cfg(feature = "sqlite-store")]
pub mod credential_store;
#[cfg(feature = "redis")]
pub(crate) mod redis_codec;
#[cfg(feature = "redis")]
pub mod redis_session_registry;
#[cfg(feature = "redis")]
pub mod redis_store;
pub(crate) mod rkyv_envelope;
pub mod safety;
pub mod schema;
pub mod schema_cache;
#[cfg(any(feature = "sqlx", feature = "sqlite-store"))]
pub(crate) mod session_mapper;
pub mod session_registry;
pub mod session_tracker;
#[cfg(any(
    feature = "sqlx",
    feature = "sqlite-store",
    feature = "durable-streams-sqlite"
))]
pub(crate) mod sqlite_pool;
pub mod task_ledger;

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[cfg(feature = "agentfs")]
pub use backend::agentfs_kv::{AgentFsMemoryPersistence, AgentFsPersistenceBackend};
#[cfg(feature = "sqlx")]
pub use backend::postgres::{
    ConversationTableConfig, ConversationTableConfigBuilder, PostgresConversationStore,
};
#[cfg(feature = "sqlx")]
pub use backend::postgres_session::PostgresSessionRegistry;
#[cfg(feature = "sqlite-store")]
pub use backend::sqlite::SqliteConversationStore;
#[cfg(feature = "sqlite-store")]
pub use backend::sqlite_kv::SqlitePersistenceBackend;
#[cfg(feature = "sqlite-store")]
pub use backend::sqlite_session::SqliteSessionRegistry;
pub use backend::{
    ConversationListQuery, ConversationStats, ConversationStoreBackend, MetadataQuery,
};
#[cfg(feature = "sqlx")]
pub use connection_manager::{
    DatabaseConfig, DatabaseConnection, DatabaseConnectionManager, DatabasePool, DatabaseType,
};
pub use conversation_store::ConversationStore;
#[cfg(feature = "sqlite-store")]
pub use credential_store::{CredentialMetadata, CredentialStoreBackend, SqliteCredentialStore};
#[cfg(feature = "redis")]
pub use redis_session_registry::RedisSessionRegistry;
#[cfg(feature = "redis")]
pub use redis_store::RedisSessionStore;
#[cfg(feature = "sqlx")]
pub use schema::PostgresSchemaIntrospector;
pub use schema::{
    ColumnSchema, DatabaseIntrospector, DatabaseSchema, ForeignKeySchema, IndexSchema,
    SchemaDocument, TableSchema,
};
pub use schema_cache::{CacheStats, SchemaCache};
pub use session_registry::{
    SessionQuery, SessionRegistry, SessionStatus, TrackedSession, generate_session_id,
    generate_time_id,
};
pub use session_tracker::{SessionTracker, SlugGenerator};

#[cfg(feature = "tools-rag")]
pub use schema::{SchemaEmbeddingPipeline, SchemaSyncReport};

/// Generic persistence trait for different backends
#[async_trait]
pub trait PersistenceBackend: Send + Sync + std::fmt::Debug {
    /// Store data with key-value semantics
    async fn set(&self, key: &str, value: &[u8], ttl: Option<u64>) -> Result<()>;

    /// Retrieve data by key
    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Delete data by key
    async fn delete(&self, key: &str) -> Result<()>;

    /// Check if key exists
    async fn exists(&self, key: &str) -> Result<bool>;

    /// Set expiration time
    async fn expire(&self, key: &str, ttl: u64) -> Result<()>;
}

/// Conversation record for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRecord {
    pub id: String,
    pub agent_id: String,
    pub messages: Vec<crate::messages::Message>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Session data for fast access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub session_id: String,
    pub agent_id: String,
    pub conversation_id: String,
    pub user_id: Option<String>,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
}
