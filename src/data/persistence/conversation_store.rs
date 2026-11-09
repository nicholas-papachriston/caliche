use std::sync::Arc;

use anyhow::Result;
use serde_json::Value;

#[cfg(feature = "sqlite-store")]
use super::SqliteConversationStore;
use super::{
    ConversationListQuery, ConversationRecord, ConversationStats, ConversationStoreBackend,
    MetadataQuery,
};
#[cfg(feature = "sqlx")]
use super::{ConversationTableConfig, PostgresConversationStore};

/// High-level facade for managing conversation persistence.
pub struct ConversationStore {
    backend: Arc<dyn ConversationStoreBackend>,
}

impl ConversationStore {
    /// Construct a store using the default `PostgreSQL` backend.
    #[cfg(feature = "sqlx")]
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn new(database_url: &str) -> Result<Self> {
        Self::with_postgres(database_url).await
    }

    /// Construct a store using the `PostgreSQL` backend.
    #[cfg(feature = "sqlx")]
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn with_postgres(database_url: &str) -> Result<Self> {
        let backend = PostgresConversationStore::new(database_url).await?;
        Ok(Self {
            backend: Arc::new(backend),
        })
    }

    /// Construct a store using the `PostgreSQL` backend and custom table configuration.
    #[cfg(feature = "sqlx")]
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn with_postgres_config(
        database_url: &str,
        config: ConversationTableConfig,
    ) -> Result<Self> {
        let backend = PostgresConversationStore::new_with_config(database_url, config).await?;
        Ok(Self {
            backend: Arc::new(backend),
        })
    }

    /// Construct a store using the SQLite backend.
    #[cfg(feature = "sqlite-store")]
    pub async fn with_sqlite(database_url: &str) -> Result<Self> {
        let backend = SqliteConversationStore::new(database_url).await?;
        Ok(Self {
            backend: Arc::new(backend),
        })
    }

    /// Construct a store using the SQLite backend from a file path.
    #[cfg(feature = "sqlite-store")]
    pub async fn with_sqlite_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let backend = SqliteConversationStore::from_path(path).await?;
        Ok(Self {
            backend: Arc::new(backend),
        })
    }

    /// Create a store from an arbitrary backend.
    pub fn from_backend<B: ConversationStoreBackend + 'static>(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Create a store from an existing backend handle.
    pub fn from_backend_arc(backend: Arc<dyn ConversationStoreBackend>) -> Self {
        Self { backend }
    }

    /// Access the underlying backend.
    #[must_use]
    pub fn backend(&self) -> Arc<dyn ConversationStoreBackend> {
        Arc::clone(&self.backend)
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn save(&self, conversation: &ConversationRecord) -> Result<()> {
        self.backend.save(conversation).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get(&self, id: &str) -> Result<Option<ConversationRecord>> {
        self.backend.fetch(id).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn list_by_agent(
        &self,
        agent_id: &str,
        limit: i64,
        offset: i64,
    ) -> Result<Vec<ConversationRecord>> {
        let query = ConversationListQuery::new(agent_id, limit, offset);
        self.backend.list(&query).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn delete(&self, id: &str) -> Result<()> {
        self.backend.delete(id).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn cleanup_expired(&self) -> Result<u64> {
        self.backend.cleanup_expired().await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn update_ttl(&self, id: &str, ttl_seconds: i64) -> Result<()> {
        self.backend.update_ttl(id, ttl_seconds).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn search_by_metadata(
        &self,
        json_path: &str,
        value: &Value,
        limit: i64,
    ) -> Result<Vec<ConversationRecord>> {
        let mut query = MetadataQuery::new(value.clone(), limit);
        if !json_path.is_empty() {
            query = query.with_json_path(json_path);
        }
        self.backend.search(&query).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get_stats(&self) -> Result<ConversationStats> {
        self.backend.stats().await
    }
}
