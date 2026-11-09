#[cfg(feature = "sqlx")]
use crate::data::persistence::backend::postgres_session::PostgresSessionRegistry;
#[cfg(feature = "sqlite-store")]
use crate::data::persistence::backend::sqlite_session::SqliteSessionRegistry;
#[cfg(feature = "redis")]
use crate::data::persistence::redis_session_registry::RedisSessionRegistry;
use crate::data::persistence::session_registry::{SessionQuery, SessionRegistry, TrackedSession};
use crate::messages::Message;
use anyhow::Result;
use async_trait::async_trait;
use chrono::Duration;
use std::sync::Arc;
use uuid::Uuid;

#[async_trait]
pub trait SlugGenerator: Send + Sync {
    async fn generate_slug(&self, messages: &[Message]) -> Result<String>;
}

/// Facade for session tracking across different backends.
#[derive(Debug, Clone)]
pub struct SessionTracker {
    registry: Arc<dyn SessionRegistry>,
}

impl SessionTracker {
    pub fn new(registry: Arc<dyn SessionRegistry>) -> Self {
        Self { registry }
    }

    #[must_use]
    pub fn registry(&self) -> &Arc<dyn SessionRegistry> {
        &self.registry
    }

    #[cfg(feature = "sqlite-store")]
    pub async fn with_sqlite(database_url: &str) -> Result<Self> {
        let registry = SqliteSessionRegistry::new(database_url).await?;
        Ok(Self::new(Arc::new(registry)))
    }

    #[cfg(feature = "sqlx")]
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn with_postgres(database_url: &str) -> Result<Self> {
        let registry = PostgresSessionRegistry::new(database_url).await?;
        Ok(Self::new(Arc::new(registry)))
    }

    #[cfg(feature = "redis")]
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub fn with_redis(redis_url: &str, prefix: Option<String>) -> Result<Self> {
        let registry = RedisSessionRegistry::new(redis_url, prefix)?;
        Ok(Self::new(Arc::new(registry)))
    }

    /// Run any one-time backend initialization (schema creation, table
    /// setup, Redis key warmup, etc.).
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn init(&self) -> Result<()> {
        self.registry.init().await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn create(&self, agent_id: &str, user_id: Option<&str>) -> Result<TrackedSession> {
        self.registry.create(agent_id, user_id).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn create_with_id(
        &self,
        id: Uuid,
        agent_id: &str,
        user_id: Option<&str>,
    ) -> Result<TrackedSession> {
        self.registry.create_with_id(id, agent_id, user_id).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get(&self, id: Uuid) -> Result<Option<TrackedSession>> {
        self.registry.get(id).await
    }

    /// Look up a session by its human-readable slug.
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn get_by_slug(&self, slug: &str) -> Result<Option<TrackedSession>> {
        self.registry.get_by_slug(slug).await
    }

    /// Persist the given session record, overwriting the existing row.
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn update(&self, session: &TrackedSession) -> Result<()> {
        self.registry.update(session).await
    }

    /// Replace only the name slug on the given session.
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn update_slug(&self, id: Uuid, slug: &str) -> Result<()> {
        self.registry.update_slug(id, slug).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn list(&self, query: &SessionQuery) -> Result<Vec<TrackedSession>> {
        self.registry.list(query).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn touch(&self, id: Uuid) -> Result<()> {
        self.registry.touch(id).await
    }

    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn delete(&self, id: Uuid) -> Result<()> {
        self.registry.delete(id).await
    }

    /// Mark the session as archived without deleting it.
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn archive(&self, id: Uuid) -> Result<()> {
        self.registry.archive(id).await
    }

    /// Delete sessions that have been inactive for longer than `older_than`.
    /// Returns the number of sessions removed.
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn cleanup_expired(&self, older_than: Duration) -> Result<u64> {
        self.registry.cleanup_expired(older_than).await
    }

    /// Ensure a session has a name slug. If missing, generate one lazily using an LLM.
    /// # Errors
    ///
    /// Returns an error if the persistence or storage operation fails.
    pub async fn ensure_slug(
        &self,
        session_id: Uuid,
        messages: &[Message],
        slug_generator: &dyn SlugGenerator,
    ) -> Result<String> {
        let Some(session) = self.registry.get(session_id).await? else {
            anyhow::bail!("Session not found: {session_id}");
        };

        if let Some(slug) = session.name_slug {
            return Ok(slug);
        }

        let slug = slug_generator.generate_slug(messages).await?;
        self.registry.update_slug(session_id, &slug).await?;
        Ok(slug)
    }
}
