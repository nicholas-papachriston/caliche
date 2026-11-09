use std::{fmt, sync::Arc};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{Duration, Utc};
use serde_json::{self, Value};
use uuid::Uuid;

use crate::{
    agent::memory::base::{MemoryPersistence, MemorySnapshot, MemoryStats, MemoryStrategy},
    data::persistence::{ConversationRecord, ConversationStore, PersistenceBackend},
    messages::Message,
};

/// Wraps a memory strategy with persistence support.
///
/// Propagates errors from the inner strategy and from persistence; does not silently overwrite
/// good state with a failed save. Transient persistence failures may be retried once with backoff.
pub struct PersistentMemory<M: MemoryStrategy> {
    inner: M,
    persistence: Arc<dyn MemoryPersistence>,
    conversation_id: String,
    autosave: bool,
}

impl<M: MemoryStrategy> PersistentMemory<M> {
    pub fn new(
        inner: M,
        persistence: Arc<dyn MemoryPersistence>,
        conversation_id: impl Into<String>,
    ) -> Self {
        Self {
            inner,
            persistence,
            conversation_id: conversation_id.into(),
            autosave: true,
        }
    }

    /// Enable or disable automatic persistence after each update.
    pub const fn with_autosave(mut self, autosave: bool) -> Self {
        self.autosave = autosave;
        self
    }

    /// Load existing memory state from persistence.
    /// On failure, the inner state is left unchanged (or partially loaded depending on the strategy);
    /// the error is returned so the caller can decide (e.g. start fresh).
    pub async fn hydrate(&mut self) -> Result<()> {
        self.inner
            .load(Some(&*self.persistence), &self.conversation_id)
            .await
            .context("PersistentMemory hydrate failed")
    }

    /// Persist the current memory state.
    /// On failure, retries once after a short backoff (50ms) for transient I/O errors.
    pub async fn flush(&self) -> Result<()> {
        const RETRY_BACKOFF_MS: u64 = 50;
        let result = self
            .inner
            .save(Some(&*self.persistence), &self.conversation_id)
            .await
            .context("PersistentMemory flush failed");
        if let Err(ref e) = result {
            tracing::warn!(error = %e, conversation_id = %self.conversation_id, "PersistentMemory flush failed, retrying once");
            tokio::time::sleep(std::time::Duration::from_millis(RETRY_BACKOFF_MS)).await;
            return self
                .inner
                .save(Some(&*self.persistence), &self.conversation_id)
                .await
                .context("PersistentMemory flush retry failed");
        }
        result
    }

    pub fn conversation_id(&self) -> &str {
        &self.conversation_id
    }

    pub fn into_inner(self) -> M {
        self.inner
    }
}

#[async_trait]
impl<M: MemoryStrategy> MemoryStrategy for PersistentMemory<M> {
    async fn add_messages(&mut self, messages: Vec<Message>) -> Result<()> {
        self.inner
            .add_messages(messages)
            .await
            .context("PersistentMemory add_messages: inner strategy failed")?;
        if self.autosave {
            self.flush()
                .await
                .context("PersistentMemory add_messages: flush after add failed")?;
        }
        Ok(())
    }

    async fn get_context(&self) -> Result<Vec<Message>> {
        self.inner.get_context().await
    }

    async fn clear(&mut self) -> Result<()> {
        self.inner
            .clear()
            .await
            .context("PersistentMemory clear: inner strategy failed")?;
        if self.autosave {
            self.flush()
                .await
                .context("PersistentMemory clear: flush after clear failed")?;
        }
        Ok(())
    }

    fn stats(&self) -> MemoryStats {
        self.inner.stats()
    }

    async fn save(
        &self,
        persistence: Option<&dyn MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        if let Some(persistence) = persistence {
            self.inner.save(Some(persistence), conversation_id).await
        } else {
            self.inner
                .save(Some(&*self.persistence), &self.conversation_id)
                .await
        }
    }

    async fn load(
        &mut self,
        persistence: Option<&dyn MemoryPersistence>,
        conversation_id: &str,
    ) -> Result<()> {
        if let Some(persistence) = persistence {
            self.inner.load(Some(persistence), conversation_id).await
        } else {
            self.inner
                .load(Some(&*self.persistence), &self.conversation_id)
                .await
        }
    }

    async fn save_state(&self) -> Result<MemorySnapshot> {
        self.inner.save_state().await
    }

    async fn load_state(&mut self, snapshot: MemorySnapshot) -> Result<()> {
        self.inner.load_state(snapshot).await?;
        if self.autosave {
            self.flush().await?;
        }
        Ok(())
    }

    async fn fork(&self) -> Result<Box<dyn MemoryStrategy>> {
        let forked = self.inner.fork().await?;
        let fork_id = format!("{}-fork-{}", self.conversation_id, Uuid::now_v7());
        Ok(Box::new(PersistentMemory::new(
            forked,
            Arc::clone(&self.persistence),
            fork_id,
        )))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self.inner.as_any()
    }
}

/// Memory persistence backed by a generic key-value backend.
#[derive(Debug)]
pub struct BackendMemoryPersistence<B: PersistenceBackend + 'static + fmt::Debug> {
    backend: Arc<B>,
    key_prefix: String,
    ttl_seconds: Option<u64>,
}

impl<B: PersistenceBackend + 'static + fmt::Debug> BackendMemoryPersistence<B> {
    pub fn new(backend: Arc<B>, key_prefix: impl Into<String>, ttl_seconds: Option<u64>) -> Self {
        Self {
            backend,
            key_prefix: key_prefix.into(),
            ttl_seconds,
        }
    }

    fn key(&self, conversation_id: &str) -> String {
        format!("{}{conversation_id}", self.key_prefix)
    }
}

#[async_trait]
impl<B: PersistenceBackend + 'static + fmt::Debug> MemoryPersistence
    for BackendMemoryPersistence<B>
{
    async fn save_snapshot(&self, conversation_id: &str, snapshot: &MemorySnapshot) -> Result<()> {
        let data = serde_json::to_vec(snapshot)?;
        self.backend
            .set(&self.key(conversation_id), &data, self.ttl_seconds)
            .await
    }

    async fn load_snapshot(&self, conversation_id: &str) -> Result<Option<MemorySnapshot>> {
        if let Some(bytes) = self.backend.get(&self.key(conversation_id)).await? {
            let snapshot = serde_json::from_slice(&bytes)?;
            Ok(Some(snapshot))
        } else {
            Ok(None)
        }
    }

    async fn clear_snapshot(&self, conversation_id: &str) -> Result<()> {
        self.backend.delete(&self.key(conversation_id)).await
    }
}

/// Memory persistence backed by the `PostgreSQL` conversation store.
pub struct ConversationStoreMemoryPersistence {
    store: Arc<ConversationStore>,
    agent_id: String,
    ttl_seconds: Option<u64>,
}

impl ConversationStoreMemoryPersistence {
    pub fn new(
        store: Arc<ConversationStore>,
        agent_id: impl Into<String>,
        ttl_seconds: Option<u64>,
    ) -> Self {
        Self {
            store,
            agent_id: agent_id.into(),
            ttl_seconds,
        }
    }
}

impl fmt::Debug for ConversationStoreMemoryPersistence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConversationStoreMemoryPersistence")
            .field("agent_id", &self.agent_id)
            .field("ttl_seconds", &self.ttl_seconds)
            .finish()
    }
}

#[async_trait]
impl MemoryPersistence for ConversationStoreMemoryPersistence {
    async fn save_snapshot(&self, conversation_id: &str, snapshot: &MemorySnapshot) -> Result<()> {
        let mut record = if let Some(existing) = self.store.get(conversation_id).await? {
            existing
        } else {
            ConversationRecord {
                id: conversation_id.to_string(),
                agent_id: self.agent_id.clone(),
                messages: Vec::new(),
                metadata: Value::Null,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                expires_at: None,
            }
        };

        record.messages = snapshot.messages.clone();
        record.metadata = snapshot.metadata.clone();
        record.updated_at = Utc::now();

        if let Some(ttl) = self.ttl_seconds {
            record.expires_at = Some(record.updated_at + Duration::seconds(ttl as i64));
        }

        self.store.save(&record).await
    }

    async fn load_snapshot(&self, conversation_id: &str) -> Result<Option<MemorySnapshot>> {
        if let Some(record) = self.store.get(conversation_id).await? {
            Ok(Some(MemorySnapshot {
                messages: record.messages,
                metadata: record.metadata,
            }))
        } else {
            Ok(None)
        }
    }

    async fn clear_snapshot(&self, conversation_id: &str) -> Result<()> {
        self.store.delete(conversation_id).await
    }
}
