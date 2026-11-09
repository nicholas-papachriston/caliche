use crate::gateway::types::ChannelKind;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique key used to bind channel users/threads to durable agent sessions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BindingKey {
    pub tenant_id: Option<String>,
    pub channel_kind: ChannelKind,
    pub channel_id: String,
    pub user_id: String,
    pub thread_id: Option<String>,
}

/// Persistent mapping between a channel identity and an agent session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SessionBinding {
    pub key: BindingKey,
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[async_trait]
pub trait SessionBindingStore: Send + Sync {
    async fn get(&self, key: &BindingKey) -> Result<Option<SessionBinding>>;
    async fn put(&self, binding: SessionBinding) -> Result<()>;
    /// Atomically return an existing binding or insert `candidate`.
    ///
    /// Returns `(binding, created)` where `created` is true only when `candidate` was inserted.
    async fn get_or_insert(&self, candidate: SessionBinding) -> Result<(SessionBinding, bool)>;
    async fn delete(&self, key: &BindingKey) -> Result<()>;
    async fn list_for_session(&self, session_id: &str) -> Result<Vec<SessionBinding>>;
}

/// Session factory abstraction so routing can create sessions without coupling
/// directly to API adapters.
#[async_trait]
pub trait SessionFactory: Send + Sync {
    async fn create_session(
        &self,
        tenant_id: Option<&str>,
        system_prompt_hint: Option<&str>,
    ) -> Result<String>;
}

/// In-memory binding store. Useful as a default and for tests.
#[derive(Debug, Default)]
pub struct MemorySessionBindingStore {
    map: DashMap<BindingKey, SessionBinding>,
}

impl MemorySessionBindingStore {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionBindingStore for MemorySessionBindingStore {
    async fn get(&self, key: &BindingKey) -> Result<Option<SessionBinding>> {
        Ok(self.map.get(key).as_deref().cloned())
    }

    async fn put(&self, binding: SessionBinding) -> Result<()> {
        self.map.insert(binding.key.clone(), binding);
        Ok(())
    }

    async fn get_or_insert(&self, candidate: SessionBinding) -> Result<(SessionBinding, bool)> {
        use dashmap::mapref::entry::Entry;

        match self.map.entry(candidate.key.clone()) {
            Entry::Occupied(entry) => Ok((entry.get().clone(), false)),
            Entry::Vacant(entry) => {
                entry.insert(candidate.clone());
                Ok((candidate, true))
            }
        }
    }

    async fn delete(&self, key: &BindingKey) -> Result<()> {
        self.map.remove(key);
        Ok(())
    }

    async fn list_for_session(&self, session_id: &str) -> Result<Vec<SessionBinding>> {
        let mut out = Vec::new();
        for row in &self.map {
            if row.value().session_id == session_id {
                out.push(row.value().clone());
            }
        }
        Ok(out)
    }
}
