//! AgentFS-backed PersistenceBackend and MemoryPersistence implementations.
//!
//! Uses the AgentFS SDK's KvStore for key-value storage. Raw bytes are stored
//! as base64-encoded JSON. TTL is implemented by storing expiry in the value
//! and filtering on get (AgentFS kv_store has no native TTL).

use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::super::PersistenceBackend;
use crate::agent::memory::{MemoryPersistence, MemorySnapshot};

/// Wrapper for raw bytes with optional TTL. Stored as JSON in AgentFS KvStore.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct KvValue {
    /// Base64-encoded bytes
    #[serde(rename = "b")]
    data: String,
    /// Unix timestamp when this entry expires (None = no expiry)
    #[serde(rename = "e")]
    expires_at: Option<i64>,
}

/// Persistence backend backed by AgentFS KvStore.
#[derive(Clone)]
pub struct AgentFsPersistenceBackend {
    agentfs: Arc<agentfs_sdk::AgentFS>,
}

impl std::fmt::Debug for AgentFsPersistenceBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentFsPersistenceBackend")
            .finish_non_exhaustive()
    }
}

impl AgentFsPersistenceBackend {
    /// Create a new persistence backend from an AgentFS instance.
    pub fn new(agentfs: Arc<agentfs_sdk::AgentFS>) -> Self {
        Self { agentfs }
    }
}

#[async_trait]
impl PersistenceBackend for AgentFsPersistenceBackend {
    async fn set(&self, key: &str, value: &[u8], ttl: Option<u64>) -> Result<()> {
        let expires_at = if let Some(seconds) = ttl {
            let ttl_i64 = i64::try_from(seconds).context("AgentFS kv set TTL exceeds i64 range")?;
            Some(chrono::Utc::now().timestamp() + ttl_i64)
        } else {
            None
        };
        let kv_value = KvValue {
            data: BASE64.encode(value),
            expires_at,
        };
        self.agentfs
            .kv
            .set(key, &kv_value)
            .await
            .context("AgentFS kv set failed")
    }

    async fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let Some(kv_value) = self
            .agentfs
            .kv
            .get::<KvValue>(key)
            .await
            .context("AgentFS kv get failed")?
        else {
            return Ok(None);
        };
        if let Some(exp) = kv_value.expires_at
            && chrono::Utc::now().timestamp() >= exp
        {
            return Ok(None);
        }
        let bytes = BASE64
            .decode(&kv_value.data)
            .context("AgentFS kv value base64 decode failed")?;
        Ok(Some(bytes))
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.agentfs
            .kv
            .delete(key)
            .await
            .context("AgentFS kv delete failed")
    }

    async fn exists(&self, key: &str) -> Result<bool> {
        let val = self.get(key).await?;
        Ok(val.is_some())
    }

    async fn expire(&self, key: &str, ttl: u64) -> Result<()> {
        let Some(bytes) = self.get(key).await? else {
            return Ok(());
        };
        let ttl_i64 = i64::try_from(ttl).context("AgentFS kv expire TTL exceeds i64 range")?;
        let expires_at = chrono::Utc::now().timestamp() + ttl_i64;
        let kv_value = KvValue {
            data: BASE64.encode(&bytes),
            expires_at: Some(expires_at),
        };
        self.agentfs
            .kv
            .set(key, &kv_value)
            .await
            .context("AgentFS kv expire (re-set) failed")?;
        Ok(())
    }
}

/// Memory persistence backed by AgentFS KvStore.
#[derive(Clone)]
pub struct AgentFsMemoryPersistence {
    agentfs: Arc<agentfs_sdk::AgentFS>,
    prefix: String,
}

impl std::fmt::Debug for AgentFsMemoryPersistence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentFsMemoryPersistence")
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl AgentFsMemoryPersistence {
    /// Create memory persistence with the given key prefix (e.g. "memory").
    pub fn new(agentfs: Arc<agentfs_sdk::AgentFS>, prefix: impl Into<String>) -> Self {
        Self {
            agentfs,
            prefix: prefix.into(),
        }
    }

    fn key(&self, conversation_id: &str) -> String {
        format!("{}:{}", self.prefix, conversation_id)
    }
}

#[async_trait]
impl MemoryPersistence for AgentFsMemoryPersistence {
    async fn save_snapshot(&self, conversation_id: &str, snapshot: &MemorySnapshot) -> Result<()> {
        let key = self.key(conversation_id);
        self.agentfs
            .kv
            .set(&key, snapshot)
            .await
            .context("AgentFS memory save_snapshot failed")
    }

    async fn load_snapshot(&self, conversation_id: &str) -> Result<Option<MemorySnapshot>> {
        let key = self.key(conversation_id);
        self.agentfs
            .kv
            .get::<MemorySnapshot>(&key)
            .await
            .context("AgentFS memory load_snapshot failed")
    }

    async fn clear_snapshot(&self, conversation_id: &str) -> Result<()> {
        let key = self.key(conversation_id);
        self.agentfs
            .kv
            .delete(&key)
            .await
            .context("AgentFS memory clear_snapshot failed")
    }
}
