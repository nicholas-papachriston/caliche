//! AgentFS-backed SessionStore implementation.
//!
//! Uses the AgentFS KvStore to persist session snapshots. Keys:
//! - `session:{id}` -> full SessionSnapshot
//! - `session_meta:{id}` -> SessionMetadata (for list_sessions without loading full data)

use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::Arc;

use super::{SessionMetadata, SessionSnapshot, SessionStore};

impl std::fmt::Debug for AgentFsSessionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentFsSessionStore")
            .finish_non_exhaustive()
    }
}

/// Session store backed by AgentFS KvStore.
#[derive(Clone)]
pub struct AgentFsSessionStore {
    agentfs: Arc<agentfs_sdk::AgentFS>,
}

impl AgentFsSessionStore {
    const SESSION_PREFIX: &'static str = "session:";
    const META_PREFIX: &'static str = "session_meta:";

    /// Create a session store from an AgentFS instance.
    pub fn new(agentfs: Arc<agentfs_sdk::AgentFS>) -> Self {
        Self { agentfs }
    }

    fn session_key(&self, session_id: &str) -> String {
        format!("{}{session_id}", Self::SESSION_PREFIX)
    }

    fn meta_key(&self, session_id: &str) -> String {
        format!("{}{session_id}", Self::META_PREFIX)
    }
}

#[async_trait]
impl SessionStore for AgentFsSessionStore {
    async fn save_session(&self, snapshot: &SessionSnapshot) -> Result<()> {
        let session_key = self.session_key(&snapshot.session_id);
        let meta_key = self.meta_key(&snapshot.session_id);

        self.agentfs
            .kv
            .set(&session_key, snapshot)
            .await
            .context("AgentFS session save failed")?;

        self.agentfs
            .kv
            .set(&meta_key, &snapshot.metadata)
            .await
            .context("AgentFS session metadata save failed")?;

        Ok(())
    }

    async fn load_session(&self, session_id: &str) -> Result<Option<SessionSnapshot>> {
        let session_key = self.session_key(session_id);
        self.agentfs
            .kv
            .get::<SessionSnapshot>(&session_key)
            .await
            .context("AgentFS session load failed")
    }

    async fn list_sessions(&self) -> Result<Vec<SessionMetadata>> {
        let keys = self
            .agentfs
            .kv
            .keys()
            .await
            .context("AgentFS session list keys failed")?;

        let mut metas = Vec::new();
        for key in keys {
            if key.starts_with(Self::META_PREFIX)
                && let Some(meta) = self
                    .agentfs
                    .kv
                    .get::<SessionMetadata>(&key)
                    .await
                    .ok()
                    .flatten()
            {
                metas.push(meta);
            }
        }

        Ok(metas)
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        let session_key = self.session_key(session_id);
        let meta_key = self.meta_key(session_id);

        self.agentfs
            .kv
            .delete(&session_key)
            .await
            .context("AgentFS session delete failed")?;

        self.agentfs
            .kv
            .delete(&meta_key)
            .await
            .context("AgentFS session metadata delete failed")?;

        Ok(())
    }
}
