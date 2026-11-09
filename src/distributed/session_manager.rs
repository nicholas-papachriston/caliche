use crate::data::persistence::rkyv_envelope::{
    SESSION_AFFINITY_MAGIC_V1, prepend_magic, strip_magic_prefix,
};
use anyhow::{Context, Result};
use deadpool_redis::redis::AsyncCommands;
use deadpool_redis::{Config, Pool, Runtime};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionAffinity {
    pub session_id: String,
    pub node_id: String,
    pub fallback_nodes: Vec<String>,
}

#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
struct SessionAffinityStoredV1 {
    session_id: String,
    node_id: String,
    fallback_nodes: Vec<String>,
}

impl From<&SessionAffinity> for SessionAffinityStoredV1 {
    fn from(a: &SessionAffinity) -> Self {
        Self {
            session_id: a.session_id.clone(),
            node_id: a.node_id.clone(),
            fallback_nodes: a.fallback_nodes.clone(),
        }
    }
}

impl From<SessionAffinityStoredV1> for SessionAffinity {
    fn from(s: SessionAffinityStoredV1) -> Self {
        Self {
            session_id: s.session_id,
            node_id: s.node_id,
            fallback_nodes: s.fallback_nodes,
        }
    }
}

fn encode_session_affinity(a: &SessionAffinity) -> Result<Vec<u8>> {
    let stored = SessionAffinityStoredV1::from(a);
    let body = rkyv::to_bytes::<rkyv::rancor::Error>(&stored)
        .map(|b| b.into_vec())
        .map_err(|e| anyhow::anyhow!("rkyv serialize SessionAffinity: {e}"))?;
    Ok(prepend_magic(SESSION_AFFINITY_MAGIC_V1, &body))
}

fn decode_session_affinity(bytes: &[u8]) -> Result<SessionAffinity> {
    let body = strip_magic_prefix(SESSION_AFFINITY_MAGIC_V1, bytes, "SessionAffinity")?;
    let stored: SessionAffinityStoredV1 = rkyv::from_bytes::<_, rkyv::rancor::Error>(body)
        .map_err(|e| anyhow::anyhow!("rkyv deserialize SessionAffinity: {e}"))?;
    Ok(SessionAffinity::from(stored))
}

/// Session manager for distributed session affinity
///
/// Compatible with both Redis and Valkey (Redis protocol-compatible).
/// Stores session-to-node mappings with TTL support.
pub struct SessionManager {
    pool: Pool,
    ttl_secs: u64,
}

impl SessionManager {
    /// Create a new session manager
    ///
    /// # Arguments
    /// * `coordinator_url` - Redis or Valkey connection URL (e.g., "redis://localhost:6379" or "redis://localhost:6380")
    /// * `ttl_secs` - TTL in seconds for session affinity entries
    ///
    /// # Compatibility
    /// Works with both Redis and Valkey (Redis protocol-compatible).
    pub fn new(coordinator_url: &str, ttl_secs: u64) -> Result<Self> {
        let cfg = Config::from_url(coordinator_url);
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1))
            .context("Failed to create Redis/Valkey pool for SessionManager")?;

        Ok(Self { pool, ttl_secs })
    }

    pub async fn set_affinity(
        &self,
        session_id: String,
        node_id: String,
        fallback_nodes: Vec<String>,
    ) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let key = format!("distributed:session:{}", session_id);
        let affinity = SessionAffinity {
            session_id: session_id.clone(),
            node_id,
            fallback_nodes,
        };
        let value =
            encode_session_affinity(&affinity).context("Failed to serialize SessionAffinity")?;

        conn.set_ex::<_, _, ()>(&key, value, self.ttl_secs).await?;

        tracing::debug!(session_id = %session_id, "Set session affinity");
        Ok(())
    }

    pub async fn get_affinity(&self, session_id: &str) -> Result<Option<SessionAffinity>> {
        let mut conn = self.pool.get().await?;
        let key = format!("distributed:session:{}", session_id);

        if let Some(data) = conn.get::<_, Option<Vec<u8>>>(&key).await? {
            let affinity =
                decode_session_affinity(&data).context("Failed to deserialize SessionAffinity")?;
            Ok(Some(affinity))
        } else {
            Ok(None)
        }
    }

    pub async fn remove_affinity(&self, session_id: &str) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let key = format!("distributed:session:{}", session_id);
        conn.del::<_, ()>(&key).await?;

        tracing::debug!(session_id = %session_id, "Removed session affinity");
        Ok(())
    }

    pub async fn reassign_session(
        &self,
        session_id: &str,
        new_node_id: impl Into<String>,
    ) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let key = format!("distributed:session:{}", session_id);

        let Some(data) = conn.get::<_, Option<Vec<u8>>>(&key).await? else {
            return Ok(());
        };
        let mut affinity =
            decode_session_affinity(&data).context("Failed to deserialize session affinity")?;
        affinity.node_id = new_node_id.into();
        let value = encode_session_affinity(&affinity)?;
        conn.set_ex::<_, _, ()>(&key, value, self.ttl_secs).await?;

        tracing::debug!(
            session_id = %session_id,
            new_node_id = %affinity.node_id,
            "Reassigned session to new node"
        );
        Ok(())
    }

    pub async fn refresh_ttl(&self, session_id: &str) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let key = format!("distributed:session:{}", session_id);
        conn.expire::<_, ()>(&key, self.ttl_secs as i64).await?;
        Ok(())
    }
}

#[cfg(test)]
mod affinity_codec_tests {
    use super::*;

    #[test]
    fn session_affinity_roundtrip() {
        let a = SessionAffinity {
            session_id: "sess-1".into(),
            node_id: "node-a".into(),
            fallback_nodes: vec!["node-b".into(), "node-c".into()],
        };
        let bytes = encode_session_affinity(&a).expect("encode");
        assert!(
            bytes.starts_with(crate::data::persistence::rkyv_envelope::SESSION_AFFINITY_MAGIC_V1)
        );
        let got = decode_session_affinity(&bytes).expect("decode");
        assert_eq!(got.session_id, a.session_id);
        assert_eq!(got.node_id, a.node_id);
        assert_eq!(got.fallback_nodes, a.fallback_nodes);
    }

    #[test]
    fn decode_affinity_rejects_bad_magic() {
        let err = decode_session_affinity(b"bad!").expect_err("bad");
        assert!(
            err.to_string().contains("SessionAffinity") || err.to_string().contains("magic"),
            "{err}"
        );
    }
}
