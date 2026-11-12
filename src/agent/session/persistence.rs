//! Unified session persistence: snapshot types and store trait.

use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::codec::{decode_session_snapshot_v1, encode_session_snapshot_v1};
use crate::agent::memory::MemorySnapshot;
use crate::harness::SessionDiffState;
use crate::messages::Message;

/// Full snapshot of a session for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// Session identifier.
    pub session_id: String,
    /// Conversation history.
    pub conversation_history: Vec<Message>,
    /// Memory strategy state (if any).
    pub memory_snapshot: Option<MemorySnapshot>,
    /// Knowledge store state (if any; used by `EnhancedMemory`).
    pub knowledge_snapshot: Option<MemorySnapshot>,
    /// Plan document state (populated during Plan mode turns).
    #[serde(default)]
    pub plan_document: Option<crate::agent::modes::PlanDocument>,
    /// Durable per-message diff state for undo/redo.
    #[serde(default)]
    pub diff_state: SessionDiffState,
    /// Shared state as JSON.
    pub shared_state: Option<serde_json::Value>,
    /// Snapshot metadata.
    pub metadata: SessionMetadata,
}

/// Metadata for a persisted session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// When the session was first created (or first saved).
    pub created_at: DateTime<Utc>,
    /// When this snapshot was saved.
    pub saved_at: DateTime<Utc>,
    /// Human-readable memory strategy type (e.g. "buffer", "enhanced").
    pub memory_type: String,
    /// Total messages in conversation history.
    pub total_messages: usize,
    /// Estimated total tokens.
    pub total_tokens_estimated: usize,
}

/// Store for saving and loading session snapshots.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Save a session snapshot.
    async fn save_session(&self, snapshot: &SessionSnapshot) -> Result<()>;

    /// Load a session snapshot by id.
    async fn load_session(&self, session_id: &str) -> Result<Option<SessionSnapshot>>;

    /// List metadata for all stored sessions.
    async fn list_sessions(&self) -> Result<Vec<SessionMetadata>>;

    /// Delete a session by id.
    async fn delete_session(&self, session_id: &str) -> Result<()>;
}

/// File-based session store using binary rkyv v1 snapshots (`*.cls1`) under a directory.
#[derive(Debug, Clone)]
pub struct BinaryFileSessionStore {
    dir: std::path::PathBuf,
}

impl BinaryFileSessionStore {
    /// Create a binary session store using the given directory.
    pub fn new(dir: impl AsRef<std::path::Path>) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
        }
    }

    fn path_for(&self, session_id: &str) -> std::path::PathBuf {
        let sanitized = session_id.replace(UNSAFE_ID_CHARS, "_");
        self.dir.join(format!("{sanitized}.cls1"))
    }

    /// Where a session was stored before [`UNSAFE_ID_CHARS`] covered every
    /// character Windows reserves. There a `:` made an alternate data
    /// stream, which loads but is never listed. Read and delete still find
    /// such a session.
    fn legacy_path_for(&self, session_id: &str) -> Option<std::path::PathBuf> {
        let sanitized = session_id.replace(['/', '\\', '.'], "_");
        let legacy = self.dir.join(format!("{sanitized}.cls1"));
        (cfg!(windows) && legacy != self.path_for(session_id)).then_some(legacy)
    }

    /// The stored file for `session_id`: the current name, else the legacy
    /// one.
    fn existing_path_for(&self, session_id: &str) -> Option<std::path::PathBuf> {
        std::iter::once(self.path_for(session_id))
            .chain(self.legacy_path_for(session_id))
            .find(|path| path.is_file())
    }
}

/// Characters replaced by `_` in a session file name: path separators, `.`,
/// and on Windows every other character it reserves in a file name (`:`
/// would name an alternate data stream).
#[cfg(windows)]
const UNSAFE_ID_CHARS: &[char] = &['/', '\\', '.', ':', '*', '?', '"', '<', '>', '|'];
#[cfg(not(windows))]
const UNSAFE_ID_CHARS: &[char] = &['/', '\\', '.'];

#[async_trait]
impl SessionStore for BinaryFileSessionStore {
    async fn save_session(&self, snapshot: &SessionSnapshot) -> Result<()> {
        tokio::fs::create_dir_all(&self.dir).await?;
        let path = self.path_for(&snapshot.session_id);
        let bytes = encode_session_snapshot_v1(snapshot)?;
        tokio::fs::write(path, bytes).await?;
        Ok(())
    }

    async fn load_session(&self, session_id: &str) -> Result<Option<SessionSnapshot>> {
        let Some(path) = self.existing_path_for(session_id) else {
            return Ok(None);
        };
        let data = tokio::fs::read(&path).await?;
        let snapshot = decode_session_snapshot_v1(&data)?;
        Ok(Some(snapshot))
    }

    async fn list_sessions(&self) -> Result<Vec<SessionMetadata>> {
        if !self.dir.is_dir() {
            return Ok(vec![]);
        }
        let mut out = Vec::new();
        let mut read_dir = tokio::fs::read_dir(&self.dir).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "cls1")
                && let Ok(data) = tokio::fs::read(&path).await
                && let Ok(snap) = decode_session_snapshot_v1(&data)
            {
                out.push(snap.metadata);
            }
        }
        Ok(out)
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        if let Some(path) = self.existing_path_for(session_id) {
            tokio::fs::remove_file(path).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::Message;
    use chrono::{DateTime, TimeZone, Utc};

    fn fixed_time(year: i32, month: u32, day: u32) -> Result<DateTime<Utc>> {
        Utc.with_ymd_and_hms(year, month, day, 0, 0, 0)
            .single()
            .ok_or_else(|| anyhow::anyhow!("invalid fixed test time"))
    }

    fn minimal_snapshot(id: &str) -> Result<SessionSnapshot> {
        Ok(SessionSnapshot {
            session_id: id.to_string(),
            conversation_history: vec![Message::user("ping")],
            memory_snapshot: None,
            knowledge_snapshot: None,
            plan_document: None,
            diff_state: SessionDiffState::default(),
            shared_state: None,
            metadata: SessionMetadata {
                created_at: fixed_time(2024, 1, 1)?,
                saved_at: fixed_time(2024, 1, 2)?,
                memory_type: "none".to_string(),
                total_messages: 1,
                total_tokens_estimated: 1,
            },
        })
    }

    /// A session stored under the old name, where `:` made an alternate
    /// data stream, still loads and deletes.
    #[cfg(windows)]
    #[tokio::test]
    async fn a_session_in_a_legacy_stream_still_loads_and_deletes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        let snap = minimal_snapshot("id:legacy")?;
        let legacy = dir.path().join("id:legacy.cls1");
        std::fs::write(&legacy, encode_session_snapshot_v1(&snap)?)?;

        let loaded = store
            .load_session("id:legacy")
            .await?
            .ok_or_else(|| anyhow::anyhow!("expected the legacy session"))?;
        assert_eq!(loaded.session_id, "id:legacy");
        assert_eq!(loaded.conversation_history[0].as_text(), Some("ping"));

        store.delete_session("id:legacy").await?;
        assert!(store.load_session("id:legacy").await?.is_none());
        Ok(())
    }

    #[tokio::test]
    async fn binary_file_session_store_roundtrip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let store = BinaryFileSessionStore::new(dir.path());
        let snap = minimal_snapshot("id/with:unsafe")?;
        store.save_session(&snap).await?;
        let loaded = store
            .load_session("id/with:unsafe")
            .await?
            .ok_or_else(|| anyhow::anyhow!("expected stored session"))?;
        assert_eq!(loaded.session_id, snap.session_id);
        assert_eq!(loaded.conversation_history.len(), 1);
        assert_eq!(loaded.conversation_history[0].as_text(), Some("ping"));
        assert!(loaded.diff_state.active.is_empty());
        let listed = store.list_sessions().await?;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].total_messages, 1);
        store.delete_session("id/with:unsafe").await?;
        assert!(store.load_session("id/with:unsafe").await?.is_none());
        Ok(())
    }
}
