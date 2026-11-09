//! Agent session management for conversation state and forking.

#[cfg(feature = "agentfs")]
pub mod agentfs_store;
mod codec;
pub mod jsonl_tree;
#[cfg(feature = "session-tools")]
pub mod live_registry;
mod persistence;
pub mod share;
pub mod share_store;

#[cfg(feature = "agentfs")]
pub use agentfs_store::AgentFsSessionStore;
pub use codec::{
    SESSION_SNAPSHOT_FILE_MAGIC, decode_session_snapshot_v1, encode_session_snapshot_v1,
    load_session_snapshot_from_path,
};
pub use jsonl_tree::{
    JSONL_SESSION_VERSION, JsonlSessionEntry, JsonlSessionHeader, JsonlSessionStore,
    JsonlSessionTree, new_entry_id, parse_jsonl_session,
};
#[cfg(feature = "session-tools")]
pub use live_registry::{
    LiveSessionHandle, LiveSessionRegistry, SessionLiveStatus, SessionMessage, SessionResponse,
    SessionSummary,
};
pub use persistence::{BinaryFileSessionStore, SessionMetadata, SessionSnapshot, SessionStore};
pub use share::{
    RedactedTranscript, RedactionResult, ShareConfig, ShareMode, SharePolicy, ShareVisibility,
    TranscriptRedactor,
};
pub use share_store::{
    DirectoryShareStore, MemoryShareStore, SharePayload, ShareStore, default_share_store,
    escape_html, is_safe_share_slug, render_share_html,
};

use crate::agent::memory::{MemorySnapshot, MemoryStrategy};
use crate::agent::modes::PlanDocument;
use crate::harness::SessionDiffState;
use crate::messages::Message;
use crate::orchestration::state::SharedState;
use anyhow::Result;
use chrono::Utc;
use std::sync::{Arc, RwLock};

/// Agent session for managing conversation state and enabling session forking
#[derive(Clone)]
pub struct AgentSession {
    /// Unique session identifier
    pub id: String,
    /// Conversation history (thread-safe)
    pub conversation_history: Arc<RwLock<Vec<Message>>>,
    /// Shared state for coordination between sessions
    pub shared_state: SharedState,
    /// Memory strategy for this session (forked/merged with session)
    pub memory: Option<Arc<tokio::sync::RwLock<Box<dyn MemoryStrategy>>>>,
    /// Evolving plan document captured during Plan mode turns.
    ///
    /// Guarded by a `tokio::sync::RwLock` because the agent service updates it
    /// asynchronously from the post-turn plan-extraction pass while
    /// [`AgentSession::save`] reads it concurrently.
    pub plan_document: Arc<tokio::sync::RwLock<Option<PlanDocument>>>,
    /// Durable per-message diff state used to hydrate harness undo/redo stacks.
    diff_state: Arc<RwLock<SessionDiffState>>,
}

impl AgentSession {
    /// Create a new agent session
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            conversation_history: Arc::new(RwLock::new(Vec::new())),
            shared_state: SharedState::new(),
            memory: None,
            plan_document: Arc::new(tokio::sync::RwLock::new(None)),
            diff_state: Arc::new(RwLock::new(SessionDiffState::default())),
        }
    }

    /// Create a new session with existing conversation history
    pub fn with_history(id: impl Into<String>, history: Vec<Message>) -> Self {
        Self {
            id: id.into(),
            conversation_history: Arc::new(RwLock::new(history)),
            shared_state: SharedState::new(),
            memory: None,
            plan_document: Arc::new(tokio::sync::RwLock::new(None)),
            diff_state: Arc::new(RwLock::new(SessionDiffState::default())),
        }
    }

    /// Create a new session with shared state
    pub fn with_shared_state(id: impl Into<String>, shared_state: SharedState) -> Self {
        Self {
            id: id.into(),
            conversation_history: Arc::new(RwLock::new(Vec::new())),
            shared_state,
            memory: None,
            plan_document: Arc::new(tokio::sync::RwLock::new(None)),
            diff_state: Arc::new(RwLock::new(SessionDiffState::default())),
        }
    }

    /// Fork this session into a new isolated session (async to fork memory).
    ///
    /// The forked session has:
    /// - A new unique ID
    /// - A copy of the current conversation history
    /// - A forked memory strategy if present
    /// - A reference to the same shared state (for coordination)
    pub async fn fork(&self, new_id: impl Into<String>) -> anyhow::Result<Self> {
        let history = self
            .conversation_history
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to read conversation history for fork: {e}"))?
            .clone();

        let forked_memory = match &self.memory {
            Some(mem) => {
                let strategy = mem.read().await.fork().await?;
                Some(Arc::new(tokio::sync::RwLock::new(strategy)))
            }
            None => None,
        };

        let plan_document = self.plan_document.read().await.clone();
        let diff_state = self.diff_state_snapshot()?;

        Ok(Self {
            id: new_id.into(),
            conversation_history: Arc::new(RwLock::new(history)),
            shared_state: self.shared_state.clone(),
            memory: forked_memory,
            plan_document: Arc::new(tokio::sync::RwLock::new(plan_document)),
            diff_state: Arc::new(RwLock::new(diff_state)),
        })
    }

    /// Merge another session's conversation history and memory into this session.
    pub async fn merge(&mut self, other: &Self) -> Result<()> {
        let other_history_vec: Vec<Message> = {
            let other_history = other
                .conversation_history
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to read other session history: {e}"))?;
            other_history.iter().cloned().collect()
        };

        {
            let mut self_history = self
                .conversation_history
                .write()
                .map_err(|e| anyhow::anyhow!("Failed to write session history: {e}"))?;
            self_history.extend(other_history_vec);
        }

        if let (Some(self_mem), Some(other_mem)) = (&self.memory, &other.memory) {
            let other_guard = other_mem.read().await;
            let mut self_guard = self_mem.write().await;
            self_guard
                .merge_from(other_guard.as_ref())
                .await
                .map_err(|e| anyhow::anyhow!("Failed to merge memory: {e}"))?;
        }

        Ok(())
    }

    /// Get the current conversation history
    pub fn get_history(&self) -> Result<Vec<Message>> {
        self.conversation_history
            .read()
            .map(|h| h.clone())
            .map_err(|e| anyhow::anyhow!("Failed to read conversation history: {e}"))
    }

    /// Add a message to the conversation history
    pub fn add_message(&self, message: Message) -> Result<()> {
        self.conversation_history
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to write conversation history: {e}"))?
            .push(message);
        Ok(())
    }

    /// Clear the conversation history
    pub fn clear_history(&self) -> Result<()> {
        self.conversation_history
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to write conversation history: {e}"))?
            .clear();
        Ok(())
    }

    /// Snapshot durable per-message diff state.
    pub fn diff_state_snapshot(&self) -> Result<SessionDiffState> {
        self.diff_state
            .read()
            .map(|state| state.clone())
            .map_err(|e| anyhow::anyhow!("Failed to read session diff state: {e}"))
    }

    /// Replace durable per-message diff state.
    pub fn replace_diff_state(&self, state: SessionDiffState) -> Result<()> {
        *self
            .diff_state
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to write session diff state: {e}"))? = state;
        Ok(())
    }

    /// Save this session to a store.
    pub async fn save(&self, store: &dyn SessionStore) -> Result<()> {
        let history = self
            .conversation_history
            .read()
            .map_err(|e| anyhow::anyhow!("conversation_history: {e}"))?
            .clone();
        let (memory_snapshot, total_tokens_estimated) = match &self.memory {
            Some(mem) => {
                let guard = mem.read().await;
                // Prefer the strategy's authoritative state (summaries, entities,
                // vector metadata, composite children) over a bare messages slice
                // so `from_snapshot` can faithfully restore it.
                let snapshot = guard.save_state().await?;
                let tokens = guard.stats().total_tokens_estimated;
                (Some(snapshot), tokens)
            }
            None => (None, 0_usize),
        };
        let shared_state = Some(self.shared_state.get_all().await);
        let plan_document = self.plan_document.read().await.clone();
        let diff_state = self.diff_state_snapshot()?;
        let total_messages = history.len();
        let memory_type = memory_snapshot
            .as_ref()
            .map_or("none", |_| "memory")
            .to_string();
        let now = Utc::now();
        let snapshot = SessionSnapshot {
            session_id: self.id.clone(),
            conversation_history: history,
            memory_snapshot,
            knowledge_snapshot: None,
            plan_document,
            diff_state,
            shared_state,
            metadata: SessionMetadata {
                created_at: now,
                saved_at: now,
                memory_type,
                total_messages,
                total_tokens_estimated,
            },
        };
        store.save_session(&snapshot).await
    }

    /// Load a session from a store. Returns the snapshot; use `from_snapshot` to build an `AgentSession`.
    pub async fn load(
        store: &dyn SessionStore,
        session_id: &str,
    ) -> Result<Option<SessionSnapshot>> {
        store.load_session(session_id).await
    }

    /// Build an agent session from a snapshot (history and shared state; memory is not restored).
    ///
    /// Memory cannot be reconstructed from a snapshot alone because strategies
    /// depend on runtime dependencies (token counters, embedding providers,
    /// vector backends). Use [`AgentSession::attach_memory`] to wire an
    /// already-built memory strategy and replay the persisted snapshot into it.
    pub fn from_snapshot(snapshot: SessionSnapshot) -> Self {
        let shared_state = snapshot
            .shared_state
            .map(SharedState::from_value)
            .unwrap_or_default();
        Self {
            id: snapshot.session_id,
            conversation_history: Arc::new(RwLock::new(snapshot.conversation_history)),
            shared_state,
            memory: None,
            plan_document: Arc::new(tokio::sync::RwLock::new(snapshot.plan_document)),
            diff_state: Arc::new(RwLock::new(snapshot.diff_state)),
        }
    }

    /// Attach a pre-built memory strategy to this session and, if provided,
    /// replay a persisted snapshot into it so summaries, entities, and other
    /// strategy-specific state are restored.
    pub async fn attach_memory(
        &mut self,
        memory: Box<dyn MemoryStrategy>,
        snapshot: Option<MemorySnapshot>,
    ) -> Result<()> {
        let memory = Arc::new(tokio::sync::RwLock::new(memory));
        if let Some(snap) = snapshot {
            memory.write().await.load_state(snap).await?;
        }
        self.memory = Some(memory);
        Ok(())
    }
}
