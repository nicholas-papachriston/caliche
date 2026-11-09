//! In-memory registry of live agent sessions for inter-agent communication.

use crate::messages::TokenUsage;
use anyhow::Result;
use dashmap::DashMap;
use parking_lot::Mutex;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};

/// Maximum number of (role, content) pairs retained per live session.
pub const MAX_HISTORY_ENTRIES: usize = 64;

/// Live status of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionLiveStatus {
    Active,
    Idle,
    Processing,
    Terminated,
}

/// Message sent to another session.
#[derive(Debug)]
pub struct SessionMessage {
    pub from_session_id: String,
    pub content: String,
    pub reply_back: bool,
    pub reply_tx: Option<oneshot::Sender<SessionResponse>>,
}

/// Response from a session (when `reply_back` is true).
#[derive(Debug, Clone)]
pub struct SessionResponse {
    pub session_id: String,
    pub content: String,
    pub usage: TokenUsage,
}

/// Single history entry kept in the bounded ring buffer on a live session.
#[derive(Debug, Clone, Serialize)]
pub struct HistoryEntry {
    pub role: String,
    pub content: String,
    /// Unix milliseconds since the epoch.
    pub recorded_at_ms: u64,
}

/// Handle to a live running agent session.
pub struct LiveSessionHandle {
    pub session_id: String,
    pub agent_id: String,
    pub model: String,
    pub status: SessionLiveStatus,
    pub created_at: Instant,
    pub message_tx: mpsc::Sender<SessionMessage>,
    pub tags: Vec<String>,
    pub depth: u32,
    /// Bounded ring buffer of recent (role, content) pairs so that
    /// `sessions_history` can return useful data without requiring a
    /// persistent session store.
    pub history: Arc<Mutex<VecDeque<HistoryEntry>>>,
}

/// Summary of a session for list endpoints.
#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub agent_id: String,
    pub model: String,
    pub status: SessionLiveStatus,
    pub tags: Vec<String>,
    pub depth: u32,
}

impl LiveSessionHandle {
    #[must_use]
    pub fn to_summary(&self) -> SessionSummary {
        SessionSummary {
            session_id: self.session_id.clone(),
            agent_id: self.agent_id.clone(),
            model: self.model.clone(),
            status: self.status,
            tags: self.tags.clone(),
            depth: self.depth,
        }
    }
}

/// Global live session registry (thread-safe, singleton per process).
pub struct LiveSessionRegistry {
    sessions: DashMap<String, LiveSessionHandle>,
    max_spawn_depth: u32,
}

static REGISTRY: std::sync::OnceLock<Arc<LiveSessionRegistry>> = std::sync::OnceLock::new();

impl LiveSessionRegistry {
    /// Create a new registry (used to initialize the global).
    #[must_use]
    pub fn new(max_spawn_depth: u32) -> Self {
        Self {
            sessions: DashMap::new(),
            max_spawn_depth,
        }
    }

    /// Get the global registry (initializes with default `max_spawn_depth` 3 if not set).
    pub fn global() -> &'static Arc<Self> {
        REGISTRY.get_or_init(|| Arc::new(Self::new(3)))
    }

    /// Register a live session.
    pub fn register(&self, handle: LiveSessionHandle) -> Result<()> {
        if self.sessions.contains_key(&handle.session_id) {
            return Err(anyhow::anyhow!(
                "Session {} already registered",
                handle.session_id
            ));
        }
        self.sessions.insert(handle.session_id.clone(), handle);
        Ok(())
    }

    /// Unregister a session by id.
    pub fn unregister(&self, session_id: &str) {
        self.sessions.remove(session_id);
    }

    /// Get a session by id (returns a ref guard).
    #[must_use]
    pub fn get(
        &self,
        session_id: &str,
    ) -> Option<dashmap::mapref::one::Ref<'_, String, LiveSessionHandle>> {
        self.sessions.get(session_id)
    }

    /// List all sessions.
    #[must_use]
    pub fn list(&self) -> Vec<SessionSummary> {
        self.sessions
            .iter()
            .map(|r| r.value().to_summary())
            .collect()
    }

    /// List sessions by tag.
    #[must_use]
    pub fn list_by_tag(&self, tag: &str) -> Vec<SessionSummary> {
        self.sessions
            .iter()
            .filter(|r| r.value().tags.iter().any(|t| t == tag))
            .map(|r| r.value().to_summary())
            .collect()
    }

    /// Send a message to a session. Caller can pass `reply_tx` in msg to receive a response.
    pub async fn send_message(&self, target_id: &str, msg: SessionMessage) -> Result<()> {
        let handle = self
            .sessions
            .get(target_id)
            .ok_or_else(|| anyhow::anyhow!("Session {target_id} not found"))?;
        let tx = handle.message_tx.clone();
        drop(handle);
        tx.send(msg)
            .await
            .map_err(|e| anyhow::anyhow!("Send to session {target_id} failed: {e}"))?;
        Ok(())
    }

    /// Max spawn depth allowed (for `SessionsSpawnTool`).
    #[must_use]
    pub const fn max_spawn_depth(&self) -> u32 {
        self.max_spawn_depth
    }

    /// Append a history entry to the session's ring buffer (no-op if the
    /// session is not currently registered).
    pub fn record_history(&self, session_id: &str, role: &str, content: &str) {
        let Some(handle) = self.sessions.get(session_id) else {
            return;
        };
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let entry = HistoryEntry {
            role: role.to_string(),
            content: content.to_string(),
            recorded_at_ms: now_ms,
        };
        let mut buf = handle.history.lock();
        if buf.len() >= MAX_HISTORY_ENTRIES {
            buf.pop_front();
        }
        buf.push_back(entry);
    }

    /// Snapshot the last `limit` history entries for a session (newest last).
    #[must_use]
    pub fn history_snapshot(&self, session_id: &str, limit: usize) -> Option<Vec<HistoryEntry>> {
        let handle = self.sessions.get(session_id)?;
        let buf = handle.history.lock();
        let start = buf.len().saturating_sub(limit);
        Some(buf.iter().skip(start).cloned().collect())
    }
}
