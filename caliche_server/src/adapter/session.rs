//! Session types stored and returned by the API adapter.

use caliche::harness::{AgentEvent, AgentHarness, SharedHarness};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Session ID type
pub type SessionId = String;

/// Owner metadata stored on API-created sessions for IDOR checks.
#[derive(Debug, Clone, Default)]
pub struct SessionOwner {
    pub user_id: Option<String>,
    pub tenant_id: Option<String>,
}

/// Session state stored in the adapter
pub struct SessionState {
    /// Shared agent harness for this session (also registered in [`AgentRuntime`]).
    pub harness: SharedHarness,
    /// Event sender for broadcasting to subscribers
    pub event_sender: broadcast::Sender<AgentEvent>,
    /// Created timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// Authenticated owner when session auth is enforced.
    pub owner: SessionOwner,
    /// Caller-supplied short title. Absent when create omitted it.
    pub title: Option<String>,
}

impl SessionState {
    /// Create a new session state from an owned harness.
    pub fn new(harness: AgentHarness) -> Self {
        let event_stream = harness.event_stream();
        let event_sender = event_stream.sender();
        Self {
            harness: Arc::new(tokio::sync::RwLock::new(harness)),
            event_sender,
            created_at: chrono::Utc::now(),
            last_activity: chrono::Utc::now(),
            owner: SessionOwner::default(),
            title: None,
        }
    }

    /// Shared harness handle registered with the process runtime.
    pub fn shared_harness(&self) -> SharedHarness {
        Arc::clone(&self.harness)
    }

    /// Update last activity timestamp
    pub fn touch(&mut self) {
        self.last_activity = chrono::Utc::now();
    }

    /// Subscribe to events
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_sender.subscribe()
    }
}

/// Session information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name_slug: Option<String>,
    /// Short human title from create. Omitted when the session has none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub turn_count: usize,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct StreamingTurnResult {
    pub cancelled: bool,
}
