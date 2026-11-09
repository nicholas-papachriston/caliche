//! Durable suspend / resume snapshots for harness sessions.
//!
//! In-process checkpoints stay in [`crate::harness::state`]. This module
//! persists enough conversation and actor metadata to survive a process
//! restart and continue at the same logical point.

use crate::harness::persistence::{ConversationExport, ExportedCheckpoint, ExportedMessage};
use crate::harness::state::{Checkpoint, CheckpointId};
use crate::messages::{Message, TokenUsage};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Logical point in a session that resume should restore.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SessionPhase {
    #[default]
    Idle,
    InTurn,
    Complete,
}

/// On-disk snapshot written by `/suspend` (or [`crate::harness::AgentRuntime::suspend`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspendedSession {
    /// Snapshot format version.
    pub version: String,
    /// Session id restored into the new process.
    pub session_id: String,
    /// When the snapshot was written.
    pub suspended_at: chrono::DateTime<chrono::Utc>,
    /// Logical phase to restore (not "suspended").
    pub phase: SessionPhase,
    /// Completed user turns.
    pub turn_count: usize,
    /// Last harness checkpoint id, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_checkpoint_id: Option<String>,
    /// Workspace root at suspend time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    /// Conversation messages required to continue.
    pub messages: Vec<ExportedMessage>,
    /// Accumulated token usage.
    pub total_usage: TokenUsage,
    /// Checkpoint metadata (identity / label / turn).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoints: Vec<ExportedCheckpoint>,
    /// Session inbox (seen ids) so redelivery after resume is a no-op.
    #[serde(
        default,
        skip_serializing_if = "crate::harness::inbox::SessionInbox::is_empty"
    )]
    pub inbox: crate::harness::inbox::SessionInbox,
    /// Compaction reports captured before suspend.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compaction_reports: Vec<crate::messages::CompactionReport>,
    /// Hard-stop latch.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub hard_stopped: bool,
    /// When-idle stop still pending.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending_when_idle: bool,
    /// Inbox settings snapshot.
    #[serde(default, skip_serializing_if = "inbox_settings_is_default")]
    pub inbox_settings: crate::harness::inbox::InboxSettings,
}

fn inbox_settings_is_default(settings: &crate::harness::inbox::InboxSettings) -> bool {
    settings.reasoning_effort.is_none()
}

impl SuspendedSession {
    /// Snapshot envelope for `session_id`.
    #[must_use]
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            version: "1".to_string(),
            session_id: session_id.into(),
            suspended_at: chrono::Utc::now(),
            phase: SessionPhase::Idle,
            turn_count: 0,
            last_checkpoint_id: None,
            workspace: None,
            messages: Vec::new(),
            total_usage: TokenUsage::default(),
            checkpoints: Vec::new(),
            inbox: crate::harness::inbox::SessionInbox::new(),
            compaction_reports: Vec::new(),
            hard_stopped: false,
            pending_when_idle: false,
            inbox_settings: crate::harness::inbox::InboxSettings::default(),
        }
    }

    /// Restore conversation messages.
    #[must_use]
    pub fn to_messages(&self) -> Vec<Message> {
        ConversationExport {
            version: self.version.clone(),
            session_id: self.session_id.clone(),
            exported_at: self.suspended_at,
            workspace: self.workspace.clone(),
            messages: self.messages.clone(),
            total_usage: self.total_usage,
            checkpoints: self.checkpoints.clone(),
            modified_files: Vec::new(),
            metadata: None,
        }
        .to_messages()
    }

    /// Checkpoint metadata restored into [`crate::harness::StateManager`].
    #[must_use]
    pub fn checkpoints_as_state(&self) -> Vec<Checkpoint> {
        self.checkpoints
            .iter()
            .map(|exported| Checkpoint {
                id: CheckpointId::new(exported.id.clone()),
                label: exported.label.clone(),
                timestamp: exported.timestamp,
                turn: exported.turn,
                file_versions: HashMap::new(),
            })
            .collect()
    }

    /// Serialize to pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns an error when serialization fails.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("serialize suspended session")
    }

    /// Parse a JSON snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when the JSON is invalid.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("parse suspended session")
    }

    /// Write `{session_id}.suspend.json` under `dir`.
    ///
    /// # Errors
    ///
    /// Returns an error when the session id is unsafe or the write fails.
    pub fn save_to_dir(&self, dir: &Path) -> Result<PathBuf> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create suspend dir {}", dir.display()))?;
        let path = suspend_path(dir, &self.session_id)?;
        std::fs::write(&path, self.to_json()?)
            .with_context(|| format!("write suspend snapshot {}", path.display()))?;
        Ok(path)
    }

    /// Load `{session_id}.suspend.json` from `dir`.
    ///
    /// # Errors
    ///
    /// Returns an error when the file is missing or invalid.
    pub fn load_from_dir(dir: &Path, session_id: &str) -> Result<Self> {
        let path = suspend_path(dir, session_id)?;
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read suspend snapshot {}", path.display()))?;
        Self::from_json(&raw)
    }
}

/// Directory-backed store for suspended sessions.
#[derive(Debug, Clone)]
pub struct SuspendStore {
    dir: PathBuf,
}

impl SuspendStore {
    /// Store snapshots under `dir`.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Directory used for `{session_id}.suspend.json` files.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Persist `snapshot` and return the file path.
    ///
    /// # Errors
    ///
    /// Returns an error when the write fails.
    pub fn save(&self, snapshot: &SuspendedSession) -> Result<PathBuf> {
        snapshot.save_to_dir(&self.dir)
    }

    /// Load a snapshot by session id.
    ///
    /// # Errors
    ///
    /// Returns an error when the snapshot is missing or invalid.
    pub fn load(&self, session_id: &str) -> Result<SuspendedSession> {
        SuspendedSession::load_from_dir(&self.dir, session_id)
    }

    /// True when a snapshot exists for `session_id`.
    #[must_use]
    pub fn contains(&self, session_id: &str) -> bool {
        suspend_path(&self.dir, session_id)
            .map(|path| path.exists())
            .unwrap_or(false)
    }
}

fn suspend_path(dir: &Path, session_id: &str) -> Result<PathBuf> {
    let sanitized = sanitize_session_id(session_id)?;
    Ok(dir.join(format!("{sanitized}.suspend.json")))
}

fn sanitize_session_id(id: &str) -> Result<String> {
    if id.is_empty()
        || id.contains('/')
        || id.contains('\\')
        || id.contains("..")
        || !id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_')
    {
        bail!("invalid suspend session id '{id}'");
    }
    Ok(id.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::Message;

    #[test]
    fn snapshot_round_trip_preserves_history_and_phase() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let store = SuspendStore::new(dir.path());
        let mut snap = SuspendedSession::new("sess-1");
        snap.phase = SessionPhase::InTurn;
        snap.turn_count = 2;
        snap.last_checkpoint_id = Some("cp-9".to_string());
        snap.messages = vec![ExportedMessage::from(&Message::user("continue"))];
        store.save(&snap).expect("save");
        let loaded = store.load("sess-1").expect("load");
        assert_eq!(loaded.session_id, "sess-1");
        assert_eq!(loaded.phase, SessionPhase::InTurn);
        assert_eq!(loaded.turn_count, 2);
        assert_eq!(loaded.last_checkpoint_id.as_deref(), Some("cp-9"));
        assert_eq!(loaded.to_messages().len(), 1);
    }

    #[test]
    fn rejects_path_traversal_session_id() {
        let err = sanitize_session_id("../etc/passwd").expect_err("unsafe id");
        assert!(err.to_string().contains("invalid suspend session id"));
    }
}
