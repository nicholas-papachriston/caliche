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
}

/// Directory-backed store for suspended sessions.
///
/// Snapshots are read and written below an anchor directory without
/// following any link below it: the workspace for the store a workspace
/// keeps in `.caliche/suspended`, and the store directory itself for one
/// named elsewhere. A link committed into a project therefore cannot feed
/// `/resume` a snapshot from outside it, or take a save outside it.
#[derive(Debug, Clone)]
pub struct SuspendStore {
    dir: PathBuf,
    anchor: PathBuf,
}

impl SuspendStore {
    /// Store snapshots under `dir`, which is trusted as named.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        Self {
            anchor: dir.clone(),
            dir,
        }
    }

    /// The store a workspace keeps in `.caliche/suspended`.
    #[must_use]
    pub fn for_workspace(workspace: &Path) -> Self {
        Self {
            dir: workspace.join(".caliche").join("suspended"),
            anchor: workspace.to_path_buf(),
        }
    }

    /// Directory used for `{session_id}.suspend.json` files.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The canonical anchor and the snapshot path for `session_id` below it.
    fn place(&self, session_id: &str) -> Result<(PathBuf, PathBuf)> {
        let anchor = self
            .anchor
            .canonicalize()
            .with_context(|| format!("open suspend dir {}", self.dir.display()))?;
        let below = self
            .dir
            .strip_prefix(&self.anchor)
            .unwrap_or_else(|_| Path::new(""));
        Ok((
            anchor.clone(),
            suspend_path(&anchor.join(below), session_id)?,
        ))
    }

    /// Persist `snapshot` and return the file path.
    ///
    /// # Errors
    ///
    /// Returns an error when the session id is unsafe, a component below
    /// the anchor is a link, or the write fails.
    pub fn save(&self, snapshot: &SuspendedSession) -> Result<PathBuf> {
        use std::io::Write as _;
        if self.anchor == self.dir {
            std::fs::create_dir_all(&self.dir)
                .with_context(|| format!("create suspend dir {}", self.dir.display()))?;
        }
        let (anchor, path) = self.place(&snapshot.session_id)?;
        let json = snapshot.to_json()?;
        caliche_sandbox::create_file_beneath(&anchor, &path)
            .and_then(|mut file| file.write_all(json.as_bytes()))
            .with_context(|| format!("write suspend snapshot {}", path.display()))?;
        Ok(path)
    }

    /// Load a snapshot by session id.
    ///
    /// # Errors
    ///
    /// Returns an error when the snapshot is missing, invalid, or reached
    /// through a link.
    pub fn load(&self, session_id: &str) -> Result<SuspendedSession> {
        use std::io::Read as _;
        let (anchor, path) = self.place(session_id)?;
        let mut raw = String::new();
        caliche_sandbox::open_file_beneath(&anchor, &path)
            .and_then(|mut file| file.read_to_string(&mut raw))
            .with_context(|| format!("read suspend snapshot {}", path.display()))?;
        SuspendedSession::from_json(&raw)
    }

    /// True when a snapshot exists for `session_id`.
    #[must_use]
    pub fn contains(&self, session_id: &str) -> bool {
        self.place(session_id)
            .is_ok_and(|(anchor, path)| caliche_sandbox::open_file_beneath(&anchor, &path).is_ok())
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

    /// A workspace whose `.caliche` is a link to an outside directory, or
    /// whose snapshot is a link to an outside file: `/resume` loads nothing
    /// from outside, and a save writes nothing there.
    #[cfg(unix)]
    #[test]
    fn a_linked_workspace_snapshot_is_not_loaded_or_written() {
        for layout in ["linked dir", "linked file"] {
            let workspace = tempfile::TempDir::new().expect("workspace");
            let root = workspace
                .path()
                .canonicalize()
                .expect("canonical workspace");
            let outside = tempfile::TempDir::new().expect("outside");
            let outside_dir = outside.path().join("suspended");
            std::fs::create_dir(&outside_dir).expect("outside suspended");
            let mut planted = SuspendedSession::new("sess-1");
            planted.turn_count = 7;
            let planted_json = planted.to_json().expect("json");
            std::fs::write(outside_dir.join("sess-1.suspend.json"), &planted_json)
                .expect("planted");
            if layout == "linked dir" {
                std::os::unix::fs::symlink(outside.path(), root.join(".caliche"))
                    .expect("dir link");
            } else {
                std::fs::create_dir_all(root.join(".caliche/suspended")).expect("suspended");
                std::os::unix::fs::symlink(
                    outside_dir.join("sess-1.suspend.json"),
                    root.join(".caliche/suspended/sess-1.suspend.json"),
                )
                .expect("file link");
            }
            let store = SuspendStore::for_workspace(&root);

            let loaded = store.load("sess-1");
            let contained = store.contains("sess-1");
            let saved = store.save(&SuspendedSession::new("sess-1"));

            assert!(loaded.is_err(), "{layout}: loaded {loaded:?}");
            assert!(!contained, "{layout}");
            assert!(saved.is_err(), "{layout}: saved {saved:?}");
            assert_eq!(
                std::fs::read_to_string(outside_dir.join("sess-1.suspend.json")).expect("planted"),
                planted_json,
                "{layout}"
            );
        }
    }

    #[test]
    fn a_workspace_snapshot_round_trips() {
        let workspace = tempfile::TempDir::new().expect("workspace");
        let store = SuspendStore::for_workspace(workspace.path());
        let mut snap = SuspendedSession::new("sess-2");
        snap.turn_count = 3;

        let path = store.save(&snap).expect("save");

        assert_eq!(
            path,
            workspace
                .path()
                .canonicalize()
                .expect("canonical")
                .join(".caliche/suspended/sess-2.suspend.json")
        );
        assert!(store.contains("sess-2"));
        assert_eq!(store.load("sess-2").expect("load").turn_count, 3);
    }
}
