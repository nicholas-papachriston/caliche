//! Per-message diff tracking for session-level undo/redo.
//!
//! Groups file mutations by user message so that `/undo` reverts the last
//! message's worth of changes and `/redo` restores the most recently reverted.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// High-level change classification for a file snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotChangeKind {
    Created,
    Modified,
    Deleted,
}

/// Snapshot of a single file before/after a change (one user message).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSnapshot {
    /// File path
    pub path: PathBuf,
    /// Content before the change (None if file did not exist)
    pub before: Option<Vec<u8>>,
    /// Content after the change (None if file was deleted)
    pub after: Option<Vec<u8>>,
}

impl FileSnapshot {
    /// Classify this snapshot for display and API summaries.
    pub fn change_kind(&self) -> SnapshotChangeKind {
        match (self.before.as_ref(), self.after.as_ref()) {
            (None, Some(_)) => SnapshotChangeKind::Created,
            (Some(_), None) => SnapshotChangeKind::Deleted,
            _ => SnapshotChangeKind::Modified,
        }
    }
}

/// All file changes produced by one user message/turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDiff {
    /// Identifier for this message/turn
    pub message_id: Uuid,
    /// File snapshots for this message
    pub snapshots: Vec<FileSnapshot>,
}

/// Durable diff tracker state stored with session snapshots.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionDiffState {
    /// Active, undoable change sets.
    #[serde(default)]
    pub active: Vec<MessageDiff>,
    /// Reverted change sets available for redo.
    #[serde(default)]
    pub reverted: Vec<MessageDiff>,
}

/// Lightweight summary for one file snapshot, suitable for UI/API use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSnapshotSummary {
    /// File path
    pub path: PathBuf,
    /// High-level change kind
    pub change: SnapshotChangeKind,
    /// Size of the file before the change, if known
    pub before_len: Option<usize>,
    /// Size of the file after the change, if known
    pub after_len: Option<usize>,
}

impl From<&FileSnapshot> for FileSnapshotSummary {
    fn from(snapshot: &FileSnapshot) -> Self {
        Self {
            path: snapshot.path.clone(),
            change: snapshot.change_kind(),
            before_len: snapshot.before.as_ref().map(Vec::len),
            after_len: snapshot.after.as_ref().map(Vec::len),
        }
    }
}

/// Lightweight summary for one recorded message-level change set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageDiffSummary {
    /// Identifier for this message/turn
    pub message_id: Uuid,
    /// Number of files changed in this message
    pub changed_files: usize,
    /// Per-file summaries
    pub snapshots: Vec<FileSnapshotSummary>,
}

impl From<&MessageDiff> for MessageDiffSummary {
    fn from(diff: &MessageDiff) -> Self {
        Self {
            message_id: diff.message_id,
            changed_files: diff.snapshots.len(),
            snapshots: diff
                .snapshots
                .iter()
                .map(FileSnapshotSummary::from)
                .collect(),
        }
    }
}

/// Summary of the current session diff state for UI/API use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDiffSummary {
    /// Active, undoable change sets.
    pub messages: Vec<MessageDiffSummary>,
    /// Change sets currently available for redo.
    pub reverted_messages: Vec<MessageDiffSummary>,
    /// Unique file paths touched by active change sets.
    pub files_changed: Vec<PathBuf>,
    /// Whether undo is currently available.
    pub can_undo: bool,
    /// Whether redo is currently available.
    pub can_redo: bool,
    /// Count of active change sets.
    pub active_change_sets: usize,
    /// Count of reverted change sets.
    pub reverted_change_sets: usize,
}

/// Tracks diffs per message and supports undo (revert last message) and redo (unrevert).
#[derive(Debug)]
pub struct DiffTracker {
    /// Current message being recorded (set at turn start, cleared at turn end)
    current_message_id: Option<Uuid>,
    /// Snapshots accumulated for the current message
    current_snapshots: Vec<FileSnapshot>,
    /// Completed message diffs (oldest first)
    stack: Vec<MessageDiff>,
    /// Reverted message diffs (most recently reverted last)
    reverted_stack: Vec<MessageDiff>,
}

impl DiffTracker {
    /// Create a new diff tracker.
    pub fn new() -> Self {
        Self {
            current_message_id: None,
            current_snapshots: Vec::new(),
            stack: Vec::new(),
            reverted_stack: Vec::new(),
        }
    }

    /// Create a tracker from durable session diff state.
    #[must_use]
    pub fn from_state(state: SessionDiffState) -> Self {
        let mut tracker = Self::new();
        tracker.replace_state(state);
        tracker
    }

    /// Snapshot committed diff state for persistence.
    #[must_use]
    pub fn snapshot_state(&self) -> SessionDiffState {
        SessionDiffState {
            active: self.stack.clone(),
            reverted: self.reverted_stack.clone(),
        }
    }

    /// Replace committed diff state, clearing any in-progress message.
    pub fn replace_state(&mut self, state: SessionDiffState) {
        self.current_message_id = None;
        self.current_snapshots.clear();
        self.stack = state.active;
        self.reverted_stack = state.reverted;
    }

    /// Start recording file changes for a message. Call at turn start.
    pub fn start_message(&mut self, message_id: Uuid) {
        self.finish_message_inner();
        self.current_message_id = Some(message_id);
        self.current_snapshots = Vec::new();
    }

    /// Finish the current message and push it onto the stack. Call at turn end.
    pub fn finish_message(&mut self) {
        self.finish_message_inner();
    }

    fn finish_message_inner(&mut self) {
        if let Some(id) = self.current_message_id.take()
            && !self.current_snapshots.is_empty()
        {
            self.stack.push(MessageDiff {
                message_id: id,
                snapshots: std::mem::take(&mut self.current_snapshots),
            });
            self.reverted_stack.clear();
        }
    }

    /// Record the state of a file before a write/edit. Call from hooks before tool runs.
    pub fn record_before(&mut self, path: &Path, content: Option<&[u8]>) {
        if self.current_message_id.is_none() {
            return;
        }
        let path_buf = path.to_path_buf();
        // Preserve the original "before" state: if a snapshot already exists for this
        // path in the current message, ignore subsequent record_before calls so that
        // only the first pre-edit content (the true original) is retained.
        if self.current_snapshots.iter().any(|s| s.path == path_buf) {
            return;
        }
        self.current_snapshots.push(FileSnapshot {
            path: path_buf,
            before: content.map(<[u8]>::to_vec),
            after: None,
        });
    }

    /// Record the state of a file after a write/edit. Call from hooks after tool runs.
    pub fn record_after(&mut self, path: &Path, content: Option<&[u8]>) {
        if self.current_message_id.is_none() {
            return;
        }
        let path_buf = path.to_path_buf();
        let after = content.map(<[u8]>::to_vec);
        if let Some(s) = self
            .current_snapshots
            .iter_mut()
            .find(|s| s.path == path_buf)
        {
            // Move `after` directly — the early return means it is not used below.
            s.after = after;
            return;
        }
        self.current_snapshots.push(FileSnapshot {
            path: path_buf,
            before: None,
            after,
        });
    }

    /// Undo the last message: restore files to their "before" state and push that message onto the reverted stack.
    pub fn undo_last(&mut self) -> Result<Option<MessageDiff>> {
        self.finish_message_inner();
        let diff = match self.stack.pop() {
            Some(d) => d,
            None => return Ok(None),
        };
        for snap in &diff.snapshots {
            apply_snapshot_restore(snap, true)?;
        }
        self.reverted_stack.push(diff.clone());
        Ok(Some(diff))
    }

    /// Undo the message identified by `message_id` and every newer message after it.
    ///
    /// Returns the reverted diffs in reverse chronological order (newest first).
    pub fn undo_to_message(&mut self, message_id: Uuid) -> Result<Vec<MessageDiff>> {
        self.finish_message_inner();
        let Some(idx) = self.stack.iter().position(|d| d.message_id == message_id) else {
            return Ok(Vec::new());
        };
        let mut reverted = Vec::with_capacity(self.stack.len() - idx);
        while self.stack.len() > idx {
            let Some(diff) = self.undo_last()? else {
                break;
            };
            reverted.push(diff);
        }
        Ok(reverted)
    }

    /// Redo the last reverted message: re-apply "after" state and move it back to the main stack.
    pub fn redo_last(&mut self) -> Result<Option<MessageDiff>> {
        let diff = match self.reverted_stack.pop() {
            Some(d) => d,
            None => return Ok(None),
        };
        for snap in &diff.snapshots {
            apply_snapshot_restore(snap, false)?;
        }
        self.stack.push(diff.clone());
        Ok(Some(diff))
    }

    /// Immutable view of all message diffs for this session (for API).
    pub fn diffs_for_session(&self) -> &[MessageDiff] {
        &self.stack
    }

    /// Immutable view of reverted message diffs for this session (for redo/UI).
    pub fn reverted_diffs_for_session(&self) -> &[MessageDiff] {
        &self.reverted_stack
    }

    /// Build a lightweight summary of the session diff state.
    pub fn session_summary(&self, message_id: Option<Uuid>) -> SessionDiffSummary {
        let matches_message =
            |diff: &&MessageDiff| message_id.is_none_or(|id| diff.message_id == id);

        let messages: Vec<MessageDiffSummary> = self
            .stack
            .iter()
            .filter(matches_message)
            .map(MessageDiffSummary::from)
            .collect();
        let reverted_messages: Vec<MessageDiffSummary> = self
            .reverted_stack
            .iter()
            .filter(matches_message)
            .map(MessageDiffSummary::from)
            .collect();
        let files_changed = messages
            .iter()
            .flat_map(|message| {
                message
                    .snapshots
                    .iter()
                    .map(|snapshot| snapshot.path.clone())
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        SessionDiffSummary {
            active_change_sets: messages.len(),
            reverted_change_sets: reverted_messages.len(),
            files_changed,
            can_undo: self.can_undo(),
            can_redo: self.can_redo(),
            messages,
            reverted_messages,
        }
    }

    /// Whether there is something to undo.
    ///
    /// True when the stack has committed change sets, or when the in-progress
    /// message has recorded at least one snapshot (empty in-progress turns are
    /// not undoable and must not advertise undo).
    pub fn can_undo(&self) -> bool {
        !self.stack.is_empty()
            || (self.current_message_id.is_some() && !self.current_snapshots.is_empty())
    }

    /// Whether there is something to redo.
    pub fn can_redo(&self) -> bool {
        !self.reverted_stack.is_empty()
    }
}

impl Default for DiffTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Restore file to `before` (undo) or `after` (redo). `use_before == true` => write `before` (or delete if None); else write `after`.
fn apply_snapshot_restore(snap: &FileSnapshot, use_before: bool) -> Result<()> {
    let content: Option<&[u8]> = if use_before {
        snap.before.as_deref()
    } else {
        snap.after.as_deref()
    };
    if let Some(data) = content {
        if let Some(parent) = snap.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir for {:?}", snap.path))?;
        }
        std::fs::write(&snap.path, data).with_context(|| format!("write {:?}", snap.path))?;
    } else if snap.path.exists() {
        std::fs::remove_file(&snap.path).with_context(|| format!("remove {:?}", snap.path))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn record_write(
        tracker: &mut DiffTracker,
        message_id: Uuid,
        path: &Path,
        before: &[u8],
        after: &[u8],
    ) -> Result<()> {
        tracker.start_message(message_id);
        tracker.record_before(path, Some(before));
        std::fs::write(path, after)?;
        tracker.record_after(path, Some(after));
        tracker.finish_message();
        Ok(())
    }

    #[test]
    fn undo_to_message_reverts_target_and_newer() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"v0")?;

        let id_a = Uuid::now_v7();
        let id_b = Uuid::now_v7();
        let id_c = Uuid::now_v7();

        let mut tracker = DiffTracker::new();
        record_write(&mut tracker, id_a, &path, b"v0", b"v1")?;
        record_write(&mut tracker, id_b, &path, b"v1", b"v2")?;
        record_write(&mut tracker, id_c, &path, b"v2", b"v3")?;

        let reverted = tracker.undo_to_message(id_b)?;
        assert_eq!(reverted.len(), 2);
        assert_eq!(reverted[0].message_id, id_c);
        assert_eq!(reverted[1].message_id, id_b);
        assert_eq!(std::fs::read(&path)?, b"v1");
        assert_eq!(tracker.diffs_for_session().len(), 1);
        assert_eq!(tracker.diffs_for_session()[0].message_id, id_a);
        assert!(tracker.can_redo());
        Ok(())
    }

    #[test]
    fn undo_to_message_unknown_id_is_noop() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"v0")?;

        let id_a = Uuid::now_v7();
        let mut tracker = DiffTracker::new();
        record_write(&mut tracker, id_a, &path, b"v0", b"v1")?;

        let reverted = tracker.undo_to_message(Uuid::now_v7())?;
        assert!(reverted.is_empty());
        assert_eq!(std::fs::read(&path)?, b"v1");
        assert_eq!(tracker.diffs_for_session().len(), 1);
        Ok(())
    }

    #[test]
    fn undo_last_then_redo_restores_file() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"before")?;

        let id = Uuid::now_v7();
        let mut tracker = DiffTracker::new();
        record_write(&mut tracker, id, &path, b"before", b"after")?;

        let undone = tracker
            .undo_last()?
            .ok_or_else(|| anyhow::anyhow!("undo should return diff"))?;
        assert_eq!(undone.message_id, id);
        assert_eq!(std::fs::read(&path)?, b"before");

        let redone = tracker
            .redo_last()?
            .ok_or_else(|| anyhow::anyhow!("redo should return diff"))?;
        assert_eq!(redone.message_id, id);
        assert_eq!(std::fs::read(&path)?, b"after");
        assert!(!tracker.can_redo());
        Ok(())
    }

    #[test]
    fn change_kind_classifies_created_modified_deleted() {
        assert_eq!(
            FileSnapshot {
                path: PathBuf::from("a"),
                before: None,
                after: Some(b"x".to_vec()),
            }
            .change_kind(),
            SnapshotChangeKind::Created
        );
        assert_eq!(
            FileSnapshot {
                path: PathBuf::from("b"),
                before: Some(b"x".to_vec()),
                after: Some(b"y".to_vec()),
            }
            .change_kind(),
            SnapshotChangeKind::Modified
        );
        assert_eq!(
            FileSnapshot {
                path: PathBuf::from("c"),
                before: Some(b"x".to_vec()),
                after: None,
            }
            .change_kind(),
            SnapshotChangeKind::Deleted
        );
    }

    #[test]
    fn finish_message_with_empty_snapshots_does_not_push_or_clear_redo() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"v0")?;

        let id = Uuid::now_v7();
        let mut tracker = DiffTracker::new();
        record_write(&mut tracker, id, &path, b"v0", b"v1")?;
        tracker.undo_last()?;
        assert!(tracker.can_redo());

        tracker.start_message(Uuid::now_v7());
        tracker.finish_message();

        assert!(tracker.diffs_for_session().is_empty());
        assert!(tracker.can_redo());
        assert!(!tracker.can_undo());
        Ok(())
    }

    #[test]
    fn snapshot_state_roundtrips_active_and_reverted_diffs() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"v0")?;

        let id_a = Uuid::now_v7();
        let id_b = Uuid::now_v7();
        let mut tracker = DiffTracker::new();
        record_write(&mut tracker, id_a, &path, b"v0", b"v1")?;
        record_write(&mut tracker, id_b, &path, b"v1", b"v2")?;
        tracker.undo_last()?;

        let restored = DiffTracker::from_state(tracker.snapshot_state());
        assert_eq!(restored.diffs_for_session().len(), 1);
        assert_eq!(restored.diffs_for_session()[0].message_id, id_a);
        assert_eq!(restored.reverted_diffs_for_session().len(), 1);
        assert_eq!(restored.reverted_diffs_for_session()[0].message_id, id_b);
        assert!(restored.can_undo());
        assert!(restored.can_redo());
        Ok(())
    }

    #[test]
    fn replace_state_clears_in_progress_message() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"v0")?;

        let id = Uuid::now_v7();
        let mut source = DiffTracker::new();
        record_write(&mut source, id, &path, b"v0", b"v1")?;

        let mut tracker = DiffTracker::new();
        tracker.start_message(Uuid::now_v7());
        tracker.record_before(&path, Some(b"dirty"));
        assert!(tracker.can_undo());

        tracker.replace_state(source.snapshot_state());
        assert_eq!(tracker.diffs_for_session().len(), 1);
        assert_eq!(tracker.diffs_for_session()[0].message_id, id);
        assert!(tracker.undo_last()?.is_some());
        Ok(())
    }

    #[test]
    fn empty_in_progress_message_is_not_undoable() -> Result<()> {
        let mut tracker = DiffTracker::new();
        tracker.start_message(Uuid::now_v7());
        assert!(!tracker.can_undo());
        assert!(tracker.undo_last()?.is_none());
        Ok(())
    }

    #[test]
    fn records_ignored_without_active_message() {
        let mut tracker = DiffTracker::new();
        tracker.record_before(Path::new("x.txt"), Some(b"a"));
        tracker.record_after(Path::new("x.txt"), Some(b"b"));
        tracker.finish_message();
        assert!(tracker.diffs_for_session().is_empty());
        assert!(!tracker.can_undo());
    }

    #[test]
    fn session_summary_lists_active_and_reverted_change_sets() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"v0")?;

        let id_a = Uuid::now_v7();
        let id_b = Uuid::now_v7();
        let mut tracker = DiffTracker::new();
        record_write(&mut tracker, id_a, &path, b"v0", b"v1")?;
        record_write(&mut tracker, id_b, &path, b"v1", b"v2")?;
        tracker.undo_last()?;

        let summary = tracker.session_summary(None);
        assert_eq!(summary.active_change_sets, 1);
        assert_eq!(summary.reverted_change_sets, 1);
        assert_eq!(summary.messages[0].message_id, id_a);
        assert_eq!(summary.reverted_messages[0].message_id, id_b);
        assert!(summary.can_undo);
        assert!(summary.can_redo);
        assert_eq!(summary.files_changed, vec![path.clone()]);
        Ok(())
    }

    #[test]
    fn session_summary_filter_keeps_global_undo_redo_flags() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("note.txt");
        std::fs::write(&path, b"v0")?;

        let id_a = Uuid::now_v7();
        let id_b = Uuid::now_v7();
        let mut tracker = DiffTracker::new();
        record_write(&mut tracker, id_a, &path, b"v0", b"v1")?;
        record_write(&mut tracker, id_b, &path, b"v1", b"v2")?;
        tracker.undo_last()?;

        let filtered = tracker.session_summary(Some(id_a));
        assert_eq!(filtered.messages.len(), 1);
        assert!(filtered.reverted_messages.is_empty());
        // Flags remain session-global even when message lists are filtered.
        assert!(filtered.can_undo);
        assert!(filtered.can_redo);
        Ok(())
    }
}
