//! Shared mapping from harness diff state into session DTO contracts.

use caliche_contracts::sessions::{
    DiffChangeKindDto, DiffFileSummaryDto, DiffMessageSummaryDto, DiffResponseDto, RedoResponseDto,
    UndoResponseDto,
};

use crate::harness::{MessageDiff, MessageDiffSummary, SessionDiffSummary, SnapshotChangeKind};

/// Map a harness snapshot change kind to the shared DTO enum.
#[must_use]
pub const fn change_kind_to_dto(change: SnapshotChangeKind) -> DiffChangeKindDto {
    match change {
        SnapshotChangeKind::Created => DiffChangeKindDto::Created,
        SnapshotChangeKind::Modified => DiffChangeKindDto::Modified,
        SnapshotChangeKind::Deleted => DiffChangeKindDto::Deleted,
    }
}

/// Map a per-message diff summary into a shared DTO.
#[must_use]
pub fn message_summary_to_dto(summary: &MessageDiffSummary) -> DiffMessageSummaryDto {
    DiffMessageSummaryDto {
        message_id: summary.message_id.to_string(),
        changed_files: summary.changed_files,
        snapshots: summary
            .snapshots
            .iter()
            .map(|snapshot| DiffFileSummaryDto {
                path: snapshot.path.display().to_string(),
                change: change_kind_to_dto(snapshot.change),
                before_len: snapshot.before_len,
                after_len: snapshot.after_len,
            })
            .collect(),
    }
}

/// Map a full session diff summary into a shared DTO response.
#[must_use]
pub fn diff_response_from_summary(summary: &SessionDiffSummary) -> DiffResponseDto {
    DiffResponseDto {
        messages: summary
            .messages
            .iter()
            .map(message_summary_to_dto)
            .collect(),
        reverted_messages: summary
            .reverted_messages
            .iter()
            .map(message_summary_to_dto)
            .collect(),
        files_changed: summary
            .files_changed
            .iter()
            .map(|path| path.display().to_string())
            .collect(),
        can_undo: summary.can_undo,
        can_redo: summary.can_redo,
        active_change_sets: summary.active_change_sets,
        reverted_change_sets: summary.reverted_change_sets,
    }
}

fn files_from_message_diff(diff: &MessageDiff) -> Vec<String> {
    diff.snapshots
        .iter()
        .map(|snapshot| snapshot.path.display().to_string())
        .collect()
}

/// Build a successful undo DTO after a diff has been reverted.
#[must_use]
pub fn undo_response_from_success(
    diff: &MessageDiff,
    summary: &SessionDiffSummary,
) -> UndoResponseDto {
    UndoResponseDto {
        undone: true,
        reverted: Some(true),
        message_id: Some(diff.message_id.to_string()),
        files: files_from_message_diff(diff),
        files_restored: diff.snapshots.len(),
        message: None,
        change_set: Some(message_summary_to_dto(&MessageDiffSummary::from(diff))),
        can_undo: summary.can_undo,
        can_redo: summary.can_redo,
        remaining_change_sets: summary.active_change_sets,
        reverted_change_sets: summary.reverted_change_sets,
    }
}

/// Build a no-op undo DTO.
#[must_use]
pub fn undo_response_from_noop(
    message: impl Into<String>,
    summary: &SessionDiffSummary,
) -> UndoResponseDto {
    UndoResponseDto {
        undone: false,
        reverted: Some(false),
        message_id: None,
        files: Vec::new(),
        files_restored: 0,
        message: Some(message.into()),
        change_set: None,
        can_undo: summary.can_undo,
        can_redo: summary.can_redo,
        remaining_change_sets: summary.active_change_sets,
        reverted_change_sets: summary.reverted_change_sets,
    }
}

/// Build a successful redo DTO after a diff has been restored.
#[must_use]
pub fn redo_response_from_success(
    diff: &MessageDiff,
    summary: &SessionDiffSummary,
) -> RedoResponseDto {
    RedoResponseDto {
        redone: true,
        unreverted: Some(true),
        message_id: Some(diff.message_id.to_string()),
        files: files_from_message_diff(diff),
        files_restored: diff.snapshots.len(),
        message: None,
        change_set: Some(message_summary_to_dto(&MessageDiffSummary::from(diff))),
        can_undo: summary.can_undo,
        can_redo: summary.can_redo,
        remaining_change_sets: summary.active_change_sets,
        reverted_change_sets: summary.reverted_change_sets,
    }
}

/// Build a no-op redo DTO.
#[must_use]
pub fn redo_response_from_noop(
    message: impl Into<String>,
    summary: &SessionDiffSummary,
) -> RedoResponseDto {
    RedoResponseDto {
        redone: false,
        unreverted: Some(false),
        message_id: None,
        files: Vec::new(),
        files_restored: 0,
        message: Some(message.into()),
        change_set: None,
        can_undo: summary.can_undo,
        can_redo: summary.can_redo,
        remaining_change_sets: summary.active_change_sets,
        reverted_change_sets: summary.reverted_change_sets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{
        FileSnapshotSummary, MessageDiff, MessageDiffSummary, SessionDiffSummary,
    };
    use std::path::PathBuf;
    use uuid::Uuid;

    fn summary_with_active_and_reverted() -> (Uuid, Uuid, SessionDiffSummary) {
        let active_id = Uuid::now_v7();
        let reverted_id = Uuid::now_v7();
        (
            active_id,
            reverted_id,
            SessionDiffSummary {
                messages: vec![MessageDiffSummary {
                    message_id: active_id,
                    changed_files: 1,
                    snapshots: vec![FileSnapshotSummary {
                        path: PathBuf::from("src/lib.rs"),
                        change: SnapshotChangeKind::Modified,
                        before_len: Some(3),
                        after_len: Some(4),
                    }],
                }],
                reverted_messages: vec![MessageDiffSummary {
                    message_id: reverted_id,
                    changed_files: 1,
                    snapshots: vec![FileSnapshotSummary {
                        path: PathBuf::from("new.txt"),
                        change: SnapshotChangeKind::Created,
                        before_len: None,
                        after_len: Some(2),
                    }],
                }],
                files_changed: vec![PathBuf::from("src/lib.rs")],
                can_undo: true,
                can_redo: true,
                active_change_sets: 1,
                reverted_change_sets: 1,
            },
        )
    }

    #[test]
    fn diff_response_preserves_active_reverted_files_and_flags() {
        let (active_id, reverted_id, summary) = summary_with_active_and_reverted();

        let dto = diff_response_from_summary(&summary);

        assert_eq!(dto.messages[0].message_id, active_id.to_string());
        assert_eq!(dto.messages[0].snapshots[0].path, "src/lib.rs");
        assert_eq!(
            dto.messages[0].snapshots[0].change,
            DiffChangeKindDto::Modified
        );
        assert_eq!(dto.reverted_messages[0].message_id, reverted_id.to_string());
        assert_eq!(
            dto.reverted_messages[0].snapshots[0].change,
            DiffChangeKindDto::Created
        );
        assert_eq!(dto.files_changed, vec!["src/lib.rs"]);
        assert!(dto.can_undo);
        assert!(dto.can_redo);
    }

    #[test]
    fn undo_success_includes_change_set_and_remaining_counts() {
        let (_, reverted_id, summary) = summary_with_active_and_reverted();
        let diff = MessageDiff {
            message_id: reverted_id,
            snapshots: vec![crate::harness::FileSnapshot {
                path: PathBuf::from("new.txt"),
                before: None,
                after: Some(b"hi".to_vec()),
            }],
        };

        let dto = undo_response_from_success(&diff, &summary);

        assert!(dto.undone);
        assert_eq!(
            dto.message_id.as_deref(),
            Some(reverted_id.to_string().as_str())
        );
        assert_eq!(dto.files, vec!["new.txt"]);
        assert_eq!(dto.files_restored, 1);
        assert_eq!(
            dto.change_set.as_ref().map(|set| set.changed_files),
            Some(1)
        );
        assert_eq!(dto.remaining_change_sets, 1);
        assert_eq!(dto.reverted_change_sets, 1);
    }

    #[test]
    fn noop_responses_keep_flags_and_messages() {
        let (_, _, summary) = summary_with_active_and_reverted();

        let undo = undo_response_from_noop("Nothing to undo", &summary);
        let redo = redo_response_from_noop("Nothing to redo", &summary);

        assert!(!undo.undone);
        assert_eq!(undo.message.as_deref(), Some("Nothing to undo"));
        assert!(undo.can_redo);
        assert!(!redo.redone);
        assert_eq!(redo.message.as_deref(), Some("Nothing to redo"));
        assert!(redo.can_undo);
    }

    #[test]
    fn redo_success_sets_unreverted_and_change_set() {
        let (active_id, _, summary) = summary_with_active_and_reverted();
        let diff = MessageDiff {
            message_id: active_id,
            snapshots: vec![crate::harness::FileSnapshot {
                path: PathBuf::from("src/lib.rs"),
                before: Some(b"old".to_vec()),
                after: Some(b"new".to_vec()),
            }],
        };

        let dto = redo_response_from_success(&diff, &summary);

        assert!(dto.redone);
        assert_eq!(dto.unreverted, Some(true));
        assert_eq!(
            dto.message_id.as_deref(),
            Some(active_id.to_string().as_str())
        );
        assert_eq!(
            dto.change_set.as_ref().map(|set| set.changed_files),
            Some(1)
        );
    }
}
