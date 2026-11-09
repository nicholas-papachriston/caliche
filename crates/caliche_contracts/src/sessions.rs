use serde::{Deserialize, Serialize};

use crate::settings::InteractionModeDto;

/// Maximum length of a caller-supplied session title, in Unicode scalar values.
pub const SESSION_TITLE_MAX_CHARS: usize = 80;

/// A caller-supplied session title is longer than [`SESSION_TITLE_MAX_CHARS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionTitleTooLong;

impl core::fmt::Display for SessionTitleTooLong {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "session title is longer than {SESSION_TITLE_MAX_CHARS} characters"
        )
    }
}

impl core::error::Error for SessionTitleTooLong {}

/// Trim a caller-supplied session title.
///
/// Missing, blank, and whitespace-only titles are `Ok(None)`. A title longer
/// than [`SESSION_TITLE_MAX_CHARS`] Unicode scalar values is an error. The
/// stored value is the trimmed title.
pub fn normalize_session_title(raw: Option<&str>) -> Result<Option<String>, SessionTitleTooLong> {
    let Some(raw) = raw else {
        return Ok(None);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    if trimmed.chars().count() > SESSION_TITLE_MAX_CHARS {
        return Err(SessionTitleTooLong);
    }
    Ok(Some(trimmed.to_owned()))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateSessionRequestDto {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<String>,
    #[serde(default)]
    pub context_files: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    /// Short human title. Omitted, null, and blank mean no title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

impl Default for CreateSessionRequestDto {
    fn default() -> Self {
        Self {
            workspace: None,
            context_files: Vec::new(),
            max_turns: Some(20),
            system_prompt: None,
            title: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreateSessionResponseDto {
    pub session_id: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiffChangeKindDto {
    Created,
    Modified,
    Deleted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffFileSummaryDto {
    pub path: String,
    pub change: DiffChangeKindDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_len: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffMessageSummaryDto {
    pub message_id: String,
    pub changed_files: usize,
    #[serde(default)]
    pub snapshots: Vec<DiffFileSummaryDto>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffResponseDto {
    #[serde(default)]
    pub messages: Vec<DiffMessageSummaryDto>,
    #[serde(default)]
    pub reverted_messages: Vec<DiffMessageSummaryDto>,
    #[serde(default)]
    pub files_changed: Vec<String>,
    #[serde(default)]
    pub can_undo: bool,
    #[serde(default)]
    pub can_redo: bool,
    #[serde(default)]
    pub active_change_sets: usize,
    #[serde(default)]
    pub reverted_change_sets: usize,
}

impl DiffResponseDto {
    #[must_use]
    pub const fn has_any_changes(&self) -> bool {
        !self.messages.is_empty() || !self.reverted_messages.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UndoResponseDto {
    pub undone: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reverted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub files_restored: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_set: Option<DiffMessageSummaryDto>,
    #[serde(default)]
    pub can_undo: bool,
    #[serde(default)]
    pub can_redo: bool,
    #[serde(default)]
    pub remaining_change_sets: usize,
    #[serde(default)]
    pub reverted_change_sets: usize,
}

impl UndoResponseDto {
    #[must_use]
    pub fn status_message(&self) -> String {
        format_change_action_message("undone", self.undone, &self.files, self.message.as_deref())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RedoResponseDto {
    pub redone: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unreverted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    #[serde(default)]
    pub files: Vec<String>,
    #[serde(default)]
    pub files_restored: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub change_set: Option<DiffMessageSummaryDto>,
    #[serde(default)]
    pub can_undo: bool,
    #[serde(default)]
    pub can_redo: bool,
    #[serde(default)]
    pub remaining_change_sets: usize,
    #[serde(default)]
    pub reverted_change_sets: usize,
}

impl RedoResponseDto {
    #[must_use]
    pub fn status_message(&self) -> String {
        format_change_action_message("redone", self.redone, &self.files, self.message.as_deref())
    }
}

fn format_change_action_message(
    verb: &str,
    applied: bool,
    files: &[String],
    fallback_message: Option<&str>,
) -> String {
    if applied {
        if files.is_empty() {
            format!("{verb} last change set")
        } else {
            format!("{verb} {}", files.join(", "))
        }
    } else {
        fallback_message
            .unwrap_or(match verb {
                "undone" => "Nothing to undo",
                "redone" => "Nothing to redo",
                _ => "No changes applied",
            })
            .to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointResponseDto {
    pub checkpoint_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollbackResponseDto {
    pub files_restored: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShareVisibilityDto {
    #[default]
    Public,
    Authenticated,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ShareCreateRequestDto {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visibility: Option<ShareVisibilityDto>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareCreateResponseDto {
    pub slug: String,
    pub session_id: String,
    pub created_at: String,
    pub url: String,
    #[serde(default)]
    pub redaction_count: usize,
    #[serde(default)]
    pub visibility: ShareVisibilityDto,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareMessageDto {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareResolveResponseDto {
    pub slug: String,
    pub session_id: String,
    pub created_at: String,
    #[serde(default)]
    pub messages: Vec<ShareMessageDto>,
    #[serde(default)]
    pub redaction_count: usize,
    #[serde(default)]
    pub visibility: ShareVisibilityDto,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModeResponseDto {
    pub mode: InteractionModeDto,
    pub write_tools_enabled: bool,
    pub description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_session_request_default_max_turns_and_roundtrip() {
        let default = CreateSessionRequestDto::default();
        assert_eq!(default.max_turns, Some(20));

        let parsed: CreateSessionRequestDto =
            serde_json::from_str("{}").expect("empty object should deserialize");
        assert!(parsed.workspace.is_none());
        assert!(parsed.context_files.is_empty());
        // serde default for Option is None unless #[serde(default = "...")]
        assert_eq!(parsed.max_turns, None);

        let with_fields = CreateSessionRequestDto {
            workspace: Some("/tmp/ws".into()),
            context_files: vec!["a.rs".into()],
            max_turns: Some(8),
            system_prompt: Some("be brief".into()),
            title: Some("desk".into()),
        };
        let json = serde_json::to_value(&with_fields).expect("serialize");
        let again: CreateSessionRequestDto = serde_json::from_value(json).expect("deserialize");
        assert_eq!(again, with_fields);
        assert!(parsed.title.is_none());
    }

    #[test]
    fn normalize_session_title_drops_blank_and_rejects_overlong() {
        assert_eq!(normalize_session_title(None).expect("missing"), None);
        assert_eq!(normalize_session_title(Some("")).expect("empty"), None);
        assert_eq!(normalize_session_title(Some("  \n")).expect("blank"), None);
        assert_eq!(
            normalize_session_title(Some("  desk  ")).expect("trim"),
            Some("desk".to_owned())
        );

        let max = "a".repeat(SESSION_TITLE_MAX_CHARS);
        assert_eq!(normalize_session_title(Some(&max)).expect("max"), Some(max));
        let over = "a".repeat(SESSION_TITLE_MAX_CHARS + 1);
        assert_eq!(
            normalize_session_title(Some(&over)),
            Err(SessionTitleTooLong)
        );
    }

    #[test]
    fn create_session_request_omits_title_when_absent() {
        let parsed: CreateSessionRequestDto =
            serde_json::from_str(r#"{"title":null}"#).expect("null title");
        assert!(parsed.title.is_none());

        let bare: CreateSessionRequestDto = serde_json::from_str("{}").expect("bare object");
        let json = serde_json::to_value(&bare).expect("serialize");
        assert!(json.get("title").is_none());
    }

    #[test]
    fn diff_response_has_any_changes_checks_both_stacks() {
        let empty = DiffResponseDto {
            messages: vec![],
            reverted_messages: vec![],
            files_changed: vec![],
            can_undo: false,
            can_redo: false,
            active_change_sets: 0,
            reverted_change_sets: 0,
        };
        assert!(!empty.has_any_changes());

        let active = DiffResponseDto {
            messages: vec![DiffMessageSummaryDto {
                message_id: "m1".into(),
                changed_files: 1,
                snapshots: vec![],
            }],
            ..empty.clone()
        };
        assert!(active.has_any_changes());

        let reverted = DiffResponseDto {
            reverted_messages: vec![DiffMessageSummaryDto {
                message_id: "m2".into(),
                changed_files: 1,
                snapshots: vec![],
            }],
            ..empty
        };
        assert!(reverted.has_any_changes());
    }

    #[test]
    fn undo_redo_status_messages_cover_files_and_noop_fallbacks() {
        let undone_with_files = UndoResponseDto {
            undone: true,
            reverted: Some(true),
            message_id: Some("msg-1".into()),
            files: vec!["a.rs".into(), "b.rs".into()],
            files_restored: 2,
            message: None,
            change_set: None,
            can_undo: true,
            can_redo: true,
            remaining_change_sets: 1,
            reverted_change_sets: 1,
        };
        assert_eq!(undone_with_files.status_message(), "undone a.rs, b.rs");

        let noop_undo = UndoResponseDto {
            undone: false,
            reverted: None,
            message_id: None,
            files: vec![],
            files_restored: 0,
            message: None,
            change_set: None,
            can_undo: false,
            can_redo: false,
            remaining_change_sets: 0,
            reverted_change_sets: 0,
        };
        assert_eq!(noop_undo.status_message(), "Nothing to undo");

        let redone_empty_files = RedoResponseDto {
            redone: true,
            unreverted: Some(true),
            message_id: Some("msg-2".into()),
            files: vec![],
            files_restored: 0,
            message: None,
            change_set: None,
            can_undo: true,
            can_redo: false,
            remaining_change_sets: 1,
            reverted_change_sets: 0,
        };
        assert_eq!(
            redone_empty_files.status_message(),
            "redone last change set"
        );

        let noop_redo = RedoResponseDto {
            redone: false,
            unreverted: None,
            message_id: None,
            files: vec![],
            files_restored: 0,
            message: Some("custom noop".into()),
            change_set: None,
            can_undo: true,
            can_redo: false,
            remaining_change_sets: 0,
            reverted_change_sets: 0,
        };
        assert_eq!(noop_redo.status_message(), "custom noop");
    }
}
