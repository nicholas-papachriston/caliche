//! Conversation history state.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::api::types::MessageRole;
use crate::state::Message;

const TITLE_MAX_CHARS: usize = 72;

/// A completed or in-progress conversation saved for later review.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SavedConversation {
    /// Stable key — equals the server session_id when one exists.
    pub id: String,
    /// Friendly display name derived from the first user turn.
    pub title: String,
    /// When this snapshot was last updated.
    pub saved_at: DateTime<Utc>,
    /// Server session ID, if one was created.
    pub session_id: Option<String>,
    /// Full message list at the time of save.
    pub messages: Vec<Message>,
    /// Cumulative token count.
    pub total_tokens: u32,
}

impl SavedConversation {
    /// Build a snapshot from the current chat, keyed by session_id or the first user message.
    pub fn from_chat(
        messages: Vec<Message>,
        session_id: Option<String>,
        total_tokens: u32,
    ) -> Self {
        let title = friendly_title(&messages);
        let id = conversation_key(&messages, session_id.as_deref());

        Self {
            id,
            title,
            saved_at: Utc::now(),
            session_id,
            messages,
            total_tokens,
        }
    }

    /// Insert or replace a conversation snapshot, keeping the newest entry first.
    pub fn upsert(entries: &mut Vec<Self>, conversation: Self) {
        entries.retain(|entry| !entry.matches_conversation(&conversation));
        entries.insert(0, conversation);
    }

    /// Collapse duplicate snapshots from older UI versions and keep newest first.
    pub fn normalize_list(entries: Vec<Self>) -> Vec<Self> {
        let mut normalized = Vec::with_capacity(entries.len());
        let mut sorted = entries;
        sorted.sort_by(|left, right| left.saved_at.cmp(&right.saved_at));
        for entry in sorted {
            Self::upsert(&mut normalized, entry);
        }
        normalized
    }

    /// Whether this snapshot belongs to the same logical conversation.
    pub fn matches_conversation(&self, other: &Self) -> bool {
        self.id == other.id
            || shared_session_id(self.session_id.as_deref(), other.session_id.as_deref())
            || shared_first_user_message(self, other)
    }
}

fn conversation_key(messages: &[Message], session_id: Option<&str>) -> String {
    session_id
        .filter(|id| !id.trim().is_empty())
        .map(str::to_string)
        .or_else(|| first_user_message_id(messages))
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

fn first_user_message_id(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .find(|message| message.role == MessageRole::User)
        .map(|message| message.message_id.clone())
}

fn shared_session_id(left: Option<&str>, right: Option<&str>) -> bool {
    matches!((left, right), (Some(left), Some(right)) if !left.is_empty() && left == right)
}

fn shared_first_user_message(left: &SavedConversation, right: &SavedConversation) -> bool {
    matches!(
        (
            first_user_message_id(&left.messages),
            first_user_message_id(&right.messages)
        ),
        (Some(left_id), Some(right_id)) if left_id == right_id
    )
}

fn friendly_title(messages: &[Message]) -> String {
    messages
        .iter()
        .find(|message| message.role == MessageRole::User)
        .map(|message| title_from_content(&message.content))
        .filter(|title| !title.is_empty())
        .unwrap_or_else(|| "Untitled conversation".to_string())
}

fn title_from_content(content: &str) -> String {
    let candidate = content
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with("[attached:"))
        .unwrap_or(content.trim());

    let title = if let Some(command) = candidate.strip_prefix('/') {
        command_title(command)
    } else {
        sentence_case(&collapse_whitespace(candidate))
    };
    truncate_title(title.trim())
}

fn command_title(command: &str) -> String {
    let mut parts = command.split_whitespace();
    let name = parts.next().unwrap_or("command");
    let rest = parts.collect::<Vec<_>>().join(" ");
    let label = name.replace(['-', '_'], " ");
    let label = sentence_case(&label);
    if rest.is_empty() {
        label
    } else {
        format!("{label}: {}", collapse_whitespace(&rest))
    }
}

fn collapse_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn sentence_case(value: &str) -> String {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let first = first.to_uppercase().collect::<String>();
    format!("{first}{}", chars.as_str())
}

fn truncate_title(value: &str) -> String {
    if value.chars().count() <= TITLE_MAX_CHARS {
        value.to_string()
    } else {
        let truncated = value.chars().take(TITLE_MAX_CHARS).collect::<String>();
        format!("{}…", truncated.trim_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Message;

    #[test]
    fn snapshot_uses_session_id_as_stable_key() {
        let conversation = SavedConversation::from_chat(
            vec![Message::user("hello")],
            Some("session-123".to_string()),
            42,
        );

        assert_eq!(conversation.id, "session-123");
        assert_eq!(conversation.session_id.as_deref(), Some("session-123"));
        assert_eq!(conversation.total_tokens, 42);
    }

    #[test]
    fn snapshot_title_uses_first_user_message() {
        let conversation = SavedConversation::from_chat(
            vec![
                Message::system("system notice"),
                Message::user("actual task"),
            ],
            None,
            0,
        );

        assert_eq!(conversation.title, "Actual task");
    }

    #[test]
    fn snapshot_title_truncates_on_character_boundaries() {
        let long = "é".repeat(80);
        let conversation = SavedConversation::from_chat(vec![Message::user(long)], None, 0);

        assert_eq!(conversation.title.chars().count(), 73);
        assert!(conversation.title.ends_with('…'));
    }

    #[test]
    fn snapshot_without_user_message_is_untitled() {
        let conversation = SavedConversation::from_chat(vec![Message::system("ready")], None, 0);

        assert_eq!(conversation.title, "Untitled conversation");
        assert!(
            conversation
                .messages
                .iter()
                .all(|message| message.role != MessageRole::User)
        );
    }
}
