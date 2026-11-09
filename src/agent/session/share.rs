//! Redacted transcript sharing primitives.

use anyhow::{Context, Result, bail};
use caliche_contracts::sessions::ShareMessageDto;
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::messages::{Message, MessageRole};

const DEFAULT_SECRET_PATTERNS: &[&str] =
    &[r"sk-[A-Za-z0-9_-]{16,}", r"Bearer\s+[A-Za-z0-9._~+/=-]+"];

/// Product policy controlling whether share creation is allowed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SharePolicy {
    /// Reject all share creation.
    Disabled,
    /// Allow explicit share create/unshare (default).
    #[default]
    Manual,
    /// Reserved for automatic share-on-turn; currently behaves like [`Manual`].
    Auto,
}

/// Who may resolve a shared transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShareVisibility {
    /// Anyone with the slug URL may resolve the share.
    #[default]
    Public,
    /// Resolve requires an authenticated API principal.
    Authenticated,
}

/// Backing store for session shares.
///
/// Historical configs used `mode = "memory" | "remote"` for this field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ShareMode {
    /// Process-local (or directory-backed) store.
    #[default]
    Memory,
    /// External backend URL. API hosts can interpret this in a higher layer.
    Remote,
}

/// Configuration for creating redacted transcript shares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShareConfig {
    /// Storage backend selection.
    #[serde(default)]
    pub mode: ShareMode,
    /// Whether sharing is disabled / manual / auto.
    #[serde(default)]
    pub policy: SharePolicy,
    /// Default visibility applied when create requests omit one.
    #[serde(default)]
    pub default_visibility: ShareVisibility,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_url: Option<String>,
    #[serde(default)]
    pub redact_patterns: Vec<String>,
}

impl Default for ShareConfig {
    fn default() -> Self {
        Self {
            mode: ShareMode::Memory,
            policy: SharePolicy::Manual,
            default_visibility: ShareVisibility::Public,
            backend_url: None,
            redact_patterns: DEFAULT_SECRET_PATTERNS
                .iter()
                .map(|pattern| (*pattern).to_string())
                .collect(),
        }
    }
}

impl ShareConfig {
    /// Return an error when share creation is not allowed by policy.
    pub fn ensure_create_allowed(&self) -> Result<()> {
        match self.policy {
            SharePolicy::Disabled => bail!("session sharing is disabled by configuration"),
            SharePolicy::Manual | SharePolicy::Auto => Ok(()),
        }
    }

    /// Resolve effective visibility for a create request.
    #[must_use]
    pub fn resolve_visibility(&self, requested: Option<ShareVisibility>) -> ShareVisibility {
        requested.unwrap_or(self.default_visibility)
    }
}

/// Result of applying redaction to one transcript string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactionResult {
    pub text: String,
    pub count: usize,
}

/// Redacted transcript payload copied at share creation time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RedactedTranscript {
    pub messages: Vec<ShareMessageDto>,
    pub redaction_count: usize,
}

/// Regex-backed transcript redactor.
#[derive(Debug, Clone)]
pub struct TranscriptRedactor {
    patterns: Vec<Regex>,
}

impl TranscriptRedactor {
    /// Build a redactor from a share configuration.
    pub fn from_config(config: &ShareConfig) -> Result<Self> {
        Self::from_patterns(config.redact_patterns.iter().map(String::as_str))
    }

    /// Build a redactor from regex pattern strings.
    pub fn from_patterns<'a>(patterns: impl IntoIterator<Item = &'a str>) -> Result<Self> {
        let mut compiled = Vec::new();
        for pattern in patterns {
            compiled.push(
                Regex::new(pattern).with_context(|| format!("compile redact pattern {pattern}"))?,
            );
        }
        Ok(Self { patterns: compiled })
    }

    /// Build a redactor using Caliche's default secret patterns.
    pub fn with_default_patterns() -> Result<Self> {
        Self::from_patterns(DEFAULT_SECRET_PATTERNS.iter().copied())
    }

    /// Redact one text payload.
    #[must_use]
    pub fn redact_text(&self, input: &str) -> RedactionResult {
        let mut text = input.to_string();
        let mut count = 0usize;
        for pattern in &self.patterns {
            count = count.saturating_add(pattern.find_iter(&text).count());
            text = pattern.replace_all(&text, "[REDACTED]").into_owned();
        }
        RedactionResult { text, count }
    }

    /// Redact and copy conversation messages into share DTOs.
    ///
    /// Public shares stay text-only. This copies [`Message::extract_text`] and
    /// does not copy `metadata.reasoning_content`. Chain-of-thought stays on
    /// the local session (autosave, and export with `--include-thinking`) and
    /// is not published on a share link.
    #[must_use]
    pub fn redact_messages(&self, messages: &[Message]) -> RedactedTranscript {
        let mut redaction_count = 0usize;
        let messages = messages
            .iter()
            .map(|message| {
                let redacted = self.redact_text(&message.extract_text());
                redaction_count = redaction_count.saturating_add(redacted.count);
                ShareMessageDto {
                    role: message_role_name(&message.role).to_string(),
                    content: redacted.text,
                }
            })
            .collect();
        RedactedTranscript {
            messages,
            redaction_count,
        }
    }
}

const fn message_role_name(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_patterns_redact_openai_and_bearer_secrets() -> Result<()> {
        let redactor = TranscriptRedactor::with_default_patterns()?;

        let result = redactor.redact_text(&format!(
            "OPENAI_API_KEY=s{}-{} Bearer abc.def-ghi_jkl",
            "k", "abcdefghijklmnopqrstuvwxyz"
        ));

        assert_eq!(result.count, 2);
        assert_eq!(result.text, "OPENAI_API_KEY=[REDACTED] [REDACTED]");
        Ok(())
    }

    #[test]
    fn public_share_omits_reasoning_content() -> Result<()> {
        let redactor = TranscriptRedactor::with_default_patterns()?;
        let messages = vec![Message::assistant_with_reasoning(
            "visible answer",
            "hidden-cot-token",
        )];

        let transcript = redactor.redact_messages(&messages);
        let json = serde_json::to_string(&transcript.messages).expect("serialize share");

        assert_eq!(transcript.messages[0].content, "visible answer");
        assert!(!json.contains("hidden-cot-token"));
        assert!(!json.contains("reasoning_content"));
        Ok(())
    }

    #[test]
    fn transcript_redaction_preserves_roles_and_counts_replacements() -> Result<()> {
        let redactor = TranscriptRedactor::from_patterns([r"secret-[0-9]+"])?;
        let messages = vec![
            Message::system("setup"),
            Message::user("token secret-123"),
            Message::assistant("ok secret-456"),
        ];

        let transcript = redactor.redact_messages(&messages);

        assert_eq!(transcript.redaction_count, 2);
        assert_eq!(transcript.messages[0].role, "system");
        assert_eq!(transcript.messages[1].content, "token [REDACTED]");
        assert_eq!(transcript.messages[2].content, "ok [REDACTED]");
        Ok(())
    }

    #[test]
    fn disabled_policy_blocks_create() {
        let mut config = ShareConfig::default();
        config.policy = SharePolicy::Disabled;
        assert!(config.ensure_create_allowed().is_err());
    }

    #[test]
    fn resolve_visibility_prefers_request_over_default() {
        let mut config = ShareConfig::default();
        config.default_visibility = ShareVisibility::Authenticated;
        assert_eq!(
            config.resolve_visibility(Some(ShareVisibility::Public)),
            ShareVisibility::Public
        );
        assert_eq!(
            config.resolve_visibility(None),
            ShareVisibility::Authenticated
        );
    }
}
