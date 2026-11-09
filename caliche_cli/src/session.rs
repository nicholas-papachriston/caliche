//! CLI session persistence for resuming conversations.

use caliche::messages::{Message, REASONING_CONTENT_METADATA_KEY};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// CLI session data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliSession {
    pub id: String,
    /// Optional human-readable name for the session
    #[serde(default)]
    pub name: Option<String>,
    pub working_directory: PathBuf,
    pub conversation_history: Vec<Message>,
    pub context_files: Vec<PathBuf>,
    pub permissions_granted: Vec<String>,
    /// When the session was created
    #[serde(default = "Utc::now")]
    pub created_at: DateTime<Utc>,
    /// When the session was last updated
    #[serde(default = "Utc::now")]
    pub updated_at: DateTime<Utc>,
    /// Total tokens used in this session
    #[serde(default)]
    pub total_tokens: u64,
    /// Provider used for this session
    #[serde(default)]
    pub provider: Option<String>,
    /// Model used for this session
    #[serde(default)]
    pub model: Option<String>,
    /// Cache of recently read file paths -> (mtime, line_count). Not persisted.
    #[serde(skip, default)]
    pub file_content_cache: HashMap<PathBuf, (std::time::SystemTime, u64)>,
    /// Files created during this session (for doc-sprawl limits).
    #[serde(default)]
    pub files_created: Vec<PathBuf>,
}

/// Session summary for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub name: Option<String>,
    pub working_directory: PathBuf,
    pub message_count: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub total_tokens: u64,
    pub provider: Option<String>,
    pub model: Option<String>,
}

/// Aggregate statistics across all sessions
#[derive(Debug, Clone, Default)]
pub struct AggregateStats {
    /// Total number of sessions
    pub total_sessions: usize,
    /// Total tokens used across all sessions
    pub total_tokens: u64,
    /// Total messages across all sessions
    pub total_messages: usize,
    /// Provider usage: provider name -> (session_count, total_tokens)
    pub provider_usage: HashMap<String, (usize, u64)>,
    /// Model usage: model name -> (session_count, total_tokens)
    pub model_usage: HashMap<String, (usize, u64)>,
    /// Earliest session date
    pub earliest_session: Option<DateTime<Utc>>,
    /// Latest session date
    pub latest_session: Option<DateTime<Utc>>,
}

impl AggregateStats {
    /// Get the date range as a human-readable string
    pub fn date_range(&self) -> String {
        match (self.earliest_session, self.latest_session) {
            (Some(earliest), Some(latest)) => {
                let days = (latest - earliest).num_days();
                format!(
                    "{} to {} ({} days)",
                    earliest.format("%Y-%m-%d"),
                    latest.format("%Y-%m-%d"),
                    days
                )
            }
            _ => "No sessions".to_string(),
        }
    }

    /// Get providers sorted by token usage (descending)
    pub fn providers_by_usage(&self) -> Vec<(&String, &(usize, u64))> {
        let mut providers: Vec<_> = self.provider_usage.iter().collect();
        providers.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        providers
    }

    /// Get models sorted by token usage (descending)
    pub fn models_by_usage(&self) -> Vec<(&String, &(usize, u64))> {
        let mut models: Vec<_> = self.model_usage.iter().collect();
        models.sort_by(|a, b| b.1.1.cmp(&a.1.1));
        models
    }
}

impl CliSession {
    /// Create a new session
    pub fn new(id: String, working_directory: PathBuf) -> Self {
        let now = Utc::now();
        Self {
            id,
            name: None,
            working_directory,
            conversation_history: Vec::new(),
            context_files: Vec::new(),
            permissions_granted: Vec::new(),
            created_at: now,
            updated_at: now,
            total_tokens: 0,
            provider: None,
            model: None,
            file_content_cache: HashMap::new(),
            files_created: Vec::new(),
        }
    }

    /// Create a new session with a name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the provider and model for this session
    pub fn with_provider(mut self, provider: impl Into<String>, model: Option<String>) -> Self {
        self.provider = Some(provider.into());
        self.model = model;
        self
    }

    // Note: All persistence methods (save, load, autosave, etc.) have been removed.
    // Use SessionStore from storage::sessions for all session persistence.
    // SessionManager in core::session provides a unified interface.

    /// Add tokens to the session total
    pub fn add_tokens(&mut self, tokens: u64) {
        self.total_tokens += tokens;
    }

    /// Get a summary of this session
    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            id: self.id.clone(),
            name: self.name.clone(),
            working_directory: self.working_directory.clone(),
            message_count: self.conversation_history.len(),
            created_at: self.created_at,
            updated_at: self.updated_at,
            total_tokens: self.total_tokens,
            provider: self.provider.clone(),
            model: self.model.clone(),
        }
    }

    /// Export session to markdown format. Thinking is omitted.
    pub fn to_markdown(&self) -> String {
        self.to_markdown_with(false)
    }

    /// Render a session export.
    ///
    /// `include_thinking` keeps `metadata.reasoning_content`. The default
    /// omits it so a file you might hand off stays text-only. Public share
    /// links never include thinking; use the flag for local review.
    ///
    /// # Errors
    ///
    /// Returns an error when `format` is not `json`, `markdown`, or `md`, or
    /// when JSON serialization fails.
    pub fn export_text(&self, format: &str, include_thinking: bool) -> anyhow::Result<String> {
        match format {
            "json" => {
                let session = self.export_view(include_thinking);
                Ok(serde_json::to_string_pretty(&session)?)
            }
            "markdown" | "md" => Ok(self.to_markdown_with(include_thinking)),
            _ => anyhow::bail!("Unknown format: {format}. Use 'json' or 'markdown'"),
        }
    }

    fn export_view(&self, include_thinking: bool) -> Self {
        let mut session = self.clone();
        if !include_thinking {
            for message in &mut session.conversation_history {
                strip_reasoning_content(message);
            }
        }
        session
    }

    fn to_markdown_with(&self, include_thinking: bool) -> String {
        let mut md = String::new();

        // Header
        let name = self.name.as_deref().unwrap_or(&self.id);
        md.push_str(&format!("# Session: {}\n\n", name));

        // Metadata
        md.push_str("## Metadata\n\n");
        md.push_str(&format!("- **ID**: {}\n", self.id));
        md.push_str(&format!(
            "- **Working Directory**: {}\n",
            self.working_directory.display()
        ));
        md.push_str(&format!("- **Created**: {}\n", self.created_at));
        md.push_str(&format!("- **Updated**: {}\n", self.updated_at));
        md.push_str(&format!("- **Total Tokens**: {}\n", self.total_tokens));
        if let Some(ref provider) = self.provider {
            md.push_str(&format!("- **Provider**: {}\n", provider));
        }
        if let Some(ref model) = self.model {
            md.push_str(&format!("- **Model**: {}\n", model));
        }
        md.push('\n');

        // Conversation
        md.push_str("## Conversation\n\n");
        for msg in &self.conversation_history {
            let role = format!("{:?}", msg.role);
            md.push_str(&format!("### {}\n\n", role));
            md.push_str(&msg.content.extract_text());
            if include_thinking && let Some(reasoning) = msg.reasoning_content() {
                md.push_str("\n\n**Thinking**\n\n");
                md.push_str(reasoning);
            }
            md.push_str("\n\n");
        }

        md
    }
}

fn strip_reasoning_content(message: &mut Message) {
    let Some(mut meta) = message.metadata.take() else {
        return;
    };
    if let Some(obj) = meta.as_object_mut() {
        obj.remove(REASONING_CONTENT_METADATA_KEY);
    }
    if meta.as_object().is_some_and(|obj| !obj.is_empty()) || !meta.is_object() {
        message.metadata = Some(meta);
    }
}

impl std::fmt::Display for SessionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = self.name.as_deref().unwrap_or("-");
        let id_short = caliche_contracts::text::truncate_utf8(&self.id, 8);

        write!(
            f,
            "{:<10} {:<20} {:>5} msgs  {:>8} tokens  {}",
            id_short,
            name,
            self.message_count,
            self.total_tokens,
            self.updated_at.format("%Y-%m-%d %H:%M")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::CliSession;
    use caliche::messages::Message;
    use std::path::PathBuf;

    fn session_with_reasoning() -> CliSession {
        let mut session = CliSession::new("sess-1".to_string(), PathBuf::from("/tmp/work"));
        session
            .conversation_history
            .push(Message::assistant_with_reasoning(
                "the answer",
                "hidden chain of thought",
            ));
        session
    }

    #[test]
    fn export_omits_thinking_unless_requested() {
        let session = session_with_reasoning();

        let plain = session.export_text("json", false).expect("json");
        assert!(!plain.contains("hidden chain of thought"));
        assert!(!plain.contains("reasoning_content"));

        let with_thinking = session.export_text("json", true).expect("json");
        assert!(with_thinking.contains("hidden chain of thought"));

        let markdown = session.export_text("markdown", false).expect("markdown");
        assert!(markdown.contains("the answer"));
        assert!(!markdown.contains("hidden chain of thought"));

        let markdown_thinking = session.export_text("md", true).expect("markdown");
        assert!(markdown_thinking.contains("**Thinking**"));
        assert!(markdown_thinking.contains("hidden chain of thought"));
    }
}
