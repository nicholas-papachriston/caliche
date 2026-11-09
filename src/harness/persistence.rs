//! Session persistence for saving and restoring agent state.
//!
//! This module provides functionality for:
//! - Exporting conversations to JSON/YAML
//! - Importing saved conversations
//! - Session state persistence

use crate::harness::state::Checkpoint;
use crate::messages::{Message, TokenUsage};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Exported conversation data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationExport {
    /// Export format version
    pub version: String,
    /// Session ID
    pub session_id: String,
    /// Timestamp of export
    pub exported_at: chrono::DateTime<chrono::Utc>,
    /// Workspace root (relative or absolute)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workspace: Option<PathBuf>,
    /// Conversation messages
    pub messages: Vec<ExportedMessage>,
    /// Total token usage
    pub total_usage: TokenUsage,
    /// Checkpoints created during session
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub checkpoints: Vec<ExportedCheckpoint>,
    /// Files modified during session
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub modified_files: Vec<PathBuf>,
    /// Metadata
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Exported message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedMessage {
    /// Role: "user", "assistant", "system"
    pub role: String,
    /// Message content
    pub content: String,
    /// Timestamp if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
    /// Tool calls if any
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ExportedToolCall>,
}

impl From<&Message> for ExportedMessage {
    fn from(msg: &Message) -> Self {
        Self {
            role: format!("{:?}", msg.role).to_lowercase(),
            content: msg.content.as_text().unwrap_or_default().to_string(),
            timestamp: None,
            tool_calls: Vec::new(),
        }
    }
}

/// Exported tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    pub success: bool,
}

/// Exported checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportedCheckpoint {
    pub id: String,
    pub label: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub file_count: usize,
    /// Turn number captured with the checkpoint (0 when unknown).
    #[serde(default)]
    pub turn: usize,
}

impl From<&Checkpoint> for ExportedCheckpoint {
    fn from(cp: &Checkpoint) -> Self {
        Self {
            id: cp.id.to_string(),
            label: cp.label.clone(),
            timestamp: cp.timestamp,
            file_count: cp.file_versions.len(),
            turn: cp.turn,
        }
    }
}

impl ConversationExport {
    /// Create a new conversation export
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            version: "1.0".to_string(),
            session_id: session_id.into(),
            exported_at: chrono::Utc::now(),
            workspace: None,
            messages: Vec::new(),
            total_usage: TokenUsage::default(),
            checkpoints: Vec::new(),
            modified_files: Vec::new(),
            metadata: None,
        }
    }

    /// Set workspace
    pub fn with_workspace(mut self, workspace: PathBuf) -> Self {
        self.workspace = Some(workspace);
        self
    }

    /// Set messages
    pub fn with_messages(mut self, messages: &[Message]) -> Self {
        self.messages = messages.iter().map(ExportedMessage::from).collect();
        self
    }

    /// Set total usage
    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.total_usage = usage;
        self
    }

    /// Set checkpoints
    pub fn with_checkpoints(mut self, checkpoints: &[Checkpoint]) -> Self {
        self.checkpoints = checkpoints.iter().map(ExportedCheckpoint::from).collect();
        self
    }

    /// Set modified files
    pub fn with_modified_files(mut self, files: Vec<PathBuf>) -> Self {
        self.modified_files = files;
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Export to JSON string
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).context("Failed to serialize conversation to JSON")
    }

    /// Export to YAML string
    pub fn to_yaml(&self) -> Result<String> {
        serde_norway::to_string(self).context("Failed to serialize conversation to YAML")
    }

    /// Parse from JSON string
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("Failed to parse conversation from JSON")
    }

    /// Parse from YAML string
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        serde_norway::from_str(yaml).context("Failed to parse conversation from YAML")
    }

    /// Save to file (auto-detects format from extension)
    pub fn save_to_file(&self, path: &Path) -> Result<()> {
        let content = if path
            .extension()
            .map(|e| e == "yaml" || e == "yml")
            .unwrap_or(false)
        {
            self.to_yaml()?
        } else {
            self.to_json()?
        };

        std::fs::write(path, content).context("Failed to write conversation file")
    }

    /// Load from file (auto-detects format from extension)
    pub fn load_from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).context("Failed to read conversation file")?;

        if path
            .extension()
            .map(|e| e == "yaml" || e == "yml")
            .unwrap_or(false)
        {
            Self::from_yaml(&content)
        } else {
            Self::from_json(&content)
        }
    }

    /// Convert messages back to Message format for import
    pub fn to_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .map(|m| match m.role.as_str() {
                "user" => Message::user(&m.content),
                "assistant" => Message::assistant(&m.content),
                "system" => Message::system(&m.content),
                _ => Message::user(&m.content),
            })
            .collect()
    }
}

/// Session persistence manager
#[derive(Debug)]
pub struct SessionPersistence {
    /// Directory for storing sessions
    sessions_dir: PathBuf,
    /// Auto-save interval (if enabled)
    auto_save_interval: Option<std::time::Duration>,
}

impl SessionPersistence {
    /// Create a new session persistence manager
    pub fn new(sessions_dir: PathBuf) -> Self {
        Self {
            sessions_dir,
            auto_save_interval: None,
        }
    }

    /// Enable auto-save with specified interval
    pub fn with_auto_save(mut self, interval: std::time::Duration) -> Self {
        self.auto_save_interval = Some(interval);
        self
    }

    /// Ensure sessions directory exists
    pub fn ensure_dir(&self) -> Result<()> {
        std::fs::create_dir_all(&self.sessions_dir).context("Failed to create sessions directory")
    }

    /// Get path for a session file
    pub fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir.join(format!("{}.json", session_id))
    }

    /// Save a conversation export
    pub fn save(&self, export: &ConversationExport) -> Result<PathBuf> {
        self.ensure_dir()?;
        let path = self.session_path(&export.session_id);
        export.save_to_file(&path)?;
        Ok(path)
    }

    /// Load a conversation by session ID
    pub fn load(&self, session_id: &str) -> Result<ConversationExport> {
        let path = self.session_path(session_id);
        ConversationExport::load_from_file(&path)
    }

    /// List all saved sessions
    pub fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        self.ensure_dir()?;

        let mut sessions = Vec::new();

        for entry in std::fs::read_dir(&self.sessions_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path
                .extension()
                .map(|e| e == "json" || e == "yaml" || e == "yml")
                .unwrap_or(false)
                && let Ok(export) = ConversationExport::load_from_file(&path)
            {
                sessions.push(SessionInfo {
                    session_id: export.session_id,
                    exported_at: export.exported_at,
                    message_count: export.messages.len(),
                    total_tokens: export.total_usage.total_tokens.unwrap_or(0),
                    path,
                });
            }
        }

        // Sort by export time, newest first
        sessions.sort_by(|a, b| b.exported_at.cmp(&a.exported_at));

        Ok(sessions)
    }

    /// Delete a saved session
    pub fn delete(&self, session_id: &str) -> Result<()> {
        let path = self.session_path(session_id);
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }
}

/// Info about a saved session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub session_id: String,
    pub exported_at: chrono::DateTime<chrono::Utc>,
    pub message_count: usize,
    pub total_tokens: u32,
    pub path: PathBuf,
}
