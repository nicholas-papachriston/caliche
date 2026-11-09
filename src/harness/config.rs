//! Configuration types for the Agent Harness.

use crate::agent::ToolExecutionDriver;
use crate::harness::formatter::FormatterRule;
use crate::llm::ProviderType;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;

/// Configuration for the Agent Harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    /// Provider type to use
    #[serde(default)]
    pub provider: Option<ProviderType>,

    /// Model name to use
    #[serde(default)]
    pub model: Option<String>,

    /// Tools to enable (None = all default tools)
    #[serde(default)]
    pub tools: Option<Vec<String>>,

    /// Working directory for file operations
    #[serde(default)]
    pub workspace_root: Option<PathBuf>,

    /// Maximum turns per task
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,

    /// Maximum concurrent tool executions
    #[serde(default = "default_max_concurrent_tools")]
    pub max_concurrent_tools: usize,

    /// Enable automatic checkpointing before file writes
    #[serde(default = "default_auto_checkpoint")]
    pub auto_checkpoint: bool,

    /// Enable context tracking (file reads, git state, etc.)
    #[serde(default = "default_context_tracking")]
    pub context_tracking: bool,

    /// System prompt override
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Permission mode for tool execution
    #[serde(default)]
    pub permission_mode: PermissionModeConfig,

    /// Event streaming configuration
    #[serde(default)]
    pub event_config: EventConfig,

    /// Tool execution driver mode (optional override)
    #[serde(default)]
    pub tool_execution_driver: Option<ToolExecutionDriver>,

    /// Durable stream backend for event persistence (replay, resume). Not serialized.
    #[serde(skip)]
    pub durable_stream_backend: Option<Arc<dyn crate::streams::DurableStreamBackend>>,

    /// Key prefix for durable stream keys (e.g. "events:" -> "events:{session_id}").
    #[serde(skip)]
    pub durable_stream_key_prefix: String,

    /// Formatter rules to run after fs write/edit operations (CLI config, not serialized).
    #[serde(skip)]
    pub formatter_rules: Vec<FormatterRule>,

    /// Shell command to run after each agent turn completes.
    /// Receives turn metadata via CALICHE_* environment variables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notify_command: Option<String>,
}

fn default_max_turns() -> usize {
    50
}

fn default_max_concurrent_tools() -> usize {
    24
}

fn default_auto_checkpoint() -> bool {
    true
}

fn default_context_tracking() -> bool {
    true
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            provider: None,
            model: None,
            tools: None,
            workspace_root: None,
            max_turns: default_max_turns(),
            max_concurrent_tools: default_max_concurrent_tools(),
            auto_checkpoint: default_auto_checkpoint(),
            context_tracking: default_context_tracking(),
            system_prompt: None,
            permission_mode: PermissionModeConfig::default(),
            event_config: EventConfig::default(),
            tool_execution_driver: None,
            durable_stream_backend: None,
            durable_stream_key_prefix: "events:".to_string(),
            formatter_rules: vec![],
            notify_command: None,
        }
    }
}

impl HarnessConfig {
    /// Create a new config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider type
    pub fn with_provider(mut self, provider: ProviderType) -> Self {
        self.provider = Some(provider);
        self
    }

    /// Set the model name
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the tools to enable
    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Set the workspace root
    pub fn with_workspace(mut self, path: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(path.into());
        self
    }

    /// Set the maximum turns
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Set the system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Enable or disable auto-checkpointing
    pub fn with_auto_checkpoint(mut self, enabled: bool) -> Self {
        self.auto_checkpoint = enabled;
        self
    }

    pub fn with_tool_execution_driver(mut self, driver: ToolExecutionDriver) -> Self {
        self.tool_execution_driver = Some(driver);
        self
    }

    /// Set durable stream backend and key prefix for event persistence.
    pub fn with_durable_stream(
        mut self,
        backend: Arc<dyn crate::streams::DurableStreamBackend>,
        key_prefix: impl Into<String>,
    ) -> Self {
        self.durable_stream_backend = Some(backend);
        self.durable_stream_key_prefix = key_prefix.into();
        self
    }

    /// Set formatter rules to run after file writes/edits.
    pub fn with_formatters(mut self, rules: Vec<FormatterRule>) -> Self {
        self.formatter_rules = rules;
        self
    }

    /// Set the shell command to run after each agent turn completes.
    pub fn with_notify_command(mut self, cmd: impl Into<String>) -> Self {
        self.notify_command = Some(cmd.into());
        self
    }
}

/// Permission mode configuration
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionModeConfig {
    /// Default permission mode (prompt for dangerous operations)
    #[default]
    Default,
    /// Accept all operations without prompting
    AcceptAll,
    /// Accept file edits without prompting
    AcceptEdits,
    /// Deny all dangerous operations
    DenyAll,
    /// Interactive prompting for all operations
    Interactive,
}

/// Event streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventConfig {
    /// Include thinking events
    #[serde(default = "default_true")]
    pub include_thinking: bool,

    /// Include tool argument previews
    #[serde(default = "default_true")]
    pub include_tool_args: bool,

    /// Include file diff previews
    #[serde(default = "default_true")]
    pub include_diffs: bool,

    /// Maximum length for preview strings
    #[serde(default = "default_preview_length")]
    pub max_preview_length: usize,
}

fn default_true() -> bool {
    true
}

fn default_preview_length() -> usize {
    500
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            include_thinking: default_true(),
            include_tool_args: default_true(),
            include_diffs: default_true(),
            max_preview_length: default_preview_length(),
        }
    }
}

/// Configuration for a specific session
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionConfig {
    /// Session ID (auto-generated if not provided)
    #[serde(default)]
    pub session_id: Option<String>,

    /// Working directory for this session
    #[serde(default)]
    pub workspace: Option<PathBuf>,

    /// Initial context files to include
    #[serde(default)]
    pub context_files: Vec<PathBuf>,

    /// Session-specific system prompt addition
    #[serde(default)]
    pub system_prompt_addition: Option<String>,

    /// Maximum turns for this session (overrides harness config)
    #[serde(default)]
    pub max_turns: Option<usize>,

    /// Initial conversation history
    #[serde(default)]
    pub history: Vec<crate::messages::Message>,
}

impl SessionConfig {
    /// Create a new session config
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the session ID
    pub fn with_session_id(mut self, id: impl Into<String>) -> Self {
        self.session_id = Some(id.into());
        self
    }

    /// Set the workspace
    pub fn with_workspace(mut self, path: impl Into<PathBuf>) -> Self {
        self.workspace = Some(path.into());
        self
    }

    /// Add context files
    pub fn with_context_files(mut self, files: Vec<PathBuf>) -> Self {
        self.context_files = files;
        self
    }

    /// Set max turns for this session
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }
}

#[cfg(test)]
mod session_config_tests {
    use super::*;

    #[test]
    fn session_config_builders_and_empty_json_defaults() {
        let config = SessionConfig::new()
            .with_session_id("sess-1")
            .with_workspace("/tmp/ws")
            .with_max_turns(12)
            .with_context_files(vec![PathBuf::from("README.md")]);
        assert_eq!(config.session_id.as_deref(), Some("sess-1"));
        assert_eq!(config.workspace, Some(PathBuf::from("/tmp/ws")));
        assert_eq!(config.max_turns, Some(12));
        assert_eq!(config.context_files.len(), 1);

        let parsed: SessionConfig = serde_json::from_str("{}").expect("empty object defaults");
        assert!(parsed.session_id.is_none());
        assert!(parsed.history.is_empty());
        assert!(parsed.context_files.is_empty());
    }
}
