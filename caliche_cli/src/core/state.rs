//! Unified interactive state shared across REPL and CLI modes.

use crate::cost::TokenBudget;
use crate::export::ConversationExport;
use crate::session::CliSession;
use caliche::agent::modes::{InteractionMode, PlanDocument};
use caliche::messages::{Message, MessageRole, TokenUsage};
use std::path::PathBuf;
use std::time::Instant;

/// Unified state for all interactive modes.
///
/// This state contains all the shared data that REPL and CLI modes need.
/// Mode-specific state (like REPL history) remains in their respective modules.
// Note: We manually implement Debug because KnowledgeStore doesn't implement it
pub struct InteractiveState {
    // Identity
    /// Session ID
    pub session_id: String,
    /// Working directory
    pub working_directory: PathBuf,
    /// Session data
    pub session: CliSession,

    // Mode
    /// Current interaction mode (Agent/Plan/Ask)
    pub interaction_mode: InteractionMode,
    /// Plan document (when in Plan mode)
    pub plan_document: Option<PlanDocument>,
    /// Original system prompt (before mode modifications)
    pub original_system_prompt: String,

    // Conversation
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Conversation export for saving
    pub conversation_export: ConversationExport,

    // Tracking
    /// Token budget tracking
    pub token_budget: TokenBudget,
    /// The window set with `/context-window` in this REPL, if any.
    pub context_window_override: Option<usize>,
    /// Tracked files for context
    pub tracked_files: Vec<PathBuf>,
    /// Total tokens used in session
    pub total_tokens: u64,
    /// Number of completed LLM turns in this session.
    pub turn_count: u32,
    /// Context window tokens (current context size)
    pub context_tokens: Option<u64>,
    /// Provider prompt-cache write tokens accumulated this session.
    pub cache_creation_tokens: u64,
    /// Provider prompt-cache read tokens accumulated this session.
    pub cache_read_tokens: u64,
    /// Prompt tokens accumulated this session (for cache hit ratio).
    pub session_prompt_tokens: u64,

    // Settings
    /// REPL / turn output verbosity (thinking, tools, cache detail).
    pub output_verbosity: caliche::observability::OutputVerbosity,
    /// Whether autosave is enabled
    pub autosave_enabled: bool,
    /// Whether to save session on exit
    pub save_session: bool,
    /// Last autosave timestamp
    pub last_autosave: Instant,
    /// Whether auto-compact is enabled
    pub auto_compact_enabled: bool,
    /// Auto-compact threshold percentage (default: 80%)
    pub auto_compact_threshold: f64,

    /// Optional provider for subagents (research, delegated agents). If set, persists across sessions.
    pub subagent_provider: Option<String>,
    /// Optional model for subagents. If set, persists across sessions.
    pub subagent_model: Option<String>,
    /// Optional provider for embeddings. If set, persists across sessions.
    pub embedding_provider: Option<String>,
    /// Optional model for embeddings. If set, persists across sessions.
    pub embedding_model: Option<String>,
    /// Optional provider for image generation. If set, persists across sessions.
    pub image_generation_provider: Option<String>,
    /// Optional model for image generation. If set, persists across sessions.
    pub image_generation_model: Option<String>,
    /// Optional generated artifact directory override. If set, persists across sessions.
    pub artifact_dir: Option<PathBuf>,
    /// Optional primary agent profile name for the next message(s). When set, the next turn uses this profile instead of the default agent.
    pub current_agent_override: Option<String>,

    // Undo system
    /// Undo states for conversation rollback
    undo_states: Vec<UndoState>,
    /// Maximum number of undo states to keep
    max_undo_states: usize,

    // Knowledge store
    /// Shared knowledge store for cross-session memory
    // Note: KnowledgeStore doesn't implement Debug, so we can't derive Debug for this field
    // We'll skip it in Debug output
    pub knowledge_store: Option<crate::core::knowledge_store::SharedKnowledgeStore>,

    // Media attachments queued for the next message (via /attach)
    /// Pending media attachments to include with the next send
    pub pending_attachments: Vec<caliche::messages::Attachment>,

    // File tracking
    /// Files modified during this session
    pub modified_files: std::collections::HashMap<PathBuf, FileChange>,
    /// Session quality/accuracy counters for REPL UX and reliability.
    pub quality_metrics: ReplQualityMetrics,
}

impl std::fmt::Debug for InteractiveState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveState")
            .field("session_id", &self.session_id)
            .field("working_directory", &self.working_directory)
            .field("interaction_mode", &self.interaction_mode)
            .field("messages", &self.messages.len())
            .field("token_budget", &self.token_budget)
            .field("total_tokens", &self.total_tokens)
            .field("turn_count", &self.turn_count)
            .field("context_tokens", &self.context_tokens)
            .field("output_verbosity", &self.output_verbosity)
            .field("autosave_enabled", &self.autosave_enabled)
            .field("auto_compact_enabled", &self.auto_compact_enabled)
            .field("auto_compact_threshold", &self.auto_compact_threshold)
            .field("subagent_provider", &self.subagent_provider)
            .field("subagent_model", &self.subagent_model)
            .field("embedding_provider", &self.embedding_provider)
            .field("embedding_model", &self.embedding_model)
            .field("image_generation_provider", &self.image_generation_provider)
            .field("image_generation_model", &self.image_generation_model)
            .field("artifact_dir", &self.artifact_dir)
            .field("current_agent_override", &self.current_agent_override)
            .field(
                "knowledge_store",
                &self.knowledge_store.as_ref().map(|_| "<KnowledgeStore>"),
            )
            .field("modified_files", &self.modified_files.len())
            .field("quality_metrics", &self.quality_metrics)
            .finish()
    }
}

/// Session quality counters surfaced in `/stats` and `/status`.
#[derive(Debug, Clone, Default)]
pub struct ReplQualityMetrics {
    pub narration_only_responses: u64,
    pub tool_calls_started: u64,
    pub tool_calls_succeeded: u64,
    pub tool_calls_failed: u64,
    pub tool_retries_attempted: u64,
    pub tool_retries_succeeded: u64,
    pub text_loop_detections: u64,
    pub tool_loop_detections: u64,
    pub cancellations: u64,
    pub iteration_limits_reached: u64,
}

impl ReplQualityMetrics {
    pub fn tool_success_rate(&self) -> f64 {
        if self.tool_calls_started == 0 {
            return 0.0;
        }
        (self.tool_calls_succeeded as f64 / self.tool_calls_started as f64) * 100.0
    }
}

/// Information about a file change
#[derive(Debug, Clone)]
pub struct FileChange {
    /// Path to the file
    pub path: PathBuf,
    /// Type of change
    pub change_type: FileChangeType,
    /// Original content (if available)
    pub original_content: Option<String>,
    /// New content (if available)
    pub new_content: Option<String>,
    /// When the change occurred
    pub timestamp: Instant,
}

/// Type of file change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileChangeType {
    Created,
    Modified,
    Deleted,
}

/// State snapshot for undo functionality
#[derive(Clone, Debug)]
pub struct UndoState {
    /// Message count at this state
    pub message_count: usize,
    /// Token count at this state
    pub token_count: u32,
    /// Timestamp of this state
    pub timestamp: Instant,
}

impl InteractiveState {
    /// Create a new interactive state
    pub fn new(session: CliSession, provider: &str, model: &str, system_prompt: String) -> Self {
        let working_directory = session.working_directory.clone();
        let session_id = session.id.clone();
        let session_id_for_export = session_id.clone();
        let messages = session.conversation_history.clone();
        let total_tokens = session.total_tokens;
        let turn_count = u32::try_from(
            messages
                .iter()
                .filter(|message| message.role == MessageRole::User)
                .count(),
        )
        .unwrap_or(u32::MAX);
        let mut token_budget = TokenBudget::new(provider, model);
        token_budget.set_used(total_tokens);

        Self {
            session_id,
            working_directory,
            session,
            interaction_mode: InteractionMode::Agent,
            plan_document: None,
            original_system_prompt: system_prompt,
            messages,
            conversation_export: ConversationExport::new(&session_id_for_export, provider, model),
            token_budget,
            context_window_override: None,
            tracked_files: Vec::new(),
            total_tokens,
            turn_count,
            context_tokens: None,
            cache_creation_tokens: 0,
            cache_read_tokens: 0,
            session_prompt_tokens: 0,
            output_verbosity: caliche::observability::OutputVerbosity::Normal,
            autosave_enabled: true,
            save_session: false,
            last_autosave: Instant::now(),
            auto_compact_enabled: true,
            auto_compact_threshold: 80.0,
            subagent_provider: None,
            subagent_model: None,
            embedding_provider: None,
            embedding_model: None,
            image_generation_provider: None,
            image_generation_model: None,
            artifact_dir: None,
            current_agent_override: None,
            undo_states: Vec::new(),
            max_undo_states: 50,
            knowledge_store: None,
            pending_attachments: Vec::new(),
            modified_files: std::collections::HashMap::new(),
            quality_metrics: ReplQualityMetrics::default(),
        }
    }

    /// Set the interaction mode
    pub fn set_interaction_mode(&mut self, mode: InteractionMode) {
        self.interaction_mode = mode;
    }

    /// Toggle compact (quiet) output mode
    pub fn toggle_compact(&mut self) {
        self.output_verbosity = self.output_verbosity.toggle_quiet();
    }

    /// Toggle verbose output mode
    pub fn toggle_verbose(&mut self) {
        self.output_verbosity = self.output_verbosity.toggle_verbose();
    }

    /// Set output verbosity from a slash-command or CLI value.
    pub fn set_output_verbosity(&mut self, verbosity: caliche::observability::OutputVerbosity) {
        self.output_verbosity = verbosity;
    }

    /// Add a message to the conversation
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Update token usage
    pub fn update_token_usage(&mut self, usage: &TokenUsage) {
        let input = usage.prompt_tokens.unwrap_or(0) as u64;
        let output = usage.completion_tokens.unwrap_or(0) as u64;
        let total = input + output;
        self.total_tokens += total;
        self.turn_count += 1;
        self.token_budget.add_usage(total);
        self.context_tokens = usage.total_tokens.map(|t| t as u64);
        self.cache_creation_tokens = self
            .cache_creation_tokens
            .saturating_add(u64::from(usage.cache_creation_tokens.unwrap_or(0)));
        self.cache_read_tokens = self
            .cache_read_tokens
            .saturating_add(u64::from(usage.cache_read_tokens.unwrap_or(0)));
        self.session_prompt_tokens = self.session_prompt_tokens.saturating_add(input);
    }

    /// Save an undo state
    pub fn save_undo_state(&mut self) {
        let state = UndoState {
            message_count: self.messages.len(),
            token_count: self.total_tokens as u32,
            timestamp: Instant::now(),
        };
        self.undo_states.push(state);

        // Trim if over limit
        if self.undo_states.len() > self.max_undo_states {
            self.undo_states.remove(0);
        }
    }

    /// Pop the last undo state
    pub fn pop_undo_state(&mut self) -> Option<UndoState> {
        self.undo_states.pop()
    }

    /// Check if undo is available
    pub fn can_undo(&self) -> bool {
        !self.undo_states.is_empty()
    }

    /// Check if autosave should run (debounced)
    pub fn should_autosave(&self, debounce_secs: u64) -> bool {
        if !self.autosave_enabled {
            return false;
        }
        self.last_autosave.elapsed().as_secs() >= debounce_secs
    }

    /// Mark autosave as completed
    pub fn mark_autosaved(&mut self) {
        self.last_autosave = Instant::now();
    }

    /// Clear conversation history
    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.conversation_export = ConversationExport::new(
            &self.session_id,
            &self.token_budget.provider,
            &self.token_budget.model,
        );
    }

    /// Check if auto-compact should trigger based on context usage
    pub fn should_auto_compact(&self) -> bool {
        if !self.auto_compact_enabled {
            return false;
        }
        // Use context_tokens if available, otherwise fall back to token_budget
        let current_tokens = self.context_tokens.unwrap_or(self.token_budget.used_tokens);
        let max_tokens = self.token_budget.max_tokens;
        if max_tokens == 0 {
            return false;
        }
        let usage_percent = (current_tokens as f64 / max_tokens as f64) * 100.0;
        usage_percent >= self.auto_compact_threshold
    }

    /// Toggle auto-compact
    pub fn toggle_auto_compact(&mut self) {
        self.auto_compact_enabled = !self.auto_compact_enabled;
    }

    /// Set auto-compact threshold
    pub fn set_auto_compact_threshold(&mut self, threshold: f64) {
        self.auto_compact_threshold = threshold.clamp(50.0, 99.0);
    }

    /// Record a file change
    pub fn record_file_change(
        &mut self,
        path: PathBuf,
        change_type: FileChangeType,
        original_content: Option<String>,
        new_content: Option<String>,
    ) {
        self.modified_files.insert(
            path.clone(),
            FileChange {
                path,
                change_type,
                original_content,
                new_content,
                timestamp: Instant::now(),
            },
        );
    }

    /// Get all modified files
    pub fn get_modified_files(&self) -> Vec<&FileChange> {
        self.modified_files.values().collect()
    }

    /// Clear file change tracking
    pub fn clear_file_changes(&mut self) {
        self.modified_files.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn new_loads_saved_transcript_into_the_active_conversation() {
        let mut session = CliSession::new(
            "01a10958-0c28-7720-8d8d-ccc32c0ae47d".to_string(),
            PathBuf::from("/tmp/workspace"),
        );
        session.conversation_history = vec![
            Message::user("Please rename parser.rs"),
            Message::assistant("Renamed parser.rs with the fs tool"),
        ];
        session.total_tokens = 42;

        let state = InteractiveState::new(
            session,
            "anthropic",
            "claude-haiku-4-5-20251001",
            "system".to_string(),
        );

        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.turn_count, 1);
        assert_eq!(state.total_tokens, 42);
        assert!(
            state.messages[1]
                .extract_text()
                .contains("Renamed parser.rs with the fs tool")
        );
    }
}
