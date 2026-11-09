//! Core Agent Harness implementation.
//!
//! The AgentHarness wraps AgentService with enhanced capabilities for
//! event streaming, context management, and state tracking.

use crate::agent::hooks::HookMatcher;
use crate::agent::service::AgentService;
use crate::harness::command_ledger::{CommandLedger, CommandLedgerHook};
use crate::harness::config::{HarnessConfig, PermissionModeConfig};
use crate::harness::context::ContextManager;
use crate::harness::diff_tracker::DiffTracker;
use crate::harness::events::EventStream;
use crate::harness::formatter::FormatterRule;
use crate::harness::hooks::ToolHooks;
use crate::harness::metrics::HarnessDisciplineMetrics;
use crate::harness::session_diff_hook::SessionDiffHook;
use crate::harness::state::StateManager;
use crate::harness::suspend::SessionPhase;
use crate::messages::{Message, TokenUsage};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Strip shell-metacharacters and control bytes from notify prompt previews.
///
/// The value is injected into a child environment; treat it as data, never as
/// part of the command string.
#[must_use]
pub fn sanitize_notify_preview(value: &str) -> String {
    value
        .chars()
        .map(|ch| match ch {
            '\n' | '\r' | '$' | '`' => ' ',
            ch if ch.is_control() => ' ',
            ch => ch,
        })
        .take(100)
        .collect()
}

/// Result of a task execution
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Final response from the agent
    pub response: String,
    /// Total token usage across all turns
    pub total_usage: TokenUsage,
    /// Number of turns executed
    pub turns: usize,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Files modified during the task
    pub files_modified: Vec<PathBuf>,
    /// Checkpoint ID for the task (if auto-checkpoint is enabled)
    pub checkpoint_id: Option<String>,
    /// Tokens/turn, tool-call, and estimated-cost discipline metrics.
    pub discipline: HarnessDisciplineMetrics,
}

/// Result of a single turn
#[derive(Debug, Clone)]
pub struct HarnessTurnResult {
    /// Response from this turn
    pub response: String,
    /// Token usage for this turn
    pub usage: TokenUsage,
    /// Whether tools were used
    pub tools_used: bool,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Whether the agent indicates task completion
    pub task_complete: bool,
    /// True when the inner chat loop stopped on its iteration cap.
    pub limit_reached: bool,
    /// Tokens/turn, tool-call, and estimated-cost discipline metrics.
    pub discipline: HarnessDisciplineMetrics,
    /// Stable inbox id for this turn (generated when the caller omitted one).
    pub input_id: Option<String>,
    /// Provider chain-of-thought for this turn.
    ///
    /// Live service memory stays text-only, so this is not copied into the
    /// next model request. Callers that persist a review transcript read it
    /// from here.
    pub reasoning_content: String,
}

/// Agent Harness - unified execution layer for agentic workflows.
///
/// The harness provides:
/// - Event streaming for real-time observability
/// - Context management for workspace awareness
/// - State management with undo/rollback
/// - Parallel tool execution with progress tracking
pub struct AgentHarness {
    /// Underlying agent service
    pub(crate) service: AgentService,
    /// Configuration
    pub(crate) config: HarnessConfig,
    /// Event stream for broadcasting events
    pub(crate) event_stream: EventStream,
    /// Context manager for workspace awareness
    pub(crate) context_manager: Arc<RwLock<ContextManager>>,
    /// State manager for checkpointing and rollback
    pub(crate) state_manager: Arc<RwLock<StateManager>>,
    /// Per-message diff tracker for undo/redo
    pub(crate) diff_tracker: Arc<RwLock<DiffTracker>>,
    /// Current session ID
    pub(crate) session_id: String,
    /// Current turn count
    pub(crate) turn_count: usize,
    /// Accumulated token usage
    pub(crate) total_usage: TokenUsage,
    /// Accumulated tool invocations across turns
    pub(crate) total_tool_calls: usize,
    /// Accumulated context-window compaction events across turns
    pub(crate) total_compactions: usize,
    /// Accumulated readonly / allowed-tools filter denials across turns
    pub(crate) total_readonly_filter_hits: usize,
    /// Conversation history for multi-turn
    pub(crate) history: Vec<Message>,
    /// Logical session phase restored by durable suspend/resume.
    pub(crate) phase: SessionPhase,
    /// Last checkpoint id created in this session, if any.
    pub(crate) last_checkpoint_id: Option<String>,
    /// Shell command to run after each turn (fire-and-forget)
    pub(crate) notify_command: Option<String>,
    /// Last edit or bash command, used to skip duplicate coding verification.
    pub(crate) command_ledger: Arc<RwLock<CommandLedger>>,
    /// Inbox + control flags behind a mutex so WhenIdle/Hard/Heartbeat
    /// can land while a turn holds the harness write lock.
    pub(crate) side: std::sync::Arc<parking_lot::Mutex<SessionSideState>>,
    /// Compaction reports accumulated this session (Jev + LLM).
    pub(crate) compaction_reports: Vec<crate::messages::CompactionReport>,
}

/// Inbox and control-plane state that can mutate independently of a turn write lock.
#[derive(Debug, Clone, Default)]
pub(crate) struct SessionSideState {
    pub inbox: crate::harness::inbox::SessionInbox,
    pub hard_stopped: bool,
    pub pending_when_idle: bool,
    pub last_heartbeat: Option<String>,
    pub inbox_settings: crate::harness::inbox::InboxSettings,
    pub suspending: bool,
    /// UUID v7 for the open thinking segment, when the stream chunk omitted one.
    pub thinking_action_id: Option<String>,
}

pub struct SessionSummary {
    pub session_id: String,
    pub turn_count: usize,
    pub total_tokens: u32,
    pub workspace: Option<PathBuf>,
    pub project_type: Option<crate::harness::context::ProjectType>,
    pub git_branch: Option<String>,
    pub files_tracked: usize,
    pub files_modified: usize,
    pub checkpoints: usize,
    pub history_messages: usize,
}

/// Register session-diff and tool-event hooks on the harness agent.
///
/// Tool hooks must be attached here: `run_turn` uses `AgentService::chat`,
/// which never calls `ToolHooks` unless they are registered as lifecycle hooks.
pub(crate) fn register_harness_lifecycle_hooks(
    service: &mut AgentService,
    event_stream: &EventStream,
    state_manager: &Arc<RwLock<StateManager>>,
    diff_tracker: &Arc<RwLock<DiffTracker>>,
    workspace_root: PathBuf,
    formatter_rules: &[FormatterRule],
    command_ledger: Arc<RwLock<CommandLedger>>,
) {
    let session_diff_hook =
        SessionDiffHook::new(Arc::clone(diff_tracker)).with_workspace_root(workspace_root.clone());
    service
        .agent_mut()
        .register_hook(HookMatcher::All, Arc::new(session_diff_hook));

    let command_ledger_hook = CommandLedgerHook::new(command_ledger, workspace_root);
    service
        .agent_mut()
        .register_hook(HookMatcher::All, Arc::new(command_ledger_hook));

    let action_ids = Arc::clone(&service.agent().action_ids);
    let tool_hooks = ToolHooks::new(event_stream.clone(), Arc::clone(state_manager))
        .with_formatters(formatter_rules.to_vec())
        .with_action_ids(action_ids);
    service
        .agent_mut()
        .register_hook(HookMatcher::All, Arc::new(tool_hooks));
}

/// Apply a permission mode config to an `AgentService`, mapping the harness-level
/// enum to the agent-level `PermissionMode`. The `Default` variant is a no-op
/// (the service keeps its built-in default).
pub(crate) fn apply_permission_mode(service: &mut AgentService, mode: PermissionModeConfig) {
    use crate::agent::hooks::permission::PermissionMode;
    let pm = match mode {
        PermissionModeConfig::AcceptAll => PermissionMode::AcceptAll,
        PermissionModeConfig::AcceptEdits => PermissionMode::AcceptEdits,
        PermissionModeConfig::DenyAll => PermissionMode::DenyAll,
        PermissionModeConfig::Interactive => PermissionMode::Interactive,
        PermissionModeConfig::Default => return,
    };
    service.agent_mut().permission_mode = pm;
}

impl std::fmt::Debug for AgentHarness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHarness")
            .field("session_id", &self.session_id)
            .field("turn_count", &self.turn_count)
            .field("total_usage", &self.total_usage)
            .field("config", &self.config)
            .finish()
    }
}
