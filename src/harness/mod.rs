//! Agent harness — shared execution layer for CLI and API.
//!
//! Wraps `AgentService` with event streaming, workspace context, and undo/rollback.
//!
//! # Architecture
//!
//! The harness wraps `AgentService` and adds:
//! - **Event streaming**: Events for tool execution, thinking, and file changes
//! - **Context management**: Workspace and file tracking for the active session
//! - **State management**: Checkpointing and undo/rollback for file modifications
//! - **Parallel execution**: Concurrent tool calls with progress events
//! - **Session inbox**: Stable input ids and control-message dedupe
//! - **Operations** (opt-in): `ToolTranslator` + `OperationManager` for durable tool specs
//!
//! # Example
//!
//! ```rust,no_run
//! use caliche::harness::{AgentHarness, HarnessConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = HarnessConfig::default();
//! let mut harness = AgentHarness::new(config).await?;
//!
//! // Subscribe to events
//! let mut events = harness.subscribe();
//! tokio::spawn(async move {
//!     while let Ok(event) = events.recv().await {
//!         println!("Event: {:?}", event);
//!     }
//! });
//!
//! // Run a task
//! let result = harness.run_task("Implement a simple hello world function").await?;
//! println!("Result: {}", result.response);
//! # Ok(())
//! # }
//! ```

mod broadcast_resync;
pub mod coding;
mod command_ledger;
pub mod complex_task;
pub mod config;
mod context;
mod core;
pub mod diff_contracts;
pub mod diff_tracker;
mod events;
mod executor;
pub mod inbox;
pub use executor::ParallelToolExecutor;
pub mod formatter;
mod hooks;
mod metrics;
mod persistence;
mod planning;
mod resume_context;
mod runtime;
mod session_diff_hook;
#[cfg(feature = "session-tools")]
mod session_spawn;
mod state;
mod streaming;
mod suspend;

pub use broadcast_resync::{BroadcastRecv, BroadcastResync, broadcast_tail, classify_recv};
pub use coding::{
    CodingContextPolicy, CodingMode, CodingTaskExecutor, CodingTaskRequest, CodingTaskResult,
    CodingTaskStatus, ProjectKind, VerificationPolicy, VerificationReport, VerificationRun,
    WorkspaceProfile, profile_workspace, run_with_harness as run_coding_task_with_harness,
};
pub use command_ledger::{CommandLedger, normalize_shell_command};
pub use complex_task::{
    COMPLEX_TASK_PROMPT, ComplexTaskConfig, ComplexTaskOutcome, Finding, RoundStatus,
    apply_complex_task_prompt, continuation_prompt, parse_findings, parse_gaps, parse_round_status,
    run_complex_task, verification_prompt,
};
pub use config::{HarnessConfig, SessionConfig};
pub use context::{
    ContextItem, ContextManager, FileContext, GitContext, ProjectType, detect_language,
};
pub use core::{
    AgentHarness, HarnessTurnResult, SessionSummary, TaskResult, sanitize_notify_preview,
};
pub use diff_tracker::{
    DiffTracker, FileSnapshot, FileSnapshotSummary, MessageDiff, MessageDiffSummary,
    SessionDiffState, SessionDiffSummary, SnapshotChangeKind,
};
pub use events::{AgentEvent, EventStream, public_agent_sse_event_name};
pub use hooks::{ToolCallAnalysis, ToolHooks};
pub use inbox::{
    ControlMessage, ControlMode, ControlOutcome, InboxAck, InboxInput, InboxSettings,
    InboxTurnOutcome, InputId, InputKind, SessionInbox,
};
pub use metrics::{HarnessDisciplineMetrics, estimate_usage_cost_usd};
pub use persistence::{
    ConversationExport, ExportedCheckpoint, ExportedMessage, SessionInfo, SessionPersistence,
};
pub use planning::{PlanStatus, StepStatus, TaskPlan, TaskPlanner, TaskStep};
pub use resume_context::{
    FittedResumeContext, RESUME_CONTEXT_TOKEN_BUDGET, fit_resume_messages,
    seed_service_resume_context,
};
pub use runtime::{AgentRuntime, SharedHarness};
#[cfg(feature = "session-tools")]
pub use session_spawn::{HarnessSessionSpawner, run_recorded_child};
pub use state::{Checkpoint, CheckpointId, FileVersion, RollbackAction, StateManager};
pub use streaming::{StreamingResponseHandler, detect_thinking_blocks, extract_code_blocks};
pub use suspend::{SessionPhase, SuspendStore, SuspendedSession};
