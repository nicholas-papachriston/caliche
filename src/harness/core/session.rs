//! Core Agent Harness implementation.
//!
//! The AgentHarness wraps AgentService with enhanced capabilities for
//! event streaming, context management, and state tracking.

use super::types::{AgentHarness, HarnessTurnResult, SessionSummary, sanitize_notify_preview};
use crate::agent::service::AgentService;
use crate::harness::events::AgentEvent;
use crate::harness::hooks::ToolHooks;
use crate::harness::persistence::ConversationExport;
use crate::harness::resume_context::{
    FittedResumeContext, RESUME_CONTEXT_TOKEN_BUDGET, fit_resume_messages,
    seed_service_resume_context,
};
use crate::harness::streaming::StreamingResponseHandler;
use crate::messages::{Message, StreamEvent};
use anyhow::Result;
use regex::Regex;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock};

/// Matches `"status": "complete"` or `"task_complete": true`, with optional
/// whitespace around the colon. The key must be followed directly by its value.
#[expect(
    clippy::expect_used,
    reason = "static Regex/LazyLock literals are reviewed constants; failure is a programmer bug"
)]
static RE_COMPLETION_FIELD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#""status"\s*:\s*"complete"|"task_complete"\s*:\s*true\b"#)
        .expect("RE_COMPLETION_FIELD")
});

impl AgentHarness {
    /// Rollback to a specific checkpoint.
    pub async fn rollback(&mut self, checkpoint_id: &str) -> Result<Vec<PathBuf>> {
        let actions = self
            .state_manager
            .write()
            .await
            .rollback_to_str(checkpoint_id)?;

        let mut restored_files = Vec::new();
        for action in &actions {
            restored_files.push(action.path.clone());
            self.event_stream.emit(AgentEvent::FileRollback {
                path: action.path.clone(),
                checkpoint_id: checkpoint_id.to_string(),
                version: action.version,
            });
        }

        self.event_stream.emit(AgentEvent::RolledBack {
            checkpoint_id: checkpoint_id.to_string(),
            files_restored: restored_files.clone(),
        });

        Ok(restored_files)
    }

    /// Undo the last message's file changes (per-message undo).
    pub async fn undo_last(&mut self) -> Result<Option<crate::harness::MessageDiff>> {
        let result = {
            let mut tracker = self.diff_tracker.write().await;
            tracker.undo_last()
        };
        match result {
            Ok(Some(diff)) => {
                for snap in &diff.snapshots {
                    self.event_stream.emit(AgentEvent::FileRollback {
                        path: snap.path.clone(),
                        checkpoint_id: "undo".to_string(),
                        version: 0,
                    });
                }
                self.sync_session_diff_state().await?;
                Ok(Some(diff))
            }
            Ok(None) => {
                self.sync_session_diff_state().await?;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// Undo a specific message and every newer message after it.
    pub async fn undo_to_message(
        &mut self,
        message_id: uuid::Uuid,
    ) -> Result<Vec<crate::harness::MessageDiff>> {
        let diffs = {
            let mut tracker = self.diff_tracker.write().await;
            tracker.undo_to_message(message_id)?
        };
        for diff in &diffs {
            for snap in &diff.snapshots {
                self.event_stream.emit(AgentEvent::FileRollback {
                    path: snap.path.clone(),
                    checkpoint_id: "undo".to_string(),
                    version: 0,
                });
            }
        }
        self.sync_session_diff_state().await?;
        Ok(diffs)
    }

    /// Redo the last reverted message's file changes.
    pub async fn redo_last(&mut self) -> Result<Option<crate::harness::MessageDiff>> {
        let result = {
            let mut tracker = self.diff_tracker.write().await;
            tracker.redo_last()
        }?;
        self.sync_session_diff_state().await?;
        Ok(result)
    }

    /// Create a manual checkpoint.
    pub async fn checkpoint(&mut self, label: &str) -> String {
        let id = self.state_manager.write().await.checkpoint(label);
        let file_count = self.state_manager.read().await.checkpoint_file_count(&id);

        self.event_stream.emit(AgentEvent::CheckpointCreated {
            checkpoint_id: id.to_string(),
            label: label.to_string(),
            file_count,
            timestamp: chrono::Utc::now(),
        });

        self.last_checkpoint_id = Some(id.to_string());
        id.to_string()
    }

    /// Logical session phase (idle / in-turn / complete).
    #[must_use]
    pub fn phase(&self) -> crate::harness::suspend::SessionPhase {
        self.phase
    }

    /// Session inbox (idempotent input dedupe).
    #[must_use]
    pub fn inbox(&self) -> crate::harness::inbox::SessionInbox {
        self.side.lock().inbox.clone()
    }

    pub(crate) fn is_hard_stopped(&self) -> bool {
        self.side.lock().hard_stopped
    }

    pub(crate) fn is_suspending(&self) -> bool {
        self.side.lock().suspending
    }

    pub(crate) fn set_suspending(&self, suspending: bool) {
        self.side.lock().suspending = suspending;
    }

    /// Compaction reports persisted with this session.
    #[must_use]
    pub fn compaction_reports(&self) -> &[crate::messages::CompactionReport] {
        &self.compaction_reports
    }

    pub(crate) fn finish_turn_side_effects(
        &mut self,
        _result: &mut crate::harness::HarnessTurnResult,
    ) {
        self.capture_compaction_report();
        self.apply_pending_when_idle();
        self.apply_pending_inbox_settings();
    }

    pub(crate) fn finish_turn_side_effects_after_error(&mut self) {
        self.capture_compaction_report();
        self.apply_pending_when_idle();
        self.apply_pending_inbox_settings();
    }

    fn apply_pending_when_idle(&mut self) {
        let mut side = self.side.lock();
        if side.pending_when_idle {
            side.hard_stopped = true;
            side.pending_when_idle = false;
            self.phase = crate::harness::suspend::SessionPhase::Complete;
        } else if !side.hard_stopped {
            self.phase = crate::harness::suspend::SessionPhase::Idle;
        }
    }

    fn apply_pending_inbox_settings(&mut self) {
        let effort = {
            let side = self.side.lock();
            side.inbox_settings
                .reasoning_effort
                .as_deref()
                .and_then(crate::messages::ReasoningEffort::from_inbox_token)
        };
        if let Some(effort) = effort {
            self.service.agent_mut().apply_reasoning_effort(effort);
        }
    }

    fn capture_compaction_report(&mut self) {
        if let Some(report) = self.service.agent().last_compaction_report()
            && !report.is_empty()
        {
            self.compaction_reports.push(report);
        }
    }

    /// Submit an input to the session inbox without applying it.
    pub fn submit_inbox(
        &mut self,
        input: crate::harness::inbox::InboxInput,
    ) -> anyhow::Result<crate::harness::inbox::InboxAck> {
        self.side.lock().inbox.submit(input)
    }

    /// Apply a previously submitted inbox item.
    pub async fn apply_inbox(
        &mut self,
        input_id: &str,
    ) -> anyhow::Result<crate::harness::InboxTurnOutcome> {
        let input = {
            let side = self.side.lock();
            let record = side
                .inbox
                .get(input_id)
                .ok_or_else(|| anyhow::anyhow!("unknown inbox input '{input_id}'"))?;
            if record.applied {
                return Ok(crate::harness::InboxTurnOutcome {
                    ack: crate::harness::inbox::InboxAck {
                        input_id: input_id.to_string(),
                        duplicate: true,
                        kind: record.input.kind,
                        applied: true,
                    },
                    turn: None,
                    control: None,
                });
            }
            record.input.clone()
        };
        match input.kind {
            crate::harness::inbox::InputKind::External
            | crate::harness::inbox::InputKind::Crash => {
                let mut turn = self.run_turn(&input.payload).await?;
                turn.input_id = Some(input.id.clone());
                self.side
                    .lock()
                    .inbox
                    .mark_applied(&input.id, Some(turn.response.clone()))?;
                Ok(crate::harness::InboxTurnOutcome {
                    ack: crate::harness::inbox::InboxAck {
                        input_id: input.id,
                        duplicate: false,
                        kind: input.kind,
                        applied: true,
                    },
                    turn: Some(turn),
                    control: None,
                })
            }
            crate::harness::inbox::InputKind::Control => {
                let control = input
                    .control
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("control input missing payload"))?;
                let idle = matches!(
                    self.phase,
                    crate::harness::suspend::SessionPhase::Idle
                        | crate::harness::suspend::SessionPhase::Complete
                );
                let outcome = {
                    let mut side = self.side.lock();
                    apply_control_flags(&mut side, control, idle)?
                };
                if matches!(
                    outcome,
                    crate::harness::inbox::ControlOutcome::HardStopped { .. }
                        | crate::harness::inbox::ControlOutcome::WhenIdle {
                            applied_immediately: true,
                            ..
                        }
                ) {
                    self.phase = crate::harness::suspend::SessionPhase::Complete;
                }
                if let crate::harness::inbox::ControlOutcome::SettingsApplied { ref settings } =
                    outcome
                {
                    if let Some(raw) = settings.reasoning_effort.as_deref() {
                        let effort = crate::messages::ReasoningEffort::from_inbox_token(raw)
                            .ok_or_else(|| {
                                anyhow::anyhow!("unsupported reasoning_effort '{raw}'")
                            })?;
                        self.service.agent_mut().apply_reasoning_effort(effort);
                    }
                }
                self.side
                    .lock()
                    .inbox
                    .mark_applied(&input.id, Some(format!("{outcome:?}")))?;
                Ok(crate::harness::InboxTurnOutcome {
                    ack: crate::harness::inbox::InboxAck {
                        input_id: input.id,
                        duplicate: false,
                        kind: crate::harness::inbox::InputKind::Control,
                        applied: true,
                    },
                    turn: None,
                    control: Some(outcome),
                })
            }
        }
    }
}

pub(crate) fn apply_control_flags(
    side: &mut crate::harness::core::SessionSideState,
    control: crate::harness::inbox::ControlMessage,
    idle: bool,
) -> anyhow::Result<crate::harness::inbox::ControlOutcome> {
    use crate::harness::inbox::{ControlMode, ControlOutcome};
    Ok(match control.mode {
        ControlMode::Hard => {
            side.hard_stopped = true;
            ControlOutcome::HardStopped {
                reason: control.reason,
            }
        }
        ControlMode::WhenIdle => {
            if idle {
                side.hard_stopped = true;
                side.pending_when_idle = false;
            } else {
                side.pending_when_idle = true;
            }
            ControlOutcome::WhenIdle {
                applied_immediately: idle,
                reason: control.reason,
            }
        }
        ControlMode::Heartbeat => {
            side.last_heartbeat = Some(control.reason.clone());
            ControlOutcome::Heartbeat {
                reason: control.reason,
            }
        }
        ControlMode::Settings => {
            if let Some(settings) = control.settings {
                if let Some(raw) = settings.reasoning_effort.as_deref()
                    && crate::messages::ReasoningEffort::from_inbox_token(raw).is_none()
                {
                    anyhow::bail!("unsupported reasoning_effort '{raw}'");
                }
                side.inbox_settings = settings;
            }
            ControlOutcome::SettingsApplied {
                settings: side.inbox_settings.clone(),
            }
        }
    })
}

impl AgentHarness {
    /// Set the logical session phase (tests and hosts).
    pub fn set_phase(&mut self, phase: crate::harness::suspend::SessionPhase) {
        self.phase = phase;
    }

    /// Last checkpoint id recorded by this harness, if any.
    #[must_use]
    pub fn last_checkpoint_id(&self) -> Option<&str> {
        self.last_checkpoint_id.as_deref()
    }

    /// Build a durable suspend snapshot (does not drop in-memory state).
    #[must_use]
    pub fn suspend_snapshot(&self) -> crate::harness::suspend::SuspendedSession {
        let workspace = self
            .context_manager
            .try_read()
            .map(|ctx| ctx.workspace_root().to_path_buf())
            .ok();
        let checkpoints: Vec<_> = self
            .state_manager
            .try_read()
            .map(|state| state.list_checkpoints().to_vec())
            .unwrap_or_default();
        let last_checkpoint_id = self.last_checkpoint_id.clone().or_else(|| {
            checkpoints
                .last()
                .map(|checkpoint| checkpoint.id.to_string())
        });
        let side = self.side.lock();
        crate::harness::suspend::SuspendedSession {
            version: "1".to_string(),
            session_id: self.session_id.clone(),
            suspended_at: chrono::Utc::now(),
            phase: self.phase,
            turn_count: self.turn_count,
            last_checkpoint_id,
            workspace,
            messages: self
                .history
                .iter()
                .map(crate::harness::persistence::ExportedMessage::from)
                .collect(),
            total_usage: self.total_usage,
            checkpoints: checkpoints
                .iter()
                .map(crate::harness::persistence::ExportedCheckpoint::from)
                .collect(),
            inbox: side.inbox.clone(),
            compaction_reports: self.compaction_reports.clone(),
            hard_stopped: side.hard_stopped,
            pending_when_idle: side.pending_when_idle,
            inbox_settings: side.inbox_settings.clone(),
        }
    }

    /// Restore conversation, phase, and checkpoint identity from a snapshot.
    pub async fn apply_suspended(
        &mut self,
        snapshot: &crate::harness::suspend::SuspendedSession,
    ) -> Result<()> {
        self.session_id = snapshot.session_id.clone();
        self.history = snapshot.to_messages();
        self.turn_count = snapshot.turn_count;
        self.total_usage = snapshot.total_usage;
        self.phase = snapshot.phase;
        self.last_checkpoint_id = snapshot.last_checkpoint_id.clone();
        self.state_manager
            .write()
            .await
            .restore_checkpoints(snapshot.checkpoints_as_state());
        self.compaction_reports = snapshot.compaction_reports.clone();
        {
            let mut side = self.side.lock();
            side.inbox = snapshot.inbox.clone();
            side.hard_stopped = snapshot.hard_stopped;
            side.pending_when_idle = snapshot.pending_when_idle;
            side.inbox_settings = snapshot.inbox_settings.clone();
            side.suspending = false;
            if let Some(raw) = side.inbox_settings.reasoning_effort.as_deref()
                && let Some(effort) = crate::messages::ReasoningEffort::from_inbox_token(raw)
            {
                self.service.agent_mut().apply_reasoning_effort(effort);
            }
        }
        Ok(())
    }

    /// Clear conversation history and start fresh.
    pub fn clear_history(&mut self) {
        self.history.clear();
        self.turn_count = 0;
    }

    /// Get the underlying agent service for advanced operations.
    pub fn service(&self) -> &AgentService {
        &self.service
    }

    /// Get a mutable reference to the underlying agent service.
    pub fn service_mut(&mut self) -> &mut AgentService {
        &mut self.service
    }

    /// Spawn the optional notify command in a fire-and-forget manner.
    ///
    /// Injects turn metadata via environment variables:
    /// - `CALICHE_TURN_STATUS` — `"success"`, `"error"`, or `"interrupted"`
    /// - `CALICHE_TURN_TOKENS` — total tokens as a decimal string, or empty
    /// - `CALICHE_SESSION_ID` — current session UUID
    /// - `CALICHE_PROMPT_PREVIEW` — first 100 characters of the user prompt
    pub(crate) fn fire_notify_command(
        &self,
        status: &str,
        prompt_preview: &str,
        total_tokens: Option<u64>,
    ) {
        let Some(ref cmd) = self.notify_command else {
            return;
        };
        let tokens_str = total_tokens.map(|t| t.to_string()).unwrap_or_default();
        let prompt_preview = sanitize_notify_preview(prompt_preview);
        let _ = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(cmd)
            .env("CALICHE_TURN_STATUS", status)
            .env("CALICHE_TURN_TOKENS", &tokens_str)
            .env("CALICHE_SESSION_ID", &self.session_id)
            .env("CALICHE_PROMPT_PREVIEW", prompt_preview)
            .spawn();
    }

    /// Detect if the agent is signaling task completion.
    /// Uses phrase heuristics and optional structured JSON (status/summary/next_steps schema).
    pub(crate) fn detect_task_completion(&self, response: &str) -> bool {
        let lower = response.to_lowercase();

        // Structured completion: look for JSON with "status": "complete" or "task_complete": true
        if self.detect_structured_completion(response) {
            return true;
        }

        // Common completion phrases (fallback)
        let completion_phrases = [
            "task complete",
            "task completed",
            "task is complete",
            "i have completed",
            "i've completed",
            "successfully completed",
            "finished the task",
            "done with the task",
            "all done",
            "that's everything",
            "let me know if you need anything else",
            "is there anything else",
        ];

        for phrase in &completion_phrases {
            if lower.contains(phrase) {
                return true;
            }
        }

        // Explicit completion markers
        if response.contains("[TASK_COMPLETE]") || response.contains("<<DONE>>") {
            return true;
        }

        false
    }

    /// Try to detect structured completion from JSON in the response (e.g. ```json {"status":"complete"} ```).
    pub(crate) fn detect_structured_completion(&self, response: &str) -> bool {
        // Extract potential JSON from a fenced block or the surrounding text
        if let Some(json_str) = caliche_contracts::text::extract_json(response)
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str)
            && let Some(obj) = v.as_object()
        {
            if obj.get("status").and_then(|s| s.as_str()) == Some("complete") {
                return true;
            }
            if obj.get("task_complete").and_then(|b| b.as_bool()) == Some(true) {
                return true;
            }
        }
        // Inline JSON: "status":"complete" or "task_complete":true (flexible whitespace)
        RE_COMPLETION_FIELD.is_match(response)
    }

    // === Context Injection Methods ===

    /// Build context-aware prompt with relevant file and project context
    pub async fn build_context_prompt(&self, user_input: &str) -> String {
        let ctx = self.context_manager.read().await;
        let mut prompt_parts = Vec::new();

        // Add project type context
        if let Some(project_type) = ctx.project_type() {
            prompt_parts.push(format!("Project Type: {:?}", project_type));
        }

        // Add git context
        if let Some(git) = ctx.git_context() {
            prompt_parts.push(format!(
                "Git: branch='{}', {} modified files, {} staged",
                git.branch, git.modified_count, git.staged_count
            ));
        }

        // Add recently accessed files as context
        let recent_files: Vec<String> = ctx
            .tracked_files()
            .filter(|f| f.read_count > 0)
            .take(5)
            .map(|f| format!("  - {} ({:?})", f.path.display(), f.language))
            .collect();

        if !recent_files.is_empty() {
            prompt_parts.push(format!("Recent Files:\n{}", recent_files.join("\n")));
        }

        // Combine with user input
        if prompt_parts.is_empty() {
            user_input.to_string()
        } else {
            format!(
                "[Context]\n{}\n\n[Task]\n{}",
                prompt_parts.join("\n"),
                user_input
            )
        }
    }

    /// Run a turn with automatic context injection
    pub async fn run_turn_with_context(&mut self, input: &str) -> Result<HarnessTurnResult> {
        // Build context-enhanced prompt
        let enhanced_input = self.build_context_prompt(input).await;

        // Detect project type if not already done
        {
            let mut ctx = self.context_manager.write().await;
            if ctx.project_type().is_none() {
                let _ = ctx.detect_project();
            }
            // Refresh git context
            let _ = ctx.refresh_git_context().await;
        }

        // Run the turn
        self.run_turn(&enhanced_input).await
    }

    // === Export/Import Methods ===

    /// Export the current conversation
    pub fn export_conversation(&self) -> ConversationExport {
        let workspace = self
            .context_manager
            .try_read()
            .map(|c| c.workspace_root().to_path_buf())
            .unwrap_or_else(|_| PathBuf::from("."));

        let checkpoints: Vec<_> = self
            .state_manager
            .try_read()
            .map(|s| s.list_checkpoints().to_vec())
            .unwrap_or_default();

        let modified_files = self
            .state_manager
            .try_read()
            .map(|s| s.modified_files())
            .unwrap_or_default();

        ConversationExport::new(&self.session_id)
            .with_workspace(workspace)
            .with_messages(&self.history)
            .with_usage(self.total_usage)
            .with_checkpoints(&checkpoints)
            .with_modified_files(modified_files)
    }

    /// Save the conversation to a file
    pub fn save_conversation(&self, path: &std::path::Path) -> Result<()> {
        let export = self.export_conversation();
        export.save_to_file(path)
    }

    /// Import a conversation from export
    pub fn import_conversation(&mut self, export: &ConversationExport) {
        self.history = export.to_messages();
        self.total_usage = export.total_usage;
        self.turn_count = self
            .history
            .iter()
            .filter(|m| matches!(m.role, crate::messages::MessageRole::User))
            .count();
    }

    /// Load and import a conversation from a file
    pub fn load_conversation(&mut self, path: &std::path::Path) -> Result<()> {
        let export = ConversationExport::load_from_file(path)?;
        self.import_conversation(&export);
        Ok(())
    }

    // === Streaming Methods ===

    /// Create a streaming response handler for this harness
    pub fn create_streaming_handler(&self) -> StreamingResponseHandler {
        StreamingResponseHandler::new(self.event_stream.clone())
    }

    /// Mirror high-traffic stream chunks onto the session event bus for SSE clients.
    pub(crate) fn emit_public_stream_event(&self, event: &StreamEvent) {
        match event {
            StreamEvent::TextChunk { content } if !content.is_empty() => {
                self.clear_thinking_action();
                self.event_stream.emit(AgentEvent::TextChunk {
                    content: content.clone(),
                });
            }
            StreamEvent::ThinkingChunk { content, action_id } if !content.is_empty() => {
                let action_id = self.thinking_action_id(action_id.as_deref());
                self.event_stream.emit(AgentEvent::Thinking {
                    content: content.clone(),
                    action_id,
                });
            }
            StreamEvent::ToolCallStart { .. } | StreamEvent::ToolCallResult { .. } => {
                self.clear_thinking_action();
            }
            StreamEvent::Status { message } => {
                self.event_stream.emit(AgentEvent::Status {
                    message: message.clone(),
                });
            }
            _ => {}
        }
    }

    fn thinking_action_id(&self, stamped: Option<&str>) -> String {
        let mut side = self.side.lock();
        if let Some(id) = stamped.filter(|id| !id.is_empty()) {
            side.thinking_action_id = Some(id.to_string());
            return id.to_string();
        }
        if let Some(id) = side.thinking_action_id.clone() {
            return id;
        }
        let id = crate::utils::time_id::generate_time_id().to_string();
        side.thinking_action_id = Some(id.clone());
        id
    }

    fn clear_thinking_action(&self) {
        self.side.lock().thinking_action_id = None;
    }

    // === Tool Hooks ===

    /// Create tool hooks for this harness.
    ///
    /// The hooks share the agent action-id map, so a tool start and its result
    /// keep the UUID v7 the stream loop already stamped.
    #[must_use]
    pub fn create_tool_hooks(&self) -> ToolHooks {
        ToolHooks::new(self.event_stream.clone(), Arc::clone(&self.state_manager))
            .with_formatters(self.config.formatter_rules.clone())
            .with_action_ids(Arc::clone(&self.service.agent().action_ids))
    }

    // === Advanced Operations ===

    /// Get a summary of the current session state
    pub async fn session_summary(&self) -> SessionSummary {
        let context = self.context_manager.read().await;
        let state = self.state_manager.read().await;

        SessionSummary {
            session_id: self.session_id.clone(),
            turn_count: self.turn_count,
            total_tokens: self.total_usage.total_tokens.unwrap_or(0),
            workspace: Some(context.workspace_root().to_path_buf()),
            project_type: context.project_type().cloned(),
            git_branch: context.git_context().map(|g| g.branch.clone()),
            files_tracked: context.tracked_files().count(),
            files_modified: state.modified_files().len(),
            checkpoints: state.list_checkpoints().len(),
            history_messages: self.history.len(),
        }
    }

    /// Add a file to context tracking
    pub async fn track_file(&mut self, path: &std::path::Path) -> Result<()> {
        let mut ctx = self.context_manager.write().await;
        ctx.track_file(path).await?;
        Ok(())
    }

    /// Get conversation history
    pub fn history(&self) -> &[Message] {
        &self.history
    }

    /// Copy service memory into harness history when a streamed or chat turn
    /// persisted a full tool transcript. Fall back to user/assistant text.
    pub(crate) async fn replace_history_from_memory(
        &mut self,
        fallback_user: &str,
        fallback_assistant: &str,
    ) {
        if let Some(memory) = self.service.memory() {
            match memory.read().await.get_context().await {
                Ok(messages) if !messages.is_empty() => {
                    self.history = messages;
                    return;
                }
                Ok(_) => {}
                Err(error) => {
                    tracing::warn!(
                        error = %error,
                        "failed to copy service memory into harness history"
                    );
                }
            }
        }
        self.history.push(Message::user(fallback_user.to_string()));
        self.history
            .push(Message::assistant(fallback_assistant.to_string()));
    }

    /// Set conversation history directly
    pub fn set_history(&mut self, history: Vec<Message>) {
        self.history = history;
        self.turn_count = self
            .history
            .iter()
            .filter(|m| matches!(m.role, crate::messages::MessageRole::User))
            .count();
    }

    /// Load a stored transcript into harness history and service memory.
    ///
    /// The next turn reads service memory, so history alone would not reach
    /// the model. Oversized transcripts are summarized first.
    ///
    /// # Errors
    ///
    /// Returns an error when memory cannot store the fitted transcript.
    pub async fn restore_resume_context(
        &mut self,
        stored: Vec<Message>,
    ) -> Result<FittedResumeContext> {
        let fitted = fit_resume_messages(&stored, RESUME_CONTEXT_TOKEN_BUDGET);
        self.set_history(fitted.messages.clone());
        seed_service_resume_context(self.service_mut(), fitted.messages.clone()).await?;
        Ok(fitted)
    }

    /// Add a message to history
    pub fn add_to_history(&mut self, message: Message) {
        self.history.push(message);
    }
}
