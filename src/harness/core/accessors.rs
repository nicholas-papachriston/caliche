//! Core Agent Harness implementation.
//!
//! The AgentHarness wraps AgentService with enhanced capabilities for
//! event streaming, context management, and state tracking.

use super::types::AgentHarness;
use crate::harness::command_ledger::CommandLedger;
use crate::harness::context::ContextManager;
use crate::harness::diff_tracker::DiffTracker;
use crate::harness::events::{AgentEvent, EventStream};
use crate::harness::state::StateManager;
use crate::messages::TokenUsage;
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;

impl AgentHarness {
    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Set the session ID (e.g. when creating from shared service with session config).
    pub fn set_session_id(&mut self, id: impl Into<String>) {
        self.session_id = id.into();
    }

    /// Set max turns for this harness (overrides config).
    pub fn set_max_turns(&mut self, max_turns: usize) {
        self.config.max_turns = max_turns;
    }

    /// Subscribe to agent events
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<AgentEvent> {
        self.event_stream.subscribe()
    }

    /// Get the event stream for external use
    pub fn event_stream(&self) -> &EventStream {
        &self.event_stream
    }

    /// Get the context manager
    pub fn context_manager(&self) -> Arc<RwLock<ContextManager>> {
        Arc::clone(&self.context_manager)
    }

    /// Get the state manager
    pub fn state_manager(&self) -> Arc<RwLock<StateManager>> {
        Arc::clone(&self.state_manager)
    }

    /// Get the per-message diff tracker (for undo/redo and API).
    pub fn diff_tracker(&self) -> Arc<RwLock<DiffTracker>> {
        Arc::clone(&self.diff_tracker)
    }

    /// Summarize recorded per-message file changes for undo/redo (CLI, API, UI).
    ///
    /// This mirrors the diff tracker's [`DiffTracker::session_summary`] for the
    /// full session without filtering by message id.
    pub async fn session_diff_summary(&self) -> crate::harness::SessionDiffSummary {
        self.diff_tracker.read().await.session_summary(None)
    }

    /// Last edit or bash command recorded for this session.
    #[must_use]
    pub fn command_ledger(&self) -> &Arc<RwLock<CommandLedger>> {
        &self.command_ledger
    }

    pub(crate) async fn sync_session_diff_state(&self) -> Result<()> {
        let state = self.diff_tracker.read().await.snapshot_state();
        self.service
            .session()
            .replace_diff_state(state)
            .context("sync session diff state")?;
        self.service.save_session_snapshot().await
    }

    /// Get the current turn count
    pub fn turn_count(&self) -> usize {
        self.turn_count
    }

    /// Get the total token usage so far
    pub fn total_usage(&self) -> &TokenUsage {
        &self.total_usage
    }
}
