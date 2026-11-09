//! Event types for real-time agent observability.
//!
//! This module defines a unified event system that both CLI and API can use
//! to stream agent activity in real-time.

use crate::agent::tools::ToolResult;
use crate::messages::TokenUsage;
use crate::messages::stream_contracts::sse_name;
use crate::streams::{DurableEvent, DurableStreamWriter};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::broadcast;

/// Channel capacity for event broadcasting
const EVENT_CHANNEL_CAPACITY: usize = 1024;

/// Public SSE event name for [`AgentEvent`] (ROADMAP 2.5 taxonomy).
///
/// Maps the high-traffic lifecycle/tool/thinking variants to the dotted
/// [`sse_name`] constants; remaining variants keep their snake_case serde tag
/// names.
#[must_use]
pub fn public_agent_sse_event_name(event: &AgentEvent) -> &'static str {
    if let Some(name) = lifecycle_sse_name(event) {
        return name;
    }
    activity_sse_name(event)
}

fn lifecycle_sse_name(event: &AgentEvent) -> Option<&'static str> {
    Some(match event {
        AgentEvent::Thinking { .. } => sse_name::AGENT_THINKING,
        AgentEvent::ToolExecuting { .. } => sse_name::TOOL_START,
        AgentEvent::ToolComplete { .. } => sse_name::TOOL_END,
        AgentEvent::TurnComplete { .. } => sse_name::TURN_COMPLETE,
        AgentEvent::TurnError { .. } => sse_name::TURN_ERROR,
        AgentEvent::TurnStart { .. } => sse_name::TURN_START,
        AgentEvent::SessionStart { .. } => sse_name::SESSION_UPDATED,
        AgentEvent::SessionEnd { .. } => sse_name::SESSION_UPDATED,
        _ => return None,
    })
}

fn activity_sse_name(event: &AgentEvent) -> &'static str {
    match event {
        AgentEvent::Planning { .. } => "planning",
        AgentEvent::ReasoningStep { .. } => "reasoning_step",
        AgentEvent::ToolQueued { .. } => "tool_queued",
        AgentEvent::ToolError { .. } => "tool_error",
        AgentEvent::FileRead { .. } => "file_read",
        AgentEvent::FileWrite { .. } => "file_write",
        AgentEvent::FileDelete { .. } => "file_delete",
        AgentEvent::FileRollback { .. } => "file_rollback",
        AgentEvent::DirectoryListed { .. } => "directory_listed",
        AgentEvent::TextChunk { .. } => "text_chunk",
        AgentEvent::CodeBlock { .. } => "code_block",
        AgentEvent::CheckpointCreated { .. } => "checkpoint_created",
        AgentEvent::RolledBack { .. } => "rolled_back",
        AgentEvent::Done { .. } => "done",
        AgentEvent::Error { .. } => "error",
        AgentEvent::Warning { .. } => "warning",
        AgentEvent::Info { .. } => "info",
        AgentEvent::Status { .. } => "status",
        // Lifecycle events are named by `lifecycle_sse_name`.
        _ => sse_name::AGENT_THINKING,
    }
}

/// Unified event types for agent execution observability.
///
/// These events are emitted during agent execution and can be consumed
/// by CLI (for TUI display) or API (for SSE streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    // === Lifecycle Events ===
    /// Session has started
    SessionStart {
        session_id: String,
        workspace: Option<PathBuf>,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Session has ended
    SessionEnd {
        session_id: String,
        total_turns: usize,
        total_usage: TokenUsage,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// A new turn has started
    TurnStart {
        turn_id: usize,
        input: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// A turn has completed
    TurnComplete {
        turn_id: usize,
        response: String,
        usage: TokenUsage,
        duration_ms: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// A turn failed before completion. Terminal for session event streams.
    TurnError { message: String },

    // === Reasoning Events ===
    /// Agent is thinking/reasoning (extended thinking block).
    ///
    /// Chunks in one contiguous burst share `action_id` (UUID v7).
    Thinking {
        content: String,
        /// Time-sortable UUID v7 for this thinking segment. Empty on legacy events.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        action_id: String,
    },

    /// Agent is planning steps
    Planning { steps: Vec<String> },

    /// Agent generated reasoning step (for ReAct, etc.)
    ReasoningStep {
        step_type: String,
        content: String,
        metadata: Option<serde_json::Value>,
    },

    // === Tool Events ===
    /// Tool execution has been queued
    ToolQueued {
        tool_id: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        args_preview: Option<String>,
        timestamp: chrono::DateTime<chrono::Utc>,
        /// Time-sortable UUID v7 shared with the matching complete or error event.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        action_id: String,
    },

    /// Tool is currently executing
    ToolExecuting {
        tool_id: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        progress: Option<f32>,
        /// Time-sortable UUID v7 shared with the matching complete or error event.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        action_id: String,
    },

    /// Tool execution completed successfully
    ToolComplete {
        tool_id: String,
        tool_name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        result_preview: Option<String>,
        success: bool,
        duration_ms: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
        /// Time-sortable UUID v7 shared with the matching start event.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        action_id: String,
    },

    /// Tool execution failed
    ToolError {
        tool_id: String,
        tool_name: String,
        error: String,
        timestamp: chrono::DateTime<chrono::Utc>,
        /// Time-sortable UUID v7 shared with the matching start event.
        #[serde(default, skip_serializing_if = "String::is_empty")]
        action_id: String,
    },

    // === Context Events ===
    /// A file was read
    FileRead {
        path: PathBuf,
        #[serde(skip_serializing_if = "Option::is_none")]
        lines: Option<(usize, usize)>,
        size_bytes: usize,
    },

    /// A file was written/modified
    FileWrite {
        path: PathBuf,
        #[serde(skip_serializing_if = "Option::is_none")]
        diff_preview: Option<String>,
        is_new: bool,
        checkpoint_id: Option<String>,
    },

    /// A file was deleted
    FileDelete {
        path: PathBuf,
        checkpoint_id: Option<String>,
    },

    /// A file change was rolled back
    FileRollback {
        path: PathBuf,
        checkpoint_id: String,
        version: usize,
    },

    /// Directory was listed
    DirectoryListed { path: PathBuf, file_count: usize },

    // === Streaming Text Events ===
    /// A chunk of response text
    TextChunk { content: String },

    /// Code block detected in response
    CodeBlock {
        language: Option<String>,
        content: String,
    },

    // === State Events ===
    /// A checkpoint was created
    CheckpointCreated {
        checkpoint_id: String,
        label: String,
        file_count: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// State was rolled back to a checkpoint
    RolledBack {
        checkpoint_id: String,
        files_restored: Vec<PathBuf>,
    },

    // === Terminal Events ===
    /// Task completed successfully
    Done {
        final_response: String,
        total_usage: TokenUsage,
        duration_ms: u64,
    },

    /// An error occurred
    Error {
        error: String,
        recoverable: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        context: Option<String>,
    },

    /// A warning was generated
    Warning { message: String },

    /// Informational message
    Info { message: String },

    /// A progress line while the model has not answered yet, for example
    /// while a local server loads it. Replaced by the next output.
    Status { message: String },
}

impl AgentEvent {
    /// Create a session start event
    pub fn session_start(session_id: impl Into<String>, workspace: Option<PathBuf>) -> Self {
        Self::SessionStart {
            session_id: session_id.into(),
            workspace,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a session end event
    pub fn session_end(
        session_id: impl Into<String>,
        total_turns: usize,
        total_usage: TokenUsage,
    ) -> Self {
        Self::SessionEnd {
            session_id: session_id.into(),
            total_turns,
            total_usage,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a turn start event
    pub fn turn_start(turn_id: usize, input: impl Into<String>) -> Self {
        Self::TurnStart {
            turn_id,
            input: input.into(),
            timestamp: chrono::Utc::now(),
        }
    }

    /// Create a tool queued event
    pub fn tool_queued(
        tool_id: impl Into<String>,
        tool_name: impl Into<String>,
        args: &serde_json::Value,
        action_id: impl Into<String>,
    ) -> Self {
        let args_preview = serde_json::to_string(args).ok().map(|s| {
            if s.len() > 200 {
                format!("{}...", caliche_contracts::text::truncate_utf8(&s, 200))
            } else {
                s
            }
        });

        Self::ToolQueued {
            tool_id: tool_id.into(),
            tool_name: tool_name.into(),
            args_preview,
            timestamp: chrono::Utc::now(),
            action_id: action_id.into(),
        }
    }

    /// Create a tool complete event
    pub fn tool_complete(
        tool_id: impl Into<String>,
        tool_name: impl Into<String>,
        result: &ToolResult,
        duration_ms: u64,
        action_id: impl Into<String>,
    ) -> Self {
        let result_preview = if result.content.len() > 200 {
            Some(format!(
                "{}...",
                caliche_contracts::text::truncate_utf8(&result.content, 200)
            ))
        } else if !result.content.is_empty() {
            Some(result.content.clone())
        } else {
            None
        };

        Self::ToolComplete {
            tool_id: tool_id.into(),
            tool_name: tool_name.into(),
            result_preview,
            success: result.success,
            duration_ms,
            timestamp: chrono::Utc::now(),
            action_id: action_id.into(),
        }
    }

    /// Create a file write event
    pub fn file_write(
        path: PathBuf,
        diff_preview: Option<String>,
        is_new: bool,
        checkpoint_id: Option<String>,
    ) -> Self {
        Self::FileWrite {
            path,
            diff_preview,
            is_new,
            checkpoint_id,
        }
    }

    /// Create an error event
    pub fn error(error: impl Into<String>, recoverable: bool) -> Self {
        Self::Error {
            error: error.into(),
            recoverable,
            context: None,
        }
    }

    /// Create a terminal turn-failure event.
    pub fn turn_error(message: impl Into<String>) -> Self {
        Self::TurnError {
            message: message.into(),
        }
    }

    /// Create a done event
    pub fn done(
        final_response: impl Into<String>,
        total_usage: TokenUsage,
        duration_ms: u64,
    ) -> Self {
        Self::Done {
            final_response: final_response.into(),
            total_usage,
            duration_ms,
        }
    }

    /// Check if this is a terminal event (session end, done, or fatal error)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            AgentEvent::SessionEnd { .. }
                | AgentEvent::Done { .. }
                | AgentEvent::TurnError { .. }
                | AgentEvent::Error {
                    recoverable: false,
                    ..
                }
        )
    }
}

/// Event stream for broadcasting agent events to multiple subscribers.
///
/// When a durable writer is configured, emitted events are also appended to the
/// backend so that [replay](EventStream::replay) and [subscribe_durable](EventStream::subscribe_durable) can be used.
#[derive(Clone)]
pub struct EventStream {
    sender: broadcast::Sender<AgentEvent>,
    durable_writer: Option<DurableStreamWriter>,
}

impl EventStream {
    /// Create a new event stream
    pub fn new() -> Self {
        Self {
            sender: broadcast::channel(EVENT_CHANNEL_CAPACITY).0,
            durable_writer: None,
        }
    }

    /// Attach a durable stream writer so that emitted events are persisted.
    pub fn with_durable_writer(mut self, writer: DurableStreamWriter) -> Self {
        self.durable_writer = Some(writer);
        self
    }

    /// Emit an event to all subscribers and, if configured, append to the durable backend.
    pub fn emit(&self, event: AgentEvent) {
        let _ = self.sender.send(event.clone());
        if let Some(ref w) = self.durable_writer {
            let w = w.clone();
            let durable = DurableEvent::agent(event, None);
            tokio::spawn(async move {
                let _ = w.append(&durable).await;
            });
        }
    }

    /// Subscribe to events (live only).
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.sender.subscribe()
    }

    /// Replay persisted events from the durable backend from `from_id` (use `streams::STREAM_START` for beginning).
    /// Returns `None` if no durable backend is configured.
    pub fn replay(
        &self,
        from_id: String,
    ) -> Option<
        impl futures_util::Stream<Item = anyhow::Result<(String, AgentEvent)>> + Send + 'static,
    > {
        let writer = self.durable_writer.clone()?;
        let reader = writer.reader();
        Some(reader.replay_stream(from_id).then(
            |r: anyhow::Result<(String, DurableEvent)>| async move {
                match r {
                    Ok((id, de)) => de
                        .as_agent_event()
                        .cloned()
                        .map(|e| Ok((id, e)))
                        .unwrap_or_else(|| {
                            Err(anyhow::anyhow!("durable stream expected agent event"))
                        }),
                    Err(e) => Err(e),
                }
            },
        ))
    }

    /// Subscribe with replay then live: first yields persisted events from `from_id`, then live events from broadcast.
    /// If `tail_timeout` is `Some(d)`, after replay blocks for new events with timeout `d`; if `None`, only replays.
    /// Returns `None` if no durable backend is configured.
    pub fn subscribe_durable(
        &self,
        from_id: String,
        tail_timeout: Option<std::time::Duration>,
    ) -> Option<
        impl futures_util::Stream<Item = anyhow::Result<(String, AgentEvent)>> + Send + 'static,
    > {
        let writer = self.durable_writer.clone()?;
        let reader = writer.reader();
        Some(reader.replay_then_tail_stream(from_id, tail_timeout).then(
            |r: anyhow::Result<(String, DurableEvent)>| async move {
                match r {
                    Ok((id, de)) => de
                        .as_agent_event()
                        .cloned()
                        .map(|e| Ok((id, e)))
                        .unwrap_or_else(|| {
                            Err(anyhow::anyhow!("durable stream expected agent event"))
                        }),
                    Err(e) => Err(e),
                }
            },
        ))
    }

    /// Get a clone of the sender for use in other contexts
    pub fn sender(&self) -> broadcast::Sender<AgentEvent> {
        self.sender.clone()
    }

    /// Get the number of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for EventStream {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod public_sse_name_tests {
    use super::*;

    #[test]
    fn public_agent_sse_event_name_uses_roadmap_taxonomy() {
        assert_eq!(
            public_agent_sse_event_name(&AgentEvent::Thinking {
                content: "think".into(),
                action_id: String::new(),
            }),
            "agent.thinking"
        );
        assert_eq!(
            public_agent_sse_event_name(&AgentEvent::ToolExecuting {
                tool_id: "t1".into(),
                tool_name: "lookup".into(),
                progress: None,
                action_id: String::new(),
            }),
            "tool.start"
        );
        assert_eq!(
            public_agent_sse_event_name(&AgentEvent::ToolComplete {
                tool_id: "t1".into(),
                tool_name: "lookup".into(),
                result_preview: None,
                success: true,
                duration_ms: 1,
                timestamp: chrono::Utc::now(),
                action_id: String::new(),
            }),
            "tool.end"
        );
        assert_eq!(
            public_agent_sse_event_name(&AgentEvent::TurnComplete {
                turn_id: 1,
                response: "done".into(),
                usage: TokenUsage::default(),
                duration_ms: 1,
                timestamp: chrono::Utc::now(),
            }),
            "turn.complete"
        );
        assert_eq!(
            public_agent_sse_event_name(&AgentEvent::turn_error("anthropic 400: rejected")),
            sse_name::TURN_ERROR
        );
        assert!(AgentEvent::turn_error("anthropic 400: rejected").is_terminal());
        assert_eq!(
            public_agent_sse_event_name(&AgentEvent::TurnStart {
                turn_id: 1,
                input: "hi".into(),
                timestamp: chrono::Utc::now(),
            }),
            "turn.start"
        );
        assert_eq!(
            public_agent_sse_event_name(&AgentEvent::SessionStart {
                session_id: "s1".into(),
                workspace: None,
                timestamp: chrono::Utc::now(),
            }),
            "session.updated"
        );
        assert_eq!(
            public_agent_sse_event_name(&AgentEvent::SessionEnd {
                session_id: "s1".into(),
                total_turns: 1,
                total_usage: TokenUsage::default(),
                timestamp: chrono::Utc::now(),
            }),
            "session.updated"
        );
    }
}

impl std::fmt::Debug for EventStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventStream")
            .field("subscriber_count", &self.subscriber_count())
            .finish()
    }
}
