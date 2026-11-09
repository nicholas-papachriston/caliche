//! Streaming event types for real-time LLM responses.

use crate::messages::{PerformanceMetrics, TokenUsage};
use crate::utils::beef_helpers::{
    deserialize_cow_str, deserialize_option_cow_str, serialize_cow_str, serialize_option_cow_str,
};
use beef::lean::Cow;
use serde::{Deserialize, Serialize};

/// Stream events for real-time LLM responses
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    /// Text chunk from the LLM
    TextChunk { content: String },
    /// Reasoning/thinking chunk from models that stream deliberation traces.
    ///
    /// Chunks in one contiguous burst share [`Self::ThinkingChunk::action_id`].
    ThinkingChunk {
        content: String,
        /// Time-sortable UUID v7 for this thinking segment.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action_id: Option<String>,
    },
    /// Tool call detected (for native tool calling)
    ToolCallStart {
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        id: Cow<'static, str>,
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        name: Cow<'static, str>,
        arguments: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
        /// Time-sortable UUID v7 shared with the matching [`Self::ToolCallResult`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action_id: Option<String>,
    },
    /// Tool execution result (when streaming with tools)
    ToolCallResult {
        #[serde(
            serialize_with = "serialize_cow_str",
            deserialize_with = "deserialize_cow_str"
        )]
        name: Cow<'static, str>,
        result: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        typed_payload: Option<serde_json::Value>,
        /// Provider tool-call id. Same value as [`Self::ToolCallStart::id`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Time-sortable UUID v7 shared with the matching [`Self::ToolCallStart`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        action_id: Option<String>,
    },
    /// Streaming audio payload chunk.
    AudioChunk { data: Vec<u8>, format: String },
    /// Generated image payload.
    ImageGenerated {
        data: Vec<u8>,
        media_type: String,
        revised_prompt: Option<String>,
    },
    /// Notification for processed attachments.
    AttachmentProcessed {
        filename: String,
        attachment_type: String,
        summary: String,
    },
    /// Metadata including token usage, finish reason, and optional performance metrics
    Metadata {
        usage: Option<TokenUsage>,
        #[serde(
            default,
            serialize_with = "serialize_option_cow_str",
            deserialize_with = "deserialize_option_cow_str"
        )]
        finish_reason: Option<Cow<'static, str>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metrics: Option<PerformanceMetrics>,
    },
    /// Iteration limit was reached - stream will end with partial results
    LimitReached {
        /// Maximum iterations that were allowed
        max_iterations: usize,
        /// Total token usage up to this point
        usage: TokenUsage,
    },
    /// Context was compacted to fit within the window
    Compaction {
        tokens_before: usize,
        tokens_after: usize,
        messages_removed: usize,
        summary: Option<String>,
        /// Machine-readable omitted / truncated / compacted changes.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        report: Option<crate::messages::CompactionReport>,
    },
    /// Planner-first phased: current phase changed (for REPL/UI to show e.g. "Phase 2/3: editing").
    PhaseTransition {
        phase_index: usize,
        phase_total: usize,
        tool_names: Vec<String>,
    },

    /// A progress note while nothing else streams, for example while a
    /// local server loads the model before its first token.
    Status { message: String },

    /// Stream complete
    Done,

    /// Agent run started (lifecycle; emitted once at stream start).
    AgentStart,
    /// Agent run finished (lifecycle; emitted once before Done).
    AgentEnd,
    /// New turn started (one LLM call plus any tool executions).
    TurnStart,
    /// Turn finished (assistant message and optional tool results).
    TurnEnd,
}

/// Versioned envelope used for transport-stable stream contracts.
#[derive(Debug, Clone, Serialize)]
pub struct StreamEventEnvelope {
    pub version: Cow<'static, str>,
    #[serde(serialize_with = "serialize_cow_str")]
    pub r#type: Cow<'static, str>,
    pub payload: StreamEvent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
}

impl StreamEventEnvelope {
    pub fn v1(payload: StreamEvent, event_id: Option<String>) -> Self {
        let event_type = match &payload {
            StreamEvent::TextChunk { .. } => "text_chunk",
            StreamEvent::ThinkingChunk { .. } => "thinking_chunk",
            StreamEvent::ToolCallStart { .. } => "tool_call_start",
            StreamEvent::ToolCallResult { .. } => "tool_call_result",
            StreamEvent::AudioChunk { .. } => "audio_chunk",
            StreamEvent::ImageGenerated { .. } => "image_generated",
            StreamEvent::AttachmentProcessed { .. } => "attachment_processed",
            StreamEvent::Metadata { .. } => "metadata",
            StreamEvent::LimitReached { .. } => "limit_reached",
            StreamEvent::Compaction { .. } => "compaction",
            StreamEvent::PhaseTransition { .. } => "phase_transition",
            StreamEvent::Status { .. } => "status",
            StreamEvent::Done => "done",
            StreamEvent::AgentStart => "agent_start",
            StreamEvent::AgentEnd => "agent_end",
            StreamEvent::TurnStart => "turn_start",
            StreamEvent::TurnEnd => "turn_end",
        }
        .to_string();
        Self {
            version: Cow::const_str("1"),
            r#type: Cow::owned(event_type),
            payload,
            event_id,
        }
    }
}
