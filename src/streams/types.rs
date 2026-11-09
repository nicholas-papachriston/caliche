//! Types for durable event streams.
//!
//! Defines the event envelope and trim/position types used by durable stream backends.

use crate::harness::AgentEvent;
use crate::messages::StreamEvent;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Envelope for a single event persisted in a durable stream.
///
/// Wraps either an agent-level observability event or an LLM-level stream event,
/// with a timestamp and optional metadata for replay and audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableEvent {
    /// When the event was recorded (server time).
    pub timestamp: DateTime<Utc>,
    /// Optional metadata (e.g. session_id, turn_id) for filtering or indexing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<DurableEventMetadata>,
    /// The event payload: either an agent event or a stream event.
    #[serde(flatten)]
    pub payload: DurableEventPayload,
}

/// Optional metadata attached to a durable event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurableEventMetadata {
    /// Session identifier.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Turn index within the session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub turn_id: Option<usize>,
    /// Arbitrary extra key-value data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

/// Payload of a durable event: either agent observability or LLM stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data", rename_all = "snake_case")]
pub enum DurableEventPayload {
    /// Agent-level observability event (lifecycle, tools, files, etc.).
    Agent(AgentEvent),
    /// LLM-level stream event (text chunks, tool calls, metadata, done).
    Stream(StreamEvent),
}

impl DurableEvent {
    /// Build a durable event from an agent event.
    pub fn agent(event: AgentEvent, metadata: Option<DurableEventMetadata>) -> Self {
        Self {
            timestamp: Utc::now(),
            metadata,
            payload: DurableEventPayload::Agent(event),
        }
    }

    /// Build a durable event from a stream event.
    pub fn stream(event: StreamEvent, metadata: Option<DurableEventMetadata>) -> Self {
        Self {
            timestamp: Utc::now(),
            metadata,
            payload: DurableEventPayload::Stream(event),
        }
    }

    /// Reference to the inner agent event, if the payload is Agent.
    pub fn as_agent_event(&self) -> Option<&AgentEvent> {
        match &self.payload {
            DurableEventPayload::Agent(e) => Some(e),
            DurableEventPayload::Stream(_) => None,
        }
    }

    /// Reference to the inner stream event, if the payload is Stream.
    pub fn as_stream_event(&self) -> Option<&StreamEvent> {
        match &self.payload {
            DurableEventPayload::Agent(_) => None,
            DurableEventPayload::Stream(e) => Some(e),
        }
    }
}

/// Policy for trimming old events from a stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrimPolicy {
    /// Keep at most `count` most recent events (drop oldest).
    MaxLen(usize),
    /// Remove all events with ID strictly before `min_id`.
    MinId(String),
    /// Remove events older than the given number of seconds (best-effort; backends may approximate).
    MaxAgeSecs(u64),
}

/// Sentinel for "read from the beginning" when passing position to backends.
pub const STREAM_START: &str = "0";
