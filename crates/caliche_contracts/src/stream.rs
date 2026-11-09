use serde::{Deserialize, Serialize};

use crate::chat::TokenUsageDto;

/// JSON object key for the UUID v7 shared by one action's start and result.
pub const ACTION_ID_FIELD: &str = "action_id";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientStreamEventTypeDto {
    TextChunk,
    ThinkingChunk,
    Content,
    Metadata,
    Usage,
    Error,
    Done,
    ToolCallStart,
    ToolCallResult,
    LimitReached,
    AudioChunk,
    ImageGenerated,
    AttachmentProcessed,
    Compaction,
    PhaseTransition,
    Status,
    AgentStart,
    AgentEnd,
    TurnStart,
    TurnEnd,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientStreamEventDto {
    #[serde(rename = "type")]
    pub event_type: ClientStreamEventTypeDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsageDto>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Time-sortable UUID v7 shared by a tool call or thinking segment's start and result.
    ///
    /// Additive on stream schema version 1. Absent on older payloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typed_payload: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thought_signature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Vec<u8>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attachment_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_before: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_after: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages_removed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_total: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_names: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compaction_report: Option<CompactionReportDto>,
}

/// Machine-readable compaction mutation (kind / source / reason).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionChangeKindDto {
    Omitted,
    Truncated,
    Compacted,
}

/// One omitted, truncated, or compacted change on the wire.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionChangeDto {
    pub kind: CompactionChangeKindDto,
    pub source: String,
    pub reason: String,
}

/// Audit trail persisted with a compaction summary.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompactionReportDto {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changes: Vec<CompactionChangeDto>,
}

impl ClientStreamEventDto {
    #[must_use]
    pub const fn done() -> Self {
        Self {
            event_type: ClientStreamEventTypeDto::Done,
            content: None,
            usage: None,
            error: None,
            name: None,
            id: None,
            action_id: None,
            arguments: None,
            result: None,
            typed_payload: None,
            thought_signature: None,
            finish_reason: None,
            metrics: None,
            data: None,
            format: None,
            media_type: None,
            revised_prompt: None,
            filename: None,
            attachment_type: None,
            summary: None,
            max_iterations: None,
            tokens_before: None,
            tokens_after: None,
            messages_removed: None,
            phase_index: None,
            phase_total: None,
            tool_names: None,
            compaction_report: None,
        }
    }

    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            event_type: ClientStreamEventTypeDto::Error,
            error: Some(message.into()),
            ..Self::done()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamEventEnvelopeDto {
    pub version: String,
    #[serde(rename = "type")]
    pub event_type: ClientStreamEventTypeDto,
    pub payload: ClientStreamEventDto,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_event_envelope_dto_roundtrips_tool_result_payload() {
        let mut payload = ClientStreamEventDto::done();
        payload.event_type = ClientStreamEventTypeDto::ToolCallResult;
        payload.id = Some("call-1".into());
        payload.name = Some("lookup".into());
        payload.result = Some(r#"{"answer":42}"#.into());
        payload.typed_payload = Some(serde_json::json!({"answer": 42}));

        let envelope = StreamEventEnvelopeDto {
            version: "1".into(),
            event_type: ClientStreamEventTypeDto::ToolCallResult,
            payload,
            event_id: Some("evt-9".into()),
        };
        let value = serde_json::to_value(&envelope).expect("serialize");
        assert_eq!(value["type"], "tool_call_result");
        assert_eq!(value["version"], "1");
        assert_eq!(value["event_id"], "evt-9");
        assert_eq!(value["payload"]["type"], "tool_call_result");
        assert_eq!(value["payload"]["typed_payload"]["answer"], 42);
        assert_eq!(value["payload"]["result"], r#"{"answer":42}"#);

        let decoded: StreamEventEnvelopeDto = serde_json::from_value(value).expect("deserialize");
        assert_eq!(decoded.event_id.as_deref(), Some("evt-9"));
        assert_eq!(decoded.event_type, ClientStreamEventTypeDto::ToolCallResult);
        assert_eq!(decoded.payload.name.as_deref(), Some("lookup"));
        assert_eq!(
            decoded.payload.typed_payload,
            Some(serde_json::json!({"answer": 42}))
        );
    }

    #[test]
    fn client_stream_event_dto_error_and_done_helpers() {
        let done = ClientStreamEventDto::done();
        assert_eq!(done.event_type, ClientStreamEventTypeDto::Done);
        assert!(done.error.is_none());

        let err = ClientStreamEventDto::error("boom");
        assert_eq!(err.event_type, ClientStreamEventTypeDto::Error);
        assert_eq!(err.error.as_deref(), Some("boom"));
    }

    #[test]
    fn client_stream_event_dto_roundtrips_media_lifecycle_compaction_and_phase_fields() {
        let mut audio = ClientStreamEventDto::done();
        audio.event_type = ClientStreamEventTypeDto::AudioChunk;
        audio.data = Some(vec![1, 2, 3]);
        audio.format = Some("wav".into());

        let mut image = ClientStreamEventDto::done();
        image.event_type = ClientStreamEventTypeDto::ImageGenerated;
        image.data = Some(vec![9]);
        image.media_type = Some("image/png".into());
        image.revised_prompt = Some("brighter".into());

        let mut attachment = ClientStreamEventDto::done();
        attachment.event_type = ClientStreamEventTypeDto::AttachmentProcessed;
        attachment.filename = Some("notes.pdf".into());
        attachment.attachment_type = Some("document".into());
        attachment.summary = Some("scanned".into());

        let mut compaction = ClientStreamEventDto::done();
        compaction.event_type = ClientStreamEventTypeDto::Compaction;
        compaction.tokens_before = Some(100);
        compaction.tokens_after = Some(40);
        compaction.messages_removed = Some(3);
        compaction.summary = Some("trimmed".into());
        compaction.compaction_report = Some(CompactionReportDto {
            changes: vec![CompactionChangeDto {
                kind: CompactionChangeKindDto::Omitted,
                source: "jev".into(),
                reason: "drop tool result".into(),
            }],
        });

        let mut phase = ClientStreamEventDto::done();
        phase.event_type = ClientStreamEventTypeDto::PhaseTransition;
        phase.phase_index = Some(1);
        phase.phase_total = Some(3);
        phase.tool_names = Some(vec!["fs".into(), "bash".into()]);

        let mut limit = ClientStreamEventDto::done();
        limit.event_type = ClientStreamEventTypeDto::LimitReached;
        limit.max_iterations = Some(5);
        limit.usage = Some(TokenUsageDto {
            prompt_tokens: Some(1),
            completion_tokens: Some(2),
            total_tokens: Some(3),
            embedding_tokens: None,
            cache_creation_tokens: None,
            cache_read_tokens: None,
        });

        for payload in [audio, image, attachment, compaction, phase, limit] {
            let event_type = payload.event_type;
            let envelope = StreamEventEnvelopeDto {
                version: "1".into(),
                event_type,
                payload: payload.clone(),
                event_id: Some("evt-media".into()),
            };
            let value = serde_json::to_value(&envelope).expect("serialize");
            let decoded: StreamEventEnvelopeDto =
                serde_json::from_value(value).expect("deserialize");
            assert_eq!(decoded.event_type, event_type);
            assert_eq!(decoded.payload, payload);
            assert_eq!(decoded.event_id.as_deref(), Some("evt-media"));
        }
    }
}
