//! Canonical `StreamEvent` → client wire DTO mapping.
//!
//! Kept in the root SDK so CLI, API, and bindings share one mapping. Transport
//! layers (SSE framing, WebSocket) should call these helpers rather than
//! re-implementing field copies.

use crate::messages::{StreamEvent, TokenUsage};
use caliche_contracts::chat::TokenUsageDto;
use caliche_contracts::stream::{
    ClientStreamEventDto, ClientStreamEventTypeDto, CompactionChangeDto, CompactionChangeKindDto,
    CompactionReportDto, StreamEventEnvelopeDto,
};

pub use caliche_contracts::stream::ACTION_ID_FIELD;
use serde_json::Value;

/// Wire schema version for versioned stream envelopes.
///
/// `action_id` on tool and thinking payloads is additive. Clients that ignore
/// unknown JSON fields keep working, so this version stays `"1"`.
pub const STREAM_EVENT_SCHEMA_VERSION: &str = "1";

/// Public SSE event names for the ROADMAP 2.5 taxonomy.
///
/// [`StreamEvent`] and `AgentEvent` describe the same run from two levels, so
/// both map onto these names. Clients match on them to follow a stream.
pub mod sse_name {
    /// Agent reasoning text.
    pub const AGENT_THINKING: &str = "agent.thinking";
    /// Agent run started.
    pub const AGENT_START: &str = "agent.start";
    /// Agent run finished.
    pub const AGENT_END: &str = "agent.end";
    /// Tool call started.
    pub const TOOL_START: &str = "tool.start";
    /// Tool call finished.
    pub const TOOL_END: &str = "tool.end";
    /// Turn started.
    pub const TURN_START: &str = "turn.start";
    /// Turn finished. This is the terminal event of a successful turn stream.
    pub const TURN_COMPLETE: &str = "turn.complete";
    /// Turn failed. This is the terminal event when the turn returns an error.
    pub const TURN_ERROR: &str = "turn.error";
    /// Session state changed.
    pub const SESSION_UPDATED: &str = "session.updated";
}

/// Map SDK [`TokenUsage`] into the shared contracts DTO.
#[must_use]
pub fn token_usage_to_dto(usage: &TokenUsage) -> TokenUsageDto {
    TokenUsageDto {
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        total_tokens: usage.total_tokens,
        embedding_tokens: usage.embedding_tokens,
        cache_creation_tokens: usage.cache_creation_tokens,
        cache_read_tokens: usage.cache_read_tokens,
    }
}

/// Map a runtime [`StreamEvent`] into the client-facing stream DTO.
pub fn client_event_from_stream_event(
    event: StreamEvent,
) -> Result<ClientStreamEventDto, serde_json::Error> {
    let mut dto = ClientStreamEventDto::done();
    match event {
        StreamEvent::TextChunk { content } => {
            dto.event_type = ClientStreamEventTypeDto::TextChunk;
            dto.content = Some(content);
        }
        StreamEvent::ThinkingChunk { content, action_id } => {
            dto.event_type = ClientStreamEventTypeDto::ThinkingChunk;
            dto.content = Some(content);
            dto.action_id = action_id;
        }
        StreamEvent::ToolCallStart {
            id,
            name,
            arguments,
            thought_signature,
            action_id,
        } => {
            dto.event_type = ClientStreamEventTypeDto::ToolCallStart;
            dto.id = Some(id.into_owned());
            dto.name = Some(name.into_owned());
            dto.arguments = Some(arguments);
            dto.thought_signature = thought_signature;
            dto.action_id = action_id;
        }
        StreamEvent::ToolCallResult {
            name,
            result,
            typed_payload,
            id,
            action_id,
        } => {
            dto.event_type = ClientStreamEventTypeDto::ToolCallResult;
            dto.name = Some(name.into_owned());
            dto.result = Some(result);
            dto.typed_payload = typed_payload;
            dto.id = id;
            dto.action_id = action_id;
        }
        StreamEvent::AudioChunk { data, format } => {
            dto.event_type = ClientStreamEventTypeDto::AudioChunk;
            dto.data = Some(data);
            dto.format = Some(format);
        }
        StreamEvent::ImageGenerated {
            data,
            media_type,
            revised_prompt,
        } => {
            dto.event_type = ClientStreamEventTypeDto::ImageGenerated;
            dto.data = Some(data);
            dto.media_type = Some(media_type);
            dto.revised_prompt = revised_prompt;
        }
        StreamEvent::AttachmentProcessed {
            filename,
            attachment_type,
            summary,
        } => {
            dto.event_type = ClientStreamEventTypeDto::AttachmentProcessed;
            dto.filename = Some(filename);
            dto.attachment_type = Some(attachment_type);
            dto.summary = Some(summary);
        }
        StreamEvent::Metadata {
            usage,
            finish_reason,
            metrics,
        } => {
            dto.event_type = ClientStreamEventTypeDto::Metadata;
            dto.usage = usage.as_ref().map(token_usage_to_dto);
            dto.finish_reason = finish_reason.map(|reason| reason.to_string());
            dto.metrics = metrics.map(serde_json::to_value).transpose()?;
        }
        StreamEvent::LimitReached {
            max_iterations,
            usage,
        } => {
            dto.event_type = ClientStreamEventTypeDto::LimitReached;
            dto.max_iterations = Some(max_iterations);
            dto.usage = Some(token_usage_to_dto(&usage));
        }
        StreamEvent::Compaction {
            tokens_before,
            tokens_after,
            messages_removed,
            summary,
            report,
        } => {
            dto.event_type = ClientStreamEventTypeDto::Compaction;
            dto.tokens_before = Some(tokens_before);
            dto.tokens_after = Some(tokens_after);
            dto.messages_removed = Some(messages_removed);
            dto.summary = summary;
            dto.compaction_report = report
                .as_ref()
                .filter(|r| !r.is_empty())
                .map(compaction_report_to_dto);
        }
        StreamEvent::PhaseTransition {
            phase_index,
            phase_total,
            tool_names,
        } => {
            dto.event_type = ClientStreamEventTypeDto::PhaseTransition;
            dto.phase_index = Some(phase_index);
            dto.phase_total = Some(phase_total);
            dto.tool_names = Some(tool_names);
        }
        StreamEvent::Status { message } => {
            dto.event_type = ClientStreamEventTypeDto::Status;
            dto.content = Some(message);
        }
        StreamEvent::Done => {}
        StreamEvent::AgentStart => dto.event_type = ClientStreamEventTypeDto::AgentStart,
        StreamEvent::AgentEnd => dto.event_type = ClientStreamEventTypeDto::AgentEnd,
        StreamEvent::TurnStart => dto.event_type = ClientStreamEventTypeDto::TurnStart,
        StreamEvent::TurnEnd => dto.event_type = ClientStreamEventTypeDto::TurnEnd,
    }
    Ok(dto)
}

fn compaction_report_to_dto(report: &crate::messages::CompactionReport) -> CompactionReportDto {
    CompactionReportDto {
        changes: report
            .changes
            .iter()
            .map(|change| CompactionChangeDto {
                kind: match change.kind {
                    crate::messages::CompactionChangeKind::Omitted => {
                        CompactionChangeKindDto::Omitted
                    }
                    crate::messages::CompactionChangeKind::Truncated => {
                        CompactionChangeKindDto::Truncated
                    }
                    crate::messages::CompactionChangeKind::Compacted => {
                        CompactionChangeKindDto::Compacted
                    }
                },
                source: change.source.clone(),
                reason: change.reason.clone(),
            })
            .collect(),
    }
}

/// Build a versioned stream envelope DTO from a runtime event.
pub fn envelope_dto_from_stream_event(
    event: StreamEvent,
    event_id: Option<String>,
) -> Result<StreamEventEnvelopeDto, serde_json::Error> {
    let payload = client_event_from_stream_event(event)?;
    Ok(StreamEventEnvelopeDto {
        version: STREAM_EVENT_SCHEMA_VERSION.to_string(),
        event_type: payload.event_type,
        payload,
        event_id,
    })
}

/// Serialize a runtime stream event into the versioned envelope JSON value.
pub fn envelope_value_from_stream_event(
    event: StreamEvent,
    event_id: Option<String>,
) -> Result<Value, serde_json::Error> {
    serde_json::to_value(envelope_dto_from_stream_event(event, event_id)?)
}

/// Public SSE event name for transport framing (ROADMAP 2.5 taxonomy).
///
/// Lifecycle/tool/thinking events use the dotted [`sse_name`] constants.
/// Remaining variants keep the `ClientStreamEventTypeDto` snake_case wire
/// names (`text_chunk`, `compaction`, …).
#[must_use]
pub fn public_sse_event_name(event: &StreamEvent) -> &'static str {
    match event {
        StreamEvent::ThinkingChunk { .. } => sse_name::AGENT_THINKING,
        StreamEvent::ToolCallStart { .. } => sse_name::TOOL_START,
        StreamEvent::ToolCallResult { .. } => sse_name::TOOL_END,
        StreamEvent::TurnEnd | StreamEvent::Done => sse_name::TURN_COMPLETE,
        StreamEvent::TurnStart => sse_name::TURN_START,
        StreamEvent::AgentStart => sse_name::AGENT_START,
        StreamEvent::AgentEnd => sse_name::AGENT_END,
        StreamEvent::TextChunk { .. } => "text_chunk",
        StreamEvent::AudioChunk { .. } => "audio_chunk",
        StreamEvent::ImageGenerated { .. } => "image_generated",
        StreamEvent::AttachmentProcessed { .. } => "attachment_processed",
        StreamEvent::Metadata { .. } => "metadata",
        StreamEvent::LimitReached { .. } => "limit_reached",
        StreamEvent::Compaction { .. } => "compaction",
        StreamEvent::PhaseTransition { .. } => "phase_transition",
        StreamEvent::Status { .. } => "status",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{PerformanceMetrics, TokenUsage};
    use beef::lean::Cow;
    use std::collections::BTreeSet;

    fn assert_maps(
        event: StreamEvent,
        expected: ClientStreamEventTypeDto,
        check: impl FnOnce(&ClientStreamEventDto),
    ) {
        let dto = client_event_from_stream_event(event).expect("map stream event");
        assert_eq!(dto.event_type, expected);
        check(&dto);
    }

    #[test]
    fn maps_every_stream_event_variant_to_client_dto() {
        let mut seen = BTreeSet::new();
        let cases: Vec<(StreamEvent, ClientStreamEventTypeDto)> = vec![
            (
                StreamEvent::TextChunk {
                    content: "hi".into(),
                },
                ClientStreamEventTypeDto::TextChunk,
            ),
            (
                StreamEvent::ThinkingChunk {
                    content: "think".into(),
                    action_id: None,
                },
                ClientStreamEventTypeDto::ThinkingChunk,
            ),
            (
                StreamEvent::ToolCallStart {
                    id: Cow::borrowed("c1"),
                    name: Cow::borrowed("lookup"),
                    arguments: serde_json::json!({"q": "x"}),
                    thought_signature: Some("sig".into()),
                    action_id: None,
                },
                ClientStreamEventTypeDto::ToolCallStart,
            ),
            (
                StreamEvent::ToolCallResult {
                    name: Cow::borrowed("lookup"),
                    result: "ok".into(),
                    typed_payload: Some(serde_json::json!({"ok": true})),
                    id: None,
                    action_id: None,
                },
                ClientStreamEventTypeDto::ToolCallResult,
            ),
            (
                StreamEvent::AudioChunk {
                    data: vec![1, 2],
                    format: "wav".into(),
                },
                ClientStreamEventTypeDto::AudioChunk,
            ),
            (
                StreamEvent::ImageGenerated {
                    data: vec![9],
                    media_type: "image/png".into(),
                    revised_prompt: Some("brighter".into()),
                },
                ClientStreamEventTypeDto::ImageGenerated,
            ),
            (
                StreamEvent::AttachmentProcessed {
                    filename: "a.pdf".into(),
                    attachment_type: "document".into(),
                    summary: "scanned".into(),
                },
                ClientStreamEventTypeDto::AttachmentProcessed,
            ),
            (
                StreamEvent::Metadata {
                    usage: Some(TokenUsage::from_counts(Some(1), Some(2), None)),
                    finish_reason: Some(Cow::borrowed("stop")),
                    metrics: Some(PerformanceMetrics {
                        total_duration_ms: Some(12),
                        ..PerformanceMetrics::default()
                    }),
                },
                ClientStreamEventTypeDto::Metadata,
            ),
            (
                StreamEvent::LimitReached {
                    max_iterations: 3,
                    usage: TokenUsage::from_counts(Some(1), Some(1), Some(2)),
                },
                ClientStreamEventTypeDto::LimitReached,
            ),
            (
                StreamEvent::Compaction {
                    tokens_before: 10,
                    tokens_after: 4,
                    messages_removed: 2,
                    summary: Some("trimmed".into()),
                    report: None,
                },
                ClientStreamEventTypeDto::Compaction,
            ),
            (
                StreamEvent::PhaseTransition {
                    phase_index: 1,
                    phase_total: 2,
                    tool_names: vec!["fs".into()],
                },
                ClientStreamEventTypeDto::PhaseTransition,
            ),
            (StreamEvent::Done, ClientStreamEventTypeDto::Done),
            (
                StreamEvent::AgentStart,
                ClientStreamEventTypeDto::AgentStart,
            ),
            (StreamEvent::AgentEnd, ClientStreamEventTypeDto::AgentEnd),
            (StreamEvent::TurnStart, ClientStreamEventTypeDto::TurnStart),
            (StreamEvent::TurnEnd, ClientStreamEventTypeDto::TurnEnd),
        ];

        assert_eq!(cases.len(), 16, "cover every StreamEvent variant");
        for (event, expected) in cases {
            let dto = client_event_from_stream_event(event).expect("map");
            assert_eq!(dto.event_type, expected);
            seen.insert(format!("{expected:?}"));
        }
        assert_eq!(seen.len(), 16);
    }

    #[test]
    fn compaction_event_maps_typed_report() {
        let mut report = crate::messages::CompactionReport::new();
        report.record(
            crate::messages::CompactionChangeKind::Omitted,
            "jev",
            "drop tool result c1",
        );
        assert_maps(
            StreamEvent::Compaction {
                tokens_before: 10,
                tokens_after: 4,
                messages_removed: 1,
                summary: Some("trimmed".into()),
                report: Some(report),
            },
            ClientStreamEventTypeDto::Compaction,
            |dto| {
                let mapped = dto
                    .compaction_report
                    .as_ref()
                    .expect("typed compaction report");
                assert_eq!(mapped.changes.len(), 1);
                assert_eq!(mapped.changes[0].kind, CompactionChangeKindDto::Omitted);
                assert_eq!(mapped.changes[0].source, "jev");
            },
        );
    }

    #[test]
    fn tool_call_start_preserves_thought_signature_and_arguments() {
        assert_maps(
            StreamEvent::ToolCallStart {
                id: Cow::borrowed("id-1"),
                name: Cow::borrowed("search"),
                arguments: serde_json::json!({"q": "weather"}),
                thought_signature: Some("sig-abc".into()),
                action_id: None,
            },
            ClientStreamEventTypeDto::ToolCallStart,
            |dto| {
                assert_eq!(dto.id.as_deref(), Some("id-1"));
                assert_eq!(dto.name.as_deref(), Some("search"));
                assert_eq!(dto.arguments, Some(serde_json::json!({"q": "weather"})));
                assert_eq!(dto.thought_signature.as_deref(), Some("sig-abc"));
            },
        );
    }

    #[test]
    fn envelope_value_sets_version_type_and_event_id() {
        let value = envelope_value_from_stream_event(
            StreamEvent::TextChunk {
                content: "chunk".into(),
            },
            Some("evt-42".into()),
        )
        .expect("envelope value");
        assert_eq!(value["version"], STREAM_EVENT_SCHEMA_VERSION);
        assert_eq!(value["type"], "text_chunk");
        assert_eq!(value["event_id"], "evt-42");
        assert_eq!(value["payload"]["type"], "text_chunk");
        assert_eq!(value["payload"]["content"], "chunk");
    }

    #[test]
    fn token_usage_mapping_preserves_embedding_tokens() {
        let usage = TokenUsage {
            prompt_tokens: Some(1),
            completion_tokens: Some(2),
            embedding_tokens: Some(4),
            total_tokens: Some(7),
            context_window: Some(8_192),
            cache_creation_tokens: Some(3),
            cache_read_tokens: Some(5),
            last_prompt_tokens: None,
        };
        let dto = token_usage_to_dto(&usage);
        assert_eq!(dto.prompt_tokens, Some(1));
        assert_eq!(dto.completion_tokens, Some(2));
        assert_eq!(dto.embedding_tokens, Some(4));
        assert_eq!(dto.total_tokens, Some(7));
        assert_eq!(dto.cache_creation_tokens, Some(3));
        assert_eq!(dto.cache_read_tokens, Some(5));
    }

    #[test]
    fn public_sse_event_name_uses_roadmap_taxonomy() {
        assert_eq!(
            public_sse_event_name(&StreamEvent::ThinkingChunk {
                content: "think".into(),
                action_id: None,
            }),
            "agent.thinking"
        );
        assert_eq!(
            public_sse_event_name(&StreamEvent::ToolCallStart {
                id: Cow::borrowed("c1"),
                name: Cow::borrowed("lookup"),
                arguments: serde_json::json!({}),
                thought_signature: None,
                action_id: None,
            }),
            "tool.start"
        );
        assert_eq!(
            public_sse_event_name(&StreamEvent::ToolCallResult {
                name: Cow::borrowed("lookup"),
                result: "ok".into(),
                typed_payload: None,
                id: None,
                action_id: None,
            }),
            "tool.end"
        );
        assert_eq!(
            public_sse_event_name(&StreamEvent::TurnEnd),
            "turn.complete"
        );
        assert_eq!(public_sse_event_name(&StreamEvent::Done), "turn.complete");
        assert_eq!(
            public_sse_event_name(&StreamEvent::TextChunk {
                content: "hi".into(),
            }),
            "text_chunk"
        );
        assert_eq!(
            public_sse_event_name(&StreamEvent::Compaction {
                tokens_before: 10,
                tokens_after: 4,
                messages_removed: 2,
                summary: None,
                report: None,
            }),
            "compaction"
        );
    }
}
