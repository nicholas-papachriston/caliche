use caliche::messages::{
    STREAM_EVENT_SCHEMA_VERSION, StreamEvent, envelope_value_from_stream_event,
};
use caliche_contracts::stream::{ClientStreamEventDto, StreamEventEnvelopeDto};
use serde::Serialize;
use serde_json::{Value, json};

pub const EVENT_SCHEMA_VERSION: &str = STREAM_EVENT_SCHEMA_VERSION;

pub fn envelope_from_serializable<T: Serialize>(event: &T) -> serde_json::Result<Value> {
    let payload = serde_json::to_value(event)?;
    let event_type = payload
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("event");
    Ok(json!({
        "version": EVENT_SCHEMA_VERSION,
        "type": event_type,
        "payload": payload
    }))
}

pub fn envelope_from_stream_event(
    event: StreamEvent,
    event_id: Option<String>,
) -> serde_json::Result<Value> {
    // Canonical mapping lives in the root SDK (`messages::stream_contracts`).
    envelope_value_from_stream_event(event, event_id)
}

/// Encode an SSE frame with both `event:` and `data:` fields.
pub fn encode_sse_event(event_name: &str, envelope: &Value) -> serde_json::Result<String> {
    let json = serde_json::to_string(envelope)?;
    Ok(format!("event: {event_name}\ndata: {json}\n\n"))
}

/// Encode an SSE data-only frame (compat; prefer [`encode_sse_event`] for StreamEvents).
pub fn encode_sse_data(envelope: &Value) -> serde_json::Result<String> {
    let json = serde_json::to_string(envelope)?;
    Ok(format!("data: {}\n\n", json))
}

pub fn encode_sse_error(error: &str) -> serde_json::Result<String> {
    let payload = ClientStreamEventDto::error(error);
    let envelope = StreamEventEnvelopeDto {
        version: EVENT_SCHEMA_VERSION.to_string(),
        event_type: payload.event_type,
        payload,
        event_id: None,
    };
    encode_sse_data(&serde_json::to_value(envelope)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliche::messages::{TokenUsage, client_event_from_stream_event, public_sse_event_name};

    #[test]
    fn envelope_from_stream_event_delegates_to_sdk_mapping() {
        let value = envelope_from_stream_event(
            StreamEvent::TextChunk {
                content: "chunk".into(),
            },
            Some("evt-1".into()),
        )
        .expect("envelope");
        assert_eq!(value["version"], EVENT_SCHEMA_VERSION);
        assert_eq!(value["type"], "text_chunk");
        assert_eq!(value["event_id"], "evt-1");
        assert_eq!(value["payload"]["content"], "chunk");
    }

    #[test]
    fn encode_sse_error_wraps_error_dto() {
        let encoded = encode_sse_error("boom").expect("encode");
        assert!(encoded.starts_with("data: "));
        assert!(encoded.ends_with("\n\n"));
        let json = encoded.trim_start_matches("data: ").trim_end();
        let value: Value = serde_json::from_str(json).expect("json");
        assert_eq!(value["type"], "error");
        assert_eq!(value["payload"]["error"], "boom");
    }

    #[test]
    fn client_event_mapping_covers_limit_reached_usage() {
        let dto = client_event_from_stream_event(StreamEvent::LimitReached {
            max_iterations: 2,
            usage: TokenUsage::from_counts(Some(1), Some(1), Some(2)),
        })
        .expect("map");
        assert_eq!(dto.max_iterations, Some(2));
        assert_eq!(dto.usage.as_ref().and_then(|u| u.total_tokens), Some(2));
    }

    #[test]
    fn encode_sse_event_maps_thinking_chunk_to_agent_thinking() {
        let event = StreamEvent::ThinkingChunk {
            content: "hmm".into(),
            action_id: None,
        };
        let event_name = public_sse_event_name(&event);
        assert_eq!(event_name, "agent.thinking");
        let envelope = envelope_from_stream_event(event, None).expect("envelope");
        let framed = encode_sse_event(event_name, &envelope).expect("encode");
        assert!(framed.starts_with("event: agent.thinking\n"));
        assert!(framed.contains("data: "));
        assert!(framed.ends_with("\n\n"));
    }
}
