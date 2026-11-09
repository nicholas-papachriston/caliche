use caliche_contracts::rpc::methods as rpc_methods;
use caliche_web::api::client::{
    done_stream_event, error_stream_event, format_rpc_error, stream_event_from_envelope_str,
    stream_event_from_envelope_value,
};
use caliche_web::api::types::StreamingEventType;
use serde_json::json;

#[test]
fn rpc_error_formats_openai_quota_as_actionable_message() {
    let error = json!({
        "code": -32500,
        "data": { "status": 500 },
        "message": "HTTP 500: Failed to process chat: Failed to get LLM completion: openai API error (429 Too Many Requests): { \"error\": { \"code\": \"insufficient_quota\" } }"
    });

    assert_eq!(
        format_rpc_error(rpc_methods::CHAT_SEND, &error),
        "OpenAI quota exceeded. Check your OpenAI plan/billing, update the API key, or switch to another provider/model."
    );
}

#[test]
fn rpc_error_uses_message_without_dumping_full_error_json() {
    let error = json!({
        "code": -32500,
        "data": { "status": 500 },
        "message": "HTTP 500: Failed to process chat"
    });

    assert_eq!(
        format_rpc_error(rpc_methods::CHAT_SEND, &error),
        "chat.send failed: HTTP 500: Failed to process chat"
    );
}

#[test]
fn parses_versioned_stream_envelope_payload() {
    let event = stream_event_from_envelope_str(
        r#"{
            "version": "1",
            "type": "text_chunk",
            "payload": {
                "type": "text_chunk",
                "content": "hello"
            }
        }"#,
    )
    .expect("versioned stream envelope should parse");

    assert_eq!(event.event_type, StreamingEventType::TextChunk);
    assert_eq!(event.content.as_deref(), Some("hello"));
}

#[test]
fn rejects_legacy_flat_stream_event_payload() {
    let err = stream_event_from_envelope_str(
        r#"{
            "type": "text_chunk",
            "content": "hello"
        }"#,
    )
    .expect_err("flat stream events should not bypass the shared envelope");

    assert!(err.to_string().contains("stream event envelope"));
}

#[test]
fn parses_rpc_stream_event_envelope_value() {
    let event = stream_event_from_envelope_value(json!({
        "version": "1",
        "type": "tool_call_result",
        "payload": {
            "type": "tool_call_result",
            "name": "read_file",
            "result": "ok"
        }
    }))
    .expect("RPC stream event envelope should parse");

    assert_eq!(event.event_type, StreamingEventType::ToolCallResult);
    assert_eq!(event.name.as_deref(), Some("read_file"));
    assert_eq!(event.result.as_deref(), Some("ok"));
}

#[test]
fn synthesized_stream_events_are_terminal_or_error_events() {
    let done = done_stream_event();
    assert_eq!(done.event_type, StreamingEventType::Done);
    assert!(done.error.is_none());

    let error = error_stream_event("network down");
    assert_eq!(error.event_type, StreamingEventType::Error);
    assert_eq!(error.error.as_deref(), Some("network down"));
}
