use caliche::llm::provider_utils::{AnthropicStreamFeed, AnthropicStreamParser};
use caliche::messages::StreamEvent;

#[test]
fn feed_assembles_tool_call_from_content_blocks() {
    let mut parser = AnthropicStreamParser::new();

    let start_events = parser.feed(AnthropicStreamFeed::ContentBlockStart {
        index: Some(0),
        block_type: "tool_use",
        tool_id: Some("call_1"),
        tool_name: Some("get_weather"),
    });
    assert!(start_events.is_empty());

    let delta_events = parser.feed(AnthropicStreamFeed::ContentBlockDelta {
        index: Some(0),
        text: None,
        thinking: None,
        partial_json: Some(r#"{"city":"NYC"}"#),
    });
    assert!(delta_events.is_empty());

    let stop_events = parser.feed(AnthropicStreamFeed::ContentBlockStop { index: Some(0) });
    assert_eq!(stop_events.len(), 1);
    match &stop_events[0] {
        StreamEvent::ToolCallStart { id, name, .. } => {
            assert_eq!(id, "call_1");
            assert_eq!(name, "get_weather");
        }
        other => panic!("expected ToolCallStart, got {other:?}"),
    }

    let finish_events = parser.finish();
    assert_eq!(finish_events.len(), 1);
    assert!(matches!(finish_events[0], StreamEvent::Done));
}

#[test]
fn feed_emits_text_chunks() {
    let mut parser = AnthropicStreamParser::new();
    let events = parser.feed(AnthropicStreamFeed::ContentBlockDelta {
        index: Some(0),
        text: Some("hello"),
        thinking: None,
        partial_json: None,
    });
    assert_eq!(events.len(), 1);
    match &events[0] {
        StreamEvent::TextChunk { content } => assert_eq!(content, "hello"),
        other => panic!("expected TextChunk, got {other:?}"),
    }
}

#[test]
fn feed_ignores_empty_text_deltas() {
    let mut parser = AnthropicStreamParser::new();
    let events = parser.feed(AnthropicStreamFeed::ContentBlockDelta {
        index: Some(0),
        text: Some(""),
        thinking: None,
        partial_json: None,
    });
    assert!(events.is_empty());
}

#[test]
fn feed_emits_thinking_chunks() {
    let mut parser = AnthropicStreamParser::new();
    let events = parser.feed(AnthropicStreamFeed::ContentBlockDelta {
        index: Some(0),
        text: None,
        thinking: Some("consider the cache"),
        partial_json: None,
    });
    assert_eq!(events.len(), 1);
    match &events[0] {
        StreamEvent::ThinkingChunk { content, .. } => {
            assert_eq!(content, "consider the cache");
        }
        other => panic!("expected ThinkingChunk, got {other:?}"),
    }
}

#[test]
fn feed_emits_placeholder_for_redacted_thinking() {
    let mut parser = AnthropicStreamParser::new();
    let events = parser.feed(AnthropicStreamFeed::ContentBlockStart {
        index: Some(0),
        block_type: "redacted_thinking",
        tool_id: None,
        tool_name: None,
    });
    assert_eq!(events.len(), 1);
    match &events[0] {
        StreamEvent::ThinkingChunk { content, .. } => assert_eq!(content, "[redacted]"),
        other => panic!("expected ThinkingChunk, got {other:?}"),
    }
}
