use caliche::llm::provider_utils::StreamingToolCallAccumulator;
use caliche::messages::StreamEvent;

#[test]
fn accumulator_assembles_fragmented_openai_tool_calls() {
    let mut acc = StreamingToolCallAccumulator::new();
    acc.accumulate(
        0,
        Some("call_1".to_string()),
        Some("get_weather".to_string()),
        None,
    );
    acc.accumulate(0, None, None, Some("{\"loc".to_string()));
    acc.accumulate(0, None, None, Some("ation\":\"NYC\"}".to_string()));

    let evt = acc.finalize(0, true).expect("tool call should finalize");
    match evt {
        StreamEvent::ToolCallStart {
            id,
            name,
            arguments,
            ..
        } => {
            assert_eq!(id, "call_1");
            assert_eq!(name, "get_weather");
            assert_eq!(arguments["location"], "NYC");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn accumulator_dedupes_emitted_ids() {
    let mut acc = StreamingToolCallAccumulator::new();
    acc.accumulate(
        0,
        Some("call_1".to_string()),
        Some("search".to_string()),
        Some("{}".to_string()),
    );
    assert!(acc.finalize(0, false).is_some());
    acc.accumulate(
        1,
        Some("call_1".to_string()),
        Some("search".to_string()),
        Some("{}".to_string()),
    );
    assert!(acc.finalize(1, false).is_none());
}

#[test]
fn accumulator_cumulative_replaces_growing_args() {
    let mut acc = StreamingToolCallAccumulator::new();
    acc.accumulate_cumulative(0, None, Some("fn".to_string()), Some("{\"a\"".to_string()));
    acc.accumulate_cumulative(0, None, None, Some("{\"a\":\"b\"}".to_string()));

    let evt = acc.finalize(0, true).expect("tool call should finalize");
    match evt {
        StreamEvent::ToolCallStart { arguments, .. } => {
            assert_eq!(arguments["a"], "b");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn finalize_all_flushes_remaining_tool_calls() {
    let mut acc = StreamingToolCallAccumulator::new();
    acc.accumulate(
        0,
        Some("a".to_string()),
        Some("one".to_string()),
        Some("{}".to_string()),
    );
    acc.accumulate(
        1,
        Some("b".to_string()),
        Some("two".to_string()),
        Some("{}".to_string()),
    );

    let events = acc.finalize_all(false);
    assert_eq!(events.len(), 2);
}
