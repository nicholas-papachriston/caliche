use beef::lean::Cow;
use caliche::messages::{StreamEvent, TokenUsage};

fn event_type(event: &StreamEvent) -> String {
    serde_json::to_value(event)
        .expect("stream event serializes")
        .get("type")
        .and_then(|v| v.as_str())
        .expect("stream event has type tag")
        .to_string()
}

#[test]
fn tool_events_use_core_contract_names() {
    let start = StreamEvent::ToolCallStart {
        id: Cow::borrowed("call-1"),
        name: Cow::borrowed("calculator"),
        arguments: serde_json::json!({"x": 1}),
        thought_signature: None,
        action_id: None,
    };
    assert_eq!(event_type(&start), "tool_call_start");

    let result = StreamEvent::ToolCallResult {
        name: Cow::borrowed("calculator"),
        result: "42".to_string(),
        typed_payload: None,
        id: None,
        action_id: None,
    };
    assert_eq!(event_type(&result), "tool_call_result");
}

#[test]
fn lifecycle_events_are_observable() {
    for (event, expected_type) in [
        (StreamEvent::AgentStart, "agent_start"),
        (StreamEvent::AgentEnd, "agent_end"),
        (StreamEvent::TurnStart, "turn_start"),
        (StreamEvent::TurnEnd, "turn_end"),
    ] {
        assert_eq!(event_type(&event), expected_type);
    }
}

#[test]
fn limit_reached_includes_usage() {
    let event = StreamEvent::LimitReached {
        max_iterations: 3,
        usage: TokenUsage {
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            ..Default::default()
        },
    };
    assert_eq!(event_type(&event), "limit_reached");
    let value = serde_json::to_value(&event).expect("limit reached serializes");
    assert_eq!(
        value.get("max_iterations").and_then(|v| v.as_u64()),
        Some(3)
    );
}
