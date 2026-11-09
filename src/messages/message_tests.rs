use super::*;

#[test]
fn message_text_roundtrips_through_json() {
    let msg = Message {
        role: MessageRole::User,
        content: MessageContent::Text(Arc::new("hello world".to_string())),
        metadata: None,
    };

    let json = serde_json::to_string(&msg).expect("serialize message");
    let parsed: Message = serde_json::from_str(&json).expect("deserialize message");
    assert_eq!(parsed.role, MessageRole::User);
    assert_eq!(parsed.extract_text(), "hello world");
}

#[test]
fn reasoning_effort_wire_and_budget_mappings() {
    assert_eq!(ReasoningEffort::Low.as_str(), "low");
    assert_eq!(ReasoningEffort::Medium.as_str(), "medium");
    assert_eq!(ReasoningEffort::High.as_str(), "high");

    assert_eq!(ReasoningEffort::Low.anthropic_budget_tokens(), 1_024);
    assert_eq!(ReasoningEffort::Medium.anthropic_budget_tokens(), 8_192);
    assert_eq!(ReasoningEffort::High.anthropic_budget_tokens(), 32_768);

    assert_eq!(ReasoningEffort::Low.google_thinking_budget(), 512);
    assert_eq!(ReasoningEffort::Medium.google_thinking_budget(), 4_096);
    assert_eq!(ReasoningEffort::High.google_thinking_budget(), 16_384);
}

#[test]
fn completion_request_preserves_reasoning_effort() {
    let request = CompletionRequest {
        messages: Arc::new(vec![Message::user("hi")]),
        max_tokens: Some(100),
        temperature: Some(0.2),
        reasoning_effort: Some(ReasoningEffort::High),
        tools: None,
        tool_choice: None,
        response_format: None,
    };
    let json = serde_json::to_value(&request).expect("serialize completion request");
    assert_eq!(json["reasoning_effort"], "high");
    let parsed: CompletionRequest =
        serde_json::from_value(json).expect("deserialize completion request");
    assert_eq!(parsed.reasoning_effort, Some(ReasoningEffort::High));
}

#[test]
fn completion_request_omits_reasoning_effort_when_none() {
    let request = CompletionRequest::new(Arc::new(vec![Message::user("hi")]));
    let json = serde_json::to_value(&request).expect("serialize completion request");
    assert!(json.get("reasoning_effort").is_none());
}

#[test]
fn message_constructors_set_role_and_text() {
    assert_eq!(Message::user("u").role, MessageRole::User);
    assert_eq!(Message::assistant("a").extract_text(), "a");
    assert_eq!(Message::system("s").role, MessageRole::System);
    assert_eq!(Message::tool("t").role, MessageRole::Tool);
}

#[test]
fn assistant_reasoning_round_trips_through_metadata() {
    let message = Message::assistant_with_reasoning("answer", "hidden chain");
    assert_eq!(message.extract_text(), "answer");
    assert_eq!(message.reasoning_content(), Some("hidden chain"));

    let json = serde_json::to_string(&message).expect("serialize");
    let parsed: Message = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.reasoning_content(), Some("hidden chain"));
    assert_eq!(parsed.extract_text(), "answer");

    let blank = Message::assistant_with_reasoning("answer", "   ");
    assert!(blank.reasoning_content().is_none());

    let mut kept = Message::assistant_with_reasoning("answer", "hidden chain");
    kept.metadata
        .as_mut()
        .and_then(|meta| meta.as_object_mut())
        .expect("metadata object")
        .insert("cache".to_string(), serde_json::json!("keep"));
    kept.clear_reasoning_content();
    assert!(kept.reasoning_content().is_none());
    assert_eq!(
        kept.metadata
            .as_ref()
            .and_then(|meta| meta.get("cache"))
            .and_then(|value| value.as_str()),
        Some("keep")
    );
    assert!(
        !serde_json::to_string(&kept)
            .expect("serialize")
            .contains(super::REASONING_CONTENT_METADATA_KEY)
    );
}

#[test]
fn message_structured_tool_calls_roundtrip_json() {
    let msg = Message {
        role: MessageRole::Assistant,
        content: MessageContent::Structured {
            text: Some(Arc::new("calling".into())),
            tool_calls: Some(vec![ToolCallInfo::from_json(
                "call-1",
                "search",
                r#"{"q":"weather"}"#,
            )]),
            tool_call_id: None,
        },
        metadata: None,
    };
    let json = serde_json::to_value(&msg).expect("serialize");
    let parsed: Message = serde_json::from_value(json).expect("deserialize");
    assert_eq!(parsed.extract_text(), "calling");
    let calls = parsed.content.as_tool_calls().expect("tool calls");
    assert_eq!(calls[0].name_str(), "search");
    assert_eq!(calls[0].arguments, serde_json::json!({"q": "weather"}));
}

#[test]
fn completion_content_extract_helpers() {
    let text = CompletionContent::Text {
        content: "hi".into(),
    };
    assert_eq!(text.as_text(), "hi");
    assert!(!text.has_tool_calls());

    let with_tools = CompletionContent::WithToolCalls {
        content: "ok".into(),
        tool_calls: vec![ToolCallInfo::from_json("1", "calc", r#"{"x":1}"#)],
    };
    assert_eq!(with_tools.as_text(), "ok");
    assert!(with_tools.has_tool_calls());
    assert_eq!(with_tools.extract_tool_calls()[0].name_str(), "calc");
}

#[test]
fn completion_response_with_tool_calls_roundtrips() {
    let response = CompletionResponse {
        content: CompletionContent::WithToolCalls {
            content: String::new(),
            tool_calls: vec![ToolCallInfo::from_json("c1", "fs_read", r#"{"path":"a"}"#)],
        },
        finish_reason: Some(Cow::borrowed("tool_calls")),
        usage: Some(TokenUsage::from_counts(Some(3), Some(7), None)),
        reasoning_content: None,
    };
    let json = serde_json::to_value(&response).expect("serialize");
    let parsed: CompletionResponse = serde_json::from_value(json).expect("deserialize");
    assert_eq!(parsed.finish_reason.as_deref(), Some("tool_calls"));
    assert_eq!(parsed.usage.and_then(|u| u.total_tokens), Some(10));
    assert_eq!(parsed.content.as_tool_calls()[0].name_str(), "fs_read");
}

#[test]
fn completion_response_multimodal_tool_calls_roundtrips_flattened_json() {
    let response = CompletionResponse {
        content: CompletionContent::Multimodal {
            content: "see image".into(),
            media: vec![CompletionMediaItem {
                media_type: "image/png".into(),
                data: b"raw".to_vec(),
                filename: Some("a.png".into()),
            }],
            tool_calls: vec![ToolCallInfo::from_json("c2", "lookup", r#"{"q":"x"}"#)],
        },
        finish_reason: Some(Cow::borrowed("tool_calls")),
        usage: None,
        reasoning_content: None,
    };
    let json = serde_json::to_value(&response).expect("serialize");
    assert!(json.get("tool_calls").is_some());
    assert!(json.get("media").is_some());
    let parsed: CompletionResponse = serde_json::from_value(json).expect("deserialize");
    assert_eq!(parsed.content.as_text(), "see image");
    assert!(parsed.content.has_tool_calls());
    assert_eq!(parsed.content.as_tool_calls()[0].name_str(), "lookup");
    match &parsed.content {
        CompletionContent::Multimodal { media, .. } => {
            assert_eq!(media.len(), 1);
            assert_eq!(media[0].media_type, "image/png");
            assert_eq!(media[0].filename.as_deref(), Some("a.png"));
        }
        other => panic!("expected Multimodal, got {other:?}"),
    }
}

#[test]
fn stream_event_wire_tags_roundtrip() {
    let events = [
        StreamEvent::TextChunk {
            content: "x".into(),
        },
        StreamEvent::ToolCallStart {
            id: Cow::borrowed("1"),
            name: Cow::borrowed("t"),
            arguments: serde_json::json!({}),
            thought_signature: None,
            action_id: None,
        },
        StreamEvent::Metadata {
            usage: None,
            finish_reason: Some(Cow::borrowed("stop")),
            metrics: None,
        },
        StreamEvent::LimitReached {
            max_iterations: 3,
            usage: TokenUsage::default(),
        },
        StreamEvent::Done,
        StreamEvent::AgentStart,
        StreamEvent::TurnEnd,
    ];
    for event in events {
        let json = serde_json::to_value(&event).expect("serialize stream event");
        let tag = json
            .get("type")
            .and_then(|v| v.as_str())
            .expect("type tag")
            .to_string();
        assert!(!tag.is_empty());
        let parsed: StreamEvent = serde_json::from_value(json).expect("deserialize stream event");
        let again = serde_json::to_value(&parsed).expect("re-serialize");
        assert_eq!(again["type"], tag);
    }
}

#[test]
fn stream_event_envelope_v1_sets_type_and_version() {
    let envelope = StreamEventEnvelope::v1(
        StreamEvent::TextChunk {
            content: "hi".into(),
        },
        Some("evt-1".into()),
    );
    assert_eq!(envelope.version.as_ref(), "1");
    assert_eq!(envelope.r#type.as_ref(), "text_chunk");
    assert_eq!(envelope.event_id.as_deref(), Some("evt-1"));
}

#[test]
fn stream_event_envelope_v1_names_every_stream_variant() {
    let cases = [
        (
            StreamEvent::TextChunk {
                content: "t".into(),
            },
            "text_chunk",
        ),
        (
            StreamEvent::ThinkingChunk {
                content: "think".into(),
                action_id: None,
            },
            "thinking_chunk",
        ),
        (
            StreamEvent::ToolCallStart {
                id: Cow::borrowed("1"),
                name: Cow::borrowed("t"),
                arguments: serde_json::json!({}),
                thought_signature: Some("sig".into()),
                action_id: None,
            },
            "tool_call_start",
        ),
        (
            StreamEvent::ToolCallResult {
                name: Cow::borrowed("t"),
                result: "ok".into(),
                typed_payload: None,
                id: None,
                action_id: None,
            },
            "tool_call_result",
        ),
        (
            StreamEvent::AudioChunk {
                data: vec![1],
                format: "wav".into(),
            },
            "audio_chunk",
        ),
        (
            StreamEvent::ImageGenerated {
                data: vec![2],
                media_type: "image/png".into(),
                revised_prompt: None,
            },
            "image_generated",
        ),
        (
            StreamEvent::AttachmentProcessed {
                filename: "a.pdf".into(),
                attachment_type: "document".into(),
                summary: "ok".into(),
            },
            "attachment_processed",
        ),
        (
            StreamEvent::Metadata {
                usage: None,
                finish_reason: None,
                metrics: None,
            },
            "metadata",
        ),
        (
            StreamEvent::LimitReached {
                max_iterations: 1,
                usage: TokenUsage::default(),
            },
            "limit_reached",
        ),
        (
            StreamEvent::Compaction {
                tokens_before: 10,
                tokens_after: 5,
                messages_removed: 1,
                summary: None,
                report: None,
            },
            "compaction",
        ),
        (
            StreamEvent::PhaseTransition {
                phase_index: 1,
                phase_total: 2,
                tool_names: vec!["fs".into()],
            },
            "phase_transition",
        ),
        (StreamEvent::Done, "done"),
        (StreamEvent::AgentStart, "agent_start"),
        (StreamEvent::AgentEnd, "agent_end"),
        (StreamEvent::TurnStart, "turn_start"),
        (StreamEvent::TurnEnd, "turn_end"),
    ];
    let mut seen = std::collections::BTreeSet::new();
    for (event, expected) in cases {
        let envelope = StreamEventEnvelope::v1(event, None);
        assert_eq!(envelope.r#type.as_ref(), expected);
        seen.insert(expected);
    }
    assert_eq!(seen.len(), 16);
    assert!(seen.contains("thinking_chunk"));
    assert!(seen.contains("tool_call_result"));
    assert!(seen.contains("audio_chunk"));
    assert!(seen.contains("compaction"));
    assert!(seen.contains("phase_transition"));
    assert!(seen.contains("agent_end"));
    assert!(seen.contains("turn_start"));
}

#[test]
fn tool_call_thought_signature_survives_completion_and_stream_wire_roundtrips() {
    let call = ToolCallInfo {
        id: Cow::borrowed("c1"),
        name: Cow::borrowed("lookup"),
        arguments: serde_json::json!({"q": "x"}),
        thought_signature: Some("sig-abc".into()),
        action_id: None,
        arguments_json: None,
    };
    let response = CompletionResponse {
        content: CompletionContent::WithToolCalls {
            content: String::new(),
            tool_calls: vec![call.clone()],
        },
        finish_reason: Some(Cow::borrowed("tool_calls")),
        usage: None,
        reasoning_content: None,
    };
    let json = serde_json::to_value(&response).expect("serialize completion");
    assert_eq!(json["tool_calls"][0]["thought_signature"], "sig-abc");
    let parsed: CompletionResponse = serde_json::from_value(json).expect("deserialize completion");
    assert_eq!(
        parsed.content.as_tool_calls()[0]
            .thought_signature
            .as_deref(),
        Some("sig-abc")
    );

    let stream = StreamEvent::ToolCallStart {
        id: Cow::borrowed("c1"),
        name: Cow::borrowed("lookup"),
        arguments: serde_json::json!({"q": "x"}),
        thought_signature: Some("sig-abc".into()),
        action_id: None,
    };
    let stream_json = serde_json::to_value(&stream).expect("serialize stream");
    assert_eq!(stream_json["thought_signature"], "sig-abc");
    let stream_parsed: StreamEvent =
        serde_json::from_value(stream_json).expect("deserialize stream");
    match stream_parsed {
        StreamEvent::ToolCallStart {
            thought_signature, ..
        } => assert_eq!(thought_signature.as_deref(), Some("sig-abc")),
        other => panic!("expected ToolCallStart, got {other:?}"),
    }
    assert_eq!(call.arguments_json(), r#"{"q":"x"}"#);
}

#[test]
fn chat_response_builders_and_accessors() {
    let response = ChatResponse::new(
        "hello",
        TokenUsage::from_counts(Some(1), Some(2), None),
        true,
    )
    .with_tools_executed(vec!["calc".into()])
    .with_limit_reached();
    assert_eq!(response.text(), "hello");
    assert!(response.used_tools());
    assert!(response.limit_reached());
    assert_eq!(response.tools_executed, vec!["calc".to_string()]);
    assert_eq!(format!("{response}"), "hello");
    assert_eq!(&*response, "hello");
    assert_eq!(response.usage().total_tokens, Some(3));
}

#[test]
fn tool_schema_and_tool_choice_roundtrip() {
    let schema = ToolSchema {
        name: Cow::borrowed("search"),
        description: Cow::borrowed("Search the web"),
        parameters: ParametersSchema {
            schema_type: Cow::borrowed("object"),
            properties: serde_json::json!({"q": {"type": "string"}}),
            required: vec![Cow::borrowed("q")],
        },
    };
    let json = serde_json::to_value(&schema).expect("serialize schema");
    let parsed: ToolSchema = serde_json::from_value(json).expect("deserialize schema");
    assert_eq!(parsed.name.as_ref(), "search");
    assert_eq!(parsed.parameters.required.len(), 1);

    let choice = ToolChoice::Specific(Cow::borrowed("search"));
    let choice_json = serde_json::to_value(&choice).expect("serialize choice");
    let parsed_choice: ToolChoice =
        serde_json::from_value(choice_json).expect("deserialize choice");
    match parsed_choice {
        ToolChoice::Specific(name) => assert_eq!(name.as_ref(), "search"),
        other => panic!("expected Specific, got {other:?}"),
    }
    let auto = serde_json::to_value(&ToolChoice::Auto).expect("serialize auto");
    assert_eq!(auto, "auto");
}

#[test]
fn token_usage_from_counts_fills_total() {
    let computed = TokenUsage::from_counts(Some(10), Some(5), None);
    assert_eq!(computed.total_tokens, Some(15));
    let explicit = TokenUsage::from_counts(Some(10), Some(5), Some(99));
    assert_eq!(explicit.total_tokens, Some(99));
}

#[test]
fn token_usage_embedding_and_add_saturates() {
    let embed = TokenUsage::from_embedding_tokens(4, Some(4));
    assert_eq!(embed.embedding_tokens, Some(4));
    assert_eq!(embed.total_tokens, Some(4));

    let a = TokenUsage::from_counts(Some(u32::MAX - 1), Some(1), None);
    let b = TokenUsage::from_counts(Some(10), Some(0), None);
    let sum = a + b;
    assert_eq!(sum.prompt_tokens, Some(u32::MAX));
    assert_eq!(sum.completion_tokens, Some(1));
}

#[test]
fn token_usage_ensure_total_skips_all_zero_components() {
    let usage = TokenUsage {
        prompt_tokens: Some(0),
        completion_tokens: None,
        embedding_tokens: None,
        total_tokens: None,
        context_window: None,
        cache_creation_tokens: None,
        cache_read_tokens: None,
        last_prompt_tokens: None,
    }
    .ensure_total_tokens();
    assert_eq!(usage.total_tokens, None);
}

#[test]
fn token_usage_adds_cache_counters() {
    let a = TokenUsage::from_counts(Some(10), Some(2), None).with_cache_tokens(Some(4), Some(6));
    let b = TokenUsage::from_counts(Some(1), Some(1), None).with_cache_tokens(Some(0), Some(3));
    let sum = a + b;
    assert_eq!(sum.cache_creation_tokens, Some(4));
    assert_eq!(sum.cache_read_tokens, Some(9));
}

#[test]
fn tool_call_from_json_caches_arguments_string() {
    let call = ToolCallInfo::from_json("id", "name", r#"{"a":1}"#);
    assert_eq!(call.arguments, serde_json::json!({"a": 1}));
    assert_eq!(call.arguments_json(), r#"{"a":1}"#);
    assert_eq!(call.arguments_json_arc().as_str(), r#"{"a":1}"#);
}

#[test]
fn tool_call_from_json_invalid_aligns_cache_with_empty_object() {
    let call = ToolCallInfo::from_json("id", "name", "not-json");
    assert_eq!(call.arguments, serde_json::json!({}));
    assert_eq!(call.arguments_json(), "{}");
}

#[test]
fn detect_media_type_maps_common_extensions() {
    assert_eq!(
        detect_media_type(std::path::Path::new("a.png")),
        "image/png"
    );
    assert_eq!(
        detect_media_type(std::path::Path::new("a.JPG")),
        "image/jpeg"
    );
    assert_eq!(
        detect_media_type(std::path::Path::new("a.mp3")),
        "audio/mpeg"
    );
    assert_eq!(
        detect_media_type(std::path::Path::new("a.mp4")),
        "video/mp4"
    );
    assert_eq!(
        detect_media_type(std::path::Path::new("a.pdf")),
        "application/pdf"
    );
    assert_eq!(
        detect_media_type(std::path::Path::new("a.bin")),
        "application/octet-stream"
    );
}

#[test]
fn detect_attachment_type_classifies_image_audio_video_document() {
    assert_eq!(detect_attachment_type("image/png"), AttachmentType::Image);
    assert_eq!(detect_attachment_type("audio/wav"), AttachmentType::Audio);
    assert_eq!(detect_attachment_type("video/webm"), AttachmentType::Video);
    assert_eq!(
        detect_attachment_type("application/pdf"),
        AttachmentType::Document
    );
    assert_eq!(
        detect_attachment_type("application/octet-stream"),
        AttachmentType::Document
    );
}

#[test]
fn user_with_image_and_url_produce_multimodal_parts() {
    let with_bytes = Message::user_with_image("see", b"raw".to_vec(), "image/png");
    assert!(with_bytes.content.is_multimodal());
    assert_eq!(with_bytes.extract_text(), "see");
    let parts = with_bytes.content.as_multimodal().expect("parts");
    assert!(matches!(parts[1], ContentPart::Image { .. }));

    let with_url = Message::user_with_image_url("link", "https://example.test/a.png");
    assert!(with_url.content.is_multimodal());
    let url_parts = with_url.content.as_multimodal().expect("parts");
    match &url_parts[1] {
        ContentPart::ImageUrl { url, .. } => {
            assert_eq!(url.as_ref(), "https://example.test/a.png")
        }
        other => panic!("expected ImageUrl, got {other:?}"),
    }
}
