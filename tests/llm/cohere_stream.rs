use caliche::llm::provider_utils::CohereStreamParser;
use caliche::messages::StreamEvent;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct CohereStreamChunk {
    #[serde(default)]
    index: Option<u32>,
    tool_call_delta: Option<CohereStreamToolCallDelta>,
}

#[derive(Debug, Deserialize, Default)]
struct CohereStreamToolCallDelta {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    index: Option<u32>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

fn accumulate_cohere_tool_delta(
    parser: &mut CohereStreamParser,
    delta: CohereStreamToolCallDelta,
    index: Option<u32>,
) {
    let index = index.or(delta.index).unwrap_or(0);
    let arguments = delta
        .parameters
        .as_ref()
        .and_then(|parameters| match parameters {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Null => None,
            other if other.is_object() && !other.as_object().is_some_and(|m| m.is_empty()) => {
                serde_json::to_string(other).ok()
            }
            other if !other.is_null() => Some(other.to_string()),
            _ => None,
        });
    parser
        .accumulator_mut()
        .accumulate(index, delta.id, delta.name, arguments);
}

fn finalize_cohere_chunks(chunks: &[&str]) -> anyhow::Result<Option<StreamEvent>> {
    let mut parser = CohereStreamParser::new();
    for line in chunks {
        let chunk: CohereStreamChunk = serde_json::from_str(line)?;
        if let Some(tool_call) = chunk.tool_call_delta {
            accumulate_cohere_tool_delta(&mut parser, tool_call, chunk.index);
        }
    }
    Ok(parser.finalize_index(0, false))
}

#[test]
fn cohere_tool_calls_chunk_accumulates_fragments() {
    let evt = finalize_cohere_chunks(&[
        r#"{"event_type":"tool-calls-chunk","index":0,"tool_call_delta":{"id":"tc_1","name":"get_weather","parameters":""}}"#,
        r#"{"event_type":"tool-calls-chunk","index":0,"tool_call_delta":{"parameters":"{\"location\":\"NYC\"}"}}"#,
    ])
    .expect("chunks should process")
    .expect("tool call should finalize");

    match evt {
        StreamEvent::ToolCallStart {
            id,
            name,
            arguments,
            ..
        } => {
            assert_eq!(id, "tc_1");
            assert_eq!(name, "get_weather");
            assert_eq!(arguments["location"], "NYC");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn cohere_tool_calls_chunk_accepts_object_parameters() {
    let evt = finalize_cohere_chunks(&[
        r#"{"event_type":"tool-calls-chunk","index":0,"tool_call_delta":{"id":"tc_2","name":"search","parameters":{"q":"rust"}}}"#,
    ])
    .expect("chunk should process")
    .expect("tool call should finalize");

    match evt {
        StreamEvent::ToolCallStart { arguments, .. } => {
            assert_eq!(arguments["q"], "rust");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}
