use caliche::llm::google::GoogleProviderConfig;
use caliche::llm::provider_utils::GoogleStreamAccumulator;
use caliche::messages::StreamEvent;
use serde::Deserialize;

const WEATHER_CHUNK: &str = r#"{
    "candidates": [{
        "content": {
            "role": "model",
            "parts": [{
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "NYC"}
                }
            }]
        }
    }]
}"#;

#[test]
fn google_config_keeps_text_chat_models() {
    assert_eq!(
        GoogleProviderConfig::normalize_text_chat_model("models/gemini-2.5-pro"),
        "models/gemini-2.5-pro"
    );
    assert_eq!(
        GoogleProviderConfig::normalize_text_chat_model("gemini-2.5-flash"),
        "models/gemini-2.5-flash"
    );
}

#[test]
fn google_config_normalizes_tts_to_text_model() {
    assert_eq!(
        GoogleProviderConfig::normalize_text_chat_model("models/gemini-2.5-pro-preview-tts"),
        "models/gemini-2.5-flash"
    );
    assert_eq!(
        GoogleProviderConfig::normalize_text_chat_model("gemini-2.5-flash-preview-tts"),
        "models/gemini-2.5-flash"
    );
}

#[derive(Debug, Deserialize)]
struct GoogleStreamChunk {
    candidates: Vec<GoogleStreamCandidate>,
}

#[derive(Debug, Deserialize)]
struct GoogleStreamCandidate {
    content: Option<GoogleStreamContent>,
}

#[derive(Debug, Deserialize)]
struct GoogleStreamContent {
    parts: Vec<GoogleStreamPart>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleStreamPart {
    function_call: Option<GoogleFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct GoogleFunctionCall {
    name: String,
    args: serde_json::Value,
}

fn finalize_google_chunks(chunks: &[&str]) -> anyhow::Result<Vec<StreamEvent>> {
    let mut accumulator = GoogleStreamAccumulator::new();
    for line in chunks {
        let chunk: GoogleStreamChunk = serde_json::from_str(line)?;
        for candidate in &chunk.candidates {
            let Some(content) = candidate.content.as_ref() else {
                continue;
            };
            for part in &content.parts {
                if let Some(function_call) = &part.function_call {
                    let arguments = serde_json::to_string(&function_call.args)?;
                    accumulator.feed_function_call(function_call.name.clone(), arguments, None);
                }
            }
        }
    }
    Ok(accumulator.finalize_all())
}

#[test]
fn google_stream_accumulator_dedupes_repeated_function_calls() {
    let events =
        finalize_google_chunks(&[WEATHER_CHUNK, WEATHER_CHUNK]).expect("chunks should process");

    assert_eq!(events.len(), 1);
    match &events[0] {
        StreamEvent::ToolCallStart {
            name, arguments, ..
        } => {
            assert_eq!(name, "get_weather");
            assert_eq!(arguments["location"], "NYC");
        }
        other => panic!("unexpected event: {other:?}"),
    }
}
