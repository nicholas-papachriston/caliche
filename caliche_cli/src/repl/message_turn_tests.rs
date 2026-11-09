use super::{thin_client_final_response, thin_client_streamed_text};
use caliche::messages::stream_contracts::sse_name;
use serde_json::json;

fn envelope(event_type: &str, payload: serde_json::Value) -> serde_json::Value {
    json!({
        "version": "1",
        "type": event_type,
        "payload": payload,
    })
}

#[test]
fn streamed_text_reads_nested_text_chunk_payload() {
    let data = envelope(
        "text_chunk",
        json!({"type": "text_chunk", "content": "Hello from the agent"}),
    );
    assert_eq!(
        thin_client_streamed_text("text_chunk", &data),
        Some("Hello from the agent")
    );
}

#[test]
fn streamed_text_reads_flat_text_chunk_body() {
    let data = json!({"content": "flat chunk"});
    assert_eq!(
        thin_client_streamed_text("text_chunk", &data),
        Some("flat chunk")
    );
}

#[test]
fn final_response_reads_nested_turn_complete_payload() {
    let data = envelope(
        "turn_complete",
        json!({
            "type": "turn_complete",
            "turn_id": 1,
            "response": "done",
            "duration_ms": 12
        }),
    );
    assert_eq!(
        thin_client_final_response(sse_name::TURN_COMPLETE, &data),
        Some("done")
    );
}

#[test]
fn missing_nested_content_does_not_print_from_envelope_root() {
    let data = envelope("text_chunk", json!({"type": "text_chunk"}));
    assert_eq!(thin_client_streamed_text("text_chunk", &data), None);
}

#[test]
fn parse_token_usage_reads_prompt_cache_counters() {
    let usage = super::parse_token_usage_value(&json!({
        "prompt_tokens": 80,
        "completion_tokens": 12,
        "total_tokens": 92,
        "cache_creation_tokens": 20,
        "cache_read_tokens": 40
    }));
    assert_eq!(usage.prompt_tokens, Some(80));
    assert_eq!(usage.cache_creation_tokens, Some(20));
    assert_eq!(usage.cache_read_tokens, Some(40));
}

use crate::core::slash_commands::SlashCommandResult;
use crate::core::state::InteractiveState;
use crate::core::traits::{InputSource, OutputRenderer};
use crate::cost::TokenBudget;
use crate::session::CliSession;
use async_trait::async_trait;
use caliche::observability::OutputVerbosity;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct IdleInput;

#[async_trait]
impl InputSource for IdleInput {
    async fn read_line(&mut self, _prompt: &str) -> anyhow::Result<Option<String>> {
        Ok(None)
    }

    fn add_history(&mut self, _line: &str) {}

    fn save_history(&mut self) -> anyhow::Result<()> {
        Ok(())
    }
}

struct QuietRenderer;

impl OutputRenderer for QuietRenderer {
    fn handle_command_result(&mut self, _result: SlashCommandResult) {}
    fn render_chunk(&mut self, _text: &str) {}
    fn render_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
    fn render_tool_complete(&mut self, _name: &str, _result: &str) {}
    fn render_tool_error(&mut self, _name: &str, _error: &str) {}
    fn render_thinking(&mut self, _content: &str) {}
    fn render_token_usage(&mut self, _usage: &caliche::messages::TokenUsage) {}
    fn show_error(&mut self, _message: &str) {}
    fn show_status(&mut self, _message: &str) {}
    fn clear(&mut self) {}
}

struct PostedTurn {
    session_id: String,
    body: serde_json::Value,
}

async fn recording_server() -> (String, Arc<Mutex<Vec<PostedTurn>>>) {
    use axum::Json;
    use axum::Router;
    use axum::extract::{Path, State};
    use axum::response::sse::{Event, Sse};
    use axum::routing::{get, post};
    use futures_util::stream::Stream;
    use std::convert::Infallible;
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;

    let posted = Arc::new(Mutex::new(Vec::new()));

    async fn events(Path(_id): Path<String>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
        let (tx, rx) = mpsc::channel(2);
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(Event::default().event(sse_name::TURN_COMPLETE).data(
                    r#"{"type":"turn_complete","response":"Caliche runtime.","duration_ms":1}"#,
                )))
                .await;
        });
        Sse::new(ReceiverStream::new(rx))
    }

    async fn turn(
        State(posted): State<Arc<Mutex<Vec<PostedTurn>>>>,
        Path(session_id): Path<String>,
        Json(body): Json<serde_json::Value>,
    ) -> Json<serde_json::Value> {
        posted
            .lock()
            .expect("posted turns")
            .push(PostedTurn { session_id, body });
        Json(serde_json::json!({"response": "Caliche runtime.", "duration_ms": 1}))
    }

    let app = Router::new()
        .route("/api/agent/sessions/{id}/events", get(events))
        .route("/api/agent/sessions/{id}/turn", post(turn))
        .with_state(Arc::clone(&posted));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind test listener");
    let addr = listener.local_addr().expect("listener addr");
    tokio::spawn(async move {
        axum::serve(listener, app).await.expect("serve test app");
    });
    (format!("http://{addr}"), posted)
}

/// `/model` updates the prompt label (`token_budget`). The next thin-client
/// turn must post that provider and model on the same remote session.
#[tokio::test]
async fn model_switch_label_is_posted_on_the_same_session() {
    let (base_url, posted) = recording_server().await;
    let client = crate::client::AgentClient::new(base_url);
    let state = InteractiveState::new(
        CliSession::new("local".to_string(), std::path::PathBuf::from(".")),
        "anthropic",
        "claude-sonnet-4",
        "You are Caliche.".to_string(),
    );
    let mut engine = super::super::ReplEngine::new(IdleInput, QuietRenderer, state)
        .with_thin_client(client.clone(), "sess-live");

    engine.state.token_budget = TokenBudget::new("openrouter", "z-ai/glm-5.3");
    let prompt = engine.build_prompt();
    assert!(
        prompt.contains("openrouter/z-ai/glm-5.3"),
        "label should show the switched model: {prompt}"
    );
    engine
        .process_message_via_thin_client(&client, "sess-live", "who are you?", true, None)
        .await
        .expect("first thin-client turn");

    engine.state.token_budget = TokenBudget::new("anthropic", "claude-opus-4");
    engine
        .process_message_via_thin_client(&client, "sess-live", "and now?", true, None)
        .await
        .expect("second thin-client turn");

    let turns = posted.lock().expect("posted turns");
    assert_eq!(
        turns.len(),
        2,
        "each prompt is one turn on the remote session"
    );
    assert_eq!(turns[0].session_id, "sess-live");
    assert_eq!(turns[1].session_id, "sess-live");
    assert_eq!(turns[0].body["input"], "who are you?");
    assert_eq!(turns[0].body["provider"], "openrouter");
    assert_eq!(turns[0].body["model"], "z-ai/glm-5.3");
    assert_eq!(turns[1].body["input"], "and now?");
    assert_eq!(turns[1].body["provider"], "anthropic");
    assert_eq!(turns[1].body["model"], "claude-opus-4");
    assert!(turns[0].body.get("model").is_some());
    assert_ne!(turns[0].body["model"], "claude-sonnet-4");
}

/// A label that still matches session create must not retarget the remote.
///
/// Posting the startup pair on every turn makes a server without that
/// provider's credentials fail instead of answering with its own model.
#[tokio::test]
async fn unchanged_label_omits_model_on_the_same_session() {
    let (base_url, posted) = recording_server().await;
    let client = crate::client::AgentClient::new(base_url);
    let state = InteractiveState::new(
        CliSession::new("local".to_string(), std::path::PathBuf::from(".")),
        "anthropic",
        "claude-sonnet-4",
        "You are Caliche.".to_string(),
    );
    let mut engine = super::super::ReplEngine::new(IdleInput, QuietRenderer, state)
        .with_thin_client(client.clone(), "sess-live");
    let prompt = engine.build_prompt();
    assert!(
        prompt.contains("anthropic/claude-sonnet-4"),
        "label should show the session model: {prompt}"
    );

    engine
        .process_message_via_thin_client(&client, "sess-live", "who are you?", true, None)
        .await
        .expect("startup turn");

    engine.state.token_budget = TokenBudget::new("openrouter", "z-ai/glm-5.3");
    engine
        .process_message_via_thin_client(&client, "sess-live", "switch", true, None)
        .await
        .expect("switch turn");
    engine
        .process_message_via_thin_client(&client, "sess-live", "again", true, None)
        .await
        .expect("sticky turn");

    let turns = posted.lock().expect("posted turns");
    assert_eq!(turns.len(), 3);
    assert!(turns.iter().all(|turn| turn.session_id == "sess-live"));
    assert_eq!(turns[0].body["input"], "who are you?");
    assert!(turns[0].body.get("provider").is_none());
    assert!(turns[0].body.get("model").is_none());
    assert_eq!(turns[1].body["provider"], "openrouter");
    assert_eq!(turns[1].body["model"], "z-ai/glm-5.3");
    assert_eq!(turns[2].body["input"], "again");
    assert!(
        turns[2].body.get("provider").is_none(),
        "an applied switch stays on the server without another retarget"
    );
    assert!(turns[2].body.get("model").is_none());
}

struct StatusLog {
    lines: Vec<String>,
    chunks: Vec<String>,
    thinking: Vec<String>,
    verbosity: OutputVerbosity,
}

impl StatusLog {
    fn new(verbosity: OutputVerbosity) -> Self {
        Self {
            lines: Vec::new(),
            chunks: Vec::new(),
            thinking: Vec::new(),
            verbosity,
        }
    }
}

impl OutputRenderer for StatusLog {
    fn handle_command_result(&mut self, _result: SlashCommandResult) {}
    fn render_chunk(&mut self, text: &str) {
        self.chunks.push(text.to_string());
    }
    fn render_tool_start(&mut self, _name: &str, _args: &serde_json::Value) {}
    fn render_tool_complete(&mut self, _name: &str, _result: &str) {}
    fn render_tool_error(&mut self, _name: &str, _error: &str) {}
    fn render_thinking(&mut self, content: &str) {
        self.thinking.push(content.to_string());
    }
    fn render_token_usage(&mut self, _usage: &caliche::messages::TokenUsage) {}
    fn show_error(&mut self, _message: &str) {}
    fn show_status(&mut self, message: &str) {
        self.lines.push(message.to_string());
    }
    fn clear(&mut self) {}
    fn output_verbosity(&self) -> OutputVerbosity {
        self.verbosity
    }
}

#[test]
fn thin_client_tool_and_thinking_lines_share_action_ids() {
    let action = "01936b8e-7c3a-7000-8000-000000000001";
    let mut turn = super::ThinClientTurnState::default();
    let mut renderer = StatusLog::new(OutputVerbosity::Thinking);
    let thinking = serde_json::json!({
        "version": "1",
        "type": "thinking",
        "payload": {"type": "thinking", "content": "look", "action_id": action}
    });
    super::apply_thin_client_sse_event(&mut renderer, &mut turn, "agent.thinking", &thinking);
    super::apply_thin_client_sse_event(
        &mut renderer,
        &mut turn,
        "agent.thinking",
        &serde_json::json!({"payload": {"content": " again", "action_id": action}}),
    );
    let start = serde_json::json!({
        "payload": {"tool_name": "bash", "tool_id": "call_1", "action_id": action}
    });
    super::apply_thin_client_sse_event(&mut renderer, &mut turn, "tool.start", &start);
    let end = serde_json::json!({
        "payload": {"tool_name": "bash", "tool_id": "call_1", "action_id": action, "success": true}
    });
    super::apply_thin_client_sse_event(&mut renderer, &mut turn, "tool.end", &end);
    let short = super::display_action_id(action);
    assert_eq!(short, "00000001");
    assert_eq!(
        renderer.lines,
        vec![
            format!("thinking.start {short}"),
            format!("thinking.end {short}"),
            format!("tool.start {short} bash"),
            format!("tool.end {short} bash ok=true"),
        ]
    );
    assert!(
        renderer.lines.iter().all(|line| !line.contains(action)),
        "status lines must not print the full action id: {:?}",
        renderer.lines
    );
    assert_eq!(
        renderer.thinking,
        vec!["look".to_string(), " again".to_string()]
    );
    assert!(
        renderer.chunks.is_empty(),
        "thinking must not render as answer text"
    );
}

#[test]
fn thin_client_tool_error_reuses_the_start_action_id() {
    let action = "01936b8e-7c3a-7000-8000-000000000002";
    let mut turn = super::ThinClientTurnState::default();
    let mut renderer = StatusLog::new(OutputVerbosity::Normal);
    super::apply_thin_client_sse_event(
        &mut renderer,
        &mut turn,
        "tool.start",
        &serde_json::json!({"payload": {"tool_name": "bash", "action_id": action}}),
    );
    super::apply_thin_client_sse_event(
        &mut renderer,
        &mut turn,
        "tool_error",
        &serde_json::json!({"payload": {"tool_name": "bash", "action_id": action, "error": "no"}}),
    );
    let short = super::display_action_id(action);
    assert_eq!(short, "00000002");
    assert_eq!(
        renderer.lines,
        vec![
            format!("tool.start {short} bash"),
            format!("tool.end {short} bash ok=false"),
        ]
    );
    assert!(
        renderer.lines.iter().all(|line| !line.contains(action)),
        "status lines must not print the full action id: {:?}",
        renderer.lines
    );
}

#[test]
fn normal_verbosity_hides_thinking_but_keeps_it_for_persistence() {
    let action = "01936b8e-7c3a-7000-8000-000000000003";
    let mut turn = super::ThinClientTurnState::default();
    let mut renderer = StatusLog::new(OutputVerbosity::Normal);
    super::apply_thin_client_sse_event(
        &mut renderer,
        &mut turn,
        "agent.thinking",
        &serde_json::json!({"payload": {"content": "look", "action_id": action}}),
    );
    super::apply_thin_client_sse_event(
        &mut renderer,
        &mut turn,
        "agent.thinking",
        &serde_json::json!({"payload": {"content": " again", "action_id": action}}),
    );
    super::apply_thin_client_sse_event(
        &mut renderer,
        &mut turn,
        "text_chunk",
        &serde_json::json!({"payload": {"content": "the answer"}}),
    );
    super::apply_thin_client_sse_event(
        &mut renderer,
        &mut turn,
        sse_name::TURN_COMPLETE,
        &serde_json::json!({"payload": {"response": "the answer"}}),
    );

    assert!(
        renderer
            .lines
            .iter()
            .all(|line| !line.starts_with("thinking.")),
        "normal verbosity must not print thinking markers: {:?}",
        renderer.lines
    );
    assert!(renderer.thinking.is_empty());
    assert_eq!(renderer.chunks, vec!["the answer".to_string()]);
    assert!(!renderer.chunks.iter().any(|chunk| chunk.contains("look")));

    let saved = super::persisted_thin_client_message(&turn).expect("assistant turn");
    assert_eq!(saved.extract_text(), "the answer");
    assert_eq!(saved.reasoning_content(), Some("look again"));
}

fn reasoned_completion(text: &str, reasoning: &str) -> caliche::messages::CompletionResponse {
    caliche::messages::CompletionResponse {
        content: caliche::messages::CompletionContent::Text {
            content: text.to_string(),
        },
        finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
        usage: Some(caliche::messages::TokenUsage::default()),
        reasoning_content: Some(reasoning.to_string()),
    }
}

fn repl_engine() -> super::super::ReplEngine<IdleInput, QuietRenderer> {
    let state = InteractiveState::new(
        CliSession::new("local".to_string(), std::path::PathBuf::from(".")),
        "openai",
        "mock-model",
        "prompt".to_string(),
    );
    super::super::ReplEngine::new(IdleInput, QuietRenderer, state)
}

fn last_assistant_reasoning(messages: &[caliche::messages::Message]) -> Option<&str> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == caliche::messages::MessageRole::Assistant)
        .and_then(caliche::messages::Message::reasoning_content)
}

/// Direct stream (`process_message`) must store sink-captured thinking.
/// Replacing that append with `Message::assistant` fails this test.
#[tokio::test]
async fn direct_stream_persists_reasoning_when_display_is_off() {
    let provider = std::sync::Arc::new(caliche::testing::MockProvider::with_responses(vec![
        reasoned_completion("the answer", "hidden cot"),
    ]));
    let agent = caliche::Agent::new(provider, std::sync::Arc::new(caliche::ToolRegistry::new()));
    let mut service =
        caliche::AgentService::new_with_agent(agent, caliche::llm::ProviderType::OpenAI)
            .expect("mock service");
    let mut engine = repl_engine();
    engine
        .process_message(&mut service, "hello", None, Vec::new(), Vec::new(), false)
        .await
        .expect("direct stream");
    assert_eq!(
        engine
            .state
            .messages
            .last()
            .expect("assistant")
            .extract_text(),
        "the answer"
    );
    assert_eq!(
        last_assistant_reasoning(&engine.state.messages),
        Some("hidden cot")
    );
}

/// Harness non-stream append reads `HarnessTurnResult.reasoning_content`.
/// Dropping that field at the append fails this test. Service memory stays
/// text-only, so this must not depend on history metadata.
#[tokio::test]
async fn harness_non_stream_persists_reasoning_from_history() {
    let provider = std::sync::Arc::new(caliche::testing::MockProvider::with_responses(vec![
        reasoned_completion("single step only", "plan cot"),
        reasoned_completion("the answer", "hidden cot"),
    ]));
    let agent = caliche::Agent::new(provider, std::sync::Arc::new(caliche::ToolRegistry::new()));
    let service = caliche::AgentService::new_with_agent(agent, caliche::llm::ProviderType::OpenAI)
        .expect("mock service");
    let mut harness =
        caliche::AgentHarness::from_service(service, caliche::HarnessConfig::default())
            .expect("harness");
    let mut engine = repl_engine();
    engine
        .process_message_via_harness_decomposed(&mut harness, "do the thing", None)
        .await
        .expect("harness turn");
    assert_eq!(
        last_assistant_reasoning(&engine.state.messages),
        Some("hidden cot")
    );
    assert_eq!(
        engine
            .state
            .messages
            .last()
            .expect("assistant")
            .extract_text(),
        "the answer"
    );
}

/// Harness stream append (`record_harness_metrics_and_response`) stores sink thinking.
#[test]
fn harness_stream_record_persists_thinking() {
    let mut engine = repl_engine();
    let stats = super::turn_helpers::HarnessStreamStats {
        thinking: "hidden cot".to_string(),
        ..super::turn_helpers::HarnessStreamStats::default()
    };
    let turn = caliche::harness::HarnessTurnResult {
        response: "the answer".to_string(),
        usage: caliche::messages::TokenUsage::default(),
        tools_used: false,
        duration_ms: 1,
        task_complete: true,
        limit_reached: false,
        discipline: caliche::harness::HarnessDisciplineMetrics::for_turn(
            &caliche::messages::TokenUsage::default(),
            0,
            0,
            0,
            None,
            None,
        ),
        input_id: None,
        reasoning_content: String::new(),
    };
    engine.record_harness_metrics_and_response(&stats, &turn);
    assert_eq!(
        last_assistant_reasoning(&engine.state.messages),
        Some("hidden cot")
    );
    assert_eq!(
        engine
            .state
            .messages
            .last()
            .expect("assistant")
            .extract_text(),
        "the answer"
    );
}

#[test]
fn a_status_line_shows_while_waiting_and_gives_way_to_the_answer() {
    let mut turn = super::ThinClientTurnState::default();
    let mut renderer = StatusLog::new(OutputVerbosity::Normal);
    let status = serde_json::json!({
        "version": "1",
        "type": "status",
        "payload": {"type": "status", "message": "Loading m in LM Studio..."}
    });
    super::apply_thin_client_sse_event(&mut renderer, &mut turn, "status", &status);
    super::apply_thin_client_sse_event(&mut renderer, &mut turn, "status", &status);
    super::apply_thin_client_sse_event(
        &mut renderer,
        &mut turn,
        "text_chunk",
        &serde_json::json!({"payload": {"content": "Hi"}}),
    );

    assert_eq!(
        renderer.lines,
        vec![
            "Loading m in LM Studio...",
            "Loading m in LM Studio...",
            "Thinking...",
        ]
    );
    assert_eq!(renderer.chunks, vec!["Hi".to_string()]);
}

/// `/context-window` in the REPL reaches the remote session: the next turn
/// posts the window once, and later turns omit it while it is unchanged.
#[tokio::test]
async fn context_window_is_posted_once_on_the_next_turn() {
    let (base_url, posted) = recording_server().await;
    let client = crate::client::AgentClient::new(base_url);
    let state = InteractiveState::new(
        CliSession::new("local".to_string(), std::path::PathBuf::from(".")),
        "lmstudio",
        "big-model",
        "You are Caliche.".to_string(),
    );
    let mut engine = super::super::ReplEngine::new(IdleInput, QuietRenderer, state)
        .with_thin_client(client.clone(), "sess-live");

    engine
        .process_message_via_thin_client(&client, "sess-live", "before", true, None)
        .await
        .expect("turn before the window");
    engine.state.context_window_override = Some(16_384);
    engine
        .process_message_via_thin_client(&client, "sess-live", "after", true, None)
        .await
        .expect("turn after the window");
    engine
        .process_message_via_thin_client(&client, "sess-live", "again", true, None)
        .await
        .expect("turn with the same window");

    let turns = posted.lock().expect("posted turns");
    let windows: Vec<Option<u64>> = turns
        .iter()
        .map(|turn| {
            turn.body
                .get("context_window")
                .and_then(serde_json::Value::as_u64)
        })
        .collect();
    assert_eq!(windows, [None, Some(16_384), None]);
}
