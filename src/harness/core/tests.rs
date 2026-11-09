use super::{AgentHarness, sanitize_notify_preview};
use crate::agent::hooks::CONTINUE_TASK_PROMPT;
use crate::agent::memory::{BufferMemory, MemoryStrategy};
use crate::agent::service::AgentService;
use crate::agent::tools::ToolRegistry;
use crate::harness::config::{HarnessConfig, PermissionModeConfig};
use crate::harness::events::AgentEvent;
use crate::llm::ProviderType;
use crate::messages::{Message, MessageContent, MessageRole};
use crate::testing::MockProvider;
use std::sync::Arc;

#[tokio::test]
async fn from_service_history_set_clear_roundtrip() {
    let provider = Arc::new(MockProvider::new(vec!["ok".into()]));
    let tools = Arc::new(ToolRegistry::new());
    let agent = crate::agent::Agent::new(provider, tools);
    let service =
        AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock agent service");
    assert_eq!(service.provider(), ProviderType::OpenAI);
    assert_eq!(service.model(), "mock-model");

    let mut harness = AgentHarness::from_service(service, HarnessConfig::default())
        .expect("harness from mock service");
    harness.set_history(vec![Message::user("a"), Message::assistant("b")]);
    assert_eq!(harness.history().len(), 2);
    assert_eq!(harness.history()[0].role, MessageRole::User);
    harness.add_to_history(Message::user("c"));
    assert_eq!(harness.history().len(), 3);
    harness.clear_history();
    assert!(harness.history().is_empty());
}

#[test]
fn sanitize_notify_preview_strips_metacharacters() {
    let dirty = "hello\nworld`$HOME\u{7}end";
    let clean = sanitize_notify_preview(dirty);
    assert!(!clean.contains('\n'));
    assert!(!clean.contains('`'));
    assert!(!clean.contains('$'));
    assert_eq!(clean.chars().count(), dirty.chars().count());
}

fn mock_harness(
    provider: Arc<MockProvider>,
    tools: Arc<ToolRegistry>,
    max_turns: usize,
) -> AgentHarness {
    let agent = crate::agent::Agent::new(provider, tools);
    let service =
        AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock agent service");
    AgentHarness::from_service(
        service,
        HarnessConfig {
            max_turns,
            auto_checkpoint: false,
            permission_mode: PermissionModeConfig::AcceptAll,
            ..HarnessConfig::default()
        },
    )
    .expect("harness from mock service")
}

fn user_texts(harness: &AgentHarness) -> Vec<String> {
    harness
        .history()
        .iter()
        .filter(|message| matches!(message.role, MessageRole::User))
        .filter_map(|message| message.as_text().map(ToOwned::to_owned))
        .collect()
}

#[tokio::test]
async fn resume_context_reaches_the_next_prompt() {
    let provider = Arc::new(MockProvider::new(vec!["I renamed parser.rs".into()]));
    let mut harness = mock_harness(Arc::clone(&provider), Arc::new(ToolRegistry::new()), 1);
    let stored = vec![
        Message::user("Please rename parser.rs"),
        Message::assistant("Renamed parser.rs with the fs tool"),
    ];
    let fitted = harness
        .restore_resume_context(stored)
        .await
        .expect("restore resume context");
    assert!(!fitted.summarized);
    assert_eq!(harness.history().len(), 2);

    let _ = harness
        .run_turn_stream("what did you just do", |_| {})
        .await
        .expect("resumed turn");

    let prompts = provider.recorded_prompts();
    let prompt = prompts.last().map(String::as_str).unwrap_or("");
    assert!(
        prompt.contains("Renamed parser.rs with the fs tool"),
        "resume prompt missing prior assistant turn: {prompt}"
    );
    assert!(
        prompt.contains("what did you just do"),
        "resume prompt missing the new user question: {prompt}"
    );
}

const EARLY_TURN: &str = "EARLY-HARNESS-MARKER name the harness";

async fn seeded_buffer_max_tokens(harness: &AgentHarness) -> usize {
    let memory = harness
        .service()
        .memory()
        .expect("a new harness must seed memory or the next turn is a blank prompt");
    let guard = memory.read().await;
    guard
        .as_any()
        .downcast_ref::<BufferMemory>()
        .expect("seeded turn memory must be a buffer")
        .max_tokens()
}

#[tokio::test]
async fn streamed_turn_reaches_the_next_prompt_without_manual_memory() {
    let provider = Arc::new(MockProvider::new(vec![
        "The harness is Caliche.".into(),
        "Still Caliche.".into(),
        "Third reply.".into(),
        "It is Caliche.".into(),
    ]));
    let mut harness = mock_harness(Arc::clone(&provider), Arc::new(ToolRegistry::new()), 4);
    let window = harness.service().agent().cached_context_window;
    assert_eq!(
        seeded_buffer_max_tokens(&harness).await,
        crate::agent::memory::session_memory_token_budget(window),
        "unset harness memory must use the session transcript budget"
    );

    // Three stored turns are six messages. A zero cap keeps only min_messages (4),
    // so the first turn is gone before the fourth prompt. The session budget holds the fixture.
    harness
        .run_turn_stream(EARLY_TURN, |_| {})
        .await
        .expect("first turn");
    harness
        .run_turn_stream("second turn", |_| {})
        .await
        .expect("second turn");
    harness
        .run_turn_stream("third turn", |_| {})
        .await
        .expect("third turn");
    harness
        .run_turn_stream("what did you just call it", |_| {})
        .await
        .expect("fourth turn");

    let prompts = provider.recorded_prompts();
    let last = prompts.last().map(String::as_str).unwrap_or("");
    assert!(
        last.contains("The harness is Caliche."),
        "later prompt dropped the prior assistant turn: {last}"
    );
    assert!(
        last.contains(EARLY_TURN),
        "later prompt dropped the early user turn; a zero token cap keeps only min_messages: {last}"
    );
    assert!(
        last.contains("what did you just call it"),
        "later prompt missing the new question: {last}"
    );
}

#[tokio::test]
async fn run_turn_stream_emits_text_chunks_on_event_bus() {
    let provider = Arc::new(MockProvider::new(vec!["hello from stream".into()]));
    let mut harness = mock_harness(provider, Arc::new(ToolRegistry::new()), 1);
    let mut rx = harness.subscribe();
    let result = harness
        .run_turn_stream("say hello", |_| {})
        .await
        .expect("streamed turn");
    assert_eq!(result.response, "hello from stream");

    let mut saw_text = false;
    while let Ok(event) = rx.try_recv() {
        if let AgentEvent::TextChunk { content } = event {
            assert_eq!(content, "hello from stream");
            saw_text = true;
        }
    }
    assert!(saw_text, "session bus should receive live text chunks");
}

#[tokio::test]
async fn stream_turns_keep_tool_messages_in_service_memory() {
    use crate::agent::memory::{ApproximateTokenCounter, BufferMemory, BufferMemoryConfig};
    use crate::agent::tools::CalculatorTool;
    use futures_util::StreamExt;

    let tools = Arc::new(ToolRegistry::new());
    tools.register(Arc::new(CalculatorTool));
    let provider = Arc::new(MockProvider::with_tool_call(
        "calculator",
        serde_json::json!({"operation": "add", "a": 1, "b": 1}),
    ));
    let agent = crate::agent::Agent::new(provider, tools);
    let mut service =
        AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock agent service");
    service.set_memory_strategy(Some(Box::new(BufferMemory::new(
        BufferMemoryConfig::default().with_max_tokens(16_000),
        Arc::new(ApproximateTokenCounter::default()),
    ))));

    let mut stream = service.stream_chat("Add 1 and 1");
    while let Some(event) = stream.next().await {
        event.expect("first streamed turn");
    }
    drop(stream);

    let memory = service.memory().expect("memory");
    let first = memory.read().await.get_context().await.expect("context");
    assert!(
        first.iter().any(message_has_tool_transcript),
        "first streamed turn should keep tool transcript: {first:?}"
    );

    let mut stream = service.stream_chat("What was the sum?");
    while let Some(event) = stream.next().await {
        event.expect("second streamed turn");
    }
    drop(stream);

    let second = memory.read().await.get_context().await.expect("context");
    let users: Vec<String> = second
        .iter()
        .filter(|message| matches!(message.role, MessageRole::User))
        .filter_map(|message| message.as_text().map(ToOwned::to_owned))
        .collect();
    assert!(users.iter().any(|text| text.contains("Add 1 and 1")));
    assert!(users.iter().any(|text| text.contains("What was the sum?")));
    assert!(
        second.iter().any(message_has_tool_transcript),
        "second turn must still see the first tool transcript: {second:?}"
    );
}

fn message_has_tool_transcript(message: &Message) -> bool {
    matches!(message.role, MessageRole::Tool)
        || matches!(
            message.content,
            MessageContent::Structured {
                tool_calls: Some(ref calls),
                ..
            } if !calls.is_empty()
        )
}

#[tokio::test]
async fn run_task_stops_after_inner_chat_without_repeating_task() {
    let provider = Arc::new(MockProvider::new(vec![
        "The helper lives in src/lib.rs.".into(),
    ]));
    let mut harness = mock_harness(provider, Arc::new(ToolRegistry::new()), 3);
    let result = harness
        .run_task("Where is the helper defined?")
        .await
        .expect("run_task");

    assert_eq!(result.turns, 1);
    assert_eq!(result.response, "The helper lives in src/lib.rs.");
    assert_eq!(
        user_texts(&harness),
        vec!["Where is the helper defined?".to_string()]
    );
}

#[tokio::test]
async fn run_task_continues_only_when_inner_loop_hits_limit() {
    use crate::agent::tools::CalculatorTool;

    let tools = Arc::new(ToolRegistry::new());
    tools.register(Arc::new(CalculatorTool));
    let provider = Arc::new(MockProvider::with_tool_call(
        "calculator",
        serde_json::json!({"operation": "add", "a": 1, "b": 1}),
    ));
    let mut harness = mock_harness(provider, tools, 3);
    harness.service_mut().agent_mut().set_max_iterations(1);

    let result = harness
        .run_task("Add 1 and 1, then report the sum.")
        .await
        .expect("run_task");

    let users = user_texts(&harness);
    assert_eq!(result.turns, 2);
    assert_eq!(users.len(), 2);
    assert_eq!(users[0], "Add 1 and 1, then report the sum.");
    assert_eq!(users[1], CONTINUE_TASK_PROMPT);
    assert!(
        !users
            .iter()
            .skip(1)
            .any(|text| text.contains("Add 1 and 1"))
    );
}

fn reports_completion(reply: &str) -> bool {
    let harness = mock_harness(
        Arc::new(MockProvider::new(Vec::new())),
        Arc::new(ToolRegistry::new()),
        1,
    );
    harness.detect_task_completion(reply)
}

#[test]
fn status_other_than_complete_is_not_completion() {
    assert!(!reports_completion(
        r#"{"status": "in_progress", "note": "not \"complete\" yet"}"#
    ));
    assert!(!reports_completion(r#"{"status":"incomplete"}"#));
    assert!(!reports_completion(
        r#"{"status": "in_progress", "next": "complete"}"#
    ));
}

#[test]
fn task_complete_false_is_not_completion() {
    assert!(!reports_completion(
        r#"{"task_complete": false, "tests_passed": true}"#
    ));
}

#[test]
fn prose_without_a_completion_signal_is_not_completion() {
    assert!(!reports_completion(
        "I read src/lib.rs and will update the parser next."
    ));
}

#[test]
fn completion_fields_are_completion() {
    assert!(reports_completion(
        "```json\n{\"status\":\"complete\"}\n```"
    ));
    assert!(reports_completion(r#"Result: {"task_complete": true}"#));
    assert!(reports_completion(
        r#"I set "status" : "complete" in the summary."#
    ));
    assert!(reports_completion(
        r#"{"summary": "parser fixed", "status": "complete", "next_steps": []}"#
    ));
}

/// A reply at the inner loop limit that reports completion in JSON ends the task.
#[tokio::test]
async fn run_task_stops_at_the_limit_when_the_reply_reports_completion() {
    use crate::agent::tools::CalculatorTool;
    use crate::messages::{CompletionContent, CompletionResponse, ToolCallInfo};

    let tools = Arc::new(ToolRegistry::new());
    tools.register(Arc::new(CalculatorTool));
    let reply = CompletionResponse {
        content: CompletionContent::WithToolCalls {
            content: "Checked.\n```json\n{\"status\": \"complete\"}\n```".to_string(),
            tool_calls: vec![ToolCallInfo::from_json(
                "call_1".to_string(),
                "calculator".to_string(),
                r#"{"operation": "add", "a": 1, "b": 1}"#.to_string(),
            )],
        },
        finish_reason: Some(beef::lean::Cow::borrowed("tool_calls")),
        usage: None,
        reasoning_content: None,
    };
    let provider = Arc::new(MockProvider::with_responses(vec![reply]));
    let mut harness = mock_harness(provider, tools, 3);
    harness.service_mut().agent_mut().set_max_iterations(1);

    let result = harness
        .run_task("Add 1 and 1, then report the sum.")
        .await
        .expect("run_task");

    assert_eq!(result.turns, 1);
}

#[cfg(feature = "tools-rag")]
#[tokio::test]
async fn self_hosted_task_turns_skip_rag_wait_and_continuation_embeds() {
    use crate::agent::workspace_indexing::WorkspaceIndexingReady;
    use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
    use crate::rag::RagService;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    #[derive(Debug, Default)]
    struct CountingEmbedding {
        texts: AtomicUsize,
    }

    #[async_trait]
    impl EmbeddingProvider for CountingEmbedding {
        async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
            self.texts.fetch_add(request.texts.len(), Ordering::SeqCst);
            Ok(EmbeddingResponse {
                embeddings: request.texts.iter().map(|_| vec![0.1_f32; 8]).collect(),
                usage: None,
                model: "counting-stub".to_string(),
            })
        }

        fn name(&self) -> &str {
            "counting-stub"
        }
    }

    let embedding = Arc::new(CountingEmbedding::default());
    let rag = Arc::new(RagService::new(Arc::clone(&embedding) as _));
    rag.index_document("fn unique_identifier_xyz() {}", "helper.rs")
        .await
        .expect("index self-hosted helper");
    let embeds_after_index = embedding.texts.load(Ordering::SeqCst);
    assert!(embeds_after_index > 0, "indexing should embed the helper");

    let ready = Arc::new(WorkspaceIndexingReady::new());
    let provider = Arc::new(MockProvider::new(vec![
        "The helper is in helper.rs.".into(),
        "Continuing from the last step.".into(),
    ]));
    let tools = Arc::new(ToolRegistry::new());
    let mut agent = crate::agent::Agent::new(provider, tools);
    agent.set_workspace_rag(Some(Arc::clone(&rag)), Some(ready));
    let service =
        AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock agent service");
    let mut harness = AgentHarness::from_service(
        service,
        HarnessConfig {
            max_turns: 2,
            auto_checkpoint: false,
            permission_mode: PermissionModeConfig::AcceptAll,
            ..HarnessConfig::default()
        },
    )
    .expect("harness");

    let first_start = Instant::now();
    harness
        .run_turn("Where is unique_identifier_xyz defined?")
        .await
        .expect("first turn");
    let first_ms = first_start.elapsed();
    assert!(
        first_ms >= Duration::from_millis(150),
        "first turn should wait briefly for partial index, got {first_ms:?}"
    );
    assert!(
        first_ms < Duration::from_millis(1500),
        "first turn must not pay the old 5s indexing wait, got {first_ms:?}"
    );
    let embeds_after_first = embedding.texts.load(Ordering::SeqCst);
    assert!(
        embeds_after_first > embeds_after_index,
        "real task query should embed for workspace context"
    );

    let second_start = Instant::now();
    harness
        .run_turn(CONTINUE_TASK_PROMPT)
        .await
        .expect("continuation turn");
    let second_ms = second_start.elapsed();
    assert!(
        second_ms < Duration::from_millis(200),
        "continuation must not wait on ingest, got {second_ms:?}"
    );
    assert_eq!(
        embedding.texts.load(Ordering::SeqCst),
        embeds_after_first,
        "continuation prompt must not re-run hybrid workspace search"
    );
    assert_eq!(user_texts(&harness)[1], CONTINUE_TASK_PROMPT);
}

#[tokio::test]
async fn stream_chat_pairs_same_name_tool_start_and_result_ids() {
    use crate::agent::tools::CalculatorTool;
    use crate::messages::{CompletionContent, CompletionResponse, StreamEvent, ToolCallInfo};
    use futures_util::StreamExt;

    let tools = Arc::new(ToolRegistry::new());
    tools.register(Arc::new(CalculatorTool));
    let tool_turn = CompletionResponse {
        content: CompletionContent::WithToolCalls {
            content: String::new(),
            tool_calls: vec![
                ToolCallInfo::from_json(
                    "call-a",
                    "calculator",
                    r#"{"operation":"add","a":1,"b":1}"#,
                ),
                ToolCallInfo::from_json(
                    "call-b",
                    "calculator",
                    r#"{"operation":"add","a":2,"b":2}"#,
                ),
            ],
        },
        finish_reason: Some(beef::lean::Cow::borrowed("tool_calls")),
        usage: None,
        reasoning_content: None,
    };
    let provider = Arc::new(MockProvider::with_responses(vec![tool_turn]));
    let agent = crate::agent::Agent::new(provider, tools);
    let service =
        AgentService::new_with_agent(agent, ProviderType::OpenAI).expect("mock agent service");

    let mut starts = Vec::new();
    let mut results = Vec::new();
    let mut stream = service.stream_chat("Add the pairs");
    while let Some(event) = stream.next().await {
        match event.expect("streamed turn") {
            StreamEvent::ToolCallStart { id, action_id, .. } => {
                starts.push((id.to_string(), action_id));
            }
            StreamEvent::ToolCallResult { id, action_id, .. } => {
                results.push((id, action_id));
            }
            _ => {}
        }
    }
    drop(stream);

    assert_eq!(starts.len(), 2, "expected two tool starts: {starts:?}");
    assert_eq!(results.len(), 2, "expected two tool results: {results:?}");
    let mut paired = Vec::new();
    for (provider_id, start_action) in &starts {
        let result_action = results
            .iter()
            .find(|(id, _)| id.as_deref() == Some(provider_id.as_str()))
            .and_then(|(_, action_id)| action_id.clone());
        let start_action = start_action.clone().expect("start action id");
        let result_action = result_action.expect("result action id");
        assert_eq!(
            start_action, result_action,
            "start and result for {provider_id} must share an action id"
        );
        paired.push(start_action);
    }
    assert_ne!(
        paired[0], paired[1],
        "two calculator calls must not share one action id"
    );
}
