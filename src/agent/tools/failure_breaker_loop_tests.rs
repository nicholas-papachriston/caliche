//! One breaker per user turn, shared across iterations of each chat loop.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use futures_util::StreamExt;

use crate::agent::core::Agent;
use crate::agent::kernel::turn::TurnResult;
use crate::agent::tools::ToolRegistry;
use crate::agent::tools::names::TOOL_BASH;
use crate::agent::tools::tool::{Tool, ToolMeta, ToolResult};
use crate::messages::{CompletionContent, CompletionResponse, TokenUsage, ToolCallInfo};
use crate::testing::mock_provider::MockProvider;

#[tokio::test]
async fn chat_loop_blocks_the_second_identical_failure() {
    super::with_injected_breaker_env(Some("1"), chat_loop_body()).await;
}

async fn chat_loop_body() {
    let hits = counting_bash();
    let mut agent = agent_with(Arc::clone(&hits));
    let response = agent.chat("typecheck").await.expect("chat");
    assert!(
        response.text.contains("done"),
        "chat loop stopped before the scripted reply: {}",
        response.text
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn handoff_loop_blocks_the_second_identical_failure() {
    super::with_injected_breaker_env(Some("1"), handoff_loop_body()).await;
}

async fn handoff_loop_body() {
    let hits = counting_bash();
    let mut agent = agent_with(Arc::clone(&hits));
    let turn = agent.execute_turn("typecheck").await.expect("handoff loop");
    let TurnResult::Reply(text, _) = turn else {
        panic!("handoff loop did not reply: {turn:?}");
    };
    assert!(
        text.contains("done"),
        "handoff loop stopped before the scripted reply: {text}"
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn stream_loop_blocks_the_second_identical_failure() {
    super::with_injected_breaker_env(Some("1"), stream_loop_body()).await;
}

async fn stream_loop_body() {
    let hits = counting_bash();
    let agent = agent_with(Arc::clone(&hits));
    let mut stream = agent.stream_chat_with_tools("typecheck");
    let mut text = String::new();
    while let Some(event) = stream.next().await {
        let event = event.expect("stream event");
        if let crate::messages::StreamEvent::TextChunk { content } = event {
            text.push_str(&content);
        }
    }
    assert!(
        text.contains("done"),
        "stream loop stopped before the scripted reply: {text}"
    );
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

fn counting_bash() -> Arc<AtomicUsize> {
    Arc::new(AtomicUsize::new(0))
}

fn agent_with(hits: Arc<AtomicUsize>) -> Agent {
    let tools = Arc::new(ToolRegistry::new());
    tools.register(Arc::new(CountingBash { hits }));
    Agent::new(Arc::new(scripted_provider()), tools)
}

fn scripted_provider() -> MockProvider {
    MockProvider::with_responses(vec![
        tool_response("call-1"),
        tool_response("call-2"),
        CompletionResponse {
            content: CompletionContent::Text {
                content: "done".to_string(),
            },
            finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
            usage: Some(usage()),
            reasoning_content: None,
        },
    ])
}

fn tool_response(id: &str) -> CompletionResponse {
    CompletionResponse {
        content: CompletionContent::WithToolCalls {
            tool_calls: vec![ToolCallInfo::from_json(
                id,
                TOOL_BASH,
                r#"{"command":"bun run typecheck","working_directory":"/workspace"}"#,
            )],
            content: String::new(),
        },
        finish_reason: Some(beef::lean::Cow::borrowed("tool_calls")),
        usage: Some(usage()),
        reasoning_content: None,
    }
}

fn usage() -> TokenUsage {
    TokenUsage {
        prompt_tokens: Some(10),
        completion_tokens: Some(5),
        total_tokens: Some(15),
        embedding_tokens: None,
        context_window: None,
        cache_creation_tokens: None,
        cache_read_tokens: None,
        last_prompt_tokens: None,
    }
}

#[derive(Debug)]
struct CountingBash {
    hits: Arc<AtomicUsize>,
}

impl ToolMeta for CountingBash {
    fn name(&self) -> &str {
        TOOL_BASH
    }
    fn description(&self) -> &str {
        "counting bash"
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}})
    }
}

#[async_trait]
impl Tool for CountingBash {
    async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
        self.hits.fetch_add(1, Ordering::SeqCst);
        let mut result = ToolResult::error("Command failed with exit code 2:\nsrc/main.ts: nope");
        result.typed_payload = Some(serde_json::json!({"exit_code": 2}));
        result
    }
}
