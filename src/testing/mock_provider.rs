//! Mock provider implementation for testing

use crate::llm::Provider;
use crate::messages::{
    CompletionContent, CompletionRequest, CompletionResponse, StreamEvent, TokenUsage, ToolCallInfo,
};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// Mock LLM provider for testing
#[derive(Debug, Clone)]
pub struct MockProvider {
    responses: Arc<Mutex<Vec<CompletionResponse>>>,
    should_error: bool,
    token_usage: TokenUsage,
    /// Prompts observed by [`Provider::complete`], oldest first.
    recorded_prompts: Arc<Mutex<Vec<RecordedPrompt>>>,
}

#[derive(Debug, Clone)]
struct RecordedPrompt {
    text: String,
    reasoning_chars: usize,
}

impl MockProvider {
    pub fn new(responses: Vec<String>) -> Self {
        let responses = responses
            .into_iter()
            .map(|text| CompletionResponse {
                content: CompletionContent::Text { content: text },
                finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
                usage: Some(TokenUsage {
                    prompt_tokens: Some(10),
                    completion_tokens: Some(20),
                    total_tokens: Some(30),
                    embedding_tokens: None,
                    context_window: None,
                    cache_creation_tokens: None,
                    cache_read_tokens: None,
                    last_prompt_tokens: None,
                }),
                reasoning_content: None,
            })
            .collect();

        Self {
            responses: Arc::new(Mutex::new(responses)),
            should_error: false,
            token_usage: TokenUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
                embedding_tokens: None,
                context_window: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            },
            recorded_prompts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_responses(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            should_error: false,
            token_usage: TokenUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
                embedding_tokens: None,
                context_window: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            },
            recorded_prompts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_error() -> Self {
        Self {
            responses: Arc::new(Mutex::new(vec![])),
            should_error: true,
            token_usage: TokenUsage {
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: None,
                embedding_tokens: None,
                context_window: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            },
            recorded_prompts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn with_tool_call(tool_name: &str, tool_args: serde_json::Value) -> Self {
        let response = CompletionResponse {
            content: CompletionContent::WithToolCalls {
                tool_calls: vec![ToolCallInfo::from_json(
                    "test_call_1".to_string(),
                    tool_name.to_string(),
                    serde_json::to_string(&tool_args).unwrap_or_default(),
                )],
                content: "Using tool".to_string(),
            },
            finish_reason: Some(beef::lean::Cow::borrowed("tool_calls")),
            usage: Some(TokenUsage {
                prompt_tokens: Some(15),
                completion_tokens: Some(5),
                total_tokens: Some(20),
                embedding_tokens: None,
                context_window: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            }),
            reasoning_content: None,
        };

        Self {
            responses: Arc::new(Mutex::new(vec![response])),
            should_error: false,
            token_usage: TokenUsage {
                prompt_tokens: Some(15),
                completion_tokens: Some(5),
                total_tokens: Some(20),
                embedding_tokens: None,
                context_window: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            },
            recorded_prompts: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Prompt texts passed to [`Provider::complete`], oldest first.
    pub fn recorded_prompts(&self) -> Vec<String> {
        self.recorded_prompts
            .lock()
            .map(|guard| guard.iter().map(|prompt| prompt.text.clone()).collect())
            .unwrap_or_default()
    }

    /// Sum of stored reasoning lengths on each [`Provider::complete`] request.
    ///
    /// Index 0 is the first turn. A later turn that replays service memory
    /// reports the prior trace length here.
    pub fn recorded_reasoning_chars(&self) -> Vec<usize> {
        self.recorded_prompts
            .lock()
            .map(|guard| guard.iter().map(|prompt| prompt.reasoning_chars).collect())
            .unwrap_or_default()
    }

    fn record_prompt(&self, request: &CompletionRequest) {
        let text = request
            .messages
            .iter()
            .map(|message| message.extract_text())
            .collect::<Vec<_>>()
            .join("\n");
        let reasoning_chars = request
            .messages
            .iter()
            .map(|message| message.reasoning_content().map(str::len).unwrap_or(0))
            .sum();
        if let Ok(mut guard) = self.recorded_prompts.lock() {
            guard.push(RecordedPrompt {
                text,
                reasoning_chars,
            });
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.record_prompt(&request);
        if self.should_error {
            anyhow::bail!("Mock provider error");
        }

        let mut responses = self
            .responses
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock mock provider responses: {}", e))?;
        if responses.is_empty() {
            // Default response if empty
            Ok(CompletionResponse {
                content: CompletionContent::Text {
                    content: "Mock response".to_string(),
                },
                finish_reason: Some(beef::lean::Cow::borrowed("end_turn")),
                usage: Some(self.token_usage),
                reasoning_content: None,
            })
        } else {
            Ok(responses.remove(0))
        }
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<StreamEvent>> + Send + '_>> {
        let this = self.clone();
        Box::pin(async_stream::stream! {
            match this.complete(request).await {
                Ok(response) => {
                    if let Some(thinking) = response.reasoning_content
                        && !thinking.is_empty()
                    {
                        yield Ok(StreamEvent::ThinkingChunk {
                            content: thinking,
                            action_id: None,
                        });
                    }
                    let text = response.content.as_text();
                    if !text.is_empty() {
                        yield Ok(StreamEvent::TextChunk {
                            content: text.to_string(),
                        });
                    }
                    for call in response.content.extract_tool_calls() {
                        yield Ok(StreamEvent::ToolCallStart {
                            id: call.id,
                            name: call.name,
                            arguments: call.arguments,
                            thought_signature: call.thought_signature,
                            action_id: None,
                        });
                    }
                    yield Ok(StreamEvent::Metadata {
                        usage: response.usage,
                        finish_reason: response.finish_reason,
                        metrics: None,
                    });
                    yield Ok(StreamEvent::Done);
                }
                Err(err) => yield Err(err),
            }
        })
    }

    fn name(&self) -> &str {
        "mock"
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

// Re-export sandbox types for advanced use cases
pub use crate::agent::service::sandbox::{MockBehavior, RecordedInteraction};
