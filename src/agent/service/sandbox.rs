//! Sandbox mode for testing agents without making real API calls

use crate::llm::{Provider, ProviderType};
use crate::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use anyhow::Result;
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Mock behavior for sandbox providers
#[derive(Debug, Clone)]
pub enum MockBehavior {
    /// Deterministic responses based on input hash
    Deterministic,
    /// Fixed response for all requests
    Fixed(String),
    /// Record interactions for replay
    Recording,
    /// Replay from recorded interactions
    Replay(Vec<RecordedInteraction>),
}

/// Recorded interaction for replay
#[derive(Debug, Clone)]
pub struct RecordedInteraction {
    pub request_id: String,
    pub prompt: String,
    pub response: String,
    pub usage: crate::messages::TokenUsage,
}

/// Mock LLM provider for testing
pub struct MockProvider {
    behavior: MockBehavior,
    interactions: Arc<RwLock<Vec<RecordedInteraction>>>,
    provider_type: ProviderType,
}

impl std::fmt::Debug for MockProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockProvider")
            .field("behavior", &self.behavior)
            .field("interactions", &"Arc<RwLock<Vec<RecordedInteraction>>>")
            .field("provider_type", &self.provider_type)
            .finish()
    }
}

impl MockProvider {
    #[must_use]
    pub fn new(provider_type: ProviderType, behavior: MockBehavior) -> Self {
        Self {
            behavior,
            interactions: Arc::new(RwLock::new(Vec::new())),
            provider_type,
        }
    }

    /// Returns the provider type this mock simulates.
    #[must_use]
    pub const fn provider_type(&self) -> ProviderType {
        self.provider_type
    }

    fn generate_response(&self, request: &CompletionRequest) -> String {
        match &self.behavior {
            MockBehavior::Deterministic => {
                // Generate deterministic response based on input
                let input: String = request
                    .messages
                    .iter()
                    .map(crate::messages::Message::extract_text)
                    .collect::<String>();
                let hash = input.len() as u64; // Simple hash
                format!(
                    "Mock response for input (hash: {}): {}",
                    hash,
                    caliche_contracts::text::truncate_utf8(&input, 50)
                )
            }
            MockBehavior::Fixed(response) => response.clone(),
            MockBehavior::Recording => {
                // Generate a simple response
                "Mock response (recording)".to_string()
            }
            MockBehavior::Replay(recordings) => {
                // Find matching recording
                let prompt = request
                    .messages
                    .last()
                    .map(crate::messages::Message::extract_text)
                    .unwrap_or_default();

                recordings
                    .iter()
                    .find(|r| r.prompt.contains(&prompt) || prompt.contains(&r.prompt))
                    .map_or_else(
                        || "Mock response (no match)".to_string(),
                        |r| r.response.clone(),
                    )
            }
        }
    }

    const fn generate_usage(&self) -> crate::messages::TokenUsage {
        crate::messages::TokenUsage {
            prompt_tokens: Some(100),
            completion_tokens: Some(50),
            total_tokens: Some(150),
            embedding_tokens: None,
            context_window: None,
            cache_creation_tokens: None,
            cache_read_tokens: None,
            last_prompt_tokens: None,
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let response_text = self.generate_response(&request);
        let usage = self.generate_usage();

        // Record interaction if in recording mode
        if matches!(self.behavior, MockBehavior::Recording) {
            let interaction = RecordedInteraction {
                request_id: Uuid::now_v7().to_string(),
                prompt: request
                    .messages
                    .last()
                    .map(crate::messages::Message::extract_text)
                    .unwrap_or_default(),
                response: response_text.clone(),
                usage,
            };
            self.interactions.write().await.push(interaction);
        }

        Ok(CompletionResponse {
            content: crate::messages::CompletionContent::Text {
                content: response_text,
            },
            usage: Some(usage),
            finish_reason: Some(beef::lean::Cow::borrowed("stop")),
            reasoning_content: None,
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + '_>> {
        let response_text = self.generate_response(&request);
        let usage = self.generate_usage();
        let chunks: Vec<String> = response_text
            .chars()
            .collect::<Vec<_>>()
            .chunks(5)
            .map(|chunk| chunk.iter().collect())
            .collect();

        Box::pin(async_stream::stream! {
            for chunk in chunks {
                yield Ok(StreamEvent::TextChunk { content: chunk });
            }
            yield Ok(StreamEvent::Metadata {
                usage: Some(usage),
                finish_reason: Some(beef::lean::Cow::borrowed("stop")),
                metrics: None,
            });
            yield Ok(StreamEvent::Done);
        })
    }

    fn name(&self) -> &'static str {
        "mock"
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_native_tools(&self) -> bool {
        true
    }
}

/// Agent sandbox for testing
pub struct AgentSandbox {
    mock_providers: std::collections::HashMap<ProviderType, MockBehavior>,
    record_interactions: bool,
    interactions: Arc<RwLock<Vec<RecordedInteraction>>>,
}

impl AgentSandbox {
    #[must_use]
    pub fn new() -> Self {
        Self {
            mock_providers: std::collections::HashMap::new(),
            record_interactions: false,
            interactions: Arc::new(RwLock::new(Vec::new())),
        }
    }

    #[must_use]
    pub fn mock_provider(mut self, provider: ProviderType, behavior: MockBehavior) -> Self {
        self.mock_providers.insert(provider, behavior);
        self
    }

    #[must_use]
    pub const fn record_interactions(mut self, record: bool) -> Self {
        self.record_interactions = record;
        self
    }

    #[must_use]
    pub const fn build(self) -> Self {
        self
    }

    #[must_use]
    pub fn create_mock_provider(&self, provider_type: ProviderType) -> Option<Arc<MockProvider>> {
        let behavior = self
            .mock_providers
            .get(&provider_type)
            .cloned()
            .unwrap_or(MockBehavior::Deterministic);

        Some(Arc::new(MockProvider::new(provider_type, behavior)))
    }

    /// Create a mock provider using the first configured behavior, falling back to `Deterministic`.
    #[must_use]
    pub fn create_default_mock_provider(&self) -> Arc<MockProvider> {
        let (provider_type, behavior) = self.mock_providers.iter().next().map_or(
            (ProviderType::External, MockBehavior::Deterministic),
            |(k, v)| (*k, v.clone()),
        );
        Arc::new(MockProvider::new(provider_type, behavior))
    }

    pub async fn get_recorded_interactions(&self) -> Vec<RecordedInteraction> {
        self.interactions.read().await.clone()
    }

    pub async fn clear_recordings(&self) {
        self.interactions.write().await.clear();
    }
}

impl Default for AgentSandbox {
    fn default() -> Self {
        Self::new()
    }
}
