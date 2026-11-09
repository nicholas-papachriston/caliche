use crate::llm::cache::ConversionCache;
use crate::llm::config::CommonProviderOptions;
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::llm::models::{CohereModel, ModelProvider, ProviderType};
use crate::llm::provider::Provider;
use crate::messages::{CompletionRequest, CompletionResponse, Message, MessageRole, StreamEvent};
use anyhow::{Context, Result};
use async_trait::async_trait;
use beef::lean::Cow;
use futures_util::Stream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;

const COHERE_BASE_URL: &str = "https://api.cohere.ai/v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereMessage {
    #[serde(
        serialize_with = "crate::llm::provider_utils::serde_helpers::serialize_cow_static_str",
        deserialize_with = "crate::llm::provider_utils::serde_helpers::deserialize_cow_static_str"
    )]
    role: Cow<'static, str>,
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereTool {
    name: String,
    description: String,
    parameter_definitions: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereRequest {
    model: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    preamble: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_history: Option<Vec<CohereMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<CohereTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereResponse {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CohereToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<CohereUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereToolCall {
    name: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereUsage {
    #[serde(rename = "input_tokens")]
    input_tokens: u32,
    #[serde(rename = "output_tokens")]
    output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereStreamChunk {
    #[serde(rename = "event_type", alias = "type", default)]
    event_type: String,
    #[serde(default)]
    index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CohereToolCall>>,
    #[serde(rename = "tool_call_delta", skip_serializing_if = "Option::is_none")]
    tool_call_delta: Option<CohereStreamToolCallDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<CohereUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereEmbeddingRequest {
    texts: Vec<String>,
    model: String,
    #[serde(rename = "input_type")]
    input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_types: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<CohereUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    meta: Option<CohereEmbeddingMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CohereEmbeddingMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    api_version: Option<CohereApiVersion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereApiVersion {
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_documents: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankResponse {
    pub results: Vec<CohereRerankResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub meta: Option<CohereRerankMeta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankResult {
    pub index: usize,
    pub relevance_score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<CohereRerankDocument>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankDocument {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohereRerankMeta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub api_version: Option<CohereApiVersion>,
}

/// Configuration for `CohereProvider`
#[derive(Debug, Clone)]
pub struct CohereProviderConfig {
    /// Cohere API key
    pub api_key: String,
    /// Model identifier
    pub model: String,
    /// Base URL (defaults to <https://api.cohere.ai/v1>)
    pub base_url: String,
    /// Common provider options
    pub common: CommonProviderOptions,
}

impl CohereProviderConfig {
    /// Create a new `CohereProviderConfig`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: COHERE_BASE_URL.to_string(),
            common: CommonProviderOptions::default(),
        }
    }

    /// Create a builder for `CohereProviderConfig`
    #[must_use]
    pub fn builder() -> CohereProviderConfigBuilder {
        CohereProviderConfigBuilder::default()
    }
}

crate::impl_provider_config!(CohereProviderConfig);

/// Builder for `CohereProviderConfig`
#[derive(Debug, Default)]
pub struct CohereProviderConfigBuilder {
    api_key: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    common: CommonProviderOptions,
}

impl CohereProviderConfigBuilder {
    /// Set the API key
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the model
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the base URL
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Build the configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn build(self) -> anyhow::Result<CohereProviderConfig> {
        let api_key = crate::llm::provider_utils::resolve_api_key(
            self.api_key,
            "COHERE_API_KEY",
            "api_key is required for CohereProviderConfig or COHERE_API_KEY must be set",
        )?;
        let model =
            crate::llm::provider_utils::resolve_model(self.model, "COHERE_MODEL", "command-r-plus");
        let base_url = self.base_url.unwrap_or_else(|| COHERE_BASE_URL.to_string());
        Ok(CohereProviderConfig {
            api_key,
            model,
            base_url,
            common: self.common,
        })
    }
}

crate::impl_common_builder_methods!(CohereProviderConfigBuilder);

pub struct CohereProvider {
    client: Arc<reqwest::Client>,
    api_key: String,
    model: String,
    base_url: String,
    retry_strategy: Arc<dyn crate::traits::retry::RetryStrategy>,
    tool_cache: ConversionCache<Vec<CohereTool>>,
    message_cache: ConversionCache<(Option<String>, Vec<CohereMessage>, String)>,
}

impl std::fmt::Debug for CohereProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CohereProvider")
            .field("client", &"Arc<reqwest::Client>")
            .field("api_key", &"***")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("retry_strategy", &self.retry_strategy)
            .field("tool_cache", &self.tool_cache)
            .field("message_cache", &self.message_cache)
            .finish()
    }
}

impl CohereProvider {
    /// Create a new `CohereProvider` with configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn with_config(config: CohereProviderConfig) -> anyhow::Result<Self> {
        let client = crate::llm::provider_utils::create_http_client(&config.common)?;
        let retry_strategy = crate::llm::provider_utils::create_retry_strategy(&config.common);

        Ok(Self {
            client,
            api_key: config.api_key,
            model: config.model,
            base_url: config.base_url,
            retry_strategy,
            tool_cache: ConversionCache::new(),
            message_cache: ConversionCache::new(),
        })
    }

    /// Create a new `CohereProvider`, returning validation errors explicitly.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new(api_key: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let config = CohereProviderConfig::new(api_key, model);
        Self::with_config(config)
    }

    /// Create a `CohereProvider` from environment variables
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        std::future::ready(()).await;
        let mut builder = CohereProviderConfig::builder();
        if let Some(model) = model {
            builder = builder.model(model);
        }
        let config = builder.build()?;
        Self::with_config(config)
    }

    fn convert_messages(
        &self,
        messages: &[Message],
    ) -> Arc<(Option<String>, Vec<CohereMessage>, String)> {
        self.message_cache.get_or_compute_with_hash(
            messages,
            crate::llm::cache::hash_messages_for_cache,
            |messages| {
                let mut system_message = None;
                let mut chat_history = Vec::new();
                let mut current_message = String::new();

                for msg in messages {
                    match msg.role {
                        MessageRole::System => {
                            if let Some(text) = msg.content.as_text() {
                                system_message = Some(text.to_string());
                            }
                        }
                        MessageRole::User | MessageRole::Tool => {
                            if let Some(text) = msg.content.as_text() {
                                // If we already have a current message, add previous to history
                                if !current_message.is_empty() {
                                    chat_history.push(CohereMessage {
                                        role: Cow::borrowed("USER"),
                                        message: current_message.clone(),
                                    });
                                }
                                current_message = text.to_string();
                            }
                        }
                        MessageRole::Assistant => {
                            if let Some(text) = msg.content.as_text() {
                                // Add user message to history if exists
                                if !current_message.is_empty() {
                                    chat_history.push(CohereMessage {
                                        role: Cow::borrowed("USER"),
                                        message: current_message.clone(),
                                    });
                                    current_message.clear();
                                }
                                // Add assistant response to history
                                chat_history.push(CohereMessage {
                                    role: Cow::borrowed("CHATBOT"),
                                    message: text.to_string(),
                                });
                            }
                        }
                    }
                }

                // If current_message is still set, it's the final user message
                let final_message = if current_message.is_empty() {
                    // Fallback: use last user message if available
                    messages
                        .iter()
                        .rev()
                        .find(|m| matches!(m.role, MessageRole::User | MessageRole::Tool))
                        .and_then(|m| m.content.as_text())
                        .unwrap_or_default()
                        .to_string()
                } else {
                    current_message
                };

                (system_message, chat_history, final_message)
            },
        )
    }

    fn convert_tools(&self, tools: &[crate::messages::ToolSchema]) -> Arc<Vec<CohereTool>> {
        crate::llm::provider_utils::cached_tool_mapping(&self.tool_cache, tools, |tools| {
            tools
                .iter()
                .map(|tool| CohereTool {
                    name: tool.name.as_ref().to_string(),
                    description: tool.description.as_ref().to_string(),
                    parameter_definitions: crate::llm::provider_utils::tool_schema_to_json_value(
                        tool,
                    ),
                })
                .collect()
        })
    }

    /// Rerank documents for RAG pipeline optimization
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
        top_n: Option<usize>,
        return_documents: bool,
    ) -> Result<CohereRerankResponse> {
        let model = "rerank-english-v3.0".to_string(); // Default rerank model
        let request = CohereRerankRequest {
            query,
            documents,
            model: Some(model),
            top_n,
            return_documents: Some(return_documents),
        };

        tracing::debug!("Cohere rerank request: {:?}", request);

        let response = self
            .client
            .post(format!("{}/rerank", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send rerank request to Cohere")?;

        let response = crate::llm::provider_utils::check_response_status(response, "Cohere")
            .await
            .context("Cohere rerank API error")?;

        let rerank_response: CohereRerankResponse = response
            .json()
            .await
            .context("Failed to parse Cohere rerank response")?;

        Ok(rerank_response)
    }

    fn accumulate_cohere_tool_delta(
        accumulator: &mut crate::llm::provider_utils::StreamingToolCallAccumulator,
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
                other
                    if other.is_object()
                        && !other.as_object().is_some_and(serde_json::Map::is_empty) =>
                {
                    serde_json::to_string(other).ok()
                }
                other if !other.is_null() => Some(other.to_string()),
                _ => None,
            });
        accumulator.accumulate(index, delta.id, delta.name, arguments);
    }

    fn accumulate_cohere_tool_call(
        accumulator: &mut crate::llm::provider_utils::StreamingToolCallAccumulator,
        tool_call: CohereToolCall,
        index: Option<u32>,
    ) {
        let arguments = match &tool_call.parameters {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Null => None,
            other
                if other.is_object()
                    && !other.as_object().is_some_and(serde_json::Map::is_empty) =>
            {
                serde_json::to_string(other).ok()
            }
            other if !other.is_null() => Some(other.to_string()),
            _ => None,
        };
        accumulator.accumulate(index.unwrap_or(0), None, Some(tool_call.name), arguments);
    }
}

fn cohere_stream_events(
    parser: &mut crate::llm::provider_utils::CohereStreamParser,
    line: &str,
) -> (Vec<StreamEvent>, bool) {
    let Ok(chunk) = serde_json::from_str::<CohereStreamChunk>(line) else {
        tracing::debug!("Failed to parse Cohere stream chunk: {line}");
        return (Vec::new(), false);
    };
    match chunk.event_type.as_str() {
        "text-generation" | "content-delta" => (
            chunk
                .text
                .map(|content| vec![StreamEvent::TextChunk { content }])
                .unwrap_or_default(),
            false,
        ),
        "tool-calls-chunk" | "tool-call-start" | "tool-call-delta" => {
            if let Some(delta) = chunk.tool_call_delta {
                CohereProvider::accumulate_cohere_tool_delta(
                    parser.accumulator_mut(),
                    delta,
                    chunk.index,
                );
            } else {
                for (index, tool_call) in
                    chunk.tool_calls.unwrap_or_default().into_iter().enumerate()
                {
                    let index = chunk
                        .index
                        .or_else(|| u32::try_from(index).ok())
                        .or(Some(u32::MAX));
                    CohereProvider::accumulate_cohere_tool_call(
                        parser.accumulator_mut(),
                        tool_call,
                        index,
                    );
                }
            }
            (Vec::new(), false)
        }
        "tool-call-end" => (
            chunk
                .index
                .and_then(|index| parser.finalize_index(index, false))
                .into_iter()
                .collect(),
            false,
        ),
        "stream-end" | "message-end" => {
            let mut events = parser.flush_remaining();
            if let Some(usage) = chunk.usage {
                events.push(StreamEvent::Metadata {
                    usage: Some(crate::messages::TokenUsage::from_counts(
                        Some(usage.input_tokens),
                        Some(usage.output_tokens),
                        Some(usage.input_tokens + usage.output_tokens),
                    )),
                    finish_reason: chunk.finish_reason.map(beef::lean::Cow::owned),
                    metrics: None,
                });
            }
            events.push(StreamEvent::Done);
            (events, true)
        }
        other => {
            tracing::debug!(
                event_type = other,
                "Cohere: unrecognised stream event type, skipping"
            );
            (Vec::new(), false)
        }
    }
}

#[async_trait]
impl Provider for CohereProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let (system_message, chat_history, current_message) =
            (*self.convert_messages(&request.messages)).clone();

        let cohere_request = CohereRequest {
            model: self.model.clone(),
            message: current_message,
            preamble: system_message,
            chat_history: if chat_history.is_empty() {
                None
            } else {
                Some(chat_history)
            },
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: Some(false),
            tools: request
                .tools
                .as_ref()
                .map(|tools| (*self.convert_tools(tools)).clone()),
            tool_choice: request
                .tool_choice
                .as_ref()
                .and_then(crate::llm::provider_utils::convert_tool_choice_cohere),
        };

        tracing::debug!("Cohere request: {:?}", cohere_request);

        let client = &self.client;
        let api_key = &self.api_key;

        let response = crate::llm::provider_utils::retry_llm_request(
            self.retry_strategy.as_ref(),
            "Cohere",
            || {
                client
                    .post(format!("{}/chat", self.base_url))
                    .header("Authorization", format!("Bearer {api_key}"))
                    .header("Content-Type", "application/json")
                    .json(&cohere_request)
                    .send()
            },
        )
        .await?;

        let cohere_response: CohereResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse Cohere response: {e}"))?;

        tracing::debug!("Cohere response: {:?}", cohere_response);

        let usage = cohere_response.usage.map(|u| {
            crate::messages::TokenUsage::from_counts(
                Some(u.input_tokens),
                Some(u.output_tokens),
                Some(u.input_tokens + u.output_tokens),
            )
        });

        crate::llm::provider_utils::log_token_usage(usage.as_ref(), "Cohere");

        let content = if let Some(ref tool_calls) = cohere_response.tool_calls {
            if tool_calls.is_empty() {
                crate::messages::CompletionContent::Text {
                    content: cohere_response.text,
                }
            } else {
                crate::messages::CompletionContent::WithToolCalls {
                    content: cohere_response.text,
                    tool_calls: tool_calls
                        .iter()
                        .map(|tc| {
                            crate::messages::ToolCallInfo::from_json(
                                format!("cohere-{}", uuid::Uuid::new_v4()),
                                tc.name.clone(),
                                tc.parameters.to_string(),
                            )
                        })
                        .collect(),
                }
            }
        } else {
            crate::messages::CompletionContent::Text {
                content: cohere_response.text,
            }
        };

        Ok(CompletionResponse {
            content,
            finish_reason: cohere_response.finish_reason.map(beef::lean::Cow::owned),
            usage,
            reasoning_content: None,
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        let (system_message, chat_history, current_message) =
            (*self.convert_messages(&request.messages)).clone();

        let model = self.model.clone();
        let client = self.client.clone();
        let api_key = self.api_key.clone();
        let base_url = self.base_url.clone();
        let tools_for_stream = request
            .tools
            .as_ref()
            .map(|tools| (*self.convert_tools(tools)).clone());
        let tool_choice_for_stream = request
            .tool_choice
            .as_ref()
            .and_then(crate::llm::provider_utils::convert_tool_choice_cohere);

        let cohere_request = CohereRequest {
            model,
            message: current_message,
            preamble: system_message,
            chat_history: if chat_history.is_empty() {
                None
            } else {
                Some(chat_history)
            },
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            stream: Some(true),
            tools: tools_for_stream,
            tool_choice: tool_choice_for_stream,
        };

        tracing::debug!("Cohere streaming request: {:?}", cohere_request);

        Box::pin(async_stream::stream! {
            let response = match client
                .post(format!("{base_url}/chat"))
                .header("Authorization", format!("Bearer {api_key}"))
                .header("Content-Type", "application/json")
                .json(&cohere_request)
                .send()
                .await
            {
                Ok(res) => res,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Failed to send request to Cohere: {e}"));
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let text = response.text().await.unwrap_or_default();
                yield Err(anyhow::anyhow!(
                    "Cohere API error: Status {status} - {text}"
                ));
                return;
            }

            let mut lines =
                crate::llm::provider_utils::sse_data_lines(response);
            use futures_util::StreamExt;

            let mut tool_parser = crate::llm::provider_utils::CohereStreamParser::new();

            while let Some(line_result) = lines.next().await {
                let line = match line_result {
                    Ok(crate::llm::provider_utils::SseDataLine::Done) => {
                        for evt in tool_parser.flush_remaining() {
                            yield Ok(evt);
                        }
                        yield Ok(StreamEvent::Done);
                        return;
                    }
                    Ok(crate::llm::provider_utils::SseDataLine::Data(json_str)) => json_str,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                let (events, done) = cohere_stream_events(&mut tool_parser, &line);
                for event in events {
                    yield Ok(event);
                }
                if done {
                    return;
                }
            }

            for evt in tool_parser.flush_remaining() {
                yield Ok(evt);
            }
            yield Ok(StreamEvent::Done);
        })
    }

    fn name(&self) -> &'static str {
        "cohere"
    }

    fn supports_native_tools(&self) -> bool {
        true
    }
}

impl ModelProvider for CohereProvider {
    fn available_models(&self) -> Vec<String> {
        CohereModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Cohere
    }
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        let model = request
            .model
            .unwrap_or_else(|| "embed-english-v3.0".to_string());

        // Determine input type based on model or default to search_document
        let input_type = "search_document";

        let cohere_request = CohereEmbeddingRequest {
            texts: request.texts,
            model: model.clone(),
            input_type: input_type.to_string(),
            embedding_types: None,
            truncate: None,
        };

        tracing::debug!("Cohere embedding request: {:?}", cohere_request);

        let response = self
            .client
            .post(format!("{}/embed", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&cohere_request)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send embedding request to Cohere: {e}"))?;

        let response =
            crate::llm::provider_utils::check_response_status(response, "Cohere").await?;

        let cohere_response: CohereEmbeddingResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse Cohere embedding response: {e}"))?;

        tracing::debug!(
            "Cohere embedding response received with {} embeddings",
            cohere_response.embeddings.len()
        );

        let usage = cohere_response.usage.map(|u| {
            crate::messages::TokenUsage::from_counts(
                Some(u.input_tokens),
                Some(u.output_tokens),
                Some(u.input_tokens + u.output_tokens),
            )
        });

        crate::llm::provider_utils::log_token_usage(usage.as_ref(), "Cohere");

        Ok(EmbeddingResponse {
            embeddings: cohere_response.embeddings,
            usage,
            model,
        })
    }

    fn name(&self) -> &'static str {
        "cohere"
    }
}
