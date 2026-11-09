//! Generic OpenAI-compatible provider implementation.
//!
//! This single implementation handles all providers that follow the `OpenAI`
//! chat completions API format, eliminating code duplication.

use super::config::OpenAICompatConfig;
use super::types::{
    OpenAICompatContent, OpenAICompatEmbeddingRequest, OpenAICompatEmbeddingResponse,
    OpenAICompatFunctionCall, OpenAICompatFunctionSchema, OpenAICompatMessage, OpenAICompatRequest,
    OpenAICompatResponse, OpenAICompatTool, OpenAICompatToolCall, OpenAICompatUsage,
};
use crate::llm::cache::ConversionCache;
use crate::llm::config::CommonProviderOptions;
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::llm::provider::Provider;
use crate::llm::provider_utils::first_token::{
    WAITING_FOR_FIRST_TOKEN, loading_status, with_waiting_status,
};
use crate::llm::provider_utils::{
    FirstByteDeadline, SseLimits, no_first_token_error, openai_stream_channel,
    openai_usage_to_token_usage, parse_sse_stream_to_channel, retry_stream_establishment,
};
use crate::messages::{
    CompletionContent, CompletionRequest, CompletionResponse, Message, MessageContent, MessageRole,
    REASONING_CONTENT_METADATA_KEY, StreamEvent, TokenUsage,
};
use anyhow::Context;
use async_trait::async_trait;
use beef::lean::Cow;
use bytes::Bytes;
use futures_util::Stream;
use std::fmt::Write;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

/// Whole-request cap for a request with a first-token limit. The
/// first-token and stall limits bound such a stream; this only ends one
/// that keeps sending for a day.
const LOCAL_REQUEST_CAP: Duration = Duration::from_secs(24 * 60 * 60);

fn full_llm_request_logging_enabled() -> bool {
    std::env::var("CALICHE_LOG_FULL_LLM_REQUESTS")
        .ok()
        .as_deref()
        .is_some_and(|value| matches!(value, "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"))
}

fn openai_compat_content_chars(content: &OpenAICompatContent) -> usize {
    match content {
        OpenAICompatContent::String(text) => text.len(),
        OpenAICompatContent::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
            .map(str::len)
            .sum(),
    }
}

fn openai_compat_message_chars(message: &OpenAICompatMessage) -> usize {
    let content_chars = message
        .content
        .as_ref()
        .map_or(0, openai_compat_content_chars);
    let reasoning_chars = message
        .reasoning_content
        .as_ref()
        .map_or(0, |content| content.len());
    content_chars.saturating_add(reasoning_chars)
}

fn convert_openai_message(message: &Message, thinking_enabled: bool) -> OpenAICompatMessage {
    let role = Cow::borrowed(crate::llm::provider_utils::role_to_str(&message.role));
    match message.role {
        MessageRole::Tool => {
            let tool_call_id = match &message.content {
                MessageContent::Structured {
                    tool_call_id: Some(id),
                    ..
                } => Some(id.clone().into_owned()),
                _ => None,
            };
            OpenAICompatMessage {
                role,
                content: Some(OpenAICompatContent::String(Arc::new(
                    crate::llm::provider_utils::extract_message_text(message),
                ))),
                tool_calls: None,
                tool_call_id,
                reasoning_content: None,
            }
        }
        MessageRole::Assistant => convert_openai_assistant(message, thinking_enabled),
        MessageRole::System | MessageRole::User => {
            let content = match &message.content {
                MessageContent::Multimodal(parts) => OpenAICompatContent::Array(
                    crate::llm::provider_utils::content_parts_to_json_array(
                        parts,
                        crate::llm::provider_utils::ContentPartJsonFormat::OpenAI,
                    ),
                ),
                _ => OpenAICompatContent::String(Arc::new(
                    crate::llm::provider_utils::extract_message_text(message),
                )),
            };
            OpenAICompatMessage {
                role,
                content: Some(content),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }
        }
    }
}

/// Read persisted thinking/reasoning from message metadata (stream or complete paths).
fn message_stored_reasoning(message: &Message) -> Option<Arc<String>> {
    let metadata = message.metadata.as_ref()?;
    let value = metadata
        .get(REASONING_CONTENT_METADATA_KEY)
        .or_else(|| metadata.get("reasoning"))?;
    let text = value.as_str()?.trim();
    if text.is_empty() {
        return None;
    }
    Some(Arc::new(text.to_string()))
}

fn convert_openai_assistant(message: &Message, thinking_enabled: bool) -> OpenAICompatMessage {
    let stored_reasoning = message_stored_reasoning(message);
    let (content, tool_calls, reasoning_content) = match &message.content {
        MessageContent::Text(text) => (Arc::clone(text), None, stored_reasoning),
        MessageContent::Multimodal(_) => (
            Arc::new(message.content.as_text().unwrap_or("").to_string()),
            None,
            stored_reasoning,
        ),
        MessageContent::Structured {
            text, tool_calls, ..
        } => {
            let converted_calls = tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .map(|call| OpenAICompatToolCall {
                        id: call.id.clone(),
                        call_type: Cow::borrowed("function"),
                        function: OpenAICompatFunctionCall {
                            name: call.name.clone(),
                            arguments: call.arguments_json_arc(),
                        },
                    })
                    .collect()
            });
            let content = text
                .as_ref()
                .map_or_else(|| Arc::new(String::new()), Arc::clone);
            // Thinking models require `reasoning_content` when replaying tool_calls.
            // Prefer stored thinking from metadata — never duplicate assistant `text`
            // into both `content` and `reasoning_content`.
            let reasoning = if thinking_enabled && converted_calls.is_some() {
                Some(stored_reasoning.unwrap_or_else(|| Arc::new(String::new())))
            } else {
                stored_reasoning
            };
            (content, converted_calls, reasoning)
        }
    };
    OpenAICompatMessage {
        role: Cow::borrowed("assistant"),
        content: Some(OpenAICompatContent::String(content)),
        tool_calls,
        tool_call_id: None,
        reasoning_content,
    }
}

/// Apply auth, provider and user headers to a request builder.
///
/// The retry closure in `complete` owns clones of the config, so this takes the
/// pieces it needs instead of `&self`.
async fn apply_request_headers<C: OpenAICompatConfig>(
    mut req: reqwest::RequestBuilder,
    config: &C,
    common: &CommonProviderOptions,
) -> reqwest::RequestBuilder {
    for (key, value) in config.resolve_auth_headers().await {
        req = req.header(key, value);
    }
    for (key, value) in config.extra_headers() {
        req = req.header(key, value);
    }
    req = req.header("Content-Type", "application/json");
    if let Some(headers) = &common.custom_headers {
        for (key, value) in headers {
            req = req.header(key, value);
        }
    }
    req
}

fn log_openai_compat_request(provider_name: &str, request: &OpenAICompatRequest, streaming: bool) {
    if full_llm_request_logging_enabled() {
        tracing::debug!("{} request: {:?}", provider_name, request);
        return;
    }

    let message_count = request.messages.len();
    let message_chars: usize = request
        .messages
        .iter()
        .map(openai_compat_message_chars)
        .sum();
    let tool_count = request.tools.as_ref().map_or(0, |tools| tools.len());

    tracing::debug!(
        target: "caliche::llm",
        provider = %provider_name,
        model = %request.model,
        streaming,
        message_count,
        message_chars,
        tool_count,
        has_tool_choice = request.tool_choice.is_some(),
        has_response_format = request.response_format.is_some(),
        max_tokens = request.max_tokens,
        max_completion_tokens = request.max_completion_tokens,
        "openai_compat.request"
    );
}

/// Generic provider for all OpenAI-compatible APIs.
///
/// This provider handles `OpenAI`, `OpenRouter`, `LMStudio`, Perplexity, and any other
/// provider that follows the `OpenAI` chat completions API format.
pub struct OpenAICompatibleProvider<C: OpenAICompatConfig> {
    client: Arc<reqwest::Client>,
    config: C,
    common: CommonProviderOptions,
    retry_strategy: Arc<dyn crate::traits::retry::RetryStrategy>,
    tool_cache: ConversionCache<Vec<OpenAICompatTool>>,
    message_cache: ConversionCache<Vec<OpenAICompatMessage>>,
    /// Cached full URL for chat/completions to avoid format! on every request.
    chat_completions_url: String,
    /// Cached full URL for embeddings.
    embeddings_url: String,
    /// Cached model string to avoid `config.model().to_string()` on every request.
    cached_model_string: String,
    /// Dynamically fetched context window for the active model (0 = not set).
    dynamic_context_window: Arc<AtomicUsize>,
}

impl<C: OpenAICompatConfig> std::fmt::Debug for OpenAICompatibleProvider<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAICompatibleProvider")
            .field("provider", &self.config.provider_name())
            .field("client", &"Arc<reqwest::Client>")
            .field("config", &self.config)
            .field("common", &self.common)
            .field("retry_strategy", &self.retry_strategy)
            .field("tool_cache", &self.tool_cache)
            .field("message_cache", &self.message_cache)
            .field("chat_completions_url", &self.chat_completions_url)
            .field("embeddings_url", &self.embeddings_url)
            .field("cached_model_string", &self.cached_model_string)
            .field(
                "dynamic_context_window",
                &self.dynamic_context_window.load(Ordering::Relaxed),
            )
            .finish()
    }
}

impl<C: OpenAICompatConfig> OpenAICompatibleProvider<C> {
    /// Create a new provider with the given configuration and common options.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn with_options(config: C, common: CommonProviderOptions) -> anyhow::Result<Self> {
        crate::gateway::egress::current_policy()?.validate_static(
            crate::gateway::egress::EgressClass::Provider,
            config.base_url(),
        )?;
        let client = crate::llm::provider_utils::create_http_client(&common)?;
        let retry_strategy = crate::llm::provider_utils::create_retry_strategy(&common);

        let chat_completions_url = config.chat_completions_url().map_or_else(
            || {
                format!(
                    "{}/chat/completions",
                    config.base_url().trim_end_matches('/')
                )
            },
            str::to_owned,
        );
        let embeddings_url = config.embeddings_url().map_or_else(
            || format!("{}/embeddings", config.base_url().trim_end_matches('/')),
            str::to_owned,
        );
        let cached_model_string = config.model().to_string();

        Ok(Self {
            client,
            config,
            common,
            retry_strategy,
            tool_cache: ConversionCache::new(),
            message_cache: ConversionCache::new(),
            chat_completions_url,
            embeddings_url,
            cached_model_string,
            dynamic_context_window: Arc::new(AtomicUsize::new(0)),
        })
    }

    /// Create a provider with default common options.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn with_defaults(config: C) -> anyhow::Result<Self> {
        Self::with_options(config, CommonProviderOptions::default())
    }

    /// Get a reference to the configuration.
    pub const fn config(&self) -> &C {
        &self.config
    }

    /// Get a reference to the HTTP client (for provider-specific API calls).
    pub fn http_client(&self) -> &reqwest::Client {
        &self.client
    }

    /// Set the dynamically fetched context window for the active model.
    pub fn set_dynamic_context_window(&self, window: usize) {
        self.dynamic_context_window.store(window, Ordering::Relaxed);
    }

    /// Get the dynamically fetched context window, if set.
    pub fn get_dynamic_context_window(&self) -> Option<usize> {
        match self.dynamic_context_window.load(Ordering::Relaxed) {
            0 => None,
            v => Some(v),
        }
    }

    /// Convert internal messages to OpenAI-compatible format.
    /// Returns `Arc<Vec<...>>` so cache hits avoid cloning the message list.
    fn convert_messages(&self, messages: &[Message]) -> Arc<Vec<OpenAICompatMessage>> {
        let thinking_enabled = super::types::is_thinking_enabled_model(self.config.model());
        self.message_cache.get_or_compute_with_hash(
            messages,
            crate::llm::cache::hash_messages_for_cache_skip_system,
            |messages| {
                messages
                    .iter()
                    .map(|message| convert_openai_message(message, thinking_enabled))
                    .collect()
            },
        )
    }

    /// Convert tool schemas to OpenAI-compatible format.
    /// Returns `Arc<Vec<...>>` so cache hits avoid cloning the tool list.
    fn convert_tools(&self, tools: &[crate::messages::ToolSchema]) -> Arc<Vec<OpenAICompatTool>> {
        crate::llm::provider_utils::cached_tool_mapping(&self.tool_cache, tools, |tools| {
            tools
                .iter()
                .map(|tool| {
                    let parameters = crate::llm::provider_utils::tool_schema_to_json_value(tool);
                    OpenAICompatTool {
                        tool_type: Cow::borrowed("function"),
                        function: OpenAICompatFunctionSchema {
                            name: tool.name.clone(),
                            description: tool.description.clone(),
                            parameters,
                        },
                    }
                })
                .collect()
        })
    }

    /// Build the request with provider-specific customizations.
    fn build_request(&self, request: &CompletionRequest, stream: bool) -> OpenAICompatRequest {
        let messages = self.convert_messages(&request.messages);
        // Only include tools if the provider supports native tools
        let tools = if self.config.supports_native_tools() {
            request.tools.as_ref().map(|t| self.convert_tools(t))
        } else {
            None
        };

        let (max_tokens, max_completion_tokens) = if self.config.uses_max_completion_tokens() {
            (None, request.max_tokens)
        } else {
            (request.max_tokens, None)
        };

        let temperature = if self.config.requires_default_temperature() {
            None
        } else {
            request.temperature
        };

        let tool_choice = request
            .tool_choice
            .as_ref()
            .and_then(|tc| self.config.convert_tool_choice(tc));

        let stream_options = if stream && self.config.include_usage_in_stream() {
            Some(serde_json::json!({"include_usage": true}))
        } else {
            None
        };

        let usage_config = if stream && self.config.needs_usage_parameter() {
            Some(serde_json::json!({ "include": true }))
        } else {
            None
        };

        let response_format = if self.config.supports_response_format() {
            request
                .response_format
                .as_ref()
                .map(crate::llm::provider_utils::response_format_to_json_value)
        } else {
            None // Skip response_format for providers that don't support it
        };

        let mut openai_request = OpenAICompatRequest {
            model: self.cached_model_string.clone(),
            messages,
            max_tokens,
            max_completion_tokens,
            temperature,
            reasoning_effort: request
                .reasoning_effort
                .map(|effort| effort.as_str().to_string()),
            stream: Some(stream),
            tools,
            tool_choice,
            stream_options,
            response_format,
            usage_config,
            prompt_cache_key: Some(crate::llm::prompt_cache::openai_prompt_cache_key(
                request.messages.as_ref(),
                request.tools.as_ref().map(|tools| tools.as_slice()),
            )),
        };
        if let Some(ref key) = openai_request.prompt_cache_key {
            tracing::debug!(
                target: "caliche::prompt_cache",
                provider = self.config.provider_name(),
                prompt_cache_key = %key,
                "openai-compat prompt cache key attached"
            );
        }

        // Allow provider-specific modifications
        self.config.customize_request(&mut openai_request);

        openai_request
    }

    /// Build HTTP request with proper headers.
    /// Uses cached URL for "chat/completions" to avoid allocation on the hot path.
    /// Send a streaming request and yield its events.
    ///
    /// With a `first_token` limit, the response headers and the first byte
    /// share one deadline (no establishment retry outlives it), and the
    /// request has no whole-request timeout, so a long local reply is not
    /// cut short. Without one, the client's timeouts and the stall limit
    /// apply as before.
    fn stream_events(
        &self,
        openai_request: OpenAICompatRequest,
        first_token: Option<Duration>,
    ) -> impl Stream<Item = anyhow::Result<StreamEvent>> + Send + '_ {
        let provider_name = self.config.provider_name();
        let strategy = self.retry_strategy.as_ref();
        let stall = self.config.stall_timeout();

        async_stream::stream! {
            let body_bytes: Bytes = match serde_json::to_vec(&openai_request) {
                Ok(b) => b.into(),
                Err(e) => {
                    yield Err(anyhow::Error::new(e).context("Failed to serialize request"));
                    return;
                }
            };

            let first_byte = first_token.map(|limit| FirstByteDeadline {
                at: tokio::time::Instant::now() + limit,
                limit,
            });
            let establish = retry_stream_establishment(strategy, provider_name, || async {
                let request = self
                    .build_http_request("chat/completions")
                    .await
                    .body(body_bytes.clone());
                match first_byte {
                    Some(_) => request.timeout(LOCAL_REQUEST_CAP).send().await,
                    None => request.send().await,
                }
            });
            let established = match first_byte {
                Some(first) => tokio::time::timeout_at(first.at, establish)
                    .await
                    .unwrap_or_else(|_| Err(anyhow::anyhow!(no_first_token_error(first.limit)))),
                None => establish.await,
            };
            let response = match established {
                Ok(resp) => resp,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            // Parser and consumer share a bounded queue. A full queue waits
            // inside the parser (`OPENAI_STREAM_CHANNEL_CAPACITY`), which stops
            // the body read and applies backpressure. Chunks are not dropped.
            let (tx, mut rx) = openai_stream_channel();
            let config = self.config.clone();
            let prompt_cache_key = openai_request.prompt_cache_key.clone();
            let limits = SseLimits { first_byte, stall };
            tokio::spawn(async move {
                let task_tx = tx.clone();
                if let Err(error) = parse_sse_stream_to_channel::<C>(
                    response,
                    config,
                    provider_name,
                    prompt_cache_key,
                    limits,
                    tx,
                )
                .await
                {
                    let _ = task_tx.send(Err(error)).await;
                }
            });

            // Forward from channel to stream
            while let Some(item) = rx.recv().await {
                yield item;
            }
        }
    }

    /// How long a model load may take: the local first-token limit.
    fn model_load_limit(&self) -> Duration {
        self.config
            .first_token_timeout()
            .unwrap_or_else(crate::llm::provider_utils::first_token::local_first_token_timeout)
    }

    /// Load `model` first when the server has it but not in memory.
    async fn load_model_if_needed(&self, model: &str) -> anyhow::Result<()> {
        let Some(loader) = self.config.model_loader() else {
            return Ok(());
        };
        if loader.needs_load(&self.client, model).await {
            loader
                .load(&self.client, model, self.model_load_limit())
                .await?;
        }
        Ok(())
    }

    async fn build_http_request(&self, endpoint: &str) -> reqwest::RequestBuilder {
        let req = if endpoint == "chat/completions" {
            self.client.post(&self.chat_completions_url)
        } else if endpoint == "embeddings" {
            self.client.post(&self.embeddings_url)
        } else {
            let base = self.config.base_url().trim_end_matches('/');
            self.client.post(format!("{base}/{endpoint}"))
        };

        apply_request_headers(req, &self.config, &self.common).await
    }

    /// Extract content and tool calls from the response.
    fn extract_response_content(
        &self,
        response: &OpenAICompatResponse,
    ) -> (CompletionContent, Option<Cow<'static, str>>) {
        let Some(choice) = response.choices.first() else {
            return (
                CompletionContent::Text {
                    content: String::new(),
                },
                None,
            );
        };

        let finish_reason = choice.finish_reason.clone().map(Cow::owned);
        let content = choice
            .message
            .content
            .as_ref()
            .map(OpenAICompatContent::extract_text)
            .map(|text| self.config.post_process_output(&text).into_owned())
            .unwrap_or_default();

        let tool_calls: Vec<crate::messages::ToolCallInfo> = choice
            .message
            .tool_calls
            .iter()
            .flatten()
            .map(|call| {
                crate::messages::ToolCallInfo::from_json(
                    call.id.clone().into_owned(),
                    call.function.name.clone().into_owned(),
                    call.function.arguments.as_ref().clone(),
                )
            })
            .collect();

        if tool_calls.is_empty() {
            return (CompletionContent::Text { content }, finish_reason);
        }

        (
            CompletionContent::WithToolCalls {
                content,
                tool_calls,
            },
            finish_reason,
        )
    }

    /// Convert usage from response.
    fn convert_usage(usage: &OpenAICompatUsage) -> TokenUsage {
        openai_usage_to_token_usage(usage)
    }
}

#[async_trait]
impl<C: OpenAICompatConfig> Provider for OpenAICompatibleProvider<C> {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        self.load_model_if_needed(self.config.model()).await?;
        let openai_request = self.build_request(&request, false);

        log_openai_compat_request(self.config.provider_name(), &openai_request, false);

        let body_bytes: Bytes = serde_json::to_vec(&openai_request)
            .map_err(anyhow::Error::new)
            .context("Failed to serialize request")?
            .into();

        let provider_name = self.config.provider_name();
        let request_start = std::time::Instant::now();

        let client = Arc::clone(&self.client);
        let chat_completions_url = self.chat_completions_url.clone();
        let config = self.config.clone();
        let common = self.common.clone();

        let response = crate::llm::provider_utils::retry_llm_request(
            self.retry_strategy.as_ref(),
            provider_name,
            || {
                let client = Arc::clone(&client);
                let chat_completions_url = chat_completions_url.clone();
                let config = config.clone();
                let common = common.clone();
                let body_bytes = body_bytes.clone();
                async move {
                    apply_request_headers(client.post(&chat_completions_url), &config, &common)
                        .await
                        .body(body_bytes)
                        .send()
                        .await
                }
            },
        )
        .await?;
        let duration_ms = request_start.elapsed().as_millis();
        let _ = tracing::info_span!(
            "llm_http_request",
            provider = %provider_name,
            duration_ms = duration_ms
        )
        .entered();

        let body = response
            .text()
            .await
            .map_err(anyhow::Error::new)
            .with_context(|| format!("Failed to read {provider_name} response body"))?;
        let openai_response: OpenAICompatResponse = serde_json::from_str(&body).map_err(|e| {
            let preview = body.chars().take(2000).collect::<String>();
            anyhow::anyhow!(
                "Failed to parse {provider_name} response: {e} | response preview: {preview}"
            )
        })?;

        tracing::debug!("{} response: {:?}", provider_name, openai_response);

        let (mut content, finish_reason) = self.extract_response_content(&openai_response);

        // Append provider-specific citations (e.g. Perplexity Sonar) as footnotes.
        if !openai_response.citations.is_empty() {
            let text_field = match &mut content {
                CompletionContent::Text { content }
                | CompletionContent::WithToolCalls { content, .. }
                | CompletionContent::Multimodal { content, .. } => content,
            };
            text_field.push_str("\n\nSources:\n");
            for (i, url) in openai_response.citations.iter().enumerate() {
                let _ = writeln!(text_field, "[{}] {url}", i + 1);
            }
        }

        let usage = openai_response.usage.as_ref().map(Self::convert_usage);

        crate::llm::provider_utils::log_prompt_cache_usage(
            usage.as_ref(),
            provider_name,
            openai_request.prompt_cache_key.as_deref(),
        );

        let reasoning_content = openai_response
            .choices
            .first()
            .and_then(|choice| choice.message.reasoning_content.as_ref())
            .map(|text| text.as_str().to_string())
            .filter(|text| !text.trim().is_empty());

        Ok(CompletionResponse {
            content,
            finish_reason,
            usage,
            reasoning_content,
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        let openai_request = self.build_request(&request, true);
        let provider_name = self.config.provider_name();

        log_openai_compat_request(provider_name, &openai_request, true);

        let first_token = self.config.first_token_timeout();
        let events = self.stream_events(openai_request, first_token);
        let events: Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> =
            match (first_token, self.config.model_load_probe()) {
                (Some(_), Some(probe)) => {
                    let client = Arc::clone(&self.client);
                    Box::pin(with_waiting_status(events, async move {
                        probe.status(&client).await
                    }))
                }
                (Some(_), None) => Box::pin(with_waiting_status(events, async {
                    WAITING_FOR_FIRST_TOKEN.to_string()
                })),
                (None, _) => Box::pin(events),
            };
        let Some(loader) = self.config.model_loader() else {
            return events;
        };
        let client = Arc::clone(&self.client);
        let model = self.config.model().to_string();
        let limit = self.model_load_limit();
        Box::pin(async_stream::stream! {
            if loader.needs_load(&client, &model).await {
                yield Ok(StreamEvent::Status { message: loading_status(&model, "LM Studio") });
                if let Err(error) = loader.load(&client, &model, limit).await {
                    yield Err(error);
                    return;
                }
            }
            let mut events = events;
            while let Some(item) = futures_util::StreamExt::next(&mut events).await {
                yield item;
            }
        })
    }

    fn name(&self) -> &str {
        self.config.provider_name()
    }

    fn supports_native_tools(&self) -> bool {
        self.config.supports_native_tools()
    }

    fn supports_text_based_tools(&self) -> bool {
        self.config.supports_text_based_tools()
    }

    fn supports_vision(&self) -> bool {
        self.config.supports_vision()
    }

    fn supports_audio_input(&self) -> bool {
        self.config.supports_audio_input()
    }

    fn supports_document_input(&self) -> bool {
        self.config.supports_document_input()
    }

    fn supported_media_types(&self) -> Vec<&str> {
        self.config.supported_media_types()
    }

    fn context_window(&self) -> Option<usize> {
        self.get_dynamic_context_window()
    }

    fn model_load_probe(&self) -> Option<crate::llm::provider_utils::first_token::ModelLoadProbe> {
        self.config.model_load_probe()
    }
}

#[async_trait]
impl<C: OpenAICompatConfig> EmbeddingProvider for OpenAICompatibleProvider<C> {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        if !self.config.supports_embeddings() {
            return Err(anyhow::anyhow!(
                "{} does not support embeddings",
                self.config.provider_name()
            ));
        }

        let model = request
            .model
            .or_else(|| self.config.default_embedding_model())
            .ok_or_else(|| {
                anyhow::anyhow!("No embedding model specified and no default available")
            })?;

        self.load_model_if_needed(&model).await?;
        let embedding_request = OpenAICompatEmbeddingRequest {
            input: request.texts,
            model,
        };

        let provider_name = self.config.provider_name();
        tracing::debug!(
            "{} embedding request: {:?}",
            provider_name,
            embedding_request
        );

        let http_req = self.build_http_request("embeddings").await;
        let response = http_req
            .json(&embedding_request)
            .send()
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to send embedding request to {provider_name}: {e}")
            })?;

        let response =
            crate::llm::provider_utils::check_response_status(response, provider_name).await?;

        let embedding_response: OpenAICompatEmbeddingResponse =
            response.json().await.map_err(|e| {
                anyhow::anyhow!("Failed to parse {provider_name} embedding response: {e}")
            })?;

        tracing::debug!(
            "{} embedding response received with {} embeddings",
            provider_name,
            embedding_response.data.len()
        );

        // Sort by index to ensure correct order
        let mut data = embedding_response.data;
        data.sort_unstable_by_key(|d| d.index);

        Ok(EmbeddingResponse {
            embeddings: data.into_iter().map(|d| d.embedding).collect(),
            usage: Some(TokenUsage::from_counts(
                Some(embedding_response.usage.prompt_tokens),
                embedding_response.usage.completion_tokens,
                Some(embedding_response.usage.total_tokens),
            )),
            model: embedding_response.model,
        })
    }

    fn name(&self) -> &str {
        self.config.provider_name()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::ToolCallInfo;

    #[test]
    fn thinking_tool_call_replay_does_not_duplicate_text_into_reasoning() {
        let message = Message {
            role: MessageRole::Assistant,
            content: MessageContent::Structured {
                text: Some(Arc::new("call the tool".into())),
                tool_calls: Some(vec![ToolCallInfo::from_json(
                    "c1",
                    "bash",
                    r#"{"command":"echo hi"}"#,
                )]),
                tool_call_id: None,
            },
            metadata: None,
        };
        let converted = convert_openai_assistant(&message, true);
        assert_eq!(
            converted
                .content
                .as_ref()
                .map(OpenAICompatContent::extract_text)
                .as_deref(),
            Some("call the tool")
        );
        assert_eq!(
            converted.reasoning_content.as_deref().map(String::as_str),
            Some("")
        );
        assert!(converted.tool_calls.is_some());
    }

    #[test]
    fn thinking_tool_call_replay_uses_stored_reasoning_metadata() {
        let message = Message {
            role: MessageRole::Assistant,
            content: MessageContent::Structured {
                text: Some(Arc::new("visible".into())),
                tool_calls: Some(vec![ToolCallInfo::from_json("c1", "bash", "{}")]),
                tool_call_id: None,
            },
            metadata: None,
        }
        .with_reasoning_content("internal chain of thought");
        let converted = convert_openai_assistant(&message, true);
        assert_eq!(
            converted
                .content
                .as_ref()
                .map(OpenAICompatContent::extract_text)
                .as_deref(),
            Some("visible")
        );
        assert_eq!(
            converted.reasoning_content.as_deref().map(String::as_str),
            Some("internal chain of thought")
        );
    }

    #[test]
    fn non_thinking_tool_call_omits_reasoning_unless_metadata_present() {
        let message = Message {
            role: MessageRole::Assistant,
            content: MessageContent::Structured {
                text: Some(Arc::new("visible".into())),
                tool_calls: Some(vec![ToolCallInfo::from_json("c1", "bash", "{}")]),
                tool_call_id: None,
            },
            metadata: None,
        };
        let converted = convert_openai_assistant(&message, false);
        assert!(converted.reasoning_content.is_none());
    }

    #[test]
    fn unstripped_reasoning_is_resent_and_trim_drops_one_char() {
        let blob = format!("{}\n", "c".repeat(25_499));
        assert_eq!(blob.chars().count(), 25_500);
        let message = Message::assistant_with_reasoning("short visible answer text here!!", &blob);
        let converted = convert_openai_assistant(&message, false);
        assert_eq!(
            converted.reasoning_content.as_ref().map(|text| text.len()),
            Some(25_499)
        );
        let stripped = message.without_reasoning_content();
        let converted = convert_openai_assistant(&stripped, false);
        assert!(converted.reasoning_content.is_none());
    }

    #[test]
    fn request_summary_counts_message_chars_without_serializing_body() {
        let message = OpenAICompatMessage {
            role: Cow::borrowed("user"),
            content: Some(OpenAICompatContent::String(Arc::new("hello".repeat(10)))),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: Some(Arc::new("why".to_string())),
        };
        let request = OpenAICompatRequest {
            model: "test-model".to_string(),
            messages: Arc::new(vec![message]),
            max_tokens: Some(100),
            max_completion_tokens: None,
            temperature: Some(0.7),
            reasoning_effort: None,
            stream: Some(true),
            tools: None,
            tool_choice: None,
            stream_options: None,
            response_format: None,
            usage_config: None,
            prompt_cache_key: None,
        };

        let message_chars: usize = request
            .messages
            .iter()
            .map(openai_compat_message_chars)
            .sum();

        assert_eq!(message_chars, 53);
    }

    fn test_provider() -> OpenAICompatibleProvider<crate::llm::openai_compat::OllamaConfig> {
        OpenAICompatibleProvider::with_defaults(crate::llm::openai_compat::OllamaConfig::new(
            "http://localhost:11434/v1",
            "test-model",
        ))
        .expect("build test provider")
    }

    fn parse_response(body: &str) -> OpenAICompatResponse {
        serde_json::from_str(body).expect("parse response fixture")
    }

    #[test]
    fn extract_response_content_returns_empty_text_without_choices() {
        let response = parse_response(
            r#"{"id":"r","object":"chat.completion","created":0,"model":"m","choices":[],"usage":null}"#,
        );
        let (content, finish_reason) = test_provider().extract_response_content(&response);
        assert!(finish_reason.is_none());
        assert!(matches!(content, CompletionContent::Text { content } if content.is_empty()));
    }

    #[test]
    fn extract_response_content_reads_text_and_finish_reason() {
        let response = parse_response(
            r#"{"id":"r","object":"chat.completion","created":0,"model":"m","usage":null,
                "choices":[{"index":0,"finish_reason":"stop",
                "message":{"role":"assistant","content":"hello"}}]}"#,
        );
        let (content, finish_reason) = test_provider().extract_response_content(&response);
        assert_eq!(finish_reason.as_deref(), Some("stop"));
        assert!(matches!(content, CompletionContent::Text { content } if content == "hello"));
    }

    #[test]
    fn extract_response_content_treats_empty_tool_calls_as_text() {
        let response = parse_response(
            r#"{"id":"r","object":"chat.completion","created":0,"model":"m","usage":null,
                "choices":[{"index":0,"finish_reason":"stop",
                "message":{"role":"assistant","content":"hello","tool_calls":[]}}]}"#,
        );
        let (content, _) = test_provider().extract_response_content(&response);
        assert!(matches!(content, CompletionContent::Text { content } if content == "hello"));
    }

    #[test]
    fn extract_response_content_keeps_text_alongside_tool_calls() {
        let response = parse_response(
            r#"{"id":"r","object":"chat.completion","created":0,"model":"m","usage":null,
                "choices":[{"index":0,"finish_reason":"tool_calls",
                "message":{"role":"assistant","content":"running",
                "tool_calls":[{"id":"c1","type":"function",
                "function":{"name":"bash","arguments":"{\"command\":\"ls\"}"}}]}}]}"#,
        );
        let (content, finish_reason) = test_provider().extract_response_content(&response);
        assert_eq!(finish_reason.as_deref(), Some("tool_calls"));
        let CompletionContent::WithToolCalls {
            content,
            tool_calls,
        } = content
        else {
            panic!("expected tool calls");
        };
        assert_eq!(content, "running");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].name, "bash");
    }

    #[test]
    fn build_request_sets_prompt_cache_key_from_system_prefix() {
        let provider = test_provider();
        let request = CompletionRequest::new(Arc::new(vec![
            crate::messages::Message::system("stable"),
            crate::messages::Message::user("hello"),
        ]));
        let body = provider.build_request(&request, true);
        let key = body.prompt_cache_key.expect("prompt cache key");
        assert!(key.starts_with("caliche-"));
        let again = provider.build_request(
            &CompletionRequest::new(Arc::new(vec![
                crate::messages::Message::system("stable"),
                crate::messages::Message::user("later"),
            ])),
            false,
        );
        assert_eq!(again.prompt_cache_key.as_deref(), Some(key.as_str()));
    }
}
