use crate::llm::cache::ConversionCache;
use crate::llm::config::CommonProviderOptions;
use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::llm::models::{BedrockModel, ModelProvider, ProviderType};
use crate::llm::provider::Provider;
use crate::messages::{CompletionRequest, CompletionResponse, Message, MessageRole, StreamEvent};
use async_trait::async_trait;
#[cfg(feature = "bedrock")]
use aws_config::{BehaviorVersion, Region};
#[cfg(feature = "bedrock")]
use aws_credential_types::Credentials;
#[cfg(feature = "bedrock")]
use aws_sdk_bedrock::Client as BedrockControlClient;
#[cfg(feature = "bedrock")]
use aws_sdk_bedrockruntime::Client as BedrockClient;
#[cfg(feature = "bedrock")]
use aws_sdk_bedrockruntime::types::ResponseStream;
use beef::lean::Cow;
use futures_util::{Stream, StreamExt, stream};
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;

mod errors;
use errors::HeaderError;

fn estimated_tokens_from_chars(chars: usize) -> u32 {
    u32::try_from(chars.div_ceil(4)).unwrap_or(u32::MAX)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BedrockRequest {
    #[serde(
        serialize_with = "crate::llm::provider_utils::serde_helpers::serialize_cow_static_str",
        deserialize_with = "crate::llm::provider_utils::serde_helpers::deserialize_cow_static_str"
    )]
    anthropic_version: Cow<'static, str>,
    max_tokens: u32,
    messages: Vec<BedrockMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<BedrockTool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BedrockMessage {
    #[serde(
        serialize_with = "crate::llm::provider_utils::serde_helpers::serialize_cow_static_str",
        deserialize_with = "crate::llm::provider_utils::serde_helpers::deserialize_cow_static_str"
    )]
    role: Cow<'static, str>,
    content: Vec<BedrockContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BedrockContent {
    Text {
        #[serde(
            serialize_with = "crate::llm::provider_utils::serde_helpers::serialize_arc_str",
            deserialize_with = "crate::llm::provider_utils::serde_helpers::deserialize_arc_str"
        )]
        text: std::sync::Arc<String>,
    },
    Image {
        source: BedrockImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(
            serialize_with = "crate::llm::provider_utils::serde_helpers::serialize_arc_str",
            deserialize_with = "crate::llm::provider_utils::serde_helpers::deserialize_arc_str"
        )]
        content: std::sync::Arc<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BedrockImageSource {
    #[serde(
        rename = "type",
        serialize_with = "crate::llm::provider_utils::serde_helpers::serialize_cow_static_str",
        deserialize_with = "crate::llm::provider_utils::serde_helpers::deserialize_cow_static_str"
    )]
    source_type: Cow<'static, str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    media_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BedrockTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BedrockResponse {
    content: Vec<BedrockContent>,
    #[serde(alias = "stop_reason", alias = "stopReason")]
    stop_reason: Option<String>,
    #[serde(alias = "usage", alias = "Usage")]
    usage: Option<BedrockUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BedrockUsage {
    #[serde(alias = "input_tokens", alias = "inputTokens")]
    input_tokens: Option<u32>,
    #[serde(alias = "output_tokens", alias = "outputTokens")]
    output_tokens: Option<u32>,
}

/// Bedrock credentials configuration
#[derive(Debug, Clone)]
pub enum BedrockCredentials {
    /// AWS credentials (access key, secret key, optional session token)
    Aws {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
    /// Bearer token authentication
    Bearer(String),
}

/// Configuration for `BedrockProvider`
#[derive(Debug, Clone)]
pub struct BedrockProviderConfig {
    /// Model identifier
    pub model_id: String,
    /// AWS region
    pub region: Option<String>,
    /// Credentials (if None, uses AWS default credential chain)
    pub credentials: Option<BedrockCredentials>,
    /// Whether embeddings are enabled
    pub embeddings_enabled: bool,
    /// Embedding model ID (if embeddings are enabled)
    pub embedding_model_id: Option<String>,
    /// Common provider options
    pub common: CommonProviderOptions,
}

impl BedrockProviderConfig {
    /// Create a new `BedrockProviderConfig`.
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            region: None,
            credentials: None,
            embeddings_enabled: false,
            embedding_model_id: None,
            common: CommonProviderOptions::default(),
        }
    }

    /// Create a builder for `BedrockProviderConfig`
    #[must_use]
    pub fn builder() -> BedrockProviderConfigBuilder {
        BedrockProviderConfigBuilder::default()
    }
}

crate::impl_provider_config!(BedrockProviderConfig);

/// Builder for `BedrockProviderConfig`
#[derive(Debug, Default)]
pub struct BedrockProviderConfigBuilder {
    model_id: Option<String>,
    region: Option<String>,
    credentials: Option<BedrockCredentials>,
    embedding_model_id: Option<String>,
    common: CommonProviderOptions,
}

impl BedrockProviderConfigBuilder {
    /// Set the model ID
    #[must_use]
    pub fn model_id(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Set the AWS region
    #[must_use]
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.region = Some(region.into());
        self
    }

    /// Set AWS credentials
    #[must_use]
    pub fn aws_credentials(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
    ) -> Self {
        self.credentials = Some(BedrockCredentials::Aws {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
        });
        self
    }

    /// Set AWS credentials with session token
    #[must_use]
    pub fn aws_credentials_with_session(
        mut self,
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: impl Into<String>,
    ) -> Self {
        self.credentials = Some(BedrockCredentials::Aws {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: Some(session_token.into()),
        });
        self
    }

    /// Set bearer token
    #[must_use]
    pub fn bearer_token(mut self, token: impl Into<String>) -> Self {
        self.credentials = Some(BedrockCredentials::Bearer(token.into()));
        self
    }

    /// Set embedding model ID
    #[must_use]
    pub fn embedding_model_id(mut self, model_id: impl Into<String>) -> Self {
        self.embedding_model_id = Some(model_id.into());
        self
    }

    /// Build the configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn build(self) -> anyhow::Result<BedrockProviderConfig> {
        let model_id = self.model_id.unwrap_or_else(|| {
            caliche_contracts::credential_env::var("BEDROCK_MODEL_ID")
                .unwrap_or_else(|_| "anthropic.claude-3-sonnet-20240229-v1:0".to_string())
        });

        // Embeddings are enabled when an embedding model ID is set (programmatically or via env).
        let embedding_model_id = self
            .embedding_model_id
            .or_else(|| caliche_contracts::credential_env::var("BEDROCK_EMBEDDING_MODEL_ID").ok());
        let embeddings_enabled = embedding_model_id.is_some();

        Ok(BedrockProviderConfig {
            model_id,
            region: self.region,
            credentials: self.credentials,
            embeddings_enabled,
            embedding_model_id,
            common: self.common,
        })
    }
}

crate::impl_common_builder_methods!(BedrockProviderConfigBuilder);

pub struct BedrockProvider {
    client: BedrockClient,
    model_id: String,
    bearer_token: Option<String>,
    retry_strategy: Arc<dyn crate::traits::retry::RetryStrategy>,
    tool_cache: ConversionCache<Vec<BedrockTool>>,
    message_cache: ConversionCache<(Option<String>, Vec<BedrockMessage>)>,
}

impl std::fmt::Debug for BedrockProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockProvider")
            .field("client", &"BedrockClient")
            .field("model_id", &self.model_id)
            .field("bearer_token", &self.bearer_token.as_ref().map(|_| "***"))
            .field("retry_strategy", &self.retry_strategy)
            .field("tool_cache", &self.tool_cache)
            .field("message_cache", &self.message_cache)
            .finish()
    }
}

impl BedrockProvider {
    pub fn new(client: BedrockClient, model_id: impl Into<String>) -> Self {
        Self {
            client,
            model_id: model_id.into(),
            bearer_token: None,
            retry_strategy: Arc::new(crate::traits::retry::ExponentialBackoffStrategy::new(
                5, 1000, 60_000, 2.0, true,
            )),
            tool_cache: ConversionCache::new(),
            message_cache: ConversionCache::new(),
        }
    }

    #[must_use]
    pub fn with_bearer_token(mut self, bearer_token: impl Into<String>) -> Self {
        self.bearer_token = Some(bearer_token.into());
        self
    }

    /// Classify Bedrock errors for retry logic
    fn classify_bedrock_error(error_debug: &str) -> crate::llm::retry::RetryError {
        // Check for throttling errors
        if error_debug.contains("ThrottlingException")
            || error_debug.contains("TooManyRequestsException")
        {
            return crate::llm::retry::RetryError::RateLimited {
                retry_after: None, // AWS doesn't provide retry-after hint
                message: format!("Bedrock API throttled: {error_debug}"),
            };
        }

        // Check for transient errors
        if error_debug.contains("ServiceUnavailableException")
            || error_debug.contains("InternalServerException")
            || error_debug.contains("RequestTimeoutException")
        {
            return crate::llm::retry::RetryError::Transient {
                status_code: 503,
                message: format!("Bedrock API transient error: {error_debug}"),
            };
        }

        // Permanent errors (auth, validation, etc.)
        let error_code = if error_debug.contains("UnrecognizedClientException") {
            "UnrecognizedClientException"
        } else if error_debug.contains("ValidationException") {
            "ValidationException"
        } else if error_debug.contains("AccessDeniedException") {
            "AccessDeniedException"
        } else {
            "Unknown"
        };

        crate::llm::retry::RetryError::Permanent {
            error: format!("Bedrock API error ({error_code}): {error_debug}"),
        }
    }

    /// Create a `BedrockProvider` with configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn with_config(config: BedrockProviderConfig) -> anyhow::Result<Self> {
        let bearer_token = match &config.credentials {
            Some(BedrockCredentials::Bearer(token)) => Some(token.clone()),
            _ => caliche_contracts::credential_env::var("AWS_BEARER_TOKEN_BEDROCK").ok(),
        };

        let mut config_builder = aws_config::defaults(BehaviorVersion::latest());

        if bearer_token.is_none() {
            // Try credentials from config first, then fall back to env
            match &config.credentials {
                Some(BedrockCredentials::Aws {
                    access_key_id,
                    secret_access_key,
                    session_token,
                }) => {
                    let credentials = Credentials::new(
                        access_key_id.as_str(),
                        secret_access_key.as_str(),
                        session_token.clone(),
                        None,
                        "provided",
                    );
                    config_builder = config_builder.credentials_provider(credentials);
                }
                _ => {
                    // Fall back to environment variables
                    if let (Ok(access_key_id), Ok(secret_access_key)) = (
                        caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID"),
                        caliche_contracts::credential_env::var("AWS_SECRET_ACCESS_KEY"),
                    ) {
                        let credentials = Credentials::new(
                            &access_key_id,
                            &secret_access_key,
                            caliche_contracts::credential_env::var("AWS_SESSION_TOKEN").ok(),
                            None,
                            "env",
                        );
                        config_builder = config_builder.credentials_provider(credentials);
                    }
                }
            }
        }

        // Try region from config first, then fall back to env
        if let Some(region_str) = config.region.as_ref() {
            config_builder = config_builder.region(Some(Region::new(region_str.clone())));
        } else if let Ok(region_str) = caliche_contracts::credential_env::var("AWS_REGION") {
            config_builder = config_builder.region(Some(Region::new(region_str)));
        } else if let Ok(region_str) = caliche_contracts::credential_env::var("AWS_DEFAULT_REGION")
        {
            config_builder = config_builder.region(Some(Region::new(region_str)));
        }

        let aws_config = config_builder.load().await;
        let client = BedrockClient::new(&aws_config);

        let retry_strategy = crate::llm::provider_utils::create_retry_strategy(&config.common);

        let mut provider = Self {
            client,
            model_id: config.model_id,
            bearer_token: None,
            retry_strategy,
            tool_cache: ConversionCache::new(),
            message_cache: ConversionCache::new(),
        };

        if let Some(token) = bearer_token {
            provider = provider.with_bearer_token(token);
        }

        Ok(provider)
    }

    /// Create a `BedrockProvider` from environment configuration.
    ///
    /// Model, credentials, and region resolve from `BEDROCK_MODEL_ID`, the AWS
    /// credential environment variables, and `AWS_REGION` / `AWS_DEFAULT_REGION`.
    ///
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model_id: Option<String>) -> anyhow::Result<Self> {
        let mut builder = BedrockProviderConfig::builder();
        if let Some(model_id) = model_id {
            builder = builder.model_id(model_id);
        }
        Self::with_config(builder.build()?).await
    }

    fn convert_messages(&self, messages: &[Message]) -> Arc<(Option<String>, Vec<BedrockMessage>)> {
        self.message_cache.get_or_compute_with_hash(
            messages,
            crate::llm::cache::hash_messages_for_cache,
            convert_bedrock_messages,
        )
    }

    fn convert_tools(&self, tools: &[crate::messages::ToolSchema]) -> Arc<Vec<BedrockTool>> {
        crate::llm::provider_utils::cached_tool_mapping(&self.tool_cache, tools, |tools| {
            tools
                .iter()
                .map(|tool| {
                    let input_schema = crate::llm::provider_utils::tool_schema_to_json_value(tool);
                    BedrockTool {
                        name: tool.name.as_ref().to_string(),
                        description: tool.description.as_ref().to_string(),
                        input_schema,
                    }
                })
                .collect()
        })
    }

    fn create_request(&self, request: &CompletionRequest) -> BedrockRequest {
        let (system, messages) = (*self.convert_messages(&request.messages)).clone();
        BedrockRequest {
            anthropic_version: Cow::borrowed("bedrock-2023-05-31"),
            max_tokens: request.max_tokens.unwrap_or(4096),
            messages,
            system,
            temperature: Some(request.temperature.unwrap_or(0.7)),
            tools: request
                .tools
                .as_ref()
                .map(|tools| (*self.convert_tools(tools)).clone()),
        }
    }
}

fn convert_bedrock_messages(messages: &[Message]) -> (Option<String>, Vec<BedrockMessage>) {
    let mut system = None;
    let mut converted = Vec::with_capacity(messages.len());
    for message in messages {
        match message.role {
            MessageRole::System => {
                let text = crate::llm::provider_utils::extract_message_text(message);
                system = Some(system.map_or_else(
                    || text.clone(),
                    |existing: String| {
                        let mut combined = String::with_capacity(existing.len() + text.len() + 1);
                        combined.push_str(&existing);
                        combined.push('\n');
                        combined.push_str(&text);
                        combined
                    },
                ));
            }
            MessageRole::User => {
                let content = bedrock_user_content(message);
                if !content.is_empty() {
                    converted.push(BedrockMessage {
                        role: Cow::borrowed("user"),
                        content,
                    });
                }
            }
            MessageRole::Assistant => {
                let content = bedrock_assistant_content(message);
                if !content.is_empty() {
                    converted.push(BedrockMessage {
                        role: Cow::borrowed("assistant"),
                        content,
                    });
                }
            }
            MessageRole::Tool => converted.push(bedrock_tool_result(message)),
        }
    }
    (system, converted)
}

fn bedrock_user_content(message: &Message) -> Vec<BedrockContent> {
    let crate::messages::MessageContent::Multimodal(parts) = &message.content else {
        return vec![BedrockContent::Text {
            text: Arc::new(crate::llm::provider_utils::extract_message_text(message)),
        }];
    };
    parts
        .iter()
        .filter_map(|part| match part {
            crate::messages::ContentPart::Text { text } if !text.is_empty() => {
                Some(BedrockContent::Text {
                    text: Arc::new(text.to_string()),
                })
            }
            crate::messages::ContentPart::Image { data, media_type } => {
                Some(BedrockContent::Image {
                    source: BedrockImageSource {
                        source_type: Cow::borrowed("base64"),
                        media_type: Some(media_type.to_string()),
                        data: Some(data.to_string()),
                        url: None,
                    },
                })
            }
            crate::messages::ContentPart::ImageUrl { url } => Some(BedrockContent::Image {
                source: BedrockImageSource {
                    source_type: Cow::borrowed("url"),
                    media_type: None,
                    data: None,
                    url: Some(url.to_string()),
                },
            }),
            crate::messages::ContentPart::Document {
                data,
                media_type,
                filename,
            } => Some(BedrockContent::Text {
                text: Arc::new(format!(
                    "[Document attachment{} received: {} ({} bytes base64)]",
                    filename
                        .as_ref()
                        .map(|name| format!(" ({name})"))
                        .unwrap_or_default(),
                    media_type,
                    data.len()
                )),
            }),
            crate::messages::ContentPart::DocumentUrl { url, filename } => {
                let title = filename
                    .as_ref()
                    .map(|name| format!(" ({name})"))
                    .unwrap_or_default();
                Some(BedrockContent::Text {
                    text: Arc::new(format!("[Document URL{title}: {url}]")),
                })
            }
            crate::messages::ContentPart::Video { .. }
            | crate::messages::ContentPart::VideoUrl { .. }
            | crate::messages::ContentPart::Audio { .. }
            | crate::messages::ContentPart::AudioUrl { .. }
            | crate::messages::ContentPart::Text { .. } => None,
        })
        .collect()
}

fn bedrock_assistant_content(message: &Message) -> Vec<BedrockContent> {
    use crate::messages::MessageContent;
    let mut blocks = Vec::new();
    match &message.content {
        MessageContent::Text(text) if !text.is_empty() => blocks.push(BedrockContent::Text {
            text: Arc::clone(text),
        }),
        MessageContent::Structured {
            text, tool_calls, ..
        } => {
            if let Some(text) = text
                && !text.is_empty()
            {
                blocks.push(BedrockContent::Text {
                    text: Arc::clone(text),
                });
            }
            blocks.extend(
                tool_calls
                    .iter()
                    .flatten()
                    .map(|call| BedrockContent::ToolUse {
                        id: call.id.to_string(),
                        name: call.name.to_string(),
                        input: call.arguments.clone(),
                    }),
            );
        }
        MessageContent::Multimodal(_) => {
            let text = message.content.as_text().unwrap_or("");
            if !text.is_empty() {
                blocks.push(BedrockContent::Text {
                    text: Arc::new(text.to_string()),
                });
            }
        }
        MessageContent::Text(_) => {}
    }
    blocks
}

fn bedrock_tool_result(message: &Message) -> BedrockMessage {
    let tool_use_id = match &message.content {
        crate::messages::MessageContent::Structured {
            tool_call_id: Some(id),
            ..
        } => id.clone().into_owned(),
        _ => "unknown".to_string(),
    };
    BedrockMessage {
        role: Cow::borrowed("user"),
        content: vec![BedrockContent::ToolResult {
            tool_use_id,
            content: Arc::new(crate::llm::provider_utils::extract_message_text(message)),
        }],
    }
}

fn bedrock_completion_content(response: &BedrockResponse) -> crate::messages::CompletionContent {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in &response.content {
        match block {
            BedrockContent::Text { text: block_text } => {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(block_text.as_str());
            }
            BedrockContent::ToolUse { id, name, input } => {
                let arguments = serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string());
                tool_calls.push(crate::messages::ToolCallInfo::from_json(
                    id.clone(),
                    name.clone(),
                    arguments,
                ));
            }
            BedrockContent::Image { .. } | BedrockContent::ToolResult { .. } => {}
        }
    }
    if tool_calls.is_empty() {
        crate::messages::CompletionContent::Text { content: text }
    } else {
        crate::messages::CompletionContent::WithToolCalls {
            content: text,
            tool_calls,
        }
    }
}

fn apply_bedrock_stream_event(
    parser: &mut crate::llm::provider_utils::AnthropicStreamParser,
    event: BedrockStreamEvent,
    input_tokens: &mut u32,
    output_tokens: &mut u32,
) -> Vec<StreamEvent> {
    match event {
        BedrockStreamEvent::MessageStart { message } => {
            if let Some(tokens) = message.usage.and_then(|usage| usage.input_tokens) {
                *input_tokens = tokens;
            }
            Vec::new()
        }
        BedrockStreamEvent::ContentBlockDelta { delta, index, .. } => parser.feed(
            crate::llm::provider_utils::AnthropicStreamFeed::ContentBlockDelta {
                index,
                text: delta.text.as_deref(),
                thinking: delta.thinking.as_deref(),
                partial_json: delta.partial_json.as_deref(),
            },
        ),
        BedrockStreamEvent::MessageDelta {
            usage, stop_reason, ..
        } => {
            if let Some(tokens) = usage.and_then(|value| value.output_tokens) {
                *output_tokens = tokens;
            }
            if let Some(reason) = stop_reason {
                tracing::debug!("Bedrock stop reason: {reason}");
            }
            Vec::new()
        }
        BedrockStreamEvent::MessageStop { stop_reason, usage } => {
            if let Some(usage) = usage {
                if let Some(tokens) = usage.input_tokens {
                    *input_tokens = tokens;
                }
                if let Some(tokens) = usage.output_tokens {
                    *output_tokens = tokens;
                }
            }
            if let Some(reason) = stop_reason {
                tracing::debug!("Bedrock stream stopped: {reason}");
            }
            Vec::new()
        }
        BedrockStreamEvent::ContentBlockStart {
            content_block,
            index,
            ..
        } => content_block.as_ref().map_or_else(Vec::new, |block| {
            parser.feed(
                crate::llm::provider_utils::AnthropicStreamFeed::ContentBlockStart {
                    index,
                    block_type: &block.block_type,
                    tool_id: block.id.as_deref(),
                    tool_name: block.name.as_deref(),
                },
            )
        }),
        BedrockStreamEvent::ContentBlockStop { index, .. } => {
            parser.feed(crate::llm::provider_utils::AnthropicStreamFeed::ContentBlockStop { index })
        }
        BedrockStreamEvent::Unknown => Vec::new(),
    }
}

#[async_trait]
impl Provider for BedrockProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let bedrock_request = self.create_request(&request);

        let body = serde_json::to_string(&bedrock_request)?;

        tracing::debug!("Bedrock request: {}", body);
        tracing::debug!("Model ID: {}", self.model_id);

        // Use retry logic for Bedrock calls
        let client = &self.client;
        let model_id = &self.model_id;
        // Clone bearer_token once before the retry loop instead of inside it
        let bearer_token_opt = self.bearer_token.as_deref();
        let body_bytes = body.as_bytes().to_vec();

        let strategy = self.retry_strategy.as_ref();
        let response = crate::llm::retry::retry_with_classifier(
            strategy,
            |e: &aws_sdk_bedrockruntime::error::SdkError<
                aws_sdk_bedrockruntime::operation::invoke_model::InvokeModelError,
                _,
            >| { Self::classify_bedrock_error(&format!("{e:?}")) },
            || async {
                let invoke_request_builder = client
                    .invoke_model()
                    .model_id(model_id)
                    .content_type("application/json")
                    .body(body_bytes.clone().into());

                if let Some(token) = bearer_token_opt {
                    let token = token.to_string();
                    invoke_request_builder
                        .customize()
                        .map_request(move |mut req| -> Result<_, HeaderError> {
                            req.headers_mut()
                                .insert("authorization", errors::bearer_header_value(&token)?);
                            Ok(req)
                        })
                        .send()
                        .await
                } else {
                    invoke_request_builder.send().await
                }
            },
        )
        .await?;

        let response_body = response.body();
        let response_str = String::from_utf8(response_body.as_ref().to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to decode response body: {e}"))?;

        tracing::debug!("Bedrock response body: {}", response_str);

        if response_str.is_empty() {
            return Err(anyhow::anyhow!("Empty response body from Bedrock"));
        }

        let bedrock_response: BedrockResponse =
            serde_json::from_str(&response_str).map_err(|e| {
                anyhow::anyhow!(
                    "Failed to parse Bedrock response: {e}. Response body: {response_str}"
                )
            })?;

        tracing::debug!("Parsed Bedrock response: {:?}", bedrock_response);

        let content = bedrock_completion_content(&bedrock_response);

        let usage = bedrock_response.usage.map(|u| {
            crate::messages::TokenUsage::from_counts(
                u.input_tokens,
                u.output_tokens,
                u.input_tokens
                    .and_then(|input| u.output_tokens.map(|output| input + output)),
            )
        });

        crate::llm::provider_utils::log_token_usage(usage.as_ref(), "Bedrock");

        Ok(CompletionResponse {
            content,
            finish_reason: bedrock_response.stop_reason.map(beef::lean::Cow::owned),
            usage,
            reasoning_content: None,
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        let bedrock_request = self.create_request(&request);
        let model_id = self.model_id.clone();
        // Store bearer_token as Option<&str> to avoid cloning until needed
        let bearer_token_opt = self.bearer_token.as_deref();
        let client = self.client.clone();

        Box::pin(async_stream::stream! {
            let body = match serde_json::to_string(&bedrock_request) {
                Ok(b) => b,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Failed to serialize Bedrock request: {e}"));
                    yield Ok(StreamEvent::Done);
                    return;
                }
            };

            tracing::debug!("Bedrock streaming request: {}", body);

            let invoke_request_builder = client
                .invoke_model_with_response_stream()
                .model_id(&model_id)
                .content_type("application/json")
                .body(body.as_bytes().into());

            let response_result = if let Some(token) = bearer_token_opt {
                let token = token.to_string();
                invoke_request_builder
                    .customize()
                    .map_request(move |mut req| -> Result<_, HeaderError> {
                        req.headers_mut().insert(
                            "authorization",
                            errors::bearer_header_value(&token)?,
                        );
                        Ok(req)
                    })
                    .send()
                    .await
            } else {
                invoke_request_builder.send().await
            };

            let mut event_stream = match response_result {
                Ok(response) => response.body,
                Err(e) => {
                    let error_debug = format!("{e:?}");
                    let error_code = if error_debug.contains("UnrecognizedClientException") {
                        "UnrecognizedClientException"
                    } else if error_debug.contains("ValidationException") {
                        "ValidationException"
                    } else if error_debug.contains("AccessDeniedException") {
                        "AccessDeniedException"
                    } else {
                        "Unknown"
                    };
                    yield Err(anyhow::anyhow!(
                        "Bedrock streaming API error ({error_code}): {e}. Please check your AWS credentials."
                    ));
                    yield Ok(StreamEvent::Done);
                    return;
                }
            };

            let mut accumulated_input_tokens = 0u32;
            let mut accumulated_output_tokens = 0u32;
            let mut stream_parser = crate::llm::provider_utils::AnthropicStreamParser::new();

            // Process the event stream
            loop {
                match event_stream.recv().await {
                    Ok(Some(event)) => {
                        if let ResponseStream::Chunk(payload) = event
                            && let Some(bytes) = payload.bytes()
                        {
                            match serde_json::from_slice::<BedrockStreamEvent>(bytes.as_ref()) {
                                Ok(event) => {
                                    for output in apply_bedrock_stream_event(
                                        &mut stream_parser,
                                        event,
                                        &mut accumulated_input_tokens,
                                        &mut accumulated_output_tokens,
                                    ) {
                                        yield Ok(output);
                                    }
                                }
                                Err(error) => {
                                    tracing::warn!("Failed to parse Bedrock stream event: {error}");
                                }
                            }
                        }
                    }
                    Ok(None) => {
                        // Stream ended
                        tracing::debug!("Bedrock stream ended");
                        break;
                    }
                    Err(e) => {
                        tracing::error!("Bedrock stream error: {:?}", e);
                        yield Err(anyhow::anyhow!("Bedrock stream error: {e}"));
                        break;
                    }
                }
            }

            for evt in stream_parser.finish_tool_calls() {
                yield Ok(evt);
            }

            // Yield final token usage
            if accumulated_input_tokens > 0 || accumulated_output_tokens > 0 {
                let total = accumulated_input_tokens + accumulated_output_tokens;
                let token_usage = crate::messages::TokenUsage::from_counts(
                    Some(accumulated_input_tokens),
                    Some(accumulated_output_tokens),
                    Some(total),
                );
                crate::llm::provider_utils::log_token_usage(Some(&token_usage), "Bedrock");
                yield Ok(StreamEvent::Metadata {
                    usage: Some(token_usage),
                    finish_reason: None,
                    metrics: None,
                });
            } else {
                crate::llm::provider_utils::log_token_usage(None, "Bedrock");
            }

            yield Ok(StreamEvent::Done);
        })
    }

    fn name(&self) -> &'static str {
        "bedrock"
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true // Bedrock Claude models support vision
    }

    fn supports_audio_input(&self) -> bool {
        false
    }

    fn supports_document_input(&self) -> bool {
        true
    }
}

impl ModelProvider for BedrockProvider {
    fn available_models(&self) -> Vec<String> {
        BedrockModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Bedrock
    }
}

impl BedrockProvider {
    /// Fetch all available models from AWS Bedrock API
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models() -> anyhow::Result<Vec<String>> {
        let bearer_token = caliche_contracts::credential_env::var("AWS_BEARER_TOKEN_BEDROCK").ok();
        let mut config_builder = aws_config::defaults(BehaviorVersion::latest());

        if bearer_token.is_none()
            && let (Ok(access_key_id), Ok(secret_access_key)) = (
                caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID"),
                caliche_contracts::credential_env::var("AWS_SECRET_ACCESS_KEY"),
            )
        {
            let credentials = Credentials::new(
                &access_key_id,
                &secret_access_key,
                caliche_contracts::credential_env::var("AWS_SESSION_TOKEN").ok(),
                None,
                "env",
            );
            config_builder = config_builder.credentials_provider(credentials);
        }

        if let Ok(region_str) = caliche_contracts::credential_env::var("AWS_REGION")
            .or_else(|_| caliche_contracts::credential_env::var("AWS_DEFAULT_REGION"))
        {
            config_builder = config_builder.region(Region::new(region_str));
        }

        let config = config_builder.load().await;
        let client = BedrockControlClient::new(&config);

        let mut models = Vec::new();

        // Fetch models from Bedrock API
        // Note: The AWS SDK may require pagination handling differently
        // For now, we'll fetch the first page - pagination can be added if needed
        match client.list_foundation_models().send().await {
            Ok(response) => {
                // model_summaries() returns a slice, not an Option
                for model in response.model_summaries() {
                    models.push(model.model_id().to_string());
                }
            }
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to fetch Bedrock models: {e}"));
            }
        }

        if models.is_empty() {
            return Err(anyhow::anyhow!("No models found from Bedrock API"));
        }

        Ok(models)
    }
}

// Bedrock streaming event structures
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[expect(
    dead_code,
    reason = "deserialized from the Bedrock reply; unread fields keep the wire format complete"
)]
enum BedrockStreamEvent {
    /// Message start event - contains input token usage
    MessageStart { message: BedrockStreamMessage },
    /// Content block delta event - contains text content chunks
    ContentBlockDelta {
        delta: BedrockStreamDelta,
        #[serde(default)]
        index: Option<u32>,
    },
    /// Message delta event - contains output token usage
    MessageDelta {
        #[serde(default)]
        usage: Option<BedrockStreamUsage>,
        #[serde(default)]
        delta: Option<BedrockStreamMessageDelta>,
        #[serde(default)]
        stop_reason: Option<String>,
    },
    /// Content block start event
    ContentBlockStart {
        #[serde(default)]
        index: Option<u32>,
        #[serde(default)]
        #[serde(rename = "content_block")]
        content_block: Option<BedrockStreamContentBlock>,
    },
    /// Content block stop event
    ContentBlockStop {
        #[serde(default)]
        index: Option<u32>,
    },
    /// Message stop event - stream completion
    MessageStop {
        #[serde(default)]
        stop_reason: Option<String>,
        #[serde(default)]
        usage: Option<BedrockStreamUsage>,
    },
    /// Unknown event type (for forward compatibility)
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[expect(
    dead_code,
    reason = "deserialized from the Bedrock reply; unread fields keep the wire format complete"
)]
struct BedrockStreamMessage {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    content: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    usage: Option<BedrockStreamUsage>,
}

#[derive(Debug, Deserialize)]
#[expect(
    dead_code,
    reason = "deserialized from the Bedrock reply; unread fields keep the wire format complete"
)]
struct BedrockStreamDelta {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    #[serde(rename = "partial_json")]
    partial_json: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BedrockStreamUsage {
    #[serde(alias = "input_tokens", alias = "inputTokens")]
    #[serde(default)]
    input_tokens: Option<u32>,
    #[serde(alias = "output_tokens", alias = "outputTokens")]
    #[serde(default)]
    output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[expect(
    dead_code,
    reason = "deserialized from the Bedrock reply; unread fields keep the wire format complete"
)]
struct BedrockStreamMessageDelta {
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    stop_sequence: Option<String>,
}

#[derive(Debug, Deserialize)]
#[expect(
    dead_code,
    reason = "deserialized from the Bedrock reply; unread fields keep the wire format complete"
)]
struct BedrockStreamContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

// Embedding structures for Amazon Titan
#[derive(Debug, Clone, Serialize)]
struct TitanEmbeddingRequest {
    #[serde(rename = "inputText")]
    input_text: String,
}

#[derive(Debug, Clone, Deserialize)]
struct TitanEmbeddingResponse {
    embedding: Vec<f32>,
}

// Embedding structures for Cohere
#[derive(Debug, Clone, Serialize)]
struct CohereEmbeddingRequest {
    texts: Vec<String>,
    input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[expect(
    dead_code,
    reason = "deserialized from the Bedrock reply; unread fields keep the wire format complete"
)]
struct CohereEmbeddingResponse {
    embeddings: Vec<Vec<f32>>,
    id: String,
    response_type: String,
}

impl BedrockProvider {
    /// Get the default embedding model ID
    fn default_embedding_model() -> String {
        caliche_contracts::credential_env::var("BEDROCK_EMBEDDING_MODEL_ID")
            .unwrap_or_else(|_| "amazon.titan-embed-text-v1".to_string())
    }

    /// Create a `BedrockProvider` for embeddings with configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn with_config_for_embeddings(config: BedrockProviderConfig) -> anyhow::Result<Self> {
        let mut config = config;
        config.embeddings_enabled = true;
        if config.embedding_model_id.is_none() {
            config.embedding_model_id = Some(Self::default_embedding_model());
        }
        Self::with_config(config).await
    }

    /// Create a new `BedrockProvider` specifically for embeddings
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env_for_embeddings() -> anyhow::Result<Self> {
        Self::from_env(None).await
    }

    /// Check if the model is a Cohere embedding model
    fn is_cohere_model(model_id: &str) -> bool {
        model_id.starts_with("cohere.embed-")
    }

    /// Embed texts using Cohere model (batched request)
    async fn embed_cohere(
        &self,
        texts: Vec<String>,
        model_id: &str,
    ) -> anyhow::Result<(Vec<Vec<f32>>, u32)> {
        let cohere_request = CohereEmbeddingRequest {
            texts: texts.clone(),
            input_type: "search_document".to_string(),
            truncate: Some("END".to_string()),
        };

        let body = serde_json::to_string(&cohere_request)?;

        let invoke_request_builder = self
            .client
            .invoke_model()
            .model_id(model_id)
            .content_type("application/json")
            .body(body.as_bytes().into());

        let response = if let Some(token) = self.bearer_token.as_ref() {
            let token = token.clone();
            invoke_request_builder
                .customize()
                .map_request(move |mut req| -> Result<_, HeaderError> {
                    req.headers_mut()
                        .insert("authorization", errors::bearer_header_value(&token)?);
                    Ok(req)
                })
                .send()
                .await
        } else {
            invoke_request_builder.send().await
        }
        .map_err(|e| {
            anyhow::anyhow!(
                "Bedrock Cohere embedding API error: {e}. Please check your AWS credentials."
            )
        })?;

        let response_body = response.body();
        let response_str = String::from_utf8(response_body.as_ref().to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to decode Cohere response body: {e}"))?;

        tracing::debug!("Bedrock Cohere embedding response: {}", response_str);

        let cohere_response: CohereEmbeddingResponse = serde_json::from_str(&response_str)
            .map_err(|e| {
                anyhow::anyhow!(
                    "Failed to parse Bedrock Cohere embedding response: {e}. Response body: {response_str}"
                )
            })?;

        // Estimate tokens (rough approximation: ~4 chars per token)
        let total_chars: usize = texts.iter().map(std::string::String::len).sum();
        let tokens = estimated_tokens_from_chars(total_chars);

        Ok((cohere_response.embeddings, tokens))
    }
}

#[async_trait]
impl EmbeddingProvider for BedrockProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        let model_id = request.model.unwrap_or_else(Self::default_embedding_model);

        tracing::debug!(
            "Bedrock embedding request for {} texts using model {}",
            request.texts.len(),
            model_id
        );

        let (all_embeddings, total_input_tokens) = if Self::is_cohere_model(&model_id) {
            // Cohere supports batch embedding - send all texts at once
            tracing::debug!("Using Cohere batch embedding API");
            self.embed_cohere(request.texts, &model_id).await?
        } else {
            // Titan requires individual requests - process in parallel
            tracing::debug!("Using Titan individual embedding API with parallel processing");
            let max_concurrent = 20; // Limit concurrent requests to avoid rate limits
            // Store bearer_token as Option<&str> to avoid cloning until needed
            let bearer_token_opt = self.bearer_token.as_deref();
            let client = self.client.clone();

            let tasks = request.texts.into_iter().map(|text| {
                let model_id = model_id.clone();
                // Clone only when needed for the closure
                let bearer_token_opt = bearer_token_opt.map(std::string::ToString::to_string);
                let client = client.clone();

                async move {
                    // Store text length before moving into request
                    let text_len = text.len();
                    let titan_request = TitanEmbeddingRequest {
                        input_text: text,
                    };

                    let body = serde_json::to_string(&titan_request)?;

                    let invoke_request_builder = client
                        .invoke_model()
                        .model_id(&model_id)
                        .content_type("application/json")
                        .body(body.as_bytes().into());

                    let response = if let Some(token) = bearer_token_opt {
                        invoke_request_builder
                            .customize()
                            .map_request(move |mut req| -> Result<_, HeaderError> {
                                req.headers_mut().insert(
                                    "authorization",
                                    errors::bearer_header_value(&token)?,
                                );
                                Ok(req)
                            })
                            .send()
                            .await
                    } else {
                        invoke_request_builder.send().await
                    }
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "Bedrock Titan embedding API error: {e}. Please check your AWS credentials."
                        )
                    })?;

                    let response_body = response.body();
                    let response_str =
                        String::from_utf8(response_body.as_ref().to_vec()).map_err(|e| {
                            anyhow::anyhow!("Failed to decode Titan response body: {e}")
                        })?;

                    tracing::debug!("Bedrock Titan embedding response: {}", response_str);

                    let titan_response: TitanEmbeddingResponse =
                        serde_json::from_str(&response_str).map_err(|e| {
                            anyhow::anyhow!(
                                "Failed to parse Bedrock Titan embedding response: {e}. Response body: {response_str}"
                            )
                        })?;

                    // Estimate tokens (rough approximation: ~4 chars per token)
                    let tokens = estimated_tokens_from_chars(text_len);

                    Ok::<_, anyhow::Error>((titan_response.embedding, tokens))
                }
            });

            // Execute tasks with concurrency limit
            let results: Vec<_> = stream::iter(tasks)
                .buffer_unordered(max_concurrent)
                .collect()
                .await;

            // Collect embeddings and tokens
            let mut all_embeddings = Vec::new();
            let mut total_input_tokens = 0u32;

            for result in results {
                let (embedding, tokens) = result?;
                all_embeddings.push(embedding);
                total_input_tokens += tokens;
            }

            (all_embeddings, total_input_tokens)
        };

        tracing::info!(
            "Bedrock embedding generated {} embeddings, estimated tokens: {}",
            all_embeddings.len(),
            total_input_tokens
        );

        Ok(EmbeddingResponse {
            embeddings: all_embeddings,
            usage: Some(crate::messages::TokenUsage::from_embedding_tokens(
                total_input_tokens,
                Some(total_input_tokens),
            )),
            model: model_id,
        })
    }

    fn name(&self) -> &'static str {
        "bedrock"
    }
}

#[cfg(test)]
mod tests {
    use super::estimated_tokens_from_chars;

    #[test]
    fn estimated_tokens_rounds_up_and_saturates() {
        assert_eq!(estimated_tokens_from_chars(0), 0);
        assert_eq!(estimated_tokens_from_chars(1), 1);
        assert_eq!(estimated_tokens_from_chars(4), 1);
        assert_eq!(estimated_tokens_from_chars(5), 2);
        assert_eq!(estimated_tokens_from_chars(usize::MAX), u32::MAX);
    }
}
