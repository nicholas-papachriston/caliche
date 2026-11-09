use crate::llm::cache::ConversionCache;
use crate::llm::config::CommonProviderOptions;
use crate::llm::models::{AnthropicModel, ModelProvider, ProviderType};
use crate::llm::provider::Provider;
use crate::messages::{CompletionRequest, CompletionResponse, Message, MessageRole, StreamEvent};
use async_stream::stream;
use async_trait::async_trait;
use beef::lean::Cow;
use futures_util::Stream as FuturesStream;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;

const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";

#[derive(Debug, Clone)]
struct AnthropicMessage {
    role: Cow<'static, str>,
    content: serde_json::Value, // Can be string or array of content blocks
    cache_control: Option<serde_json::Value>,
}

impl serde::Serialize for AnthropicMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("role", &self.role)?;
        map.serialize_entry("content", &self.content)?;
        if let Some(ref cache_control) = self.cache_control {
            map.serialize_entry("cache_control", cache_control)?;
        }
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for AnthropicMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            role: String,
            content: serde_json::Value,
            #[serde(skip_serializing_if = "Option::is_none")]
            cache_control: Option<serde_json::Value>,
        }
        let helper = Helper::deserialize(deserializer)?;
        Ok(Self {
            role: Cow::owned(helper.role),
            content: helper.content,
            cache_control: helper.cache_control,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum AnthropicSystem {
    Text(String),
    Blocks(Vec<AnthropicSystemBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    thinking_type: String,
    budget_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicRequest {
    model: String,
    #[serde(rename = "max_tokens")]
    max_tokens: u32,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<AnthropicSystem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    response_type: String,
    role: String,
    content: Vec<AnthropicContentBlock>,
    model: String,
    #[serde(rename = "stop_reason")]
    stop_reason: Option<String>,
    #[serde(rename = "stop_sequence")]
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
enum AnthropicContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        #[serde(default)]
        signature: Option<String>,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking {
        #[serde(default)]
        data: Option<String>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicUsage {
    #[serde(rename = "input_tokens")]
    input_tokens: u32,
    #[serde(rename = "output_tokens")]
    output_tokens: u32,
    #[serde(default, rename = "cache_creation_input_tokens")]
    cache_creation_input_tokens: Option<u32>,
    #[serde(default, rename = "cache_read_input_tokens")]
    cache_read_input_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamChunk {
    #[serde(rename = "type")]
    chunk_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    index: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    delta: Option<AnthropicDelta>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "message")]
    message: Option<AnthropicStreamMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "content_block")]
    content_block: Option<AnthropicStreamContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamMessage {
    id: String,
    #[serde(rename = "type")]
    message_type: String,
    role: String,
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicStreamContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    delta_type: Option<String>,
    text: Option<String>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "partial_json")]
    partial_json: Option<String>,
}

/// Configuration for `AnthropicProvider`
#[derive(Debug, Clone)]
pub struct AnthropicProviderConfig {
    /// Anthropic API key
    pub api_key: String,
    /// Model identifier
    pub model: String,
    /// Common provider options
    pub common: CommonProviderOptions,
}

impl AnthropicProviderConfig {
    /// Create a new `AnthropicProviderConfig`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            common: CommonProviderOptions::default(),
        }
    }

    /// Create a builder for `AnthropicProviderConfig`
    #[must_use]
    pub fn builder() -> AnthropicProviderConfigBuilder {
        AnthropicProviderConfigBuilder::default()
    }
}

crate::impl_provider_config!(AnthropicProviderConfig);

/// Builder for `AnthropicProviderConfig`
#[derive(Debug, Default)]
pub struct AnthropicProviderConfigBuilder {
    api_key: Option<String>,
    model: Option<String>,
    common: CommonProviderOptions,
}

impl AnthropicProviderConfigBuilder {
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

    /// Build the configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn build(self) -> anyhow::Result<AnthropicProviderConfig> {
        let api_key = crate::llm::provider_utils::resolve_api_key(
            self.api_key,
            "ANTHROPIC_API_KEY",
            "api_key is required for AnthropicProviderConfig or ANTHROPIC_API_KEY must be set",
        )?;
        let model = crate::llm::provider_utils::resolve_model(
            self.model,
            "ANTHROPIC_MODEL",
            crate::llm::defaults::ANTHROPIC,
        );
        Ok(AnthropicProviderConfig {
            api_key,
            model,
            common: self.common,
        })
    }
}

crate::impl_common_builder_methods!(AnthropicProviderConfigBuilder);

pub struct AnthropicProvider {
    client: std::sync::Arc<reqwest::Client>,
    api_key: String,
    model: String,
    retry_strategy: Arc<dyn crate::traits::retry::RetryStrategy>,
    // Cache converted tool schemas for efficient reuse
    tool_cache: ConversionCache<Vec<AnthropicTool>>,
    // Cache converted messages for efficient reuse
    message_cache: ConversionCache<(Vec<AnthropicMessage>, Option<String>)>,
}

impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("client", &"Arc<reqwest::Client>")
            .field("api_key", &"***")
            .field("model", &self.model)
            .field("retry_strategy", &self.retry_strategy)
            .field("tool_cache", &self.tool_cache)
            .field("message_cache", &self.message_cache)
            .finish()
    }
}

impl AnthropicProvider {
    /// Create a new `AnthropicProvider` with configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn with_config(config: AnthropicProviderConfig) -> anyhow::Result<Self> {
        let client = crate::llm::provider_utils::create_http_client(&config.common)?;
        let retry_strategy = crate::llm::provider_utils::create_retry_strategy(&config.common);

        Ok(Self {
            client,
            api_key: config.api_key,
            model: config.model,
            retry_strategy,
            tool_cache: ConversionCache::new(),
            message_cache: ConversionCache::new(),
        })
    }

    /// Create a new `AnthropicProvider`, returning validation errors explicitly.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new(api_key: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let config = AnthropicProviderConfig::new(api_key, model);
        Self::with_config(config)
    }

    /// Create an `AnthropicProvider` from environment variables
    ///
    /// Reads the following environment variables:
    /// - `ANTHROPIC_API_KEY` - Required: Your Anthropic API key
    /// - `ANTHROPIC_MODEL` - Optional: Model to use (defaults to `claude-sonnet-4-6`)
    ///
    /// If `model` parameter is provided, it takes precedence over `ANTHROPIC_MODEL`.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        std::future::ready(()).await;
        let mut builder = AnthropicProviderConfig::builder();
        if let Some(model) = model {
            builder = builder.model(model);
        }
        let config = builder.build()?;
        Self::with_config(config)
    }

    /// Convert `MessageRole` to static string slice (zero-cost)
    const fn role_to_str(role: &MessageRole) -> &'static str {
        match role {
            MessageRole::System => "system",
            MessageRole::User | MessageRole::Tool => "user",
            MessageRole::Assistant => "assistant",
            // Tool messages converted to user in Anthropic format
        }
    }

    fn convert_messages(
        &self,
        messages: &[Message],
    ) -> Arc<(Vec<AnthropicMessage>, Option<String>)> {
        self.message_cache.get_or_compute_with_hash(
            messages,
            crate::llm::cache::hash_messages_for_cache_skip_system,
            convert_anthropic_messages,
        )
    }

    fn convert_tools(&self, tools: &[crate::messages::ToolSchema]) -> Arc<Vec<AnthropicTool>> {
        crate::llm::provider_utils::cached_tool_mapping(&self.tool_cache, tools, |tools| {
            tools
                .iter()
                .map(|schema| AnthropicTool {
                    name: schema.name.as_ref().to_string(),
                    description: schema.description.as_ref().to_string(),
                    input_schema: crate::llm::provider_utils::tool_schema_to_json_value(schema),
                    cache_control: None,
                })
                .collect()
        })
    }

    fn create_stream_request(&self, request: &CompletionRequest) -> AnthropicRequest {
        let (messages, system) = (*self.convert_messages(&request.messages)).clone();
        let (thinking, max_tokens) = anthropic_thinking_and_max_tokens(request);
        let tools = request
            .tools
            .as_ref()
            .map(|schemas| (*self.convert_tools(schemas)).clone());
        let tool_choice = crate::llm::provider_utils::convert_tool_choice_anthropic(
            request.tool_choice.as_ref(),
            tools.is_some(),
        );
        let (system, tools) = apply_anthropic_prompt_cache(system, tools);
        AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            messages,
            temperature: request.temperature,
            system,
            stream: Some(true),
            tools,
            tool_choice,
            response_format: request
                .response_format
                .as_ref()
                .map(crate::llm::provider_utils::response_format_to_json_value),
            thinking,
        }
    }
}

fn apply_anthropic_prompt_cache(
    system: Option<String>,
    mut tools: Option<Vec<AnthropicTool>>,
) -> (Option<AnthropicSystem>, Option<Vec<AnthropicTool>>) {
    let cache_control = crate::llm::prompt_cache::ephemeral_cache_control();
    if let Some(ref mut tools) = tools
        && let Some(last) = tools.last_mut()
    {
        last.cache_control = Some(cache_control.clone());
    }
    let system = system.map(|text| {
        AnthropicSystem::Blocks(vec![AnthropicSystemBlock {
            block_type: "text".to_string(),
            text,
            cache_control: Some(cache_control),
        }])
    });
    tracing::debug!(
        target: "caliche::prompt_cache",
        provider = "anthropic",
        breakpoints = "system+last_tool",
        "anthropic cache_control attached"
    );
    (system, tools)
}

fn convert_anthropic_messages(messages: &[Message]) -> (Vec<AnthropicMessage>, Option<String>) {
    let mut converted = Vec::with_capacity(messages.len());
    let mut system_message = None;
    for message in messages {
        let cache_control = message
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get("cache_control").cloned());
        match message.role {
            MessageRole::System => {
                system_message = Some(message.content.as_text().unwrap_or("").to_string());
            }
            MessageRole::User => converted.push(anthropic_user_message(message, cache_control)),
            MessageRole::Assistant => {
                converted.push(anthropic_assistant_message(message, cache_control));
            }
            MessageRole::Tool => converted.push(anthropic_tool_message(message, cache_control)),
        }
    }
    (converted, system_message)
}

fn anthropic_user_message(
    message: &Message,
    cache_control: Option<serde_json::Value>,
) -> AnthropicMessage {
    let content = match &message.content {
        crate::messages::MessageContent::Multimodal(parts) => {
            serde_json::Value::Array(crate::llm::provider_utils::content_parts_to_json_array(
                parts,
                crate::llm::provider_utils::ContentPartJsonFormat::Anthropic,
            ))
        }
        _ => serde_json::Value::String(message.content.as_text().unwrap_or("").to_string()),
    };
    AnthropicMessage {
        role: Cow::borrowed(AnthropicProvider::role_to_str(&message.role)),
        content,
        cache_control,
    }
}

fn anthropic_assistant_message(
    message: &Message,
    cache_control: Option<serde_json::Value>,
) -> AnthropicMessage {
    let content = message.content.as_tool_calls().map_or_else(
        || serde_json::Value::String(message.content.as_text().unwrap_or("").to_string()),
        |tool_calls| {
            let mut blocks = Vec::with_capacity(tool_calls.len() + 1);
            if let Some(text) = message.content.as_text()
                && !text.is_empty()
            {
                blocks.push(serde_json::json!({"type": "text", "text": text}));
            }
            blocks.extend(tool_calls.iter().map(|tool_call| {
                serde_json::json!({
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "input": tool_call.arguments
                })
            }));
            serde_json::Value::Array(blocks)
        },
    );
    AnthropicMessage {
        role: Cow::borrowed("assistant"),
        content,
        cache_control,
    }
}

fn anthropic_tool_message(
    message: &Message,
    cache_control: Option<serde_json::Value>,
) -> AnthropicMessage {
    let content = message.content.as_tool_call_id().map_or_else(
        || serde_json::Value::String(message.content.as_text().unwrap_or("").to_string()),
        |tool_call_id| {
            let result_text = match &message.content {
                crate::messages::MessageContent::Text(text) => text.as_ref().clone(),
                crate::messages::MessageContent::Multimodal(parts) => parts
                    .iter()
                    .filter_map(|part| match part {
                        crate::messages::ContentPart::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                crate::messages::MessageContent::Structured { text, .. } => text
                    .as_ref()
                    .map(|value| value.as_str().to_owned())
                    .unwrap_or_default(),
            };
            serde_json::json!([{
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": result_text
            }])
        },
    );
    AnthropicMessage {
        role: Cow::borrowed("user"),
        content,
        cache_control,
    }
}

fn anthropic_usage_event(usage: &AnthropicUsage) -> StreamEvent {
    let usage = Some(crate::llm::prompt_cache::token_usage_from_anthropic(
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_creation_input_tokens,
        usage.cache_read_input_tokens,
    ));
    crate::llm::provider_utils::log_prompt_cache_usage(
        usage.as_ref(),
        "Anthropic",
        Some(crate::llm::prompt_cache::ANTHROPIC_PROMPT_CACHE_KEY),
    );
    StreamEvent::Metadata {
        usage,
        finish_reason: None,
        metrics: None,
    }
}

fn parse_anthropic_stream_chunk(
    parser: &mut crate::llm::provider_utils::AnthropicStreamParser,
    json: &str,
) -> Vec<StreamEvent> {
    let Ok(chunk) = serde_json::from_str::<AnthropicStreamChunk>(json) else {
        tracing::debug!("Failed to parse Anthropic chunk: {json}");
        return Vec::new();
    };
    tracing::debug!(
        "Received chunk type: {}, index: {:?}",
        chunk.chunk_type,
        chunk.index
    );
    match chunk.chunk_type.as_str() {
        "content_block_delta" => chunk.delta.map_or_else(Vec::new, |delta| {
            parser.feed(
                crate::llm::provider_utils::AnthropicStreamFeed::ContentBlockDelta {
                    index: chunk.index,
                    text: delta.text.as_deref(),
                    thinking: delta.thinking.as_deref(),
                    partial_json: delta.partial_json.as_deref(),
                },
            )
        }),
        "content_block_start" => chunk.content_block.as_ref().map_or_else(Vec::new, |block| {
            parser.feed(
                crate::llm::provider_utils::AnthropicStreamFeed::ContentBlockStart {
                    index: chunk.index,
                    block_type: &block.block_type,
                    tool_id: block.id.as_deref(),
                    tool_name: block.name.as_deref(),
                },
            )
        }),
        "content_block_stop" => parser.feed(
            crate::llm::provider_utils::AnthropicStreamFeed::ContentBlockStop {
                index: chunk.index,
            },
        ),
        "message_start" | "message_delta" | "message_stop" => chunk
            .usage
            .as_ref()
            .map_or_else(Vec::new, |usage| vec![anthropic_usage_event(usage)]),
        _ => {
            tracing::debug!("Unknown Anthropic chunk type: {}", chunk.chunk_type);
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::ReasoningEffort;
    use std::sync::Arc;

    #[test]
    fn try_new_builds_provider() {
        let provider = AnthropicProvider::try_new(
            "test-key".to_string(),
            crate::llm::defaults::ANTHROPIC.to_string(),
        )
        .expect("provider should build");

        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.model, crate::llm::defaults::ANTHROPIC);
    }

    #[test]
    fn thinking_helper_maps_effort_and_raises_max_tokens() {
        let request = CompletionRequest {
            messages: Arc::new(vec![Message::user("hi")]),
            max_tokens: Some(100),
            temperature: None,
            reasoning_effort: Some(ReasoningEffort::High),
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let (thinking, max_tokens) = anthropic_thinking_and_max_tokens(&request);
        let thinking = thinking.expect("thinking should be enabled");
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, 32_768);
        assert_eq!(max_tokens, 32_769);
    }

    #[test]
    fn thinking_helper_is_noop_without_effort() {
        let request = CompletionRequest::new(Arc::new(vec![Message::user("hi")]));
        let (thinking, max_tokens) = anthropic_thinking_and_max_tokens(&request);
        assert!(thinking.is_none());
        assert_eq!(max_tokens, 4096);
    }

    #[test]
    fn thinking_helper_preserves_max_tokens_above_budget() {
        let request = CompletionRequest {
            messages: Arc::new(vec![Message::user("hi")]),
            max_tokens: Some(40_000),
            temperature: None,
            reasoning_effort: Some(ReasoningEffort::High),
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let (thinking, max_tokens) = anthropic_thinking_and_max_tokens(&request);
        assert_eq!(thinking.expect("thinking").budget_tokens, 32_768);
        assert_eq!(max_tokens, 40_000);
    }

    #[test]
    fn thinking_helper_maps_low_and_medium_budgets() {
        for (effort, budget) in [
            (ReasoningEffort::Low, 1_024u32),
            (ReasoningEffort::Medium, 8_192u32),
        ] {
            let request = CompletionRequest {
                messages: Arc::new(vec![Message::user("hi")]),
                max_tokens: Some(100),
                temperature: None,
                reasoning_effort: Some(effort),
                tools: None,
                tool_choice: None,
                response_format: None,
            };
            let (thinking, max_tokens) = anthropic_thinking_and_max_tokens(&request);
            assert_eq!(thinking.expect("thinking").budget_tokens, budget);
            assert_eq!(max_tokens, budget + 1);
        }
    }

    #[test]
    fn prompt_cache_marks_system_and_last_tool() {
        let tools = vec![
            AnthropicTool {
                name: "fs".into(),
                description: "files".into(),
                input_schema: serde_json::json!({"type": "object"}),
                cache_control: None,
            },
            AnthropicTool {
                name: "bash".into(),
                description: "shell".into(),
                input_schema: serde_json::json!({"type": "object"}),
                cache_control: None,
            },
        ];
        let (system, tools) =
            apply_anthropic_prompt_cache(Some("You are helpful.".into()), Some(tools));
        let AnthropicSystem::Blocks(blocks) = system.expect("system blocks") else {
            panic!("expected content-block system prompt");
        };
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].text, "You are helpful.");
        assert_eq!(
            blocks[0].cache_control,
            Some(crate::llm::prompt_cache::ephemeral_cache_control())
        );
        let tools = tools.expect("tools");
        assert!(tools[0].cache_control.is_none());
        assert_eq!(
            tools[1].cache_control,
            Some(crate::llm::prompt_cache::ephemeral_cache_control())
        );
        let json = serde_json::to_value(&tools[1]).expect("serialize last tool");
        assert_eq!(json["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn create_stream_request_serializes_cache_breakpoints() {
        let provider = AnthropicProvider::try_new(
            "test-key".to_string(),
            crate::llm::defaults::ANTHROPIC.to_string(),
        )
        .expect("provider should build");
        let tool = crate::messages::ToolSchema {
            name: beef::lean::Cow::borrowed("fs"),
            description: beef::lean::Cow::borrowed("files"),
            parameters: crate::messages::ParametersSchema {
                schema_type: beef::lean::Cow::borrowed("object"),
                properties: serde_json::json!({}),
                required: vec![],
            },
        };
        let request = CompletionRequest {
            messages: Arc::new(vec![Message::system("stable prefix"), Message::user("hi")]),
            max_tokens: Some(32),
            temperature: None,
            reasoning_effort: None,
            tools: Some(Arc::new(vec![tool])),
            tool_choice: None,
            response_format: None,
        };
        let body = provider.create_stream_request(&request);
        let json = serde_json::to_value(&body).expect("serialize anthropic request");
        assert_eq!(json["system"][0]["type"], "text");
        assert_eq!(json["system"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(json["tools"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn usage_event_maps_cache_tokens() {
        let event = anthropic_usage_event(&AnthropicUsage {
            input_tokens: 8,
            output_tokens: 2,
            cache_creation_input_tokens: Some(16),
            cache_read_input_tokens: Some(32),
        });
        let StreamEvent::Metadata { usage, .. } = event else {
            panic!("expected metadata");
        };
        let usage = usage.expect("usage");
        assert_eq!(usage.prompt_tokens, Some(56));
        assert_eq!(usage.cache_creation_tokens, Some(16));
        assert_eq!(usage.cache_read_tokens, Some(32));
    }
}

fn anthropic_thinking_and_max_tokens(
    request: &CompletionRequest,
) -> (Option<AnthropicThinking>, u32) {
    let thinking = request.reasoning_effort.map(|effort| AnthropicThinking {
        thinking_type: "enabled".to_string(),
        budget_tokens: effort.anthropic_budget_tokens(),
    });
    let mut max_tokens = request.max_tokens.unwrap_or(4096);
    if let Some(ref thinking) = thinking {
        // Anthropic requires max_tokens > thinking.budget_tokens.
        max_tokens = max_tokens.max(thinking.budget_tokens.saturating_add(1));
    }
    (thinking, max_tokens)
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let (messages, system) = (*self.convert_messages(&request.messages)).clone();

        let (thinking, max_tokens) = anthropic_thinking_and_max_tokens(&request);

        // Convert tool schemas to Anthropic format (with caching)
        let tools = request
            .tools
            .as_ref()
            .map(|schemas| (*self.convert_tools(schemas)).clone());

        let tool_choice = crate::llm::provider_utils::convert_tool_choice_anthropic(
            request.tool_choice.as_ref(),
            tools.is_some(),
        );

        let (system, tools) = apply_anthropic_prompt_cache(system, tools);

        let anthropic_request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens,
            messages,
            temperature: request.temperature,
            system,
            stream: Some(false),
            tools,
            tool_choice,
            response_format: request
                .response_format
                .as_ref()
                .map(crate::llm::provider_utils::response_format_to_json_value),
            thinking,
        };

        tracing::debug!("Anthropic request: {:?}", anthropic_request);

        let client = &self.client;
        let api_key = &self.api_key;

        let response = crate::llm::provider_utils::retry_llm_request(
            self.retry_strategy.as_ref(),
            "Anthropic",
            || {
                client
                    .post(format!("{ANTHROPIC_BASE_URL}/messages"))
                    .header("x-api-key", api_key)
                    .header("anthropic-version", "2023-06-01")
                    .header("Content-Type", "application/json")
                    .json(&anthropic_request)
                    .send()
            },
        )
        .await?;

        let anthropic_response: AnthropicResponse = response
            .json()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to parse Anthropic response: {e}"))?;

        tracing::debug!("Anthropic response: {:?}", anthropic_response);

        // Extract text content and tool calls from content blocks
        let mut text_content = String::new();
        let mut thinking_content = String::new();
        let mut tool_calls = Vec::new();

        for block in &anthropic_response.content {
            match block {
                AnthropicContentBlock::Text { text } => {
                    text_content.push_str(text);
                }
                AnthropicContentBlock::Thinking { thinking, .. } => {
                    thinking_content.push_str(thinking);
                }
                AnthropicContentBlock::RedactedThinking { .. } => {
                    if !thinking_content.is_empty() {
                        thinking_content.push('\n');
                    }
                    thinking_content.push_str("[redacted]");
                }
                AnthropicContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(crate::messages::ToolCallInfo {
                        id: beef::lean::Cow::owned(id.clone()),
                        name: beef::lean::Cow::owned(name.clone()),
                        arguments: input.clone(),
                        thought_signature: None,
                        action_id: None,
                        arguments_json: None,
                    });
                }
            }
        }

        let content = if !tool_calls.is_empty() {
            crate::messages::CompletionContent::WithToolCalls {
                content: text_content,
                tool_calls,
            }
        } else if text_content.is_empty() && thinking_content.is_empty() {
            return Err(anyhow::anyhow!("No content in Anthropic response"));
        } else {
            crate::messages::CompletionContent::Text {
                content: text_content,
            }
        };

        let usage = Some(crate::llm::prompt_cache::token_usage_from_anthropic(
            anthropic_response.usage.input_tokens,
            anthropic_response.usage.output_tokens,
            anthropic_response.usage.cache_creation_input_tokens,
            anthropic_response.usage.cache_read_input_tokens,
        ));

        crate::llm::provider_utils::log_prompt_cache_usage(
            usage.as_ref(),
            "Anthropic",
            Some(crate::llm::prompt_cache::ANTHROPIC_PROMPT_CACHE_KEY),
        );

        Ok(CompletionResponse {
            content,
            finish_reason: anthropic_response.stop_reason.map(beef::lean::Cow::owned),
            usage,
            reasoning_content: if thinking_content.is_empty() {
                None
            } else {
                Some(thinking_content)
            },
        })
    }

    fn supports_native_tools(&self) -> bool {
        true // Anthropic supports native tool use
    }

    fn streaming_usage_policy(&self) -> crate::llm::UsageMergePolicy {
        // Anthropic emits usage at both `message_start` (initial input tokens
        // and near-zero output tokens) and `message_delta` (cumulative update
        // for output tokens). Treat them as deltas so the input_tokens count
        // from `message_start` is preserved when `message_delta` reports 0.
        crate::llm::UsageMergePolicy::Delta
    }

    fn supports_vision(&self) -> bool {
        true // Anthropic Claude models support vision
    }

    fn supports_audio_input(&self) -> bool {
        false
    }

    fn supports_document_input(&self) -> bool {
        true
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn FuturesStream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        let api_key = self.api_key.clone();
        let client = self.client.clone();
        let anthropic_request = self.create_stream_request(&request);

        tracing::debug!(
            "Anthropic streaming request - model: {}, tools: {:?}, tool_choice: {:?}",
            anthropic_request.model,
            anthropic_request.tools.as_ref().map(std::vec::Vec::len),
            anthropic_request.tool_choice
        );

        Box::pin(stream! {
            let response = match client
                .post(format!("{ANTHROPIC_BASE_URL}/messages"))
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("Content-Type", "application/json")
                .json(&anthropic_request)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Failed to send streaming request to Anthropic: {e}"));
                    return;
                }
            };

            let response = match crate::llm::provider_utils::check_response_status(
                response,
                "Anthropic",
            )
            .await
            {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let mut stream_parser = crate::llm::provider_utils::AnthropicStreamParser::new();
            let mut lines =
                crate::llm::provider_utils::sse_data_lines(response);
            use futures_util::StreamExt;

            while let Some(line_result) = lines.next().await {
                let json_str = match line_result {
                    Ok(crate::llm::provider_utils::SseDataLine::Done) => {
                        for evt in stream_parser.finish() {
                            yield Ok(evt);
                        }
                        return;
                    }
                    Ok(crate::llm::provider_utils::SseDataLine::Data(s)) => s,
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                };

                for event in parse_anthropic_stream_chunk(&mut stream_parser, &json_str) {
                    yield Ok(event);
                }
            }

            for evt in stream_parser.finish() {
                yield Ok(evt);
            }
        })
    }

    fn name(&self) -> &'static str {
        "anthropic"
    }
}

impl ModelProvider for AnthropicProvider {
    fn available_models(&self) -> Vec<String> {
        AnthropicModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }
}

impl AnthropicProvider {
    /// Fetch all available models from Anthropic's API
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<String>> {
        let fallback_models = AnthropicModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect();

        // Anthropic uses a different auth header, so we need custom handling
        let response = self
            .client
            .get(format!("{ANTHROPIC_BASE_URL}/models"))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await;

        match response {
            Ok(resp) => {
                if !resp.status().is_success() {
                    tracing::debug!(
                        "Anthropic /models endpoint returned status {}, using enum models",
                        resp.status()
                    );
                    return Ok(fallback_models);
                }

                match resp
                    .json::<crate::llm::provider_utils::ModelsResponse>()
                    .await
                {
                    Ok(models_response) => {
                        let models: Vec<String> =
                            models_response.data.into_iter().map(|m| m.id).collect();

                        if models.is_empty() {
                            Ok(fallback_models)
                        } else {
                            Ok(models)
                        }
                    }
                    Err(e) => {
                        tracing::debug!(
                            "Failed to parse Anthropic models response: {}, using enum models",
                            e
                        );
                        Ok(fallback_models)
                    }
                }
            }
            Err(e) => {
                tracing::debug!(
                    "Failed to fetch Anthropic models from API: {}, using enum models",
                    e
                );
                Ok(fallback_models)
            }
        }
    }
}
