use crate::llm::cache::ConversionCache;
use crate::llm::config::CommonProviderOptions;
use crate::llm::models::{GoogleModel, ModelProvider, ProviderType};
use crate::llm::provider::Provider;
use crate::messages::{
    CompletionContent, CompletionRequest, CompletionResponse, Message, MessageRole, StreamEvent,
};
use async_trait::async_trait;
use beef::lean::Cow;
use futures_util::Stream;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use std::pin::Pin;
use std::sync::Arc;

const GOOGLE_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleContent {
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "crate::llm::provider_utils::serde_helpers::serialize_option_cow_static_str",
        deserialize_with = "crate::llm::provider_utils::serde_helpers::deserialize_option_cow_static_str"
    )]
    role: Option<Cow<'static, str>>,
    parts: Vec<GooglePart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum GooglePart {
    // Check FunctionCall first since it's more specific and has a required field
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GoogleFunctionCall,
        #[serde(
            rename = "thoughtSignature",
            alias = "thought_signature",
            skip_serializing_if = "Option::is_none"
        )]
        thought_signature: Option<String>,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GoogleFunctionResponse,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GoogleInlineData,
    },
    // Text must be last and only matches when text field is present
    // Note: text can be None in the Option, but the field must exist in JSON
    Text {
        #[serde(
            default,
            serialize_with = "crate::llm::provider_utils::serde_helpers::serialize_option_arc_str",
            deserialize_with = "crate::llm::provider_utils::serde_helpers::deserialize_option_arc_str"
        )]
        text: Option<std::sync::Arc<String>>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleInlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleFunctionCall {
    name: String,
    args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GoogleFunctionResponse {
    name: String,
    response: serde_json::Value,
}

impl GooglePart {
    fn from_text(text: impl Into<std::sync::Arc<String>>) -> Option<Self> {
        let arc = text.into();
        if arc.trim().is_empty() {
            return None;
        }
        Some(Self::Text { text: Some(arc) })
    }

    /// Map a [`ContentPart`] to a Gemini part using shared [`content_part`] conversion.
    fn from_content_part(part: &crate::messages::ContentPart) -> Option<Self> {
        use crate::llm::provider_utils::ContentPartJsonFormat;
        use crate::messages::ContentPart;

        match part {
            ContentPart::Text { text } => Self::from_text(text.to_string()),
            ContentPart::Document {
                data,
                media_type,
                filename,
            } if media_type == "application/pdf" => Some(Self::InlineData {
                inline_data: GoogleInlineData {
                    mime_type: media_type.to_string(),
                    data: data.to_string(),
                },
            }),
            ContentPart::Document { filename, .. } => Self::from_text(format!(
                "[Document attachment{} included; provide extracted text for best results]",
                filename
                    .as_ref()
                    .map(|f| format!(" ({f})"))
                    .unwrap_or_default()
            )),
            _ => {
                let blocks = crate::llm::provider_utils::content_parts_to_json_array(
                    std::slice::from_ref(part),
                    ContentPartJsonFormat::Google,
                );
                Self::from_google_json_block(blocks.first()?)
            }
        }
    }

    fn from_google_json_block(block: &serde_json::Value) -> Option<Self> {
        if let Some(text) = block.get("text").and_then(serde_json::Value::as_str) {
            return Self::from_text(text.to_string());
        }
        let inline = block
            .get("inline_data")
            .or_else(|| block.get("inlineData"))?;
        Some(Self::InlineData {
            inline_data: GoogleInlineData {
                mime_type: inline
                    .get("mime_type")
                    .or_else(|| inline.get("mimeType"))
                    .and_then(serde_json::Value::as_str)?
                    .to_string(),
                data: inline
                    .get("data")
                    .and_then(serde_json::Value::as_str)?
                    .to_string(),
            },
        })
    }
}

impl GoogleContent {
    fn with_role(
        role: Option<&'static str>,
        text: impl Into<std::sync::Arc<String>>,
    ) -> Option<Self> {
        GooglePart::from_text(text).map(|part| Self {
            role: role.map(Cow::borrowed),
            parts: vec![part],
        })
    }
}

fn google_assistant_parts(message: &Message) -> Vec<GooglePart> {
    use crate::messages::MessageContent;

    match &message.content {
        MessageContent::Text(text) => GooglePart::from_text(std::sync::Arc::clone(text))
            .into_iter()
            .collect(),
        MessageContent::Multimodal(_) => GooglePart::from_text(std::sync::Arc::new(
            message.content.as_text().unwrap_or("").to_string(),
        ))
        .into_iter()
        .collect(),
        MessageContent::Structured {
            text, tool_calls, ..
        } => {
            let mut parts = text
                .as_ref()
                .and_then(|text| GooglePart::from_text(std::sync::Arc::clone(text)))
                .into_iter()
                .collect::<Vec<_>>();
            parts.extend(
                tool_calls
                    .iter()
                    .flatten()
                    .map(|call| GooglePart::FunctionCall {
                        function_call: GoogleFunctionCall {
                            name: call.name.to_string(),
                            args: call.arguments.clone(),
                        },
                        thought_signature: call.thought_signature.clone(),
                    }),
            );
            parts
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleThinkingConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_budget: Option<i32>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<GoogleThinkingConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GoogleGenerateContentRequest {
    contents: Vec<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GoogleContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GoogleGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GoogleTool>>,
}

#[derive(Debug, Clone, Serialize)]
struct GoogleTool {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GoogleFunctionDeclaration>,
}

#[derive(Debug, Clone, Serialize)]
struct GoogleFunctionDeclaration {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleGenerateContentResponse {
    #[serde(default)]
    candidates: Vec<GoogleCandidate>,
    #[serde(default)]
    usage_metadata: Option<GoogleUsageMetadata>,
    #[serde(default)]
    prompt_feedback: Option<GooglePromptFeedback>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleCandidate {
    #[serde(default)]
    content: Option<GoogleContent>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    safety_ratings: Vec<GoogleSafetyRating>,
}

#[derive(Debug, Deserialize)]
struct GoogleUsageMetadata {
    #[serde(default, rename = "promptTokenCount")]
    prompt: i32,
    #[serde(default, rename = "candidatesTokenCount")]
    candidates: i32,
    #[serde(default, rename = "totalTokenCount")]
    total: i32,
    #[serde(default, rename = "cachedContentTokenCount")]
    cached: i32,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GooglePromptFeedback {
    #[serde(default)]
    safety_ratings: Vec<GoogleSafetyRating>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleSafetyRating {
    category: String,
    #[serde(default)]
    _probability: Option<String>,
    #[serde(default)]
    blocked: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct GoogleModelsResponse {
    #[serde(default)]
    models: Vec<GoogleModelInfo>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleModelInfo {
    name: String,
    #[serde(default)]
    supported_generation_methods: Vec<String>,
    #[serde(default)]
    input_token_limit: Option<usize>,
}

/// Configuration for `GoogleProvider`
#[derive(Debug, Clone)]
pub struct GoogleProviderConfig {
    /// Google API key
    pub api_key: String,
    /// Model identifier
    pub model: String,
    /// Optional base URL (defaults to Google's API)
    pub base_url: Option<String>,
    /// Common provider options
    pub common: CommonProviderOptions,
}

impl GoogleProviderConfig {
    /// Create a new `GoogleProviderConfig`.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        let model = model.into();
        Self {
            api_key: api_key.into(),
            model: Self::normalize_model(&model),
            base_url: None,
            common: CommonProviderOptions::default(),
        }
    }

    /// Create a builder for `GoogleProviderConfig`
    #[must_use]
    pub fn builder() -> GoogleProviderConfigBuilder {
        GoogleProviderConfigBuilder::default()
    }

    /// Return a Gemini text-chat model id. Audio-only/TTS model ids are not
    /// usable through this provider's text completion surface.
    pub fn normalize_text_chat_model(model: impl AsRef<str>) -> String {
        Self::normalize_model(model.as_ref())
    }

    fn normalize_model(model: &str) -> String {
        let trimmed = model.trim();
        if trimmed.is_empty() {
            return format!("models/{}", crate::llm::defaults::GOOGLE);
        }

        let normalized = if trimmed.starts_with("models/") {
            trimmed.to_string()
        } else {
            format!("models/{trimmed}")
        };

        if is_google_text_chat_model_id(&normalized) {
            normalized
        } else {
            tracing::warn!(
                model = %normalized,
                "Google model is not supported for text chat; falling back to gemini-2.5-flash",
            );
            crate::llm::defaults::GOOGLE_API.to_string()
        }
    }
}

fn is_google_text_chat_model_id(model: &str) -> bool {
    let id = model
        .trim()
        .trim_start_matches("models/")
        .to_ascii_lowercase();
    if id.is_empty() {
        return false;
    }

    // Gemini lists include modality-specific models such as
    // `gemini-2.5-pro-preview-tts`; those accept AUDIO-only response
    // modalities and fail text chat requests.
    ![
        "tts",
        "speech",
        "audio",
        "imagen",
        "image-generation",
        "veo",
    ]
    .iter()
    .any(|marker| id.contains(marker))
}

crate::impl_provider_config!(GoogleProviderConfig);

/// Builder for `GoogleProviderConfig`
#[derive(Debug, Default)]
pub struct GoogleProviderConfigBuilder {
    api_key: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    common: CommonProviderOptions,
}

impl GoogleProviderConfigBuilder {
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

    /// Add a custom header
    #[must_use]
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.common = self.common.with_header(key.into(), value.into());
        self
    }

    /// Set connection pool size
    #[must_use]
    pub fn connection_pool_size(mut self, size: usize) -> Self {
        self.common = self.common.with_connection_pool_size(size);
        self
    }

    /// Build the configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn build(self) -> anyhow::Result<GoogleProviderConfig> {
        let api_key = self
            .api_key
            .ok_or_else(|| anyhow::anyhow!("api_key is required for GoogleProviderConfig"))?;
        let model = self
            .model
            .unwrap_or_else(|| crate::llm::defaults::GOOGLE_API.to_string());
        Ok(GoogleProviderConfig {
            api_key,
            model: GoogleProviderConfig::normalize_model(&model),
            base_url: self.base_url,
            common: self.common,
        })
    }
}

crate::impl_common_builder_methods!(GoogleProviderConfigBuilder);

pub struct GoogleProvider {
    client: std::sync::Arc<reqwest::Client>,
    api_key: String,
    model: String,
    base_url: String,
    retry_strategy: Arc<dyn crate::traits::retry::RetryStrategy>,
    tool_cache: ConversionCache<Vec<GoogleTool>>,
    message_cache: ConversionCache<(Vec<GoogleContent>, Option<GoogleContent>)>,
    dynamic_context_window: std::sync::atomic::AtomicUsize,
}

impl std::fmt::Debug for GoogleProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GoogleProvider")
            .field("client", &"Arc<reqwest::Client>")
            .field("api_key", &"***")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .field("retry_strategy", &self.retry_strategy)
            .field("tool_cache", &self.tool_cache)
            .field("message_cache", &self.message_cache)
            .field(
                "dynamic_context_window",
                &self
                    .dynamic_context_window
                    .load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish()
    }
}

impl GoogleProvider {
    /// Create a new `GoogleProvider` with configuration
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn with_config(config: GoogleProviderConfig) -> anyhow::Result<Self> {
        let normalized_model = GoogleProviderConfig::normalize_model(&config.model);

        let client = crate::llm::provider_utils::create_http_client(&config.common)?;
        let retry_strategy = crate::llm::provider_utils::create_retry_strategy(&config.common);

        let base_url = config
            .base_url
            .unwrap_or_else(|| GOOGLE_BASE_URL.to_string());

        Ok(Self {
            client,
            api_key: config.api_key,
            model: normalized_model,
            base_url,
            retry_strategy,
            tool_cache: ConversionCache::new(),
            message_cache: ConversionCache::new(),
            dynamic_context_window: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    /// Create a new `GoogleProvider`, returning validation errors explicitly.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new(api_key: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let config = GoogleProviderConfig::new(api_key, model);
        Self::with_config(config)
    }

    /// Create a `GoogleProvider` from environment variables
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        std::future::ready(()).await;
        let api_key = caliche_contracts::credential_env::var("GOOGLE_API_KEY")
            .or_else(|_| caliche_contracts::credential_env::var("GEMINI_API_KEY"))
            .map_err(|_| {
                anyhow::anyhow!("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
            })?;

        let candidate = model
            .or_else(|| caliche_contracts::credential_env::var("GOOGLE_MODEL").ok())
            .or_else(|| caliche_contracts::credential_env::var("GEMINI_MODEL").ok())
            .unwrap_or_else(|| crate::llm::defaults::GOOGLE_API.to_string());

        let config = GoogleProviderConfig::new(api_key, candidate);
        Self::with_config(config)
    }

    fn convert_messages(
        &self,
        messages: &[Message],
    ) -> anyhow::Result<Arc<(Vec<GoogleContent>, Option<GoogleContent>)>> {
        let result = self.message_cache.get_or_compute_with_hash(
            messages,
            crate::llm::cache::hash_messages_for_cache,
            |messages| {
                use crate::messages::MessageContent;

                let mut contents = Vec::with_capacity(messages.len());
                let mut system_parts = Vec::new();

                for message in messages {
                    match message.role {
                        MessageRole::System => {
                            let text = crate::llm::provider_utils::extract_message_text(message);
                            if let Some(part) = GooglePart::from_text(std::sync::Arc::new(text)) {
                                system_parts.push(part);
                            }
                        }
                        MessageRole::User => {
                            if let MessageContent::Multimodal(parts) = &message.content {
                                let google_parts: Vec<GooglePart> = parts
                                    .iter()
                                    .filter_map(GooglePart::from_content_part)
                                    .collect();
                                if !google_parts.is_empty() {
                                    contents.push(GoogleContent {
                                        role: Some(Cow::borrowed("user")),
                                        parts: google_parts,
                                    });
                                }
                            } else {
                                let text =
                                    crate::llm::provider_utils::extract_message_text(message);
                                if let Some(content) = GoogleContent::with_role(
                                    Some("user"),
                                    std::sync::Arc::new(text),
                                ) {
                                    contents.push(content);
                                }
                            }
                        }
                        MessageRole::Assistant => {
                            let parts = google_assistant_parts(message);
                            if !parts.is_empty() {
                                contents.push(GoogleContent {
                                    role: Some(Cow::borrowed("model")),
                                    parts,
                                });
                            }
                        }
                        MessageRole::Tool => {
                            let text = crate::llm::provider_utils::extract_message_text(message);

                            // Extract function name from MessageContent
                            let function_name = match &message.content {
                                MessageContent::Structured {
                                    tool_call_id: Some(id),
                                    ..
                                } => id.clone().into_owned(),
                                _ => "unknown".to_string(),
                            };

                            // Convert text result to JSON value
                            let response_value = serde_json::json!({ "result": text });

                            contents.push(GoogleContent {
                                role: Some(Cow::borrowed("user")),
                                parts: vec![GooglePart::FunctionResponse {
                                    function_response: GoogleFunctionResponse {
                                        name: function_name,
                                        response: response_value,
                                    },
                                }],
                            });
                        }
                    }
                }

                let system_instruction = if system_parts.is_empty() {
                    None
                } else {
                    Some(GoogleContent {
                        role: None,
                        parts: system_parts,
                    })
                };

                (contents, system_instruction)
            },
        );

        // Check error condition after caching
        if result.0.is_empty() {
            return Err(anyhow::anyhow!(
                "Google provider requires at least one non-system message"
            ));
        }

        Ok(result)
    }

    /// Clean a property schema to remove fields unsupported by Google's API
    /// Google doesn't support: additionalProperties, and type must be a single string (not array)
    fn clean_property_schema(prop: &mut serde_json::Value) {
        if let Some(obj) = prop.as_object_mut() {
            // Remove unsupported fields
            obj.remove("additionalProperties");
            obj.remove("$schema");
            obj.remove("definitions");

            // Ensure type is a single string, not an array
            if let Some(type_val) = obj.get("type") {
                if let Some(type_array) = type_val.as_array() {
                    // If type is an array, take the first element
                    if let Some(first_type) = type_array.first().and_then(|t| t.as_str()) {
                        obj.insert(
                            "type".to_string(),
                            serde_json::Value::String(first_type.to_string()),
                        );
                    } else {
                        // If array is empty or invalid, default to "string"
                        obj.insert(
                            "type".to_string(),
                            serde_json::Value::String("string".to_string()),
                        );
                    }
                } else if !type_val.is_string() {
                    // If type is not a string or array, convert to string
                    obj.insert(
                        "type".to_string(),
                        serde_json::Value::String("string".to_string()),
                    );
                }
            }

            // Recursively clean nested objects (for object/array types)
            if let Some(properties) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
                for prop_value in properties.values_mut() {
                    Self::clean_property_schema(prop_value);
                }
            }

            // Clean items for array types
            if let Some(items) = obj.get_mut("items") {
                Self::clean_property_schema(items);
            }
        }
    }

    fn convert_tools(&self, tools: &[crate::messages::ToolSchema]) -> Arc<Vec<GoogleTool>> {
        crate::llm::provider_utils::cached_tool_mapping(&self.tool_cache, tools, |tools| {
            let function_declarations = tools
                .iter()
                .map(|tool| {
                    let mut parameters =
                        crate::llm::provider_utils::tool_schema_to_json_value(tool);

                    // Clean each property for Google API compatibility
                    if let Some(props) = parameters
                        .get_mut("properties")
                        .and_then(|p| p.as_object_mut())
                    {
                        for prop_value in props.values_mut() {
                            Self::clean_property_schema(prop_value);
                        }
                    }

                    // Ensure schema_type is a single string (not array)
                    let schema_type = if tool.parameters.schema_type.starts_with('[') {
                        serde_json::from_str::<Vec<String>>(tool.parameters.schema_type.as_ref())
                            .ok()
                            .and_then(|types| types.first().cloned())
                            .unwrap_or_else(|| "object".to_string())
                    } else {
                        tool.parameters.schema_type.as_ref().to_string()
                    };
                    parameters["type"] = serde_json::json!(schema_type);

                    GoogleFunctionDeclaration {
                        name: tool.name.as_ref().to_string(),
                        description: tool.description.as_ref().to_string(),
                        parameters,
                    }
                })
                .collect();

            vec![GoogleTool {
                function_declarations,
            }]
        })
    }

    fn build_url(&self, action: &str) -> anyhow::Result<Url> {
        let endpoint = format!("{}/{}:{}", self.base_url, self.model, action);
        let mut url = Url::parse(&endpoint)
            .map_err(|err| anyhow::anyhow!("Failed to build Google API URL: {err}"))?;
        url.query_pairs_mut().append_pair("key", &self.api_key);
        Ok(url)
    }

    fn extract_content(
        response: &GoogleGenerateContentResponse,
    ) -> (String, Vec<crate::messages::ToolCallInfo>) {
        let mut text_content = String::new();
        let mut tool_calls = Vec::new();

        for candidate in &response.candidates {
            if let Some(content) = candidate.content.as_ref() {
                for part in &content.parts {
                    match part {
                        GooglePart::Text { text: Some(text) } => {
                            if !text_content.is_empty() {
                                text_content.push(' ');
                            }
                            text_content.push_str(text);
                        }
                        GooglePart::FunctionCall {
                            function_call,
                            thought_signature,
                        } => {
                            // Use cached JSON serialization for optimization
                            let arguments_json = serde_json::to_string(&function_call.args)
                                .unwrap_or_else(|_| "{}".to_string());

                            // Generate a unique ID for the function call using UUIDv7
                            let id = format!("call_{}", uuid::Uuid::now_v7().simple());

                            let mut call_info = crate::messages::ToolCallInfo::from_json(
                                id,
                                function_call.name.clone(),
                                arguments_json,
                            );
                            call_info.thought_signature.clone_from(thought_signature);
                            tool_calls.push(call_info);
                        }
                        GooglePart::Text { text: None } => {}
                        GooglePart::FunctionResponse { .. } | GooglePart::InlineData { .. } => {
                            tracing::debug!(
                                "Google: skipping unhandled response part variant in completion extraction"
                            );
                        }
                    }
                }
            }
        }

        (text_content, tool_calls)
    }

    fn feed_google_stream_chunk(
        chunk: &GoogleGenerateContentResponse,
        accumulator: &mut crate::llm::provider_utils::GoogleStreamAccumulator,
    ) -> String {
        let mut text_content = String::new();

        for candidate in &chunk.candidates {
            let Some(content) = candidate.content.as_ref() else {
                continue;
            };
            for part in &content.parts {
                match part {
                    GooglePart::Text { text: Some(text) } => {
                        if !text_content.is_empty() {
                            text_content.push(' ');
                        }
                        text_content.push_str(text);
                    }
                    GooglePart::FunctionCall {
                        function_call,
                        thought_signature,
                    } => {
                        let arguments = serde_json::to_string(&function_call.args)
                            .unwrap_or_else(|_| "{}".to_string());
                        accumulator.feed_function_call(
                            function_call.name.clone(),
                            arguments,
                            thought_signature.clone(),
                        );
                    }
                    GooglePart::Text { text: None }
                    | GooglePart::FunctionResponse { .. }
                    | GooglePart::InlineData { .. } => {}
                }
            }
        }

        text_content
    }

    fn convert_usage(metadata: &GoogleUsageMetadata) -> crate::messages::TokenUsage {
        let prompt_tokens = u32::try_from(metadata.prompt).ok();
        let completion_tokens = u32::try_from(metadata.candidates).ok();
        let total_tokens = u32::try_from(metadata.total).ok();

        crate::messages::TokenUsage::from_counts(prompt_tokens, completion_tokens, total_tokens)
            .with_cache_tokens(None, u32::try_from(metadata.cached).ok().filter(|&n| n > 0))
    }

    fn build_safety_message(response: &GoogleGenerateContentResponse) -> Option<String> {
        let mut blocked_categories = Vec::new();

        for candidate in &response.candidates {
            if let Some(reason) = candidate.finish_reason.as_deref()
                && (reason.eq_ignore_ascii_case("safety") || reason.eq_ignore_ascii_case("blocked"))
            {
                blocked_categories.extend(
                    candidate
                        .safety_ratings
                        .iter()
                        .filter(|rating| rating.blocked.unwrap_or(false))
                        .map(|rating| rating.category.clone()),
                );
            }
        }

        if let Some(feedback) = response.prompt_feedback.as_ref() {
            let prompt_categories = feedback
                .safety_ratings
                .iter()
                .filter(|rating| rating.blocked.unwrap_or(false))
                .map(|rating| rating.category.clone());
            blocked_categories.extend(prompt_categories);
        }

        blocked_categories.sort();
        blocked_categories.dedup();

        if blocked_categories.is_empty() {
            None
        } else {
            let categories = blocked_categories.join(", ");
            Some(format!(
                "Google Gemini blocked the response due to safety filters in categories: {categories}. Try rephrasing the request."
            ))
        }
    }
}

fn google_generation_config(request: &CompletionRequest) -> Option<GoogleGenerationConfig> {
    let max_output_tokens = request
        .max_tokens
        .and_then(|value| i32::try_from(value).ok());
    let thinking_config = request.reasoning_effort.map(|effort| GoogleThinkingConfig {
        thinking_budget: Some(effort.google_thinking_budget()),
    });
    if max_output_tokens.is_none() && request.temperature.is_none() && thinking_config.is_none() {
        None
    } else {
        Some(GoogleGenerationConfig {
            max_output_tokens,
            temperature: request.temperature,
            thinking_config,
        })
    }
}

#[async_trait]
impl Provider for GoogleProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let (contents, system_instruction) = (*self.convert_messages(&request.messages)?).clone();

        let generation_config = google_generation_config(&request);

        // Convert tools if present
        let tools = request
            .tools
            .as_ref()
            .map(|t| (*self.convert_tools(t)).clone());

        let payload = GoogleGenerateContentRequest {
            contents,
            system_instruction,
            generation_config,
            tools,
        };

        tracing::debug!("Google request: {:?}", payload);

        let url = self.build_url("generateContent")?;
        let client = self.client.as_ref();

        let response = crate::llm::provider_utils::retry_llm_request(
            self.retry_strategy.as_ref(),
            "Google",
            || client.post(url.clone()).json(&payload).send(),
        )
        .await?;

        let raw_body = response.text().await.map_err(|err| {
            tracing::error!("Failed to read Google response body: {}", err);
            anyhow::anyhow!("Failed to read Google response body: {err}")
        })?;

        tracing::debug!("Raw Google response body: {}", raw_body);

        let google_response: GoogleGenerateContentResponse = serde_json::from_str(&raw_body)
            .map_err(|err| {
                tracing::error!("Failed to parse Google response: {}", err);
                anyhow::anyhow!("Failed to parse Google response: {err}")
            })?;

        tracing::debug!("Google response: {:?}", google_response);

        // Debug: Log parts for each candidate
        for (idx, candidate) in google_response.candidates.iter().enumerate() {
            if let Some(content) = candidate.content.as_ref() {
                tracing::debug!("Candidate {} parts: {:?}", idx, content.parts);
                for (part_idx, part) in content.parts.iter().enumerate() {
                    match part {
                        GooglePart::FunctionCall {
                            function_call,
                            thought_signature,
                        } => {
                            tracing::debug!(
                                "Part {} is FunctionCall: name={}, args={:?}, thought_signature={:?}",
                                part_idx,
                                function_call.name,
                                function_call.args,
                                thought_signature
                            );
                        }
                        GooglePart::Text { text } => {
                            tracing::debug!("Part {} is Text: {:?}", part_idx, text);
                        }
                        _ => {
                            tracing::debug!("Part {} is other type: {:?}", part_idx, part);
                        }
                    }
                }
            }
        }

        // Extract content and tool calls
        let (text_content, tool_calls) = Self::extract_content(&google_response);

        tracing::debug!(
            "Extracted text_content: '{}', tool_calls: {}",
            text_content,
            tool_calls.len()
        );

        // Check for safety blocks if no content was returned
        let content = if text_content.is_empty() && tool_calls.is_empty() {
            Self::build_safety_message(&google_response).map_or_else(|| {
                tracing::warn!("Google response did not include any content");
                CompletionContent::Text {
                    content: "Google Gemini returned no content. This often means the request was blocked by safety filters. Try rephrasing or simplifying the prompt.".to_string()
                }
            }, |message| {
                tracing::warn!("{}", message);
                CompletionContent::Text { content: message }
            })
        } else if !tool_calls.is_empty() {
            CompletionContent::WithToolCalls {
                content: text_content,
                tool_calls,
            }
        } else {
            CompletionContent::Text {
                content: text_content,
            }
        };

        let usage = google_response
            .usage_metadata
            .as_ref()
            .map(Self::convert_usage);

        crate::llm::provider_utils::log_token_usage(usage.as_ref(), "Google");

        let finish_reason = google_response
            .candidates
            .first()
            .and_then(|candidate| candidate.finish_reason.clone().map(beef::lean::Cow::owned));

        Ok(CompletionResponse {
            content,
            finish_reason,
            usage,
            reasoning_content: None,
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        Box::pin(async_stream::stream! {
            let (contents, system_instruction) = match self.convert_messages(&request.messages) {
                Ok(c) => (*c).clone(),
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let generation_config = google_generation_config(&request);

            let tools = request
                .tools
                .as_ref()
                .map(|t| (*self.convert_tools(t)).clone());

            let payload = GoogleGenerateContentRequest {
                contents,
                system_instruction,
                generation_config,
                tools,
            };

            let mut url = match self.build_url("streamGenerateContent") {
                Ok(u) => u,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            url.query_pairs_mut().append_pair("alt", "sse");

            let client = self.client.clone();

            let response = match crate::llm::provider_utils::retry_llm_request(
                self.retry_strategy.as_ref(),
                "Google",
                || client.post(url.clone()).json(&payload).send(),
            ).await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(anyhow::anyhow!("Failed to send stream request: {e}"));
                    return;
                }
            };

            let mut stream = crate::llm::provider_utils::sse_data_lines(response);
            use futures_util::StreamExt;

            let mut yielded_done = false;
            let mut stream_accumulator =
                crate::llm::provider_utils::GoogleStreamAccumulator::new();

            while let Some(line_result) = stream.next().await {
                match line_result {
                    Ok(crate::llm::provider_utils::SseDataLine::Done) => {
                        for evt in stream_accumulator.finalize_all() {
                            yield Ok(evt);
                        }
                        yield Ok(StreamEvent::Done);
                        yielded_done = true;
                        break;
                    }
                    Ok(crate::llm::provider_utils::SseDataLine::Data(data)) => {
                        let chunk: GoogleGenerateContentResponse = match serde_json::from_str(&data) {
                            Ok(c) => c,
                            Err(e) => {
                                tracing::error!("Failed to parse Google stream chunk: {} (data: {})", e, data);
                                continue;
                            }
                        };

                        let text_content =
                            Self::feed_google_stream_chunk(&chunk, &mut stream_accumulator);

                        if !text_content.is_empty() {
                            yield Ok(StreamEvent::TextChunk { content: text_content });
                        }

                        let finish_reason = chunk
                            .candidates
                            .first()
                            .and_then(|c| c.finish_reason.clone().map(beef::lean::Cow::owned));

                        if finish_reason.is_some() {
                            for evt in stream_accumulator.finalize_all() {
                                yield Ok(evt);
                            }
                        }

                        let usage = chunk.usage_metadata.as_ref().map(Self::convert_usage);

                        if usage.is_some() || finish_reason.is_some() {
                            yield Ok(StreamEvent::Metadata {
                                usage,
                                finish_reason,
                                metrics: None,
                            });
                        }
                    }
                    Err(e) => {
                        yield Err(e);
                        return;
                    }
                }
            }
            if !yielded_done {
                for evt in stream_accumulator.finalize_all() {
                    yield Ok(evt);
                }
                yield Ok(StreamEvent::Done);
            }
        })
    }

    fn name(&self) -> &'static str {
        "google"
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supports_native_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true // Google Gemini models support vision
    }

    fn supports_audio_input(&self) -> bool {
        true
    }

    fn supports_document_input(&self) -> bool {
        true
    }

    fn supported_media_types(&self) -> Vec<&str> {
        vec![
            "image/png",
            "image/jpeg",
            "image/webp",
            "video/mp4",
            "video/webm",
            "audio/mpeg",
            "audio/wav",
            "audio/ogg",
            "application/pdf",
        ]
    }

    fn context_window(&self) -> Option<usize> {
        match self
            .dynamic_context_window
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => None,
            v => Some(v),
        }
    }
}

impl ModelProvider for GoogleProvider {
    fn available_models(&self) -> Vec<String> {
        GoogleModel::all()
            .into_iter()
            .map(|model| model.as_str().to_owned())
            .collect()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Google
    }
}

impl GoogleProvider {
    /// Fetch available generative models from Google's Gemini API.
    /// Falls back to the static enum list when the API is unreachable
    /// or returns an empty payload.
    /// Also stores the context window for the active model when available.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<String>> {
        let fallback_models = GoogleModel::all()
            .into_iter()
            .map(|model| model.as_str().to_owned())
            .collect::<Vec<_>>();

        let response = self
            .client
            .get(format!("{}/models", self.base_url))
            .query(&[("key", &self.api_key)])
            .send()
            .await;

        let response = match response {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!(
                    "Failed to fetch Google models from API: {}, using fallback models",
                    e
                );
                return Ok(fallback_models);
            }
        };

        if !response.status().is_success() {
            tracing::debug!(
                "Google /models endpoint returned status {}, using fallback models",
                response.status()
            );
            return Ok(fallback_models);
        }

        match response.json::<GoogleModelsResponse>().await {
            Ok(models_response) => {
                for model_info in &models_response.models {
                    if model_info.name == self.model {
                        if let Some(limit) = model_info.input_token_limit {
                            tracing::debug!(
                                "Google: model {} has inputTokenLimit={}",
                                self.model,
                                limit
                            );
                            self.dynamic_context_window
                                .store(limit, std::sync::atomic::Ordering::Relaxed);
                        }
                        break;
                    }
                }

                let generative_models: Vec<String> = models_response
                    .models
                    .into_iter()
                    .filter(|model| {
                        model
                            .supported_generation_methods
                            .iter()
                            .any(|method| method.eq_ignore_ascii_case("generatecontent"))
                            && is_google_text_chat_model_id(&model.name)
                    })
                    .map(|model| model.name)
                    .collect();

                if generative_models.is_empty() {
                    tracing::debug!(
                        "Google /models endpoint returned no generative models, using fallback"
                    );
                    Ok(fallback_models)
                } else {
                    Ok(generative_models)
                }
            }
            Err(e) => {
                tracing::debug!(
                    "Failed to parse Google models response: {}, using fallback models",
                    e
                );
                Ok(fallback_models)
            }
        }
    }

    /// Fetch context window for the active model from Google's API.
    pub async fn fetch_context_window(&self) -> Option<usize> {
        let existing = self
            .dynamic_context_window
            .load(std::sync::atomic::Ordering::Relaxed);
        if existing != 0 {
            return Some(existing);
        }

        let response = self
            .client
            .get(format!("{}/{}", self.base_url, self.model))
            .query(&[("key", &self.api_key)])
            .send()
            .await
            .ok()?;

        if !response.status().is_success() {
            return None;
        }

        let model_info: GoogleModelInfo = response.json().await.ok()?;
        if let Some(limit) = model_info.input_token_limit {
            self.dynamic_context_window
                .store(limit, std::sync::atomic::Ordering::Relaxed);
            return Some(limit);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::ReasoningEffort;

    #[test]
    fn generation_config_maps_reasoning_effort_budget() {
        let request = CompletionRequest {
            messages: Arc::new(vec![Message::user("hi")]),
            max_tokens: None,
            temperature: None,
            reasoning_effort: Some(ReasoningEffort::Medium),
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let config = google_generation_config(&request).expect("config should exist");
        let thinking = config.thinking_config.expect("thinking config");
        assert_eq!(thinking.thinking_budget, Some(4_096));
    }

    #[test]
    fn generation_config_omitted_without_overrides() {
        let request = CompletionRequest::new(Arc::new(vec![Message::user("hi")]));
        assert!(google_generation_config(&request).is_none());
    }

    #[test]
    fn generation_config_present_for_temperature_only() {
        let request = CompletionRequest {
            messages: Arc::new(vec![Message::user("hi")]),
            max_tokens: None,
            temperature: Some(0.4),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let config = google_generation_config(&request).expect("config should exist");
        assert_eq!(config.temperature, Some(0.4));
        assert!(config.thinking_config.is_none());
    }

    #[test]
    fn generation_config_maps_high_thinking_budget() {
        let request = CompletionRequest {
            messages: Arc::new(vec![Message::user("hi")]),
            max_tokens: None,
            temperature: None,
            reasoning_effort: Some(ReasoningEffort::High),
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let config = google_generation_config(&request).expect("config should exist");
        assert_eq!(
            config.thinking_config.expect("thinking").thinking_budget,
            Some(16_384)
        );
    }
}
