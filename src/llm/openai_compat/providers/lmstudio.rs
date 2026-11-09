//! LM Studio provider configuration.

use crate::llm::config::CommonProviderOptions;
use crate::llm::models::{ModelProvider, ProviderType};
use crate::llm::openai_compat::{OpenAICompatConfig, OpenAICompatibleProvider};
use crate::llm::provider_utils::first_token::{LocalStreamLimits, ModelLoadProbe, server_root};
use serde::Deserialize;

const LMSTUDIO_DEFAULT_BASE_URL: &str = "http://localhost:1234/v1";

/// Response structure for LM Studio's /v1/models endpoint (OpenAI-compatible)
#[derive(Debug, Deserialize)]
struct LMStudioModelsResponse {
    data: Vec<LMStudioModelInfo>,
}

/// Individual model info from LM Studio API
#[derive(Debug, Deserialize)]
struct LMStudioModelInfo {
    id: String,
    #[serde(default)]
    max_model_len: Option<usize>,
    #[serde(default)]
    context_length: Option<usize>,
}

/// Clean special control tokens from LM Studio output.
/// Some models output tokens like <|channel|>, <|constrain|>, <|message|>
///
/// NOTE: This is called per streaming chunk, so we must NOT trim whitespace.
/// Trimming would strip spaces between words since spaces are often their own tokens.
fn clean_lmstudio_output(content: &str) -> String {
    let mut cleaned = content.to_string();

    // Strip common control tokens
    let tokens_to_remove = [
        "<|channel|>commentary",
        "<|channel|>",
        "to=echo",
        "<|constrain|>json",
        "<|constrain|>",
        "<|message|>",
        "<|im_start|>",
        "<|im_end|>",
    ];

    for token in &tokens_to_remove {
        cleaned = cleaned.replace(token, "");
    }

    // If the content looks like {"text":"..."}, extract just the text value
    // Use trim() only for the JSON check, not for the final output
    let trimmed = cleaned.trim();
    if trimmed.starts_with("{\"text\":\"")
        && let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed)
        && let Some(text) = json.get("text").and_then(|v| v.as_str())
    {
        return text.to_string();
    }

    // Don't trim - preserve whitespace for proper word spacing in streaming
    cleaned
}

/// Configuration for LM Studio provider.
#[derive(Debug, Clone)]
pub struct LMStudioConfig {
    pub base_url: String,
    pub model: String,
    pub embedding_model: Option<String>,
    /// Common provider options
    pub common: CommonProviderOptions,
    /// First-token and stall limits for streamed replies.
    pub stream_limits: LocalStreamLimits,
}

impl LMStudioConfig {
    /// Create a new LM Studio configuration.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            embedding_model: None,
            common: CommonProviderOptions::default(),
            stream_limits: LocalStreamLimits::default(),
        }
    }

    /// Create a builder for LM Studio configuration.
    #[must_use]
    pub fn builder() -> LMStudioConfigBuilder {
        LMStudioConfigBuilder::default()
    }
}

impl OpenAICompatConfig for LMStudioConfig {
    fn first_token_timeout(&self) -> Option<std::time::Duration> {
        Some(self.stream_limits.first_token())
    }

    fn stall_timeout(&self) -> std::time::Duration {
        self.stream_limits.stall()
    }

    fn model_load_probe(&self) -> Option<ModelLoadProbe> {
        Some(ModelLoadProbe::LmStudio {
            url: format!(
                "{}/api/v0/models/{}",
                server_root(&self.base_url),
                self.model
            ),
            model: self.model.clone(),
        })
    }
    fn model_loader(&self) -> Option<crate::llm::provider_utils::model_load::ModelLoader> {
        Some(crate::llm::provider_utils::model_load::ModelLoader::lm_studio(&self.base_url))
    }

    fn provider_name(&self) -> &'static str {
        "lmstudio"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn auth_headers(&self) -> Vec<(&'static str, String)> {
        vec![] // LM Studio doesn't require authentication
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn post_process_output<'a>(&self, content: &'a str) -> beef::Cow<'a, str> {
        beef::Cow::owned(clean_lmstudio_output(content))
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn default_embedding_model(&self) -> Option<String> {
        self.embedding_model.clone().or_else(|| {
            std::env::var("LMSTUDIO_EMBEDDING_MODEL")
                .ok()
                .or_else(|| Some("text-embedding-qwen3-embedding-4b".to_string()))
        })
    }

    /// Enable usage tracking in streaming responses.
    /// LM Studio supports OpenAI-compatible streaming with usage data.
    fn include_usage_in_stream(&self) -> bool {
        true
    }

    /// Convert `tool_choice` to LM Studio format.
    /// LM Studio only supports string values: "none", "auto", "required".
    /// It does not support object format for specific tool selection.
    /// For `ToolChoice::Specific`, we convert to "auto" to allow tool usage without forcing it on every turn.
    /// This prevents infinite loops where the model is forced to use tools but never provides a final response.
    fn convert_tool_choice(
        &self,
        tool_choice: &crate::messages::ToolChoice,
    ) -> Option<serde_json::Value> {
        match tool_choice {
            crate::messages::ToolChoice::None => Some(serde_json::json!("none")),
            crate::messages::ToolChoice::Required => Some(serde_json::json!("required")),
            // LM Studio doesn't support specific tool selection.
            // Convert to "auto" instead of "required" to allow the model to provide a final text response.
            // The system prompt can guide it to use the specific tool when needed.
            crate::messages::ToolChoice::Auto | crate::messages::ToolChoice::Specific(_) => {
                Some(serde_json::json!("auto"))
            }
        }
    }
}

/// Builder for LM Studio configuration.
#[derive(Debug, Default)]
pub struct LMStudioConfigBuilder {
    base_url: Option<String>,
    model: Option<String>,
    embedding_model: Option<String>,
    common: CommonProviderOptions,
    stream_limits: LocalStreamLimits,
}

impl LMStudioConfigBuilder {
    /// Set the base URL.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Set the model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the embedding model.
    #[must_use]
    pub fn embedding_model(mut self, model: impl Into<String>) -> Self {
        self.embedding_model = Some(model.into());
        self
    }

    /// Set request timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.common = self.common.with_timeout(timeout);
        self
    }

    /// Set maximum retries.
    #[must_use]
    pub fn max_retries(mut self, max_retries: usize) -> Self {
        self.common = self.common.with_max_retries(max_retries);
        self
    }

    /// Set the first-token and stall limits for streamed replies.
    #[must_use]
    pub const fn stream_limits(mut self, limits: LocalStreamLimits) -> Self {
        self.stream_limits = limits;
        self
    }

    /// Use a shared HTTP client for connection pooling across multiple agents.
    #[cfg(feature = "reqwest")]
    #[must_use]
    pub fn shared_http_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.common = self.common.with_client(client);
        self
    }

    /// Build the configuration and create a provider.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn build(self) -> anyhow::Result<OpenAICompatibleProvider<LMStudioConfig>> {
        let base_url = self.base_url.unwrap_or_else(|| {
            crate::llm::provider_utils::read_base_url_from_env(
                "LMSTUDIO_BASE_URL",
                LMSTUDIO_DEFAULT_BASE_URL,
            )
        });

        let model = self
            .model
            .or_else(|| std::env::var("LMSTUDIO_MODEL").ok())
            .filter(|model| !model.trim().is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "LM Studio model is required. Set LMSTUDIO_MODEL, pass .model(...), or use LMStudioProvider::from_env(None) for live discovery."
                )
            })?;

        // Read embedding model from env if not set programmatically
        let embedding_model = self
            .embedding_model
            .or_else(|| std::env::var("LMSTUDIO_EMBEDDING_MODEL").ok());

        let config = LMStudioConfig {
            base_url,
            model,
            embedding_model,
            common: self.common,
            stream_limits: self.stream_limits,
        };
        OpenAICompatibleProvider::with_options(config.clone(), config.common)
    }
}

// Canonical ergonomic provider name for LM Studio configuration.
pub type LMStudioProvider = OpenAICompatibleProvider<LMStudioConfig>;

impl LMStudioProvider {
    /// Create a new LM Studio provider, returning validation errors explicitly.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new(base_url: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let config = LMStudioConfig::new(base_url, model);
        Self::with_defaults(config)
    }

    /// Create an LM Studio provider from configuration.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn with_config(config: LMStudioConfig) -> anyhow::Result<Self> {
        Self::with_options(config.clone(), config.common)
    }

    /// Create an LM Studio provider from environment variables.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        let base_url = crate::llm::provider_utils::read_base_url_from_env(
            "LMSTUDIO_BASE_URL",
            LMSTUDIO_DEFAULT_BASE_URL,
        );
        let model = match model.or_else(|| std::env::var("LMSTUDIO_MODEL").ok()) {
            Some(model) if !model.trim().is_empty() => model,
            _ => Self::discover_default_model(&base_url).await?,
        };
        let builder = LMStudioConfig::builder().base_url(base_url).model(model);
        builder.build()
    }

    /// Discover the first chat-capable model currently served by LM Studio.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn discover_default_model(base_url: &str) -> anyhow::Result<String> {
        let discovery_provider = Self::try_new(base_url.to_string(), "model-discovery")?;
        let models = discovery_provider.fetch_available_models().await?;
        models
            .iter()
            .find(|model| !looks_like_embedding_model(model))
            .or_else(|| models.first())
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No LM Studio models discovered at {base_url}. Load a model in LM Studio or set LMSTUDIO_MODEL."
                )
            })
    }

    /// Check if LM Studio is available at the given base URL.
    /// Makes a quick GET request to /v1/models endpoint.
    pub async fn is_available(base_url: Option<&str>) -> bool {
        let url = base_url.unwrap_or(LMSTUDIO_DEFAULT_BASE_URL);

        let Ok(client) = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
        else {
            return false;
        };

        (client.get(format!("{url}/models")).send().await)
            .map_or(false, |resp| resp.status().is_success())
    }

    /// Fetch available models from LM Studio's API.
    /// Also stores context window for the active model if available.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<String>> {
        let fallback_models = Vec::new();

        let base_url = self.config().base_url();
        let client = self.http_client();
        let url = format!("{base_url}/models");

        let response = match client.get(&url).send().await {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!("Failed to fetch LM Studio models: {}, using fallback", e);
                return Ok(fallback_models);
            }
        };

        if !response.status().is_success() {
            tracing::debug!(
                "LM Studio /v1/models returned status {}, using fallback",
                response.status()
            );
            return Ok(fallback_models);
        }

        match response.json::<LMStudioModelsResponse>().await {
            Ok(models_response) => {
                let active_model = self.config().model.as_str();
                for model_info in &models_response.data {
                    if model_info.id == active_model {
                        let ctx = model_info.max_model_len.or(model_info.context_length);
                        if let Some(ctx_len) = ctx {
                            tracing::debug!(
                                "LM Studio: model {} has context_length={}",
                                active_model,
                                ctx_len
                            );
                            self.set_dynamic_context_window(ctx_len);
                        }
                        break;
                    }
                }

                let models: Vec<String> = models_response.data.into_iter().map(|m| m.id).collect();

                if models.is_empty() {
                    tracing::debug!("LM Studio returned no models, using fallback");
                    return Ok(fallback_models);
                }

                tracing::debug!("Fetched {} models from LM Studio API", models.len());
                Ok(models)
            }
            Err(e) => {
                tracing::debug!("Failed to parse LM Studio models: {}, using fallback", e);
                Ok(fallback_models)
            }
        }
    }
}

impl ModelProvider for LMStudioProvider {
    fn available_models(&self) -> Vec<String> {
        Vec::new()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::LMStudio
    }
}

fn looks_like_embedding_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("embed") || model.contains("embedding")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_lmstudio_output() {
        // Test special token removal with JSON extraction
        let input = "<|channel|>commentary to=echo <|constrain|>json<|message|>{\"text\":\"Your name is Alex.\"}";
        let expected = "Your name is Alex.";
        assert_eq!(clean_lmstudio_output(input), expected);

        // Test simple text (no cleaning needed)
        let input = "Hello, world!";
        assert_eq!(clean_lmstudio_output(input), "Hello, world!");

        // Test JSON text extraction
        let input = "{\"text\":\"This is the answer\"}";
        assert_eq!(clean_lmstudio_output(input), "This is the answer");

        // Test partial token removal (whitespace preserved)
        let input = "<|channel|>Hello <|message|>there";
        assert_eq!(clean_lmstudio_output(input), "Hello there");

        // Test whitespace preservation (important for streaming!)
        // Spaces should NOT be trimmed since they're meaningful between tokens
        let input = "  <|channel|>commentary  Hello  ";
        assert_eq!(clean_lmstudio_output(input), "    Hello  ");

        // Test im_start/im_end tokens
        let input = "<|im_start|>assistant\nHello<|im_end|>";
        assert_eq!(clean_lmstudio_output(input), "assistant\nHello");

        // Test that leading space is preserved (common in streaming tokens)
        let input = " How";
        assert_eq!(clean_lmstudio_output(input), " How");

        // Test that trailing space is preserved
        let input = "word ";
        assert_eq!(clean_lmstudio_output(input), "word ");
    }
}
