//! Ollama provider configuration.

use crate::llm::config::CommonProviderOptions;
use crate::llm::models::{ModelProvider, ProviderType};
use crate::llm::openai_compat::{OpenAICompatConfig, OpenAICompatibleProvider};
use crate::llm::provider_utils::first_token::{LocalStreamLimits, ModelLoadProbe, server_root};
use serde::Deserialize;

const OLLAMA_DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

/// Response structure for Ollama's /api/tags endpoint
#[derive(Debug, Deserialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

/// Individual model info from Ollama API
#[derive(Debug, Deserialize)]
struct OllamaModelInfo {
    name: String,
}

/// Response from Ollama /api/show endpoint (model details)
#[derive(Debug, Deserialize)]
struct OllamaShowResponse {
    #[serde(default)]
    model_info: Option<OllamaModelParameters>,
}

/// Model parameters from Ollama /api/show (nested under `model_info`)
#[derive(Debug, Deserialize)]
struct OllamaModelParameters {
    #[serde(default, alias = "context_length")]
    context_length: Option<usize>,
}

/// Configuration for Ollama provider.
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub base_url: String,
    pub model: String,
    pub embedding_model: Option<String>,
    /// Common provider options
    pub common: CommonProviderOptions,
    /// First-token and stall limits for streamed replies.
    pub stream_limits: LocalStreamLimits,
}

impl OllamaConfig {
    /// Create a new Ollama configuration.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model: model.into(),
            embedding_model: None,
            common: CommonProviderOptions::default(),
            stream_limits: LocalStreamLimits::default(),
        }
    }

    /// Create a builder for Ollama configuration.
    #[must_use]
    pub fn builder() -> OllamaConfigBuilder {
        OllamaConfigBuilder::default()
    }
}

impl OpenAICompatConfig for OllamaConfig {
    fn first_token_timeout(&self) -> Option<std::time::Duration> {
        Some(self.stream_limits.first_token())
    }

    fn stall_timeout(&self) -> std::time::Duration {
        self.stream_limits.stall()
    }

    fn model_load_probe(&self) -> Option<ModelLoadProbe> {
        Some(ModelLoadProbe::Ollama {
            url: format!("{}/api/ps", server_root(&self.base_url)),
            model: self.model.clone(),
        })
    }
    fn provider_name(&self) -> &'static str {
        "ollama"
    }

    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn auth_headers(&self) -> Vec<(&'static str, String)> {
        vec![] // Ollama doesn't require authentication by default
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn default_embedding_model(&self) -> Option<String> {
        self.embedding_model.clone().or_else(|| {
            std::env::var("OLLAMA_EMBEDDING_MODEL")
                .ok()
                .or_else(|| Some("nomic-embed-text".to_string()))
        })
    }

    /// Enable usage tracking in streaming responses.
    /// Ollama supports OpenAI-compatible streaming with usage data.
    fn include_usage_in_stream(&self) -> bool {
        true
    }

    /// Convert `tool_choice` to Ollama format.
    /// Ollama only supports string values: "none", "auto", "required".
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
            // Ollama doesn't support specific tool selection.
            // Convert to "auto" instead of "required" to allow the model to provide a final text response.
            // The system prompt can guide it to use the specific tool when needed.
            crate::messages::ToolChoice::Auto | crate::messages::ToolChoice::Specific(_) => {
                Some(serde_json::json!("auto"))
            }
        }
    }
}

/// Builder for Ollama configuration.
#[derive(Debug, Default)]
pub struct OllamaConfigBuilder {
    base_url: Option<String>,
    model: Option<String>,
    embedding_model: Option<String>,
    common: CommonProviderOptions,
    stream_limits: LocalStreamLimits,
}

impl OllamaConfigBuilder {
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
    pub fn build(self) -> anyhow::Result<OpenAICompatibleProvider<OllamaConfig>> {
        let base_url = self.base_url.unwrap_or_else(|| {
            crate::llm::provider_utils::read_base_url_from_env(
                "OLLAMA_BASE_URL",
                OLLAMA_DEFAULT_BASE_URL,
            )
        });

        let model = self
            .model
            .or_else(|| std::env::var("OLLAMA_MODEL").ok())
            .filter(|model| !model.trim().is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Ollama model is required. Set OLLAMA_MODEL, pass .model(...), or use OllamaProvider::from_env(None) for live discovery."
                )
            })?;

        // Read embedding model from env if not set programmatically
        let embedding_model = self
            .embedding_model
            .or_else(|| std::env::var("OLLAMA_EMBEDDING_MODEL").ok());

        let config = OllamaConfig {
            base_url,
            model,
            embedding_model,
            common: self.common,
            stream_limits: self.stream_limits,
        };
        OpenAICompatibleProvider::with_options(config.clone(), config.common)
    }
}

pub type OllamaProvider = OpenAICompatibleProvider<OllamaConfig>;

impl OllamaProvider {
    /// Create a new Ollama provider, returning validation errors explicitly.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new(base_url: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let config = OllamaConfig::new(base_url, model);
        Self::with_defaults(config)
    }

    /// Create an Ollama provider from configuration.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn with_config(config: OllamaConfig) -> anyhow::Result<Self> {
        Self::with_options(config.clone(), config.common)
    }

    /// Create an Ollama provider from environment variables.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        let base_url = crate::llm::provider_utils::read_base_url_from_env(
            "OLLAMA_BASE_URL",
            OLLAMA_DEFAULT_BASE_URL,
        );
        let model = match model.or_else(|| std::env::var("OLLAMA_MODEL").ok()) {
            Some(model) if !model.trim().is_empty() => model,
            _ => Self::discover_default_model(&base_url).await?,
        };
        let builder = OllamaConfig::builder().base_url(base_url).model(model);
        builder.build()
    }

    /// Discover the first model currently installed in Ollama.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn discover_default_model(base_url: &str) -> anyhow::Result<String> {
        let discovery_provider = Self::try_new(base_url.to_string(), "model-discovery")?;
        let models = discovery_provider.fetch_available_models().await?;
        models.into_iter().next().ok_or_else(|| {
            anyhow::anyhow!(
                "No Ollama models discovered at {base_url}. Run `ollama pull <model>` or set OLLAMA_MODEL."
            )
        })
    }

    /// Strip an OpenAI-compat `/v1` suffix (with optional trailing slash) to get
    /// the native Ollama API root used by `/api/tags` and `/api/show`.
    #[must_use]
    pub fn native_api_base(base_url: &str) -> &str {
        server_root(base_url)
    }

    /// Check if Ollama is available at the given base URL.
    /// Makes a quick GET request to /api/tags endpoint.
    pub async fn is_available(base_url: Option<&str>) -> bool {
        let base = base_url.unwrap_or(OLLAMA_DEFAULT_BASE_URL);
        let api_base_url = Self::native_api_base(base);

        let Ok(client) = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(500))
            .build()
        else {
            return false;
        };

        let url = format!("{api_base_url}/api/tags");

        (client.get(&url).send().await).map_or(false, |resp| resp.status().is_success())
    }

    /// Fetch the context window size for the active model from Ollama's /api/show endpoint.
    pub async fn fetch_context_window(&self) -> Option<usize> {
        if let Some(existing) = self.get_dynamic_context_window() {
            return Some(existing);
        }

        let api_base_url = Self::native_api_base(self.config().base_url());

        let client = self.http_client();
        let model_name = self.config().model.as_str();

        let body = serde_json::json!({ "name": model_name });
        let response = client
            .post(format!("{api_base_url}/api/show"))
            .json(&body)
            .send()
            .await
            .ok()?;

        if !response.status().is_success() {
            return None;
        }

        let show_resp: OllamaShowResponse = response.json().await.ok()?;
        if let Some(params) = show_resp.model_info
            && let Some(ctx_len) = params.context_length
        {
            tracing::debug!(
                "Ollama: model {} has context_length={}",
                model_name,
                ctx_len
            );
            self.set_dynamic_context_window(ctx_len);
            return Some(ctx_len);
        }

        None
    }

    /// Fetch available models from Ollama's API.
    /// Calls the /api/tags endpoint to get list of installed models.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<String>> {
        let fallback_models = Vec::new();

        // Extract base URL without /v1 suffix for /api/tags endpoint
        let api_base_url = self
            .config()
            .base_url()
            .strip_suffix("/v1")
            .unwrap_or_else(|| self.config().base_url());

        let client = self.http_client();
        let url = format!("{api_base_url}/api/tags");

        let response = match client.get(&url).send().await {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!("Failed to fetch Ollama models: {}, using fallback", e);
                return Ok(fallback_models);
            }
        };

        if !response.status().is_success() {
            tracing::debug!(
                "Ollama /api/tags returned status {}, using fallback",
                response.status()
            );
            return Ok(fallback_models);
        }

        match response.json::<OllamaTagsResponse>().await {
            Ok(tags_response) => {
                let models: Vec<String> =
                    tags_response.models.into_iter().map(|m| m.name).collect();

                if models.is_empty() {
                    tracing::debug!("Ollama returned no models, using fallback");
                    return Ok(fallback_models);
                }

                tracing::debug!("Fetched {} models from Ollama API", models.len());
                Ok(models)
            }
            Err(e) => {
                tracing::debug!("Failed to parse Ollama models: {}, using fallback", e);
                Ok(fallback_models)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_new_builds_provider() {
        let provider = OllamaProvider::try_new(
            "http://localhost:11434/v1".to_string(),
            "llama3.2".to_string(),
        )
        .expect("ollama provider should build");

        assert_eq!(provider.config().base_url, "http://localhost:11434/v1");
        assert_eq!(provider.config().model, "llama3.2");
    }

    #[test]
    fn native_api_base_strips_v1_with_trailing_slash() {
        assert_eq!(
            OllamaProvider::native_api_base("http://localhost:11434/v1/"),
            "http://localhost:11434"
        );
        assert_eq!(
            OllamaProvider::native_api_base("http://localhost:11434/v1"),
            "http://localhost:11434"
        );
        assert_eq!(
            OllamaProvider::native_api_base("http://localhost:11434"),
            "http://localhost:11434"
        );
    }

    #[test]
    fn try_new_accepts_trailing_slash_base_url() {
        let provider = OllamaProvider::try_new(
            "http://localhost:11434/v1/".to_string(),
            "llama3.2".to_string(),
        )
        .expect("provider should build with trailing slash");
        assert_eq!(
            OllamaProvider::native_api_base(provider.config().base_url()),
            "http://localhost:11434"
        );
    }
}

impl ModelProvider for OllamaProvider {
    fn available_models(&self) -> Vec<String> {
        Vec::new()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Ollama
    }
}
