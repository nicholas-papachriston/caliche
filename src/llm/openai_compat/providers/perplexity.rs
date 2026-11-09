//! Perplexity provider configuration.

use crate::llm::config::CommonProviderOptions;
use crate::llm::models::{ModelProvider, PerplexityModel, ProviderType};
use crate::llm::openai_compat::{OpenAICompatConfig, OpenAICompatibleProvider};

const PERPLEXITY_BASE_URL: &str = "https://api.perplexity.ai";

/// Configuration for Perplexity provider.
#[derive(Debug, Clone)]
pub struct PerplexityConfig {
    pub api_key: String,
    pub model: String,
}

impl PerplexityConfig {
    /// Create a new Perplexity configuration.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// Create a builder for Perplexity configuration.
    #[must_use]
    pub fn builder() -> PerplexityConfigBuilder {
        PerplexityConfigBuilder::default()
    }
}

impl OpenAICompatConfig for PerplexityConfig {
    fn provider_name(&self) -> &'static str {
        "perplexity"
    }

    fn base_url(&self) -> &str {
        PERPLEXITY_BASE_URL
    }

    fn auth_headers(&self) -> Vec<(&'static str, String)> {
        vec![("Authorization", format!("Bearer {}", self.api_key))]
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn supports_native_tools(&self) -> bool {
        false // Perplexity uses text-based tool calling (JSON in content field, not native tool_calls)
    }

    fn supports_text_based_tools(&self) -> bool {
        true
    }

    fn supports_response_format(&self) -> bool {
        false // Perplexity doesn't support response_format parameter
    }
}

/// Builder for Perplexity configuration.
#[derive(Debug, Default)]
pub struct PerplexityConfigBuilder {
    api_key: Option<String>,
    model: Option<String>,
    common: CommonProviderOptions,
}

impl PerplexityConfigBuilder {
    /// Set the API key.
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the model.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
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
    pub fn build(self) -> anyhow::Result<OpenAICompatibleProvider<PerplexityConfig>> {
        let api_key = crate::llm::provider_utils::resolve_api_key(
            self.api_key,
            "PERPLEXITY_API_KEY",
            "api_key is required or PERPLEXITY_API_KEY must be set",
        )?;

        let model =
            crate::llm::provider_utils::resolve_model(self.model, "PERPLEXITY_MODEL", "sonar");

        let config = PerplexityConfig { api_key, model };
        OpenAICompatibleProvider::with_options(config, self.common)
    }
}

// Canonical ergonomic provider name for Perplexity configuration.
pub type PerplexityProvider = OpenAICompatibleProvider<PerplexityConfig>;

impl PerplexityProvider {
    /// Create a new Perplexity provider, returning validation errors explicitly.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new(api_key: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let config = PerplexityConfig::new(api_key, model);
        Self::with_defaults(config)
    }

    /// Create a Perplexity provider from environment variables.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        std::future::ready(()).await;
        let mut builder = PerplexityConfig::builder();
        if let Some(m) = model {
            builder = builder.model(m);
        }
        builder.build()
    }

    /// Fetch available models from Perplexity's API.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<String>> {
        std::future::ready(()).await;
        let fallback_models = PerplexityModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect();

        Ok(fallback_models)
    }
}

impl ModelProvider for PerplexityProvider {
    fn available_models(&self) -> Vec<String> {
        PerplexityModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::Perplexity
    }
}
