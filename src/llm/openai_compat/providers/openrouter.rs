//! `OpenRouter` provider configuration.

use crate::llm::config::CommonProviderOptions;
use crate::llm::models::{ModelProvider, OpenRouterModel, ProviderType};
use crate::llm::openai_compat::{OpenAICompatConfig, OpenAICompatibleProvider};
use serde::Deserialize;

const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";

/// Response structure for `OpenRouter`'s /models endpoint
#[derive(Debug, Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModelInfo>,
}

/// Individual model info from `OpenRouter` API
#[derive(Debug, Deserialize)]
struct OpenRouterModelInfo {
    id: String,
    #[serde(default)]
    context_length: Option<usize>,
}

/// Configuration for `OpenRouter` provider.
#[derive(Debug, Clone)]
pub struct OpenRouterConfig {
    pub api_key: String,
    pub model: String,
}

impl OpenRouterConfig {
    /// Create a new `OpenRouter` configuration.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// Create a builder for `OpenRouter` configuration.
    #[must_use]
    pub fn builder() -> OpenRouterConfigBuilder {
        OpenRouterConfigBuilder::default()
    }
}

impl OpenAICompatConfig for OpenRouterConfig {
    fn provider_name(&self) -> &'static str {
        "openrouter"
    }

    fn base_url(&self) -> &str {
        OPENROUTER_BASE_URL
    }

    fn auth_headers(&self) -> Vec<(&'static str, String)> {
        vec![("Authorization", format!("Bearer {}", self.api_key))]
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn extra_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![
            ("HTTP-Referer", crate::llm::OPENROUTER_HTTP_REFERER),
            ("X-Title", crate::llm::OPENROUTER_X_TITLE),
        ]
    }

    fn include_usage_in_stream(&self) -> bool {
        true
    }

    fn needs_usage_parameter(&self) -> bool {
        true // OpenRouter requires the 'usage' parameter
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn default_embedding_model(&self) -> Option<String> {
        if self.model.contains("embed") || self.model.contains("embedding") {
            Some(self.model.clone())
        } else {
            std::env::var("OPENROUTER_EMBEDDING_MODEL")
                .ok()
                .or_else(|| Some("openai/text-embedding-3-small".to_string()))
        }
    }

    fn supports_vision(&self) -> bool {
        true
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
            "audio/wav",
            "audio/mpeg",
            "application/pdf",
            "text/plain",
        ]
    }

    fn enhance_error(&self, status: u16, body: &str) -> Option<String> {
        if status == 404
            && (body.contains("data policy")
                || body.contains("No endpoints found matching your data policy"))
        {
            return Some(format!(
                "{body}\n\n\
                 This error occurs when using free models that require specific data policy settings.\n\
                 To fix this:\n\
                 1. Visit https://openrouter.ai/settings/privacy\n\
                 2. Configure your data policy to allow free model publication\n\
                 3. Or use a non-free model variant (remove ':free' suffix)"
            ));
        }
        None
    }
}

/// Builder for `OpenRouter` configuration.
#[derive(Debug, Default)]
pub struct OpenRouterConfigBuilder {
    api_key: Option<String>,
    model: Option<String>,
    common: CommonProviderOptions,
}

impl OpenRouterConfigBuilder {
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
    pub fn build(self) -> anyhow::Result<OpenAICompatibleProvider<OpenRouterConfig>> {
        let api_key = crate::llm::provider_utils::resolve_api_key(
            self.api_key,
            "OPENROUTER_API_KEY",
            "api_key is required or OPENROUTER_API_KEY must be set",
        )?;

        let model = crate::llm::provider_utils::resolve_model(
            self.model,
            "OPENROUTER_MODEL",
            "anthropic/claude-sonnet-4",
        );

        let config = OpenRouterConfig { api_key, model };
        OpenAICompatibleProvider::with_options(config, self.common)
    }
}

// Canonical ergonomic provider name for OpenRouter configuration.
pub type OpenRouterProvider = OpenAICompatibleProvider<OpenRouterConfig>;

impl OpenRouterProvider {
    /// Create a new `OpenRouter` provider, returning validation errors explicitly.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new(api_key: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let config = OpenRouterConfig::new(api_key, model);
        Self::with_defaults(config)
    }

    /// Create an `OpenRouter` provider from environment variables.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        std::future::ready(()).await;
        let mut builder = OpenRouterConfig::builder();
        if let Some(m) = model {
            builder = builder.model(m);
        }
        builder.build()
    }

    /// Fetch available models from `OpenRouter`'s API.
    /// Returns model IDs sorted by popularity/relevance.
    /// Also stores the context window for the active model when available.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<String>> {
        let fallback_models: Vec<String> = OpenRouterModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect();

        let client = self.http_client();

        let response = match client
            .get(format!("{OPENROUTER_BASE_URL}/models"))
            .header("Authorization", format!("Bearer {}", self.config().api_key))
            .header("HTTP-Referer", crate::llm::OPENROUTER_HTTP_REFERER)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!("Failed to fetch OpenRouter models: {}, using fallback", e);
                return Ok(fallback_models);
            }
        };

        if !response.status().is_success() {
            tracing::debug!(
                "OpenRouter /models returned status {}, using fallback",
                response.status()
            );
            return Ok(fallback_models);
        }

        match response.json::<OpenRouterModelsResponse>().await {
            Ok(models_response) => {
                let active_model = self.config().model.as_str();
                for model_info in &models_response.data {
                    if model_info.id == active_model {
                        if let Some(ctx_len) = model_info.context_length {
                            tracing::debug!(
                                "OpenRouter: model {} has context_length={}",
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
                    tracing::debug!("OpenRouter returned no models, using fallback");
                    Ok(fallback_models)
                } else {
                    tracing::debug!("Fetched {} models from OpenRouter API", models.len());
                    Ok(models)
                }
            }
            Err(e) => {
                tracing::debug!("Failed to parse OpenRouter models: {}, using fallback", e);
                Ok(fallback_models)
            }
        }
    }

    /// Fetch context window size for the active model from `OpenRouter`'s API.
    /// This is a lightweight alternative to `fetch_available_models` when you
    /// only need the context window.
    pub async fn fetch_context_window(&self) -> Option<usize> {
        if let Some(existing) = self.get_dynamic_context_window() {
            return Some(existing);
        }

        let client = self.http_client();
        let active_model = self.config().model.as_str();

        let response = client
            .get(format!("{OPENROUTER_BASE_URL}/models"))
            .header("Authorization", format!("Bearer {}", self.config().api_key))
            .header("HTTP-Referer", crate::llm::OPENROUTER_HTTP_REFERER)
            .send()
            .await
            .map_err(|error| {
                tracing::debug!(error = %error, "Failed to fetch OpenRouter model metadata");
                error
            })
            .ok()?;

        if !response.status().is_success() {
            return None;
        }

        let models_response: OpenRouterModelsResponse = response.json().await.ok()?;

        for model_info in &models_response.data {
            if model_info.id == active_model
                && let Some(ctx_len) = model_info.context_length
            {
                self.set_dynamic_context_window(ctx_len);
                return Some(ctx_len);
            }
        }

        None
    }
}

impl ModelProvider for OpenRouterProvider {
    fn available_models(&self) -> Vec<String> {
        OpenRouterModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenRouter
    }
}
