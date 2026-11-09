//! `OpenAI` provider configuration.

use crate::llm::config::CommonProviderOptions;
use crate::llm::models::{ModelProvider, OpenAIModel, ProviderType};
use crate::llm::openai_compat::{OpenAICompatConfig, OpenAICompatibleProvider};
use serde::Deserialize;

const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// Response structure for `OpenAI`'s /models endpoint
#[derive(Debug, Deserialize)]
struct OpenAIModelsResponse {
    data: Vec<OpenAIModelInfo>,
}

/// Individual model info from `OpenAI` API
#[derive(Debug, Deserialize)]
struct OpenAIModelInfo {
    id: String,
}

/// Configuration for `OpenAI` provider.
#[derive(Debug, Clone)]
pub struct OpenAIConfig {
    pub api_key: String,
    pub model: String,
}

impl OpenAIConfig {
    /// Create a new `OpenAI` configuration.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
        }
    }

    /// Create a builder for `OpenAI` configuration.
    #[must_use]
    pub fn builder() -> OpenAIConfigBuilder {
        OpenAIConfigBuilder::default()
    }
}

impl OpenAICompatConfig for OpenAIConfig {
    fn provider_name(&self) -> &'static str {
        "openai"
    }

    fn base_url(&self) -> &str {
        OPENAI_BASE_URL
    }

    fn auth_headers(&self) -> Vec<(&'static str, String)> {
        vec![("Authorization", format!("Bearer {}", self.api_key))]
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn uses_max_completion_tokens(&self) -> bool {
        self.model.starts_with("gpt-4o")
            || self.model.starts_with("gpt-4.1")
            || self.model.starts_with("gpt-4.5")
            || self.model.starts_with("o1")
            || self.model.starts_with("o3")
            || self.model.starts_with("o4")
            || self.model.starts_with("gpt-5")
            || self.model.starts_with("chatgpt-4o")
    }

    fn requires_default_temperature(&self) -> bool {
        self.model.starts_with("o1")
            || self.model.starts_with("o3")
            || self.model.starts_with("o4")
            || self.model.starts_with("gpt-5")
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn default_embedding_model(&self) -> Option<String> {
        Some("text-embedding-3-small".to_string())
    }

    fn include_usage_in_stream(&self) -> bool {
        true // OpenAI supports stream_options: { include_usage: true }
    }

    fn supports_vision(&self) -> bool {
        self.model.starts_with("gpt-4o")
            || self.model.starts_with("gpt-4.1")
            || self.model.starts_with("gpt-4-turbo")
    }

    fn supports_audio_input(&self) -> bool {
        self.model.starts_with("gpt-4o") || self.model.starts_with("gpt-4.1")
    }

    fn supported_media_types(&self) -> Vec<&str> {
        vec![
            "image/png",
            "image/jpeg",
            "image/webp",
            "audio/wav",
            "audio/mpeg",
        ]
    }
}

/// Builder for `OpenAI` configuration.
#[derive(Debug, Default)]
pub struct OpenAIConfigBuilder {
    api_key: Option<String>,
    model: Option<String>,
    common: CommonProviderOptions,
}

impl OpenAIConfigBuilder {
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

    /// Set retry strategy.
    #[must_use]
    pub fn retry_strategy(
        mut self,
        strategy: std::sync::Arc<dyn crate::traits::retry::RetryStrategy>,
    ) -> Self {
        self.common = self.common.with_retry_strategy(strategy);
        self
    }

    /// Use a shared HTTP client for connection pooling across multiple agents.
    #[must_use]
    pub fn shared_http_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.common = self.common.with_client(client);
        self
    }

    /// Build the configuration and create a provider.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn build(self) -> anyhow::Result<OpenAICompatibleProvider<OpenAIConfig>> {
        let api_key = crate::llm::provider_utils::resolve_api_key(
            self.api_key,
            "OPENAI_API_KEY",
            "api_key is required or OPENAI_API_KEY must be set",
        )?;

        let model = crate::llm::provider_utils::resolve_model(
            self.model,
            "OPENAI_MODEL",
            crate::llm::defaults::OPENAI,
        );

        let config = OpenAIConfig { api_key, model };
        OpenAICompatibleProvider::with_options(config, self.common)
    }
}

// Canonical ergonomic provider name for OpenAI-compatible configuration.
pub type OpenAIProvider = OpenAICompatibleProvider<OpenAIConfig>;

impl OpenAIProvider {
    /// Create a new `OpenAI` provider, returning validation errors explicitly.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new(api_key: impl Into<String>, model: impl Into<String>) -> anyhow::Result<Self> {
        let model = model.into();
        let model = model.trim();
        if model.is_empty() {
            anyhow::bail!("OpenAI model must be non-empty");
        }
        let config = OpenAIConfig::new(api_key, model);
        Self::with_defaults(config)
    }

    /// Create an `OpenAI` provider from environment variables.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env(model: Option<String>) -> anyhow::Result<Self> {
        std::future::ready(()).await;
        let mut builder = OpenAIConfig::builder();
        if let Some(m) = model {
            builder = builder.model(m);
        }
        builder.build()
    }

    /// Fetch available models from `OpenAI`'s API.
    /// Filters to chat/completion models and sorts by relevance.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<String>> {
        let fallback_models: Vec<String> = OpenAIModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect();

        let client = self.http_client();

        let response = match client
            .get(format!("{OPENAI_BASE_URL}/models"))
            .header("Authorization", format!("Bearer {}", self.config().api_key))
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                tracing::debug!("Failed to fetch OpenAI models: {}, using fallback", e);
                return Ok(fallback_models);
            }
        };

        if !response.status().is_success() {
            tracing::debug!(
                "OpenAI /models returned status {}, using fallback",
                response.status()
            );
            return Ok(fallback_models);
        }

        match response.json::<OpenAIModelsResponse>().await {
            Ok(models_response) => {
                // Filter to chat/completion models (gpt-*, o1-*, o3-*, chatgpt-*)
                // Exclude embedding, tts, whisper, dall-e, etc.
                let mut models: Vec<String> = models_response
                    .data
                    .into_iter()
                    .filter(|m| {
                        let id = m.id.as_str();
                        (id.starts_with("gpt-") 
                            || id.starts_with("o1") 
                            || id.starts_with("o3")
                            || id.starts_with("chatgpt-"))
                        && !id.contains("instruct")
                        && !id.contains("vision")  // vision is deprecated
                        && !id.contains("0301")    // old snapshots
                        && !id.contains("0314")
                        && !id.contains("0613")
                    })
                    .map(|m| m.id)
                    .collect();

                if models.is_empty() {
                    tracing::debug!("OpenAI returned no chat models, using fallback");
                    return Ok(fallback_models);
                }

                // Sort by preference (latest models first)
                models.sort_by(|a, b| {
                    let score_a = model_priority_score(a);
                    let score_b = model_priority_score(b);
                    score_b.cmp(&score_a)
                });

                tracing::debug!("Fetched {} chat models from OpenAI API", models.len());
                Ok(models)
            }
            Err(e) => {
                tracing::debug!("Failed to parse OpenAI models: {}, using fallback", e);
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
        let provider = OpenAIProvider::try_new("test-key".to_string(), "gpt-4o".to_string())
            .expect("provider should build");

        assert_eq!(provider.config().api_key, "test-key");
        assert_eq!(provider.config().model, "gpt-4o");
    }

    #[test]
    fn try_new_rejects_empty_or_whitespace_model() {
        let err = OpenAIProvider::try_new("test-key", "").expect_err("empty model should fail");
        assert!(err.to_string().contains("non-empty"));
        let err =
            OpenAIProvider::try_new("test-key", "   ").expect_err("whitespace model should fail");
        assert!(err.to_string().contains("non-empty"));
    }
}

/// Score models by preference (higher = more preferred)
fn model_priority_score(model: &str) -> i32 {
    gpt_41_priority(model)
        .or_else(|| gpt_4o_priority(model))
        .or_else(|| o_series_priority(model))
        .or_else(|| gpt_legacy_priority(model))
        .unwrap_or(10)
}

fn gpt_41_priority(model: &str) -> Option<i32> {
    if model.starts_with("gpt-4.1") && !model.contains("mini") && !model.contains("nano") {
        Some(110)
    } else if model.starts_with("gpt-4.1-mini") {
        Some(105)
    } else if model.starts_with("gpt-4.1-nano") {
        Some(103)
    } else {
        None
    }
}

fn gpt_4o_priority(model: &str) -> Option<i32> {
    if model.starts_with("gpt-4o") && !model.contains("mini") {
        Some(100)
    } else if model.starts_with("gpt-4o-mini") {
        Some(95)
    } else {
        None
    }
}

fn o_series_priority(model: &str) -> Option<i32> {
    if model.starts_with("o4") && !model.contains("mini") {
        Some(92)
    } else if model.starts_with("o4-mini") {
        Some(91)
    } else if model.starts_with("o3") && !model.contains("mini") {
        Some(90)
    } else if model.starts_with("o3-mini") {
        Some(85)
    } else if model.starts_with("o1") && !model.contains("mini") && !model.contains("preview") {
        Some(80)
    } else if model.starts_with("o1-mini") {
        Some(75)
    } else if model.starts_with("o1-preview") {
        Some(70)
    } else {
        None
    }
}

fn gpt_legacy_priority(model: &str) -> Option<i32> {
    if model.starts_with("gpt-4.5") {
        Some(65)
    } else if model.starts_with("gpt-4-turbo") {
        Some(60)
    } else if model.starts_with("gpt-4") && !model.contains("turbo") {
        Some(50)
    } else if model.starts_with("gpt-3.5") {
        Some(30)
    } else if model.starts_with("chatgpt") {
        Some(20)
    } else {
        None
    }
}

impl ModelProvider for OpenAIProvider {
    fn available_models(&self) -> Vec<String> {
        OpenAIModel::all()
            .iter()
            .map(|m| m.as_str().to_owned())
            .collect()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenAI
    }
}
