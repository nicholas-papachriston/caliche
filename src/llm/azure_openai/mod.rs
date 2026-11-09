//! Azure `OpenAI` provider — thin wrapper over [`OpenAICompatibleProvider`].

mod auth;
mod config;

use config::AzureOpenAICompatConfig;

pub use auth::AzureAuth;
pub use config::{AzureOpenAIProviderConfig, AzureOpenAIProviderConfigBuilder};

use crate::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use crate::llm::models::{ModelProvider, ProviderType};
use crate::llm::openai_compat::{OpenAICompatConfig, OpenAICompatibleProvider};
use crate::llm::provider::Provider;
use crate::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use async_trait::async_trait;
use futures_util::Stream;
use std::pin::Pin;

type AzureOpenAIInnerProvider = OpenAICompatibleProvider<AzureOpenAICompatConfig>;

/// Azure `OpenAI` provider delegating chat and embedding requests to the shared
/// OpenAI-compatible implementation.
pub struct AzureOpenAIProvider {
    inner: AzureOpenAIInnerProvider,
}

impl std::fmt::Debug for AzureOpenAIProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAIProvider")
            .field("inner", &self.inner)
            .finish()
    }
}

impl AzureOpenAIProvider {
    /// Create a provider from [`AzureOpenAIProviderConfig`].
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn with_config(config: AzureOpenAIProviderConfig) -> anyhow::Result<Self> {
        let (compat, common) = AzureOpenAICompatConfig::from_provider_config_with_common(config);
        let inner = AzureOpenAIInnerProvider::with_options(compat, common)?;
        Ok(Self { inner })
    }

    /// Create a provider with API key authentication.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn try_new(
        resource_name: String,
        deployment_id: String,
        api_key: String,
    ) -> anyhow::Result<Self> {
        let config = AzureOpenAIProviderConfig::new_api_key(resource_name, deployment_id, api_key);
        Self::with_config(config)
    }

    /// Create a provider from environment variables.
    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub async fn from_env(deployment_id: Option<String>) -> anyhow::Result<Self> {
        std::future::ready(()).await;
        let mut builder = AzureOpenAIProviderConfig::builder();
        if let Some(deployment_id) = deployment_id {
            builder = builder.deployment_id(deployment_id);
        }
        let config = builder.build()?;
        Self::with_config(config)
    }

    /// Deployment ID configured for this provider.
    #[must_use]
    pub fn deployment_id(&self) -> &str {
        self.inner.config().deployment_id()
    }

    /// OpenAI-compat configuration surface for integration tests.
    #[doc(hidden)]
    #[must_use]
    pub fn openai_compat(&self) -> &impl OpenAICompatConfig {
        self.inner.config()
    }
}

#[async_trait]
impl Provider for AzureOpenAIProvider {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        self.inner.complete(request).await
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        self.inner.stream(request)
    }

    fn name(&self) -> &str {
        Provider::name(&self.inner)
    }

    fn supports_native_tools(&self) -> bool {
        self.inner.supports_native_tools()
    }

    fn supports_text_based_tools(&self) -> bool {
        self.inner.supports_text_based_tools()
    }

    fn supports_vision(&self) -> bool {
        self.inner.supports_vision()
    }

    fn supports_audio_input(&self) -> bool {
        self.inner.supports_audio_input()
    }

    fn supports_document_input(&self) -> bool {
        self.inner.supports_document_input()
    }

    fn supported_media_types(&self) -> Vec<&str> {
        self.inner.supported_media_types()
    }

    fn context_window(&self) -> Option<usize> {
        self.inner.context_window()
    }
}

impl ModelProvider for AzureOpenAIProvider {
    fn available_models(&self) -> Vec<String> {
        vec![self.deployment_id().to_owned()]
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::AzureOpenAI
    }
}

#[async_trait]
impl EmbeddingProvider for AzureOpenAIProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        self.inner.embed(request).await
    }

    fn name(&self) -> &str {
        EmbeddingProvider::name(&self.inner)
    }
}
