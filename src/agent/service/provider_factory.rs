//! Provider factory for creating LLM provider instances.
//!
//! This module provides a centralized factory for initializing LLM providers with
//! credential management. It handles the complexity of different authentication
//! mechanisms (API keys, AWS credentials, base URLs) and provides a consistent
//! interface for provider creation.
//!
//! # Examples
//!
//! ```no_run
//! use caliche::agent::service::ProviderFactory;
//! use caliche::llm::ProviderType;
//! use caliche::traits::credentials::{CredentialProvider, StaticCredentialProvider, Credentials};
//! use std::sync::Arc;
//! use std::time::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create provider with credential provider
//! let cred_provider = Arc::new(StaticCredentialProvider::new(
//!     Credentials::ApiKey("sk-...".to_string())
//! ));
//! let provider = ProviderFactory::create(
//!     ProviderType::OpenAI,
//!     "gpt-4o".to_string(),
//!     cred_provider,
//!     None,
//!     Some(Duration::from_secs(30)),
//!     Some(3),
//!     None,
//! ).await?;
//!
//! // Create provider from environment variables
//! let provider = ProviderFactory::create_from_env(
//!     ProviderType::OpenAI,
//!     "gpt-4o".to_string(),
//!     Some(Duration::from_secs(30)),
//!     Some(3),
//! ).await?;
//!
//! // Create provider using builder pattern
//! let provider = ProviderFactory::builder()
//!     .provider_type(ProviderType::OpenAI)
//!     .model("gpt-4o")
//!     .timeout(Duration::from_secs(30))
//!     .max_retries(3)
//!     .build()
//!     .await?;
//! # Ok(())
//! # }
//! ```

use super::provider_build::{self, ProviderCredentialMode};
use crate::llm::anthropic::AnthropicProviderConfig;
#[cfg(feature = "bedrock")]
use crate::llm::bedrock::{BedrockProvider, BedrockProviderConfig};
use crate::llm::google::GoogleProviderConfig;
use crate::llm::openai_compat::{
    CloudflareAIGatewayConfig, CloudflareAIGatewayConfigBuilder, LMStudioConfig, OllamaConfig,
    OpenAIConfig, OpenRouterConfig, PerplexityConfig,
};
use crate::llm::{
    AnthropicProvider, CloudflareAIGatewayProvider, EmbeddingProvider, ExternalCompletionCallback,
    ExternalEmbeddingCallback, ExternalEmbeddingProvider, ExternalProvider, GoogleProvider,
    ImageGenerationProvider, LMStudioProvider, OpenAIChatImageGenerationProvider,
    OpenAIImageGenerationProvider, OpenAIProvider, OpenRouterProvider, PerplexityProvider,
    Provider, ProviderType,
};
use crate::llm::{GoogleImageGenerationProvider, resolve_image_generation_model};
use crate::traits::credentials::{CredentialProvider, Credentials};
use std::sync::Arc;
use std::time::Duration;

/// Credential resolution mode is defined in [`super::provider_build`].

/// Factory for creating LLM provider instances with credential management.
pub struct ProviderFactory;

/// Builder for `ProviderFactory` with fluent configuration API
#[derive(Debug, Default)]
pub struct ProviderFactoryBuilder {
    provider_type: Option<ProviderType>,
    model: Option<String>,
    timeout: Option<Duration>,
    max_retries: Option<usize>,
    credentials: Option<Arc<dyn CredentialProvider>>,
    base_url: Option<String>,
}

impl ProviderFactoryBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the provider type
    #[must_use]
    pub const fn provider_type(mut self, provider: ProviderType) -> Self {
        self.provider_type = Some(provider);
        self
    }

    /// Set the model name
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set request timeout
    #[must_use]
    pub const fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set maximum retries
    #[must_use]
    pub const fn max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Set credential provider
    pub fn credentials(mut self, cred_provider: Arc<dyn CredentialProvider>) -> Self {
        self.credentials = Some(cred_provider);
        self
    }

    /// Set base URL (for `LMStudio`)
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Build the provider
    pub async fn build(self) -> anyhow::Result<Arc<dyn Provider>> {
        let provider_type = self.provider_type.ok_or_else(|| {
            anyhow::anyhow!("provider_type is required for ProviderFactoryBuilder")
        })?;
        let model = self
            .model
            .ok_or_else(|| anyhow::anyhow!("model is required for ProviderFactoryBuilder"))?;

        if let Some(cred_provider) = self.credentials {
            ProviderFactory::create(
                provider_type,
                model,
                cred_provider,
                self.base_url.as_deref(),
                self.timeout,
                self.max_retries,
                #[cfg(feature = "reqwest")]
                None,
            )
            .await
        } else {
            ProviderFactory::create_from_env(provider_type, model, self.timeout, self.max_retries)
                .await
        }
    }
}

impl ProviderFactory {
    /// Create a new builder for constructing providers with fluent API
    #[must_use]
    pub fn builder() -> ProviderFactoryBuilder {
        ProviderFactoryBuilder::new()
    }

    /// Create a provider instance with provider-specific configuration
    ///
    /// This is a convenience method that accepts provider-specific config types.
    /// For type-safe configuration, use the provider's `with_config` method directly.
    #[cfg(feature = "bedrock")]
    pub async fn create_with_bedrock_config(
        config: BedrockProviderConfig,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        Ok(Arc::new(BedrockProvider::with_config(config).await?))
    }

    /// Create a Google provider with configuration
    pub fn create_with_google_config(
        config: GoogleProviderConfig,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        Ok(Arc::new(GoogleProvider::with_config(config)?))
    }

    /// Create an `LMStudio` provider with configuration
    pub fn create_with_lmstudio_config(
        config: LMStudioConfig,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        Ok(Arc::new(LMStudioProvider::with_defaults(config)?))
    }

    /// Create an Anthropic provider with configuration
    pub fn create_with_anthropic_config(
        config: AnthropicProviderConfig,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        Ok(Arc::new(AnthropicProvider::with_config(config)?))
    }

    /// Create an `OpenAI` provider with configuration
    pub fn create_with_openai_config(config: OpenAIConfig) -> anyhow::Result<Arc<dyn Provider>> {
        Ok(Arc::new(OpenAIProvider::with_defaults(config)?))
    }

    /// Create an `OpenRouter` provider with configuration
    pub fn create_with_openrouter_config(
        config: OpenRouterConfig,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        Ok(Arc::new(OpenRouterProvider::with_defaults(config)?))
    }

    /// Create a Cloudflare AI Gateway provider with configuration
    pub fn create_with_cloudflare_ai_gateway_config(
        config: CloudflareAIGatewayConfig,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        Ok(Arc::new(CloudflareAIGatewayProvider::with_defaults(
            config,
        )?))
    }

    /// Create a Perplexity provider with configuration
    pub fn create_with_perplexity_config(
        config: PerplexityConfig,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        Ok(Arc::new(PerplexityProvider::with_defaults(config)?))
    }

    pub(super) fn apply_cloudflare_credentials(
        mut builder: CloudflareAIGatewayConfigBuilder,
        creds: Credentials,
    ) -> CloudflareAIGatewayConfigBuilder {
        match creds {
            Credentials::ApiKey(key) => builder.api_key(key),
            Credentials::Custom(mut values) => {
                if let Some(api_key) = values.remove("api_key") {
                    builder = builder.api_key(api_key);
                }
                if let Some(account_id) = values.remove("account_id") {
                    builder = builder.account_id(account_id);
                }
                if let Some(gateway_id) = values.remove("gateway_id") {
                    builder = builder.gateway_id(gateway_id);
                }
                if let Some(auth_token) = values
                    .remove("cf_aig_authorization")
                    .or_else(|| values.remove("auth_token"))
                {
                    builder = builder.cf_aig_authorization(auth_token);
                }
                builder
            }
            _ => builder,
        }
    }

    fn cloudflare_image_config(
        creds: Credentials,
    ) -> anyhow::Result<(String, String, String, Option<String>)> {
        let (api_key, account_id, gateway_id, auth_token) = match creds {
            Credentials::ApiKey(key) => (Some(key), None, None, None),
            Credentials::Custom(mut values) => (
                values.remove("api_key"),
                values.remove("account_id"),
                values.remove("gateway_id"),
                values
                    .remove("cf_aig_authorization")
                    .or_else(|| values.remove("auth_token")),
            ),
            _ => (None, None, None, None),
        };

        let api_key = api_key
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_API_KEY").ok())
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_API_TOKEN").ok());
        let account_id = account_id
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_ACCOUNT_ID").ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "CLOUDFLARE_ACCOUNT_ID is required for Cloudflare AI Gateway image generation"
                )
            })?;
        let gateway_id = gateway_id
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_GATEWAY_ID").ok())
            .unwrap_or_else(|| "default".to_string());
        let auth_token = auth_token
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_AUTH_TOKEN").ok());
        if api_key.as_deref().is_none_or(|key| key.trim().is_empty())
            && auth_token
                .as_deref()
                .is_none_or(|key| key.trim().is_empty())
        {
            anyhow::bail!(
                "CLOUDFLARE_AIG_API_KEY or CLOUDFLARE_AIG_AUTH_TOKEN is required for Cloudflare AI Gateway image generation"
            );
        }

        Ok((
            api_key.unwrap_or_default(),
            account_id,
            gateway_id,
            auth_token,
        ))
    }

    pub(super) fn credentials_for_builder(cred_mode: &ProviderCredentialMode<'_>) -> Credentials {
        match cred_mode {
            ProviderCredentialMode::Environment => Credentials::None,
            ProviderCredentialMode::Explicit(creds) => (*creds).clone(),
        }
    }

    /// Create a provider instance with credential provider.
    pub async fn create(
        provider: ProviderType,
        model_name: String,
        credential_provider: Arc<dyn CredentialProvider>,
        base_url: Option<&str>,
        timeout: Option<Duration>,
        max_retries: Option<usize>,
        #[cfg(feature = "reqwest")] shared_http_client: Option<std::sync::Arc<reqwest::Client>>,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        let provider_name = provider.to_string();
        let creds = {
            #[cfg(feature = "local-inference-base")]
            {
                if matches!(provider, ProviderType::LocalInference) {
                    Credentials::None
                } else {
                    credential_provider.get_credentials(&provider_name).await?
                }
            }
            #[cfg(not(feature = "local-inference-base"))]
            {
                credential_provider.get_credentials(&provider_name).await?
            }
        };

        Self::build_provider(
            provider,
            model_name,
            ProviderCredentialMode::Explicit(&creds),
            base_url,
            timeout,
            max_retries,
            #[cfg(feature = "reqwest")]
            shared_http_client,
        )
        .await
    }

    /// Unified provider construction from explicit credentials or environment only.
    async fn build_provider(
        provider: ProviderType,
        model_name: String,
        cred_mode: ProviderCredentialMode<'_>,
        base_url: Option<&str>,
        timeout: Option<Duration>,
        max_retries: Option<usize>,
        #[cfg(feature = "reqwest")] shared_http_client: Option<std::sync::Arc<reqwest::Client>>,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        provider_build::build_llm_provider(
            provider,
            model_name,
            cred_mode,
            base_url,
            timeout,
            max_retries,
            #[cfg(feature = "reqwest")]
            shared_http_client,
        )
        .await
    }

    /// Create an external provider with a completion callback
    ///
    /// This method allows Python/JS bindings to provide their own LLM implementations
    /// (e.g., `LangChain`) that will be used by Caliche.
    pub fn create_with_external_callback(
        name: impl Into<String>,
        _model_name: String,
        completion_callback: Arc<dyn ExternalCompletionCallback>,
        supports_native_tools: bool,
        supports_streaming: bool,
    ) -> Arc<dyn Provider> {
        Arc::new(ExternalProvider::new(
            name.into(),
            completion_callback,
            supports_native_tools,
            supports_streaming,
        ))
    }

    /// Create an external embedding provider with an embedding callback
    pub fn create_external_embedding_provider(
        name: impl Into<String>,
        embedding_callback: Arc<dyn ExternalEmbeddingCallback>,
    ) -> Arc<dyn EmbeddingProvider> {
        Arc::new(ExternalEmbeddingProvider::new(
            name.into(),
            embedding_callback,
        ))
    }

    /// Create a provider instance from environment variables only.
    pub async fn create_from_env(
        provider: ProviderType,
        model_name: String,
        timeout: Option<Duration>,
        max_retries: Option<usize>,
    ) -> anyhow::Result<Arc<dyn Provider>> {
        Self::build_provider(
            provider,
            model_name,
            ProviderCredentialMode::Environment,
            None,
            timeout,
            max_retries,
            #[cfg(feature = "reqwest")]
            None,
        )
        .await
    }

    /// Create an embedding provider if the LLM provider supports embeddings.
    pub async fn create_embedding_provider(
        provider: ProviderType,
        model_name: String,
        credential_provider: Arc<dyn CredentialProvider>,
    ) -> anyhow::Result<Option<Arc<dyn EmbeddingProvider>>> {
        #[cfg(feature = "local-inference-base")]
        if matches!(provider, ProviderType::LocalInference) {
            return Ok(None);
        }

        let provider_name = provider.to_string();
        let creds = credential_provider.get_credentials(&provider_name).await?;

        let embedding_provider: Option<Arc<dyn EmbeddingProvider>> = match provider {
            ProviderType::OpenAI => {
                let openai_provider = match creds {
                    Credentials::ApiKey(key) => OpenAIProvider::try_new(key, model_name)?,
                    _ => OpenAIProvider::from_env(Some(model_name)).await?,
                };
                Some(Arc::new(openai_provider) as Arc<dyn EmbeddingProvider>)
            }
            ProviderType::OpenRouter => {
                let openrouter_provider = match creds {
                    Credentials::ApiKey(key) => OpenRouterProvider::try_new(key, model_name)?,
                    _ => OpenRouterProvider::from_env(Some(model_name)).await?,
                };
                Some(Arc::new(openrouter_provider) as Arc<dyn EmbeddingProvider>)
            }
            ProviderType::CloudflareAIGateway => {
                let builder = Self::apply_cloudflare_credentials(
                    CloudflareAIGatewayConfig::builder().model(model_name),
                    creds,
                );
                Some(Arc::new(builder.build()?) as Arc<dyn EmbeddingProvider>)
            }
            ProviderType::Bedrock => bedrock_embedding(creds).await?,
            ProviderType::LMStudio => {
                let lmstudio_provider = LMStudioProvider::from_env(Some(model_name)).await?;
                Some(Arc::new(lmstudio_provider) as Arc<dyn EmbeddingProvider>)
            }
            ProviderType::Ollama => {
                let base_url = caliche_contracts::credential_env::var("OLLAMA_BASE_URL")
                    .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
                let ollama_provider = OllamaConfig::builder()
                    .base_url(base_url)
                    .model("dummy")
                    .embedding_model(model_name)
                    .build()?;
                Some(Arc::new(ollama_provider) as Arc<dyn EmbeddingProvider>)
            }
            _ => None,
        };

        Ok(embedding_provider)
    }

    /// Create an image generation provider when the selected provider supports it.
    pub async fn create_image_generation_provider(
        provider: ProviderType,
        model_name: Option<String>,
        credential_provider: Arc<dyn CredentialProvider>,
    ) -> anyhow::Result<Option<Arc<dyn ImageGenerationProvider>>> {
        let Some(model_name) = resolve_image_generation_model(provider, model_name) else {
            return Ok(None);
        };
        let provider_name = provider.to_string();
        let creds = credential_provider.get_credentials(&provider_name).await?;

        let image_provider: Option<Arc<dyn ImageGenerationProvider>> = match provider {
            ProviderType::OpenAI => {
                if let Credentials::ApiKey(key) = creds {
                    Some(
                        Arc::new(OpenAIImageGenerationProvider::new(key, model_name))
                            as Arc<dyn ImageGenerationProvider>,
                    )
                } else {
                    let key =
                        caliche_contracts::credential_env::var("OPENAI_API_KEY").map_err(|_| {
                            anyhow::anyhow!(
                                "OPENAI_API_KEY is required for OpenAI image generation"
                            )
                        })?;
                    Some(
                        Arc::new(OpenAIImageGenerationProvider::new(key, model_name))
                            as Arc<dyn ImageGenerationProvider>,
                    )
                }
            }
            ProviderType::Google => {
                if let Credentials::ApiKey(key) = creds {
                    Some(
                        Arc::new(GoogleImageGenerationProvider::new(key, model_name))
                            as Arc<dyn ImageGenerationProvider>,
                    )
                } else {
                    let key = caliche_contracts::credential_env::var("GOOGLE_API_KEY")
                    .or_else(|_| caliche_contracts::credential_env::var("GEMINI_API_KEY"))
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "GOOGLE_API_KEY or GEMINI_API_KEY is required for Google image generation"
                        )
                    })?;
                    Some(
                        Arc::new(GoogleImageGenerationProvider::new(key, model_name))
                            as Arc<dyn ImageGenerationProvider>,
                    )
                }
            }
            ProviderType::OpenRouter => {
                let key = match creds {
                    Credentials::ApiKey(key) => key,
                    _ => caliche_contracts::credential_env::var("OPENROUTER_API_KEY").map_err(
                        |_| {
                            anyhow::anyhow!(
                                "OPENROUTER_API_KEY is required for OpenRouter image generation"
                            )
                        },
                    )?,
                };
                Some(Arc::new(
                    OpenAIChatImageGenerationProvider::new(
                        "openrouter",
                        key,
                        model_name,
                        "https://openrouter.ai/api/v1",
                    )
                    .with_header("HTTP-Referer", crate::llm::OPENROUTER_HTTP_REFERER)
                    .with_header("X-Title", crate::llm::OPENROUTER_X_TITLE),
                ) as Arc<dyn ImageGenerationProvider>)
            }
            ProviderType::CloudflareAIGateway => {
                let (key, account_id, gateway_id, auth_token) =
                    Self::cloudflare_image_config(creds)?;
                let base_url = format!(
                    "https://gateway.ai.cloudflare.com/v1/{account_id}/{gateway_id}/compat"
                );
                let mut provider = OpenAIChatImageGenerationProvider::new(
                    "cloudflare-ai-gateway",
                    key,
                    model_name,
                    base_url,
                );
                if let Some(token) = auth_token {
                    provider =
                        provider.with_header("cf-aig-authorization", format!("Bearer {token}"));
                }
                Some(Arc::new(provider) as Arc<dyn ImageGenerationProvider>)
            }
            _ => None,
        };

        Ok(image_provider)
    }
}

#[cfg_attr(
    not(feature = "bedrock"),
    expect(
        clippy::unused_async,
        reason = "same signature as the async feature build"
    )
)]
async fn bedrock_embedding(
    creds: Credentials,
) -> anyhow::Result<Option<Arc<dyn EmbeddingProvider>>> {
    #[cfg(feature = "bedrock")]
    {
        if caliche_contracts::credential_env::var("BEDROCK_EMBEDDING_MODEL_ID").is_err() {
            return Ok(None);
        }
        let mut builder = BedrockProviderConfig::builder();
        match creds {
            Credentials::AwsAuth {
                access_key,
                secret_key,
                session_token,
                region,
            } => {
                builder = match session_token {
                    Some(token) => {
                        builder.aws_credentials_with_session(access_key, secret_key, token)
                    }
                    None => builder.aws_credentials(access_key, secret_key),
                };
                if let Some(region) = region {
                    builder = builder.region(region);
                }
            }
            Credentials::Bearer(token) => {
                builder = builder.bearer_token(token);
            }
            _ => {}
        }
        let bedrock_provider = BedrockProvider::with_config(builder.build()?).await?;
        return Ok(Some(
            Arc::new(bedrock_provider) as Arc<dyn EmbeddingProvider>
        ));
    }
    #[cfg(not(feature = "bedrock"))]
    {
        let _ = creds;
        Ok(None)
    }
}
