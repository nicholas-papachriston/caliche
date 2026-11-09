//! Per-provider constructors for [`super::provider_factory::ProviderFactory`].

use super::provider_factory::ProviderFactory;
use crate::capabilities::providers;
use crate::llm::anthropic::AnthropicProviderConfig;
#[cfg(feature = "bedrock")]
use crate::llm::bedrock::{BedrockProvider, BedrockProviderConfig};
use crate::llm::google::GoogleProviderConfig;
use crate::llm::openai_compat::{
    CloudflareAIGatewayConfig, LMStudioConfig, OllamaConfig, OpenAIConfig, OpenRouterConfig,
    PerplexityConfig,
};
use crate::llm::{
    AnthropicProvider, GoogleProvider, LMStudioProvider, OllamaProvider, OpenAIProvider,
    OpenRouterProvider, PerplexityProvider, Provider, ProviderType,
};
use crate::traits::credentials::Credentials;
use std::sync::Arc;
use std::time::Duration;

/// Applies common options (`api_key`, timeout, `max_retries`, `shared_http_client`) to a builder.
macro_rules! apply_common_builder_options {
    ($builder:expr, $creds:expr, $timeout:expr, $max_retries:expr $(, $shared_http_client:expr)?) => {{
        let mut __b = $builder;
        if let Credentials::ApiKey(key) = $creds {
            __b = __b.api_key(key);
        }
        if let Some(t) = $timeout {
            __b = __b.timeout(t);
        }
        if let Some(r) = $max_retries {
            __b = __b.max_retries(r);
        }
        $(
            #[cfg(feature = "reqwest")]
            if let Some(ref client) = $shared_http_client {
                __b = __b.shared_http_client(client.clone());
            }
        )?
        __b
    }};
}

/// Credential resolution mode for [`ProviderFactory::build_provider`].
pub(super) enum ProviderCredentialMode<'a> {
    /// Use provider `from_env` paths only (no explicit credential provider).
    Environment,
    /// Use resolved credentials from a [`crate::traits::credentials::CredentialProvider`].
    Explicit(&'a Credentials),
}

pub(super) struct ProviderBuildCtx<'a> {
    model_name: String,
    cred_mode: ProviderCredentialMode<'a>,
    creds: Credentials,
    base_url: Option<&'a str>,
    timeout: Option<Duration>,
    max_retries: Option<usize>,
    needs_builder: bool,
    env_only: bool,
    #[cfg(feature = "reqwest")]
    shared_http_client: Option<std::sync::Arc<reqwest::Client>>,
}

pub(super) async fn build_llm_provider(
    provider: ProviderType,
    model_name: String,
    cred_mode: ProviderCredentialMode<'_>,
    base_url: Option<&str>,
    timeout: Option<Duration>,
    max_retries: Option<usize>,
    #[cfg(feature = "reqwest")] shared_http_client: Option<std::sync::Arc<reqwest::Client>>,
) -> anyhow::Result<Arc<dyn Provider>> {
    if !providers::is_provider_available(provider) {
        return Err(providers::provider_not_available_error(provider));
    }

    let creds = ProviderFactory::credentials_for_builder(&cred_mode);

    #[cfg(feature = "reqwest")]
    let has_shared_client = shared_http_client.as_ref().is_some();
    #[cfg(not(feature = "reqwest"))]
    let has_shared_client = false;
    let needs_builder = timeout.is_some() || max_retries.is_some() || has_shared_client;
    let env_only = matches!(cred_mode, ProviderCredentialMode::Environment);

    let ctx = ProviderBuildCtx {
        model_name,
        cred_mode,
        creds,
        base_url,
        timeout,
        max_retries,
        needs_builder,
        env_only,
        #[cfg(feature = "reqwest")]
        shared_http_client,
    };

    match provider {
        ProviderType::Bedrock => build_bedrock(ctx).await,
        ProviderType::OpenRouter => build_openrouter(ctx).await,
        ProviderType::CloudflareAIGateway => build_cloudflare(ctx),
        ProviderType::Perplexity => build_perplexity(ctx).await,
        ProviderType::OpenAI => build_openai(ctx).await,
        ProviderType::LMStudio => build_lmstudio(ctx).await,
        ProviderType::Ollama => build_ollama(ctx).await,
        ProviderType::Anthropic => build_anthropic(ctx).await,
        ProviderType::Google => build_google(ctx).await,
        ProviderType::Cohere => build_cohere(ctx).await,
        ProviderType::AzureOpenAI => build_azure_openai(ctx).await,
        #[cfg(feature = "local-inference-base")]
        ProviderType::LocalInference => Ok(Arc::new(
            crate::llm::LazyLocalInferenceProvider::from_env(Some(ctx.model_name))?,
        )),
        ProviderType::External => Err(anyhow::anyhow!(
            "External provider requires a callback. Use ProviderFactory::create_with_external_callback instead."
        )),
    }
}

#[cfg_attr(
    not(feature = "bedrock"),
    expect(
        clippy::unused_async,
        reason = "same signature as the async feature build"
    )
)]
async fn build_bedrock(
    #[cfg_attr(
        not(feature = "bedrock"),
        expect(unused_variables, reason = "read by the bedrock build")
    )]
    ctx: ProviderBuildCtx<'_>,
) -> anyhow::Result<Arc<dyn Provider>> {
    #[cfg(feature = "bedrock")]
    {
        if ctx.env_only {
            return Ok(Arc::new(
                BedrockProvider::from_env(Some(ctx.model_name)).await?,
            ));
        }
        let mut builder = BedrockProviderConfig::builder().model_id(ctx.model_name);
        match &ctx.cred_mode {
            ProviderCredentialMode::Explicit(Credentials::AwsAuth {
                access_key,
                secret_key,
                session_token,
                region,
            }) => {
                builder = match session_token {
                    Some(token) => builder.aws_credentials_with_session(
                        access_key.clone(),
                        secret_key.clone(),
                        token.clone(),
                    ),
                    None => builder.aws_credentials(access_key.clone(), secret_key.clone()),
                };
                if let Some(region) = region {
                    builder = builder.region(region.clone());
                }
            }
            ProviderCredentialMode::Explicit(Credentials::Bearer(token)) => {
                builder = builder.bearer_token(token.clone());
            }
            _ => {}
        }
        Ok(Arc::new(
            BedrockProvider::with_config(builder.build()?).await?,
        ))
    }
    #[cfg(not(feature = "bedrock"))]
    {
        Err(providers::provider_not_available_error(
            ProviderType::Bedrock,
        ))
    }
}

async fn build_openrouter(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    if ctx.needs_builder {
        #[cfg(feature = "reqwest")]
        let builder = apply_common_builder_options!(
            OpenRouterConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries,
            ctx.shared_http_client
        );
        #[cfg(not(feature = "reqwest"))]
        let builder = apply_common_builder_options!(
            OpenRouterConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries
        );
        Ok(Arc::new(builder.build()?))
    } else {
        match ctx.creds {
            Credentials::ApiKey(key) => {
                Ok(Arc::new(OpenRouterProvider::try_new(key, ctx.model_name)?))
            }
            _ => Ok(Arc::new(
                OpenRouterProvider::from_env(Some(ctx.model_name)).await?,
            )),
        }
    }
}

fn build_cloudflare(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    let mut builder = ProviderFactory::apply_cloudflare_credentials(
        CloudflareAIGatewayConfig::builder().model(ctx.model_name),
        ctx.creds,
    );
    if let Some(t) = ctx.timeout {
        builder = builder.timeout(t);
    }
    if let Some(r) = ctx.max_retries {
        builder = builder.max_retries(r);
    }
    #[cfg(feature = "reqwest")]
    if let Some(ref client) = ctx.shared_http_client {
        builder = builder.shared_http_client(client.clone());
    }
    Ok(Arc::new(builder.build()?))
}

async fn build_perplexity(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    if ctx.needs_builder {
        #[cfg(feature = "reqwest")]
        let builder = apply_common_builder_options!(
            PerplexityConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries,
            ctx.shared_http_client
        );
        #[cfg(not(feature = "reqwest"))]
        let builder = apply_common_builder_options!(
            PerplexityConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries
        );
        Ok(Arc::new(builder.build()?))
    } else {
        match ctx.creds {
            Credentials::ApiKey(key) => {
                Ok(Arc::new(PerplexityProvider::try_new(key, ctx.model_name)?))
            }
            _ => Ok(Arc::new(
                PerplexityProvider::from_env(Some(ctx.model_name)).await?,
            )),
        }
    }
}

async fn build_openai(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    if ctx.needs_builder {
        #[cfg(feature = "reqwest")]
        let builder = apply_common_builder_options!(
            OpenAIConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries,
            ctx.shared_http_client
        );
        #[cfg(not(feature = "reqwest"))]
        let builder = apply_common_builder_options!(
            OpenAIConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries
        );
        Ok(Arc::new(builder.build()?))
    } else if ctx.env_only {
        Ok(Arc::new(
            OpenAIProvider::from_env(Some(ctx.model_name)).await?,
        ))
    } else {
        match ctx.cred_mode {
            ProviderCredentialMode::Explicit(Credentials::ApiKey(key)) => Ok(Arc::new(
                OpenAIProvider::try_new(key.clone(), ctx.model_name)?,
            )),
            ProviderCredentialMode::Explicit(_) => {
                let key = crate::llm::provider_utils::read_api_key_from_env("OPENAI_API_KEY")?;
                Ok(Arc::new(OpenAIProvider::try_new(key, ctx.model_name)?))
            }
            ProviderCredentialMode::Environment => Ok(Arc::new(
                OpenAIProvider::from_env(Some(ctx.model_name)).await?,
            )),
        }
    }
}

async fn build_lmstudio(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    if ctx.needs_builder {
        let mut builder = LMStudioConfig::builder().model(ctx.model_name);
        if let Some(url) = ctx.base_url {
            builder = builder.base_url(url);
        }
        if let Some(t) = ctx.timeout {
            builder = builder.timeout(t);
        }
        if let Some(r) = ctx.max_retries {
            builder = builder.max_retries(r);
        }
        #[cfg(feature = "reqwest")]
        if let Some(ref client) = ctx.shared_http_client {
            builder = builder.shared_http_client(client.clone());
        }
        Ok(Arc::new(builder.build()?))
    } else if let Some(url) = ctx.base_url {
        Ok(Arc::new(LMStudioProvider::try_new(
            url.to_string(),
            ctx.model_name,
        )?))
    } else {
        Ok(Arc::new(
            LMStudioProvider::from_env(Some(ctx.model_name)).await?,
        ))
    }
}

async fn build_ollama(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    if ctx.needs_builder {
        let mut builder = OllamaConfig::builder().model(ctx.model_name);
        if let Some(url) = ctx.base_url {
            builder = builder.base_url(url);
        }
        if let Some(t) = ctx.timeout {
            builder = builder.timeout(t);
        }
        if let Some(r) = ctx.max_retries {
            builder = builder.max_retries(r);
        }
        #[cfg(feature = "reqwest")]
        if let Some(ref client) = ctx.shared_http_client {
            builder = builder.shared_http_client(client.clone());
        }
        Ok(Arc::new(builder.build()?))
    } else if let Some(url) = ctx.base_url {
        Ok(Arc::new(OllamaProvider::try_new(
            url.to_string(),
            ctx.model_name,
        )?))
    } else {
        Ok(Arc::new(
            OllamaProvider::from_env(Some(ctx.model_name)).await?,
        ))
    }
}

async fn build_anthropic(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    if ctx.needs_builder {
        #[cfg(feature = "reqwest")]
        let builder = apply_common_builder_options!(
            AnthropicProviderConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries,
            ctx.shared_http_client
        );
        #[cfg(not(feature = "reqwest"))]
        let builder = apply_common_builder_options!(
            AnthropicProviderConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries
        );
        let config = builder.build()?;
        Ok(Arc::new(AnthropicProvider::with_config(config)?))
    } else {
        match ctx.creds {
            Credentials::ApiKey(key) => {
                Ok(Arc::new(AnthropicProvider::try_new(key, ctx.model_name)?))
            }
            _ => Ok(Arc::new(
                AnthropicProvider::from_env(Some(ctx.model_name)).await?,
            )),
        }
    }
}

async fn build_google(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    if ctx.needs_builder {
        #[cfg(feature = "reqwest")]
        let builder = apply_common_builder_options!(
            GoogleProviderConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries,
            ctx.shared_http_client
        );
        #[cfg(not(feature = "reqwest"))]
        let builder = apply_common_builder_options!(
            GoogleProviderConfig::builder().model(ctx.model_name),
            ctx.creds,
            ctx.timeout,
            ctx.max_retries
        );
        let config = builder.build()?;
        Ok(Arc::new(GoogleProvider::with_config(config)?))
    } else {
        match ctx.creds {
            Credentials::ApiKey(key) => Ok(Arc::new(GoogleProvider::try_new(key, ctx.model_name)?)),
            _ => Ok(Arc::new(
                GoogleProvider::from_env(Some(ctx.model_name)).await?,
            )),
        }
    }
}

async fn build_cohere(ctx: ProviderBuildCtx<'_>) -> anyhow::Result<Arc<dyn Provider>> {
    #[cfg(feature = "cohere")]
    {
        use crate::llm::cohere::CohereProvider;
        if ctx.env_only {
            Ok(Arc::new(
                CohereProvider::from_env(Some(ctx.model_name)).await?,
            ))
        } else if ctx.needs_builder {
            #[cfg(feature = "reqwest")]
            let builder = apply_common_builder_options!(
                crate::llm::cohere::CohereProviderConfig::builder().model(ctx.model_name),
                ctx.creds,
                ctx.timeout,
                ctx.max_retries,
                ctx.shared_http_client
            );
            #[cfg(not(feature = "reqwest"))]
            let builder = apply_common_builder_options!(
                crate::llm::cohere::CohereProviderConfig::builder().model(ctx.model_name),
                ctx.creds,
                ctx.timeout,
                ctx.max_retries
            );
            let config = builder.build()?;
            Ok(Arc::new(CohereProvider::with_config(config)?))
        } else {
            match ctx.cred_mode {
                ProviderCredentialMode::Explicit(Credentials::ApiKey(key)) => Ok(Arc::new(
                    CohereProvider::try_new(key.clone(), ctx.model_name)?,
                )),
                _ => Ok(Arc::new(
                    CohereProvider::from_env(Some(ctx.model_name)).await?,
                )),
            }
        }
    }
    #[cfg(not(feature = "cohere"))]
    {
        Err(providers::provider_not_available_error(
            ProviderType::Cohere,
        ))
    }
}

#[cfg_attr(
    not(feature = "azure-openai"),
    expect(
        clippy::unused_async,
        reason = "same signature as the async feature build"
    )
)]
async fn build_azure_openai(
    #[cfg_attr(
        not(feature = "azure-openai"),
        expect(unused_variables, reason = "read by the azure-openai build")
    )]
    ctx: ProviderBuildCtx<'_>,
) -> anyhow::Result<Arc<dyn Provider>> {
    #[cfg(feature = "azure-openai")]
    {
        use crate::llm::azure_openai::AzureOpenAIProvider;
        if ctx.env_only {
            Ok(Arc::new(
                AzureOpenAIProvider::from_env(Some(ctx.model_name)).await?,
            ))
        } else if ctx.needs_builder {
            let resource_name = std::env::var("AZURE_OPENAI_RESOURCE_NAME")
                .map_err(|e| anyhow::anyhow!("AZURE_OPENAI_RESOURCE_NAME must be set: {e}"))?;
            let deployment_id = std::env::var("AZURE_OPENAI_DEPLOYMENT_ID")
                .unwrap_or_else(|_| ctx.model_name.clone());
            let base_builder = crate::llm::azure_openai::AzureOpenAIProviderConfig::builder()
                .resource_name(resource_name)
                .deployment_id(deployment_id);
            #[cfg(feature = "reqwest")]
            let builder = apply_common_builder_options!(
                base_builder,
                ctx.creds,
                ctx.timeout,
                ctx.max_retries,
                ctx.shared_http_client
            );
            #[cfg(not(feature = "reqwest"))]
            let builder = apply_common_builder_options!(
                base_builder,
                ctx.creds,
                ctx.timeout,
                ctx.max_retries
            );
            let config = builder.build()?;
            Ok(Arc::new(AzureOpenAIProvider::with_config(config)?))
        } else {
            match ctx.cred_mode {
                ProviderCredentialMode::Explicit(Credentials::ApiKey(key)) => {
                    let resource_name =
                        std::env::var("AZURE_OPENAI_RESOURCE_NAME").map_err(|e| {
                            anyhow::anyhow!("AZURE_OPENAI_RESOURCE_NAME must be set: {e}")
                        })?;
                    let deployment_id = std::env::var("AZURE_OPENAI_DEPLOYMENT_ID")
                        .unwrap_or_else(|_| ctx.model_name.clone());
                    Ok(Arc::new(AzureOpenAIProvider::try_new(
                        resource_name,
                        deployment_id,
                        key.clone(),
                    )?))
                }
                _ => Ok(Arc::new(
                    AzureOpenAIProvider::from_env(Some(ctx.model_name)).await?,
                )),
            }
        }
    }
    #[cfg(not(feature = "azure-openai"))]
    {
        Err(providers::provider_not_available_error(
            ProviderType::AzureOpenAI,
        ))
    }
}
