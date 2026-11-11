//! Credential and model resolution utilities.
//!
//! This module provides helper functions for resolving provider types, model names,
//! and API keys from environment variables with appropriate fallbacks. It centralizes
//! the logic for determining which provider and model to use based on explicit
//! configuration, environment variables, or sensible defaults.
//!
//! # Environment Variables
//!
//! The following environment variables are supported:
//!
//! - `AGENT_PROVIDER`: Default provider type (bedrock, openai, anthropic, etc.)
//! - `DEFAULT_PROVIDER`: Documented alias used when `AGENT_PROVIDER` is unset
//! - `AGENT_MODEL`: Fallback model name for any provider
//! - `BEDROCK_MODEL_ID`: Bedrock-specific model ID
//! - `OPENAI_MODEL`: OpenAI-specific model name
//! - `ANTHROPIC_MODEL`: Anthropic-specific model name
//! - `OPENROUTER_MODEL`: OpenRouter-specific model name
//! - `PERPLEXITY_MODEL`: Perplexity-specific model name
//! - `LMSTUDIO_MODEL`: LM Studio-specific model name
//! - `GOOGLE_MODEL` or `GEMINI_MODEL`: Google/Gemini-specific model name
//!
//! # Resolution Priority
//!
//! Model resolution follows this priority order:
//! 1. Explicitly provided model parameter
//! 2. Provider-specific environment variable
//! 3. Generic `AGENT_MODEL` environment variable
//! 4. Provider-specific default model
//!
//! # Examples
//!
//! ```
//! use caliche::agent::service::credentials;
//! use caliche::llm::ProviderType;
//!
//! // Resolve provider (defaults to Bedrock if not set)
//! let provider = credentials::resolve_provider(None);
//!
//! // Resolve model with explicit value
//! let model = credentials::resolve_model_for_provider(
//!     ProviderType::OpenAI,
//!     Some("gpt-4-turbo".to_string())
//! );
//! assert_eq!(model, "gpt-4-turbo");
//!
//! // Resolve model from environment or default
//! let model = credentials::resolve_model_for_provider(
//!     ProviderType::OpenAI,
//!     None
//! );
//! // Returns value from OPENAI_MODEL env var, or the current OpenAI default
//! ```

use crate::llm::openai_compat::{
    normalize_cloudflare_ai_gateway_model, validate_cloudflare_ai_gateway_model,
};
use crate::llm::{
    CloudflareAIGatewayProvider, LMStudioProvider, ModelCatalogOptions, OllamaProvider,
    ProviderType, discover_provider_models,
};

/// Resolved provider + model pair for runtime construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRuntimeIdentity {
    pub provider: ProviderType,
    pub model: String,
}

/// Resolve provider and model from optional overrides (env/defaults for gaps).
#[must_use]
pub fn resolve_runtime_identity(
    provider: Option<ProviderType>,
    model: Option<String>,
) -> ResolvedRuntimeIdentity {
    let provider = resolve_provider(provider);
    let model = resolve_model_for_provider(provider, model);
    ResolvedRuntimeIdentity { provider, model }
}

/// Resolve provider and model, using live discovery for local providers when needed.
///
/// `options` is accepted for callers that already load catalog options; discovery
/// currently reuses [`resolve_model_for_provider_with_discovery`].
pub async fn resolve_runtime_identity_with_discovery(
    provider: Option<ProviderType>,
    model: Option<String>,
    options: Option<&ModelCatalogOptions>,
) -> anyhow::Result<ResolvedRuntimeIdentity> {
    let _ = options;
    let provider = resolve_provider(provider);
    let model = resolve_model_for_provider_with_discovery(provider, model).await?;
    Ok(ResolvedRuntimeIdentity { provider, model })
}

/// Resolve the provider from environment or use default.
#[must_use]
pub fn resolve_provider(provider: Option<ProviderType>) -> ProviderType {
    if let Some(provider) = provider {
        return provider;
    }
    env_provider("AGENT_PROVIDER")
        .or_else(|| env_provider("DEFAULT_PROVIDER"))
        .unwrap_or(ProviderType::OpenRouter)
}

fn env_provider(key: &str) -> Option<ProviderType> {
    let Ok(raw) = std::env::var(key) else {
        return None;
    };
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    match raw.parse() {
        Ok(provider) => Some(provider),
        Err(err) => {
            tracing::warn!("Invalid {key} value '{raw}': {err}");
            None
        }
    }
}

/// Resolve the model name for a given provider.
pub fn resolve_model_for_provider(provider: ProviderType, model: Option<String>) -> String {
    match provider {
        ProviderType::Bedrock => resolve_env_then_agent_then_default(
            model,
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-sonnet-20240229-v1:0",
        ),
        ProviderType::OpenRouter => resolve_env_then_agent_then_default(
            model,
            "OPENROUTER_MODEL",
            crate::llm::defaults::OPENROUTER,
        ),
        ProviderType::Perplexity => resolve_env_then_agent_then_default(
            model,
            "PERPLEXITY_MODEL",
            crate::llm::defaults::PERPLEXITY,
        ),
        ProviderType::OpenAI => {
            resolve_env_then_agent_then_default(model, "OPENAI_MODEL", crate::llm::defaults::OPENAI)
        }
        ProviderType::LMStudio => resolve_env_then_default(model, "LMSTUDIO_MODEL", "local-model"),
        ProviderType::Ollama => resolve_env_then_default(model, "OLLAMA_MODEL", "llama3.2"),
        ProviderType::CloudflareAIGateway => resolve_cloudflare_ai_gateway_model(model),
        ProviderType::Anthropic => resolve_env_then_agent_then_default(
            model,
            "ANTHROPIC_MODEL",
            crate::llm::defaults::ANTHROPIC,
        ),
        ProviderType::Google => resolve_google_model(model),
        #[cfg(feature = "cohere")]
        ProviderType::Cohere => {
            resolve_env_then_agent_then_plain_default(model, "COHERE_MODEL", "command-r-plus")
        }
        #[cfg(not(feature = "cohere"))]
        ProviderType::Cohere => model.unwrap_or_else(|| "command-r-plus".to_string()),
        #[cfg(feature = "azure-openai")]
        ProviderType::AzureOpenAI => {
            resolve_env_then_agent_then_plain_default(model, "AZURE_OPENAI_DEPLOYMENT_ID", "gpt-4o")
        }
        #[cfg(not(feature = "azure-openai"))]
        ProviderType::AzureOpenAI => model.unwrap_or_else(|| "gpt-4o".to_string()),
        #[cfg(feature = "local-inference-base")]
        ProviderType::LocalInference => resolve_local_inference_model(model),
        ProviderType::External => model
            .or_else(|| std::env::var("AGENT_MODEL").ok())
            .unwrap_or_else(|| "external".to_string()),
    }
}

fn first_env(keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|key| std::env::var(key).ok())
}

fn resolve_env_then_agent_then_default(
    model: Option<String>,
    specific_env: &str,
    default: &str,
) -> String {
    model
        .or_else(|| first_env(&[specific_env, "AGENT_MODEL"]))
        .unwrap_or_else(|| std::env::var(specific_env).unwrap_or_else(|_| default.to_string()))
}

fn resolve_env_then_agent_then_plain_default(
    model: Option<String>,
    specific_env: &str,
    default: &str,
) -> String {
    model
        .or_else(|| first_env(&[specific_env, "AGENT_MODEL"]))
        .unwrap_or_else(|| default.to_string())
}

fn resolve_env_then_default(model: Option<String>, specific_env: &str, default: &str) -> String {
    model
        .or_else(|| std::env::var(specific_env).ok())
        .unwrap_or_else(|| std::env::var(specific_env).unwrap_or_else(|_| default.to_string()))
}

fn resolve_cloudflare_ai_gateway_model(model: Option<String>) -> String {
    model
        .or_else(|| std::env::var("CLOUDFLARE_AIG_MODEL").ok())
        .or_else(|| std::env::var("AGENT_MODEL").ok())
        .map(normalize_cloudflare_ai_gateway_model)
        .unwrap_or_default()
}

fn resolve_google_model(model: Option<String>) -> String {
    crate::llm::google::GoogleProviderConfig::normalize_text_chat_model(
        model
            .or_else(|| first_env(&["GOOGLE_MODEL", "GEMINI_MODEL", "AGENT_MODEL"]))
            .unwrap_or_else(|| crate::llm::defaults::GOOGLE_API.to_string()),
    )
}

#[cfg(feature = "local-inference-base")]
fn resolve_local_inference_model(model: Option<String>) -> String {
    model
        .or_else(|| crate::llm::env_nonempty("CALICHE_LOCAL_INFERENCE_DOWNLOAD_REPO"))
        .or_else(|| crate::llm::env_nonempty("CALICHE_LOCAL_INFERENCE_MODEL"))
        .or_else(crate::llm::LocalInferenceProvider::preferred_downloaded_model)
        .unwrap_or_else(|| crate::llm::LocalInferenceProvider::default_model_name().to_string())
}

/// Resolve a model, using live discovery for local providers when no explicit
/// model or provider-specific environment variable is set.
///
/// This intentionally does not apply the generic `AGENT_MODEL` to local
/// providers because it often names a hosted model that local runtimes cannot
/// serve.
pub async fn resolve_model_for_provider_with_discovery(
    provider: ProviderType,
    model: Option<String>,
) -> anyhow::Result<String> {
    resolve_model_for_provider_with_discovery_at(provider, model, None).await
}

pub(crate) async fn resolve_model_for_provider_with_discovery_at(
    provider: ProviderType,
    model: Option<String>,
    base_url: Option<&str>,
) -> anyhow::Result<String> {
    if !is_live_discovery_provider(provider) {
        return validate_resolved_model(provider, resolve_model_for_provider(provider, model));
    }

    if let Some(model) = model.filter(|model| !model.trim().is_empty()) {
        return validate_resolved_model(provider, model);
    }

    if let Some(model) = provider_specific_model_env(provider) {
        return validate_resolved_model(provider, model);
    }

    if let Some(base_url) = base_url {
        return discover_default_model_at_base_url(provider, base_url).await;
    }

    if matches!(provider, ProviderType::CloudflareAIGateway) {
        let live_models =
            CloudflareAIGatewayProvider::fetch_live_available_models_from_credentials(None).await?;
        let discovered = live_models
            .into_iter()
            .find(|model| !looks_like_embedding_model(model))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No Cloudflare AI Gateway models discovered. Set CLOUDFLARE_ACCOUNT_ID and Cloudflare credentials for discovery, or set CLOUDFLARE_AIG_MODEL in '{{provider}}/{{model}}' format."
                )
            })?;
        return validate_resolved_model(provider, discovered);
    }

    let entry =
        discover_provider_models(provider, &ModelCatalogOptions::from_environment()).await?;
    let models = entry.models;
    let discovered = models
        .iter()
        .find(|model| !looks_like_embedding_model(&model.id))
        .or_else(|| models.first())
        .map(|model| model.id.clone());

    let discovered = discovered.ok_or_else(|| match provider {
        ProviderType::CloudflareAIGateway => anyhow::anyhow!(
            "No Cloudflare AI Gateway models discovered. Set CLOUDFLARE_ACCOUNT_ID and Cloudflare credentials for discovery, or set CLOUDFLARE_AIG_MODEL in '{{provider}}/{{model}}' format."
        ),
        _ => anyhow::anyhow!(
            "No models discovered for provider '{}'. Start the local runtime or set {}.",
            provider,
            provider_model_env_name(provider).unwrap_or("AGENT_MODEL")
        ),
    })?;
    validate_resolved_model(provider, discovered)
}

const fn is_live_discovery_provider(provider: ProviderType) -> bool {
    matches!(
        provider,
        ProviderType::LMStudio | ProviderType::Ollama | ProviderType::CloudflareAIGateway
    )
}

fn provider_specific_model_env(provider: ProviderType) -> Option<String> {
    let env_name = provider_model_env_name(provider)?;
    std::env::var(env_name)
        .ok()
        .filter(|model| !model.trim().is_empty())
}

const fn provider_model_env_name(provider: ProviderType) -> Option<&'static str> {
    match provider {
        ProviderType::LMStudio => Some("LMSTUDIO_MODEL"),
        ProviderType::Ollama => Some("OLLAMA_MODEL"),
        ProviderType::CloudflareAIGateway => Some("CLOUDFLARE_AIG_MODEL"),
        _ => None,
    }
}

async fn discover_default_model_at_base_url(
    provider: ProviderType,
    base_url: &str,
) -> anyhow::Result<String> {
    match provider {
        ProviderType::LMStudio => LMStudioProvider::discover_default_model(base_url).await,
        ProviderType::Ollama => OllamaProvider::discover_default_model(base_url).await,
        _ => Err(anyhow::anyhow!(
            "Live model discovery is not supported for provider '{provider}'"
        )),
    }
}

fn looks_like_embedding_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("embed") || model.contains("embedding")
}

fn validate_resolved_model(provider: ProviderType, model: String) -> anyhow::Result<String> {
    match provider {
        ProviderType::CloudflareAIGateway => validate_cloudflare_ai_gateway_model(model),
        _ => Ok(model),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn resolve_provider_reads_default_provider_env() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("AGENT_PROVIDER");
        env.set("DEFAULT_PROVIDER", "anthropic");
        assert_eq!(resolve_provider(None), ProviderType::Anthropic);
        env.set("AGENT_PROVIDER", "openai");
        assert_eq!(resolve_provider(None), ProviderType::OpenAI);
    }

    #[test]
    fn resolve_provider_honors_explicit_value() {
        assert_eq!(
            resolve_provider(Some(ProviderType::Anthropic)),
            ProviderType::Anthropic
        );
        assert_eq!(
            resolve_provider(Some(ProviderType::OpenAI)),
            ProviderType::OpenAI
        );
    }

    #[test]
    fn resolve_runtime_identity_uses_explicit_provider_and_model() {
        let identity =
            resolve_runtime_identity(Some(ProviderType::OpenAI), Some("gpt-test-identity".into()));
        assert_eq!(identity.provider, ProviderType::OpenAI);
        assert_eq!(identity.model, "gpt-test-identity");
    }

    #[test]
    fn resolve_model_for_provider_prefers_explicit_over_defaults() {
        assert_eq!(
            resolve_model_for_provider(ProviderType::OpenAI, Some("gpt-test".into())),
            "gpt-test"
        );
        assert_eq!(
            resolve_model_for_provider(ProviderType::Anthropic, Some("claude-test".into())),
            "claude-test"
        );
        assert_eq!(
            resolve_model_for_provider(ProviderType::OpenRouter, Some("or/test".into())),
            "or/test"
        );
    }

    #[test]
    #[serial]
    fn resolve_model_for_provider_falls_back_to_provider_defaults() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("OPENAI_MODEL");
        env.remove("AGENT_MODEL");
        let model = resolve_model_for_provider(ProviderType::OpenAI, None);
        assert_eq!(model, crate::llm::defaults::OPENAI);
    }
}
