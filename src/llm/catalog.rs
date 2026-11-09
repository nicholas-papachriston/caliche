//! Shared provider and model discovery catalog.
//!
//! The catalog is the single boundary used by the API, CLI, and UI to discover
//! configured providers and their available models.

use crate::capabilities::providers;
#[cfg(feature = "bedrock")]
use crate::llm::BedrockProvider;
#[cfg(feature = "local-inference-base")]
use crate::llm::LocalInferenceProvider;
use crate::llm::{
    AnthropicModel, AnthropicProvider, AzureOpenAIModel, BedrockModel, CloudflareAIGatewayModel,
    CloudflareAIGatewayProvider, CohereModel, GoogleModel, GoogleProvider, LMStudioProvider,
    OllamaProvider, OpenAIModel, OpenAIProvider, OpenRouterModel, OpenRouterProvider,
    PerplexityModel, PerplexityProvider, ProviderType,
};
use crate::traits::credentials::{
    CredentialProvider, Credentials, EnvironmentCredentialProvider, StaticCredentialProvider,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Provider information returned by the shared catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCatalogEntry {
    /// Stable lowercase provider identifier, e.g. `openrouter`.
    pub id: String,
    /// Human-readable provider label.
    pub display_name: String,
    /// Whether the provider is configured or discoverable in this runtime.
    pub configured: bool,
    /// Whether the provider runs locally.
    pub local: bool,
    /// Sorted models discovered for this provider.
    pub models: Vec<ModelCatalogEntry>,
}

/// Model information returned by the shared catalog.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCatalogEntry {
    pub id: String,
    pub name: String,
    pub provider_id: String,
    pub provider_display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
}

/// Options for provider/model catalog discovery.
#[derive(Debug, Clone)]
pub struct ModelCatalogOptions {
    pub include_unconfigured: bool,
    pub live_fetch: bool,
    pub credential_provider: Arc<dyn CredentialProvider>,
}

impl ModelCatalogOptions {
    /// Build catalog options backed by process environment variables.
    #[must_use]
    pub fn from_environment() -> Self {
        Self {
            include_unconfigured: false,
            live_fetch: true,
            credential_provider: Arc::new(EnvironmentCredentialProvider::new()),
        }
    }

    /// Build catalog options that intentionally avoid configured credentials.
    #[must_use]
    pub fn without_credentials() -> Self {
        Self {
            include_unconfigured: true,
            live_fetch: false,
            credential_provider: Arc::new(StaticCredentialProvider::new(Credentials::None)),
        }
    }

    #[must_use]
    pub const fn include_unconfigured(mut self, include_unconfigured: bool) -> Self {
        self.include_unconfigured = include_unconfigured;
        self
    }

    #[must_use]
    pub const fn live_fetch(mut self, live_fetch: bool) -> Self {
        self.live_fetch = live_fetch;
        self
    }

    #[must_use]
    pub fn with_credential_provider(
        mut self,
        credential_provider: Arc<dyn CredentialProvider>,
    ) -> Self {
        self.credential_provider = credential_provider;
        self
    }
}

impl Default for ModelCatalogOptions {
    fn default() -> Self {
        Self::from_environment()
    }
}

/// List all provider types compiled into this binary.
pub fn compiled_provider_types() -> Vec<ProviderType> {
    let mut provider_types = vec![
        #[cfg(feature = "bedrock")]
        ProviderType::Bedrock,
        #[cfg(feature = "openai")]
        ProviderType::OpenAI,
        #[cfg(feature = "anthropic")]
        ProviderType::Anthropic,
        #[cfg(feature = "openrouter")]
        ProviderType::OpenRouter,
        #[cfg(feature = "perplexity")]
        ProviderType::Perplexity,
        #[cfg(feature = "lmstudio")]
        ProviderType::LMStudio,
        #[cfg(feature = "ollama")]
        ProviderType::Ollama,
        #[cfg(feature = "google")]
        ProviderType::Google,
        #[cfg(feature = "cohere")]
        ProviderType::Cohere,
        #[cfg(feature = "azure-openai")]
        ProviderType::AzureOpenAI,
        #[cfg(feature = "cloudflare-ai-gateway")]
        ProviderType::CloudflareAIGateway,
        #[cfg(feature = "local-inference-base")]
        ProviderType::LocalInference,
    ];
    provider_types.sort_by_key(ToString::to_string);
    provider_types
}

/// Discover all configured providers and their sorted model lists.
/// # Errors
///
/// Returns an error if the language-model operation cannot be completed.
pub async fn discover_provider_catalog(
    options: &ModelCatalogOptions,
) -> Result<Vec<ProviderCatalogEntry>> {
    let mut entries = Vec::new();

    for provider in compiled_provider_types() {
        let entry = discover_provider_models(provider, options).await?;
        if options.include_unconfigured || entry.configured {
            entries.push(entry);
        }
    }

    entries.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(entries)
}

/// Discover one provider and its sorted model list.
/// # Errors
///
/// Returns an error if the language-model operation cannot be completed.
pub async fn discover_provider_models(
    provider: ProviderType,
    options: &ModelCatalogOptions,
) -> Result<ProviderCatalogEntry> {
    let id = provider.to_string();
    let display_name = provider_display_name(provider).to_string();
    let local = is_local_provider(provider);

    let credential_state = if local {
        CredentialState::Configured(Credentials::None)
    } else {
        credential_state(provider, &options.credential_provider).await
    };
    let configured = credential_state.is_configured();

    let models = if configured {
        let live_models = if options.live_fetch {
            fetch_live_models(provider, credential_state.credentials()).await
        } else {
            None
        };

        live_models.map_or_else(|| fallback_models(provider), |models| models)
    } else if options.include_unconfigured {
        fallback_models(provider)
    } else {
        Vec::new()
    };

    let models = model_entries(provider, &display_name, models);

    Ok(ProviderCatalogEntry {
        id,
        display_name,
        configured,
        local,
        models,
    })
}

/// Flatten provider entries into a sorted model list.
#[must_use]
pub fn flatten_catalog_models(entries: &[ProviderCatalogEntry]) -> Vec<ModelCatalogEntry> {
    let mut models = entries
        .iter()
        .flat_map(|entry| entry.models.iter().cloned())
        .collect::<Vec<_>>();
    models.sort_by(|left, right| {
        left.provider_id
            .cmp(&right.provider_id)
            .then_with(|| left.id.cmp(&right.id))
    });
    models
}

fn model_entries(
    provider: ProviderType,
    provider_display_name: &str,
    models: Vec<String>,
) -> Vec<ModelCatalogEntry> {
    let provider_id = provider.to_string();
    dedupe_and_sort_model_ids(models)
        .into_iter()
        .map(|id| ModelCatalogEntry {
            name: id.clone(),
            id,
            provider_id: provider_id.clone(),
            provider_display_name: provider_display_name.to_string(),
            description: None,
            context_length: None,
        })
        .collect()
}

fn dedupe_and_sort_model_ids(mut models: Vec<String>) -> Vec<String> {
    models.retain(|model| !model.trim().is_empty());
    models.sort();
    models.dedup();
    models
}

#[derive(Debug)]
enum CredentialState {
    Configured(Credentials),
    Unconfigured,
}

impl CredentialState {
    const fn is_configured(&self) -> bool {
        matches!(self, Self::Configured(_))
    }

    const fn credentials(&self) -> Option<&Credentials> {
        match self {
            Self::Configured(credentials) => Some(credentials),
            Self::Unconfigured => None,
        }
    }
}

async fn credential_state(
    provider: ProviderType,
    credential_provider: &Arc<dyn CredentialProvider>,
) -> CredentialState {
    if !providers::is_provider_available(provider) {
        return CredentialState::Unconfigured;
    }

    let provider_id = provider.to_string();
    match credential_provider.get_credentials(&provider_id).await {
        Ok(credentials) if credentials_are_usable(provider, &credentials) => {
            CredentialState::Configured(credentials)
        }
        _ => CredentialState::Unconfigured,
    }
}

fn credentials_are_usable(provider: ProviderType, credentials: &Credentials) -> bool {
    match credentials {
        Credentials::ApiKey(key) | Credentials::Bearer(key) => !key.trim().is_empty(),
        Credentials::AwsAuth {
            access_key,
            secret_key,
            ..
        } => !access_key.trim().is_empty() && !secret_key.trim().is_empty(),
        Credentials::BasicAuth { username, password } => {
            !username.trim().is_empty() && !password.trim().is_empty()
        }
        Credentials::Custom(values) => !values.is_empty(),
        Credentials::None => {
            matches!(provider, ProviderType::Bedrock) || is_local_provider(provider)
        }
    }
}

const fn is_local_provider(provider: ProviderType) -> bool {
    matches!(provider, ProviderType::LMStudio | ProviderType::Ollama) || {
        #[cfg(feature = "local-inference-base")]
        {
            matches!(provider, ProviderType::LocalInference)
        }
        #[cfg(not(feature = "local-inference-base"))]
        {
            false
        }
    }
}

const fn provider_display_name(provider: ProviderType) -> &'static str {
    match provider {
        ProviderType::Bedrock => "Bedrock",
        ProviderType::OpenRouter => "OpenRouter",
        ProviderType::Perplexity => "Perplexity",
        ProviderType::OpenAI => "OpenAI",
        ProviderType::LMStudio => "LM Studio",
        ProviderType::Anthropic => "Anthropic",
        ProviderType::Google => "Google",
        ProviderType::Cohere => "Cohere",
        ProviderType::AzureOpenAI => "Azure OpenAI",
        ProviderType::Ollama => "Ollama",
        ProviderType::CloudflareAIGateway => "Cloudflare AI Gateway",
        #[cfg(feature = "local-inference-base")]
        ProviderType::LocalInference => "Local Inference",
        ProviderType::External => "External",
    }
}

async fn fetch_live_models(
    provider: ProviderType,
    credentials: Option<&Credentials>,
) -> Option<Vec<String>> {
    match provider {
        ProviderType::Bedrock => {
            #[cfg(feature = "bedrock")]
            {
                BedrockProvider::fetch_available_models().await.ok()
            }
            #[cfg(not(feature = "bedrock"))]
            {
                None
            }
        }
        ProviderType::OpenRouter
        | ProviderType::Perplexity
        | ProviderType::OpenAI
        | ProviderType::Anthropic
        | ProviderType::Google => fetch_with_api_key(provider, credentials).await,
        ProviderType::LMStudio => {
            let base_url = std::env::var("LMSTUDIO_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:1234/v1".to_string());
            let provider = LMStudioProvider::try_new(base_url, "dummy").ok()?;
            provider.fetch_available_models().await.ok()
        }
        ProviderType::Ollama => {
            let base_url = std::env::var("OLLAMA_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());
            let provider = OllamaProvider::try_new(base_url, "dummy").ok()?;
            provider.fetch_available_models().await.ok()
        }
        ProviderType::CloudflareAIGateway => {
            CloudflareAIGatewayProvider::fetch_available_models_from_credentials(credentials)
                .await
                .ok()
        }
        ProviderType::Cohere | ProviderType::AzureOpenAI | ProviderType::External => None,
        #[cfg(feature = "local-inference-base")]
        ProviderType::LocalInference => None,
    }
}

async fn fetch_with_api_key(
    provider: ProviderType,
    credentials: Option<&Credentials>,
) -> Option<Vec<String>> {
    let key = credentials.and_then(Credentials::as_api_key)?;
    let key = key.to_string();
    match provider {
        ProviderType::OpenRouter => OpenRouterProvider::try_new(key, "dummy")
            .ok()?
            .fetch_available_models()
            .await
            .ok(),
        ProviderType::Perplexity => PerplexityProvider::try_new(key, "dummy")
            .ok()?
            .fetch_available_models()
            .await
            .ok(),
        ProviderType::OpenAI => OpenAIProvider::try_new(key, "dummy")
            .ok()?
            .fetch_available_models()
            .await
            .ok(),
        ProviderType::Anthropic => AnthropicProvider::try_new(key, "dummy")
            .ok()?
            .fetch_available_models()
            .await
            .ok(),
        ProviderType::Google => GoogleProvider::try_new(key, crate::llm::defaults::GOOGLE_API)
            .ok()?
            .fetch_available_models()
            .await
            .ok(),
        _ => None,
    }
}

fn fallback_models(provider: ProviderType) -> Vec<String> {
    match provider {
        ProviderType::Bedrock => model_strings(BedrockModel::all()),
        ProviderType::OpenRouter => model_strings(OpenRouterModel::all()),
        ProviderType::Perplexity => model_strings(PerplexityModel::all()),
        ProviderType::OpenAI => model_strings(OpenAIModel::all()),
        ProviderType::LMStudio | ProviderType::Ollama => Vec::new(),
        ProviderType::Anthropic => model_strings(AnthropicModel::all()),
        ProviderType::Google => model_strings(GoogleModel::all()),
        ProviderType::Cohere => model_strings(CohereModel::all()),
        ProviderType::AzureOpenAI => model_strings(AzureOpenAIModel::all()),
        ProviderType::CloudflareAIGateway => model_strings(CloudflareAIGatewayModel::all()),
        #[cfg(feature = "local-inference-base")]
        ProviderType::LocalInference => {
            let mut models = LocalInferenceProvider::discover_downloaded_models();
            models.push(LocalInferenceProvider::default_model_name().to_string());
            models
        }
        ProviderType::External => Vec::new(),
    }
}

fn model_strings<T: ToString>(models: Vec<T>) -> Vec<String> {
    models.into_iter().map(|model| model.to_string()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedupe_and_sort_drops_empty_and_collapses_duplicates() {
        let models = dedupe_and_sort_model_ids(vec![
            "zeta".into(),
            "  ".into(),
            "alpha".into(),
            String::new(),
            "alpha".into(),
            "beta".into(),
        ]);
        assert_eq!(models, vec!["alpha", "beta", "zeta"]);
    }

    #[test]
    fn flatten_catalog_models_sorts_by_provider_then_id() {
        let entries = vec![
            ProviderCatalogEntry {
                id: "openai".into(),
                display_name: "OpenAI".into(),
                configured: true,
                local: false,
                models: vec![
                    ModelCatalogEntry {
                        id: "gpt-4o".into(),
                        name: "gpt-4o".into(),
                        provider_id: "openai".into(),
                        provider_display_name: "OpenAI".into(),
                        description: None,
                        context_length: None,
                    },
                    ModelCatalogEntry {
                        id: "gpt-4.1".into(),
                        name: "gpt-4.1".into(),
                        provider_id: "openai".into(),
                        provider_display_name: "OpenAI".into(),
                        description: None,
                        context_length: None,
                    },
                ],
            },
            ProviderCatalogEntry {
                id: "anthropic".into(),
                display_name: "Anthropic".into(),
                configured: true,
                local: false,
                models: vec![ModelCatalogEntry {
                    id: "claude".into(),
                    name: "claude".into(),
                    provider_id: "anthropic".into(),
                    provider_display_name: "Anthropic".into(),
                    description: None,
                    context_length: None,
                }],
            },
        ];
        let flat = flatten_catalog_models(&entries);
        let keys: Vec<_> = flat
            .iter()
            .map(|model| (model.provider_id.as_str(), model.id.as_str()))
            .collect();
        assert_eq!(
            keys,
            vec![
                ("anthropic", "claude"),
                ("openai", "gpt-4.1"),
                ("openai", "gpt-4o"),
            ]
        );
    }

    #[test]
    fn credentials_are_usable_rejects_empty_keys_and_allows_local_none() {
        assert!(!credentials_are_usable(
            ProviderType::OpenAI,
            &Credentials::ApiKey(String::new())
        ));
        assert!(!credentials_are_usable(
            ProviderType::OpenAI,
            &Credentials::ApiKey("   ".into())
        ));
        assert!(credentials_are_usable(
            ProviderType::OpenAI,
            &Credentials::ApiKey("sk-test".into())
        ));
        assert!(credentials_are_usable(
            ProviderType::Ollama,
            &Credentials::None
        ));
        assert!(credentials_are_usable(
            ProviderType::Bedrock,
            &Credentials::None
        ));
        assert!(!credentials_are_usable(
            ProviderType::OpenAI,
            &Credentials::None
        ));

        #[cfg(feature = "local-inference-base")]
        {
            assert!(credentials_are_usable(
                ProviderType::LocalInference,
                &Credentials::None
            ));
        }
    }
}
