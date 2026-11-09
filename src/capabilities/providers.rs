//! Provider capability detection and availability functions.

use crate::llm::ProviderType;

/// Check if a provider is available at compile time.
///
/// Returns `true` if the provider feature is enabled, `false` otherwise.
pub fn is_provider_available(provider: ProviderType) -> bool {
    match provider {
        ProviderType::Bedrock => cfg!(feature = "bedrock"),
        ProviderType::OpenAI => cfg!(feature = "openai"),
        ProviderType::Anthropic => cfg!(feature = "anthropic"),
        ProviderType::OpenRouter => cfg!(feature = "openrouter"),
        ProviderType::Perplexity => cfg!(feature = "perplexity"),
        ProviderType::LMStudio => cfg!(feature = "lmstudio"),
        ProviderType::Ollama => cfg!(feature = "ollama"),
        ProviderType::Google => cfg!(feature = "google"),
        ProviderType::Cohere => cfg!(feature = "cohere"),
        ProviderType::AzureOpenAI => cfg!(feature = "azure-openai"),
        ProviderType::CloudflareAIGateway => cfg!(feature = "cloudflare-ai-gateway"),
        #[cfg(feature = "local-inference-base")]
        ProviderType::LocalInference => cfg!(feature = "local-inference-base"),
        ProviderType::External => true, // External providers are always available
    }
}

/// Get a list of all available provider names as strings.
///
/// Returns a static slice of provider names that are compiled into the binary.
pub fn available_providers() -> &'static [&'static str] {
    &[
        #[cfg(feature = "bedrock")]
        "bedrock",
        #[cfg(feature = "openai")]
        "openai",
        #[cfg(feature = "anthropic")]
        "anthropic",
        #[cfg(feature = "openrouter")]
        "openrouter",
        #[cfg(feature = "perplexity")]
        "perplexity",
        #[cfg(feature = "lmstudio")]
        "lmstudio",
        #[cfg(feature = "ollama")]
        "ollama",
        #[cfg(feature = "google")]
        "google",
        #[cfg(feature = "cohere")]
        "cohere",
        #[cfg(feature = "azure-openai")]
        "azure-openai",
        #[cfg(feature = "cloudflare-ai-gateway")]
        "cloudflare-ai-gateway",
        #[cfg(feature = "local-inference-base")]
        "local-inference",
        "external",
    ]
}

/// The error for a provider this build left out: its name and the feature
/// that builds it in.
pub fn provider_not_available_error(provider: ProviderType) -> anyhow::Error {
    anyhow::anyhow!(crate::llm::models::not_in_build_message(
        &provider.to_string(),
        provider.build_feature()
    ))
}

/// Check if a provider supports embeddings.
///
/// Returns `true` when both the provider matches **and** the corresponding
/// Cargo feature is compiled in.  Each clause is parenthesised explicitly to
/// make the `&&`/`||` interaction unambiguous at a glance.
pub fn provider_supports_embeddings(provider: ProviderType) -> bool {
    (matches!(provider, ProviderType::OpenAI) && cfg!(feature = "openai"))
        || (matches!(provider, ProviderType::Bedrock) && cfg!(feature = "bedrock"))
        || (matches!(provider, ProviderType::OpenRouter) && cfg!(feature = "openrouter"))
        || (matches!(provider, ProviderType::LMStudio) && cfg!(feature = "lmstudio"))
}
