//! Shared provider-name and environment-credential helpers.

use super::{GATEWAY_CREDENTIALS, known_providers};

/// Canonical lowercase provider id used across the secure store and CLI catalog.
///
/// Maps common aliases (`azure`, `azure_openai`) to the id in [`known_providers`].
pub fn normalize_provider_name(provider: &str) -> String {
    match provider.trim().to_ascii_lowercase().as_str() {
        "azure" | "azure_openai" => "azure-openai".to_string(),
        other => other.to_string(),
    }
}

/// Whether the current process already has API credentials for `provider` in the environment.
///
/// Uses canonical provider ids; accepts aliases via [`normalize_provider_name`].
pub fn provider_has_env_credentials(provider: &str) -> bool {
    let provider = normalize_provider_name(provider);
    if let Some(present) = hosted_provider_has_env_key(provider.as_str()) {
        return present;
    }
    catalog_provider_has_env_key(provider.as_str())
}

fn hosted_provider_has_env_key(provider: &str) -> Option<bool> {
    let api_key = caliche_contracts::credential_env::var("API_KEY").is_ok();
    Some(match provider {
        "openrouter" => {
            caliche_contracts::credential_env::var("OPENROUTER_API_KEY").is_ok() || api_key
        }
        "openai" => caliche_contracts::credential_env::var("OPENAI_API_KEY").is_ok() || api_key,
        "anthropic" => {
            caliche_contracts::credential_env::var("ANTHROPIC_API_KEY").is_ok() || api_key
        }
        "perplexity" => {
            caliche_contracts::credential_env::var("PERPLEXITY_API_KEY").is_ok() || api_key
        }
        "google" => caliche_contracts::credential_env::var("GOOGLE_API_KEY").is_ok() || api_key,
        "cohere" => caliche_contracts::credential_env::var("COHERE_API_KEY").is_ok() || api_key,
        "bedrock" => {
            caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID").is_ok()
                && caliche_contracts::credential_env::var("AWS_SECRET_ACCESS_KEY").is_ok()
        }
        "azure-openai" => {
            caliche_contracts::credential_env::var("AZURE_OPENAI_API_KEY").is_ok()
                && caliche_contracts::credential_env::var("AZURE_OPENAI_ENDPOINT").is_ok()
        }
        "ollama" | "lmstudio" | "local-inference" => true,
        _ => return None,
    })
}

fn catalog_provider_has_env_key(provider: &str) -> bool {
    if provider == "cloudflare-ai-gateway" {
        return (caliche_contracts::credential_env::var("CLOUDFLARE_AIG_API_KEY").is_ok()
            || caliche_contracts::credential_env::var("CLOUDFLARE_API_TOKEN").is_ok())
            && caliche_contracts::credential_env::var("CLOUDFLARE_ACCOUNT_ID").is_ok();
    }
    if let Some((_, env_var, _)) = GATEWAY_CREDENTIALS
        .iter()
        .find(|(gateway_name, _, _)| *gateway_name == provider)
    {
        return caliche_contracts::credential_env::var(env_var).is_ok();
    }
    known_providers()
        .iter()
        .find(|(known, _, _)| *known == provider)
        .is_some_and(|(_, env_var, _)| {
            !env_var.is_empty() && caliche_contracts::credential_env::var(env_var).is_ok()
        })
}

/// True when `provider` matches a row in [`known_providers`] (after normalization).
pub fn is_known_provider(provider: &str) -> bool {
    let provider = normalize_provider_name(provider);
    known_providers()
        .iter()
        .any(|(known, _, _)| *known == provider.as_str())
}
