//! Embedding provider auto-detection.
//!
//! Probes all available embedding providers concurrently and returns the
//! best one by priority: local first (Ollama, LM Studio), then cloud
//! (OpenAI, Bedrock, Cohere, Azure OpenAI).

use crate::llm::{EmbeddingProvider, EmbeddingRequest, ProviderType};
use crate::traits::credentials::{CredentialProvider, EnvironmentCredentialProvider};
use std::sync::Arc;
use std::time::Duration;
use tracing::debug;

/// Result of embedding provider discovery.
#[derive(Debug, Clone)]
pub struct EmbeddingProviderDiscovery {
    pub provider_type: ProviderType,
    pub embedding_model: String,
    pub embedding_dimension: Option<usize>,
    pub provider: Arc<dyn EmbeddingProvider>,
}

/// Test string used to verify embedding providers work.
const EMBED_PROBE_TEXT: &str = "test";

/// Timeout for a single embedding probe call. Prevents hanging on
/// partially-responsive providers that accept connections but never reply.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Discover the best available embedding provider.
///
/// Probes all providers **concurrently** and returns the first working one
/// by priority: Ollama > LM Studio > OpenAI > Bedrock > Cohere > Azure OpenAI.
/// Total wall-clock time is bounded by the slowest single probe (~2-5 s)
/// rather than the sum of all probes.
pub async fn discover_best_embedding_provider() -> Option<EmbeddingProviderDiscovery> {
    let cred_provider =
        Arc::new(EnvironmentCredentialProvider::new()) as Arc<dyn CredentialProvider>;

    // Probe every provider concurrently -- the total wait time is the
    // duration of the single slowest probe instead of all probes summed.
    let (ollama, lmstudio, openai, bedrock, cohere, azure) = tokio::join!(
        try_ollama(),
        try_lmstudio(),
        try_openai(Arc::clone(&cred_provider)),
        try_bedrock(Arc::clone(&cred_provider)),
        try_cohere(Arc::clone(&cred_provider)),
        try_azure_openai(Arc::clone(&cred_provider)),
    );

    // Return first available in priority order: local first (no cost, privacy), then cloud
    ollama
        .or(lmstudio)
        .or(openai)
        .or(bedrock)
        .or(cohere)
        .or(azure)
}

#[cfg(feature = "ollama")]
async fn try_ollama() -> Option<EmbeddingProviderDiscovery> {
    use crate::llm::OllamaConfig;

    let base_url = caliche_contracts::credential_env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

    if !crate::llm::OllamaProvider::is_available(Some(&base_url)).await {
        return None;
    }

    let provider = match OllamaConfig::builder()
        .base_url(base_url)
        .model("dummy")
        .build()
    {
        Ok(p) => p,
        Err(e) => {
            debug!("Ollama provider creation failed: {}", e);
            return None;
        }
    };

    let embedding_model = caliche_contracts::credential_env::var("OLLAMA_EMBEDDING_MODEL")
        .unwrap_or_else(|_| "nomic-embed-text".to_string());

    let arc = Arc::new(provider) as Arc<dyn EmbeddingProvider>;
    let embedding_dimension = probe_embedding(&arc).await?;

    Some(EmbeddingProviderDiscovery {
        provider_type: ProviderType::Ollama,
        embedding_model,
        embedding_dimension: Some(embedding_dimension),
        provider: arc,
    })
}

#[cfg(not(feature = "ollama"))]
async fn try_ollama() -> Option<EmbeddingProviderDiscovery> {
    None
}

#[cfg(feature = "lmstudio")]
async fn try_lmstudio() -> Option<EmbeddingProviderDiscovery> {
    use crate::llm::LMStudioProvider;

    let base_url = caliche_contracts::credential_env::var("LMSTUDIO_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:1234/v1".to_string());

    if !LMStudioProvider::is_available(Some(&base_url)).await {
        return None;
    }

    let provider = match LMStudioProvider::from_env(None).await {
        Ok(p) => p,
        Err(e) => {
            debug!("LM Studio provider creation failed: {}", e);
            return None;
        }
    };

    let embedding_model = caliche_contracts::credential_env::var("LMSTUDIO_EMBEDDING_MODEL")
        .unwrap_or_else(|_| "text-embedding-qwen3-embedding-4b".to_string());

    let arc = Arc::new(provider) as Arc<dyn EmbeddingProvider>;
    let embedding_dimension = probe_embedding(&arc).await?;

    Some(EmbeddingProviderDiscovery {
        provider_type: ProviderType::LMStudio,
        embedding_model,
        embedding_dimension: Some(embedding_dimension),
        provider: arc,
    })
}

#[cfg(not(feature = "lmstudio"))]
async fn try_lmstudio() -> Option<EmbeddingProviderDiscovery> {
    None
}

#[cfg(feature = "openai")]
async fn try_openai(
    cred_provider: Arc<dyn CredentialProvider>,
) -> Option<EmbeddingProviderDiscovery> {
    use crate::agent::service::ProviderFactory;

    if caliche_contracts::credential_env::var("OPENAI_API_KEY").is_err() {
        return None;
    }

    let model = caliche_contracts::credential_env::var("OPENAI_MODEL")
        .unwrap_or_else(|_| "gpt-4o".to_string());
    let embedding_model = "text-embedding-3-small".to_string();

    let provider = match ProviderFactory::create_embedding_provider(
        ProviderType::OpenAI,
        model,
        cred_provider,
    )
    .await
    {
        Ok(Some(p)) => p,
        Ok(None) | Err(_) => return None,
    };

    let embedding_dimension = probe_embedding(&provider).await?;

    Some(EmbeddingProviderDiscovery {
        provider_type: ProviderType::OpenAI,
        embedding_model,
        embedding_dimension: Some(embedding_dimension),
        provider,
    })
}

#[cfg(not(feature = "openai"))]
async fn try_openai(
    _cred_provider: Arc<dyn CredentialProvider>,
) -> Option<EmbeddingProviderDiscovery> {
    None
}

#[cfg(feature = "bedrock")]
async fn try_bedrock(
    cred_provider: Arc<dyn CredentialProvider>,
) -> Option<EmbeddingProviderDiscovery> {
    use crate::agent::service::ProviderFactory;

    if caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID").is_err()
        && caliche_contracts::credential_env::var("AWS_SECRET_ACCESS_KEY").is_err()
    {
        return None;
    }

    // Bedrock embeddings require an explicit embedding model ID.
    let Ok(embedding_model) = caliche_contracts::credential_env::var("BEDROCK_EMBEDDING_MODEL_ID")
    else {
        return None;
    };

    let model = caliche_contracts::credential_env::var("BEDROCK_MODEL_ID")
        .unwrap_or_else(|_| "anthropic.claude-3-sonnet-20240229-v1:0".to_string());

    let provider = match ProviderFactory::create_embedding_provider(
        ProviderType::Bedrock,
        model,
        cred_provider,
    )
    .await
    {
        Ok(Some(p)) => p,
        Ok(None) | Err(_) => return None,
    };

    let embedding_dimension = probe_embedding(&provider).await?;

    Some(EmbeddingProviderDiscovery {
        provider_type: ProviderType::Bedrock,
        embedding_model,
        embedding_dimension: Some(embedding_dimension),
        provider,
    })
}

#[cfg(not(feature = "bedrock"))]
#[expect(
    clippy::unused_async,
    reason = "same signature as the async bedrock build"
)]
async fn try_bedrock(
    _cred_provider: Arc<dyn CredentialProvider>,
) -> Option<EmbeddingProviderDiscovery> {
    None
}

#[cfg(feature = "cohere")]
async fn try_cohere(
    _cred_provider: Arc<dyn CredentialProvider>,
) -> Option<EmbeddingProviderDiscovery> {
    use crate::llm::CohereProvider;

    if caliche_contracts::credential_env::var("COHERE_API_KEY").is_err() {
        return None;
    }

    let provider = match CohereProvider::from_env(None).await {
        Ok(p) => p,
        Err(_) => return None,
    };

    let embedding_model = "embed-english-v3.0".to_string();
    let arc = Arc::new(provider) as Arc<dyn EmbeddingProvider>;
    let embedding_dimension = probe_embedding(&arc).await?;

    Some(EmbeddingProviderDiscovery {
        provider_type: ProviderType::Cohere,
        embedding_model,
        embedding_dimension: Some(embedding_dimension),
        provider: arc,
    })
}

#[cfg(not(feature = "cohere"))]
async fn try_cohere(
    _cred_provider: Arc<dyn CredentialProvider>,
) -> Option<EmbeddingProviderDiscovery> {
    None
}

#[cfg(feature = "azure-openai")]
async fn try_azure_openai(
    _cred_provider: Arc<dyn CredentialProvider>,
) -> Option<EmbeddingProviderDiscovery> {
    if caliche_contracts::credential_env::var("AZURE_OPENAI_API_KEY").is_err() {
        return None;
    }

    match crate::llm::AzureOpenAIProvider::from_env(None).await {
        Ok(provider) => {
            let embedding_model = "text-embedding-3-small".to_string();
            let arc = Arc::new(provider) as Arc<dyn EmbeddingProvider>;
            let embedding_dimension = probe_embedding(&arc).await?;
            Some(EmbeddingProviderDiscovery {
                provider_type: ProviderType::AzureOpenAI,
                embedding_model,
                embedding_dimension: Some(embedding_dimension),
                provider: arc,
            })
        }
        Err(_) => None,
    }
}

#[cfg(not(feature = "azure-openai"))]
#[expect(
    clippy::unused_async,
    reason = "same signature as the async azure-openai build"
)]
async fn try_azure_openai(
    _cred_provider: Arc<dyn CredentialProvider>,
) -> Option<EmbeddingProviderDiscovery> {
    None
}

async fn probe_embedding(provider: &Arc<dyn EmbeddingProvider>) -> Option<usize> {
    probe_embedding_dimension(provider).await
}

/// Probe an embedding provider once and return vector dimension.
pub async fn probe_embedding_dimension(provider: &Arc<dyn EmbeddingProvider>) -> Option<usize> {
    probe_embedding_dimension_or_stop(provider)
        .await
        .ok()
        .flatten()
}

/// Probe an embedding provider once. `Ok(None)` when the probe timed out or
/// failed in a way a retry may fix; `Err` when a retry cannot fix it (see
/// [`crate::llm::is_permanent_embedding_error`]).
///
/// # Errors
///
/// Returns the provider's error when it is permanent.
pub async fn probe_embedding_dimension_or_stop(
    provider: &Arc<dyn EmbeddingProvider>,
) -> anyhow::Result<Option<usize>> {
    let request = EmbeddingRequest {
        texts: vec![EMBED_PROBE_TEXT.to_string()],
        model: None,
    };
    match tokio::time::timeout(PROBE_TIMEOUT, provider.embed(request)).await {
        Ok(Ok(response)) => Ok(response.embeddings.first().map(Vec::len)),
        Ok(Err(error)) if crate::llm::is_permanent_embedding_error(&error) => Err(error),
        Ok(Err(_)) | Err(_) => Ok(None),
    }
}
