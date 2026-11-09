//! Local provider discovery for auto-detection of Ollama and LM Studio.

use crate::llm::openai_compat::providers::{LMStudioProvider, OllamaProvider};
use std::time::Duration;

/// Timeout for fetching the model list from a local provider.
const MODEL_FETCH_TIMEOUT: Duration = Duration::from_millis(300);

/// Status of a local provider.
#[derive(Debug, Clone)]
pub struct LocalProviderStatus {
    pub name: &'static str,
    pub base_url: String,
    pub is_available: bool,
    pub models: Vec<String>,
}

async fn check_ollama() -> LocalProviderStatus {
    let base_url = std::env::var("OLLAMA_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:11434/v1".to_string());

    let is_available = OllamaProvider::is_available(Some(&base_url)).await;
    let models = if is_available {
        match OllamaProvider::try_new(base_url.clone(), "dummy".to_string()) {
            Ok(provider) => {
                match tokio::time::timeout(MODEL_FETCH_TIMEOUT, provider.fetch_available_models())
                    .await
                {
                    Ok(Ok(models)) => models,
                    Ok(Err(_)) | Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };

    LocalProviderStatus {
        name: "ollama",
        base_url,
        is_available,
        models,
    }
}

async fn check_lmstudio() -> LocalProviderStatus {
    let base_url = std::env::var("LMSTUDIO_BASE_URL")
        .unwrap_or_else(|_| "http://localhost:1234/v1".to_string());

    let is_available = LMStudioProvider::is_available(Some(&base_url)).await;
    let models = if is_available {
        match LMStudioProvider::try_new(base_url.clone(), "dummy".to_string()) {
            Ok(provider) => {
                match tokio::time::timeout(MODEL_FETCH_TIMEOUT, provider.fetch_available_models())
                    .await
                {
                    Ok(Ok(models)) => models,
                    Ok(Err(_)) | Err(_) => Vec::new(),
                }
            }
            Err(_) => Vec::new(),
        }
    } else {
        Vec::new()
    };

    LocalProviderStatus {
        name: "lmstudio",
        base_url,
        is_available,
        models,
    }
}

/// Discover available local providers concurrently.
pub async fn discover_local_providers() -> Vec<LocalProviderStatus> {
    let (ollama, lmstudio) = tokio::join!(check_ollama(), check_lmstudio());
    vec![ollama, lmstudio]
}
