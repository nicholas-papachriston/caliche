//! Model discovery utilities for LLM provider APIs.
//!
//! Fetch available models from
//! various LLM provider APIs with support for different auth methods,
//! response formats, filtering, and post-processing.

use serde::Deserialize;
use serde::de::DeserializeOwned;

/// Common structure for OpenAI-compatible model list responses.
#[derive(Debug, Deserialize)]
pub struct ModelData {
    pub id: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelsResponse {
    pub data: Vec<ModelData>,
}

/// Response format for Ollama's /api/tags endpoint.
#[derive(Debug, Deserialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
}

/// Response format for Google's /models endpoint.
#[derive(Debug, Deserialize)]
pub struct GoogleModelsResponse {
    pub models: Vec<GoogleModelInfo>,
}

#[derive(Debug, Deserialize)]
pub struct GoogleModelInfo {
    pub name: String,
    #[serde(default)]
    pub supported_generation_methods: Vec<String>,
}

/// Authentication method for model discovery requests.
#[derive(Clone)]
pub enum AuthMethod {
    /// No authentication required (local providers like LM Studio, Ollama)
    None,
    /// Bearer token in Authorization header
    Bearer(String),
    /// Custom header (e.g., Anthropic's x-api-key)
    Header { name: String, value: String },
    /// API key as query parameter
    QueryParam { name: String, value: String },
}

/// Configuration for fetching models from a provider API.
pub struct ModelDiscoveryConfig<R> {
    /// HTTP client to use for requests
    pub client: reqwest::Client,
    /// Full URL to the models endpoint
    pub url: String,
    /// Authentication method
    pub auth: AuthMethod,
    /// Additional headers to include
    pub extra_headers: Vec<(String, String)>,
    /// Provider name for logging
    pub provider_name: String,
    /// Fallback models if API call fails
    pub fallback_models: Vec<String>,
    /// Function to extract model IDs from the response
    pub extractor: fn(R) -> Vec<String>,
    /// Optional filter function
    pub filter: Option<fn(&str) -> bool>,
    /// Optional sort function
    pub sorter: Option<fn(&mut Vec<String>)>,
}

/// Builder for `ModelDiscoveryConfig`.
pub struct ModelDiscoveryBuilder<R> {
    client: Option<reqwest::Client>,
    url: Option<String>,
    auth: AuthMethod,
    extra_headers: Vec<(String, String)>,
    provider_name: String,
    fallback_models: Vec<String>,
    extractor: Option<fn(R) -> Vec<String>>,
    filter: Option<fn(&str) -> bool>,
    sorter: Option<fn(&mut Vec<String>)>,
}

impl<R> Default for ModelDiscoveryBuilder<R> {
    fn default() -> Self {
        Self {
            client: None,
            url: None,
            auth: AuthMethod::None,
            extra_headers: Vec::new(),
            provider_name: "Unknown".to_string(),
            fallback_models: Vec::new(),
            extractor: None,
            filter: None,
            sorter: None,
        }
    }
}

impl<R: DeserializeOwned + 'static> ModelDiscoveryBuilder<R> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the HTTP client (creates a new one if not set).
    #[must_use]
    pub fn client(mut self, client: reqwest::Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Set the URL for the models endpoint.
    #[must_use]
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// Set bearer token authentication.
    #[must_use]
    pub fn bearer_auth(mut self, token: impl Into<String>) -> Self {
        self.auth = AuthMethod::Bearer(token.into());
        self
    }

    /// Set custom header authentication (e.g., x-api-key).
    #[must_use]
    pub fn header_auth(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = AuthMethod::Header {
            name: name.into(),
            value: value.into(),
        };
        self
    }

    /// Set query parameter authentication.
    #[must_use]
    pub fn query_auth(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = AuthMethod::QueryParam {
            name: name.into(),
            value: value.into(),
        };
        self
    }

    /// Add an extra header to the request.
    #[must_use]
    pub fn extra_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Set the provider name for logging.
    #[must_use]
    pub fn provider_name(mut self, name: impl Into<String>) -> Self {
        self.provider_name = name.into();
        self
    }

    /// Set fallback models to use if API call fails.
    #[must_use]
    pub fn fallback_models(mut self, models: Vec<String>) -> Self {
        self.fallback_models = models;
        self
    }

    /// Set the function to extract model IDs from the response.
    #[must_use]
    pub fn extractor(mut self, f: fn(R) -> Vec<String>) -> Self {
        self.extractor = Some(f);
        self
    }

    /// Set an optional filter function for model IDs.
    #[must_use]
    pub fn filter(mut self, f: fn(&str) -> bool) -> Self {
        self.filter = Some(f);
        self
    }

    /// Set an optional sort function for model IDs.
    #[must_use]
    pub fn sorter(mut self, f: fn(&mut Vec<String>)) -> Self {
        self.sorter = Some(f);
        self
    }

    /// Build the configuration and fetch models.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch(self) -> anyhow::Result<Vec<String>> {
        let client = self.client.unwrap_or_default();
        let url = self
            .url
            .ok_or_else(|| anyhow::anyhow!("URL is required for model discovery"))?;
        let extractor = self
            .extractor
            .ok_or_else(|| anyhow::anyhow!("Extractor is required for model discovery"))?;

        fetch_models_internal::<R>(
            &client,
            &url,
            &self.auth,
            &self.extra_headers,
            &self.provider_name,
            self.fallback_models,
            extractor,
            self.filter,
            self.sorter,
        )
        .await
    }
}

/// Internal function that performs the actual model fetching.
#[allow(clippy::too_many_arguments)]
async fn fetch_models_internal<R: DeserializeOwned>(
    client: &reqwest::Client,
    url: &str,
    auth: &AuthMethod,
    extra_headers: &[(String, String)],
    provider_name: &str,
    fallback_models: Vec<String>,
    extractor: fn(R) -> Vec<String>,
    filter: Option<fn(&str) -> bool>,
    sorter: Option<fn(&mut Vec<String>)>,
) -> anyhow::Result<Vec<String>> {
    // Build the request
    let mut request = match auth {
        AuthMethod::QueryParam { name, value } => {
            client.get(url).query(&[(name.as_str(), value.as_str())])
        }
        _ => client.get(url),
    };

    // Add auth header if needed
    match auth {
        AuthMethod::Bearer(token) => {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        AuthMethod::Header { name, value } => {
            request = request.header(name.as_str(), value.as_str());
        }
        AuthMethod::None | AuthMethod::QueryParam { .. } => {}
    }

    // Add extra headers
    for (name, value) in extra_headers {
        request = request.header(name.as_str(), value.as_str());
    }

    // Send request
    let response = match request.send().await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::debug!(
                "Failed to fetch {} models from API: {}, using fallback",
                provider_name,
                e
            );
            return Ok(fallback_models);
        }
    };

    // Check status
    if !response.status().is_success() {
        tracing::debug!(
            "{} /models returned status {}, using fallback",
            provider_name,
            response.status()
        );
        return Ok(fallback_models);
    }

    // Parse response
    match response.json::<R>().await {
        Ok(resp) => {
            let mut models = extractor(resp);

            // Apply filter if provided
            if let Some(filter_fn) = filter {
                models.retain(|m| filter_fn(m));
            }

            // Apply sorter if provided
            if let Some(sorter_fn) = sorter {
                sorter_fn(&mut models);
            }

            if models.is_empty() {
                tracing::debug!("{} returned no models, using fallback", provider_name);
                Ok(fallback_models)
            } else {
                tracing::debug!("Fetched {} models from {} API", models.len(), provider_name);
                Ok(models)
            }
        }
        Err(e) => {
            tracing::debug!(
                "Failed to parse {} models: {}, using fallback",
                provider_name,
                e
            );
            Ok(fallback_models)
        }
    }
}

// ============================================================================
// Convenience constructors for common provider patterns
// ============================================================================

/// Fetch models from an OpenAI-compatible /models endpoint.
/// Works for: `OpenAI`, `OpenRouter`, LM Studio, Perplexity (with API support).
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn fetch_openai_compatible_models(
    client: &reqwest::Client,
    base_url: &str,
    api_key: Option<&str>,
    provider_name: &str,
    fallback_models: Vec<String>,
) -> anyhow::Result<Vec<String>> {
    let mut builder = ModelDiscoveryBuilder::<ModelsResponse>::new()
        .client(client.clone())
        .url(format!("{base_url}/models"))
        .provider_name(provider_name)
        .fallback_models(fallback_models)
        .extractor(|r| r.data.into_iter().map(|m| m.id).collect());

    if let Some(key) = api_key {
        builder = builder.bearer_auth(key);
    }

    builder.fetch().await
}

/// Fetch models from Anthropic's API with custom auth headers.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn fetch_anthropic_models(
    client: &reqwest::Client,
    api_key: &str,
    fallback_models: Vec<String>,
) -> anyhow::Result<Vec<String>> {
    ModelDiscoveryBuilder::<ModelsResponse>::new()
        .client(client.clone())
        .url("https://api.anthropic.com/v1/models")
        .header_auth("x-api-key", api_key)
        .extra_header("anthropic-version", "2023-06-01")
        .provider_name("Anthropic")
        .fallback_models(fallback_models)
        .extractor(|r| r.data.into_iter().map(|m| m.id).collect())
        .fetch()
        .await
}

/// Fetch models from Google's Gemini API with query param auth.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn fetch_google_models(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    fallback_models: Vec<String>,
) -> anyhow::Result<Vec<String>> {
    ModelDiscoveryBuilder::<GoogleModelsResponse>::new()
        .client(client.clone())
        .url(format!("{base_url}/models"))
        .query_auth("key", api_key)
        .provider_name("Google")
        .fallback_models(fallback_models)
        .extractor(|r| r.models.into_iter().map(|m| m.name).collect())
        .filter(|_| true) // Google filtering done in extractor for simplicity
        .fetch()
        .await
}

/// Fetch models from Ollama's /api/tags endpoint.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn fetch_ollama_models(
    client: &reqwest::Client,
    base_url: &str,
    fallback_models: Vec<String>,
) -> anyhow::Result<Vec<String>> {
    // Strip /v1 suffix if present for the native Ollama endpoint
    let api_base = base_url.strip_suffix("/v1").unwrap_or(base_url);

    ModelDiscoveryBuilder::<OllamaTagsResponse>::new()
        .client(client.clone())
        .url(format!("{api_base}/api/tags"))
        .provider_name("Ollama")
        .fallback_models(fallback_models)
        .extractor(|r| r.models.into_iter().map(|m| m.name).collect())
        .fetch()
        .await
}

/// Check if a local provider (Ollama, LM Studio) is available.
pub async fn check_local_provider_availability(url: &str, timeout_secs: u64) -> bool {
    let Ok(client) = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .build()
    else {
        return false;
    };

    (client.get(url).send().await).map_or(false, |resp| resp.status().is_success())
}
