//! Cloudflare AI Gateway provider configuration.
//!
//! Cloudflare AI Gateway exposes an OpenAI-compatible `/chat/completions`
//! endpoint that proxies requests to upstream providers. The model identifier
//! follows the `{upstream_provider}/{model}` convention (for example
//! `openai/gpt-4o-mini`, `anthropic/claude-sonnet-4-5`,
//! `google-ai-studio/gemini-2.5-pro`,
//! `workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast`).
//!
//! Two authentication modes are supported:
//!
//! - **BYOK (stored keys)**: provide a Cloudflare API token via
//!   `CLOUDFLARE_AIG_API_KEY` (or `CLOUDFLARE_API_TOKEN`). Cloudflare injects
//!   the upstream provider key it has on file.
//! - **Passthrough**: provide the upstream provider's own API key as
//!   `CLOUDFLARE_AIG_API_KEY` and, if the gateway is "authenticated", set
//!   `CLOUDFLARE_AIG_AUTH_TOKEN` so we send the additional
//!   `cf-aig-authorization: Bearer {token}` header.

use crate::llm::config::CommonProviderOptions;
use crate::llm::models::{CloudflareAIGatewayModel, ModelProvider, ProviderType};
use crate::llm::openai_compat::{OpenAICompatConfig, OpenAICompatibleProvider};
use crate::traits::credentials::Credentials;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;

const CLOUDFLARE_API_BASE_URL: &str = "https://api.cloudflare.com/client/v4";
const CLOUDFLARE_GATEWAY_BASE_HOST: &str = "https://gateway.ai.cloudflare.com/v1";
const DEFAULT_GATEWAY_ID: &str = "default";

#[derive(Debug, Deserialize)]
struct CloudflareCompatModelsResponse {
    #[serde(default)]
    data: Vec<CloudflareCompatModelInfo>,
}

#[derive(Debug, Deserialize)]
struct CloudflareCompatModelInfo {
    id: String,
}

#[derive(Debug)]
struct CloudflareAIGatewayAuth {
    api_key: Option<String>,
    account_id: String,
    gateway_id: String,
    cf_aig_authorization: Option<String>,
}

impl CloudflareAIGatewayAuth {
    fn from_credentials(credentials: Option<&Credentials>) -> anyhow::Result<Self> {
        let mut api_key = None;
        let mut account_id = None;
        let mut gateway_id = None;
        let mut cf_aig_authorization = None;

        match credentials {
            Some(Credentials::ApiKey(key)) => {
                api_key = Some(key.clone());
            }
            Some(Credentials::Bearer(token)) => {
                cf_aig_authorization = Some(token.clone());
            }
            Some(Credentials::Custom(values)) => {
                api_key = values.get("api_key").cloned();
                account_id = values.get("account_id").cloned();
                gateway_id = values.get("gateway_id").cloned();
                cf_aig_authorization = values
                    .get("cf_aig_authorization")
                    .or_else(|| values.get("auth_token"))
                    .cloned();
            }
            _ => {}
        }

        let api_key = api_key
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_API_KEY").ok())
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_API_TOKEN").ok());
        let account_id = account_id
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_ACCOUNT_ID").ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "CLOUDFLARE_ACCOUNT_ID is required to discover Cloudflare AI Gateway models"
                )
            })?;
        let gateway_id = gateway_id
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_GATEWAY_ID").ok())
            .unwrap_or_else(|| DEFAULT_GATEWAY_ID.to_string());
        let cf_aig_authorization = cf_aig_authorization
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_AUTH_TOKEN").ok());

        Ok(Self {
            api_key,
            account_id,
            gateway_id,
            cf_aig_authorization,
        })
    }

    fn base_url(&self) -> String {
        build_base_url(&self.account_id, &self.gateway_id)
    }
}

pub fn normalize_cloudflare_ai_gateway_model(model: impl Into<String>) -> String {
    let model = model.into();
    let model = strip_duplicate_cloudflare_provider_prefix(model.trim());
    match model.as_str() {
        "google/gemini-2.5-flash" => "google-ai-studio/gemini-2.5-flash".to_string(),
        "google/gemini-2.5-pro" => "google-ai-studio/gemini-2.5-pro".to_string(),
        "anthropic/claude-4-5-haiku" | "anthropic/claude-haiku-4.5" => {
            "anthropic/claude-haiku-4-5".to_string()
        }
        "anthropic/claude-4-5-sonnet" | "anthropic/claude-sonnet-4" => {
            "anthropic/claude-sonnet-4-5".to_string()
        }
        "anthropic/claude-4-5-opus" | "anthropic/claude-opus-4" => {
            "anthropic/claude-opus-4-5".to_string()
        }
        _ => model,
    }
}

fn strip_duplicate_cloudflare_provider_prefix(model: &str) -> String {
    let Some((provider, routed_model)) = model.split_once('/') else {
        return model.to_string();
    };
    let Some(stripped) = routed_model.strip_prefix(provider) else {
        return model.to_string();
    };
    let Some(stripped) = stripped.strip_prefix('/') else {
        return model.to_string();
    };
    if stripped.is_empty() {
        model.to_string()
    } else {
        format!("{provider}/{stripped}")
    }
}

/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub fn validate_cloudflare_ai_gateway_model(model: impl Into<String>) -> anyhow::Result<String> {
    let model = normalize_cloudflare_ai_gateway_model(model);
    if !is_valid_cloudflare_ai_gateway_model(&model) {
        anyhow::bail!(
            "Cloudflare AI Gateway models must use '{{provider}}/{{model}}' format, for example 'openai/gpt-5-mini', 'anthropic/claude-sonnet-4-5', 'google-ai-studio/gemini-2.5-flash', 'workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast', or 'dynamic/<route-name>'. Got '{model}'.",
        );
    }
    Ok(model)
}

fn is_valid_cloudflare_ai_gateway_model(model: &str) -> bool {
    if model.is_empty() || model.chars().any(char::is_whitespace) {
        return false;
    }

    let Some((provider, routed_model)) = model.split_once('/') else {
        return false;
    };

    !provider.is_empty()
        && !routed_model.is_empty()
        && provider
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_'))
}

/// Configuration for the Cloudflare AI Gateway provider.
#[derive(Debug, Clone)]
pub struct CloudflareAIGatewayConfig {
    pub api_key: Option<String>,
    pub model: String,
    pub account_id: String,
    pub gateway_id: String,
    pub cf_aig_authorization: Option<String>,
    base_url_cache: String,
}

impl CloudflareAIGatewayConfig {
    /// Create a new configuration with all required fields.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        account_id: impl Into<String>,
        gateway_id: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let account_id = account_id.into();
        let gateway_id = gateway_id.into();
        let base_url_cache = build_base_url(&account_id, &gateway_id);
        Ok(Self {
            api_key: Some(api_key.into()),
            model: validate_cloudflare_ai_gateway_model(model)?,
            account_id,
            gateway_id,
            cf_aig_authorization: None,
            base_url_cache,
        })
    }

    /// Attach a Cloudflare AI Gateway authentication token, sent as the
    /// `cf-aig-authorization` header. Required for authenticated gateways
    /// when running in passthrough mode.
    #[must_use]
    pub fn with_cf_aig_authorization(mut self, token: impl Into<String>) -> Self {
        self.cf_aig_authorization = Some(token.into());
        self
    }

    /// Create a builder for Cloudflare AI Gateway configuration.
    #[must_use]
    pub fn builder() -> CloudflareAIGatewayConfigBuilder {
        CloudflareAIGatewayConfigBuilder::default()
    }
}

fn build_base_url(account_id: &str, gateway_id: &str) -> String {
    format!("{CLOUDFLARE_GATEWAY_BASE_HOST}/{account_id}/{gateway_id}/compat")
}

impl OpenAICompatConfig for CloudflareAIGatewayConfig {
    fn provider_name(&self) -> &'static str {
        "cloudflare-ai-gateway"
    }

    fn base_url(&self) -> &str {
        &self.base_url_cache
    }

    fn auth_headers(&self) -> Vec<(&'static str, String)> {
        let mut headers = Vec::with_capacity(2);
        if let Some(ref api_key) = self.api_key
            && !api_key.trim().is_empty()
        {
            headers.push(("Authorization", format!("Bearer {api_key}")));
        }
        if let Some(ref cf_token) = self.cf_aig_authorization {
            headers.push(("cf-aig-authorization", format!("Bearer {cf_token}")));
        }
        headers
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn include_usage_in_stream(&self) -> bool {
        true
    }

    fn supports_embeddings(&self) -> bool {
        true
    }

    fn default_embedding_model(&self) -> Option<String> {
        if self.model.contains("embed") || self.model.contains("embedding") {
            Some(self.model.clone())
        } else {
            caliche_contracts::credential_env::var("CLOUDFLARE_AIG_EMBEDDING_MODEL")
                .ok()
                .or_else(|| {
                    caliche_contracts::credential_env::var("CLOUDFLARE_EMBEDDING_MODEL").ok()
                })
                .or_else(|| Some("workers-ai/@cf/baai/bge-base-en-v1.5".to_string()))
        }
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn supports_audio_input(&self) -> bool {
        true
    }

    fn supports_document_input(&self) -> bool {
        true
    }

    fn supported_media_types(&self) -> Vec<&str> {
        vec![
            "image/png",
            "image/jpeg",
            "image/webp",
            "audio/wav",
            "audio/mpeg",
            "application/pdf",
            "text/plain",
        ]
    }

    fn enhance_error(&self, status: u16, body: &str) -> Option<String> {
        if matches!(status, 401 | 403) {
            return Some(format!(
                "{body}\n\n\
                 Cloudflare AI Gateway rejected the request (HTTP {status}).\n\
                 Verify the following:\n\
                 1. CLOUDFLARE_ACCOUNT_ID is correct (currently '{account}')\n\
                 2. CLOUDFLARE_AIG_GATEWAY_ID exists (currently '{gateway}')\n\
                 3. The API key matches the chosen auth mode:\n\
                    - BYOK: a Cloudflare API token with AI Gateway access\n\
                    - Passthrough: the upstream provider's own API key (and optionally CLOUDFLARE_AIG_AUTH_TOKEN if the gateway is authenticated)\n\
                 4. The selected model '{model}' uses the '{{provider}}/{{model}}' format and is enabled on the gateway.",
                account = self.account_id,
                gateway = self.gateway_id,
                model = self.model,
            ));
        }
        if status == 404 && (body.contains("model") || body.contains("provider")) {
            return Some(format!(
                "{body}\n\n\
                 Cloudflare AI Gateway returned 404 for model '{model}'.\n\
                 Use the '{{upstream_provider}}/{{model}}' format (for example 'openai/gpt-4o-mini', 'anthropic/claude-sonnet-4', 'workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast').",
                model = self.model,
            ));
        }
        None
    }
}

/// Builder for [`CloudflareAIGatewayConfig`].
#[derive(Debug, Default)]
pub struct CloudflareAIGatewayConfigBuilder {
    api_key: Option<String>,
    model: Option<String>,
    account_id: Option<String>,
    gateway_id: Option<String>,
    cf_aig_authorization: Option<String>,
    common: CommonProviderOptions,
}

impl CloudflareAIGatewayConfigBuilder {
    /// Set the API key (Cloudflare token for BYOK, upstream key for passthrough).
    #[must_use]
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Set the `{provider}/{model}` identifier.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the Cloudflare account ID.
    #[must_use]
    pub fn account_id(mut self, account_id: impl Into<String>) -> Self {
        self.account_id = Some(account_id.into());
        self
    }

    /// Set the gateway ID (defaults to `"default"`).
    #[must_use]
    pub fn gateway_id(mut self, gateway_id: impl Into<String>) -> Self {
        self.gateway_id = Some(gateway_id.into());
        self
    }

    /// Set the optional `cf-aig-authorization` token (passthrough on
    /// authenticated gateways).
    #[must_use]
    pub fn cf_aig_authorization(mut self, token: impl Into<String>) -> Self {
        self.cf_aig_authorization = Some(token.into());
        self
    }

    /// Set request timeout.
    #[must_use]
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.common = self.common.with_timeout(timeout);
        self
    }

    /// Set maximum retries.
    #[must_use]
    pub fn max_retries(mut self, max_retries: usize) -> Self {
        self.common = self.common.with_max_retries(max_retries);
        self
    }

    /// Use a shared HTTP client for connection pooling across multiple agents.
    #[cfg(feature = "reqwest")]
    #[must_use]
    pub fn shared_http_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
        self.common = self.common.with_client(client);
        self
    }

    /// Build the configuration and create a provider.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn build(self) -> anyhow::Result<OpenAICompatibleProvider<CloudflareAIGatewayConfig>> {
        let api_key = self
            .api_key
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_API_KEY").ok())
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_API_TOKEN").ok());

        let model = self
            .model
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_MODEL").ok())
            .or_else(|| caliche_contracts::credential_env::var("AGENT_MODEL").ok())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Cloudflare AI Gateway requires an explicit model in '{{provider}}/{{model}}' format; set CLOUDFLARE_AIG_MODEL or pass --model"
                )
            })
            .and_then(validate_cloudflare_ai_gateway_model)?;

        let account_id = self
            .account_id
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_ACCOUNT_ID").ok())
            .ok_or_else(|| {
                anyhow::anyhow!("account_id is required or CLOUDFLARE_ACCOUNT_ID must be set")
            })?;

        let gateway_id = self
            .gateway_id
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_GATEWAY_ID").ok())
            .unwrap_or_else(|| DEFAULT_GATEWAY_ID.to_string());

        let cf_aig_authorization = self
            .cf_aig_authorization
            .or_else(|| caliche_contracts::credential_env::var("CLOUDFLARE_AIG_AUTH_TOKEN").ok());

        if api_key.as_deref().is_none_or(|key| key.trim().is_empty())
            && cf_aig_authorization
                .as_deref()
                .is_none_or(|key| key.trim().is_empty())
        {
            anyhow::bail!(
                "api_key or cf_aig_authorization is required; set CLOUDFLARE_AIG_API_KEY for passthrough provider credentials or CLOUDFLARE_AIG_AUTH_TOKEN for Cloudflare Gateway BYOK"
            );
        }

        let base_url_cache = build_base_url(&account_id, &gateway_id);

        let config = CloudflareAIGatewayConfig {
            api_key,
            model,
            account_id,
            gateway_id,
            cf_aig_authorization,
            base_url_cache,
        };

        OpenAICompatibleProvider::with_options(config, self.common)
    }
}

/// Canonical ergonomic alias for the Cloudflare AI Gateway provider.
pub type CloudflareAIGatewayProvider = OpenAICompatibleProvider<CloudflareAIGatewayConfig>;

impl CloudflareAIGatewayProvider {
    /// Create a new Cloudflare AI Gateway provider with explicit values.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub fn try_new_cloudflare(
        api_key: impl Into<String>,
        model: impl Into<String>,
        account_id: impl Into<String>,
        gateway_id: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let config = CloudflareAIGatewayConfig::new(api_key, model, account_id, gateway_id)?;
        Self::with_defaults(config)
    }

    /// Create a Cloudflare AI Gateway provider from environment variables.
    ///
    /// Honors `CLOUDFLARE_AIG_API_KEY` (or `CLOUDFLARE_API_TOKEN`),
    /// `CLOUDFLARE_ACCOUNT_ID`, optional `CLOUDFLARE_AIG_GATEWAY_ID`
    /// (defaults to `"default"`), optional `CLOUDFLARE_AIG_AUTH_TOKEN`,
    /// and `CLOUDFLARE_AIG_MODEL`/`AGENT_MODEL` for the model.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn from_env_cloudflare(model: Option<String>) -> anyhow::Result<Self> {
        std::future::ready(()).await;
        let mut builder = CloudflareAIGatewayConfig::builder();
        if let Some(m) = model {
            builder = builder.model(m);
        }
        builder.build()
    }

    /// Fetch available Cloudflare AI Gateway model IDs.
    ///
    /// Cloudflare documents model selection for the OpenAI-compatible Gateway
    /// endpoint as `{provider}/{model}`. There is no dedicated Gateway-wide
    /// model catalog API, so discovery attempts the OpenAI-compatible
    /// `{gateway}/compat/models` path first and then augments Workers AI entries
    /// through the documented Cloudflare API model search endpoint.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models(&self) -> anyhow::Result<Vec<String>> {
        let auth = CloudflareAIGatewayAuth {
            api_key: self.config().api_key.clone(),
            account_id: self.config().account_id.clone(),
            gateway_id: self.config().gateway_id.clone(),
            cf_aig_authorization: self.config().cf_aig_authorization.clone(),
        };
        fetch_cloudflare_ai_gateway_models(self.http_client(), &auth).await
    }

    /// Fetch available Cloudflare AI Gateway model IDs from a credential source
    /// without requiring an already-selected chat model.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_available_models_from_credentials(
        credentials: Option<&Credentials>,
    ) -> anyhow::Result<Vec<String>> {
        let auth = CloudflareAIGatewayAuth::from_credentials(credentials)?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        fetch_cloudflare_ai_gateway_models(&client, &auth).await
    }

    /// Fetch only live-discovered Cloudflare AI Gateway model IDs.
    ///
    /// Unlike [`Self::fetch_available_models_from_credentials`], this does not
    /// return curated fallback models. It is used for default model resolution so
    /// startup never silently chooses a baked-in Cloudflare model.
    /// # Errors
    ///
    /// Returns an error if request construction, transport, or response decoding fails.
    pub async fn fetch_live_available_models_from_credentials(
        credentials: Option<&Credentials>,
    ) -> anyhow::Result<Vec<String>> {
        let auth = CloudflareAIGatewayAuth::from_credentials(credentials)?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()?;
        fetch_cloudflare_ai_gateway_live_models(&client, &auth).await
    }
}

impl ModelProvider for CloudflareAIGatewayProvider {
    fn available_models(&self) -> Vec<String> {
        fallback_cloudflare_ai_gateway_models()
    }

    fn provider_type(&self) -> ProviderType {
        ProviderType::CloudflareAIGateway
    }
}

async fn fetch_cloudflare_ai_gateway_models(
    client: &reqwest::Client,
    auth: &CloudflareAIGatewayAuth,
) -> anyhow::Result<Vec<String>> {
    let models = fetch_cloudflare_ai_gateway_live_models(client, auth).await?;
    if models.is_empty() {
        Ok(fallback_cloudflare_ai_gateway_models())
    } else {
        Ok(models)
    }
}

async fn fetch_cloudflare_ai_gateway_live_models(
    client: &reqwest::Client,
    auth: &CloudflareAIGatewayAuth,
) -> anyhow::Result<Vec<String>> {
    let mut models = BTreeSet::new();

    match fetch_compat_models(client, auth).await {
        Ok(compat_models) => models.extend(compat_models),
        Err(error) => tracing::debug!(
            error = %error,
            "Cloudflare AI Gateway compat /models discovery failed"
        ),
    }

    match fetch_workers_ai_models(client, auth).await {
        Ok(workers_models) => models.extend(workers_models),
        Err(error) => tracing::debug!(
            error = %error,
            "Cloudflare Workers AI model search discovery failed"
        ),
    }

    Ok(models.into_iter().collect())
}

async fn fetch_compat_models(
    client: &reqwest::Client,
    auth: &CloudflareAIGatewayAuth,
) -> anyhow::Result<Vec<String>> {
    let mut request = client.get(format!("{}/models", auth.base_url()));
    request = apply_cloudflare_auth_headers(request, auth);

    let response = request.send().await?;
    if !response.status().is_success() {
        tracing::debug!(
            status = %response.status(),
            "Cloudflare AI Gateway compat /models returned non-success status"
        );
        return Ok(Vec::new());
    }

    let models_response = response.json::<CloudflareCompatModelsResponse>().await?;
    Ok(models_response
        .data
        .into_iter()
        .filter_map(|model| validate_cloudflare_ai_gateway_model(model.id).ok())
        .collect())
}

async fn fetch_workers_ai_models(
    client: &reqwest::Client,
    auth: &CloudflareAIGatewayAuth,
) -> anyhow::Result<Vec<String>> {
    let Some(api_key) = auth.api_key.as_deref().filter(|key| !key.trim().is_empty()) else {
        return Ok(Vec::new());
    };

    let mut discovered = BTreeSet::new();
    for page in 1_u64..=5 {
        let response = client
            .get(format!(
                "{CLOUDFLARE_API_BASE_URL}/accounts/{}/ai/models/search",
                auth.account_id
            ))
            .bearer_auth(api_key)
            .query(&[
                ("hide_experimental", "true".to_string()),
                ("per_page", "100".to_string()),
                ("page", page.to_string()),
            ])
            .send()
            .await?;

        if !response.status().is_success() {
            tracing::debug!(
                status = %response.status(),
                "Cloudflare Workers AI model search returned non-success status"
            );
            break;
        }

        let payload = response.json::<Value>().await?;
        let before_len = discovered.len();
        extract_workers_ai_model_ids(payload.get("result").unwrap_or(&payload), &mut discovered);
        let total_pages = payload
            .pointer("/result_info/total_pages")
            .and_then(Value::as_u64)
            .unwrap_or(1);
        if page >= total_pages || discovered.len() == before_len {
            break;
        }
    }

    Ok(discovered.into_iter().collect())
}

fn apply_cloudflare_auth_headers(
    mut request: reqwest::RequestBuilder,
    auth: &CloudflareAIGatewayAuth,
) -> reqwest::RequestBuilder {
    if let Some(api_key) = auth.api_key.as_deref().filter(|key| !key.trim().is_empty()) {
        request = request.bearer_auth(api_key);
    }
    if let Some(token) = auth
        .cf_aig_authorization
        .as_deref()
        .filter(|key| !key.trim().is_empty())
    {
        request = request.header("cf-aig-authorization", format!("Bearer {token}"));
    }
    request
}

fn extract_workers_ai_model_ids(value: &Value, models: &mut BTreeSet<String>) {
    match value {
        Value::Array(items) => {
            for item in items {
                extract_workers_ai_model_ids(item, models);
            }
        }
        Value::Object(map) => {
            for key in ["id", "name", "model", "model_id"] {
                if let Some(model_id) = map.get(key).and_then(Value::as_str) {
                    add_workers_ai_model_id(model_id, models);
                }
            }
            for item in map.values() {
                extract_workers_ai_model_ids(item, models);
            }
        }
        Value::String(model_id) => add_workers_ai_model_id(model_id, models),
        _ => {}
    }
}

fn add_workers_ai_model_id(model_id: &str, models: &mut BTreeSet<String>) {
    let routed_model = if model_id.starts_with("workers-ai/") {
        model_id.to_string()
    } else if model_id.starts_with('@') {
        format!("workers-ai/{model_id}")
    } else {
        return;
    };

    if let Ok(model) = validate_cloudflare_ai_gateway_model(routed_model) {
        models.insert(model);
    }
}

fn fallback_cloudflare_ai_gateway_models() -> Vec<String> {
    CloudflareAIGatewayModel::all()
        .iter()
        .map(|m: &CloudflareAIGatewayModel| m.as_str().to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_uses_default_gateway() {
        let config = CloudflareAIGatewayConfig::new(
            "test-key",
            "openai/gpt-4o-mini",
            "acct123",
            DEFAULT_GATEWAY_ID,
        )
        .expect("valid config");
        assert_eq!(
            config.base_url(),
            "https://gateway.ai.cloudflare.com/v1/acct123/default/compat"
        );
    }

    #[test]
    fn base_url_uses_custom_gateway() {
        let config = CloudflareAIGatewayConfig::new(
            "test-key",
            "anthropic/claude-sonnet-4",
            "acct123",
            "my-gateway",
        )
        .expect("valid config");
        assert_eq!(
            config.base_url(),
            "https://gateway.ai.cloudflare.com/v1/acct123/my-gateway/compat"
        );
    }

    #[test]
    fn auth_headers_without_cf_token_emit_only_authorization() {
        let config = CloudflareAIGatewayConfig::new(
            "secret",
            "openai/gpt-4o-mini",
            "acct123",
            DEFAULT_GATEWAY_ID,
        )
        .expect("valid config");
        let headers = config.auth_headers();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "Authorization");
        assert_eq!(headers[0].1, "Bearer secret");
    }

    #[test]
    fn auth_headers_include_cf_aig_authorization_when_set() {
        let config = CloudflareAIGatewayConfig::new(
            "upstream-key",
            "openai/gpt-4o-mini",
            "acct123",
            DEFAULT_GATEWAY_ID,
        )
        .expect("valid config")
        .with_cf_aig_authorization("cf-token");
        let headers = config.auth_headers();
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0].0, "Authorization");
        assert_eq!(headers[0].1, "Bearer upstream-key");
        assert_eq!(headers[1].0, "cf-aig-authorization");
        assert_eq!(headers[1].1, "Bearer cf-token");
    }

    #[test]
    fn try_new_builds_provider() {
        let provider = CloudflareAIGatewayProvider::try_new_cloudflare(
            "test-key",
            "openai/gpt-4o-mini",
            "acct123",
            DEFAULT_GATEWAY_ID,
        )
        .expect("provider should build");

        assert_eq!(provider.config().api_key.as_deref(), Some("test-key"));
        assert_eq!(provider.config().model, "openai/gpt-4o-mini");
        assert_eq!(provider.config().account_id, "acct123");
        assert_eq!(provider.config().gateway_id, "default");
        assert_eq!(
            provider.config().base_url(),
            "https://gateway.ai.cloudflare.com/v1/acct123/default/compat"
        );
    }

    #[test]
    fn auth_headers_allow_gateway_only_auth() {
        let config = CloudflareAIGatewayConfig {
            api_key: None,
            model: normalize_cloudflare_ai_gateway_model("google/gemini-2.5-flash"),
            account_id: "acct123".to_string(),
            gateway_id: DEFAULT_GATEWAY_ID.to_string(),
            cf_aig_authorization: Some("cf-token".to_string()),
            base_url_cache: build_base_url("acct123", DEFAULT_GATEWAY_ID),
        };
        let headers = config.auth_headers();
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "cf-aig-authorization");
        assert_eq!(headers[0].1, "Bearer cf-token");
        assert_eq!(config.model, "google-ai-studio/gemini-2.5-flash");
    }

    #[test]
    fn validates_cloudflare_model_format() {
        let model = validate_cloudflare_ai_gateway_model(" google/gemini-2.5-pro ")
            .expect("known alias should normalize");
        assert_eq!(model, "google-ai-studio/gemini-2.5-pro");

        let model = validate_cloudflare_ai_gateway_model("anthropic/anthropic/claude-haiku-4.5")
            .expect("duplicate provider prefix should normalize");
        assert_eq!(model, "anthropic/claude-haiku-4-5");

        let error = validate_cloudflare_ai_gateway_model("gpt-4o-mini")
            .expect_err("provider prefix is required");
        assert!(error.to_string().contains("{provider}/{model}"));
    }
}
