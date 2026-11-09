//! Credential provider abstraction for authentication backends.
//!
//! This module provides a unified interface for credential management across
//! different LLM providers. Implement the `CredentialProvider` trait to add
//! custom authentication sources like Vault, AWS Secrets Manager, or OIDC.
//!
//! # Examples
//!
//! ## Using Static Credentials
//!
//! ```rust,no_run
//! use caliche::traits::credentials::{CredentialProvider, Credentials, StaticCredentialProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let provider = StaticCredentialProvider::new(Credentials::ApiKey("sk-...".to_string()));
//! let creds = provider.get_credentials("openai").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Using Environment Variables
//!
//! ```rust,no_run
//! use caliche::traits::credentials::{CredentialProvider, EnvironmentCredentialProvider};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let provider = EnvironmentCredentialProvider::new();
//! let creds = provider.get_credentials("openai").await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Chaining Multiple Providers
//!
//! ```rust,no_run
//! use caliche::traits::credentials::{CredentialProvider, ChainedCredentialProvider, StaticCredentialProvider, EnvironmentCredentialProvider, Credentials};
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let chain = ChainedCredentialProvider::new(vec![
//!     Arc::new(StaticCredentialProvider::new(Credentials::ApiKey("sk-...".to_string()))),
//!     Arc::new(EnvironmentCredentialProvider::new()),
//! ]);
//! let creds = chain.get_credentials("openai").await?;
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// Unified credential representation for all authentication types
#[derive(Debug, Clone)]
pub enum Credentials {
    /// API key authentication (OpenAI, Anthropic, etc.)
    ApiKey(String),
    /// AWS authentication with optional session token
    AwsAuth {
        access_key: String,
        secret_key: String,
        session_token: Option<String>,
        region: Option<String>,
    },
    /// Bearer token authentication
    Bearer(String),
    /// Basic HTTP authentication
    BasicAuth { username: String, password: String },
    /// Structured credential data for auth types not covered by the variants above.
    ///
    /// Use this for OAuth2 token bundles, mTLS certificates, workspace-scoped tokens,
    /// OIDC JWT flows, or any provider-specific multi-field credential.
    ///
    /// Well-known keys (used by `normalize()`):
    /// - `"api_key"` → normalized to `ApiKey`
    /// - `"access_key"` + `"secret_key"` → normalized to `AwsAuth`
    /// - `"bearer_token"` or `"token"` → normalized to `Bearer`
    /// - `"username"` + `"password"` → normalized to `BasicAuth`
    Custom(std::collections::HashMap<String, String>),
    /// No authentication required (e.g., LMStudio)
    None,
}

impl Credentials {
    /// Extract API key if this is an ApiKey credential
    pub fn as_api_key(&self) -> Option<&str> {
        match self {
            Credentials::ApiKey(key) => Some(key),
            _ => None,
        }
    }

    /// Extract AWS credentials if this is AwsAuth
    pub fn as_aws_auth(&self) -> Option<(&str, &str, Option<&str>, Option<&str>)> {
        match self {
            Credentials::AwsAuth {
                access_key,
                secret_key,
                session_token,
                region,
            } => Some((
                access_key.as_str(),
                secret_key.as_str(),
                session_token.as_deref(),
                region.as_deref(),
            )),
            _ => None,
        }
    }

    /// Extract bearer token if this is a Bearer credential
    pub fn as_bearer(&self) -> Option<&str> {
        match self {
            Credentials::Bearer(token) => Some(token),
            _ => None,
        }
    }

    /// Extract the inner map if this is a Custom credential
    pub fn as_custom(&self) -> Option<&std::collections::HashMap<String, String>> {
        match self {
            Credentials::Custom(map) => Some(map),
            _ => None,
        }
    }

    /// Convert a `Custom` credential to the most specific known variant.
    ///
    /// Falls back to returning `self` unchanged if the map does not match any
    /// well-known key pattern.  All other variants are returned as-is.
    ///
    /// # Normalization rules (checked in order)
    /// 1. `{"api_key": "..."}` → `ApiKey`
    /// 2. `{"access_key": "...", "secret_key": "..."}` → `AwsAuth`
    /// 3. `{"bearer_token" | "token": "..."}` → `Bearer`
    /// 4. `{"username": "...", "password": "..."}` → `BasicAuth`
    pub fn normalize(self) -> Self {
        let map = match self {
            Credentials::Custom(m) => m,
            other => return other,
        };

        if let Some(key) = map.get("api_key") {
            return Credentials::ApiKey(key.clone());
        }

        if let (Some(ak), Some(sk)) = (map.get("access_key"), map.get("secret_key")) {
            return Credentials::AwsAuth {
                access_key: ak.clone(),
                secret_key: sk.clone(),
                session_token: map.get("session_token").cloned(),
                region: map.get("region").cloned(),
            };
        }

        if let Some(token) = map.get("bearer_token").or_else(|| map.get("token")) {
            return Credentials::Bearer(token.clone());
        }

        if let (Some(user), Some(pass)) = (map.get("username"), map.get("password")) {
            return Credentials::BasicAuth {
                username: user.clone(),
                password: pass.clone(),
            };
        }

        Credentials::Custom(map)
    }
}

/// Trait for credential providers that can retrieve authentication credentials
#[async_trait]
pub trait CredentialProvider: Send + Sync + std::fmt::Debug {
    /// Retrieve credentials for a specific provider
    ///
    /// # Arguments
    /// * `provider` - Provider name (e.g., "openai", "bedrock", "anthropic")
    ///
    /// # Returns
    /// * `Ok(Credentials)` - Successfully retrieved credentials
    /// * `Err` - Failed to retrieve credentials
    async fn get_credentials(&self, provider: &str) -> Result<Credentials>;

    /// Refresh credentials if they're about to expire (optional)
    ///
    /// Default implementation is a no-op. Override for providers that
    /// support credential rotation.
    async fn refresh_if_needed(&self) -> Result<()> {
        Ok(())
    }

    /// Check if this provider supports a given LLM provider
    ///
    /// # Arguments
    /// * `provider` - Provider name to check
    ///
    /// # Returns
    /// `true` if this credential provider can supply credentials for the given provider
    fn supports_provider(&self, provider: &str) -> bool;

    /// Get metadata about credential source (for debugging/logging)
    fn source_name(&self) -> &str;
}

/// Credential provider that uses static/hardcoded credentials
#[derive(Debug, Clone)]
pub struct StaticCredentialProvider {
    credentials: Credentials,
}

impl StaticCredentialProvider {
    /// Create a new static credential provider
    pub fn new(credentials: Credentials) -> Self {
        Self { credentials }
    }
}

#[async_trait]
impl CredentialProvider for StaticCredentialProvider {
    async fn get_credentials(&self, _provider: &str) -> Result<Credentials> {
        Ok(self.credentials.clone())
    }

    fn supports_provider(&self, _provider: &str) -> bool {
        true
    }

    fn source_name(&self) -> &str {
        "static"
    }
}

/// Credential provider that reads from environment variables
#[derive(Debug, Clone)]
pub struct EnvironmentCredentialProvider;

impl EnvironmentCredentialProvider {
    /// Create a new environment credential provider
    pub fn new() -> Self {
        Self
    }

    /// Get provider-specific environment variable name
    fn get_env_var_name(&self, provider: &str, var_type: &str) -> String {
        match provider.to_lowercase().as_str() {
            "openai" => format!("OPENAI_{}", var_type),
            "anthropic" => format!("ANTHROPIC_{}", var_type),
            "bedrock" => format!("AWS_{}", var_type),
            "openrouter" => format!("OPENROUTER_{}", var_type),
            "perplexity" => format!("PERPLEXITY_{}", var_type),
            "lmstudio" => format!("LMSTUDIO_{}", var_type),
            "google" | "gemini" => format!("GOOGLE_{}", var_type),
            "azure-openai" | "azure" => format!("AZURE_OPENAI_{}", var_type),
            "cloudflare-ai-gateway" | "cloudflare" | "cf-aig" => {
                format!("CLOUDFLARE_AIG_{}", var_type)
            }
            _ => format!("{}_{}", provider.to_uppercase(), var_type),
        }
    }
}

impl Default for EnvironmentCredentialProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CredentialProvider for EnvironmentCredentialProvider {
    async fn get_credentials(&self, provider: &str) -> Result<Credentials> {
        let provider_lower = provider.to_lowercase();

        // Local providers typically need no API credentials. Check before
        // generic `API_KEY` so a shared key for remote providers does not
        // leak into local-inference / Ollama / LM Studio resolution.
        if matches!(
            provider_lower.as_str(),
            "lmstudio" | "ollama" | "local-inference" | "localinference"
        ) {
            return Ok(Credentials::None);
        }

        // Try provider-specific API key first
        let api_key_var = self.get_env_var_name(provider, "API_KEY");
        if let Ok(key) = caliche_contracts::credential_env::var(&api_key_var) {
            return Ok(Credentials::ApiKey(key));
        }

        if matches!(provider_lower.as_str(), "google" | "gemini")
            && let Ok(key) = caliche_contracts::credential_env::var("GEMINI_API_KEY")
        {
            return Ok(Credentials::ApiKey(key));
        }

        if provider_lower == "cloudflare-ai-gateway"
            && let Ok(key) = caliche_contracts::credential_env::var("CLOUDFLARE_API_TOKEN")
        {
            return Ok(Credentials::ApiKey(key));
        }

        if provider_lower == "cloudflare-ai-gateway"
            && let Ok(auth_token) =
                caliche_contracts::credential_env::var("CLOUDFLARE_AIG_AUTH_TOKEN")
        {
            let mut values = std::collections::HashMap::new();
            values.insert("auth_token".to_string(), auth_token);
            if let Ok(account_id) = caliche_contracts::credential_env::var("CLOUDFLARE_ACCOUNT_ID")
            {
                values.insert("account_id".to_string(), account_id);
            }
            if let Ok(gateway_id) =
                caliche_contracts::credential_env::var("CLOUDFLARE_AIG_GATEWAY_ID")
            {
                values.insert("gateway_id".to_string(), gateway_id);
            }
            return Ok(Credentials::Custom(values));
        }

        // Try generic API key
        if let Ok(key) = caliche_contracts::credential_env::var("API_KEY") {
            return Ok(Credentials::ApiKey(key));
        }

        // For AWS/Bedrock, try AWS credentials
        if provider_lower == "bedrock" || provider_lower == "aws" {
            let access_key = caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID").ok();
            let secret_key = caliche_contracts::credential_env::var("AWS_SECRET_ACCESS_KEY").ok();
            let session_token = caliche_contracts::credential_env::var("AWS_SESSION_TOKEN").ok();
            let region = caliche_contracts::credential_env::var("AWS_REGION")
                .or_else(|_| caliche_contracts::credential_env::var("AWS_DEFAULT_REGION"))
                .ok();

            if let (Some(access_key), Some(secret_key)) = (access_key, secret_key) {
                return Ok(Credentials::AwsAuth {
                    access_key,
                    secret_key,
                    session_token,
                    region,
                });
            }

            // Try bearer token for Bedrock
            if let Ok(bearer) = caliche_contracts::credential_env::var("AWS_BEARER_TOKEN_BEDROCK") {
                return Ok(Credentials::Bearer(bearer));
            }

            // If no explicit credentials found, allow AWS SDK to use default credential chain
            // (IAM roles, instance profiles, etc.) by returning None
            // The Bedrock provider will handle this and use aws_config::defaults()
            return Ok(Credentials::None);
        }

        Err(anyhow::anyhow!(
            "No credentials found for provider '{}'. Set {} or API_KEY environment variable.",
            provider,
            api_key_var
        ))
    }

    fn supports_provider(&self, _provider: &str) -> bool {
        true
    }

    fn source_name(&self) -> &str {
        "environment"
    }
}

/// Credential provider that chains multiple providers, trying each in order
#[derive(Debug)]
pub struct ChainedCredentialProvider {
    providers: Vec<Arc<dyn CredentialProvider>>,
}

impl ChainedCredentialProvider {
    /// Create a new chained credential provider
    pub fn new(providers: Vec<Arc<dyn CredentialProvider>>) -> Self {
        Self { providers }
    }
}

#[async_trait]
impl CredentialProvider for ChainedCredentialProvider {
    async fn get_credentials(&self, provider: &str) -> Result<Credentials> {
        let mut last_error: Option<anyhow::Error> = None;

        for cred_provider in &self.providers {
            if !cred_provider.supports_provider(provider) {
                continue;
            }

            match cred_provider.get_credentials(provider).await {
                Ok(creds) => return Ok(creds),
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!(
                "No credential provider in chain could supply credentials for '{}'",
                provider
            )
        }))
    }

    fn supports_provider(&self, provider: &str) -> bool {
        self.providers.iter().any(|p| p.supports_provider(provider))
    }

    fn source_name(&self) -> &str {
        "chained"
    }
}

/// Credential provider wrapper that caches credentials with TTL
#[derive(Debug)]
pub struct CachedCredentialProvider {
    inner: Arc<dyn CredentialProvider>,
    cache:
        tokio::sync::Mutex<std::collections::HashMap<String, (Credentials, std::time::SystemTime)>>,
    ttl: std::time::Duration,
}

impl CachedCredentialProvider {
    /// Create a new cached credential provider
    pub fn new(inner: Arc<dyn CredentialProvider>, ttl: std::time::Duration) -> Self {
        Self {
            inner,
            cache: tokio::sync::Mutex::new(std::collections::HashMap::new()),
            ttl,
        }
    }
}

#[async_trait]
impl CredentialProvider for CachedCredentialProvider {
    async fn get_credentials(&self, provider: &str) -> Result<Credentials> {
        let now = std::time::SystemTime::now();

        // Check cache
        {
            let cache = self.cache.lock().await;
            if let Some((creds, cached_at)) = cache.get(provider)
                && now.duration_since(*cached_at).unwrap_or(self.ttl) < self.ttl
            {
                return Ok(creds.clone());
            }
        }

        // Fetch from inner provider
        let creds = self.inner.get_credentials(provider).await?;

        // Update cache
        {
            let mut cache = self.cache.lock().await;
            cache.insert(provider.to_string(), (creds.clone(), now));
        }

        Ok(creds)
    }

    async fn refresh_if_needed(&self) -> Result<()> {
        // Clear expired entries
        let now = std::time::SystemTime::now();
        let mut cache = self.cache.lock().await;
        cache.retain(|_, (_, cached_at)| {
            now.duration_since(*cached_at).unwrap_or(self.ttl) < self.ttl
        });
        drop(cache);

        self.inner.refresh_if_needed().await
    }

    fn supports_provider(&self, provider: &str) -> bool {
        self.inner.supports_provider(provider)
    }

    fn source_name(&self) -> &str {
        // Return a static string since we can't return formatted strings
        "cached"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_static_credential_provider() {
        let provider = StaticCredentialProvider::new(Credentials::ApiKey("test-key".to_string()));
        let creds = provider
            .get_credentials("openai")
            .await
            .expect("get_credentials should succeed");
        assert_eq!(creds.as_api_key(), Some("test-key"));
    }

    #[tokio::test]
    async fn test_chained_credential_provider() {
        let static_provider = Arc::new(StaticCredentialProvider::new(Credentials::ApiKey(
            "static-key".to_string(),
        )));
        let env_provider = Arc::new(EnvironmentCredentialProvider::new());

        let chain = ChainedCredentialProvider::new(vec![static_provider.clone(), env_provider]);
        let creds = chain
            .get_credentials("openai")
            .await
            .expect("chained get_credentials should succeed");
        assert_eq!(creds.as_api_key(), Some("static-key"));
    }

    #[tokio::test]
    async fn test_cached_credential_provider() {
        let static_provider = Arc::new(StaticCredentialProvider::new(Credentials::ApiKey(
            "cached-key".to_string(),
        )));
        let cached =
            CachedCredentialProvider::new(static_provider, std::time::Duration::from_secs(60));

        // First call should fetch from inner
        let creds1 = cached
            .get_credentials("openai")
            .await
            .expect("first get_credentials should succeed");
        assert_eq!(creds1.as_api_key(), Some("cached-key"));

        // Second call should use cache
        let creds2 = cached
            .get_credentials("openai")
            .await
            .expect("second get_credentials should succeed");
        assert_eq!(creds2.as_api_key(), Some("cached-key"));
    }

    #[tokio::test]
    #[serial_test::serial(env)]
    async fn environment_provider_treats_local_backends_as_credentialless() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("API_KEY", "should-not-apply-to-local");
        let provider = EnvironmentCredentialProvider::new();
        for name in [
            "lmstudio",
            "ollama",
            "local-inference",
            "localinference",
            "LOCAL-INFERENCE",
        ] {
            let creds = provider
                .get_credentials(name)
                .await
                .unwrap_or_else(|error| panic!("expected None credentials for {name}: {error}"));
            assert!(
                matches!(creds, Credentials::None),
                "expected Credentials::None for {name}"
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(env)]
    async fn environment_provider_maps_gemini_alias_to_google_vars() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("GOOGLE_API_KEY", "google-test-key");
        env.remove("GEMINI_API_KEY");
        env.remove("API_KEY");
        let provider = EnvironmentCredentialProvider::new();
        let creds = provider
            .get_credentials("gemini")
            .await
            .expect("gemini should resolve via GOOGLE_API_KEY");
        assert_eq!(creds.as_api_key(), Some("google-test-key"));
    }

    #[tokio::test]
    #[serial_test::serial(env)]
    async fn environment_provider_accepts_gemini_api_key_for_gemini_alias() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.remove("GOOGLE_API_KEY");
        env.remove("API_KEY");
        env.set("GEMINI_API_KEY", "gemini-only-key");
        let provider = EnvironmentCredentialProvider::new();
        let creds = provider
            .get_credentials("gemini")
            .await
            .expect("gemini should resolve via GEMINI_API_KEY");
        assert_eq!(creds.as_api_key(), Some("gemini-only-key"));
    }
}
