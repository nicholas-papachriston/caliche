//! Model failover with auth profile rotation.

use crate::llm::ProviderType;
use crate::llm::provider::Provider;
use crate::llm::provider_factory::{ProviderFactoryFn, default_provider_factory};
use crate::llm::provider_utils::classify_error;
use crate::messages::{CompletionRequest, CompletionResponse, StreamEvent};
use crate::traits::credentials::{CredentialProvider, Credentials, StaticCredentialProvider};
use crate::traits::retry::RetryError;
use anyhow::Result;
use async_trait::async_trait;
use futures_util::Stream;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

use super::auth_profiles::AuthProfileManager;

/// Best-effort classification of `anyhow::Error` for failover (e.g. from provider responses).
/// Tries to downcast to `RetryError` first when the provider returned one; otherwise uses string matching.
fn classify_error_for_failover(err: &anyhow::Error) -> RetryError {
    if let Some(r) = err.downcast_ref::<RetryError>() {
        return r.clone();
    }
    if let Some(req_err) = err.downcast_ref::<reqwest::Error>()
        && let Some(status) = req_err.status()
    {
        return classify_error(status, &req_err.to_string(), "failover");
    }
    classify_failover_message(err.to_string())
}

fn classify_failover_message(msg: String) -> RetryError {
    let lower = msg.to_lowercase();
    if is_rate_limited_message(&lower) {
        return RetryError::RateLimited {
            retry_after: None,
            message: msg,
        };
    }
    if is_auth_error_message(&lower) {
        return RetryError::AuthError { message: msg };
    }
    if is_model_unavailable_message(&lower) {
        return RetryError::ModelUnavailable { message: msg };
    }
    if is_permanent_error_message(&lower) {
        return RetryError::Permanent { error: msg };
    }
    RetryError::Transient {
        status_code: 0,
        message: msg,
    }
}

fn is_rate_limited_message(lower: &str) -> bool {
    lower.contains("status: 429")
        || lower.contains("(429)")
        || lower.contains("error 429")
        || lower.contains("rate limit")
}

fn is_auth_error_message(lower: &str) -> bool {
    lower.contains("401")
        || lower.contains("403")
        || lower.contains("unauthorized")
        || lower.contains("forbidden")
}

fn is_model_unavailable_message(lower: &str) -> bool {
    lower.contains("404") || lower.contains("not found")
}

fn is_permanent_error_message(lower: &str) -> bool {
    // Avoid bare "invalid" — it false-positives transient transport messages
    // like "connection invalid after reset".
    lower.contains("400")
        || lower.contains("bad request")
        || lower.contains("invalid_request")
        || lower.contains("invalid request")
        || lower.contains("validation error")
        || lower.contains("invalid parameter")
        || lower.contains("invalid argument")
}

#[derive(Debug, Clone)]
pub struct ProviderModelPair {
    pub provider: ProviderType,
    pub model: String,
}

impl ProviderModelPair {
    pub fn new(provider: ProviderType, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

/// Configuration for model failover (primary + fallbacks + auth + cooldown).
/// Use with `LlmClientBuilder::failover_config()` to wire failover from a single config.
#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// Primary provider and model.
    pub primary: ProviderModelPair,
    /// Fallback (provider, model) pairs tried in order.
    pub fallbacks: Vec<ProviderModelPair>,
    /// Auth profiles per provider (optional; defaults from `api_key/env` when missing).
    pub auth_profiles:
        std::collections::HashMap<ProviderType, Vec<super::auth_profiles::AuthProfile>>,
    /// Cooldown base seconds for rate-limited/auth-failed profiles.
    pub cooldown_base_seconds: Option<u64>,
    /// Optional channel to receive failover events.
    pub failover_tx: Option<tokio::sync::broadcast::Sender<FailoverEvent>>,
}

/// Builder for `FailoverConfig`.
#[derive(Debug, Default)]
pub struct FailoverConfigBuilder {
    primary: Option<ProviderModelPair>,
    fallbacks: Vec<ProviderModelPair>,
    auth_profiles: std::collections::HashMap<ProviderType, Vec<super::auth_profiles::AuthProfile>>,
    cooldown_base_seconds: Option<u64>,
    failover_tx: Option<tokio::sync::broadcast::Sender<FailoverEvent>>,
}

impl FailoverConfigBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn primary(mut self, provider: ProviderType, model: impl Into<String>) -> Self {
        self.primary = Some(ProviderModelPair::new(provider, model));
        self
    }

    #[must_use]
    pub fn fallback(mut self, provider: ProviderType, model: impl Into<String>) -> Self {
        self.fallbacks.push(ProviderModelPair::new(provider, model));
        self
    }

    #[must_use]
    pub fn auth_profiles(
        mut self,
        provider: ProviderType,
        profiles: Vec<super::auth_profiles::AuthProfile>,
    ) -> Self {
        self.auth_profiles.insert(provider, profiles);
        self
    }

    #[must_use]
    pub const fn cooldown_base(mut self, seconds: u64) -> Self {
        self.cooldown_base_seconds = Some(seconds);
        self
    }

    #[must_use]
    pub fn on_failover(mut self, tx: tokio::sync::broadcast::Sender<FailoverEvent>) -> Self {
        self.failover_tx = Some(tx);
        self
    }

    /// # Errors
    ///
    /// Returns an error if the language-model operation cannot be completed.
    pub fn build(self) -> Result<FailoverConfig, anyhow::Error> {
        let primary = self
            .primary
            .ok_or_else(|| anyhow::anyhow!("primary(provider, model) required"))?;
        Ok(FailoverConfig {
            primary,
            fallbacks: self.fallbacks,
            auth_profiles: self.auth_profiles,
            cooldown_base_seconds: self.cooldown_base_seconds,
            failover_tx: self.failover_tx,
        })
    }
}

impl FailoverConfig {
    /// Fluent builder for failover config.
    #[must_use]
    pub fn builder() -> FailoverConfigBuilder {
        FailoverConfigBuilder::new()
    }
}

/// A single entry in the failover chain (provider type + model + auth manager).
#[derive(Debug)]
pub struct FailoverEntry {
    pub pair: ProviderModelPair,
    pub auth_manager: AuthProfileManager,
}

/// Reason for switching to the next provider.
#[derive(Debug, Clone)]
pub enum FailoverReason {
    RateLimited { retry_after: Option<Duration> },
    AuthError,
    ModelUnavailable,
    Timeout,
    ServerError { status: u16 },
}

/// Event emitted when failover switches to another provider.
#[derive(Debug, Clone)]
pub struct FailoverEvent {
    pub from_provider: String,
    pub from_model: String,
    pub to_provider: String,
    pub to_model: String,
    pub reason: FailoverReason,
    pub timestamp: Instant,
}

/// Wraps multiple provider+model+credential combos with automatic failover.
/// Implements the `Provider` trait so it can be used as a drop-in replacement.
type CachedProviderSlot = RwLock<Option<(usize, Arc<dyn Provider>)>>;

pub struct FailoverProvider {
    entries: Vec<FailoverEntry>,
    current: AtomicUsize,
    current_provider: RwLock<Option<Arc<dyn Provider>>>,
    cached_providers: Vec<CachedProviderSlot>,
    base_url: Option<String>,
    timeout: Option<Duration>,
    max_retries: Option<usize>,
    failover_tx: Option<tokio::sync::broadcast::Sender<FailoverEvent>>,
    provider_factory: ProviderFactoryFn,
}

impl std::fmt::Debug for FailoverProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FailoverProvider")
            .field("entries_len", &self.entries.len())
            .field("current", &self.current.load(Ordering::Relaxed))
            .field("current_provider", &"<cached provider>")
            .field("cached_providers", &self.cached_providers.len())
            .field("base_url", &self.base_url)
            .field("timeout", &self.timeout)
            .field("max_retries", &self.max_retries)
            .field("failover_tx", &self.failover_tx.is_some())
            .field("provider_factory", &"<provider factory>")
            .finish()
    }
}

impl FailoverProvider {
    /// Create a failover provider with the given entries and options.
    pub fn new(
        entries: Vec<FailoverEntry>,
        base_url: Option<String>,
        timeout: Option<Duration>,
        max_retries: Option<usize>,
        failover_tx: Option<tokio::sync::broadcast::Sender<FailoverEvent>>,
        provider_factory: Option<ProviderFactoryFn>,
    ) -> Self {
        let cached_providers = entries.iter().map(|_| RwLock::new(None)).collect();
        Self {
            entries,
            current: AtomicUsize::new(0),
            current_provider: RwLock::new(None),
            cached_providers,
            base_url,
            timeout,
            max_retries,
            failover_tx,
            provider_factory: provider_factory.unwrap_or_else(default_provider_factory),
        }
    }

    /// Create a provider for a single entry with the given API key.
    async fn create_provider_for_entry(
        &self,
        entry: &FailoverEntry,
        api_key: String,
    ) -> Result<Arc<dyn Provider>> {
        let cred_provider: Arc<dyn CredentialProvider> =
            Arc::new(StaticCredentialProvider::new(Credentials::ApiKey(api_key)));
        (self.provider_factory)(
            entry.pair.provider,
            entry.pair.model.clone(),
            cred_provider,
            self.base_url.clone(),
            self.timeout,
            self.max_retries,
        )
        .await
    }

    async fn get_or_create_provider_for_entry(
        &self,
        index: usize,
        entry: &FailoverEntry,
        profile_idx: usize,
        api_key: String,
    ) -> Result<Arc<dyn Provider>> {
        if let Some((cached_profile, provider)) = self.cached_providers[index].read().await.clone()
            && cached_profile == profile_idx
        {
            return Ok(provider);
        }

        let provider = self.create_provider_for_entry(entry, api_key).await?;
        *self.cached_providers[index].write().await = Some((profile_idx, provider.clone()));
        Ok(provider)
    }

    fn emit_failover(
        tx: Option<&tokio::sync::broadcast::Sender<FailoverEvent>>,
        from_provider: &str,
        from_model: &str,
        to_provider: &str,
        to_model: &str,
        reason: FailoverReason,
    ) {
        if let Some(sender) = tx {
            let _ = sender.send(FailoverEvent {
                from_provider: from_provider.to_string(),
                from_model: from_model.to_string(),
                to_provider: to_provider.to_string(),
                to_model: to_model.to_string(),
                reason,
                timestamp: Instant::now(),
            });
        }
    }

    async fn resolve_provider(&self) -> Result<(usize, Arc<dyn Provider>)> {
        let n = self.entries.len();
        if n == 0 {
            return Err(anyhow::anyhow!("FailoverProvider has no entries"));
        }

        let mut last_error: Option<anyhow::Error> = None;
        let start_index = self.current.load(Ordering::Relaxed) % n;

        for k in 0..n {
            let i = (start_index + k) % n;
            let entry = &self.entries[i];
            let Some((profile_idx, key)) = entry.auth_manager.next_credential().await else {
                continue;
            };

            match self
                .get_or_create_provider_for_entry(i, entry, profile_idx, key)
                .await
            {
                Ok(provider) => {
                    *self.current_provider.write().await = Some(provider.clone());
                    self.current.store(i, Ordering::Relaxed);
                    return Ok((i, provider));
                }
                Err(error) => {
                    last_error = Some(error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All failover entries failed")))
    }
}

#[async_trait]
impl Provider for FailoverProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let n = self.entries.len();
        if n == 0 {
            return Err(anyhow::anyhow!("FailoverProvider has no entries"));
        }
        let mut last_error: Option<anyhow::Error> = None;
        let start_index = self.current.load(Ordering::Relaxed) % n;

        for k in 0..n {
            let i = (start_index + k) % n;
            let entry = &self.entries[i];
            let Some((profile_idx, key)) = entry.auth_manager.next_credential().await else {
                continue;
            };
            let provider = match self
                .get_or_create_provider_for_entry(i, entry, profile_idx, key)
                .await
            {
                Ok(provider) => provider,
                Err(error) => {
                    last_error = Some(error);
                    continue;
                }
            };
            match provider.complete(request.clone()).await {
                Ok(resp) => {
                    entry.auth_manager.mark_success(profile_idx).await;
                    self.current.store(i, Ordering::Relaxed);
                    *self.current_provider.write().await = Some(provider);
                    return Ok(resp);
                }
                Err(e) => {
                    let retry_err = classify_error_for_failover(&e);
                    entry
                        .auth_manager
                        .mark_failure(profile_idx, &retry_err)
                        .await;
                    last_error = Some(e);

                    let try_next = matches!(
                        retry_err,
                        RetryError::RateLimited { .. }
                            | RetryError::AuthError { .. }
                            | RetryError::ModelUnavailable { .. }
                            | RetryError::Transient { .. }
                    );
                    if !try_next {
                        return Err(last_error.unwrap_or_else(|| {
                            anyhow::anyhow!("non-retryable failover error (last_error missing)")
                        }));
                    }

                    let from_provider = format!("{:?}", entry.pair.provider);
                    let from_model = entry.pair.model.as_str();
                    let next_i = (i + 1) % n;
                    if next_i != start_index {
                        let next = &self.entries[next_i];
                        let reason = match &retry_err {
                            RetryError::RateLimited { retry_after, .. } => {
                                FailoverReason::RateLimited {
                                    retry_after: *retry_after,
                                }
                            }
                            RetryError::AuthError { .. } => FailoverReason::AuthError,
                            RetryError::ModelUnavailable { .. } => FailoverReason::ModelUnavailable,
                            RetryError::Transient { status_code, .. } => {
                                FailoverReason::ServerError {
                                    status: *status_code,
                                }
                            }
                            RetryError::Permanent { .. } => {
                                FailoverReason::ServerError { status: 0 }
                            }
                        };
                        Self::emit_failover(
                            self.failover_tx.as_ref(),
                            &from_provider,
                            from_model,
                            &format!("{:?}", next.pair.provider),
                            &next.pair.model,
                            reason,
                        );
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All failover entries failed")))
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send + '_>> {
        Box::pin(async_stream::try_stream! {
            let (_, provider) = self.resolve_provider().await?;
            let mut inner_stream = provider.stream(request);
            while let Some(item) = futures_util::StreamExt::next(&mut inner_stream).await {
                yield item?;
            }
        })
    }

    fn name(&self) -> &'static str {
        "failover"
    }

    fn supports_native_tools(&self) -> bool {
        self.current_provider
            .try_read()
            .ok()
            .and_then(|provider| {
                provider
                    .as_ref()
                    .map(|provider| provider.supports_native_tools())
            })
            .unwrap_or(false)
    }

    fn supports_streaming(&self) -> bool {
        self.current_provider
            .try_read()
            .ok()
            .and_then(|provider| {
                provider
                    .as_ref()
                    .map(|provider| provider.supports_streaming())
            })
            .unwrap_or(true)
    }

    fn supports_vision(&self) -> bool {
        self.current_provider
            .try_read()
            .ok()
            .and_then(|provider| provider.as_ref().map(|provider| provider.supports_vision()))
            .unwrap_or(false)
    }

    fn supports_audio_input(&self) -> bool {
        self.current_provider
            .try_read()
            .ok()
            .and_then(|provider| {
                provider
                    .as_ref()
                    .map(|provider| provider.supports_audio_input())
            })
            .unwrap_or(false)
    }

    fn supports_document_input(&self) -> bool {
        self.current_provider
            .try_read()
            .ok()
            .and_then(|provider| {
                provider
                    .as_ref()
                    .map(|provider| provider.supports_document_input())
            })
            .unwrap_or(false)
    }

    fn supported_media_types(&self) -> Vec<&str> {
        vec![]
    }

    fn context_window(&self) -> Option<usize> {
        self.current_provider.try_read().ok().and_then(|provider| {
            provider
                .as_ref()
                .and_then(|provider| provider.context_window())
        })
    }

    fn model_load_probe(&self) -> Option<crate::llm::provider_utils::first_token::ModelLoadProbe> {
        self.current_provider.try_read().ok().and_then(|provider| {
            provider
                .as_ref()
                .and_then(|provider| provider.model_load_probe())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_error_keeps_connection_invalid_transient() {
        let err = anyhow::anyhow!("connection invalid after reset");
        assert!(matches!(
            classify_error_for_failover(&err),
            RetryError::Transient { .. }
        ));
    }

    #[test]
    fn classify_error_marks_invalid_request_permanent() {
        let err = anyhow::anyhow!("invalid_request_error: missing field");
        assert!(matches!(
            classify_error_for_failover(&err),
            RetryError::Permanent { .. }
        ));
        let err = anyhow::anyhow!("HTTP 400 bad request");
        assert!(matches!(
            classify_error_for_failover(&err),
            RetryError::Permanent { .. }
        ));
    }

    #[test]
    fn classify_error_detects_rate_limit_and_auth() {
        assert!(matches!(
            classify_error_for_failover(&anyhow::anyhow!("rate limit exceeded")),
            RetryError::RateLimited { .. }
        ));
        assert!(matches!(
            classify_error_for_failover(&anyhow::anyhow!("401 unauthorized")),
            RetryError::AuthError { .. }
        ));
    }
}
