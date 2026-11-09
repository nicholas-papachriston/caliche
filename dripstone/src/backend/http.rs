#![cfg(any(
    feature = "qdrant-backend",
    feature = "chroma-backend",
    feature = "weaviate-backend",
    feature = "milvus-backend",
    feature = "pinecone-backend"
))]

use reqwest::{Client, Response, header::HeaderMap};
use std::time::Duration;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc2822;

use crate::backend::retry::{RetryError, RetryStrategy};
use crate::errors::DripstoneError;

pub struct HttpClient {
    inner: Client,
    base_url: String,
    retry_strategy: Option<Box<dyn RetryStrategy>>,
}

/// Configuration for `HttpClient`
#[derive(Default, Debug)]
pub struct HttpClientConfig {
    /// Request timeout duration
    pub timeout: Option<Duration>,
    /// Default headers to include in all requests
    pub default_headers: Option<HeaderMap>,
    /// Connection pool size per host
    pub connection_pool_size: Option<usize>,
    /// Pool idle timeout
    pub pool_idle_timeout: Option<Duration>,
    /// Retry configuration
    pub retry_config: Option<RetryConfig>,
}

/// Retry configuration for HTTP clients
#[derive(Debug)]
pub struct RetryConfig {
    /// Retry strategy to use
    pub strategy: Box<dyn RetryStrategy>,
}

impl RetryConfig {
    /// Create a new `RetryConfig` with a custom strategy
    #[must_use]
    pub fn new(strategy: Box<dyn RetryStrategy>) -> Self {
        Self { strategy }
    }

    /// Create a `RetryConfig` with exponential backoff strategy
    #[must_use]
    pub fn exponential(
        max_attempts: u32,
        initial_backoff_ms: u64,
        max_backoff_ms: u64,
        backoff_multiplier: f64,
        jitter: bool,
    ) -> Self {
        use crate::backend::retry::ExponentialBackoffStrategy;
        Self {
            strategy: Box::new(ExponentialBackoffStrategy::new(
                max_attempts,
                initial_backoff_ms,
                max_backoff_ms,
                backoff_multiplier,
                jitter,
            )),
        }
    }

    /// Create a `RetryConfig` with default exponential backoff
    #[must_use]
    pub fn default_exponential() -> Self {
        use crate::backend::retry::ExponentialBackoffStrategy;
        Self {
            strategy: Box::<ExponentialBackoffStrategy>::default(),
        }
    }

    /// Create a `RetryConfig` with no retry strategy
    #[must_use]
    pub fn no_retry() -> Self {
        use crate::backend::retry::NoRetryStrategy;
        Self {
            strategy: Box::new(NoRetryStrategy::new()),
        }
    }
}

impl HttpClientConfig {
    /// Create a new `HttpClientConfig`
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set request timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set default headers
    #[must_use]
    pub fn with_headers(mut self, headers: HeaderMap) -> Self {
        self.default_headers = Some(headers);
        self
    }

    /// Set connection pool size
    #[must_use]
    pub const fn with_connection_pool_size(mut self, size: usize) -> Self {
        self.connection_pool_size = Some(size);
        self
    }

    /// Set pool idle timeout
    #[must_use]
    pub const fn with_pool_idle_timeout(mut self, timeout: Duration) -> Self {
        self.pool_idle_timeout = Some(timeout);
        self
    }

    /// Set retry configuration
    #[must_use]
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(config);
        self
    }
}

impl HttpClient {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(
        base_url: impl Into<String>,
        timeout: Option<Duration>,
    ) -> Result<Self, DripstoneError> {
        Self::with_options(
            base_url,
            HttpClientConfig {
                timeout,
                ..Default::default()
            },
        )
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn with_headers(
        base_url: impl Into<String>,
        timeout: Option<Duration>,
        headers: Option<HeaderMap>,
    ) -> Result<Self, DripstoneError> {
        Self::with_options(
            base_url,
            HttpClientConfig {
                timeout,
                default_headers: headers,
                ..Default::default()
            },
        )
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn with_options(
        base_url: impl Into<String>,
        options: HttpClientConfig,
    ) -> Result<Self, DripstoneError> {
        let mut normalized = base_url.into();
        if normalized.is_empty() {
            return Err(DripstoneError::Misconfiguration(
                "http client base url cannot be empty".to_string(),
            ));
        }
        while normalized.ends_with('/') {
            normalized.pop();
        }

        let mut builder = Client::builder();
        if let Some(value) = options.timeout {
            builder = builder.timeout(value);
        }
        if let Some(map) = options.default_headers {
            builder = builder.default_headers(map);
        }
        if let Some(max_idle) = options.connection_pool_size {
            builder = builder.pool_max_idle_per_host(max_idle);
        }
        if let Some(idle_timeout) = options.pool_idle_timeout {
            builder = builder.pool_idle_timeout(idle_timeout);
        }

        let inner = builder.build().map_err(DripstoneError::from)?;
        let retry_strategy = options.retry_config.map(|config| config.strategy);
        Ok(Self {
            inner,
            base_url: normalized,
            retry_strategy,
        })
    }

    /// Create an `HttpClient` with configuration
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn with_config(
        base_url: impl Into<String>,
        config: HttpClientConfig,
    ) -> Result<Self, DripstoneError> {
        Self::with_options(base_url, config)
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn get(&self, path: &str) -> Result<Response, DripstoneError> {
        self.execute_with_retry(|| self.inner.get(self.compose(path)).send())
            .await
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn delete(&self, path: &str) -> Result<Response, DripstoneError> {
        self.execute_with_retry(|| self.inner.delete(self.compose(path)).send())
            .await
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn post_json<T>(&self, path: &str, payload: &T) -> Result<Response, DripstoneError>
    where
        T: serde::Serialize + Sync + ?Sized,
    {
        let url = self.compose(path);
        let payload_clone = serde_json::to_value(payload).map_err(DripstoneError::from)?;
        self.execute_with_retry(|| self.inner.post(&url).json(&payload_clone).send())
            .await
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn put_json<T>(&self, path: &str, payload: &T) -> Result<Response, DripstoneError>
    where
        T: serde::Serialize + Sync + ?Sized,
    {
        let url = self.compose(path);
        let payload_clone = serde_json::to_value(payload).map_err(DripstoneError::from)?;
        self.execute_with_retry(|| self.inner.put(&url).json(&payload_clone).send())
            .await
    }

    /// Execute an HTTP operation with retry logic
    async fn execute_with_retry<F, Fut>(&self, operation: F) -> Result<Response, DripstoneError>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<Response, reqwest::Error>>,
    {
        let mut attempt = 0u32;
        loop {
            let result = operation().await;

            match result {
                Ok(response) => {
                    // Check if the response status indicates an error that should be retried
                    let status = response.status();
                    if status.is_success() {
                        return Ok(response);
                    }

                    // Convert HTTP status to RetryError
                    // Note: We need to clone status before consuming response
                    let status_code = status.as_u16();
                    let retry_error = classify_http_error(status_code, response).await?;

                    // Check if we should retry
                    if let Some(ref strategy) = self.retry_strategy
                        && strategy.should_retry(attempt, &retry_error)
                    {
                        let backoff = strategy.calculate_backoff(attempt, &retry_error);
                        attempt += 1;
                        #[cfg(feature = "tokio")]
                        tokio::time::sleep(backoff).await;
                        #[cfg(not(feature = "tokio"))]
                        std::thread::sleep(backoff);
                        continue;
                    }

                    // No retry strategy or shouldn't retry - return error
                    return Err(match retry_error {
                        RetryError::RateLimited { message, .. } => {
                            DripstoneError::storage(format!("rate limited: {message}"))
                        }
                        RetryError::Transient {
                            status_code,
                            message,
                        } => DripstoneError::storage(format!(
                            "transient error ({status_code}): {message}"
                        )),
                        RetryError::Permanent { error } => {
                            DripstoneError::storage(format!("permanent error: {error}"))
                        }
                    });
                }
                Err(e) => {
                    // Network or other reqwest error
                    let retry_error = classify_reqwest_error(&e);

                    // Check if we should retry
                    if let Some(ref strategy) = self.retry_strategy
                        && strategy.should_retry(attempt, &retry_error)
                    {
                        let backoff = strategy.calculate_backoff(attempt, &retry_error);
                        attempt += 1;
                        #[cfg(feature = "tokio")]
                        tokio::time::sleep(backoff).await;
                        #[cfg(not(feature = "tokio"))]
                        std::thread::sleep(backoff);
                        continue;
                    }

                    // No retry strategy or shouldn't retry - return error
                    return Err(DripstoneError::from(e));
                }
            }
        }
    }

    fn compose(&self, path: &str) -> String {
        if path.starts_with('/') {
            format!("{}{}", self.base_url, path)
        } else {
            format!("{}/{}", self.base_url, path)
        }
    }
}

/// Classify HTTP status code and response into a `RetryError`
async fn classify_http_error(
    status_code: u16,
    response: Response,
) -> Result<RetryError, DripstoneError> {
    let retry_after = response
        .headers()
        .get("retry-after")
        .and_then(|h| h.to_str().ok())
        .and_then(parse_retry_after);

    // Try to get error message from response body
    let message = response
        .text()
        .await
        .unwrap_or_else(|_| format!("HTTP {status_code}"));

    match status_code {
        429 => Ok(RetryError::RateLimited {
            retry_after,
            message,
        }),
        408 | 500..=599 => Ok(RetryError::Transient {
            status_code,
            message,
        }),
        _ => Ok(RetryError::Permanent {
            error: format!("HTTP {status_code}: {message}"),
        }),
    }
}

/// Parse a `Retry-After` header value.
///
/// Per RFC 7231 §7.1.3 the value is either a non-negative integer number of
/// seconds (delay-seconds) or an HTTP-date. We accept both:
///
/// - delay-seconds: `"120"` → `Some(Duration::from_secs(120))`
/// - IMF-fixdate (RFC 7231 §7.1.1.1), RFC 850 format, and ANSI C asctime
///   format are all parsed via the RFC 2822 date parser in `time`, which
///   handles all three common HTTP-date spellings in practice. The returned
///   `Duration` is the delta from "now" to the parsed timestamp, clamped at
///   zero for timestamps already in the past.
///
/// Returns `None` when neither format parses, which causes the retry
/// strategy to fall back to its configured backoff.
fn parse_retry_after(header: &str) -> Option<Duration> {
    let trimmed = header.trim();
    if let Ok(secs) = trimmed.parse::<u64>() {
        return Some(Duration::from_secs(secs));
    }

    let target = OffsetDateTime::parse(trimmed, &Rfc2822).ok()?;
    let now = OffsetDateTime::now_utc();
    let delta = target - now;
    if delta.is_negative() {
        Some(Duration::from_secs(0))
    } else {
        let seconds = u64::try_from(delta.whole_seconds().max(0)).unwrap_or(u64::MAX);
        Some(Duration::from_secs(seconds))
    }
}

/// Classify `reqwest::Error` into a `RetryError`
fn classify_reqwest_error(error: &reqwest::Error) -> RetryError {
    if error.is_timeout() {
        RetryError::Transient {
            status_code: 0,
            message: format!("request timeout: {error}"),
        }
    } else if error.is_connect() || error.is_request() {
        RetryError::Transient {
            status_code: 0,
            message: format!("network error: {error}"),
        }
    } else {
        RetryError::Permanent {
            error: format!("request error: {error}"),
        }
    }
}
