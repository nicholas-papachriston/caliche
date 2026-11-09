#![expect(
    clippy::expect_used,
    reason = "static Regex/LazyLock literals are reviewed constants; failure is a programmer bug"
)]

use crate::traits::retry::{RetryError as TraitsRetryError, RetryStrategy as TraitsRetryStrategy};
use regex::Regex;
use std::sync::LazyLock;
use std::time::Duration;

/// Matches "try again in 21.908s" style retry-after messages.
static RE_TRY_AGAIN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"try again in (\d+\.?\d*)s").expect("RE_TRY_AGAIN"));

/// Matches "retry after 30s" style retry-after messages.
static RE_RETRY_AFTER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"retry after (\d+)s").expect("RE_RETRY_AFTER"));

// Re-export RetryError and RetryStrategy from traits module.
// This is intentional - the traits module is the source of truth for retry types.
// Re-exporting here provides convenient access for LLM provider implementations.
pub use crate::traits::retry::{RetryError, RetryStrategy};

/// Retry an async operation with a retry strategy
///
/// # Arguments
/// * `strategy` - Retry strategy to use
/// * `operation` - Async operation to retry (returns Result<T, `RetryError`>)
///
/// # Returns
/// * `Ok(T)` - Success result from operation
/// * `Err(anyhow::Error)` - Final error after all retries exhausted
/// # Errors
///
/// Returns an error if the language-model operation cannot be completed.
pub async fn retry_with_backoff<T, F, Fut>(
    strategy: &dyn TraitsRetryStrategy,
    mut operation: F,
) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, TraitsRetryError>>,
{
    let mut attempt = 0;
    let mut last_error: Option<TraitsRetryError> = None;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(error) => {
                if !strategy.should_retry(attempt, &error) {
                    // Strategy says don't retry, fail immediately
                    return Err(match error {
                        TraitsRetryError::Permanent { error } => anyhow::anyhow!("{error}"),
                        e => anyhow::anyhow!("{}", e.message()),
                    });
                }

                if attempt >= strategy.max_attempts() {
                    // Max retries exceeded
                    let msg = last_error
                        .as_ref()
                        .map_or("Unknown error", RetryError::message);
                    return Err(anyhow::anyhow!(
                        "Max retries ({}) exceeded. Last error: {}",
                        strategy.max_attempts(),
                        msg
                    ));
                }

                // Calculate backoff using strategy
                let backoff = strategy.calculate_backoff(attempt, &error);

                // Log retry attempt
                match &error {
                    TraitsRetryError::RateLimited {
                        retry_after,
                        message,
                    } => {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            strategy = strategy.name(),
                            backoff_ms = backoff.as_millis(),
                            retry_after_ms = retry_after.map(|d| d.as_millis()),
                            error = %message,
                            "Rate limit encountered, retrying after backoff"
                        );
                    }
                    TraitsRetryError::Transient {
                        status_code,
                        message,
                    } => {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            strategy = strategy.name(),
                            backoff_ms = backoff.as_millis(),
                            status_code = status_code,
                            error = %message,
                            "Transient error encountered, retrying after backoff"
                        );
                    }
                    TraitsRetryError::AuthError { message } => {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            strategy = strategy.name(),
                            error = %message,
                            "Auth error encountered, will not retry after this attempt"
                        );
                    }
                    TraitsRetryError::ModelUnavailable { message } => {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            strategy = strategy.name(),
                            error = %message,
                            "Model unavailable"
                        );
                    }
                    TraitsRetryError::Permanent { error } => {
                        tracing::error!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            strategy = strategy.name(),
                            error = %error,
                            "Permanent error, failing immediately"
                        );
                    }
                }

                // Wait before retry
                tokio::time::sleep(backoff).await;

                last_error = Some(error);
                attempt += 1;
            }
        }
    }
}

/// Retry an async operation with a custom error classifier.
/// Use this when the operation returns `Result<T, E>` and you need to map `E` to `RetryError`.
///
/// # Arguments
/// * `strategy` - Retry strategy to use
/// * `classifier` - Function that converts operation errors to `RetryError`
/// * `operation` - Async operation to retry (returns `Result<T, E>`)
/// # Errors
///
/// Returns an error if the language-model operation cannot be completed.
pub async fn retry_with_classifier<T, E, C, F, Fut>(
    strategy: &dyn TraitsRetryStrategy,
    classifier: C,
    mut operation: F,
) -> anyhow::Result<T>
where
    C: Fn(&E) -> TraitsRetryError,
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>> + Send,
{
    let mut attempt = 0;
    let mut last_error: Option<TraitsRetryError> = None;

    loop {
        let result = operation().await.map_err(|e| classifier(&e));
        match result {
            Ok(t) => return Ok(t),
            Err(error) => {
                if !strategy.should_retry(attempt, &error) {
                    return Err(match error {
                        TraitsRetryError::Permanent { error } => anyhow::anyhow!("{error}"),
                        e => anyhow::anyhow!("{}", e.message()),
                    });
                }
                if attempt >= strategy.max_attempts() {
                    let msg = last_error
                        .as_ref()
                        .map_or("Unknown error", RetryError::message);
                    return Err(anyhow::anyhow!(
                        "Max retries ({}) exceeded. Last error: {}",
                        strategy.max_attempts(),
                        msg
                    ));
                }
                let backoff = strategy.calculate_backoff(attempt, &error);
                match &error {
                    TraitsRetryError::RateLimited { message, .. } => {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            error = %message,
                            "Rate limit encountered, retrying after backoff"
                        );
                    }
                    TraitsRetryError::Transient {
                        status_code,
                        message,
                        ..
                    } => {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            status_code = status_code,
                            error = %message,
                            "Transient error encountered, retrying after backoff"
                        );
                    }
                    TraitsRetryError::AuthError { message } => {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            error = %message,
                            "Auth error encountered, will not retry after this attempt"
                        );
                    }
                    TraitsRetryError::ModelUnavailable { message } => {
                        tracing::warn!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            error = %message,
                            "Model unavailable"
                        );
                    }
                    TraitsRetryError::Permanent { error } => {
                        tracing::error!(
                            attempt = attempt + 1,
                            max_retries = strategy.max_attempts(),
                            error = %error,
                            "Permanent error, failing immediately"
                        );
                    }
                }
                tokio::time::sleep(backoff).await;
                last_error = Some(error);
                attempt += 1;
            }
        }
    }
}

/// Parse retry-after duration from error message.
///
/// Recognises:
/// - `"try again in X.XXs"` (Groq, `OpenAI` style)
/// - `"retry after Xs"` (generic style)
pub fn parse_retry_after_from_message(message: &str) -> Option<Duration> {
    if let Some(captures) = RE_TRY_AGAIN.captures(message)
        && let Some(seconds_str) = captures.get(1)
        && let Ok(seconds) = seconds_str.as_str().parse::<f64>()
    {
        return Some(Duration::from_secs_f64(seconds));
    }

    if let Some(captures) = RE_RETRY_AFTER.captures(message)
        && let Some(seconds_str) = captures.get(1)
        && let Ok(seconds) = seconds_str.as_str().parse::<u64>()
    {
        return Some(Duration::from_secs(seconds));
    }

    None
}
