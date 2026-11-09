//! Retry strategy abstraction for HTTP client retry logic.
//!
//! This module provides a unified interface for retry strategies in HTTP operations.

use std::time::Duration;

use crate::utils::numeric::u64_to_f64;

/// Error types for retry logic
#[derive(Debug, Clone)]
pub enum RetryError {
    /// Rate limited error with optional retry-after hint
    RateLimited {
        retry_after: Option<Duration>,
        message: String,
    },
    /// Transient error that might succeed on retry
    Transient { status_code: u16, message: String },
    /// Permanent error that won't succeed on retry
    Permanent { error: String },
}

impl RetryError {
    /// Check if this error is retryable
    #[must_use]
    pub const fn is_retryable(&self) -> bool {
        matches!(self, Self::RateLimited { .. } | Self::Transient { .. })
    }

    /// Get the retry-after hint if available
    #[must_use]
    pub const fn retry_after(&self) -> Option<Duration> {
        match self {
            Self::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Get error message for logging (returns reference to avoid cloning)
    #[must_use]
    pub const fn message(&self) -> &str {
        match self {
            Self::RateLimited { message, .. } | Self::Transient { message, .. } => message.as_str(),
            Self::Permanent { error } => error.as_str(),
        }
    }
}

/// Trait for retry strategies that determine when and how to retry operations
pub trait RetryStrategy: Send + Sync + std::fmt::Debug {
    /// Determine if an error should be retried
    ///
    /// # Arguments
    /// * `attempt` - Current attempt number (0-indexed)
    /// * `error` - The error that occurred
    ///
    /// # Returns
    /// `true` if the operation should be retried, `false` otherwise
    fn should_retry(&self, attempt: u32, error: &RetryError) -> bool;

    /// Calculate backoff duration for next attempt
    ///
    /// # Arguments
    /// * `attempt` - Current attempt number (0-indexed)
    /// * `error` - The error that occurred
    ///
    /// # Returns
    /// Duration to wait before next retry attempt
    fn calculate_backoff(&self, attempt: u32, error: &RetryError) -> Duration;

    /// Get strategy name for logging
    fn name(&self) -> &str;

    /// Maximum attempts allowed
    fn max_attempts(&self) -> u32;
}

/// Exponential backoff retry strategy
#[derive(Debug, Clone)]
pub struct ExponentialBackoffStrategy {
    max_attempts: u32,
    initial_backoff_ms: u64,
    max_backoff_ms: u64,
    backoff_multiplier: f64,
    jitter: bool,
}

impl ExponentialBackoffStrategy {
    /// Create a new exponential backoff strategy
    ///
    /// # Arguments
    /// * `max_attempts` - Maximum number of retry attempts
    /// * `initial_backoff_ms` - Initial backoff duration in milliseconds
    /// * `max_backoff_ms` - Maximum backoff duration in milliseconds
    /// * `backoff_multiplier` - Multiplier for exponential backoff
    /// * `jitter` - Whether to add jitter to backoff duration
    #[must_use]
    pub const fn new(
        max_attempts: u32,
        initial_backoff_ms: u64,
        max_backoff_ms: u64,
        backoff_multiplier: f64,
        jitter: bool,
    ) -> Self {
        Self {
            max_attempts,
            initial_backoff_ms,
            max_backoff_ms,
            backoff_multiplier,
            jitter,
        }
    }
}

impl Default for ExponentialBackoffStrategy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60000,
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryStrategy for ExponentialBackoffStrategy {
    fn should_retry(&self, attempt: u32, error: &RetryError) -> bool {
        if !error.is_retryable() {
            return false;
        }
        attempt < self.max_attempts
    }

    fn calculate_backoff(&self, attempt: u32, error: &RetryError) -> Duration {
        // If we have a retry-after hint, use it (capped at max_backoff)
        if let Some(hint) = error.retry_after() {
            return hint.min(Duration::from_millis(self.max_backoff_ms));
        }

        // Calculate exponential backoff
        let multiplier = self.backoff_multiplier.powf(f64::from(attempt));
        let base_backoff_ms = u64_to_f64(self.initial_backoff_ms) * multiplier;
        let max_backoff = Duration::from_millis(self.max_backoff_ms);
        let mut backoff = Duration::try_from_secs_f64(base_backoff_ms / 1000.0)
            .unwrap_or(max_backoff)
            .min(max_backoff);

        // Add jitter if enabled (±25% randomization)
        if self.jitter {
            let jitter_factor = fastrand::f64().mul_add(0.5, 0.75); // Random between 0.75 and 1.25
            backoff = backoff.mul_f64(jitter_factor);
        }

        backoff
    }

    fn name(&self) -> &'static str {
        "exponential-backoff"
    }

    fn max_attempts(&self) -> u32 {
        self.max_attempts
    }
}

/// Retry strategy that always refuses to retry.
///
/// [`should_retry`](RetryStrategy::should_retry) always returns `false`, so
/// the first error produced by a request is returned to the caller
/// unchanged. This is primarily useful for:
///
/// - unit tests that assert on the exact first error,
/// - synchronous code paths that cannot tolerate any additional latency,
/// - callers that implement their own retry logic at a higher layer and
///   want the HTTP bridge to stay out of the way.
///
/// Because retries are disabled, a `Retry-After` header on a 429 response
/// is not honoured: the caller receives the classified
/// [`RetryError::RateLimited`] immediately.
#[derive(Debug, Clone)]
pub struct NoRetryStrategy;

impl NoRetryStrategy {
    /// Create a new no-retry strategy
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Default for NoRetryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl RetryStrategy for NoRetryStrategy {
    fn should_retry(&self, _attempt: u32, _error: &RetryError) -> bool {
        false
    }

    fn calculate_backoff(&self, _attempt: u32, _error: &RetryError) -> Duration {
        Duration::ZERO
    }

    fn name(&self) -> &'static str {
        "no-retry"
    }

    fn max_attempts(&self) -> u32 {
        0
    }
}
