//! Retry strategy abstraction for error handling and backoff.
//!
//! This module provides a unified interface for retry strategies across
//! different operations. Implement the `RetryStrategy` trait to add custom
//! retry behaviors like adaptive retry, circuit breakers, or custom backoff algorithms.
//!
//! # Examples
//!
//! ## Using Exponential Backoff
//!
//! ```rust,no_run
//! use caliche::traits::retry::{RetryStrategy, ExponentialBackoffStrategy};
//! use std::time::Duration;
//!
//! let strategy = ExponentialBackoffStrategy::new(
//!     3,                                    // max attempts
//!     1000,                                 // initial backoff (milliseconds)
//!     60000,                                // max backoff (milliseconds)
//!     2.0,                                  // multiplier
//!     true,                                // jitter
//! );
//! ```
//!
//! ## Using Retry Strategy Builder
//!
//! ```rust,no_run
//! use caliche::traits::retry::RetryStrategyBuilder;
//! use std::time::Duration;
//!
//! let strategy = RetryStrategyBuilder::exponential()
//!     .max_attempts(5)
//!     .initial_backoff(Duration::from_secs(2))
//!     .max_backoff(Duration::from_secs(60))
//!     .with_jitter(true)
//!     .build();
//! ```

use std::time::Duration;

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
    /// Authentication failure (invalid key, expired token, etc.)
    AuthError { message: String },
    /// Model not available (deprecated, region unavailable, etc.)
    ModelUnavailable { message: String },
}

impl RetryError {
    /// Check if this error is retryable (e.g. with a different credential or provider).
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            RetryError::RateLimited { .. } | RetryError::Transient { .. }
        )
    }

    /// Get the retry-after hint if available
    pub fn retry_after(&self) -> Option<Duration> {
        match self {
            RetryError::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
        }
    }

    /// Get error message for logging (zero-copy reference to the inner string).
    #[must_use]
    pub fn message(&self) -> &str {
        match self {
            RetryError::RateLimited { message, .. }
            | RetryError::Transient { message, .. }
            | RetryError::AuthError { message }
            | RetryError::ModelUnavailable { message } => message.as_str(),
            RetryError::Permanent { error } => error.as_str(),
        }
    }
}

impl std::fmt::Display for RetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for RetryError {}

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
    pub fn new(
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
            return Duration::from_millis(hint.as_millis().min(self.max_backoff_ms as u128) as u64);
        }

        // Calculate exponential backoff
        let base_backoff =
            self.initial_backoff_ms as f64 * self.backoff_multiplier.powi(attempt as i32);

        let mut backoff_ms = base_backoff.min(self.max_backoff_ms as f64) as u64;

        // Add jitter if enabled (±25% randomization)
        if self.jitter {
            let mut rng = fastrand::Rng::new();
            let jitter_factor = 0.75 + (rng.f64() * 0.5);
            backoff_ms = (backoff_ms as f64 * jitter_factor) as u64;
        }

        Duration::from_millis(backoff_ms)
    }

    fn name(&self) -> &str {
        "exponential-backoff"
    }

    fn max_attempts(&self) -> u32 {
        self.max_attempts
    }
}

/// Linear backoff retry strategy
#[derive(Debug, Clone)]
pub struct LinearBackoffStrategy {
    max_attempts: u32,
    initial_backoff_ms: u64,
    max_backoff_ms: u64,
    increment_ms: u64,
}

impl LinearBackoffStrategy {
    /// Create a new linear backoff strategy
    ///
    /// # Arguments
    /// * `max_attempts` - Maximum number of retry attempts
    /// * `initial_backoff_ms` - Initial backoff duration in milliseconds
    /// * `max_backoff_ms` - Maximum backoff duration in milliseconds
    /// * `increment_ms` - Amount to increment backoff by each attempt
    pub fn new(
        max_attempts: u32,
        initial_backoff_ms: u64,
        max_backoff_ms: u64,
        increment_ms: u64,
    ) -> Self {
        Self {
            max_attempts,
            initial_backoff_ms,
            max_backoff_ms,
            increment_ms,
        }
    }
}

impl RetryStrategy for LinearBackoffStrategy {
    fn should_retry(&self, attempt: u32, error: &RetryError) -> bool {
        if !error.is_retryable() {
            return false;
        }
        attempt < self.max_attempts
    }

    fn calculate_backoff(&self, attempt: u32, error: &RetryError) -> Duration {
        if let Some(hint) = error.retry_after() {
            return Duration::from_millis(hint.as_millis().min(self.max_backoff_ms as u128) as u64);
        }

        let backoff_ms = (self.initial_backoff_ms + (attempt as u64 * self.increment_ms))
            .min(self.max_backoff_ms);

        Duration::from_millis(backoff_ms)
    }

    fn name(&self) -> &str {
        "linear-backoff"
    }

    fn max_attempts(&self) -> u32 {
        self.max_attempts
    }
}

/// Fixed delay retry strategy
#[derive(Debug, Clone)]
pub struct FixedDelayStrategy {
    max_attempts: u32,
    delay: Duration,
}

impl FixedDelayStrategy {
    /// Create a new fixed delay strategy
    ///
    /// # Arguments
    /// * `max_attempts` - Maximum number of retry attempts
    /// * `delay` - Fixed delay between attempts
    pub fn new(max_attempts: u32, delay: Duration) -> Self {
        Self {
            max_attempts,
            delay,
        }
    }
}

impl RetryStrategy for FixedDelayStrategy {
    fn should_retry(&self, attempt: u32, error: &RetryError) -> bool {
        if !error.is_retryable() {
            return false;
        }
        attempt < self.max_attempts
    }

    fn calculate_backoff(&self, _attempt: u32, error: &RetryError) -> Duration {
        if let Some(hint) = error.retry_after() {
            return hint;
        }
        self.delay
    }

    fn name(&self) -> &str {
        "fixed-delay"
    }

    fn max_attempts(&self) -> u32 {
        self.max_attempts
    }
}

/// No retry strategy (never retries, useful for testing)
#[derive(Debug, Clone)]
pub struct NoRetryStrategy;

impl NoRetryStrategy {
    /// Create a new no-retry strategy
    pub fn new() -> Self {
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

    fn name(&self) -> &str {
        "no-retry"
    }

    fn max_attempts(&self) -> u32 {
        0
    }
}

/// Strategy type for builder
#[derive(Debug, Clone)]
enum StrategyType {
    Exponential,
    Linear,
    Fixed,
}

/// Builder for common retry strategies
#[derive(Debug, Clone)]
pub struct RetryStrategyBuilder {
    strategy_type: StrategyType,
    max_attempts: Option<u32>,
    initial_backoff: Option<Duration>,
    max_backoff: Option<Duration>,
    multiplier: Option<f64>,
    increment: Option<Duration>,
    fixed_delay: Option<Duration>,
    jitter: Option<bool>,
}

impl RetryStrategyBuilder {
    /// Create a builder for exponential backoff strategy
    pub fn exponential() -> Self {
        Self {
            strategy_type: StrategyType::Exponential,
            max_attempts: None,
            initial_backoff: None,
            max_backoff: None,
            multiplier: None,
            increment: None,
            fixed_delay: None,
            jitter: None,
        }
    }

    /// Create a builder for linear backoff strategy
    pub fn linear() -> Self {
        Self {
            strategy_type: StrategyType::Linear,
            max_attempts: None,
            initial_backoff: None,
            max_backoff: None,
            multiplier: None,
            increment: None,
            fixed_delay: None,
            jitter: None,
        }
    }

    /// Create a builder for fixed delay strategy
    pub fn fixed(delay: Duration) -> Self {
        Self {
            strategy_type: StrategyType::Fixed,
            max_attempts: None,
            initial_backoff: None,
            max_backoff: None,
            multiplier: None,
            increment: None,
            fixed_delay: Some(delay),
            jitter: None,
        }
    }

    /// Set maximum attempts
    pub fn max_attempts(mut self, n: u32) -> Self {
        self.max_attempts = Some(n);
        self
    }

    /// Set initial backoff duration
    pub fn initial_backoff(mut self, duration: Duration) -> Self {
        self.initial_backoff = Some(duration);
        self
    }

    /// Set maximum backoff duration
    pub fn max_backoff(mut self, duration: Duration) -> Self {
        self.max_backoff = Some(duration);
        self
    }

    /// Set backoff multiplier (for exponential)
    pub fn multiplier(mut self, multiplier: f64) -> Self {
        self.multiplier = Some(multiplier);
        self
    }

    /// Set increment duration (for linear)
    pub fn increment(mut self, duration: Duration) -> Self {
        self.increment = Some(duration);
        self
    }

    /// Enable or disable jitter
    pub fn with_jitter(mut self, enabled: bool) -> Self {
        self.jitter = Some(enabled);
        self
    }

    /// Create a builder from environment variables.
    ///
    /// Reads the following environment variables:
    /// - `LLM_MAX_RETRIES`: Maximum number of retry attempts (default: 3)
    /// - `LLM_INITIAL_BACKOFF_MS`: Initial backoff delay in milliseconds (default: 1000)
    /// - `LLM_MAX_BACKOFF_MS`: Maximum backoff delay in milliseconds (default: 60000)
    /// - `LLM_BACKOFF_MULTIPLIER`: Backoff multiplier for exponential backoff (default: 2.0)
    /// - `LLM_RETRY_JITTER`: Enable jitter to prevent thundering herd (default: true)
    ///
    /// Returns an exponential backoff builder configured with values from environment.
    pub fn from_env() -> Self {
        let defaults = ExponentialBackoffStrategy::default();

        let max_retries = std::env::var("LLM_MAX_RETRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| defaults.max_attempts());

        let initial_backoff_ms = std::env::var("LLM_INITIAL_BACKOFF_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.initial_backoff_ms);

        let max_backoff_ms = std::env::var("LLM_MAX_BACKOFF_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.max_backoff_ms);

        let multiplier = std::env::var("LLM_BACKOFF_MULTIPLIER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.backoff_multiplier);

        let jitter = std::env::var("LLM_RETRY_JITTER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(defaults.jitter);

        Self {
            strategy_type: StrategyType::Exponential,
            max_attempts: Some(max_retries),
            initial_backoff: Some(Duration::from_millis(initial_backoff_ms)),
            max_backoff: Some(Duration::from_millis(max_backoff_ms)),
            multiplier: Some(multiplier),
            increment: None,
            fixed_delay: None,
            jitter: Some(jitter),
        }
    }

    /// Build the retry strategy
    pub fn build(self) -> Box<dyn RetryStrategy> {
        match self.strategy_type {
            StrategyType::Exponential => {
                let defaults = ExponentialBackoffStrategy::default();
                Box::new(ExponentialBackoffStrategy::new(
                    self.max_attempts.unwrap_or_else(|| defaults.max_attempts()),
                    self.initial_backoff
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(defaults.initial_backoff_ms),
                    self.max_backoff
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(defaults.max_backoff_ms),
                    self.multiplier.unwrap_or(defaults.backoff_multiplier),
                    self.jitter.unwrap_or(defaults.jitter),
                ))
            }
            StrategyType::Linear => {
                let defaults = ExponentialBackoffStrategy::default();
                Box::new(LinearBackoffStrategy::new(
                    self.max_attempts.unwrap_or_else(|| defaults.max_attempts()),
                    self.initial_backoff
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(1000),
                    self.max_backoff
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(60000),
                    self.increment.map(|d| d.as_millis() as u64).unwrap_or(1000),
                ))
            }
            StrategyType::Fixed => {
                let defaults = ExponentialBackoffStrategy::default();
                Box::new(FixedDelayStrategy::new(
                    self.max_attempts.unwrap_or_else(|| defaults.max_attempts()),
                    self.fixed_delay.unwrap_or(Duration::from_secs(1)),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_backoff() {
        let strategy = ExponentialBackoffStrategy::default();
        let error = RetryError::Transient {
            status_code: 500,
            message: "Server error".to_string(),
        };

        assert!(strategy.should_retry(0, &error));
        assert!(strategy.should_retry(1, &error));
        assert!(!strategy.should_retry(3, &error));

        let backoff = strategy.calculate_backoff(0, &error);
        assert!(backoff.as_millis() > 0);
    }

    #[test]
    fn test_no_retry() {
        let strategy = NoRetryStrategy::new();
        let error = RetryError::Transient {
            status_code: 500,
            message: "Server error".to_string(),
        };

        assert!(!strategy.should_retry(0, &error));
    }

    #[test]
    fn test_retry_strategy_builder() {
        let strategy = RetryStrategyBuilder::exponential()
            .max_attempts(5)
            .initial_backoff(Duration::from_secs(2))
            .max_backoff(Duration::from_secs(60))
            .with_jitter(true)
            .build();

        assert_eq!(strategy.max_attempts(), 5);
        assert_eq!(strategy.name(), "exponential-backoff");
    }

    #[test]
    fn test_retry_strategy_from_env() {
        // Test with default values (no env vars set)
        let strategy = RetryStrategyBuilder::from_env().build();
        let defaults = ExponentialBackoffStrategy::default();
        assert_eq!(strategy.max_attempts(), defaults.max_attempts());
        assert_eq!(strategy.name(), "exponential-backoff");
    }
}
