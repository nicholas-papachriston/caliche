//! Tests for HTTP client retry logic and error classification
#![cfg(any(
    feature = "qdrant-backend",
    feature = "pinecone-backend",
    feature = "chroma-backend",
    feature = "weaviate-backend",
    feature = "milvus-backend"
))]

use dripstone::backend::http::{HttpClient, HttpClientConfig, RetryConfig};
use dripstone::backend::retry::{
    ExponentialBackoffStrategy, NoRetryStrategy, RetryError, RetryStrategy,
};
use std::time::Duration;

#[tokio::test]
async fn test_no_retry_strategy() {
    let retry_config = RetryConfig::no_retry();
    let http_config = HttpClientConfig::new().with_retry_config(retry_config);

    // Create a client (will fail to connect, but that's ok for this test)
    let result = HttpClient::with_config("http://localhost:9999", http_config);

    // Should create client successfully even if connection fails later
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_exponential_backoff_config() {
    let retry_config = RetryConfig::exponential(3, 1000, 10000, 2.0, true);

    let http_config = HttpClientConfig::new().with_retry_config(retry_config);

    let result = HttpClient::with_config("http://localhost:9999", http_config);
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_default_exponential_config() {
    let retry_config = RetryConfig::default_exponential();
    let http_config = HttpClientConfig::new().with_retry_config(retry_config);

    let result = HttpClient::with_config("http://localhost:9999", http_config);
    assert!(result.is_ok());
}

#[test]
fn test_retry_error_classification() {
    // Test RateLimited error
    let rate_limited = RetryError::RateLimited {
        retry_after: Some(Duration::from_secs(5)),
        message: "Too many requests".to_string(),
    };
    assert!(rate_limited.is_retryable());
    assert_eq!(rate_limited.retry_after(), Some(Duration::from_secs(5)));

    // Test Transient error
    let transient = RetryError::Transient {
        status_code: 500,
        message: "Internal server error".to_string(),
    };
    assert!(transient.is_retryable());
    assert_eq!(transient.retry_after(), None);

    // Test Permanent error
    let permanent = RetryError::Permanent {
        error: "Bad request".to_string(),
    };
    assert!(!permanent.is_retryable());
    assert_eq!(permanent.retry_after(), None);
}

#[test]
fn test_exponential_backoff_strategy() {
    let strategy = ExponentialBackoffStrategy::new(
        3, 1000, 10000, 2.0, false, // No jitter for deterministic testing
    );

    assert_eq!(strategy.max_attempts(), 3);
    assert_eq!(strategy.name(), "exponential-backoff");

    let transient_error = RetryError::Transient {
        status_code: 500,
        message: "Server error".to_string(),
    };

    // Should retry for first 3 attempts
    assert!(strategy.should_retry(0, &transient_error));
    assert!(strategy.should_retry(1, &transient_error));
    assert!(strategy.should_retry(2, &transient_error));
    assert!(!strategy.should_retry(3, &transient_error));

    // Should not retry permanent errors
    let permanent_error = RetryError::Permanent {
        error: "Bad request".to_string(),
    };
    assert!(!strategy.should_retry(0, &permanent_error));

    // Test backoff calculation
    let backoff = strategy.calculate_backoff(0, &transient_error);
    assert_eq!(backoff, Duration::from_millis(1000));

    let backoff = strategy.calculate_backoff(1, &transient_error);
    assert_eq!(backoff, Duration::from_millis(2000));

    let backoff = strategy.calculate_backoff(2, &transient_error);
    assert_eq!(backoff, Duration::from_millis(4000));
}

#[test]
fn test_exponential_backoff_with_retry_after() {
    let strategy = ExponentialBackoffStrategy::new(3, 1000, 10000, 2.0, false);

    let rate_limited = RetryError::RateLimited {
        retry_after: Some(Duration::from_secs(5)),
        message: "Rate limited".to_string(),
    };

    // Should use retry-after hint instead of calculated backoff
    let backoff = strategy.calculate_backoff(0, &rate_limited);
    assert_eq!(backoff, Duration::from_secs(5));
}

#[test]
fn test_no_retry_strategy_behavior() {
    let strategy = NoRetryStrategy::new();

    assert_eq!(strategy.max_attempts(), 0);
    assert_eq!(strategy.name(), "no-retry");

    let transient_error = RetryError::Transient {
        status_code: 500,
        message: "Server error".to_string(),
    };

    // Should never retry
    assert!(!strategy.should_retry(0, &transient_error));
    assert!(!strategy.should_retry(1, &transient_error));

    let backoff = strategy.calculate_backoff(0, &transient_error);
    assert_eq!(backoff, Duration::ZERO);
}

#[test]
fn test_connection_pool_config() {
    let http_config = HttpClientConfig::new()
        .with_connection_pool_size(20)
        .with_pool_idle_timeout(Duration::from_secs(120));

    // Verify configuration is stored
    assert_eq!(http_config.connection_pool_size, Some(20));
    assert_eq!(
        http_config.pool_idle_timeout,
        Some(Duration::from_secs(120))
    );
}

#[test]
fn test_http_client_config_builder() {
    let timeout = Duration::from_secs(30);
    let retry_config = RetryConfig::default_exponential();

    let config = HttpClientConfig::new()
        .with_timeout(timeout)
        .with_retry_config(retry_config)
        .with_connection_pool_size(10)
        .with_pool_idle_timeout(Duration::from_secs(90));

    assert_eq!(config.timeout, Some(timeout));
    assert!(config.retry_config.is_some());
    assert_eq!(config.connection_pool_size, Some(10));
    assert_eq!(config.pool_idle_timeout, Some(Duration::from_secs(90)));
}

#[test]
fn test_retry_config_constructors() {
    // Test custom exponential
    let custom = RetryConfig::exponential(5, 2000, 30000, 1.5, true);
    assert_eq!(custom.strategy.max_attempts(), 5);

    // Test default exponential
    let default = RetryConfig::default_exponential();
    assert_eq!(default.strategy.max_attempts(), 3);

    // Test no retry
    let no_retry = RetryConfig::no_retry();
    assert_eq!(no_retry.strategy.max_attempts(), 0);
}
