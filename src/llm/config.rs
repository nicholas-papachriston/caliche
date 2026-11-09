//! Provider configuration structures and traits for programmatic configuration.
//!
//! This module provides a unified interface for configuring LLM providers
//! programmatically; providers still fall back to environment variables when
//! a value is not set explicitly.

use crate::traits::retry::RetryStrategy;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

/// Common provider options shared across all providers
#[derive(Debug)]
pub struct CommonProviderOptions {
    /// Request timeout duration
    pub timeout: Option<Duration>,
    /// Maximum number of retry attempts
    pub max_retries: Option<usize>,
    /// Custom retry strategy
    pub retry_strategy: Option<Arc<dyn RetryStrategy>>,
    /// Custom HTTP headers to include in requests
    pub custom_headers: Option<HashMap<String, String>>,
    /// Connection pool size for HTTP client
    pub connection_pool_size: Option<usize>,
    /// Shared HTTP client for connection pooling (optional)
    pub client: Option<Arc<reqwest::Client>>,
}

impl CommonProviderOptions {
    /// Create a new set of common provider options
    #[must_use]
    pub fn new() -> Self {
        Self {
            timeout: None,
            max_retries: None,
            retry_strategy: None,
            custom_headers: None,
            connection_pool_size: None,
            client: None,
        }
    }

    /// Set request timeout
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set maximum retries
    #[must_use]
    pub const fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = Some(max_retries);
        self
    }

    /// Set retry strategy
    #[must_use]
    pub fn with_retry_strategy(mut self, strategy: Arc<dyn RetryStrategy>) -> Self {
        self.retry_strategy = Some(strategy);
        self
    }

    /// Add a custom header.
    #[must_use]
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_headers
            .get_or_insert_with(HashMap::new)
            .insert(key.into(), value.into());
        self
    }

    /// Set connection pool size
    #[must_use]
    pub const fn with_connection_pool_size(mut self, size: usize) -> Self {
        self.connection_pool_size = Some(size);
        self
    }

    /// Set shared HTTP client for connection pooling
    #[must_use]
    pub fn with_client(mut self, client: Arc<reqwest::Client>) -> Self {
        self.client = Some(client);
        self
    }
}

impl Default for CommonProviderOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CommonProviderOptions {
    fn clone(&self) -> Self {
        Self {
            timeout: self.timeout,
            max_retries: self.max_retries,
            retry_strategy: self.retry_strategy.clone(),
            custom_headers: self.custom_headers.clone(),
            connection_pool_size: self.connection_pool_size,
            client: self.client.clone(), // Arc is Clone
        }
    }
}

/// Trait for provider-specific configuration
pub trait ProviderConfig: Send + Sync + std::fmt::Debug {
    /// Get common provider options
    fn common_options(&self) -> &CommonProviderOptions;

    /// Get common provider options mutably
    fn common_options_mut(&mut self) -> &mut CommonProviderOptions;

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn Any;
}
