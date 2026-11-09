//! Tool configuration structures and traits for programmatic configuration.

use crate::agent::tools::credentials::ToolCredential;
use crate::traits::retry::RetryStrategy;
use std::time::Duration;

/// Common tool options shared across all tools
#[derive(Debug)]
pub struct CommonToolOptions {
    /// Request timeout duration
    pub timeout: Option<Duration>,
    /// Maximum number of retry attempts
    pub max_retries: Option<usize>,
    /// Custom retry strategy
    pub retry_strategy: Option<Box<dyn RetryStrategy>>,
    /// Optional credentials for tool authentication (construction-time, highest priority)
    pub credentials: Option<ToolCredential>,
}

impl CommonToolOptions {
    /// Create a new set of common tool options
    #[must_use]
    pub fn new() -> Self {
        Self {
            timeout: None,
            max_retries: None,
            retry_strategy: None,
            credentials: None,
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
    pub fn with_retry_strategy(mut self, strategy: Box<dyn RetryStrategy>) -> Self {
        self.retry_strategy = Some(strategy);
        self
    }

    /// Set credentials (construction-time, highest priority)
    #[must_use]
    pub fn with_credentials(mut self, credentials: ToolCredential) -> Self {
        self.credentials = Some(credentials);
        self
    }
}

impl Default for CommonToolOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CommonToolOptions {
    fn clone(&self) -> Self {
        Self {
            timeout: self.timeout,
            max_retries: self.max_retries,
            retry_strategy: None, // Cannot clone trait objects, set to None
            credentials: self.credentials.clone(),
        }
    }
}

/// Trait for tool-specific configuration
pub trait ToolConfig: Send + Sync + std::fmt::Debug {
    /// Get common tool options
    fn common_options(&self) -> &CommonToolOptions;

    /// Get common tool options mutably
    fn common_options_mut(&mut self) -> &mut CommonToolOptions;
}
