//! Macros for reducing boilerplate in provider implementations.

/// Implements the `ProviderConfig` trait for a struct that has a `common` field.
///
/// # Usage
/// ```text
/// impl_provider_config!(MyProviderConfig);
/// ```
///
/// This macro generates:
/// - `common_options()` method that returns `&self.common`
/// - `common_options_mut()` method that returns `&mut self.common`
/// - `as_any()` method that returns `self as &dyn std::any::Any`
#[macro_export]
macro_rules! impl_provider_config {
    ($config_type:ty) => {
        impl $crate::llm::config::ProviderConfig for $config_type {
            fn common_options(&self) -> &$crate::llm::config::CommonProviderOptions {
                &self.common
            }

            fn common_options_mut(&mut self) -> &mut $crate::llm::config::CommonProviderOptions {
                &mut self.common
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}

/// Implements common builder methods for a builder struct that has a `common` field.
///
/// # Usage
/// ```text
/// impl_common_builder_methods!(MyProviderConfigBuilder);
/// ```
///
/// This macro generates:
/// - `timeout(timeout: Duration)` method
/// - `max_retries(max_retries: usize)` method
/// - `retry_strategy(strategy: Arc<dyn RetryStrategy>)` method
#[macro_export]
macro_rules! impl_common_builder_methods {
    ($builder_type:ty) => {
        impl $builder_type {
            /// Set request timeout
            #[must_use]
            pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
                self.common = self.common.with_timeout(timeout);
                self
            }

            /// Set maximum retries
            #[must_use]
            pub fn max_retries(mut self, max_retries: usize) -> Self {
                self.common = self.common.with_max_retries(max_retries);
                self
            }

            /// Set retry strategy
            #[must_use]
            pub fn retry_strategy(
                mut self,
                strategy: std::sync::Arc<dyn $crate::traits::retry::RetryStrategy>,
            ) -> Self {
                self.common = self.common.with_retry_strategy(strategy);
                self
            }

            /// Use a shared HTTP client for connection pooling across multiple agents.
            #[cfg(feature = "reqwest")]
            #[must_use]
            pub fn shared_http_client(mut self, client: std::sync::Arc<reqwest::Client>) -> Self {
                self.common = self.common.with_client(client);
                self
            }
        }
    };
}
