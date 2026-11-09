//! Provider setup utilities for common initialization tasks.

use crate::llm::config::CommonProviderOptions;
use crate::traits::retry::RetryStrategy;
use std::sync::Arc;

/// Create an HTTP client from common provider options.
///
/// If a shared client is provided in the options, it will be used.
/// Otherwise, a new client will be created with the specified timeout and pool size.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub fn create_http_client(options: &CommonProviderOptions) -> anyhow::Result<Arc<reqwest::Client>> {
    if let Some(shared_client) = &options.client {
        return Ok(shared_client.clone());
    }

    let mut client_builder = reqwest::Client::builder()
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .pool_max_idle_per_host(20)
        .tcp_nodelay(true)
        .connect_timeout(std::time::Duration::from_secs(10));

    if let Some(timeout) = options.timeout {
        client_builder = client_builder.timeout(timeout);
    } else {
        // Default timeout catches dead or stalled upstream connections while still
        // allowing long streaming turns for slower models.
        client_builder = client_builder.timeout(std::time::Duration::from_secs(300));
    }

    if let Some(pool_size) = options.connection_pool_size {
        client_builder = client_builder.pool_max_idle_per_host(pool_size);
    }

    Ok(Arc::new(client_builder.build().map_err(|e| {
        anyhow::anyhow!("Failed to build HTTP client: {e}")
    })?))
}

/// Create a retry strategy from common provider options.
///
/// If a custom retry strategy is provided in the options, it will be used.
/// Otherwise, a default exponential backoff strategy will be created.
#[must_use]
pub fn create_retry_strategy(options: &CommonProviderOptions) -> Arc<dyn RetryStrategy> {
    options.retry_strategy.clone().unwrap_or_else(|| {
        Arc::new(crate::traits::retry::ExponentialBackoffStrategy::new(
            3, 500, 30_000, 2.0, true,
        ))
    })
}
