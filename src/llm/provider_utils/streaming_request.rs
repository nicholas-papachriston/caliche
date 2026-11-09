//! Unified retry for LLM streaming request establishment.
//!
//! Retries send + HTTP status check before any stream events are emitted.
//! Send/network failures are treated as transient (unlike [`retry_llm_request`]).

use crate::llm::provider_utils::check_response_status_with_retry;
use crate::llm::retry::{RetryError, retry_with_backoff};
use crate::traits::retry::RetryStrategy as TraitsRetryStrategy;
use reqwest::Response;

/// Retry streaming HTTP request establishment (send + success status).
///
/// Send failures use [`RetryError::Transient`] so transient network issues can be
/// retried without duplicating partial stream output.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn retry_stream_establishment<F, Fut>(
    strategy: &dyn TraitsRetryStrategy,
    provider_name: &str,
    mut send_request: F,
) -> anyhow::Result<Response>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Response, reqwest::Error>> + Send,
{
    retry_with_backoff(strategy, || {
        let fut = send_request();
        async move {
            let response = fut.await.map_err(|e| RetryError::Transient {
                status_code: 599,
                message: format!("Failed to send request to {provider_name}: {e}"),
            })?;
            check_response_status_with_retry(response, provider_name).await
        }
    })
    .await
}
