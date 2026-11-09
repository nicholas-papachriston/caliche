//! Response parsing utilities for LLM providers.

use crate::llm::retry::{RetryError, parse_retry_after_from_message, retry_with_backoff};
use crate::messages::{Message, MessageRole};
use crate::traits::retry::RetryStrategy as TraitsRetryStrategy;
use reqwest::Response;

/// Helper function to convert `MessageRole` to a string role name.
/// Used by providers that use simple string-based roles.
/// Returns a static string slice to avoid allocation.
#[must_use]
pub const fn role_to_string(role: &MessageRole) -> &'static str {
    match role {
        MessageRole::System => "system",
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
        MessageRole::Tool => "tool",
    }
}

/// Helper function to extract text content from a message.
/// This is a simple wrapper around the `MessageContent::extract_text` method.
#[must_use]
pub fn extract_message_text(msg: &Message) -> String {
    msg.content.extract_text()
}

/// Helper function to handle HTTP response errors consistently.
/// Returns the response if successful, otherwise returns a [`RetryError`]
/// (classified by status) whose text names the status and the server's reason.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn check_response_status(
    response: Response,
    provider_name: &str,
) -> anyhow::Result<Response> {
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(classify_error(status, &error_text, provider_name).into());
    }
    Ok(response)
}

/// One line naming the provider, the status and the server's reason.
///
/// A JSON body with an `error.message` (OpenAI shape), an `error` string or
/// a `message` gives that text; any other body is kept whole, with its
/// whitespace runs folded to single spaces.
#[must_use]
pub fn api_error_message(
    provider_name: &str,
    status: reqwest::StatusCode,
    error_text: &str,
) -> String {
    let reason = serde_json::from_str::<serde_json::Value>(error_text)
        .ok()
        .and_then(|body| {
            let error = body.get("error");
            error
                .and_then(|error| error.get("message"))
                .or(error)
                .or_else(|| body.get("message"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned)
        })
        .unwrap_or_else(|| error_text.to_owned());
    let reason = reason.split_whitespace().collect::<Vec<_>>().join(" ");
    format!("{provider_name} API error ({status}): {reason}")
}

/// Classify an error as a `RetryError` based on status code and message.
/// Used by providers to determine if an error is retryable.
#[must_use]
pub fn classify_error(
    status: reqwest::StatusCode,
    error_text: &str,
    provider_name: &str,
) -> RetryError {
    let status_code = status.as_u16();
    let message = api_error_message(provider_name, status, error_text);

    // Rate limit errors (429)
    if status_code == 429 {
        let retry_after = parse_retry_after_from_message(error_text);
        return RetryError::RateLimited {
            retry_after,
            message,
        };
    }

    // Authentication errors (401, 403) - failover may try another credential
    if status_code == 401 || status_code == 403 {
        return RetryError::AuthError { message };
    }

    // Model not found / unavailable (404) - failover may try another provider or model
    if status_code == 404 {
        return RetryError::ModelUnavailable { message };
    }

    // OpenRouter-specific data policy errors (404) - handled above as ModelUnavailable
    // If we need to distinguish OpenRouter data policy from generic 404, check provider_name here.

    // Transient errors (5xx server errors, 408 timeout, 503 unavailable)
    if status_code >= 500 || status_code == 408 || status_code == 503 {
        return RetryError::Transient {
            status_code,
            message,
        };
    }

    // Permanent errors (4xx client errors except 429 and 408)
    RetryError::Permanent { error: message }
}

/// Check response status and return `RetryError` for retry logic.
/// This version is used by providers with retry support.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn check_response_status_with_retry(
    response: Response,
    provider_name: &str,
) -> Result<Response, RetryError> {
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(classify_error(status, &error_text, provider_name));
    }
    Ok(response)
}

/// Retry an LLM HTTP request with backoff and response status checking.
/// Encapsulates the common pattern of `retry_with_backoff` + send + `check_response_status_with_retry`.
///
/// **Retry policy**: Send/network failures (`reqwest::Error`) are treated as `Permanent` and are
/// not retried. Only rate limits (429) and transient server errors (5xx) are retried. The `OpenAI`-
/// compatible streaming path uses a different policy (Transient for send failures) to retry
/// before any stream events are emitted.
/// # Errors
///
/// Returns an error if request construction, transport, or response decoding fails.
pub async fn retry_llm_request<F, Fut>(
    strategy: &dyn TraitsRetryStrategy,
    provider_name: &str,
    mut send_request: F,
) -> anyhow::Result<Response>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<Response, reqwest::Error>> + Send + 'static,
{
    retry_with_backoff(strategy, || {
        let fut = send_request();
        Box::pin(async move {
            let response = fut.await.map_err(|e| RetryError::Permanent {
                error: anyhow::Error::new(e)
                    .context(format!("Failed to send request to {provider_name}"))
                    .to_string(),
            })?;
            check_response_status_with_retry(response, provider_name).await
        })
    })
    .await
}
