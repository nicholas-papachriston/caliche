use caliche::llm::provider_utils::classify_error;
use caliche::llm::retry::{RetryError, retry_with_backoff};
use caliche::traits::retry::{FixedDelayStrategy, RetryError as TraitsRetryError};
use reqwest::Response;
use std::time::Duration;

#[expect(
    clippy::expect_used,
    reason = "test helper: a failure fails the calling test"
)]
fn ok_response() -> Response {
    http::Response::builder()
        .status(200)
        .body(Vec::new())
        .map(Response::from)
        .expect("response should build")
}

#[test]
fn classify_error_marks_rate_limits_retryable() {
    let err = classify_error(
        reqwest::StatusCode::TOO_MANY_REQUESTS,
        "try again in 2s",
        "test-provider",
    );
    assert!(matches!(err, RetryError::RateLimited { .. }));
    assert!(err.is_retryable());
}

#[test]
fn classify_error_marks_server_errors_transient() {
    let err = classify_error(
        reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "upstream unavailable",
        "test-provider",
    );
    assert!(matches!(
        err,
        RetryError::Transient {
            status_code: 503,
            ..
        }
    ));
    assert!(err.is_retryable());
}

#[test]
fn classify_error_marks_client_errors_permanent() {
    let err = classify_error(
        reqwest::StatusCode::BAD_REQUEST,
        "invalid request",
        "test-provider",
    );
    assert!(matches!(err, RetryError::Permanent { .. }));
    assert!(!err.is_retryable());
}

#[test]
fn classify_error_marks_auth_errors_non_retryable() {
    let err = classify_error(
        reqwest::StatusCode::UNAUTHORIZED,
        "bad key",
        "test-provider",
    );
    assert!(matches!(err, RetryError::AuthError { .. }));
    assert!(!err.is_retryable());
}

#[tokio::test]
async fn retry_stream_establishment_retries_transient_errors() {
    let strategy = FixedDelayStrategy::new(2, Duration::from_millis(1));
    let mut attempts = 0_u32;

    let response = retry_with_backoff(&strategy, || {
        attempts += 1;
        let attempt = attempts;
        async move {
            if attempt < 2 {
                Err(TraitsRetryError::Transient {
                    status_code: 503,
                    message: "temporary outage".to_string(),
                })
            } else {
                Ok(ok_response())
            }
        }
    })
    .await
    .expect("establishment should succeed after retry");

    assert_eq!(attempts, 2);
    assert!(response.status().is_success());
}

#[tokio::test]
async fn retry_stream_establishment_stops_on_permanent_errors() {
    let strategy = FixedDelayStrategy::new(3, Duration::from_millis(1));
    let mut attempts = 0_u32;

    let err = retry_with_backoff::<Response, _, _>(&strategy, || {
        attempts += 1;
        async move {
            Err(TraitsRetryError::Permanent {
                error: "bad payload".to_string(),
            })
        }
    })
    .await
    .expect_err("permanent errors should fail immediately");

    assert_eq!(attempts, 1);
    assert!(err.to_string().contains("bad payload"));
}
