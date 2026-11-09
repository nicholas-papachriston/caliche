//! AWS Lambda integration for caliche API server.
//!
//! This module provides functions to run the full caliche API server
//! as an AWS Lambda function. It handles request/response conversion
//! between Lambda's HTTP events and Axum's router.

#[cfg(feature = "lambda")]
use crate::create_router;
#[cfg(feature = "lambda")]
use caliche::agent::service::resolve_model_for_provider_with_discovery;
#[cfg(feature = "lambda")]
use caliche::llm::ProviderType;
#[cfg(feature = "lambda")]
use lambda_http::{Error, Request, Response, run, service_fn};
#[cfg(feature = "lambda")]
use std::sync::Arc;
#[cfg(feature = "lambda")]
use tokio::sync::OnceCell;

/// Global router instance for warm starts
#[cfg(feature = "lambda")]
static ROUTER: OnceCell<Arc<axum::Router>> = OnceCell::const_new();
/// Maximum response body size we will buffer when converting back to Lambda.
#[cfg(feature = "lambda")]
const MAX_LAMBDA_BODY_BYTES: usize = 10 * 1024 * 1024;

/// Get or initialize the Axum router with caching for warm starts
#[cfg(feature = "lambda")]
async fn get_router() -> Result<Arc<axum::Router>, Error> {
    ROUTER
        .get_or_try_init(|| async {
            // Read default provider from environment
            let default_provider = std::env::var("DEFAULT_PROVIDER")
                .ok()
                .and_then(|p| p.parse::<ProviderType>().ok())
                .unwrap_or(ProviderType::Bedrock);

            let default_model = resolve_model_for_provider_with_discovery(
                default_provider,
                std::env::var("DEFAULT_MODEL").ok(),
            )
            .await
            .map_err(|error| Error::from(error.to_string()))?;

            tracing::info!(
                "Initializing router with provider: {:?}, model: {}",
                default_provider,
                default_model
            );

            // Create the full API router with all endpoints
            let router = create_router(default_provider, default_model, None)
                .await
                .map_err(|error| Error::from(error.to_string()))?;

            Ok::<_, Error>(Arc::new(router))
        })
        .await
        .map(Arc::clone)
}

/// Convert lambda_http::Request to http::Request for Axum
#[cfg(feature = "lambda")]
async fn lambda_request_to_http(event: Request) -> Result<http::Request<axum::body::Body>, Error> {
    let (parts, body) = event.into_parts();
    // Convert lambda_http::Body to bytes
    let body_bytes = match body {
        lambda_http::Body::Empty => bytes::Bytes::new(),
        lambda_http::Body::Text(s) => bytes::Bytes::from(s),
        lambda_http::Body::Binary(b) => bytes::Bytes::from(b),
        _ => bytes::Bytes::new(), // Handle any other variants
    };

    let mut http_request = http::Request::builder()
        .method(parts.method)
        .uri(parts.uri)
        .version(parts.version);

    // Copy headers
    for (key, value) in parts.headers.iter() {
        http_request = http_request.header(key, value);
    }

    // Set body
    let http_body = axum::body::Body::from(body_bytes);
    let http_request = http_request.body(http_body)?;

    Ok(http_request)
}

/// Convert http::Response to lambda_http::Response
#[cfg(feature = "lambda")]
async fn http_response_to_lambda(
    http_response: http::Response<axum::body::Body>,
) -> Result<Response<String>, Error> {
    let (parts, body) = http_response.into_parts();
    let body_bytes = axum::body::to_bytes(body, MAX_LAMBDA_BODY_BYTES)
        .await
        .map_err(|e| Error::from(format!("Body conversion error: {}", e)))?;

    let body_str = String::from_utf8(body_bytes.to_vec())
        .map_err(|e| Error::from(format!("UTF-8 conversion error: {}", e)))?;

    let mut response = Response::builder().status(parts.status);

    // Copy headers
    for (key, value) in parts.headers.iter() {
        if let Ok(header_value) = value.to_str() {
            response = response.header(key, header_value);
        }
    }

    let response = response.body(body_str)?;

    Ok(response)
}

/// Lambda handler function that processes requests
#[cfg(feature = "lambda")]
pub async fn lambda_handler(event: Request) -> Result<Response<String>, Error> {
    // Get or initialize router
    let router = get_router().await?;

    // Convert lambda_http::Request to http::Request
    let http_request = lambda_request_to_http(event).await?;

    // Call the Axum router using oneshot
    use tower::ServiceExt;
    let http_response = Arc::try_unwrap(router)
        .unwrap_or_else(|arc| (*arc).clone())
        .oneshot(http_request)
        .await
        .map_err(|e| Error::from(format!("Router error: {}", e)))?;

    // Convert http::Response to lambda_http::Response
    http_response_to_lambda(http_response).await
}

/// Create a lambda handler function that can be used with lambda_http::run
#[cfg(feature = "lambda")]
pub fn create_lambda_handler() -> impl Fn(
    Request,
) -> std::pin::Pin<
    Box<dyn std::future::Future<Output = Result<Response<String>, Error>> + Send>,
> {
    Box::new(move |event: Request| {
        Box::pin(lambda_handler(event))
            as std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Response<String>, Error>> + Send>,
            >
    })
}

/// Main entry point for running the full API server in Lambda
///
/// This function initializes logging, pre-warms the router, and starts
/// the Lambda runtime. It should be called from your Lambda handler's main function.
#[cfg(feature = "lambda")]
pub async fn run_lambda_server() -> Result<(), Error> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();

    // Pre-warm router on cold start
    let _ = get_router().await?;

    tracing::info!("Lambda handler initialized and ready");

    // tokio::select! lets us exit cleanly on Lambda's SIGTERM during
    // shutdown so any pending logs flush. `run` never resolves in normal
    // operation, so the select here acts as a hard shutdown hook.
    tokio::select! {
        res = run(service_fn(lambda_handler)) => res,
        _ = shutdown_signal() => {
            tracing::info!("Lambda shutdown signal received; stopping");
            Ok(())
        }
    }
}

/// Streaming-aware Lambda entry point that uses
/// `lambda_http::run_with_streaming_response` so large/long responses (e.g.
/// SSE chat streams) flow back to the caller incrementally instead of being
/// buffered in one 10 MB response.
///
/// Build with `--features stream-lambda`.
#[cfg(feature = "stream-lambda")]
pub async fn run_lambda_streaming_server() -> Result<(), Error> {
    use lambda_http::run_with_streaming_response;

    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .without_time()
        .init();

    let _ = get_router().await?;
    tracing::info!("Lambda streaming handler initialized");

    tokio::select! {
        res = run_with_streaming_response(service_fn(lambda_streaming_handler)) => res,
        _ = shutdown_signal() => {
            tracing::info!("Lambda streaming shutdown signal received");
            Ok(())
        }
    }
}

/// Same routing as [`lambda_handler`] but returns the Axum body verbatim so
/// `run_with_streaming_response` can forward chunks as they arrive.
#[cfg(feature = "stream-lambda")]
pub async fn lambda_streaming_handler(event: Request) -> Result<Response<axum::body::Body>, Error> {
    use tower::ServiceExt;

    let router = get_router().await?;
    let http_request = lambda_request_to_http(event).await?;
    let http_response = Arc::try_unwrap(router)
        .unwrap_or_else(|arc| (*arc).clone())
        .oneshot(http_request)
        .await
        .map_err(|e| Error::from(format!("Router error: {}", e)))?;
    let (parts, body) = http_response.into_parts();
    Ok(Response::from_parts(parts, body))
}

/// Future that completes when the process receives SIGTERM / SIGINT.
#[cfg(feature = "lambda")]
async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(err) => {
                tracing::warn!(error = %err, "failed to install SIGTERM listener");
                return;
            }
        };
        let mut int = match signal(SignalKind::interrupt()) {
            Ok(s) => s,
            Err(err) => {
                tracing::warn!(error = %err, "failed to install SIGINT listener");
                return;
            }
        };
        tokio::select! {
            _ = term.recv() => {}
            _ = int.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
