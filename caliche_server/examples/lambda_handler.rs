//! Custom AWS Lambda Handler Example
//!
//! This example shows how to build a custom Lambda handler that extends
//! the base caliche lambda functionality with custom logic, middleware,
//! or request preprocessing.
//!
//! To run this example, build with the lambda feature:
//! ```bash
//! cargo run --example lambda_handler --features lambda
//! ```

#[cfg(feature = "lambda")]
use caliche_server::lambda::{lambda_handler, run_lambda_server};
#[cfg(feature = "lambda")]
use lambda_http::{Error, Request, Response};

/// Example 1: Use the default lambda handler as-is
///
/// This is equivalent to using the binary in lambda/handler.rs
#[cfg(feature = "lambda")]
#[expect(
    dead_code,
    reason = "an alternative handler shown for copying; main runs the default one"
)]
async fn example_default_handler() -> Result<(), Error> {
    run_lambda_server().await
}

/// Example 2: Custom handler with request logging
///
/// This shows how to wrap the lambda handler with custom logic
#[cfg(feature = "lambda")]
#[expect(
    dead_code,
    reason = "an alternative handler shown for copying; main runs the default one"
)]
async fn custom_handler_with_logging(event: Request) -> Result<Response<String>, Error> {
    // Log the incoming request
    tracing::info!(
        "Received request: {} {}",
        event.method(),
        event.uri().path()
    );

    // Call the base handler
    let response = lambda_handler(event).await?;

    // Log the response
    tracing::info!("Response status: {}", response.status());

    Ok(response)
}

/// Example 3: Custom handler with request filtering
///
/// This shows how to add custom routing or filtering logic
#[cfg(feature = "lambda")]
#[expect(
    dead_code,
    reason = "an alternative handler shown for copying; main runs the default one"
)]
async fn custom_handler_with_filtering(event: Request) -> Result<Response<String>, Error> {
    // Example: Block certain paths
    if event.uri().path().starts_with("/admin") {
        return Ok(Response::builder()
            .status(403)
            .body("Access denied".to_string())?);
    }

    // Call the base handler for all other requests
    lambda_handler(event).await
}

/// Example 4: Custom handler with response transformation
///
/// This shows how to modify responses before returning
#[cfg(feature = "lambda")]
#[expect(
    dead_code,
    reason = "an alternative handler shown for copying; main runs the default one"
)]
async fn custom_handler_with_transformation(event: Request) -> Result<Response<String>, Error> {
    let mut response = lambda_handler(event).await?;

    // Add custom headers
    let headers = response.headers_mut();
    if let Ok(header_value) = "caliche".parse() {
        headers.insert("X-Custom-Header", header_value);
    }
    if let Ok(header_value) = "caliche-lambda".parse() {
        headers.insert("X-Powered-By", header_value);
    }

    Ok(response)
}

/// Example 5: Custom handler with error handling
///
/// This shows how to add custom error handling and recovery
#[cfg(feature = "lambda")]
#[expect(
    dead_code,
    reason = "an alternative handler shown for copying; main runs the default one"
)]
async fn custom_handler_with_error_handling(event: Request) -> Result<Response<String>, Error> {
    match lambda_handler(event).await {
        Ok(response) => Ok(response),
        Err(e) => {
            tracing::error!("Handler error: {}", e);
            // Return a user-friendly error response
            Ok(Response::builder()
                .status(500)
                .body(r#"{"error": "Internal server error"}"#.to_string())?)
        }
    }
}

/// Main function demonstrating different custom handler patterns
#[cfg(feature = "lambda")]
#[tokio::main]
async fn main() -> Result<(), Error> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    // For demonstration, we'll use the default handler
    // In practice, you would use one of the custom handlers above
    tracing::info!("Starting custom lambda handler example");
    run_lambda_server().await
}

#[cfg(not(feature = "lambda"))]
fn main() {
    eprintln!("This example requires the 'lambda' feature to be enabled.");
    eprintln!("Build with: cargo run --example lambda_handler --features lambda");
    std::process::exit(1);
}
