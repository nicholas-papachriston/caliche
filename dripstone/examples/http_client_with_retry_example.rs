//! Example demonstrating HTTP client retry and connection pool configuration
//!
//! This example shows how to configure retry strategies and connection pool settings
//! for HTTP-based backends (Qdrant, Chroma, Pinecone).
//!
//! Run with:
//! ```bash
//! cargo run --example http_client_with_retry_example --features "qdrant-backend,chroma-backend"
//! ```

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::chroma::ChromaBackendOptions;
use dripstone::backend::http::{HttpClient, HttpClientConfig, RetryConfig};
use dripstone::backend::qdrant::QdrantBackendOptions;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example 1: Configure HTTP client with exponential backoff retry
    outputln!("Example 1: HTTP Client with Exponential Backoff Retry");

    let retry_config = RetryConfig::exponential(
        5,     // max_attempts
        1000,  // initial_backoff_ms (1 second)
        60000, // max_backoff_ms (60 seconds)
        2.0,   // backoff_multiplier
        true,  // jitter enabled
    );

    let http_config = HttpClientConfig::new()
        .with_timeout(Duration::from_secs(30))
        .with_retry_config(retry_config)
        .with_connection_pool_size(10)
        .with_pool_idle_timeout(Duration::from_secs(90));

    let _client = HttpClient::with_config("http://localhost:6333", http_config)?;
    outputln!("HTTP client created with retry and connection pool configuration\n");

    // Example 2: Configure Qdrant backend with retry
    outputln!("Example 2: Qdrant Backend with Retry Configuration");

    let retry_config = RetryConfig::default_exponential();
    let mut qdrant_options = QdrantBackendOptions::new("http://localhost:6333", 128);
    qdrant_options.retry_config = Some(retry_config);

    // Note: In a real scenario, you would create the backend:
    // let backend = QdrantEmbeddingBackend::new(qdrant_options).await?;
    outputln!("Qdrant backend options configured with retry\n");

    // Example 3: Configure Chroma backend with custom retry strategy
    outputln!("Example 3: Chroma Backend with Custom Retry Strategy");

    let retry_config = RetryConfig::exponential(
        3,     // max_attempts
        500,   // initial_backoff_ms (0.5 seconds)
        30000, // max_backoff_ms (30 seconds)
        1.5,   // backoff_multiplier (gentler backoff)
        false, // jitter disabled
    );

    let mut chroma_options = ChromaBackendOptions::new("http://localhost:8000");
    chroma_options.retry_config = Some(retry_config);

    // Note: In a real scenario, you would create the backend:
    // let backend = ChromaEmbeddingBackend::new(chroma_options).await?;
    outputln!("Chroma backend options configured with custom retry\n");

    // Example 4: No retry strategy (for testing or when retries are handled elsewhere)
    outputln!("Example 4: HTTP Client without Retry");

    let http_config = HttpClientConfig::new()
        .with_timeout(Duration::from_secs(10))
        .with_retry_config(RetryConfig::no_retry());

    let _client = HttpClient::with_config("http://localhost:6333", http_config)?;
    outputln!("HTTP client created without retry\n");

    // Example 5: Connection pool configuration
    outputln!("Example 5: Connection Pool Configuration");

    let http_config = HttpClientConfig::new()
        .with_connection_pool_size(20) // Max 20 idle connections per host
        .with_pool_idle_timeout(Duration::from_secs(120)) // Keep connections alive for 2 minutes
        .with_timeout(Duration::from_secs(60));

    let _client = HttpClient::with_config("http://localhost:6333", http_config)?;
    outputln!("HTTP client created with optimized connection pool settings\n");

    outputln!("All examples completed successfully!");
    Ok(())
}
