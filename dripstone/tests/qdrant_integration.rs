#![cfg(feature = "qdrant-backend")]

mod common;
mod integration_suite;

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::DripstoneError;
use dripstone::backend::http::RetryConfig;
use dripstone::backend::qdrant::{QdrantBackendOptions, QdrantHttpBackend};
use dripstone::embedding::SimilarityMetric;
use integration_suite::run_integration_suite;
use time::OffsetDateTime;

const DEFAULT_QDRANT_URL: &str = "http://127.0.0.1:6333";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn qdrant_integration_full_suite() -> Result<(), DripstoneError> {
    let url =
        std::env::var("EMBEDDING_QDRANT_URL").unwrap_or_else(|_| DEFAULT_QDRANT_URL.to_string());
    let api_key = common::qdrant_api_key();
    let namespace = format!(
        "qdrant-integration-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let options = QdrantBackendOptions {
        base_url: url,
        collection_prefix: Some("integration_".to_string()),
        vector_dimension: 128,
        distance: SimilarityMetric::Cosine,
        timeout: None,
        api_key,
        batch_size: None,
        retry_config: Some(RetryConfig::default_exponential()),
    };

    let backend = match QdrantHttpBackend::with_options(options).await {
        Ok(backend) => backend,
        Err(error) => {
            errorln!("skipping qdrant integration suite: {error}");
            return Ok(());
        }
    };

    run_integration_suite(backend, namespace, 128).await
}
