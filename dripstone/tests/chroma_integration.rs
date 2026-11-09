#![cfg(feature = "chroma-backend")]

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
use dripstone::backend::chroma::{ChromaBackend, ChromaBackendOptions};
use integration_suite::run_integration_suite;
use time::OffsetDateTime;

const DEFAULT_CHROMA_URL: &str = "http://127.0.0.1:8000";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chroma_integration_full_suite() -> Result<(), DripstoneError> {
    let url =
        std::env::var("EMBEDDING_CHROMA_URL").unwrap_or_else(|_| DEFAULT_CHROMA_URL.to_string());
    let namespace = format!(
        "chroma-integration-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let options = ChromaBackendOptions {
        base_url: url,
        api_key: common::chroma_api_key(),
        collection_prefix: Some("integration_".to_string()),
        expected_dimension: Some(128),
        timeout: None,
        batch_size: None,
        retry_config: None,
    };

    let backend = match ChromaBackend::with_options(options).await {
        Ok(backend) => backend,
        Err(error) => {
            errorln!("skipping chroma integration suite: {error}");
            return Ok(());
        }
    };

    run_integration_suite(backend, namespace, 128).await
}
