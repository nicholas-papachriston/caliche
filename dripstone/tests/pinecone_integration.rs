#![cfg(feature = "pinecone-backend")]

mod integration_suite;

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::DripstoneError;
use dripstone::backend::pinecone::{PineconeBackend, PineconeBackendOptions};
use integration_suite::run_integration_suite;
use std::env;
use time::OffsetDateTime;

#[ignore = "requires EMBEDDING_PINECONE_HOST and EMBEDDING_PINECONE_API_KEY; run with --ignored"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pinecone_integration_full_suite() -> Result<(), DripstoneError> {
    let host = if let Some(host) = env_var("EMBEDDING_PINECONE_HOST") {
        host
    } else {
        errorln!("skipping pinecone integration suite: EMBEDDING_PINECONE_HOST not set");
        return Err(DripstoneError::storage(
            "pinecone opt-in or connection failed".to_string(),
        ));
    };
    let api_key = if let Some(key) = env_var("EMBEDDING_PINECONE_API_KEY") {
        key
    } else {
        errorln!("skipping pinecone integration suite: EMBEDDING_PINECONE_API_KEY not set");
        return Err(DripstoneError::storage(
            "pinecone opt-in or connection failed".to_string(),
        ));
    };

    let namespace = format!(
        "pinecone-integration-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let options = PineconeBackendOptions {
        base_url: host,
        api_key,
        dimension: 128,
        timeout: None,
        batch_size: None,
        retry_config: None,
    };

    let backend = match PineconeBackend::with_options(options).await {
        Ok(backend) => backend,
        Err(error) => {
            errorln!("skipping pinecone integration suite: {error}");
            return Err(DripstoneError::storage(
                "pinecone opt-in or connection failed".to_string(),
            ));
        }
    };

    run_integration_suite(backend, namespace, 128).await
}

fn env_var(key: &str) -> Option<String> {
    match env::var(key) {
        Ok(value) if !value.trim().is_empty() => Some(value),
        _ => None,
    }
}
