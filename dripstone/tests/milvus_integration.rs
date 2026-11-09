#![cfg(feature = "milvus-backend")]

mod integration_suite;

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::DripstoneError;
use dripstone::backend::milvus::{MilvusBackend, MilvusBackendOptions};
use integration_suite::run_integration_suite;
use time::OffsetDateTime;

const DEFAULT_MILVUS_URL: &str = "http://127.0.0.1:19530";

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn milvus_integration_full_suite() -> Result<(), DripstoneError> {
    let url =
        std::env::var("EMBEDDING_MILVUS_URL").unwrap_or_else(|_| DEFAULT_MILVUS_URL.to_string());
    let namespace = format!(
        "milvus-integration-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let options = MilvusBackendOptions {
        endpoint: url,
        collection_prefix: Some("integration_".to_string()),
        vector_dimension: 128,
        index_type: Some("IVF_FLAT".to_string()),
        metric_type: Some("COSINE".to_string()),
        nlist: Some(128),
        timeout: None,
        batch_size: None,
        retry_config: None,
        flush_policy: None,
    };

    let backend = match MilvusBackend::with_options(options).await {
        Ok(backend) => backend,
        Err(error) => {
            errorln!("skipping milvus integration suite: {error}");
            return Ok(());
        }
    };

    run_integration_suite(backend, namespace, 128).await
}
