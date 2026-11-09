#![cfg(feature = "postgres-backend")]

mod common;
mod integration_suite;

use dripstone::DripstoneError;
use dripstone::backend::postgres::{PostgresBackend, PostgresBackendOptions};
use integration_suite::run_integration_suite;
use time::OffsetDateTime;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn postgres_integration_full_suite() -> Result<(), DripstoneError> {
    let url = common::postgres_url_local();
    let namespace = format!(
        "pg-integration-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let options = PostgresBackendOptions {
        table_name: "embeddings_integration_128".to_string(),
        ensure_schema: true,
        max_connections: 10,
        vector_dimension: Some(128),
        hnsw_index: None,
    };

    let backend = PostgresBackend::with_options(url.as_str(), options).await?;

    run_integration_suite(backend, namespace, 128).await
}
