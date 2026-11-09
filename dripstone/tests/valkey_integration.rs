#![cfg(feature = "redis-backend")]

mod common;
mod integration_suite;

use dripstone::DripstoneError;
use dripstone::backend::redis::RedisBackend;
use integration_suite::run_integration_suite;
use time::OffsetDateTime;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn valkey_integration_full_suite() -> Result<(), DripstoneError> {
    let url = std::env::var("VALKEY_URL").unwrap_or_else(|_| common::valkey_url_local());
    let namespace = format!(
        "valkey-integration-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend = RedisBackend::with_prefix(&url, "integration-valkey").await?;

    run_integration_suite(backend, namespace, 128).await
}
