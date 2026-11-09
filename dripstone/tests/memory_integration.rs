#![cfg(feature = "memory-backend")]

mod integration_suite;

use dripstone::DripstoneError;
use dripstone::backend::memory::MemoryBackend;
use integration_suite::run_integration_suite;
use time::OffsetDateTime;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn memory_integration_full_suite() -> Result<(), DripstoneError> {
    let namespace = format!(
        "memory-integration-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend = MemoryBackend::new();

    run_integration_suite(backend, namespace, 128).await
}
