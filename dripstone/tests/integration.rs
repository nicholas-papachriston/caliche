//! `Dripstone` AWS integration benchmarks (`ColdForge`, S3 Vector Buckets, tiered mmap hot tier).
//!
//! Run (single command):
//! `cargo test -p dripstone --test integration \
//!   --features coldforge-backend,mmap-backend,memory-backend,redis-backend -- --nocapture --test-threads=1`
//!
//! Requires AWS credentials and (unless overridden) creates a standard S3 bucket and S3 Vector
//! bucket/index. Downloads an ann-benchmarks HDF5 corpus on first run (default:
//! `nytimes-256-angular`, normalized under `$DRIPSTONE_CACHE_DIR/corpora/<name>/<rev>/`; override with
//! `DRIPSTONE_CORPUS` or `DRIPSTONE_INT_DATASET_URL`). The integration cap stays 4_000 training rows
//! unless `DRIPSTONE_CORPUS_TRAIN_MAX` or `DRIPSTONE_INT_TRAIN_MAX` is set. Recall uses subset-exact neighbors by default; see
//! `DRIPSTONE_INT_GROUND_TRUTH_SOURCE` in `dripstone/README.md`.
//! Tiered scenarios that use Redis (`s3vector_tiered_redis_mmap`) need a reachable Redis
//! (`REDIS_URL` or compose defaults via `STACK_REDIS_PASSWORD`); if Redis is down, that scenario
//! is skipped with a log line.

#[path = "integration/mod.rs"]
mod vf_integration;

use dripstone::DripstoneError;
#[cfg(feature = "memory-backend")]
use serial_test::serial;

#[ignore = "requires DRIPSTONE_INT_S3_BUCKET, DRIPSTONE_INT_S3_VECTOR_BUCKET, and AWS credentials; run with --ignored"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dripstone_integration_benchmarks() -> Result<(), DripstoneError> {
    vf_integration::run_all().await
}

#[cfg(feature = "memory-backend")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn dripstone_spfresh_filtered_recall_benchmark() -> Result<(), DripstoneError> {
    vf_integration::run_spfresh_filter_benchmark().await
}
