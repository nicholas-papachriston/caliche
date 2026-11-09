#![cfg(all(feature = "corpus", feature = "hnsw"))]
//! Speed and throughput benchmark tests for embedding operations.
//!
//! This target compiles only with `--features corpus,hnsw` (the shared loader and the HNSW index).
//!
//! Vectors are loaded from the same ann-benchmarks HDF5 pipeline as integration tests
//! (`tests/integration/datasets.rs`): `DRIPSTONE_INT_DATASET_URL`, `DRIPSTONE_INT_TRAIN_MAX`,
//! `DRIPSTONE_INT_QUERY_MAX`, `DRIPSTONE_INT_SEARCH_METRIC`, `DRIPSTONE_INT_GROUND_TRUTH_SOURCE`, etc.
//!
//! ## Smoke vs comparison tier
//!
//! By default, tests use CI-friendly sizes. For larger corpora and sweeps, set
//! `DRIPSTONE_BENCH_LONG=1` and optionally `DRIPSTONE_BENCH_MAX_POINTS` (default 50000).
//!
//! Optional multi-backend comparison: `DRIPSTONE_BENCH_BACKENDS=memory,mmap,redis,postgres,qdrant,chroma,valkey`
//!
//! ANN vs brute at 10K / 50K / 100K: `DRIPSTONE_BENCH_HNSW_SCALE=1` and
//! `DRIPSTONE_BENCH_MAX_POINTS=120000` (or higher); prefer `cargo test --release` for stable latencies.
//!
//! HNSW parameter sweep: `DRIPSTONE_BENCH_HNSW_SWEEP=1` to print recall/latency tradeoffs for
//! multiple `m`, `ef_search`, and rerank-overfetch combinations on the larger corpus cases.
//!
//! Aggregate JSON is written to `dripstone/target/dripstone/speed_benchmarks/results/speed_benchmarks.json`
//! and printed when `test_zzzz_speed_benchmarks_write_aggregate_json` runs. For a full file when
//! the test harness schedules tests out of order, use `--test-threads=1`.
//!
//! ```text
//! cargo test --test speed_benchmarks --features "corpus,memory-backend,mmap-backend" -p dripstone -- --nocapture
//! cargo test --test speed_benchmarks --features full -p dripstone -- --nocapture
//! ```

#[path = "common/mod.rs"]
mod common;

#[path = "integration/ground_truth.rs"]
mod ground_truth;

#[path = "integration/datasets.rs"]
mod datasets;

#[path = "integration/parquet_corpus.rs"]
mod parquet_corpus;

#[cfg(feature = "memory-backend")]
#[path = "speed_benchmarks/harness.rs"]
mod harness;

#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
#[path = "speed_benchmarks/report.rs"]
mod speed_bench_report;

#[path = "speed_benchmarks/cases/mod.rs"]
mod cases;
