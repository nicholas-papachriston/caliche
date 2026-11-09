//! Small local benchmark: synchronous vs deferred `IndexedBackend` upsert and search.
//!
//! Run:
//! ```bash
//! cargo test --release -p dripstone --test deferred_index_benchmark --features memory-backend -- --nocapture
//! cargo test --release -p dripstone --test deferred_index_benchmark --features memory-backend,sqlite-backend -- --nocapture
//! ```

// Benchmark harness: helpers outside `#[test]` functions report results on
// stdout and treat backend failures as fatal.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::expect_used,
    clippy::print_stdout
)]

use std::time::{Duration, Instant};

use dripstone::backend::indexed::{IndexUpdateMode, IndexedBackend, IndexedBackendConfig};
use dripstone::backend::memory::MemoryBackend;
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::index::{HnswConfig, HnswIndex};
use dripstone::traits::Backend;

const SEARCH_SAMPLES: usize = 40;

fn make_vector(seed: usize, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| ((seed.wrapping_mul(31).wrapping_add(j.wrapping_mul(17))) % 997) as f32 / 997.0)
        .collect()
}

fn make_chunk(id: usize, dim: usize) -> Chunk {
    Chunk::new(
        &format!("vec-{id}"),
        0,
        &format!("chunk {id}"),
        make_vector(id, dim),
        0,
        8,
        1,
        serde_json::json!({}),
        None,
    )
}

fn make_chunks(count: usize, dim: usize) -> Vec<Chunk> {
    (0..count).map(|id| make_chunk(id, dim)).collect()
}

fn indexed_memory_backend(
    dim: usize,
    mode: IndexUpdateMode,
    flush_threshold: usize,
    search_flush_threshold: usize,
    max_tail_scan: usize,
) -> IndexedBackend<MemoryBackend> {
    IndexedBackend::new(
        MemoryBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), dim)),
        IndexedBackendConfig {
            auto_rebuild_threshold: 0,
            fallback_to_brute_force: false,
            overfetch_factor: 2,
            index_update_mode: mode,
            scan_unindexed_tail: true,
            deferred_flush_threshold: flush_threshold,
            deferred_search_flush_threshold: search_flush_threshold,
            max_tail_scan_records: max_tail_scan,
        },
    )
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn p50(mut samples: Vec<Duration>) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    samples.sort();
    samples[samples.len() / 2]
}

struct UpsertBench {
    mode: &'static str,
    count: usize,
    total_ms: f64,
    per_vector_us: f64,
}

async fn bench_single_upserts(
    dim: usize,
    mode: IndexUpdateMode,
    count: usize,
    flush_at_end: bool,
) -> UpsertBench {
    let backend = indexed_memory_backend(dim, mode, 0, 0, 0);
    let label = match (mode, flush_at_end) {
        (IndexUpdateMode::Synchronous, _) => "sync",
        (IndexUpdateMode::Deferred, true) => "deferred+flush",
        (IndexUpdateMode::Deferred, false) => "deferred_no_flush",
    };

    let start = Instant::now();
    for id in 0..count {
        backend
            .upsert(&[make_chunk(id, dim)])
            .await
            .expect("upsert");
    }
    if mode == IndexUpdateMode::Deferred && flush_at_end {
        backend.flush_pending_index().await.expect("flush");
    }
    let elapsed = start.elapsed();

    UpsertBench {
        mode: label,
        count,
        total_ms: ms(elapsed),
        per_vector_us: elapsed.as_secs_f64() * 1_000_000.0 / count as f64,
    }
}

struct SearchBench {
    label: &'static str,
    pending: usize,
    p50_ms: f64,
}

async fn bench_pending_tail_search(
    dim: usize,
    count: usize,
    search_flush_threshold: usize,
    max_tail_scan: usize,
) -> SearchBench {
    let backend = indexed_memory_backend(
        dim,
        IndexUpdateMode::Deferred,
        0,
        search_flush_threshold,
        max_tail_scan,
    );
    backend
        .upsert(&make_chunks(count, dim))
        .await
        .expect("upsert");
    assert_eq!(backend.pending_index_count(), count);

    let mut samples = Vec::with_capacity(SEARCH_SAMPLES);
    for i in 0..SEARCH_SAMPLES {
        let query = SearchQuery::new(make_vector(i * 13, dim))
            .metric(Metric::Cosine)
            .limit(10);
        let start = Instant::now();
        let hits = backend.search(&query).await.expect("search");
        assert!(!hits.is_empty());
        samples.push(start.elapsed());
    }

    SearchBench {
        label: if search_flush_threshold > 0 {
            "search_flush"
        } else if max_tail_scan > 0 {
            "tail_cap"
        } else {
            "pending_tail"
        },
        pending: count,
        p50_ms: ms(p50(samples)),
    }
}

fn print_upsert_row(row: &UpsertBench) {
    println!(
        "  {:<16} n={:<5} total={:>8.2} ms  per_vector={:>8.1} us",
        row.mode, row.count, row.total_ms, row.per_vector_us
    );
}

fn print_search_row(row: &SearchBench) {
    println!(
        "  {:<28} pending={:<5} search_p50={:>7.2} ms",
        row.label, row.pending, row.p50_ms
    );
}

async fn run_dim_benchmark(dim: usize, counts: &[usize]) {
    println!("\n=== IndexedBackend deferred vs sync (dim={dim}) ===\n");
    println!("Upsert (single vector per call):");
    for &count in counts {
        let sync = bench_single_upserts(dim, IndexUpdateMode::Synchronous, count, false).await;
        let deferred_hot = bench_single_upserts(dim, IndexUpdateMode::Deferred, count, false).await;
        let deferred_flush =
            bench_single_upserts(dim, IndexUpdateMode::Deferred, count, true).await;
        let hot_speedup = sync.total_ms / deferred_hot.total_ms;
        print_upsert_row(&sync);
        print_upsert_row(&deferred_hot);
        print_upsert_row(&deferred_flush);
        println!(
            "    -> deferred hot-path speedup (no flush): {hot_speedup:.2}x ({:.1}% faster)",
            (1.0 - deferred_hot.total_ms / sync.total_ms) * 100.0
        );
        println!();
    }

    println!("Search with deferred tail scan (all vectors still pending in ANN):");
    for &count in counts {
        let pending = bench_pending_tail_search(dim, count, 0, 0).await;
        print_search_row(&pending);

        let search_flush = bench_pending_tail_search(dim, count, count, 0).await;
        print_search_row(&search_flush);

        let tail_cap = bench_pending_tail_search(dim, count.min(500), 0, 100).await;
        print_search_row(&tail_cap);

        let backend = indexed_memory_backend(dim, IndexUpdateMode::Deferred, 0, 0, 0);
        backend
            .upsert(&make_chunks(count, dim))
            .await
            .expect("upsert");
        backend.flush_pending_index().await.expect("flush");
        let mut flushed_samples = Vec::with_capacity(SEARCH_SAMPLES);
        for i in 0..SEARCH_SAMPLES {
            let query = SearchQuery::new(make_vector(i * 13, dim))
                .metric(Metric::Cosine)
                .limit(10);
            let start = Instant::now();
            let hits = backend.search(&query).await.expect("search");
            assert!(!hits.is_empty());
            flushed_samples.push(start.elapsed());
        }
        let flushed_p50 = ms(p50(flushed_samples));
        println!(
            "  {:<28} pending={:<5} search_p50={:>7.2} ms",
            "after_flush", 0, flushed_p50
        );
        let penalty = pending.p50_ms / flushed_p50;
        println!("    -> tail-scan search penalty vs flushed: {penalty:.2}x\n");
    }
}

async fn bench_interleaved(
    dim: usize,
    upsert_count: usize,
    searches_per_block: usize,
) -> (f64, f64) {
    async fn run(
        dim: usize,
        mode: IndexUpdateMode,
        upsert_count: usize,
        searches_per_block: usize,
    ) -> f64 {
        let backend = indexed_memory_backend(dim, mode, 0, 0, 0);
        let start = Instant::now();
        for id in 0..upsert_count {
            backend
                .upsert(&[make_chunk(id, dim)])
                .await
                .expect("upsert");
            if id > 0 && id % 10 == 0 {
                for s in 0..searches_per_block {
                    let query = SearchQuery::new(make_vector(id + s, dim))
                        .metric(Metric::Cosine)
                        .limit(5);
                    let hits = backend.search(&query).await.expect("search");
                    assert!(!hits.is_empty());
                }
            }
        }
        if mode == IndexUpdateMode::Deferred {
            backend.flush_pending_index().await.expect("flush");
        }
        ms(start.elapsed())
    }

    let sync_ms = run(
        dim,
        IndexUpdateMode::Synchronous,
        upsert_count,
        searches_per_block,
    )
    .await;
    let deferred_ms = run(
        dim,
        IndexUpdateMode::Deferred,
        upsert_count,
        searches_per_block,
    )
    .await;
    (sync_ms, deferred_ms)
}

#[tokio::test]
async fn deferred_index_local_benchmark() {
    run_dim_benchmark(384, &[500, 2_000]).await;
    run_dim_benchmark(1536, &[100, 200]).await;

    println!("Interleaved ingest + search (upsert every vector, search every 10 upserts):");
    let (sync_ms, deferred_ms) = bench_interleaved(384, 200, 3).await;
    println!("  sync total={sync_ms:.2} ms");
    println!("  deferred total={deferred_ms:.2} ms");
    if deferred_ms > 0.0 {
        println!("    -> deferred speedup: {:.2}x", sync_ms / deferred_ms);
    }

    println!("=== done ===\n");
}

#[cfg(feature = "sqlite-backend")]
mod sqlite_bench {
    use super::*;
    use dripstone::backend::sqlite::{MemoryStrategy, SqliteBackend, SqliteBackendConfig};

    async fn indexed_sqlite_backend(
        dim: usize,
        db_path: &std::path::Path,
    ) -> IndexedBackend<SqliteBackend> {
        let sqlite = SqliteBackend::new(SqliteBackendConfig {
            db_path: db_path.to_path_buf(),
            memory_strategy: MemoryStrategy::WriteThrough {
                max_cache_size: 10_000,
                ttl_secs: None,
            },
            max_connections: 4,
            table_name: "bench_vectors".to_string(),
            ensure_schema: true,
        })
        .await
        .expect("sqlite");
        IndexedBackend::new(
            sqlite,
            Box::new(HnswIndex::new(HnswConfig::default(), dim)),
            IndexedBackendConfig {
                auto_rebuild_threshold: 0,
                fallback_to_brute_force: false,
                overfetch_factor: 2,
                index_update_mode: IndexUpdateMode::Deferred,
                scan_unindexed_tail: true,
                deferred_flush_threshold: 0,
                deferred_search_flush_threshold: 0,
                max_tail_scan_records: 0,
            },
        )
    }

    #[tokio::test]
    async fn deferred_index_sqlite_benchmark() {
        let dim = 384;
        let count = 300;
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("vectors.db");

        let backend = indexed_sqlite_backend(dim, &db_path).await;
        let upsert_start = Instant::now();
        for id in 0..count {
            backend
                .upsert(&[make_chunk(id, dim)])
                .await
                .expect("upsert");
        }
        let upsert_ms = ms(upsert_start.elapsed());
        assert_eq!(backend.pending_index_count(), count);

        let mut search_samples = Vec::with_capacity(20);
        for i in 0..20 {
            let query = SearchQuery::new(make_vector(i * 7, dim))
                .metric(Metric::Cosine)
                .limit(10);
            let start = Instant::now();
            let hits = backend.search(&query).await.expect("search");
            assert!(!hits.is_empty());
            search_samples.push(start.elapsed());
        }
        let search_p50 = ms(p50(search_samples));

        backend.flush_pending_index().await.expect("flush");
        assert_eq!(backend.pending_index_count(), 0);

        println!("\n=== SQLite deferred benchmark (dim={dim}, n={count}) ===");
        println!("  upsert_no_flush={upsert_ms:.2} ms");
        println!("  search_pending_tail_p50={search_p50:.2} ms");
        println!("=== done ===\n");
    }
}
