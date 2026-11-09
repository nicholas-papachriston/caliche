//! Speed benchmark cases (see parent `speed_benchmarks.rs`).

macro_rules! outputln {
    () => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
    ($($arg:tt)+) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::sync::Arc;
use std::time::{Duration, Instant};
use time::OffsetDateTime;

use crate::common::*;
use crate::datasets;
#[cfg(feature = "memory-backend")]
use crate::harness::{
    BenchmarkCase, FilterSelectivity, PercentileStats, bench_backend_case_with_corpus,
    corpus_train_chunks_simple,
};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use crate::speed_bench_report;
#[cfg(feature = "memory-backend")]
use dripstone::backend::memory::MemoryBackend as MemBackend;
use dripstone::{Chunk, DripstoneError, SearchQuery, traits::Backend};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serde_json::json;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serial_test::serial;

/// Test search latency percentiles (smoke).
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_search_latency_percentiles() -> Result<(), DripstoneError> {
    let corpus = datasets::load_integration_corpus().await?;
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let namespace = format!(
        "latency-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let max_pts = dripstone_bench_max_points();
    let n = corpus.base.len().min(max_pts).min(1000).max(10);
    let q = corpus.queries.len().min(100).max(1);
    let case = BenchmarkCase {
        name: "latency_smoke",
        dimension: corpus.dimension,
        corpus_size: n,
        query_count: q,
        limit: 10,
        metric: corpus.search_metric,
        filter: FilterSelectivity::None,
    };
    let summary =
        bench_backend_case_with_corpus("memory", backend.clone(), &case, &namespace, &corpus)
            .await?;
    summary.search.print("Search");
    speed_bench_report::record(json!({
        "test": "test_search_latency_percentiles",
        "case": case.name,
        "search_p50_ms": speed_bench_report::duration_ms(summary.search.p50),
        "search_p99_ms": speed_bench_report::duration_ms(summary.search.p99),
        "qps": summary.qps,
    }));
    assert!(
        summary.search.p99 < Duration::from_millis(100),
        "p99 latency should be < 100ms, got {:?}",
        summary.search.p99
    );
    backend.delete_namespace(&namespace).await?;
    Ok(())
}

/// Test upsert latency percentiles.
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_upsert_latency_percentiles() -> Result<(), DripstoneError> {
    let corpus = datasets::load_integration_corpus().await?;
    let ntrain = corpus.base.len();
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let namespace = format!(
        "upsert-latency-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let iterations = 200;
    let mut stats = PercentileStats::new();
    for i in 0..iterations {
        let row = &corpus.base[i % ntrain];
        let vec_arc: Arc<[f32]> = Arc::from(row.clone().into_boxed_slice());
        let chunk = Chunk::new(
            format!("vec-{i}"),
            0,
            "",
            vec_arc,
            0,
            0,
            1,
            serde_json::json!({}),
            Some(namespace.clone()),
        );
        let start = Instant::now();
        backend.upsert(&[chunk]).await?;
        stats.add(start.elapsed());
    }
    let results = stats.calculate();
    results.print("Upsert");
    speed_bench_report::record(json!({
        "test": "test_upsert_latency_percentiles",
        "iterations": iterations,
        "upsert_p50_ms": speed_bench_report::duration_ms(results.p50),
        "upsert_p99_ms": speed_bench_report::duration_ms(results.p99),
    }));
    assert!(
        results.p99 < Duration::from_millis(50),
        "p99 upsert latency should be < 50ms, got {:?}",
        results.p99
    );
    backend.delete_namespace(&namespace).await?;
    Ok(())
}

/// Throughput over a fixed window.
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_search_throughput() -> Result<(), DripstoneError> {
    let corpus = datasets::load_integration_corpus().await?;
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let namespace = format!(
        "throughput-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let max_pts = dripstone_bench_max_points();
    let data_size = corpus.base.len().min(5000).min(max_pts).max(100);
    let chunks = corpus_train_chunks_simple(&corpus, data_size, &namespace)?;
    backend.upsert(&chunks).await?;
    let nq = corpus.queries.len().max(1);
    let query_pool: Vec<SearchQuery> = (0..nq)
        .map(|i| {
            let qv: Arc<[f32]> = Arc::from(corpus.queries[i].clone().into_boxed_slice());
            SearchQuery::new(qv)
                .metric(corpus.search_metric)
                .limit(10)
                .namespace(&namespace)
        })
        .collect();
    let duration = Duration::from_secs(5);
    let start = Instant::now();
    let mut query_count = 0;
    while start.elapsed() < duration {
        let _ = backend.search(&query_pool[query_count % nq]).await?;
        query_count += 1;
    }
    let elapsed = start.elapsed();
    let qps = query_count as f64 / elapsed.as_secs_f64();
    outputln!("\nSearch Throughput:");
    outputln!("  Total queries: {query_count}");
    outputln!("  Duration: {elapsed:?}");
    outputln!("  Queries/sec: {qps:.2}");
    speed_bench_report::record(json!({
        "test": "test_search_throughput",
        "data_size": data_size,
        "total_queries": query_count,
        "duration_secs": elapsed.as_secs_f64(),
        "qps": qps,
    }));
    assert!(
        qps >= 30.0,
        "Search throughput should be >= 30 qps, got {qps:.2}"
    );
    backend.delete_namespace(&namespace).await?;
    Ok(())
}
