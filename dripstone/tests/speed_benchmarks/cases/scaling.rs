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
use crate::harness::corpus_train_chunks_simple;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use crate::speed_bench_report;
#[cfg(feature = "memory-backend")]
use dripstone::backend::memory::MemoryBackend as MemBackend;
use dripstone::{Chunk, DripstoneError, SearchQuery, traits::Backend};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serde_json::json;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serial_test::serial;

/// Scaling curve vs dataset size.
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_performance_scaling() -> Result<(), DripstoneError> {
    let corpus = datasets::load_integration_corpus().await?;
    let max_pts = dripstone_bench_max_points();
    let cap = corpus.base.len().min(max_pts).max(100);
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let base_namespace = format!(
        "scaling-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    // Skip n=100: fixed overhead dominates vs multi-k corpora and flakes the ratio assertion.
    let mut sizes = vec![500_usize, 1000, 5000];
    for s in &mut sizes {
        *s = (*s).min(cap);
    }
    sizes.sort_unstable();
    sizes.dedup();
    sizes.retain(|&s| s >= 10);
    let mut results = Vec::new();
    for size in sizes {
        let namespace = format!("{base_namespace}-{size}");
        let insert_start = Instant::now();
        let chunks = corpus_train_chunks_simple(&corpus, size, &namespace)?;
        backend.upsert(&chunks).await?;
        let insert_time = insert_start.elapsed();
        let nq = corpus.queries.len().max(1);
        let qv: Arc<[f32]> = Arc::from(corpus.queries[(size / 2) % nq].clone().into_boxed_slice());
        let query = SearchQuery::new(Arc::clone(&qv))
            .metric(corpus.search_metric)
            .limit(10)
            .namespace(&namespace);
        for _ in 0..2 {
            let _ = backend.search(&query).await?;
        }
        let sample_runs = 9_usize;
        let mut search_samples: Vec<Duration> = Vec::with_capacity(sample_runs);
        for _ in 0..sample_runs {
            let t0 = Instant::now();
            let _ = backend.search(&query).await?;
            search_samples.push(t0.elapsed());
        }
        search_samples.sort();
        let search_time = search_samples[search_samples.len() / 2];
        results.push((size, insert_time, search_time));
        outputln!("\nDataset size: {size}");
        outputln!("  Bulk insert time: {insert_time:?}");
        outputln!("  Search time (median of {sample_runs}): {search_time:?}");
        backend.delete_namespace(&namespace).await?;
    }
    if results.len() >= 2 {
        let (size1, _, search1) = results[0];
        let (size2, _, search2) = results[results.len() - 1];
        let size_ratio = size2 as f64 / size1 as f64;
        let time_ratio = search2.as_secs_f64() / search1.as_secs_f64();
        outputln!("\nScaling Analysis:");
        outputln!("  Size increased by: {size_ratio:.2}x");
        outputln!("  Search time increased by: {time_ratio:.2}x");
        assert!(
            time_ratio < size_ratio * 2.0,
            "Search time scaling should be reasonable: {time_ratio:.2}x vs {size_ratio:.2}x size increase"
        );
        speed_bench_report::record(json!({
            "test": "test_performance_scaling",
            "points": results.iter().map(|(size, ins, st)| json!({
                "corpus_size": size,
                "bulk_insert_ms": speed_bench_report::duration_ms(*ins),
                "search_median_ms": speed_bench_report::duration_ms(*st),
            })).collect::<Vec<_>>(),
            "size_ratio": size_ratio,
            "time_ratio": time_ratio,
        }));
    } else {
        speed_bench_report::record(json!({
            "test": "test_performance_scaling",
            "points": results.iter().map(|(size, ins, st)| json!({
                "corpus_size": size,
                "bulk_insert_ms": speed_bench_report::duration_ms(*ins),
                "search_median_ms": speed_bench_report::duration_ms(*st),
            })).collect::<Vec<_>>(),
            "note": "fewer than 2 size points; no ratio",
        }));
    }
    Ok(())
}

#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_batch_size_optimization() -> Result<(), DripstoneError> {
    let corpus = datasets::load_integration_corpus().await?;
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let base_namespace = format!(
        "batch-opt-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let batch_sizes = vec![10, 50, 100, 500, 1000];
    let total_records = corpus.base.len().min(1000).max(100);
    outputln!("\nBatch Size Optimization Test:");
    let mut batch_rows = Vec::new();
    for batch_size in batch_sizes {
        let namespace = format!("{base_namespace}-{batch_size}");
        let start = Instant::now();
        for batch_start in (0..total_records).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_records);
            let mut batch = Vec::new();
            for i in batch_start..batch_end {
                let vec_arc: Arc<[f32]> = Arc::from(corpus.base[i].clone().into_boxed_slice());
                batch.push(Chunk::new(
                    format!("vec-{i}"),
                    0,
                    "",
                    vec_arc,
                    0,
                    0,
                    1,
                    serde_json::json!({}),
                    Some(namespace.clone()),
                ));
            }
            backend.upsert(&batch).await?;
        }
        let elapsed = start.elapsed();
        let throughput = total_records as f64 / elapsed.as_secs_f64();
        outputln!("  Batch size {batch_size}: {elapsed:?} ({throughput:.0} records/sec)");
        batch_rows.push(json!({
            "batch_size": batch_size,
            "elapsed_ms": speed_bench_report::duration_ms(elapsed),
            "records_per_sec": throughput,
        }));
        backend.delete_namespace(&namespace).await?;
    }
    speed_bench_report::record(json!({
        "test": "test_batch_size_optimization",
        "total_records": total_records,
        "batches": batch_rows,
    }));
    Ok(())
}

#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_cold_vs_warm_performance() -> Result<(), DripstoneError> {
    let corpus = datasets::load_integration_corpus().await?;
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let namespace = format!(
        "cold-warm-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let max_pts = dripstone_bench_max_points();
    let data_size = corpus.base.len().min(2000).min(max_pts).max(100);
    let chunks = corpus_train_chunks_simple(&corpus, data_size, &namespace)?;
    backend.upsert(&chunks).await?;
    let nq = corpus.queries.len().max(1);
    let query_vec: Arc<[f32]> = Arc::from(corpus.queries[1000 % nq].clone().into_boxed_slice());
    let query = SearchQuery::new(Arc::clone(&query_vec))
        .metric(corpus.search_metric)
        .limit(10)
        .namespace(&namespace);
    let cold_start = Instant::now();
    let _ = backend.search(&query).await?;
    let cold_time = cold_start.elapsed();
    for i in 0..20 {
        let q_vec: Arc<[f32]> =
            Arc::from(corpus.queries[(1000 + i) % nq].clone().into_boxed_slice());
        let q = SearchQuery::new(q_vec)
            .metric(corpus.search_metric)
            .limit(10)
            .namespace(&namespace);
        let _ = backend.search(&q).await?;
    }
    let mut warm_times = Vec::new();
    for _ in 0..50 {
        let warm_start = Instant::now();
        let _ = backend.search(&query).await?;
        warm_times.push(warm_start.elapsed());
    }
    let warm_avg = warm_times.iter().sum::<Duration>() / warm_times.len() as u32;
    outputln!("\nCold Start vs Warm Cache:");
    outputln!("  Cold start: {cold_time:?}");
    outputln!("  Warm average: {warm_avg:?}");
    outputln!(
        "  Speedup: {:.2}x",
        cold_time.as_secs_f64() / warm_avg.as_secs_f64().max(1e-12)
    );
    let speedup = cold_time.as_secs_f64() / warm_avg.as_secs_f64().max(1e-12);
    speed_bench_report::record(json!({
        "test": "test_cold_vs_warm_performance",
        "data_size": data_size,
        "cold_start_ms": speed_bench_report::duration_ms(cold_time),
        "warm_avg_ms": speed_bench_report::duration_ms(warm_avg),
        "cold_div_warm": speedup,
    }));
    backend.delete_namespace(&namespace).await?;
    Ok(())
}
