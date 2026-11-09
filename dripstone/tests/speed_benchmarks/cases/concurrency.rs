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
use crate::harness::{concurrency_task_counts, corpus_train_chunks_simple};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use crate::speed_bench_report;
#[cfg(feature = "memory-backend")]
use dripstone::backend::memory::MemoryBackend as MemBackend;
use dripstone::{Chunk, DripstoneError, Metric as VfMetric, SearchQuery, traits::Backend};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serde_json::json;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serial_test::serial;

#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_concurrent_read_throughput() -> Result<(), DripstoneError> {
    use tokio::task::JoinSet;
    let corpus = datasets::load_integration_corpus().await?;
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let namespace = format!(
        "concurrent-read-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let max_pts = dripstone_bench_max_points();
    let data_size = corpus.base.len().min(3000).min(max_pts).max(100);
    let chunks = corpus_train_chunks_simple(&corpus, data_size, &namespace)?;
    backend.upsert(&chunks).await?;
    let corpus = Arc::new(corpus);
    let num_tasks = 10;
    let queries_per_task = 50;
    let start = Instant::now();
    let mut tasks = JoinSet::new();
    for task_id in 0..num_tasks {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        let corpus_c = Arc::clone(&corpus);
        tasks.spawn(async move {
            let mut count = 0;
            let nq = corpus_c.queries.len().max(1);
            for i in 0..queries_per_task {
                let qv: Arc<[f32]> = Arc::from(
                    corpus_c.queries[(task_id * 1000 + i) % nq]
                        .clone()
                        .into_boxed_slice(),
                );
                let query = SearchQuery::new(qv)
                    .metric(corpus_c.search_metric)
                    .limit(10)
                    .namespace(&ns);
                let _ = backend_clone.search(&query).await?;
                count += 1;
            }
            Ok::<usize, DripstoneError>(count)
        });
    }
    let mut total_queries = 0;
    while let Some(result) = tasks.join_next().await {
        let count = result.map_err(|e| DripstoneError::storage(format!("Join error: {e}")))??;
        total_queries += count;
    }
    let elapsed = start.elapsed();
    let qps = total_queries as f64 / elapsed.as_secs_f64();
    outputln!("\nConcurrent Read Throughput:");
    outputln!("  Concurrent tasks: {num_tasks}");
    outputln!("  Total queries: {total_queries}");
    outputln!("  Duration: {elapsed:?}");
    outputln!("  Queries/sec: {qps:.2}");
    speed_bench_report::record(json!({
        "test": "test_concurrent_read_throughput",
        "data_size": data_size,
        "concurrent_tasks": num_tasks,
        "queries_per_task": queries_per_task,
        "total_queries": total_queries,
        "duration_secs": elapsed.as_secs_f64(),
        "qps": qps,
    }));
    assert!(
        qps >= 35.0,
        "Concurrent read throughput should be >= 35 qps (integration corpus dim), got {qps:.2}"
    );
    backend.delete_namespace(&namespace).await?;
    Ok(())
}

/// Concurrency sweep: task count vs QPS.
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_concurrency_sweep() -> Result<(), DripstoneError> {
    use tokio::task::JoinSet;
    let corpus = Arc::new(datasets::load_integration_corpus().await?);
    let max_pts = dripstone_bench_max_points();
    let data_size = corpus.base.len().min(3000).min(max_pts).max(100);
    let queries_per_task = 40;
    outputln!("\nConcurrency sweep (queries per task: {queries_per_task})");
    let mut sweep_rows = Vec::new();
    for num_tasks in concurrency_task_counts() {
        let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
        let namespace = format!(
            "conc-sweep-{num_tasks}-{}",
            OffsetDateTime::now_utc().unix_timestamp_nanos()
        );
        let chunks = corpus_train_chunks_simple(corpus.as_ref(), data_size, &namespace)?;
        backend.upsert(&chunks).await?;
        let start = Instant::now();
        let mut tasks = JoinSet::new();
        for task_id in 0..num_tasks {
            let backend_clone = Arc::clone(&backend);
            let ns = namespace.clone();
            let corpus_c = Arc::clone(&corpus);
            tasks.spawn(async move {
                let mut count = 0;
                let nq = corpus_c.queries.len().max(1);
                for i in 0..queries_per_task {
                    let qv: Arc<[f32]> = Arc::from(
                        corpus_c.queries[(task_id * 1000 + i) % nq]
                            .clone()
                            .into_boxed_slice(),
                    );
                    let query = SearchQuery::new(qv)
                        .metric(corpus_c.search_metric)
                        .limit(10)
                        .namespace(&ns);
                    let _ = backend_clone.search(&query).await?;
                    count += 1;
                }
                Ok::<usize, DripstoneError>(count)
            });
        }
        let mut total = 0_usize;
        while let Some(result) = tasks.join_next().await {
            total += result.map_err(|e| DripstoneError::storage(format!("Join error: {e}")))??;
        }
        let elapsed = start.elapsed();
        let qps = total as f64 / elapsed.as_secs_f64().max(1e-9);
        outputln!("  tasks={num_tasks} total_q={total} qps={qps:.1} wall={elapsed:?}");
        sweep_rows.push(json!({
            "tasks": num_tasks,
            "total_q": total,
            "qps": qps,
            "wall_ms": speed_bench_report::duration_ms(elapsed),
        }));
        backend.delete_namespace(&namespace).await?;
    }
    speed_bench_report::record(json!({
        "test": "test_concurrency_sweep",
        "queries_per_task": queries_per_task,
        "data_size": data_size,
        "sweep": sweep_rows,
    }));
    Ok(())
}

#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_mixed_read_write_throughput() -> Result<(), DripstoneError> {
    use tokio::task::JoinSet;
    let corpus = Arc::new(datasets::load_integration_corpus().await?);
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let namespace = format!(
        "mixed-rw-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let init_n = corpus.base.len().min(500).max(50);
    let chunks = corpus_train_chunks_simple(corpus.as_ref(), init_n, &namespace)?;
    backend.upsert(&chunks).await?;
    let duration = Duration::from_secs(3);
    let start = Instant::now();
    let mut tasks = JoinSet::new();
    for reader_id in 0..5 {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        let start_time = start;
        let corpus_c = Arc::clone(&corpus);
        tasks.spawn(async move {
            let mut reads = 0;
            let nq = corpus_c.queries.len().max(1);
            while start_time.elapsed() < duration {
                let qv: Arc<[f32]> = Arc::from(
                    corpus_c.queries[(reader_id * 1000 + reads) % nq]
                        .clone()
                        .into_boxed_slice(),
                );
                let query = SearchQuery::new(qv)
                    .metric(corpus_c.search_metric)
                    .limit(10)
                    .namespace(&ns);
                let _ = backend_clone.search(&query).await?;
                reads += 1;
            }
            Ok::<(usize, usize), DripstoneError>((reads, 0))
        });
    }
    for writer_id in 0..2 {
        let backend_clone = Arc::clone(&backend);
        let ns = namespace.clone();
        let start_time = start;
        let corpus_c = Arc::clone(&corpus);
        tasks.spawn(async move {
            let mut writes = 0;
            let mut idx = 0;
            let ntrain = corpus_c.base.len();
            while start_time.elapsed() < duration {
                let row = &corpus_c.base[(writer_id * 10000 + idx) % ntrain];
                let vec_arc: Arc<[f32]> = Arc::from(row.clone().into_boxed_slice());
                let chunk = Chunk::new(
                    format!("write-{writer_id}-{idx}"),
                    0,
                    "",
                    vec_arc,
                    0,
                    0,
                    1,
                    serde_json::json!({}),
                    Some(ns.clone()),
                );
                backend_clone.upsert(&[chunk]).await?;
                writes += 1;
                idx += 1;
            }
            Ok::<(usize, usize), DripstoneError>((0, writes))
        });
    }
    let mut total_reads = 0;
    let mut total_writes = 0;
    while let Some(result) = tasks.join_next().await {
        let (reads, writes) =
            result.map_err(|e| DripstoneError::storage(format!("Join error: {e}")))??;
        total_reads += reads;
        total_writes += writes;
    }
    let elapsed = start.elapsed();
    let read_qps = total_reads as f64 / elapsed.as_secs_f64();
    let write_qps = total_writes as f64 / elapsed.as_secs_f64();
    outputln!("\nMixed Read/Write Throughput:");
    outputln!("  Duration: {elapsed:?}");
    outputln!("  Total reads: {total_reads} ({read_qps:.2} reads/sec)");
    outputln!("  Total writes: {total_writes} ({write_qps:.2} writes/sec)");
    outputln!("  Combined: {:.2} ops/sec", read_qps + write_qps);
    speed_bench_report::record(json!({
        "test": "test_mixed_read_write_throughput",
        "duration_secs": elapsed.as_secs_f64(),
        "total_reads": total_reads,
        "total_writes": total_writes,
        "read_qps": read_qps,
        "write_qps": write_qps,
        "combined_qps": read_qps + write_qps,
    }));
    backend.delete_namespace(&namespace).await?;
    Ok(())
}

/// Uses synthetic vectors so multiple embedding dimensions can be compared in one run (integration
/// HDF5 is fixed-D).
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_dimension_performance_impact() -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let base_namespace = format!(
        "dim-perf-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let dimensions = vec![64, 128, 256, 384, 768, 1536];
    let record_count = 500;
    outputln!("\nDimension Performance Impact:");
    let mut dim_rows = Vec::new();
    for dim in dimensions {
        let namespace = format!("{base_namespace}-{dim}");
        let insert_start = Instant::now();
        let mut ch = Vec::with_capacity(record_count);
        for i in 0..record_count {
            ch.push(Chunk::new(
                format!("vec-{i}"),
                0,
                "",
                build_vector(i, dim),
                0,
                0,
                1,
                serde_json::json!({}),
                Some(namespace.clone()),
            ));
        }
        backend.upsert(&ch).await?;
        let insert_time = insert_start.elapsed();
        let query_vec = build_vector(250, dim);
        let query = SearchQuery::new(query_vec.clone())
            .metric(VfMetric::Cosine)
            .limit(10)
            .namespace(&namespace);
        let mut search_times = Vec::new();
        for _ in 0..20 {
            let search_start = Instant::now();
            let _ = backend.search(&query).await?;
            search_times.push(search_start.elapsed());
        }
        let avg_search = search_times.iter().sum::<Duration>() / search_times.len() as u32;
        outputln!("  Dimension {dim}:");
        outputln!("    Insert time: {insert_time:?}");
        outputln!("    Avg search time: {avg_search:?}");
        dim_rows.push(json!({
            "dimension": dim,
            "insert_ms": speed_bench_report::duration_ms(insert_time),
            "avg_search_ms": speed_bench_report::duration_ms(avg_search),
        }));
        backend.delete_namespace(&namespace).await?;
    }
    speed_bench_report::record(json!({
        "test": "test_dimension_performance_impact",
        "record_count": record_count,
        "by_dimension": dim_rows,
    }));
    Ok(())
}
