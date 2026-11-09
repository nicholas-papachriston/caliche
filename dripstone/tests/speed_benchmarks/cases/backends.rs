//! Speed benchmark cases (see parent `speed_benchmarks.rs`).

#[cfg(feature = "mmap-backend")]
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

#[cfg(feature = "mmap-backend")]
use std::sync::Arc;
#[cfg(feature = "mmap-backend")]
use std::time::{Duration, Instant};

#[cfg(feature = "mmap-backend")]
use crate::common::*;
#[cfg(feature = "mmap-backend")]
use crate::datasets;
#[cfg(feature = "memory-backend")]
use crate::harness::run_backend_matrix_if_configured;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use crate::speed_bench_report;
#[cfg(all(feature = "memory-backend", not(feature = "mmap-backend")))]
use dripstone::DripstoneError;
#[cfg(feature = "mmap-backend")]
use dripstone::{Chunk, DripstoneError, SearchQuery, traits::Backend};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serde_json::json;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serial_test::serial;

#[cfg(feature = "memory-backend")]
use super::helpers::json_benchmark_summary;

#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_backend_matrix_env() -> Result<(), DripstoneError> {
    let summaries = run_backend_matrix_if_configured().await?;
    speed_bench_report::record(json!({
        "test": "test_backend_matrix_env",
        "skipped": summaries.is_empty(),
        "rows": summaries.iter().map(json_benchmark_summary).collect::<Vec<_>>(),
    }));
    Ok(())
}

/// Memory-mapped layouts: architecture-aware v2 default, explicit v2 `f32`, and v3 int8.
#[cfg(feature = "mmap-backend")]
#[tokio::test]
#[serial]
async fn test_mmap_rkyv_cold_and_warm_search() -> Result<(), DripstoneError> {
    use dripstone::backend::mmap::{
        MmapBackend, MmapBackendConfig, MmapVectorEncoding, write_dripstone_mmap,
        write_dripstone_mmap_with_encoding,
    };
    use tempfile::tempdir;
    let corpus = datasets::load_integration_corpus().await?;
    let dim = corpus.dimension;
    let n = corpus
        .base
        .len()
        .min(2000)
        .min(dripstone_bench_max_points())
        .max(1);
    let namespace = Some("mmap-bench".to_string());
    let mut chunks = Vec::with_capacity(n);
    for i in 0..n {
        let vec_arc: Arc<[f32]> = Arc::from(corpus.base[i].clone().into_boxed_slice());
        chunks.push(Chunk::new(
            format!("doc-{i}"),
            0,
            "",
            vec_arc,
            0,
            0,
            1,
            serde_json::json!({}),
            namespace.clone(),
        ));
    }
    let nq = corpus.queries.len().max(1);
    let qv: Arc<[f32]> = Arc::from(corpus.queries[7 % nq].clone().into_boxed_slice());
    let query = SearchQuery::new(qv)
        .metric(corpus.search_metric)
        .limit(10)
        .namespace("mmap-bench");
    let mut rows = Vec::new();
    let variants = [
        #[cfg(target_arch = "aarch64")]
        ("mmap_v1_f32_default", "bench_v1_default.vf1"),
        #[cfg(not(target_arch = "aarch64"))]
        ("mmap_v1_f16_default", "bench_v1_default.vf1"),
        ("mmap_v1_f32_explicit", "bench_v1_f32.vf1"),
        ("mmap_v2_int8", "bench_v2_int8.vf2"),
    ];
    for (variant, file_name) in variants {
        let dir = tempdir().map_err(|e| DripstoneError::storage(format!("tempdir: {e}")))?;
        let path = dir.path().join(file_name);
        match variant {
            #[cfg(target_arch = "aarch64")]
            "mmap_v1_f32_default" => write_dripstone_mmap(&path, &chunks)?,
            #[cfg(not(target_arch = "aarch64"))]
            "mmap_v1_f16_default" => write_dripstone_mmap(&path, &chunks)?,
            "mmap_v1_f32_explicit" => {
                write_dripstone_mmap_with_encoding(&path, &chunks, MmapVectorEncoding::F32)?;
            }
            "mmap_v2_int8" => {
                write_dripstone_mmap_with_encoding(&path, &chunks, MmapVectorEncoding::Int8)?;
            }
            other => {
                return Err(DripstoneError::storage(format!(
                    "unknown mmap benchmark variant: {other}"
                )));
            }
        }
        let cold_start = Instant::now();
        let backend = MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), dim))?;
        let cold_open = cold_start.elapsed();
        let search_start = Instant::now();
        let _first = backend.search(&query).await?;
        let first_search = search_start.elapsed();
        let mut warm = Vec::new();
        for _ in 0..30 {
            let t = Instant::now();
            let _ = backend.search(&query).await?;
            warm.push(t.elapsed());
        }
        let avg_warm = warm.iter().copied().sum::<Duration>() / warm.len() as u32;
        outputln!("\n{variant} (n={n}, dim={dim}):");
        outputln!("  Cold open (map+validate): {cold_open:?}");
        outputln!("  First search after open: {first_search:?}");
        outputln!("  Avg warm search (30x): {avg_warm:?}");
        rows.push(json!({
            "layout": variant,
            "cold_open_ms": speed_bench_report::duration_ms(cold_open),
            "first_search_ms": speed_bench_report::duration_ms(first_search),
            "avg_warm_search_ms": speed_bench_report::duration_ms(avg_warm),
        }));
    }
    speed_bench_report::record(json!({
        "test": "test_mmap_rkyv_cold_and_warm_search",
        "n": n,
        "dim": dim,
        "rows": rows,
    }));
    Ok(())
}
