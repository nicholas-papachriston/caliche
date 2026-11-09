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
use time::OffsetDateTime;

use crate::common::*;
use crate::datasets;
#[cfg(feature = "memory-backend")]
use crate::harness::{
    FilterSelectivity, active_benchmark_cases, bench_backend_case_with_corpus,
    bench_exact_vs_hnsw_with_corpus, hnsw_sweep_configs, print_hnsw_sweep_row,
    tuned_hnsw_config_for_corpus,
};
#[cfg(all(feature = "memory-backend", feature = "mmap-backend"))]
use crate::harness::{bench_memory_vs_mmap, bench_memory_vs_mmap_with_encoding};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use crate::speed_bench_report;
#[cfg(feature = "memory-backend")]
use dripstone::backend::memory::MemoryBackend as MemBackend;
use dripstone::{DripstoneError, traits::Backend};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serde_json::json;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serial_test::serial;

#[cfg(feature = "memory-backend")]
use super::helpers::json_benchmark_summary;

/// Harness-driven cases (smoke + optional long tier).
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_benchmark_harness_matrix() -> Result<(), DripstoneError> {
    let corpus = datasets::load_integration_corpus().await?;
    let backend: Arc<dyn Backend> = Arc::new(MemBackend::new());
    let base = format!(
        "harness-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    outputln!(
        "harness matrix dataset: {} dim={} metric={:?} neighbors_gt={} ground_truth_source={:?}",
        corpus.name,
        corpus.dimension,
        corpus.search_metric,
        corpus.ground_truth.is_some(),
        corpus.diagnostics.ground_truth_source
    );
    let mut rows = Vec::new();
    for case in active_benchmark_cases(&corpus) {
        let ns = format!("{base}-{}", case.name);
        let s =
            bench_backend_case_with_corpus("memory", backend.clone(), &case, &ns, &corpus).await?;
        outputln!(
            "  {} | p50={:?} qps={:.1} upsert={:?}",
            case.name,
            s.search.p50,
            s.qps,
            s.upsert
        );
        rows.push(s);
        backend.delete_namespace(&ns).await?;
    }
    crate::harness::print_summary_table(&rows);
    speed_bench_report::record(json!({
        "test": "test_benchmark_harness_matrix",
        "dataset": corpus.name.as_str(),
        "dim": corpus.dimension,
        "metric": format!("{:?}", corpus.search_metric),
        "neighbors_gt": corpus.ground_truth.is_some(),
        "ground_truth_source": format!("{:?}", corpus.diagnostics.ground_truth_source),
        "rows": rows.iter().map(json_benchmark_summary).collect::<Vec<_>>(),
    }));
    Ok(())
}

/// Memory exact search vs HNSW: recall@k is overlap with **runtime exact** top-k (`MemoryBackend`),
/// not HDF5 `neighbors` (see `bench_exact_vs_hnsw_with_corpus` doc on `harness`).
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_exact_vs_hnsw_recall() -> Result<(), DripstoneError> {
    let corpus = datasets::load_integration_corpus().await?;
    let base = format!(
        "hnsw-bench-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let max_pts = dripstone_bench_max_points();
    for case in crate::harness::benchmark_cases_from_corpus(&corpus, max_pts, false)
        .into_iter()
        .filter(|c| matches!(c.filter, FilterSelectivity::None))
    {
        let ns = format!("{base}-{}", case.name);
        let hc = tuned_hnsw_config_for_corpus(&corpus);
        let (exact_s, hnsw_s, recall) =
            bench_exact_vs_hnsw_with_corpus(&case, &ns, &corpus, hc).await?;
        outputln!(
            "\n{}: exact p50={:?} hnsw p50={:?} recall@{}_vs_runtime_exact={:.3} hnsw_qps={:.1} hnsw_build={:?} ser_bytes={:?}",
            case.name,
            exact_s.search.p50,
            hnsw_s.search.p50,
            case.limit,
            recall,
            hnsw_s.qps,
            hnsw_s.build_or_load,
            hnsw_s.index_serialized_bytes
        );
        speed_bench_report::record(json!({
            "test": "test_exact_vs_hnsw_recall",
            "case": case.name,
            "limit": case.limit,
            "exact_p50_ms": speed_bench_report::duration_ms(exact_s.search.p50),
            "hnsw_p50_ms": speed_bench_report::duration_ms(hnsw_s.search.p50),
            "recall_at_k_vs_runtime_exact": recall,
            "hnsw_qps": hnsw_s.qps,
            "hnsw_build_ms": speed_bench_report::duration_ms(hnsw_s.build_or_load),
            "index_serialized_bytes": hnsw_s.index_serialized_bytes,
        }));
        // HNSW graph uses L2; cosine is re-scored post-search, so overlap with exact cosine
        // top-k is approximate (improves with ef_search / normalized vectors).
        assert!(
            recall >= 0.65,
            "expected recall@{} >= 0.65 for smoke case {}, got {recall}",
            case.limit,
            case.name
        );
    }
    Ok(())
}

/// Brute-force vs HNSW at 10K, 50K, and 100K vectors (paste printed lines into docs).
///
/// Enable with `DRIPSTONE_BENCH_HNSW_SCALE=1`. Set `DRIPSTONE_BENCH_MAX_POINTS` to at least `100000`
/// so the third scenario is a full 100K corpus (default cap is 50K and would clamp it).
#[cfg(feature = "memory-backend")]
#[ignore = "requires DRIPSTONE_BENCH_HNSW_SCALE=1; run with --ignored"]
#[tokio::test]
#[serial]
async fn test_hnsw_vs_brute_scale() -> Result<(), DripstoneError> {
    if !dripstone_bench_hnsw_scale() {
        return Err(DripstoneError::storage(
            "DRIPSTONE_BENCH_HNSW_SCALE must be 1 when this ignored test runs".to_string(),
        ));
    }
    let corpus = datasets::load_integration_corpus().await?;
    let base = format!(
        "hnsw-scale-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    outputln!("\nBrute vs HNSW scale benchmark");
    let mut scale_rows = Vec::new();
    for case in crate::harness::active_hnsw_scale_cases(&corpus) {
        let ns = format!("{base}-{}", case.name);
        let hc = tuned_hnsw_config_for_corpus(&corpus);
        let (exact, hnsw, recall) =
            bench_exact_vs_hnsw_with_corpus(&case, &ns, &corpus, hc).await?;
        crate::harness::print_ann_scale(&case, &exact, &hnsw, recall);
        let brute_ms = speed_bench_report::duration_ms(exact.search.p50);
        let ann_ms = speed_bench_report::duration_ms(hnsw.search.p50);
        let speedup = if ann_ms > 1.0e-9 {
            brute_ms / ann_ms
        } else {
            f64::INFINITY
        };
        scale_rows.push(json!({
            "case": case.name,
            "vectors": case.corpus_size,
            "dimension": case.dimension,
            "brute_force_search_p50_ms": brute_ms,
            "hnsw_search_p50_ms": ann_ms,
            "speedup_p50": speedup,
            "recall_at_k_mean_vs_runtime_exact": recall,
            "hnsw_qps_hnsw_search_only": hnsw.qps,
            "hnsw_build_ms": speed_bench_report::duration_ms(hnsw.build_or_load),
        }));
    }
    outputln!();
    speed_bench_report::record(json!({
        "test": "test_hnsw_vs_brute_scale",
        "skipped": false,
        "rows": scale_rows,
    }));
    Ok(())
}

/// Parameter sweep for larger HNSW cases. Opt-in because it explores many combinations.
#[cfg(feature = "memory-backend")]
#[ignore = "requires DRIPSTONE_BENCH_HNSW_SWEEP=1; run with --ignored"]
#[tokio::test]
#[serial]
async fn test_hnsw_parameter_sweep() -> Result<(), DripstoneError> {
    if !dripstone_bench_hnsw_sweep() {
        return Err(DripstoneError::storage(
            "DRIPSTONE_BENCH_HNSW_SWEEP must be 1 when this ignored test runs".to_string(),
        ));
    }
    let corpus = datasets::load_integration_corpus().await?;
    let base = format!(
        "hnsw-sweep-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    outputln!("\nHNSW parameter sweep");
    let mut sweep_out = Vec::new();
    for case in crate::harness::active_hnsw_scale_cases(&corpus) {
        let case_base = format!("{base}-{}", case.name);
        for sweep in hnsw_sweep_configs() {
            let ns = format!("{case_base}-{}", sweep.label.replace(' ', "_"));
            let mut cfg = sweep.config;
            let sweep_m = cfg.m;
            let sweep_ef_search = cfg.ef_search;
            let sweep_rerank_overfetch = cfg.rerank_overfetch;
            cfg.graph_space = corpus.hnsw_graph_space;
            let (_exact, hnsw, recall) =
                bench_exact_vs_hnsw_with_corpus(&case, &ns, &corpus, cfg).await?;
            print_hnsw_sweep_row(&case, &sweep.label, &hnsw, recall);
            sweep_out.push(json!({
                "case": case.name,
                "sweep_label": sweep.label,
                "m": sweep_m,
                "ef_search": sweep_ef_search,
                "rerank_overfetch": sweep_rerank_overfetch,
                "hnsw_p50_ms": speed_bench_report::duration_ms(hnsw.search.p50),
                "hnsw_qps": hnsw.qps,
                "recall_at_k_mean_vs_runtime_exact": recall,
                "hnsw_build_ms": speed_bench_report::duration_ms(hnsw.build_or_load),
            }));
        }
    }
    outputln!();
    speed_bench_report::record(json!({
        "test": "test_hnsw_parameter_sweep",
        "skipped": false,
        "rows": sweep_out,
    }));
    Ok(())
}

/// Same workload: memory upsert/search vs mmap cold open + search.
#[cfg(all(feature = "memory-backend", feature = "mmap-backend"))]
#[tokio::test]
#[serial]
async fn test_memory_vs_mmap_comparison() -> Result<(), DripstoneError> {
    use dripstone::backend::mmap::MmapVectorEncoding;

    let corpus = datasets::load_integration_corpus().await?;
    let ns_base = format!(
        "mem-mmap-variants-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let take_n = corpus
        .base
        .len()
        .min(2000)
        .min(dripstone_bench_max_points())
        .max(1);

    let default_stats =
        bench_memory_vs_mmap(&corpus, take_n, &format!("{ns_base}-default")).await?;
    let int8_stats = bench_memory_vs_mmap_with_encoding(
        &corpus,
        take_n,
        &format!("{ns_base}-int8"),
        "v2_int8_explicit",
        "bench_compare_int8.vf2",
        MmapVectorEncoding::Int8,
    )
    .await?;
    let n = default_stats.n;
    let dimension = default_stats.dimension;
    let rows = vec![default_stats, int8_stats]
        .into_iter()
        .map(|stats| {
            json!({
                "mmap_layout": stats.mmap_layout,
                "memory_upsert_ms": speed_bench_report::duration_ms(stats.memory_upsert),
                "memory_search_p50_ms": speed_bench_report::duration_ms(stats.memory_search_p50),
                "mmap_cold_open_ms": speed_bench_report::duration_ms(stats.mmap_cold_open),
                "mmap_first_search_ms": speed_bench_report::duration_ms(stats.mmap_first_search),
                "mmap_search_p50_ms": speed_bench_report::duration_ms(stats.mmap_search_p50),
                "memory_p50_div_mmap_p50": stats.memory_p50_div_mmap_p50,
            })
        })
        .collect::<Vec<_>>();

    speed_bench_report::record(json!({
        "test": "test_memory_vs_mmap_comparison",
        "n": n,
        "dimension": dimension,
        "rows": rows,
    }));
    Ok(())
}
