//! Shared helpers for speed benchmark cases.

use std::time::{Duration, Instant};

#[cfg(feature = "memory-backend")]
use crate::harness::PercentileStats;
use dripstone::{DripstoneError, SearchQuery, traits::Backend};
#[cfg(feature = "memory-backend")]
use serde_json::json;

#[cfg(feature = "memory-backend")]
pub(super) fn json_benchmark_summary(s: &crate::harness::BenchmarkSummary) -> serde_json::Value {
    json!({
        "subject": s.subject.as_str(),
        "case": s.case.as_str(),
        "build_or_load_ms": crate::speed_bench_report::duration_ms(s.build_or_load),
        "upsert_ms": s.upsert.map(crate::speed_bench_report::duration_ms),
        "search_p50_ms": crate::speed_bench_report::duration_ms(s.search.p50),
        "search_p99_ms": crate::speed_bench_report::duration_ms(s.search.p99),
        "qps": s.qps,
        "recall_at_k": s.recall_at_k,
        "index_serialized_bytes": s.index_serialized_bytes,
    })
}

#[cfg(feature = "memory-backend")]
pub(super) async fn measure_search_series<B, F>(
    backend: &B,
    query_count: usize,
    mut build_query: F,
) -> Result<(Duration, f64, usize), DripstoneError>
where
    B: Backend + ?Sized,
    F: FnMut(usize) -> SearchQuery,
{
    let mut stats = PercentileStats::new();
    let wall = Instant::now();
    let mut total_results = 0_usize;
    for i in 0..query_count {
        let query = build_query(i);
        let t0 = Instant::now();
        let results = backend.search(&query).await?;
        stats.add(t0.elapsed());
        total_results += results.len();
    }
    let qps = query_count as f64 / wall.elapsed().as_secs_f64().max(1e-9);
    Ok((stats.calculate().p50, qps, total_results))
}
