//! Latency aggregation and JSON result rows.

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::time::Duration;

use serde::Serialize;

use super::datasets::CorpusDiagnostics;

#[derive(Debug, Clone, Serialize)]
pub struct PercentileSummary {
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub count: usize,
}

pub fn summarize_durations(samples: &mut [Duration]) -> PercentileSummary {
    if samples.is_empty() {
        return PercentileSummary {
            mean_ms: 0.0,
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            count: 0,
        };
    }
    samples.sort();
    let len = samples.len();
    let sum: Duration = samples.iter().sum();
    let mean = sum.as_secs_f64() * 1000.0 / len as f64;
    let idx = |p: f64| ((len as f64 * p) as usize).min(len.saturating_sub(1));
    let to_ms = |d: Duration| d.as_secs_f64() * 1000.0;
    PercentileSummary {
        mean_ms: mean,
        p50_ms: to_ms(samples[idx(0.50)]),
        p95_ms: to_ms(samples[idx(0.95)]),
        p99_ms: to_ms(samples[idx(0.99)]),
        min_ms: to_ms(samples[0]),
        max_ms: to_ms(samples[len - 1]),
        count: len,
    }
}

/// Summary of per-query recall@k samples (f32 in [0,1]).
#[derive(Debug, Clone, Serialize)]
pub struct RecallPercentileSummary {
    pub mean: f32,
    pub min: f32,
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
    pub sample_count: usize,
}

pub fn summarize_recall_samples(samples: &mut [f32]) -> RecallPercentileSummary {
    if samples.is_empty() {
        return RecallPercentileSummary {
            mean: 0.0,
            min: 0.0,
            p50: 0.0,
            p95: 0.0,
            max: 0.0,
            sample_count: 0,
        };
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = samples.len();
    let sum: f32 = samples.iter().sum();
    let idx = |p: f64| ((len as f64 * p) as usize).min(len.saturating_sub(1));
    RecallPercentileSummary {
        mean: sum / len as f32,
        min: samples[0],
        p50: samples[idx(0.50)],
        p95: samples[idx(0.95)],
        max: samples[len - 1],
        sample_count: len,
    }
}

/// HNSW / indexed-backend parameters used for the scenario hot tier (when present).
#[derive(Debug, Clone, Serialize)]
pub struct ScenarioAnnParams {
    pub search_metric: String,
    pub hnsw_graph_space: String,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub hnsw_rerank_overfetch: usize,
    pub indexed_overfetch_factor: usize,
    pub indexed_auto_rebuild_threshold: usize,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkRun {
    pub scenario: String,
    pub dataset_load_ms: f64,
    pub bootstrap_ms: f64,
    pub ingest_ms: f64,
    pub vectors_ingested: usize,
    pub first_search_ms: f64,
    pub warmed_search: PercentileSummary,
    pub qps: f64,
    pub recall_at_k: Option<f32>,
    pub recall_distribution: Option<RecallPercentileSummary>,
    /// Mean Answer@k over queries that have answer labels. `None` when the corpus has none.
    pub answer_at_k: Option<f32>,
    /// Mean Support@k over queries that have support labels. `None` when support is empty.
    pub support_at_k: Option<f32>,
    pub answer_distribution: Option<RecallPercentileSummary>,
    pub support_distribution: Option<RecallPercentileSummary>,
    pub mmap_snapshot_bytes: u64,
    pub corpus_diagnostics: CorpusDiagnostics,
    pub scenario_ann: Option<ScenarioAnnParams>,
    pub untimed_prewarm_search: bool,
}

pub fn results_dir() -> std::path::PathBuf {
    let mut p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target");
    p.push("dripstone");
    p.push("integration");
    p.push("results");
    p
}

pub fn write_run_json(run: &BenchmarkRun) -> Result<(), dripstone::DripstoneError> {
    write_named_json(&run.scenario, run)
}

pub fn write_named_json<T: Serialize>(
    stem: &str,
    value: &T,
) -> Result<(), dripstone::DripstoneError> {
    let dir = results_dir();
    std::fs::create_dir_all(&dir)
        .map_err(|e| dripstone::DripstoneError::storage_with_source("results mkdir", e))?;
    let name = format!(
        "{}_{}.json",
        stem.replace(['/', ' '], "_"),
        time::OffsetDateTime::now_utc().unix_timestamp()
    );
    let path = dir.join(name);
    let json = serde_json::to_vec_pretty(value)
        .map_err(|e| dripstone::DripstoneError::storage(format!("json serialize: {e}")))?;
    std::fs::write(&path, json)
        .map_err(|e| dripstone::DripstoneError::storage_with_source("write results", e))?;
    errorln!("wrote {}", path.display());
    Ok(())
}
