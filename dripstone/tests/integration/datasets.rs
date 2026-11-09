//! Load integration corpora through `dripstone::corpus`.
//!
//! HDF5 downloads use the resumable loader and a normalized cache under
//! `$DRIPSTONE_CACHE_DIR/corpora/<name>/<rev>/`. Parquet shards use the same
//! path when `DRIPSTONE_INT_PARQUET_DIR` (or `DRIPSTONE_CORPUS_PARQUET_DIR`) is set.
//! Caps default to the integration profile (4_000 train rows, 64 queries) unless
//! `DRIPSTONE_CORPUS_TRAIN_MAX` / `DRIPSTONE_INT_TRAIN_MAX` override them.
//!
//! By default (`DRIPSTONE_INT_GROUND_TRUTH_SOURCE` unset or `subset_exact`), recall ground truth is
//! recomputed with exact search on the loaded train prefix so neighbor indices match ingested
//! `doc{i}` rows. HDF5 `neighbors` refer to the full corpus; using them under a truncated
//! train cap misaligns recall unless you opt into `hdf5` / `hdf5_neighbors`.

use dripstone::corpus::{
    self, CorpusFormat, CorpusProfile, MIN_L2_NORM_FOR_COSINE, PreparedCorpus, QueryJudgments,
    remap_graded, row_ok_for_cosine,
};
use dripstone::index::HnswGraphSpace;
use dripstone::{DripstoneError, Metric};
use serde::Serialize;

use super::ground_truth::brute_force_neighbors;

/// Default ann-benchmarks URL. Kept for callers that still name the integration default.
pub const DEFAULT_DATASET_URL: &str = corpus::NYTIMES_URL;

/// Default catalog name.
pub const DEFAULT_DATASET_NAME: &str = "nytimes-256-angular";

/// L2 norm statistics over a set of vectors (computed after any sanitization).
#[derive(Debug, Clone, Serialize)]
pub struct NormStats {
    pub min: f32,
    pub p50: f32,
    pub p95: f32,
    pub max: f32,
    /// Count of rows with L2 norm below `MIN_L2_NORM_FOR_COSINE` (should be 0 after sanitize).
    pub near_zero_count: usize,
    /// Count of rows with L2 norm within 1e-3 of 1.0 (likely pre-normalized).
    pub unit_like_count: usize,
    pub count: usize,
}

/// How integration recall ground truth was derived for the loaded corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GroundTruthSource {
    /// Exact top-k over the loaded `base` rows (indices match ingested `doc{i}` IDs).
    SubsetExact,
    /// HDF5 `neighbors` from ann-benchmarks (full-corpus indices; misleading if `train_max` truncates).
    Hdf5Neighbors,
}

/// Rows dropped during cosine/dot sanitization and norm summaries for debugging ANN runs.
///
/// Drop counts include rows the shared corpus layout already omitted for cosine.
#[derive(Debug, Clone, Serialize)]
pub struct CorpusDiagnostics {
    pub dropped_train_rows: usize,
    pub dropped_query_rows: usize,
    pub train_norm: NormStats,
    pub query_norm: NormStats,
    /// Effective source for `DatasetCorpus::ground_truth` (may fall back to `subset_exact` if HDF5 neighbors are missing).
    pub ground_truth_source: GroundTruthSource,
    /// Whether the HDF5 file contained a `neighbors` dataset (false for Parquet corpora).
    pub hdf5_neighbors_present: bool,
}

pub struct DatasetCorpus {
    pub name: String,
    pub base: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    /// Neighbor indices per query into `base` row order (`doc{i}`). See `CorpusDiagnostics::ground_truth_source`.
    pub ground_truth: Option<Vec<Vec<i32>>>,
    /// Graded qrels aligned with `queries`. `None` for hard-neighbor corpora.
    pub judgments: Option<Vec<QueryJudgments>>,
    pub dimension: usize,
    pub search_metric: Metric,
    pub hnsw_graph_space: HnswGraphSpace,
    pub diagnostics: CorpusDiagnostics,
}

fn ground_truth_source_from_env() -> GroundTruthSource {
    match std::env::var("DRIPSTONE_INT_GROUND_TRUTH_SOURCE") {
        Ok(s) => {
            let t = s.trim();
            if t.eq_ignore_ascii_case("hdf5") || t.eq_ignore_ascii_case("hdf5_neighbors") {
                GroundTruthSource::Hdf5Neighbors
            } else {
                GroundTruthSource::SubsetExact
            }
        }
        Err(_) => GroundTruthSource::SubsetExact,
    }
}

fn display_name(fallback: &str) -> String {
    std::env::var(corpus::ENV_INT_DATASET_NAME)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

fn env_metric_token() -> Option<String> {
    std::env::var(corpus::ENV_INT_SEARCH_METRIC)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
}

/// Historical HDF5 harness mapping: `dot` was scored as cosine.
fn metric_hdf5(token: Option<&str>, fallback: Metric) -> (Metric, HnswGraphSpace) {
    match token {
        Some("euclidean" | "l2") => (Metric::Euclidean, HnswGraphSpace::RawL2),
        Some("cosine" | "angular" | "dot" | "dotproduct") => {
            (Metric::Cosine, HnswGraphSpace::CosineCompatible)
        }
        _ => metric_space(fallback),
    }
}

fn metric_parquet(token: Option<&str>, fallback: Metric) -> (Metric, HnswGraphSpace) {
    match token {
        Some("euclidean" | "l2") => (Metric::Euclidean, HnswGraphSpace::RawL2),
        Some("cosine" | "angular") => (Metric::Cosine, HnswGraphSpace::CosineCompatible),
        Some("dot" | "dotproduct") => (Metric::DotProduct, HnswGraphSpace::RawL2),
        Some("manhattan" | "l1") => (Metric::Manhattan, HnswGraphSpace::RawL2),
        _ => metric_space(fallback),
    }
}

fn metric_space(metric: Metric) -> (Metric, HnswGraphSpace) {
    match metric {
        Metric::Cosine => (Metric::Cosine, HnswGraphSpace::CosineCompatible),
        Metric::Euclidean => (Metric::Euclidean, HnswGraphSpace::RawL2),
        Metric::DotProduct => (Metric::DotProduct, HnswGraphSpace::RawL2),
        Metric::Manhattan => (Metric::Manhattan, HnswGraphSpace::RawL2),
    }
}

fn l2_norm_row(v: &[f32]) -> f32 {
    let sq: f32 = v.iter().map(|x| x * x).sum();
    if !sq.is_finite() || sq <= 0.0 {
        return 0.0;
    }
    sq.sqrt()
}

pub fn norm_stats(rows: &[Vec<f32>]) -> NormStats {
    if rows.is_empty() {
        return NormStats {
            min: 0.0,
            p50: 0.0,
            p95: 0.0,
            max: 0.0,
            near_zero_count: 0,
            unit_like_count: 0,
            count: 0,
        };
    }
    let mut norms: Vec<f32> = rows.iter().map(|r| l2_norm_row(r)).collect();
    norms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let len = norms.len();
    let idx = |p: f64| ((len as f64 * p) as usize).min(len.saturating_sub(1));
    let near_zero = norms
        .iter()
        .filter(|&&n| n < MIN_L2_NORM_FOR_COSINE)
        .count();
    let unit_like = norms.iter().filter(|&&n| (n - 1.0).abs() < 1e-3).count();
    NormStats {
        min: norms[0],
        p50: norms[idx(0.50)],
        p95: norms[idx(0.95)],
        max: norms[len - 1],
        near_zero_count: near_zero,
        unit_like_count: unit_like,
        count: len,
    }
}

/// Drop degenerate training rows and queries; remap `neighbors` indices to the filtered training order.
pub fn sanitize_corpus_for_angle_metrics(
    base: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    ground_truth: Option<Vec<Vec<i32>>>,
    judgments: Option<Vec<QueryJudgments>>,
) -> Result<
    (
        Vec<Vec<f32>>,
        Vec<Vec<f32>>,
        Option<Vec<Vec<i32>>>,
        Option<Vec<QueryJudgments>>,
        usize,
        usize,
    ),
    DripstoneError,
> {
    let n_train = base.len();
    let mut new_base: Vec<Vec<f32>> = Vec::with_capacity(base.len());
    let mut old_to_new: Vec<Option<usize>> = vec![None; n_train];
    let mut dropped_train = 0_usize;

    for (old_i, row) in base.into_iter().enumerate() {
        if row_ok_for_cosine(&row) {
            old_to_new[old_i] = Some(new_base.len());
            new_base.push(row);
        } else {
            dropped_train += 1;
        }
    }

    if new_base.is_empty() {
        return Err(DripstoneError::storage(
            "integration corpus: all training vectors have near-zero L2 norm (cannot run cosine search)"
                .to_string(),
        ));
    }

    if let Some(rows) = &judgments
        && rows.len() != queries.len()
    {
        return Err(DripstoneError::storage(format!(
            "integration corpus: {} judgment rows for {} queries",
            rows.len(),
            queries.len()
        )));
    }

    let mut new_queries: Vec<Vec<f32>> = Vec::with_capacity(queries.len());
    let mut new_gt: Option<Vec<Vec<i32>>> = ground_truth.as_ref().map(|_| Vec::new());
    let mut new_judgments: Option<Vec<QueryJudgments>> = judgments.as_ref().map(|_| Vec::new());
    let mut dropped_query = 0_usize;

    for (qi, qrow) in queries.into_iter().enumerate() {
        if !row_ok_for_cosine(&qrow) {
            dropped_query += 1;
            continue;
        }

        new_queries.push(qrow);

        if let (Some(gt_in), Some(gt_out)) = (&ground_truth, &mut new_gt) {
            let neighbors = gt_in.get(qi).ok_or_else(|| {
                DripstoneError::storage(format!(
                    "integration corpus: neighbors row missing for query index {qi}"
                ))
            })?;
            let mapped: Vec<i32> = neighbors
                .iter()
                .filter_map(|&old_id| {
                    if old_id < 0 {
                        return None;
                    }
                    let u = old_id as usize;
                    if u >= old_to_new.len() {
                        return None;
                    }
                    old_to_new[u].and_then(|n| i32::try_from(n).ok())
                })
                .collect();
            gt_out.push(mapped);
        }
        if let (Some(rows), Some(out)) = (&judgments, &mut new_judgments) {
            let row = rows.get(qi).ok_or_else(|| {
                DripstoneError::storage(format!(
                    "integration corpus: judgment row missing for query index {qi}"
                ))
            })?;
            out.push(QueryJudgments {
                query_id: row.query_id.clone(),
                neighbors: remap_graded(&row.neighbors, &old_to_new),
            });
        }
    }

    if new_queries.is_empty() {
        return Err(DripstoneError::storage(
            "integration corpus: all query vectors have near-zero L2 norm (cannot run cosine search)"
                .to_string(),
        ));
    }

    Ok((
        new_base,
        new_queries,
        new_gt,
        new_judgments,
        dropped_train,
        dropped_query,
    ))
}

/// Load the integration corpus from the shared cache.
///
/// Env: `DRIPSTONE_CORPUS` / `DRIPSTONE_INT_DATASET_URL`, `DRIPSTONE_CORPUS_TRAIN_MAX` /
/// `DRIPSTONE_INT_TRAIN_MAX` (default 4_000), `DRIPSTONE_CORPUS_QUERY_MAX` /
/// `DRIPSTONE_INT_QUERY_MAX` (default 64), `DRIPSTONE_CORPUS_NEIGHBORS_K` /
/// `DRIPSTONE_INT_NEIGHBORS_K`,
/// `DRIPSTONE_INT_SEARCH_METRIC`, `DRIPSTONE_INT_PARQUET_DIR`,
/// `DRIPSTONE_INT_GROUND_TRUTH_SOURCE`.
pub async fn load_integration_corpus() -> Result<DatasetCorpus, DripstoneError> {
    let request = corpus::request_from_env(CorpusProfile::Integration)?;
    let prepared = corpus::prepare_corpus(request).await?;
    dataset_from_prepared(prepared)
}

pub(super) fn dataset_from_prepared(
    prepared: PreparedCorpus,
) -> Result<DatasetCorpus, DripstoneError> {
    let token = env_metric_token();
    let (search_metric, hnsw_graph_space) = match prepared.loaded.manifest.format {
        CorpusFormat::Hdf5 => metric_hdf5(token.as_deref(), prepared.loaded.manifest.metric),
        CorpusFormat::Parquet | CorpusFormat::Code => {
            metric_parquet(token.as_deref(), prepared.loaded.manifest.metric)
        }
    };
    let neighbors_k = prepared.loaded.manifest.neighbors_k;
    let name = display_name(&prepared.loaded.manifest.name);
    let hdf5_neighbors_present = prepared.loaded.manifest.has_neighbors;
    let base = prepared.loaded.base;
    let queries = prepared.loaded.queries;
    let hdf5_ground_truth = prepared.loaded.neighbors;
    let loaded_judgments = prepared.loaded.judgments;
    let dim = prepared.loaded.dimension;
    if base.is_empty() {
        return Err(DripstoneError::storage("empty train set".to_string()));
    }

    let gt_requested = ground_truth_source_from_env();
    let sanitize_neighbors_input = if matches!(gt_requested, GroundTruthSource::Hdf5Neighbors)
        && matches!(search_metric, Metric::Cosine | Metric::DotProduct)
    {
        hdf5_ground_truth.clone()
    } else {
        None
    };

    let (base, queries, gt_after_sanitize, judgments, dropped_train, dropped_query) =
        if matches!(search_metric, Metric::Cosine | Metric::DotProduct) {
            let (b, q, gt, judged, dt, dq) = sanitize_corpus_for_angle_metrics(
                base,
                queries,
                sanitize_neighbors_input,
                loaded_judgments,
            )?;
            (b, q, gt, judged, dt, dq)
        } else {
            (base, queries, None, loaded_judgments, 0_usize, 0_usize)
        };

    let (ground_truth, ground_truth_source) = match gt_requested {
        GroundTruthSource::SubsetExact => {
            let gt = brute_force_neighbors(&base, &queries, neighbors_k, search_metric)?;
            (Some(gt), GroundTruthSource::SubsetExact)
        }
        GroundTruthSource::Hdf5Neighbors => {
            if matches!(search_metric, Metric::Cosine | Metric::DotProduct) {
                if let Some(gt) = gt_after_sanitize {
                    (Some(gt), GroundTruthSource::Hdf5Neighbors)
                } else {
                    let gt = brute_force_neighbors(&base, &queries, neighbors_k, search_metric)?;
                    (Some(gt), GroundTruthSource::SubsetExact)
                }
            } else if let Some(gt) = hdf5_ground_truth {
                (Some(gt), GroundTruthSource::Hdf5Neighbors)
            } else {
                let gt = brute_force_neighbors(&base, &queries, neighbors_k, search_metric)?;
                (Some(gt), GroundTruthSource::SubsetExact)
            }
        }
    };

    let diagnostics = CorpusDiagnostics {
        dropped_train_rows: dropped_train
            .saturating_add(prepared.loaded.manifest.dropped_train_rows),
        dropped_query_rows: dropped_query
            .saturating_add(prepared.loaded.manifest.dropped_query_rows),
        train_norm: norm_stats(&base),
        query_norm: norm_stats(&queries),
        ground_truth_source,
        hdf5_neighbors_present,
    };

    Ok(DatasetCorpus {
        name,
        base,
        queries,
        ground_truth,
        judgments,
        dimension: dim,
        search_metric,
        hnsw_graph_space,
        diagnostics,
    })
}

#[test]
fn default_integration_dataset_is_nytimes() {
    assert_eq!(DEFAULT_DATASET_NAME, "nytimes-256-angular");
    assert_eq!(DEFAULT_DATASET_URL, corpus::NYTIMES_URL);
    assert!(DEFAULT_DATASET_URL.contains("nytimes-256-angular"));
}
