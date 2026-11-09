//! Download a catalog source and write the normalized layout.
//!
//! Reuse an existing revision when its manifest already covers the requested
//! caps (or the source was exhausted) and its row policy matches the metric.
//! A cosine layout written before near-zero rows were dropped is rebuilt.
//! A larger cap re-reads the source. A smaller cap slices the kept rows.

use std::fs;
use std::path::Path;

use crate::core::Metric;
use crate::errors::DripstoneError;
use crate::loader::download::{DownloadOptions, ensure_downloaded};

use super::catalog::{CorpusFormat, CorpusName, CorpusSelector};
use super::code::{download_cosqa, read_code_dir};
use super::env::CorpusRequest;
use super::hdf5::read_hdf5;
use super::layout::{
    CorpusManifest, CorpusRowPolicy, IDS_FILE, META_FILE, NEIGHBORS_FILE, NormalizedCorpus,
    QRELS_FILE, QUERIES_FILE, QUERY_IDS_FILE, SOURCE_DIR, VECTORS_FILE, read_f32_matrix, read_ids,
    read_manifest, read_meta, read_neighbors, write_layout,
};
use super::parquet::{download_cohere_shards, read_parquet_dir, read_parquet_paths};
use super::relevance::{QueryJudgments, judgments_from_hard_neighbors, read_judgments};
use super::sanitize::apply_row_policy;

/// Training and query rows loaded from a normalized directory, already capped.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedCorpus {
    /// Manifest the rows were read from.
    pub manifest: CorpusManifest,
    /// Training vectors, length `min(cap, train_count)`.
    pub base: Vec<Vec<f32>>,
    /// Ids aligned with `base`.
    pub ids: Vec<String>,
    /// Query vectors.
    pub queries: Vec<Vec<f32>>,
    /// Ids aligned with `queries`.
    pub query_ids: Vec<String>,
    /// Optional neighbor lists aligned with `queries`.
    pub neighbors: Option<Vec<Vec<i32>>>,
    /// Graded judgments aligned with `queries`. `None` when the corpus has no qrels.
    pub judgments: Option<Vec<QueryJudgments>>,
    /// Optional metadata aligned with `base`. Empty when the manifest has no `meta.jsonl`.
    pub train_meta: Vec<Option<serde_json::Value>>,
    /// Vector width.
    pub dimension: usize,
}

/// Normalized files plus the capped in-memory rows.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedCorpus {
    /// On-disk layout.
    pub normalized: NormalizedCorpus,
    /// Capped rows.
    pub loaded: LoadedCorpus,
}

/// Download if needed, normalize under the cache, and load the capped rows.
///
/// The blocking work runs on a blocking thread so HDF5 decode and curl do not
/// sit on the async runtime.
///
/// # Errors
///
/// Returns an error when the download, decode, or cache write fails, or when
/// the selector cannot be downloaded or decoded.
pub async fn prepare_corpus(request: CorpusRequest) -> Result<PreparedCorpus, DripstoneError> {
    tokio::task::spawn_blocking(move || prepare_corpus_blocking(&request))
        .await
        .map_err(|error| DripstoneError::storage(format!("corpus task join: {error}")))?
}

/// Synchronous [`prepare_corpus`] for callers that are already off the runtime.
///
/// # Errors
///
/// Returns an error when the download, decode, or cache write fails.
pub fn prepare_corpus_blocking(request: &CorpusRequest) -> Result<PreparedCorpus, DripstoneError> {
    let normalized = ensure_normalized(request)?;
    let loaded = load_capped(&normalized, request)?;
    Ok(PreparedCorpus { normalized, loaded })
}

fn ensure_normalized(request: &CorpusRequest) -> Result<NormalizedCorpus, DripstoneError> {
    let root = request.root();
    if let Some(existing) = try_reuse(&root, request)? {
        return Ok(existing);
    }
    fs::create_dir_all(root.join(SOURCE_DIR)).map_err(|error| {
        DripstoneError::storage_with_source(format!("create {}", root.display()), error)
    })?;
    let mut rows = materialize(request, &root)?;
    if let Some(expected) = request.expected_dimension()
        && rows.dimension != expected
    {
        return Err(DripstoneError::storage(format!(
            "corpus {} has dimension {}, catalog expects {expected}",
            request.layout_name(),
            rows.dimension
        )));
    }
    let report = apply_row_policy(
        &mut rows,
        CorpusRowPolicy::for_metric(request.default_metric()),
    )?;
    let base = CorpusManifest {
        name: request.layout_name(),
        revision: request.revision(),
        format: request.format(),
        dimension: rows.dimension,
        metric: request.default_metric(),
        train_count: 0,
        query_count: 0,
        row_policy: report.policy,
        scanned_train_rows: report.scanned_train_rows,
        scanned_query_rows: report.scanned_query_rows,
        dropped_train_rows: report.dropped_train_rows,
        dropped_query_rows: report.dropped_query_rows,
        train_cap: request.caps.train_max,
        query_cap: request.caps.query_max,
        neighbors_k: 0,
        neighbors_width: rows.neighbors_width,
        source_url: None,
        source_path: None,
        has_meta: false,
        has_neighbors: false,
        has_qrels: false,
        grade_policy: request.grade_policy,
        source_exhausted: rows.source_exhausted,
        emb_column: parquet_column(request),
        language: request.language(),
    };
    let manifest = write_layout(&root, base, &rows)?;
    Ok(NormalizedCorpus { root, manifest })
}

fn materialize(
    request: &CorpusRequest,
    root: &Path,
) -> Result<super::layout::MaterializedRows, DripstoneError> {
    match &request.selector {
        CorpusSelector::Catalog(CorpusName::Code) => {
            let source = root.join(SOURCE_DIR);
            download_cosqa(&source)?;
            let url = super::parquet::huggingface_dataset_url(
                super::catalog::COSQA_REPO,
                super::catalog::COSQA_REVISION,
                super::catalog::COSQA_CORPUS_FILE,
            );
            read_code_dir(
                &source,
                request.caps.train_max,
                request.caps.query_max,
                Some(url),
                request.grade_policy,
            )
        }
        CorpusSelector::CodeDir { path, .. } => read_code_dir(
            path,
            request.caps.train_max,
            request.caps.query_max,
            None,
            request.grade_policy,
        ),
        CorpusSelector::Catalog(name @ CorpusName::Nytimes256Angular)
        | CorpusSelector::Catalog(name @ CorpusName::Sift128Euclidean)
        | CorpusSelector::Catalog(name @ CorpusName::Gist960Euclidean) => {
            let url = name.download_url().ok_or_else(|| {
                DripstoneError::storage(format!("{} has no download url", name.as_str()))
            })?;
            let file_name = source_file_name(url)?;
            let dest = root.join(SOURCE_DIR).join(file_name);
            let options = DownloadOptions::default().with_expected_bytes(name.size_hint_bytes());
            ensure_downloaded(url, &dest, &options)?;
            read_hdf5(
                &dest,
                request.caps.train_max,
                request.caps.query_max,
                request.neighbors_k,
                Some(url.to_string()),
            )
        }
        CorpusSelector::Catalog(CorpusName::CohereWikiEn) => {
            let source = root.join(SOURCE_DIR);
            let paths = download_cohere_shards(
                &source,
                request.caps.train_max,
                request.caps.query_max,
                request.shard_max,
            )?;
            read_parquet_paths(
                &paths,
                &request.emb_column,
                request.caps.train_max,
                request.caps.query_max,
                None,
            )
        }
        CorpusSelector::Hdf5Url { url, .. } => {
            let file_name = source_file_name(url)?;
            let dest = root.join(SOURCE_DIR).join(file_name);
            ensure_downloaded(url, &dest, &DownloadOptions::default())?;
            read_hdf5(
                &dest,
                request.caps.train_max,
                request.caps.query_max,
                request.neighbors_k,
                Some(url.clone()),
            )
        }
        CorpusSelector::ParquetDir { path, .. } => read_parquet_dir(
            path,
            request.parquet_recursive,
            &request.emb_column,
            request.caps.train_max,
            request.caps.query_max,
        ),
    }
}

fn load_capped(
    normalized: &NormalizedCorpus,
    request: &CorpusRequest,
) -> Result<LoadedCorpus, DripstoneError> {
    let manifest = &normalized.manifest;
    let train_count = request.caps.train_max.min(manifest.train_count);
    let query_count = request.caps.query_max.min(manifest.query_count);
    let base = read_f32_matrix(
        &normalized.root.join(VECTORS_FILE),
        manifest.dimension,
        train_count,
    )?;
    let ids = read_ids(&normalized.root.join(IDS_FILE), train_count)?;
    let queries = if query_count == 0 {
        Vec::new()
    } else {
        read_f32_matrix(
            &normalized.root.join(QUERIES_FILE),
            manifest.dimension,
            query_count,
        )?
    };
    let query_ids = if query_count == 0 {
        Vec::new()
    } else {
        read_ids(&normalized.root.join(QUERY_IDS_FILE), query_count)?
    };
    let train_meta = if manifest.has_meta {
        read_meta(&normalized.root.join(META_FILE), train_count)?
    } else {
        Vec::new()
    };
    let neighbors = if manifest.has_neighbors {
        let stored = read_neighbors(
            &normalized.root.join(NEIGHBORS_FILE),
            query_count,
            manifest.neighbors_k,
        )?;
        let k = request.neighbors_k.min(manifest.neighbors_k);
        Some(truncate_neighbors(stored, k))
    } else {
        None
    };
    let judgments = load_judgments(normalized, query_count, &query_ids, neighbors.as_ref())?;
    if ids.len() != base.len() {
        return Err(DripstoneError::storage(format!(
            "corpus {} has {} ids for {} vectors",
            manifest.name,
            ids.len(),
            base.len()
        )));
    }
    Ok(LoadedCorpus {
        manifest: manifest.clone(),
        dimension: manifest.dimension,
        base,
        ids,
        queries,
        query_ids,
        neighbors,
        judgments,
        train_meta,
    })
}

fn load_judgments(
    normalized: &NormalizedCorpus,
    query_count: usize,
    query_ids: &[String],
    neighbors: Option<&Vec<Vec<i32>>>,
) -> Result<Option<Vec<QueryJudgments>>, DripstoneError> {
    if !normalized.manifest.has_qrels || query_count == 0 {
        return Ok(None);
    }
    let path = normalized.root.join(QRELS_FILE);
    if path.is_file() {
        return Ok(Some(read_judgments(&path, query_count)?));
    }
    match neighbors {
        Some(rows) => Ok(Some(judgments_from_hard_neighbors(query_ids, rows))),
        None => Ok(None),
    }
}

fn truncate_neighbors(rows: Vec<Vec<i32>>, k: usize) -> Vec<Vec<i32>> {
    rows.into_iter()
        .map(|row| row.into_iter().take(k).collect())
        .collect()
}

fn try_reuse(
    root: &Path,
    request: &CorpusRequest,
) -> Result<Option<NormalizedCorpus>, DripstoneError> {
    let manifest_path = root.join(super::layout::MANIFEST_FILE);
    if !manifest_path.is_file() {
        return Ok(None);
    }
    let manifest = read_manifest(&manifest_path)?;
    if !layout_reusable(&manifest, request) {
        return Ok(None);
    }
    if !root.join(VECTORS_FILE).is_file() || !root.join(IDS_FILE).is_file() {
        return Ok(None);
    }
    Ok(Some(NormalizedCorpus {
        root: root.to_path_buf(),
        manifest,
    }))
}

fn same_source(manifest: &CorpusManifest, request: &CorpusRequest) -> bool {
    manifest.name == request.layout_name()
        && manifest.revision == request.revision()
        && manifest.format == request.format()
        && manifest.metric == request.default_metric()
        && manifest.grade_policy == request.grade_policy
        && manifest.emb_column == parquet_column(request)
}

pub(super) fn layout_reusable(manifest: &CorpusManifest, request: &CorpusRequest) -> bool {
    same_source(manifest, request) && caps_covered(manifest, request)
}

fn caps_covered(manifest: &CorpusManifest, request: &CorpusRequest) -> bool {
    if !policy_reusable(manifest, request) {
        return false;
    }
    // Coverage is the pre-drop scan. `train_count` can be smaller after cosine
    // drops, and a same-cap rerun must still hit this layout. A smaller cap
    // slices the kept rows. Rebuilding it in this directory would replace a
    // larger scan, so a matching policy reuses the prefix instead.
    let train_scanned = scanned_rows(
        manifest.scanned_train_rows,
        manifest.train_count,
        manifest.row_policy,
    );
    let query_scanned = scanned_rows(
        manifest.scanned_query_rows,
        manifest.query_count,
        manifest.row_policy,
    );
    let train_ok = train_scanned >= request.caps.train_max || manifest.source_exhausted;
    let query_ok = query_scanned >= request.caps.query_max || manifest.source_exhausted;
    let neighbors_ok = if manifest.has_neighbors {
        manifest.neighbors_k >= request.neighbors_k
            || manifest.neighbors_k >= manifest.neighbors_width
    } else {
        true
    };
    train_ok && query_ok && neighbors_ok
}

fn policy_reusable(manifest: &CorpusManifest, request: &CorpusRequest) -> bool {
    let required = CorpusRowPolicy::for_metric(request.default_metric());
    match manifest.row_policy {
        CorpusRowPolicy::Unspecified => required == CorpusRowPolicy::Keep,
        CorpusRowPolicy::Keep => required == CorpusRowPolicy::Keep,
        CorpusRowPolicy::DropBelowMinL2 => required == CorpusRowPolicy::DropBelowMinL2,
    }
}

fn scanned_rows(recorded: usize, stored_fallback: usize, policy: CorpusRowPolicy) -> usize {
    match policy {
        CorpusRowPolicy::Unspecified => stored_fallback,
        CorpusRowPolicy::Keep | CorpusRowPolicy::DropBelowMinL2 => recorded,
    }
}

fn parquet_column(request: &CorpusRequest) -> Option<String> {
    match request.format() {
        CorpusFormat::Parquet => Some(request.emb_column.clone()),
        CorpusFormat::Hdf5 | CorpusFormat::Code => None,
    }
}

fn source_file_name(url: &str) -> Result<String, DripstoneError> {
    let tail = url.rsplit('/').next().unwrap_or("");
    let tail = tail.split('?').next().unwrap_or("");
    if tail.is_empty() || tail == "." || tail == ".." || tail.contains('\0') || tail.contains('/') {
        return Err(DripstoneError::storage(format!(
            "refusing corpus download file name from `{url}`"
        )));
    }
    Ok(tail.to_string())
}

/// Metric name used in live-scene headers.
#[must_use]
pub fn metric_label(metric: Metric) -> &'static str {
    match metric {
        Metric::Cosine => "cosine",
        Metric::Euclidean => "euclidean",
        Metric::DotProduct => "dot",
        Metric::Manhattan => "manhattan",
    }
}
