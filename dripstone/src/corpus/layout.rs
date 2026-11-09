//! Normalized corpus files.
//!
//! ```text
//! $DRIPSTONE_CACHE_DIR/corpora/<name>/<rev>/
//!   manifest.json
//!   vectors.f32.bin      row-major little-endian f32
//!   ids.jsonl            {"id":"doc0"}
//!   meta.jsonl           optional, one JSON object per training row
//!   queries.f32.bin      optional
//!   query_ids.jsonl      optional
//!   neighbors.i32.bin    optional, row-major little-endian i32 (answer + support, -1 pad)
//!   qrels.jsonl          optional graded judgments (class + score); absent for vector catalogs
//!   source/              resumable download (HDF5, en/*.parquet, or CoSQA)
//! ```
//!
//! Cosine / angular corpora drop train and query rows below
//! [`MIN_L2_NORM_FOR_COSINE`] before those files are written. Ids, metadata,
//! and neighbor lists stay aligned with the kept rows. The manifest records
//! how many rows were scanned and how many were dropped.

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use crate::core::Metric;
use crate::errors::DripstoneError;

use super::catalog::CorpusFormat;
use super::relevance::{GradePolicy, QueryJudgments};

/// Manifest file name.
pub const MANIFEST_FILE: &str = "manifest.json";
/// Training matrix, row-major little-endian `f32`.
pub const VECTORS_FILE: &str = "vectors.f32.bin";
/// Training ids, one JSON object per line.
pub const IDS_FILE: &str = "ids.jsonl";
/// Optional per-row JSON metadata. Code rows may use [`CodeRowMeta`] fields.
pub const META_FILE: &str = "meta.jsonl";
/// Query matrix, same layout as [`VECTORS_FILE`].
pub const QUERIES_FILE: &str = "queries.f32.bin";
/// Query ids, one JSON object per line.
pub const QUERY_IDS_FILE: &str = "query_ids.jsonl";
/// Optional neighbor indices, row-major little-endian `i32`.
pub const NEIGHBORS_FILE: &str = "neighbors.i32.bin";
/// Graded judgments, one [`QueryJudgments`] per line. Vector catalogs leave this absent.
///
/// Hard positives still live in [`NEIGHBORS_FILE`]. This file adds class and score.
pub const QRELS_FILE: &str = "qrels.jsonl";
/// Directory for the resumable source download.
pub const SOURCE_DIR: &str = "source";

/// Minimum L2 norm stored for a cosine / angular corpus.
///
/// [`crate::simd::cosine_or_zero`] scores a norm at or below
/// [`crate::simd::ZERO_NORM_EPSILON`] as `0.0`. The integration HDF5 loader
/// uses eight times that epsilon so all-zero and near-zero rows (NYTimes
/// train indices 938, 1857, and 4656 in the first 10_000) are dropped
/// together, before cosine search sees them.
pub const MIN_L2_NORM_FOR_COSINE: f32 = crate::simd::ZERO_NORM_EPSILON * 8.0;

/// How degenerate rows were treated before the matrices were written.
///
/// Cosine (ann-benchmarks angular) drops rows below [`MIN_L2_NORM_FOR_COSINE`].
/// Euclidean, dot product, and Manhattan keep zeros: those metrics are defined
/// at the origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorpusRowPolicy {
    /// Field missing on a manifest written before row policy was recorded.
    ///
    /// Cosine layouts with this value are rebuilt. Layouts whose metric keeps
    /// zeros stay reusable.
    #[default]
    Unspecified,
    /// Store every decoded row, including zeros and non-finite values.
    Keep,
    /// Drop train and query rows whose L2 norm is below [`MIN_L2_NORM_FOR_COSINE`].
    DropBelowMinL2,
}

impl CorpusRowPolicy {
    /// Policy implied by the catalog metric. Angular corpora use [`Metric::Cosine`].
    #[must_use]
    pub const fn for_metric(metric: Metric) -> Self {
        match metric {
            Metric::Cosine => Self::DropBelowMinL2,
            Metric::Euclidean | Metric::DotProduct | Metric::Manhattan => Self::Keep,
        }
    }
}

/// Row caps applied when normalizing and when reading a larger cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusCaps {
    /// Maximum training rows.
    pub train_max: usize,
    /// Maximum query rows.
    pub query_max: usize,
}

/// Per-row metadata for a code corpus. The code normalizer fills this from
/// tree-sitter when a grammar matches, and `meta.jsonl` may add `text`,
/// `source_id`, and `title` beside these fields.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CodeRowMeta {
    /// Programming language tag, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Repository-relative path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    /// Function or type name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    /// First line, 1-based when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_line: Option<u32>,
    /// Last line, inclusive when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_line: Option<u32>,
}

/// JSON manifest written after the matrices.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CorpusManifest {
    /// Catalog or override name.
    pub name: String,
    /// Revision directory segment.
    pub revision: String,
    /// Source format.
    pub format: CorpusFormat,
    /// Vector width.
    pub dimension: usize,
    /// Publisher metric. Search can still override this at query time.
    pub metric: Metric,
    /// Training rows stored in [`VECTORS_FILE`].
    pub train_count: usize,
    /// Query rows stored in [`QUERIES_FILE`].
    pub query_count: usize,
    /// How rows were filtered before the matrices were written.
    ///
    /// Absent on older manifests, which deserialize as [`CorpusRowPolicy::Unspecified`].
    #[serde(default)]
    pub row_policy: CorpusRowPolicy,
    /// Source training rows read before [`Self::row_policy`] was applied.
    ///
    /// Cap reuse compares this with the request. [`Self::train_count`] is the
    /// number of rows stored after drops, which can be smaller.
    #[serde(default)]
    pub scanned_train_rows: usize,
    /// Source query rows read before [`Self::row_policy`] was applied.
    #[serde(default)]
    pub scanned_query_rows: usize,
    /// Training rows omitted by [`Self::row_policy`].
    #[serde(default)]
    pub dropped_train_rows: usize,
    /// Query rows omitted by [`Self::row_policy`].
    #[serde(default)]
    pub dropped_query_rows: usize,
    /// Cap requested when this manifest was written.
    pub train_cap: usize,
    /// Query cap requested when this manifest was written.
    pub query_cap: usize,
    /// Neighbor columns stored per query. Zero when [`Self::has_neighbors`] is false.
    pub neighbors_k: usize,
    /// Neighbor columns available in the source, when known.
    pub neighbors_width: usize,
    /// Download URL for a single-file source.
    pub source_url: Option<String>,
    /// Staged directory or file, when the source was local.
    pub source_path: Option<String>,
    /// `meta.jsonl` is present.
    pub has_meta: bool,
    /// `neighbors.i32.bin` is present.
    pub has_neighbors: bool,
    /// `qrels.jsonl` is present, or an older code layout derived neighbors from qrels.
    /// Vector catalogs write `false`.
    pub has_qrels: bool,
    /// How qrels became answer / support labels.
    ///
    /// Absent on older manifests, which deserialize as [`GradePolicy::BinaryAnswer`].
    #[serde(default)]
    pub grade_policy: GradePolicy,
    /// The source had no further rows beyond what was written.
    pub source_exhausted: bool,
    /// Parquet embedding column, when format is Parquet.
    pub emb_column: Option<String>,
    /// Language shard (`en` for Cohere English).
    pub language: Option<String>,
}

/// A normalized directory plus its manifest.
#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedCorpus {
    /// `<cache>/corpora/<name>/<rev>`.
    pub root: PathBuf,
    /// Parsed `manifest.json`.
    pub manifest: CorpusManifest,
}

impl NormalizedCorpus {
    /// Open `manifest.json` in `root`.
    ///
    /// # Errors
    ///
    /// Returns an error when the manifest is missing or not valid JSON.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self, DripstoneError> {
        let root = root.into();
        let manifest = read_manifest(&root.join(MANIFEST_FILE))?;
        Ok(Self { root, manifest })
    }

    /// Stream training rows in `batch_size` chunks, stopping at `cap`.
    ///
    /// # Errors
    ///
    /// Returns an error when the matrix or id file cannot be read.
    pub fn train_batches(
        &self,
        batch_size: usize,
        cap: usize,
    ) -> Result<VectorBatchIter, DripstoneError> {
        let count = cap.min(self.manifest.train_count);
        VectorBatchIter::open(
            &self.root.join(VECTORS_FILE),
            &self.root.join(IDS_FILE),
            self.manifest.dimension,
            count,
            batch_size.max(1),
        )
    }
}

/// One streamed block of training rows.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorBatch {
    /// Index of the first row in this batch within the capped matrix.
    pub offset: usize,
    /// Ids aligned with `vectors`.
    pub ids: Vec<String>,
    /// Row-major vectors. Each row has the manifest dimension.
    pub vectors: Vec<Vec<f32>>,
}

/// Reads [`VECTORS_FILE`] in bounded chunks.
pub struct VectorBatchIter {
    file: File,
    ids: Vec<String>,
    dimension: usize,
    remaining: usize,
    offset: usize,
    batch_size: usize,
}

impl VectorBatchIter {
    fn open(
        vectors: &Path,
        ids: &Path,
        dimension: usize,
        count: usize,
        batch_size: usize,
    ) -> Result<Self, DripstoneError> {
        if dimension == 0 {
            return Err(DripstoneError::storage("corpus dimension is 0".to_string()));
        }
        let file = File::open(vectors).map_err(|error| {
            DripstoneError::storage_with_source(format!("open {}", vectors.display()), error)
        })?;
        let id_list = read_ids(ids, count)?;
        if id_list.len() < count {
            return Err(DripstoneError::storage(format!(
                "{} has {} ids, need {count}",
                ids.display(),
                id_list.len()
            )));
        }
        Ok(Self {
            file,
            ids: id_list,
            dimension,
            remaining: count,
            offset: 0,
            batch_size,
        })
    }
}

impl Iterator for VectorBatchIter {
    type Item = Result<VectorBatch, DripstoneError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let take = self.batch_size.min(self.remaining);
        let row_bytes = self.dimension.saturating_mul(4);
        let mut raw = vec![0_u8; take.saturating_mul(row_bytes)];
        match self.file.read_exact(&mut raw) {
            Ok(()) => {}
            Err(error) => {
                self.remaining = 0;
                return Some(Err(DripstoneError::storage_with_source(
                    "read vectors.f32.bin",
                    error,
                )));
            }
        }
        let mut vectors = Vec::with_capacity(take);
        for row in raw.chunks(row_bytes) {
            let mut vector = Vec::with_capacity(self.dimension);
            for chunk in row.chunks_exact(4) {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                vector.push(f32::from_le_bytes(bytes));
            }
            vectors.push(vector);
        }
        let ids = self.ids[self.offset..self.offset + take].to_vec();
        let batch = VectorBatch {
            offset: self.offset,
            ids,
            vectors,
        };
        self.offset += take;
        self.remaining -= take;
        Some(Ok(batch))
    }
}

/// Rows held in memory after a capped read. Caps stay in the thousands for CI and soaks.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedRows {
    /// Training vectors.
    pub train: Vec<Vec<f32>>,
    /// Training ids (`doc0`, ...).
    pub train_ids: Vec<String>,
    /// Optional metadata aligned with `train`.
    pub train_meta: Vec<Option<serde_json::Value>>,
    /// Query vectors.
    pub queries: Vec<Vec<f32>>,
    /// Query ids (`q0`, ...).
    pub query_ids: Vec<String>,
    /// Optional neighbor indices into the training matrix.
    pub neighbors: Option<Vec<Vec<i32>>>,
    /// Columns available in the source neighbor matrix.
    pub neighbors_width: usize,
    /// Vector width.
    pub dimension: usize,
    /// No unread source rows remain past this materialization.
    pub source_exhausted: bool,
    /// Download URL, when the source was remote.
    pub source_url: Option<String>,
    /// Local path, when the source was staged.
    pub source_path: Option<String>,
    /// Neighbors were built from code-retrieval qrels.
    pub has_qrels: bool,
    /// Graded judgments aligned with `queries`. `None` for vector catalogs.
    pub judgments: Option<Vec<QueryJudgments>>,
}

pub(super) fn write_layout(
    root: &Path,
    manifest_base: CorpusManifest,
    rows: &MaterializedRows,
) -> Result<CorpusManifest, DripstoneError> {
    fs::create_dir_all(root).map_err(|error| {
        DripstoneError::storage_with_source(format!("create {}", root.display()), error)
    })?;
    write_f32_matrix(&root.join(VECTORS_FILE), &rows.train)?;
    write_ids(&root.join(IDS_FILE), &rows.train_ids)?;
    let has_meta = rows.train_meta.iter().any(Option::is_some);
    if has_meta {
        write_meta(&root.join(META_FILE), &rows.train_meta)?;
    } else if root.join(META_FILE).exists() {
        let _ = fs::remove_file(root.join(META_FILE));
    }
    if rows.queries.is_empty() {
        let _ = fs::remove_file(root.join(QUERIES_FILE));
        let _ = fs::remove_file(root.join(QUERY_IDS_FILE));
    } else {
        write_f32_matrix(&root.join(QUERIES_FILE), &rows.queries)?;
        write_ids(&root.join(QUERY_IDS_FILE), &rows.query_ids)?;
    }
    let has_neighbors = rows
        .neighbors
        .as_ref()
        .is_some_and(|neighbors| !neighbors.is_empty());
    if let Some(neighbors) = rows
        .neighbors
        .as_ref()
        .filter(|neighbors| !neighbors.is_empty())
    {
        write_i32_matrix(&root.join(NEIGHBORS_FILE), neighbors)?;
    } else if root.join(NEIGHBORS_FILE).exists() {
        let _ = fs::remove_file(root.join(NEIGHBORS_FILE));
    }
    if let Some(judgments) = rows
        .judgments
        .as_ref()
        .filter(|judgments| judgments.iter().any(|row| !row.neighbors.is_empty()))
    {
        super::relevance::write_judgments(&root.join(QRELS_FILE), judgments)?;
    } else if root.join(QRELS_FILE).exists() {
        let _ = fs::remove_file(root.join(QRELS_FILE));
    }
    let mut manifest = manifest_base;
    manifest.dimension = rows.dimension;
    manifest.train_count = rows.train.len();
    manifest.query_count = rows.queries.len();
    manifest.neighbors_k = rows
        .neighbors
        .as_ref()
        .and_then(|neighbors| neighbors.first().map(Vec::len))
        .unwrap_or(0);
    manifest.neighbors_width = rows.neighbors_width;
    manifest.has_meta = has_meta;
    manifest.has_neighbors = has_neighbors;
    manifest.has_qrels = rows.has_qrels;
    manifest.source_exhausted = rows.source_exhausted;
    manifest.source_url = rows.source_url.clone();
    manifest.source_path = rows.source_path.clone();
    write_manifest_atomic(&root.join(MANIFEST_FILE), &manifest)?;
    Ok(manifest)
}

pub(super) fn read_f32_matrix(
    path: &Path,
    dimension: usize,
    cap: usize,
) -> Result<Vec<Vec<f32>>, DripstoneError> {
    if dimension == 0 || cap == 0 {
        return Ok(Vec::new());
    }
    let mut file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("open {}", path.display()), error)
    })?;
    let row_bytes = dimension.saturating_mul(4);
    let mut raw = vec![0_u8; cap.saturating_mul(row_bytes)];
    file.read_exact(&mut raw).map_err(|error| {
        DripstoneError::storage_with_source(format!("read {}", path.display()), error)
    })?;
    let mut rows = Vec::with_capacity(cap);
    for row in raw.chunks(row_bytes) {
        let mut vector = Vec::with_capacity(dimension);
        for chunk in row.chunks_exact(4) {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            vector.push(f32::from_le_bytes(bytes));
        }
        rows.push(vector);
    }
    Ok(rows)
}

pub(super) fn read_ids(path: &Path, cap: usize) -> Result<Vec<String>, DripstoneError> {
    let file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("open {}", path.display()), error)
    })?;
    let mut ids = Vec::new();
    for line in BufReader::new(file).lines() {
        if ids.len() >= cap {
            break;
        }
        let line = line.map_err(|error| {
            DripstoneError::storage_with_source(format!("read {}", path.display()), error)
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed: IdLine = serde_json::from_str(trimmed)?;
        ids.push(parsed.id);
    }
    Ok(ids)
}

/// Read `queries` rows of `stored_k` little-endian `i32` values.
pub(super) fn read_neighbors(
    path: &Path,
    queries: usize,
    stored_k: usize,
) -> Result<Vec<Vec<i32>>, DripstoneError> {
    if queries == 0 || stored_k == 0 || !path.is_file() {
        return Ok(Vec::new());
    }
    let mut file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("open {}", path.display()), error)
    })?;
    let bytes_needed = queries.saturating_mul(stored_k).saturating_mul(4);
    let mut raw = vec![0_u8; bytes_needed];
    file.read_exact(&mut raw).map_err(|error| {
        DripstoneError::storage_with_source(format!("read {}", path.display()), error)
    })?;
    let mut rows = Vec::with_capacity(queries);
    let row_bytes = stored_k.saturating_mul(4);
    for row in raw.chunks(row_bytes) {
        let mut neighbors = Vec::with_capacity(stored_k);
        for chunk in row.chunks_exact(4) {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            neighbors.push(i32::from_le_bytes(bytes));
        }
        rows.push(neighbors);
    }
    Ok(rows)
}

pub(super) fn read_manifest(path: &Path) -> Result<CorpusManifest, DripstoneError> {
    let text = fs::read_to_string(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("read {}", path.display()), error)
    })?;
    serde_json::from_str(&text)
        .map_err(|error| DripstoneError::storage(format!("parse {}: {error}", path.display())))
}

fn write_manifest_atomic(path: &Path, manifest: &CorpusManifest) -> Result<(), DripstoneError> {
    let text = serde_json::to_string_pretty(manifest)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, text.as_bytes()).map_err(|error| {
        DripstoneError::storage_with_source(format!("write {}", tmp.display()), error)
    })?;
    fs::rename(&tmp, path).map_err(|error| {
        DripstoneError::storage_with_source(
            format!("rename {} -> {}", tmp.display(), path.display()),
            error,
        )
    })?;
    Ok(())
}

fn write_f32_matrix(path: &Path, rows: &[Vec<f32>]) -> Result<(), DripstoneError> {
    let mut file = File::create(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("create {}", path.display()), error)
    })?;
    let mut buffer = Vec::with_capacity(64 * 1024);
    for row in rows {
        for value in row {
            buffer.extend_from_slice(&value.to_le_bytes());
            if buffer.len() >= 64 * 1024 {
                file.write_all(&buffer).map_err(|error| {
                    DripstoneError::storage_with_source(format!("write {}", path.display()), error)
                })?;
                buffer.clear();
            }
        }
    }
    if !buffer.is_empty() {
        file.write_all(&buffer).map_err(|error| {
            DripstoneError::storage_with_source(format!("write {}", path.display()), error)
        })?;
    }
    Ok(())
}

fn write_i32_matrix(path: &Path, rows: &[Vec<i32>]) -> Result<(), DripstoneError> {
    let mut file = File::create(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("create {}", path.display()), error)
    })?;
    for row in rows {
        for value in row {
            file.write_all(&value.to_le_bytes()).map_err(|error| {
                DripstoneError::storage_with_source(format!("write {}", path.display()), error)
            })?;
        }
    }
    Ok(())
}

fn write_ids(path: &Path, ids: &[String]) -> Result<(), DripstoneError> {
    let mut file = File::create(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("create {}", path.display()), error)
    })?;
    for id in ids {
        let line = serde_json::to_string(&IdLineRef { id })?;
        writeln!(file, "{line}").map_err(|error| {
            DripstoneError::storage_with_source(format!("write {}", path.display()), error)
        })?;
    }
    Ok(())
}

fn write_meta(path: &Path, rows: &[Option<serde_json::Value>]) -> Result<(), DripstoneError> {
    let mut file = File::create(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("create {}", path.display()), error)
    })?;
    for row in rows {
        let value = row.clone().unwrap_or(serde_json::Value::Null);
        writeln!(file, "{value}").map_err(|error| {
            DripstoneError::storage_with_source(format!("write {}", path.display()), error)
        })?;
    }
    Ok(())
}

#[derive(serde::Serialize)]
struct IdLineRef<'a> {
    id: &'a str,
}

#[derive(serde::Deserialize)]
struct IdLine {
    id: String,
}

pub(super) fn read_meta(
    path: &Path,
    cap: usize,
) -> Result<Vec<Option<serde_json::Value>>, DripstoneError> {
    if cap == 0 {
        return Ok(Vec::new());
    }
    let file = File::open(path).map_err(|error| {
        DripstoneError::storage_with_source(format!("open {}", path.display()), error)
    })?;
    let mut rows = Vec::new();
    for line in BufReader::new(file).lines() {
        if rows.len() >= cap {
            break;
        }
        let line = line.map_err(|error| {
            DripstoneError::storage_with_source(format!("read {}", path.display()), error)
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed)?;
        if value.is_null() {
            rows.push(None);
        } else {
            rows.push(Some(value));
        }
    }
    if rows.len() < cap {
        return Err(DripstoneError::storage(format!(
            "{} has {} metadata rows, need {cap}",
            path.display(),
            rows.len()
        )));
    }
    Ok(rows)
}

pub(super) fn train_id(index: usize) -> String {
    format!("doc{index}")
}

pub(super) fn query_id(index: usize) -> String {
    format!("q{index}")
}
