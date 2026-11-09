//! Environment surface for corpus caps, aliases, and concurrency.
//!
//! `DRIPSTONE_CORPUS_*` is the product name. `DRIPSTONE_INT_*` is accepted so
//! existing integration jobs keep working. When both are set, `DRIPSTONE_CORPUS_*`
//! wins.

use std::path::PathBuf;

use crate::core::Metric;
use crate::errors::DripstoneError;
use crate::loader::disk::cache_root;

use super::catalog::{CorpusFormat, CorpusName, CorpusSelector};
use super::relevance::{GradePolicy, RetrievalTechnique, parse_grade_policy, parse_technique};

/// Select a named corpus (`nytimes-256-angular`, `sift-128-euclidean`, `gist-960-euclidean`, `cohere-wiki-en`, `code`).
pub const ENV_CORPUS: &str = "DRIPSTONE_CORPUS";
/// Override the HDF5 download URL.
pub const ENV_CORPUS_URL: &str = "DRIPSTONE_CORPUS_URL";
/// Cap training rows written into the normalized layout. Soak default is 10_000.
pub const ENV_CORPUS_TRAIN_MAX: &str = "DRIPSTONE_CORPUS_TRAIN_MAX";
/// Cap query rows. Soak default is 128.
pub const ENV_CORPUS_QUERY_MAX: &str = "DRIPSTONE_CORPUS_QUERY_MAX";
/// Upsert batch size for callers that stream into ColdForge.
pub const ENV_CORPUS_BATCH: &str = "DRIPSTONE_CORPUS_BATCH";
/// Concurrent ColdForge body PUTs. Default is 32.
pub const ENV_CORPUS_PUT_CONCURRENCY: &str = "DRIPSTONE_CORPUS_PUT_CONCURRENCY";
/// Staged Parquet file or directory. For Cohere this should be the `en/` tree or a shard inside it.
pub const ENV_CORPUS_PARQUET_DIR: &str = "DRIPSTONE_CORPUS_PARQUET_DIR";
/// Recursively collect `.parquet` files. Alias: `DRIPSTONE_INT_PARQUET_RECURSIVE`.
pub const ENV_CORPUS_PARQUET_RECURSIVE: &str = "DRIPSTONE_CORPUS_PARQUET_RECURSIVE";
/// Embedding column name. Default `emb`.
pub const ENV_CORPUS_EMB_COLUMN: &str = "DRIPSTONE_CORPUS_EMB_COLUMN";
/// How many Cohere `en/NNNN.parquet` shards to download before stopping.
pub const ENV_CORPUS_SHARD_MAX: &str = "DRIPSTONE_CORPUS_SHARD_MAX";
/// Neighbor columns to keep from HDF5.
pub const ENV_CORPUS_NEIGHBORS_K: &str = "DRIPSTONE_CORPUS_NEIGHBORS_K";
/// Metric override (`cosine`, `euclidean`, `dot`, `manhattan`).
pub const ENV_CORPUS_METRIC: &str = "DRIPSTONE_CORPUS_METRIC";
/// Staged CoIR CoSQA directory. Used when [`ENV_CORPUS`] is `code` or unset.
pub const ENV_CORPUS_CODE_DIR: &str = "DRIPSTONE_CORPUS_CODE_DIR";
/// How qrels become answer / support labels (`binary`, `explicit`, `threshold[:N]`).
pub const ENV_CORPUS_GRADE_POLICY: &str = "DRIPSTONE_CORPUS_GRADE_POLICY";
/// Retrieval technique plugin (`hard_neighbors`, `graded_context`, or a stub).
pub const ENV_CORPUS_TECHNIQUE: &str = "DRIPSTONE_CORPUS_TECHNIQUE";

/// Integration alias of [`ENV_CORPUS_URL`].
pub const ENV_INT_DATASET_URL: &str = "DRIPSTONE_INT_DATASET_URL";
/// Integration label override.
pub const ENV_INT_DATASET_NAME: &str = "DRIPSTONE_INT_DATASET_NAME";
/// Integration alias of [`ENV_CORPUS_TRAIN_MAX`]. CI leaves this unset so the default stays 4_000.
pub const ENV_INT_TRAIN_MAX: &str = "DRIPSTONE_INT_TRAIN_MAX";
/// Integration alias of [`ENV_CORPUS_QUERY_MAX`].
pub const ENV_INT_QUERY_MAX: &str = "DRIPSTONE_INT_QUERY_MAX";
/// Integration alias for neighbor columns.
pub const ENV_INT_NEIGHBORS_K: &str = "DRIPSTONE_INT_NEIGHBORS_K";
/// Integration metric override.
pub const ENV_INT_SEARCH_METRIC: &str = "DRIPSTONE_INT_SEARCH_METRIC";
/// Integration alias of [`ENV_CORPUS_PARQUET_DIR`].
pub const ENV_INT_PARQUET_DIR: &str = "DRIPSTONE_INT_PARQUET_DIR";
/// Integration alias for recursive Parquet scans.
pub const ENV_INT_PARQUET_RECURSIVE: &str = "DRIPSTONE_INT_PARQUET_RECURSIVE";
/// Integration alias for the embedding column.
pub const ENV_INT_PARQUET_EMB_COLUMN: &str = "DRIPSTONE_INT_PARQUET_EMB_COLUMN";

/// Default embedding column in Cohere Parquet shards.
pub const EMB_COLUMN: &str = "emb";

/// Integration / CI training cap. Kept small so default jobs do not pull a full corpus into RAM.
pub const INTEGRATION_TRAIN_MAX: usize = 4_000;
/// Integration / CI query cap.
pub const INTEGRATION_QUERY_MAX: usize = 64;
/// Live soak training cap when `DRIPSTONE_CORPUS` is set and no cap env is present.
pub const SOAK_TRAIN_MAX: usize = 10_000;
/// Live soak query cap.
pub const SOAK_QUERY_MAX: usize = 128;
/// Default neighbor columns read from HDF5.
pub const DEFAULT_NEIGHBORS_K: usize = 100;
/// Default upsert batch for the live scene.
pub const DEFAULT_UPSERT_BATCH: usize = 256;
/// Cohere shards downloaded per ensure when the caller did not stage `en/`.
pub const DEFAULT_SHARD_MAX: usize = 4;

/// Which unset caps to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusProfile {
    /// `DRIPSTONE_INT_*` defaults: 4_000 train rows, 64 queries.
    Integration,
    /// Live soak defaults: 10_000 train rows, 128 queries.
    Soak,
}

/// Fully resolved inputs for [`super::prepare_corpus`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusRequest {
    /// Where the source bytes come from.
    pub selector: CorpusSelector,
    /// Row caps applied while normalizing and while reading.
    pub caps: super::layout::CorpusCaps,
    /// Neighbor columns to retain.
    pub neighbors_k: usize,
    /// Parent of `<name>/<rev>/`. Defaults to `$DRIPSTONE_CACHE_DIR/corpora`.
    pub cache_dir: PathBuf,
    /// Parquet embedding column.
    pub emb_column: String,
    /// Walk subdirectories for `.parquet` files.
    pub parquet_recursive: bool,
    /// Maximum Cohere shards to download.
    pub shard_max: usize,
    /// How source qrels become answer / support labels.
    pub grade_policy: GradePolicy,
}

impl CorpusRequest {
    /// Directory name (`corpora/<name>`).
    #[must_use]
    pub fn layout_name(&self) -> String {
        match &self.selector {
            CorpusSelector::Catalog(name) => name.as_str().to_string(),
            CorpusSelector::Hdf5Url { name, .. }
            | CorpusSelector::ParquetDir { name, .. }
            | CorpusSelector::CodeDir { name, .. } => name.clone(),
        }
    }

    /// Revision directory under the layout name.
    #[must_use]
    pub fn revision(&self) -> String {
        match &self.selector {
            CorpusSelector::Catalog(name) => name.revision().to_string(),
            CorpusSelector::Hdf5Url { url, .. } => format!("url-{}", short_hash(url)),
            CorpusSelector::ParquetDir { path, .. } | CorpusSelector::CodeDir { path, .. } => {
                format!("staged-{}", short_hash(&path.display().to_string()))
            }
        }
    }

    /// Normalized root: `<cache>/corpora` is already `cache_dir`.
    #[must_use]
    pub fn root(&self) -> PathBuf {
        self.cache_dir
            .join(self.layout_name())
            .join(self.revision())
    }

    /// Format implied by the selector.
    #[must_use]
    pub fn format(&self) -> CorpusFormat {
        match &self.selector {
            CorpusSelector::Catalog(name) => name.format(),
            CorpusSelector::Hdf5Url { .. } => CorpusFormat::Hdf5,
            CorpusSelector::ParquetDir { .. } => CorpusFormat::Parquet,
            CorpusSelector::CodeDir { .. } => CorpusFormat::Code,
        }
    }

    /// Catalog metric, or a URL heuristic for a custom HDF5 file.
    #[must_use]
    pub fn default_metric(&self) -> Metric {
        match &self.selector {
            CorpusSelector::Catalog(name) => name.metric(),
            CorpusSelector::Hdf5Url { url, .. } => metric_from_url(url),
            CorpusSelector::ParquetDir { .. } => Metric::DotProduct,
            CorpusSelector::CodeDir { .. } => Metric::Cosine,
        }
    }

    /// Catalog dimension, if this selector has one.
    #[must_use]
    pub fn expected_dimension(&self) -> Option<usize> {
        match &self.selector {
            CorpusSelector::Catalog(name) => name.dimension(),
            CorpusSelector::Hdf5Url { .. }
            | CorpusSelector::ParquetDir { .. }
            | CorpusSelector::CodeDir { .. } => None,
        }
    }

    /// Language shard recorded in the manifest.
    #[must_use]
    pub fn language(&self) -> Option<String> {
        match &self.selector {
            CorpusSelector::Catalog(name) => name.language().map(str::to_string),
            CorpusSelector::Hdf5Url { .. }
            | CorpusSelector::ParquetDir { .. }
            | CorpusSelector::CodeDir { .. } => None,
        }
    }
}

/// True when the process asked for a real corpus instead of the synthetic live scene.
#[must_use]
pub fn corpus_requested() -> bool {
    env_nonempty(ENV_CORPUS)
        || env_nonempty(ENV_CORPUS_URL)
        || env_nonempty(ENV_CORPUS_PARQUET_DIR)
        || env_nonempty(ENV_CORPUS_CODE_DIR)
        || env_nonempty(ENV_INT_PARQUET_DIR)
        || env_nonempty(ENV_INT_DATASET_URL)
}

/// Build a request from the process environment.
///
/// # Errors
///
/// Returns an error when a corpus name is unknown or a required path is empty.
pub fn request_from_env(profile: CorpusProfile) -> Result<CorpusRequest, DripstoneError> {
    let (train_default, query_default) = match profile {
        CorpusProfile::Integration => (INTEGRATION_TRAIN_MAX, INTEGRATION_QUERY_MAX),
        CorpusProfile::Soak => (SOAK_TRAIN_MAX, SOAK_QUERY_MAX),
    };
    let caps = super::layout::CorpusCaps {
        train_max: resolve_cap(
            env_usize(ENV_CORPUS_TRAIN_MAX),
            env_usize(ENV_INT_TRAIN_MAX),
            train_default,
        ),
        query_max: resolve_cap(
            env_usize(ENV_CORPUS_QUERY_MAX),
            env_usize(ENV_INT_QUERY_MAX),
            query_default,
        ),
    };
    let neighbors_k = resolve_cap(
        env_usize(ENV_CORPUS_NEIGHBORS_K),
        env_usize(ENV_INT_NEIGHBORS_K),
        DEFAULT_NEIGHBORS_K,
    );
    let emb_column = env_string(ENV_CORPUS_EMB_COLUMN)
        .or_else(|| env_string(ENV_INT_PARQUET_EMB_COLUMN))
        .unwrap_or_else(|| EMB_COLUMN.to_string());
    let parquet_recursive =
        env_flag(ENV_CORPUS_PARQUET_RECURSIVE) || env_flag(ENV_INT_PARQUET_RECURSIVE);
    let shard_max = env_usize(ENV_CORPUS_SHARD_MAX).unwrap_or(DEFAULT_SHARD_MAX);
    let selector = selector_from_env()?;
    let grade_policy = grade_policy_from_env()?;
    Ok(CorpusRequest {
        selector,
        caps,
        neighbors_k,
        cache_dir: corpus_cache_dir(),
        emb_column,
        parquet_recursive,
        shard_max: shard_max.max(1),
        grade_policy,
    })
}

/// Grade policy from [`ENV_CORPUS_GRADE_POLICY`]. Unset is binary answer labels.
///
/// # Errors
///
/// Returns an error when the variable is set to an unknown policy.
pub fn grade_policy_from_env() -> Result<GradePolicy, DripstoneError> {
    match env_string(ENV_CORPUS_GRADE_POLICY) {
        Some(value) => parse_grade_policy(&value),
        None => Ok(GradePolicy::BinaryAnswer),
    }
}

/// Technique plugin from [`ENV_CORPUS_TECHNIQUE`]. Unset is hard neighbors.
///
/// # Errors
///
/// Returns an error when the variable is set to an unknown technique.
pub fn technique_from_env() -> Result<RetrievalTechnique, DripstoneError> {
    match env_string(ENV_CORPUS_TECHNIQUE) {
        Some(value) => parse_technique(&value),
        None => Ok(RetrievalTechnique::HardNeighbors),
    }
}

/// First positive integer among `primary` and `alias`, otherwise `default`.
#[must_use]
pub fn resolve_cap(primary: Option<usize>, alias: Option<usize>, default: usize) -> usize {
    primary
        .or(alias)
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

/// ColdForge body PUT concurrency from [`ENV_CORPUS_PUT_CONCURRENCY`].
#[must_use]
pub fn body_put_concurrency_from_env(default: usize) -> usize {
    env_usize(ENV_CORPUS_PUT_CONCURRENCY)
        .filter(|value| *value > 0)
        .unwrap_or(default)
        .max(1)
}

/// Upsert batch size from [`ENV_CORPUS_BATCH`].
#[must_use]
pub fn upsert_batch_from_env(default: usize) -> usize {
    env_usize(ENV_CORPUS_BATCH)
        .filter(|value| *value > 0)
        .unwrap_or(default)
        .max(1)
}

/// `DRIPSTONE_CORPUS_METRIC`, then `DRIPSTONE_INT_SEARCH_METRIC`.
///
/// # Errors
///
/// Returns an error when a metric token is set but not recognized.
pub fn metric_from_env() -> Result<Option<Metric>, DripstoneError> {
    if let Some(token) = env_string(ENV_CORPUS_METRIC) {
        return Ok(Some(parse_metric_name(&token)?));
    }
    if let Some(token) = env_string(ENV_INT_SEARCH_METRIC) {
        return Ok(Some(parse_metric_name(&token)?));
    }
    Ok(None)
}

/// Parse a metric token.
///
/// # Errors
///
/// Returns an error when `token` is not a known metric name.
pub fn parse_metric_name(token: &str) -> Result<Metric, DripstoneError> {
    match token.trim().to_ascii_lowercase().as_str() {
        "euclidean" | "l2" => Ok(Metric::Euclidean),
        "cosine" | "angular" => Ok(Metric::Cosine),
        "dot" | "dotproduct" | "dot_product" => Ok(Metric::DotProduct),
        "manhattan" | "l1" => Ok(Metric::Manhattan),
        other => Err(DripstoneError::storage(format!(
            "unknown metric `{other}`; expected cosine, euclidean, dot, or manhattan"
        ))),
    }
}

/// `$DRIPSTONE_CACHE_DIR/corpora`, using the same cache root as model downloads.
#[must_use]
pub fn corpus_cache_dir() -> PathBuf {
    cache_root().join("corpora")
}

fn selector_from_env() -> Result<CorpusSelector, DripstoneError> {
    if let Some(dir) =
        env_string(ENV_CORPUS_PARQUET_DIR).or_else(|| env_string(ENV_INT_PARQUET_DIR))
    {
        let name = env_string(ENV_CORPUS)
            .or_else(|| env_string(ENV_INT_DATASET_NAME))
            .unwrap_or_else(|| "parquet".to_string());
        return Ok(CorpusSelector::ParquetDir {
            name: sanitize_layout_name(&name)?,
            path: PathBuf::from(dir),
        });
    }

    let code_dir = env_string(ENV_CORPUS_CODE_DIR);
    if let Some(name) = env_string(ENV_CORPUS) {
        let parsed = CorpusName::parse(&name)?;
        if parsed == CorpusName::Code
            && let Some(dir) = code_dir
        {
            return Ok(CorpusSelector::CodeDir {
                name: sanitize_layout_name(parsed.as_str())?,
                path: PathBuf::from(dir),
            });
        }
        return Ok(CorpusSelector::Catalog(parsed));
    }
    if let Some(dir) = code_dir {
        return Ok(CorpusSelector::CodeDir {
            name: sanitize_layout_name(CorpusName::Code.as_str())?,
            path: PathBuf::from(dir),
        });
    }

    if let Some(url) = env_string(ENV_CORPUS_URL).or_else(|| env_string(ENV_INT_DATASET_URL)) {
        if let Some(name) = CorpusName::from_url(&url) {
            return Ok(CorpusSelector::Catalog(name));
        }
        let name = env_string(ENV_INT_DATASET_NAME)
            .or_else(|| file_stem_from_url(&url))
            .unwrap_or_else(|| "hdf5".to_string());
        return Ok(CorpusSelector::Hdf5Url {
            name: sanitize_layout_name(&name)?,
            url,
        });
    }

    Ok(CorpusSelector::Catalog(CorpusName::Nytimes256Angular))
}

fn metric_from_url(url: &str) -> Metric {
    if url.to_ascii_lowercase().contains("euclidean") {
        Metric::Euclidean
    } else {
        Metric::Cosine
    }
}

fn sanitize_layout_name(name: &str) -> Result<String, DripstoneError> {
    let trimmed = name.trim();
    if trimmed.is_empty()
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || trimmed.contains("..")
        || trimmed.contains('\0')
    {
        return Err(DripstoneError::storage(format!(
            "corpus name `{trimmed}` is not a single path segment"
        )));
    }
    Ok(trimmed.to_string())
}

fn file_stem_from_url(url: &str) -> Option<String> {
    let tail = url.rsplit('/').next()?.split('?').next()?;
    let stem = tail.strip_suffix(".hdf5").unwrap_or(tail);
    if stem.is_empty() {
        None
    } else {
        Some(stem.to_string())
    }
}

fn short_hash(text: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(text.as_bytes());
    hex::encode(&digest[..6])
}

fn env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_nonempty(name: &str) -> bool {
    env_string(name).is_some()
}

fn env_usize(name: &str) -> Option<usize> {
    env_string(name).and_then(|value| value.parse().ok())
}

fn env_flag(name: &str) -> bool {
    env_string(name).is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}
