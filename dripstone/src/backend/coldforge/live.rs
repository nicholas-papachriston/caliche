//! One ColdForge scene, three backends, one table.
//!
//! [`plan_from_env`] skips unless `DRIPSTONE_LIVE_AWS=1`. The backend is
//! `DRIPSTONE_LIVE_BACKEND`: `minio`, `s3`, or `s3-redis`. When that variable is
//! unset, an `OBJECT_STORE_ENDPOINT` selects MinIO, `REDIS_URL` selects S3 plus
//! Redis, and otherwise the run is S3 cold-only.
//!
//! The default scene is dimension 128, cosine, namespace `live`, N in
//! {50, 200, 1000}. Set `DRIPSTONE_CORPUS` (or `DRIPSTONE_CORPUS_URL` /
//! `DRIPSTONE_CORPUS_PARQUET_DIR`, with `DRIPSTONE_INT_*` aliases) to ingest that
//! corpus instead: real dimension, metric, and the capped row count. Cold search
//! opens a fresh backend per sample: 2 samples on
//! AWS (`s3`, `s3-redis`) and 50 on MinIO. Warm search reuses that backend for
//! 50 samples. The report is one markdown table and one TSV from this process.
//! A release build is the measurement to keep.
//!
//! `wal_refresh_interval_ms` is [`super::DEFAULT_WAL_REFRESH_INTERVAL_MS`]
//! (1000). `0` lists the object-store WAL on every refreshing read, including
//! `get`, before Redis. This scene does not use `0`.
//!
//! Get samples prime the index once, then drop the in-process body map and the
//! LRU before each timed `get`. S3 samples measure object hydrate. S3+Redis
//! samples measure the hot tier when it still holds the id.

use std::sync::Arc;
use std::time::Instant;

use aws_sdk_s3::Client as S3Client;
use aws_sdk_s3::config::{
    BehaviorVersion, Builder as S3ConfigBuilder, Credentials, Region as S3Region,
};

use crate::core::{Chunk, Metric, SearchQuery};
use crate::corpus::graded_recall_at_k;
use crate::errors::DripstoneError;
use crate::storage::{
    ObjectStore, ObjectStoreConfig, create_object_store, parse_object_store_path_style,
};
use crate::traits::Backend;

use super::{ColdForgeBackend, ColdForgeConfig};

pub(super) const SCENE_NAMESPACE: &str = "live";
pub(super) const DIMENSION: usize = 128;
const WARM_SEARCH_SAMPLES: usize = 50;
const GET_HYDRATE_SAMPLES: usize = 50;
const MINIO_COLD_SEARCH_SAMPLES: usize = 50;
const AWS_COLD_SEARCH_SAMPLES: usize = 2;
const SEARCH_K: usize = 10;
const HEAD_SAMPLES: usize = 20;
pub(super) const SIZES: [usize; 3] = [50, 200, 1_000];

/// Which object store this process is measuring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveBackend {
    /// Local MinIO (or another generic S3 endpoint). The local floor.
    Minio,
    /// AWS S3 with no hot tier.
    S3,
    /// AWS S3 plus a Redis hot tier.
    S3Redis,
}

impl LiveBackend {
    /// Stable name used in the table.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minio => "minio",
            Self::S3 => "s3",
            Self::S3Redis => "s3-redis",
        }
    }
}

/// What [`plan_from_env`] decided.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiveAwsPlan {
    /// The live flag is unset. Callers should exit successfully.
    Skip {
        /// Why the probe did not run.
        reason: String,
    },
    /// Inputs for one backend.
    Run(LiveAwsOptions),
}

/// Inputs for one live run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveAwsOptions {
    /// Backend selected for this process.
    pub backend: LiveBackend,
    /// Bucket name.
    pub bucket: String,
    /// Region label printed in the header.
    pub region: String,
    /// Unique prefix for this process. Each N gets a child prefix.
    pub key_prefix: String,
    /// Redis URL when the backend is [`LiveBackend::S3Redis`].
    pub redis_url: Option<String>,
}

/// One N in the scene.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneRow {
    /// `minio`, `s3`, or `s3-redis`.
    pub backend: String,
    /// Corpus size.
    pub n: usize,
    /// Upsert of N vectors plus the WAL flush, milliseconds.
    pub upsert_flush_ms: f64,
    /// Cold search percentiles. Each sample is a new backend.
    ///
    /// AWS backends take 2 samples. MinIO takes 50.
    pub search_cold_p50_ms: f64,
    /// Cold search p90.
    pub search_cold_p90_ms: f64,
    /// Cold search p99.
    pub search_cold_p99_ms: f64,
    /// Warm search percentiles on the last cold backend.
    pub search_warm_p50_ms: f64,
    /// Warm search p90.
    pub search_warm_p90_ms: f64,
    /// Warm search p99.
    pub search_warm_p99_ms: f64,
    /// Get hydrate p50. Each sample drops the in-process body map and LRU.
    pub get_p50_ms: f64,
    /// Get hydrate p90.
    pub get_p90_ms: f64,
    /// HeadBucket p50. Network context, repeated on every N row.
    pub head_bucket_p50_ms: Option<f64>,
    /// Hits from the last warm search.
    pub hits: usize,
    /// Brute-force top-k count on the same N vectors.
    pub brute_hits: usize,
    /// Overlap of the warm search ids with the brute-force top-k.
    pub recall_at_k: f64,
    /// Set recall of answer-class judgments. `None` when that class is empty.
    pub answer_at_k: Option<f64>,
    /// Set recall of support-class judgments. `None` when that class is empty.
    pub support_at_k: Option<f64>,
}

/// Markdown table, TSV, and the rows behind them.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveReport {
    /// Header, markdown table, and TSV.
    pub markdown: String,
    /// TSV body including the header row.
    pub tsv: String,
    /// One row per N.
    pub rows: Vec<SceneRow>,
}

impl LiveReport {
    /// True when a warm search returned no hits.
    #[must_use]
    pub fn any_empty_search(&self) -> bool {
        self.rows.iter().any(|row| row.hits == 0)
    }
}

/// Parse `DRIPSTONE_LIVE_BACKEND`.
///
/// # Errors
///
/// Returns an error when `name` is not `minio`, `s3`, or `s3-redis`.
pub fn parse_backend_name(name: &str) -> Result<LiveBackend, DripstoneError> {
    match name.trim().to_ascii_lowercase().as_str() {
        "minio" => Ok(LiveBackend::Minio),
        "s3" => Ok(LiveBackend::S3),
        "s3-redis" => Ok(LiveBackend::S3Redis),
        other => Err(DripstoneError::storage(format!(
            "DRIPSTONE_LIVE_BACKEND must be minio, s3, or s3-redis; got {other}"
        ))),
    }
}

/// Skip unless `DRIPSTONE_LIVE_AWS=1`.
///
/// # Errors
///
/// Returns an error when the flag is set and the selected backend is missing a bucket,
/// MinIO endpoint, or Redis URL.
pub fn plan_from_env() -> Result<LiveAwsPlan, DripstoneError> {
    let flag = std::env::var("DRIPSTONE_LIVE_AWS").unwrap_or_default();
    if flag != "1" {
        return Ok(LiveAwsPlan::Skip {
            reason: "DRIPSTONE_LIVE_AWS is not 1; the live scene was not run".to_string(),
        });
    }
    let backend = match std::env::var("DRIPSTONE_LIVE_BACKEND") {
        Ok(name) if !name.trim().is_empty() => parse_backend_name(&name)?,
        _ => infer_backend(),
    };
    let bucket = std::env::var("COLDFORGE_BUCKET").unwrap_or_default();
    if bucket.is_empty() {
        return Err(DripstoneError::storage(
            "DRIPSTONE_LIVE_AWS=1 requires COLDFORGE_BUCKET",
        ));
    }
    validate_backend(backend)?;
    let region = std::env::var("AWS_REGION")
        .or_else(|_| std::env::var("OBJECT_STORE_REGION"))
        .unwrap_or_else(|_| "us-east-1".to_string());
    let stamp = time::OffsetDateTime::now_utc().unix_timestamp_nanos();
    let redis_url = std::env::var("REDIS_URL")
        .ok()
        .filter(|url| !url.is_empty());
    Ok(LiveAwsPlan::Run(LiveAwsOptions {
        backend,
        bucket,
        region,
        key_prefix: format!("coldforge-live/{stamp}/"),
        redis_url,
    }))
}

fn infer_backend() -> LiveBackend {
    let endpoint = std::env::var("OBJECT_STORE_ENDPOINT")
        .or_else(|_| std::env::var("DRIPSTONE_MINIO_ENDPOINT"))
        .ok()
        .filter(|value| !value.is_empty());
    let provider = std::env::var("OBJECT_STORE_PROVIDER").unwrap_or_default();
    if endpoint.is_some() || provider.eq_ignore_ascii_case("generic") {
        LiveBackend::Minio
    } else if std::env::var("REDIS_URL")
        .ok()
        .is_some_and(|url| !url.is_empty())
    {
        LiveBackend::S3Redis
    } else {
        LiveBackend::S3
    }
}

fn validate_backend(backend: LiveBackend) -> Result<(), DripstoneError> {
    match backend {
        LiveBackend::Minio => {
            let endpoint = std::env::var("OBJECT_STORE_ENDPOINT")
                .or_else(|_| std::env::var("DRIPSTONE_MINIO_ENDPOINT"))
                .unwrap_or_default();
            if endpoint.is_empty() {
                return Err(DripstoneError::storage(
                    "minio backend requires OBJECT_STORE_ENDPOINT or DRIPSTONE_MINIO_ENDPOINT",
                ));
            }
            Ok(())
        }
        LiveBackend::S3 => Ok(()),
        LiveBackend::S3Redis => {
            let url = std::env::var("REDIS_URL").unwrap_or_default();
            if url.is_empty() {
                return Err(DripstoneError::storage(
                    "s3-redis backend requires REDIS_URL",
                ));
            }
            #[cfg(not(feature = "redis-backend"))]
            {
                return Err(DripstoneError::storage(
                    "s3-redis backend requires the redis-backend feature",
                ));
            }
            #[cfg(feature = "redis-backend")]
            {
                Ok(())
            }
        }
    }
}

/// Run the scene for the selected backend.
///
/// # Errors
///
/// Returns an error when the object store, Redis, or a timed read fails.
/// An empty search is still returned in the report; callers should check
/// [`LiveReport::any_empty_search`].
pub async fn run(options: &LiveAwsOptions) -> Result<LiveReport, DripstoneError> {
    let store_config = object_store_config(options)?;
    let store = create_object_store(store_config).await?;
    let hot = connect_hot(options).await?;
    let (head_p50, head_note) = head_bucket_p50(options).await;
    let scene = super::live_corpus::load_scene().await?;
    let concurrency =
        crate::corpus::body_put_concurrency_from_env(super::DEFAULT_BODY_PUT_CONCURRENCY);
    let batch = crate::corpus::upsert_batch_from_env(crate::corpus::DEFAULT_UPSERT_BATCH);
    let mut rows = Vec::with_capacity(scene.slices.len());
    for slice in &scene.slices {
        rows.push(
            measure_slice(
                &store,
                options,
                hot.clone(),
                slice,
                scene.dimension,
                scene.metric,
                concurrency,
                batch,
                head_p50,
            )
            .await?,
        );
    }
    let tsv = format_tsv(&rows);
    let markdown = render(options, &scene.header, &rows, &tsv, &head_note);
    Ok(LiveReport {
        markdown,
        tsv,
        rows,
    })
}

fn object_store_config(options: &LiveAwsOptions) -> Result<ObjectStoreConfig, DripstoneError> {
    match options.backend {
        LiveBackend::Minio => minio_config(),
        LiveBackend::S3 | LiveBackend::S3Redis => Ok(ObjectStoreConfig::aws_s3(
            Some(options.region.clone()),
            None,
            None,
            None,
        )),
    }
}

fn minio_config() -> Result<ObjectStoreConfig, DripstoneError> {
    let endpoint = std::env::var("OBJECT_STORE_ENDPOINT")
        .or_else(|_| std::env::var("DRIPSTONE_MINIO_ENDPOINT"))
        .map_err(|_| {
            DripstoneError::storage(
                "minio backend requires OBJECT_STORE_ENDPOINT or DRIPSTONE_MINIO_ENDPOINT",
            )
        })?;
    let region = std::env::var("OBJECT_STORE_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let path_style = match std::env::var("OBJECT_STORE_PATH_STYLE") {
        Ok(raw) => parse_object_store_path_style(&raw)?,
        Err(_) => true,
    };
    let access_key = std::env::var("DRIPSTONE_MINIO_ACCESS_KEY")
        .or_else(|_| std::env::var("AWS_ACCESS_KEY_ID"))
        .ok()
        .filter(|value| !value.is_empty());
    let secret_key = std::env::var("DRIPSTONE_MINIO_SECRET_KEY")
        .or_else(|_| std::env::var("AWS_SECRET_ACCESS_KEY"))
        .ok()
        .filter(|value| !value.is_empty());
    let (access_key, secret_key) = match (access_key, secret_key) {
        (Some(access), Some(secret)) => (Some(access), Some(secret)),
        _ => (
            Some("minioadmin".to_string()),
            Some("minioadmin".to_string()),
        ),
    };
    Ok(ObjectStoreConfig::generic_s3(
        endpoint, region, path_style, access_key, secret_key, None,
    ))
}

async fn connect_hot(
    options: &LiveAwsOptions,
) -> Result<Option<Arc<dyn Backend + Send + Sync>>, DripstoneError> {
    match options.backend {
        LiveBackend::Minio | LiveBackend::S3 => Ok(None),
        LiveBackend::S3Redis => {
            let url = options
                .redis_url
                .as_deref()
                .filter(|url| !url.is_empty())
                .ok_or_else(|| DripstoneError::storage("s3-redis backend requires REDIS_URL"))?;
            #[cfg(not(feature = "redis-backend"))]
            {
                let _ = url;
                Err(DripstoneError::storage(
                    "s3-redis backend requires the redis-backend feature",
                ))
            }
            #[cfg(feature = "redis-backend")]
            {
                let backend = crate::backend::redis::RedisBackend::with_prefix(
                    url,
                    &format!("coldforge:live:{}", options.key_prefix),
                )
                .await?;
                Ok(Some(Arc::new(backend)))
            }
        }
    }
}

async fn measure_slice(
    store: &Arc<dyn ObjectStore>,
    options: &LiveAwsOptions,
    hot: Option<Arc<dyn Backend + Send + Sync>>,
    slice: &super::live_corpus::LiveSlice,
    dimension: usize,
    metric: crate::core::Metric,
    body_put_concurrency: usize,
    batch: usize,
    head_p50: Option<f64>,
) -> Result<SceneRow, DripstoneError> {
    let n = slice.chunks.len();
    let prefix = format!("{}n{n}/", options.key_prefix);
    let writer = open(
        store,
        options,
        &prefix,
        8 * 1024 * 1024,
        hot.clone(),
        dimension,
        body_put_concurrency,
    );
    let started = Instant::now();
    for group in slice.chunks.chunks(batch.max(1)) {
        writer.upsert(group).await?;
    }
    writer.flush().await?;
    let upsert_flush_ms = elapsed_since(started);

    let query = SearchQuery::new(slice.query.clone())
        .metric(metric)
        .limit(SEARCH_K)
        .namespace(SCENE_NAMESPACE);
    let (cold, warm_backend) = cold_searches(
        store,
        options,
        &prefix,
        hot.clone(),
        &query,
        dimension,
        body_put_concurrency,
    )
    .await?;
    let (warm, hits) = warm_searches(&warm_backend, &query).await?;
    let brute = brute_ids(&slice.chunks, &query.vector, SEARCH_K, metric)?;
    let retrieved = hits
        .iter()
        .map(|hit| hit.chunk.id.clone())
        .collect::<Vec<_>>();
    let recall = recall_at_k(&retrieved, &brute);
    let graded = slice.judgments.as_ref().map(|judgments| {
        let indices = super::live_corpus::retrieved_train_indices(&slice.chunks, &retrieved);
        graded_recall_at_k(&indices, &judgments.neighbors, SEARCH_K)
    });
    let answer_at_k = graded.as_ref().and_then(|row| row.answer_at_k);
    let support_at_k = graded.as_ref().and_then(|row| row.support_at_k);
    let (primer, hydrate_ids) = timed_get_ids(&slice.chunks);
    let gets = time_gets(
        store,
        options,
        &prefix,
        hot,
        dimension,
        body_put_concurrency,
        &primer,
        &hydrate_ids,
    )
    .await?;
    let cold_q = quantiles(&cold);
    let warm_q = quantiles(&warm);
    let get_q = quantiles(&gets);
    Ok(SceneRow {
        backend: options.backend.as_str().to_string(),
        n,
        upsert_flush_ms,
        search_cold_p50_ms: cold_q.p50,
        search_cold_p90_ms: cold_q.p90,
        search_cold_p99_ms: cold_q.p99,
        search_warm_p50_ms: warm_q.p50,
        search_warm_p90_ms: warm_q.p90,
        search_warm_p99_ms: warm_q.p99,
        get_p50_ms: get_q.p50,
        get_p90_ms: get_q.p90,
        head_bucket_p50_ms: head_p50,
        hits: retrieved.len(),
        brute_hits: brute.len(),
        recall_at_k: recall,
        answer_at_k,
        support_at_k,
    })
}

pub(super) fn open(
    store: &Arc<dyn ObjectStore>,
    options: &LiveAwsOptions,
    key_prefix: &str,
    cache_size_bytes: usize,
    hot_tier: Option<Arc<dyn Backend + Send + Sync>>,
    dimension: usize,
    body_put_concurrency: usize,
) -> ColdForgeBackend {
    let refresh_shared_wal = match options.backend {
        LiveBackend::S3Redis => None,
        LiveBackend::Minio | LiveBackend::S3 => Some(false),
    };
    ColdForgeBackend::with_object_store(
        Arc::clone(store),
        ColdForgeConfig {
            bucket: options.bucket.clone(),
            key_prefix: key_prefix.to_string(),
            region: options.region.clone(),
            cache_size_bytes,
            num_centroids: 1,
            wal_batch_interval_ms: 0,
            wal_max_batch_size: 1_000_000,
            enable_compression: false,
            vector_dimension: dimension,
            hot_tier,
            refresh_shared_wal,
            wal_refresh_interval_ms: super::DEFAULT_WAL_REFRESH_INTERVAL_MS,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            object_store_config: None,
            body_put_concurrency,
        },
    )
}

pub(super) fn sample_chunks(count: usize) -> Vec<Chunk> {
    let mut chunks = Vec::with_capacity(count);
    for index in 0..count {
        let doc_id = format!("row-{index}");
        chunks.push(Chunk::new(
            doc_id,
            0,
            "",
            scene_vector(index),
            0,
            0,
            1,
            serde_json::json!({}),
            Some(SCENE_NAMESPACE.to_string()),
        ));
    }
    chunks
}

fn scene_vector(index: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; DIMENSION];
    vector[index % DIMENSION] = 1.0;
    vector[(index.saturating_mul(3).saturating_add(1)) % DIMENSION] += 0.05;
    let bump = u32::try_from(index).map_or(0.0, |value| (value as f32) * 0.001);
    vector[0] += bump;
    vector
}

fn cold_search_samples(backend: LiveBackend) -> usize {
    match backend {
        LiveBackend::Minio => MINIO_COLD_SEARCH_SAMPLES,
        LiveBackend::S3 | LiveBackend::S3Redis => AWS_COLD_SEARCH_SAMPLES,
    }
}

async fn cold_searches(
    store: &Arc<dyn ObjectStore>,
    options: &LiveAwsOptions,
    prefix: &str,
    hot: Option<Arc<dyn Backend + Send + Sync>>,
    query: &SearchQuery,
    dimension: usize,
    body_put_concurrency: usize,
) -> Result<(Vec<f64>, ColdForgeBackend), DripstoneError> {
    let sample_count = cold_search_samples(options.backend);
    let mut samples = Vec::with_capacity(sample_count);
    let mut kept = None;
    for sample in 0..sample_count {
        let backend = open(
            store,
            options,
            prefix,
            8 * 1024 * 1024,
            hot.clone(),
            dimension,
            body_put_concurrency,
        );
        let started = Instant::now();
        backend.search(query).await?;
        samples.push(elapsed_since(started));
        if sample + 1 == sample_count {
            kept = Some(backend);
        }
    }
    let backend = kept.ok_or_else(|| {
        DripstoneError::storage("cold search produced no backend for the warm samples")
    })?;
    Ok((samples, backend))
}

async fn warm_searches(
    backend: &ColdForgeBackend,
    query: &SearchQuery,
) -> Result<(Vec<f64>, Vec<crate::core::SearchResult>), DripstoneError> {
    let mut samples = Vec::with_capacity(WARM_SEARCH_SAMPLES);
    let mut last = Vec::new();
    for _ in 0..WARM_SEARCH_SAMPLES {
        let started = Instant::now();
        last = backend.search(query).await?;
        samples.push(elapsed_since(started));
    }
    Ok((samples, last))
}

#[cfg(test)]
pub(super) fn hydrate_ids(n: usize) -> Vec<String> {
    let last = n.saturating_sub(1);
    let take = GET_HYDRATE_SAMPLES.min(last);
    (0..take).map(|index| format!("row-{index}#0")).collect()
}

pub(super) fn timed_get_ids(chunks: &[Chunk]) -> (String, Vec<String>) {
    let primer = chunks
        .last()
        .map(|chunk| chunk.id.clone())
        .unwrap_or_default();
    let take = GET_HYDRATE_SAMPLES.min(chunks.len().saturating_sub(1));
    let ids = chunks
        .iter()
        .take(take)
        .map(|chunk| chunk.id.clone())
        .collect();
    (primer, ids)
}

pub(super) async fn time_gets(
    store: &Arc<dyn ObjectStore>,
    options: &LiveAwsOptions,
    prefix: &str,
    hot: Option<Arc<dyn Backend + Send + Sync>>,
    dimension: usize,
    body_put_concurrency: usize,
    primer: &str,
    ids: &[String],
) -> Result<Vec<f64>, DripstoneError> {
    let backend = open(
        store,
        options,
        prefix,
        1,
        hot,
        dimension,
        body_put_concurrency,
    );
    let primed = backend.get(primer, Some(SCENE_NAMESPACE)).await?;
    if primed.is_none() {
        return Err(DripstoneError::storage(format!(
            "get hydrate priming read missed {primer}"
        )));
    }
    let mut samples = Vec::with_capacity(ids.len());
    for id in ids {
        // Replay fills the body map. Drop it so this sample reads the hot tier
        // or the durable object, whichever get checks next.
        backend.discard_local_envelopes().await;
        let started = Instant::now();
        let fetched = backend.get(id, Some(SCENE_NAMESPACE)).await?;
        if fetched.is_none() {
            return Err(DripstoneError::storage(format!("get hydrate missed {id}")));
        }
        samples.push(elapsed_since(started));
    }
    Ok(samples)
}

fn brute_ids(
    chunks: &[Chunk],
    query: &[f32],
    k: usize,
    metric: Metric,
) -> Result<Vec<String>, DripstoneError> {
    let mut scored = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let score = metric.compute(query, chunk.vector.as_ref())?;
        scored.push((score, chunk.id.clone()));
    }
    scored.sort_by(|left, right| {
        let order = if metric.higher_is_better() {
            right.0.total_cmp(&left.0)
        } else {
            left.0.total_cmp(&right.0)
        };
        order.then_with(|| left.1.cmp(&right.1))
    });
    scored.truncate(k.min(scored.len()));
    Ok(scored.into_iter().map(|(_, id)| id).collect())
}

fn recall_at_k(retrieved: &[String], relevant: &[String]) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let overlap = retrieved
        .iter()
        .filter(|id| relevant.iter().any(|expected| expected == *id))
        .count();
    usize_to_f64(overlap) / usize_to_f64(relevant.len())
}

async fn head_bucket_p50(options: &LiveAwsOptions) -> (Option<f64>, String) {
    let client = match head_client(options).await {
        Ok(client) => client,
        Err(err) => return (None, format!("HeadBucket client was not built: {err}")),
    };
    let mut samples = Vec::with_capacity(HEAD_SAMPLES);
    for _ in 0..HEAD_SAMPLES {
        let started = Instant::now();
        if let Err(err) = client.head_bucket().bucket(&options.bucket).send().await {
            return (None, format!("HeadBucket failed: {err}"));
        }
        samples.push(elapsed_since(started));
    }
    (
        Some(percentile(&sorted(&samples), 0.50)),
        format!("HeadBucket p50 over {HEAD_SAMPLES} samples. Network context, not search latency."),
    )
}

async fn head_client(options: &LiveAwsOptions) -> Result<S3Client, DripstoneError> {
    match options.backend {
        LiveBackend::S3 | LiveBackend::S3Redis => {
            let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
            loader = loader.region(aws_config::Region::new(options.region.clone()));
            let sdk = loader.load().await;
            Ok(S3Client::new(&sdk))
        }
        LiveBackend::Minio => {
            let config = minio_config()?;
            let ObjectStoreConfig {
                provider:
                    crate::storage::ObjectStoreProvider::GenericS3 {
                        endpoint,
                        region,
                        path_style,
                        access_key_id,
                        secret_access_key,
                        session_token,
                    },
            } = config
            else {
                return Err(DripstoneError::storage(
                    "minio head client expected a generic S3 config",
                ));
            };
            let (Some(access), Some(secret)) = (access_key_id, secret_access_key) else {
                return Err(DripstoneError::storage(
                    "minio head client is missing static credentials",
                ));
            };
            let credentials =
                Credentials::new(access, secret, session_token, None, "coldforge-live");
            let mut builder = S3ConfigBuilder::new()
                .behavior_version(BehaviorVersion::latest())
                .region(S3Region::new(region))
                .endpoint_url(endpoint)
                .credentials_provider(credentials);
            if path_style {
                builder = builder.force_path_style(true);
            }
            Ok(S3Client::from_conf(builder.build()))
        }
    }
}

fn elapsed_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

struct Quantiles {
    p50: f64,
    p90: f64,
    p99: f64,
}

fn quantiles(samples: &[f64]) -> Quantiles {
    let ordered = sorted(samples);
    Quantiles {
        p50: percentile(&ordered, 0.50),
        p90: percentile(&ordered, 0.90),
        p99: percentile(&ordered, 0.99),
    }
}

fn sorted(samples: &[f64]) -> Vec<f64> {
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    ordered
}

fn percentile(sorted_samples: &[f64], quantile: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }
    let rank = (quantile * usize_to_f64(sorted_samples.len())).ceil();
    let index = f64_to_index(rank - 1.0, sorted_samples.len());
    sorted_samples[index]
}

fn usize_to_f64(value: usize) -> f64 {
    u32::try_from(value).map_or(0.0, f64::from)
}

fn f64_to_index(value: f64, len: usize) -> usize {
    if value.is_sign_negative() {
        return 0;
    }
    let index = value as usize;
    index.min(len.saturating_sub(1))
}

fn format_tsv(rows: &[SceneRow]) -> String {
    let mut out = String::from(
        "backend\tn\tupsert_flush_ms\tsearch_cold_p50_ms\tsearch_cold_p90_ms\tsearch_cold_p99_ms\tsearch_warm_p50_ms\tsearch_warm_p90_ms\tsearch_warm_p99_ms\tget_p50_ms\tget_p90_ms\thead_bucket_p50_ms\thits\tbrute_hits\trecall_at_k\tanswer_at_k\tsupport_at_k\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            row.backend,
            row.n,
            fmt_ms(row.upsert_flush_ms),
            fmt_ms(row.search_cold_p50_ms),
            fmt_ms(row.search_cold_p90_ms),
            fmt_ms(row.search_cold_p99_ms),
            fmt_ms(row.search_warm_p50_ms),
            fmt_ms(row.search_warm_p90_ms),
            fmt_ms(row.search_warm_p99_ms),
            fmt_ms(row.get_p50_ms),
            fmt_ms(row.get_p90_ms),
            fmt_ms_opt(row.head_bucket_p50_ms),
            row.hits,
            row.brute_hits,
            format!("{:.3}", row.recall_at_k),
            fmt_ratio(row.answer_at_k),
            fmt_ratio(row.support_at_k),
        ));
    }
    out
}

fn render(
    options: &LiveAwsOptions,
    header: &super::live_corpus::SceneHeader,
    rows: &[SceneRow],
    tsv: &str,
    head_note: &str,
) -> String {
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let mut out = String::new();
    out.push_str("# ColdForge live scene\n\n");
    out.push_str(&format!("build: {profile}\n"));
    out.push_str(
        "recommended: cargo run --release --example coldforge_live_aws_example --features coldforge-backend,redis-backend\n",
    );
    out.push_str(&format!("backend: {}\n", options.backend.as_str()));
    out.push_str(&format!("region: {}\n", options.region));
    out.push_str(&format!("bucket: {}\n", options.bucket));
    out.push_str(&format!("key_prefix: {}\n", options.key_prefix));
    out.push_str(&format!("namespace: {SCENE_NAMESPACE}\n"));
    out.push_str(&format!("corpus: {}\n", header.corpus));
    out.push_str(&format!("metric: {}\n", header.metric));
    out.push_str(&format!("dimension: {}\n", header.dimension));
    out.push_str(&format!("n: {}\n", header.sizes));
    out.push_str(&format!(
        "dropped_train_rows: {}\n",
        header.dropped_train_rows
    ));
    out.push_str(&format!(
        "dropped_query_rows: {}\n",
        header.dropped_query_rows
    ));
    out.push_str(&format!("grade_policy: {}\n", header.grade_policy));
    out.push_str(&format!("technique: {}\n", header.technique));
    out.push_str(&format!(
        "cold_queries: {}\n",
        cold_search_samples(options.backend)
    ));
    out.push_str(&format!("warm_queries: {WARM_SEARCH_SAMPLES}\n"));
    out.push_str(&format!(
        "wal_refresh_interval_ms: {}\n",
        super::DEFAULT_WAL_REFRESH_INTERVAL_MS
    ));
    out.push_str(&format!("k: {SEARCH_K}\n"));
    out.push_str(&format!("{head_note}\n\n"));
    if cfg!(debug_assertions) {
        out.push_str(
            "This process is a debug build. Re-run with --release before comparing backends.\n\n",
        );
    }
    out.push_str(&format!(
        "Each N is a fresh key prefix and the same namespace, so upsert wall is that N plus its flush. \
Cold search opens a fresh backend per sample: {AWS_COLD_SEARCH_SAMPLES} on s3 and s3-redis, {MINIO_COLD_SEARCH_SAMPLES} on minio. \
Warm search reuses the last cold backend for {WARM_SEARCH_SAMPLES} samples. \
Get samples are distinct ids. One priming read loads the index. Each timed get drops the in-process body map and the LRU. \
S3 get samples measure object hydrate. S3+Redis get samples measure the hot tier when it still holds the id. \
wal_refresh_interval_ms is {interval}, the product default. \
A value of 0 lists the object-store WAL on every refreshing read, including get, before Redis. This scene uses {interval} ms. \
MinIO is the local floor. It is not an AWS network measurement. \
head_bucket_p50_ms is network context, not search latency. \
hits is the last warm search. recall_at_k is its overlap with brute-force top-k on the same N under the scene metric. \
answer_at_k and support_at_k are set recall of answer-class and support-class judgments for that query. \
`-` means that class has no positives (synthetic scene, or binary qrels with no support labels). \
They do not replace recall_at_k. An empty search shows hits 0 and recall 0.\n\n",
        interval = super::DEFAULT_WAL_REFRESH_INTERVAL_MS,
    ));
    out.push_str(&markdown_table(rows));
    out.push_str("\n```tsv\n");
    out.push_str(tsv);
    out.push_str("```\n");
    out
}

fn markdown_table(rows: &[SceneRow]) -> String {
    let mut out = String::from(
        "| backend | n | upsert_flush_ms | search_cold_p50_ms | search_cold_p90_ms | search_cold_p99_ms | search_warm_p50_ms | search_warm_p90_ms | search_warm_p99_ms | get_p50_ms | get_p90_ms | head_bucket_p50_ms | hits | brute_hits | recall_at_k | answer_at_k | support_at_k |\n",
    );
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in rows {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            row.backend,
            row.n,
            fmt_ms(row.upsert_flush_ms),
            fmt_ms(row.search_cold_p50_ms),
            fmt_ms(row.search_cold_p90_ms),
            fmt_ms(row.search_cold_p99_ms),
            fmt_ms(row.search_warm_p50_ms),
            fmt_ms(row.search_warm_p90_ms),
            fmt_ms(row.search_warm_p99_ms),
            fmt_ms(row.get_p50_ms),
            fmt_ms(row.get_p90_ms),
            fmt_ms_opt(row.head_bucket_p50_ms),
            row.hits,
            row.brute_hits,
            format!("{:.3}", row.recall_at_k),
            fmt_ratio(row.answer_at_k),
            fmt_ratio(row.support_at_k),
        ));
    }
    out
}

fn fmt_ms(millis: f64) -> String {
    format!("{millis:.1}")
}

fn fmt_ms_opt(millis: Option<f64>) -> String {
    millis.map_or_else(|| "-".to_string(), fmt_ms)
}

fn fmt_ratio(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |ratio| format!("{ratio:.3}"))
}

#[cfg(test)]
#[path = "live_tests.rs"]
mod tests;
