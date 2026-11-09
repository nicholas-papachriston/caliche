//! Benchmark scenarios: `ColdForge` vs S3 Vector cold storage, optional tiered hot tiers (mmap+HNSW,
//! S3 Vector + Redis + mmap+HNSW, S3 Vector + memory).

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::sync::Arc;
use std::time::{Duration, Instant};

use dripstone::backend::coldforge::{ColdForgeBackend, ColdForgeConfig};
use dripstone::backend::indexed::{IndexedBackend, IndexedBackendConfig};
use dripstone::backend::memory::MemoryBackend;
use dripstone::backend::mmap::{MmapBackend, MmapBackendConfig};
use dripstone::backend::redis::RedisBackend;
use dripstone::backend::s3::{S3VectorBackend, S3VectorBackendConfig};
use dripstone::index::{HnswConfig, HnswGraphSpace, HnswIndex};
use dripstone::traits::Backend;
use dripstone::{
    Chunk, DripstoneError, Metric, SearchQuery, SearchStrategy, TierConfig, TieredChainBuilder,
    WriteStrategy,
};
use serde_json::json;
use tempfile::tempdir;
use time::OffsetDateTime;
use tokio::time::sleep;

use super::aws::{self, IntegrationAwsConfig};
use super::datasets;
use super::metrics::{
    BenchmarkRun, ScenarioAnnParams, summarize_durations, summarize_recall_samples, write_run_json,
};

#[derive(Clone, Copy)]
enum ColdKind {
    ColdForge,
    S3Vector,
}

#[derive(Clone, Copy)]
enum Topology {
    ColdOnly,
    /// `ColdForge` or S3 Vector + HNSW indexed mmap hot tier (existing matrix).
    Tiered {
        write_strategy: WriteStrategy,
    },
    /// S3 Vector (cold) -> Redis (warm) -> HNSW + mmap (hot), same write strategy as other tiered runs.
    TieredS3RedisMmap {
        write_strategy: WriteStrategy,
    },
    /// S3 Vector (cold) -> in-memory (hot), similar to `tiered_advanced_example`.
    TieredS3MemoryHot,
}

struct Scenario {
    name: &'static str,
    cold: ColdKind,
    topology: Topology,
}

fn scenario_list() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "coldforge_cold_only",
            cold: ColdKind::ColdForge,
            topology: Topology::ColdOnly,
        },
        Scenario {
            name: "s3vector_cold_only",
            cold: ColdKind::S3Vector,
            topology: Topology::ColdOnly,
        },
        Scenario {
            name: "coldforge_tiered_write_source",
            cold: ColdKind::ColdForge,
            topology: Topology::Tiered {
                write_strategy: WriteStrategy::WriteToSource,
            },
        },
        Scenario {
            name: "coldforge_tiered_write_source_and_hot",
            cold: ColdKind::ColdForge,
            topology: Topology::Tiered {
                write_strategy: WriteStrategy::WriteToSourceAndHot,
            },
        },
        Scenario {
            name: "s3vector_tiered_write_source",
            cold: ColdKind::S3Vector,
            topology: Topology::Tiered {
                write_strategy: WriteStrategy::WriteToSource,
            },
        },
        Scenario {
            name: "s3vector_tiered_write_source_and_hot",
            cold: ColdKind::S3Vector,
            topology: Topology::Tiered {
                write_strategy: WriteStrategy::WriteToSourceAndHot,
            },
        },
        Scenario {
            name: "s3vector_tiered_redis_mmap_write_source_and_hot",
            cold: ColdKind::S3Vector,
            topology: Topology::TieredS3RedisMmap {
                write_strategy: WriteStrategy::WriteToSourceAndHot,
            },
        },
        Scenario {
            name: "s3vector_tiered_memory_hot",
            cold: ColdKind::S3Vector,
            topology: Topology::TieredS3MemoryHot,
        },
    ]
}

async fn prepare_s3_vectors(
    aws_cfg: &aws::IntegrationAwsConfig,
    dimension: usize,
    metric: Metric,
) -> Result<bool, DripstoneError> {
    let ensure = std::env::var("DRIPSTONE_INT_S3_VECTORS_ENSURE")
        .ok()
        .as_deref()
        == Some("1");
    if ensure {
        aws::ensure_s3_vector_sidecar_bucket(aws_cfg).await?;
        let dimension = i32::try_from(dimension)
            .map_err(|_| DripstoneError::storage("dimension does not fit i32".to_string()))?;
        aws::ensure_vector_bucket_and_index(aws_cfg, dimension, metric).await?;
        return Ok(true);
    }
    if aws::vector_index_present(aws_cfg).await? {
        aws::ensure_s3_vector_sidecar_bucket(aws_cfg).await?;
        return Ok(true);
    }
    errorln!(
        "S3 Vector index {} in bucket {} is not present; skipping those scenarios. Set DRIPSTONE_INT_S3_VECTORS_ENSURE=1 to create the bucket and index.",
        aws_cfg.s3_vector_index,
        aws_cfg.s3_vector_bucket
    );
    Ok(false)
}

fn integration_redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| {
        let p = std::env::var("STACK_REDIS_PASSWORD")
            .unwrap_or_else(|_| "dripstone_dev_redis".to_string());
        format!("redis://:{p}@127.0.0.1:6379/0")
    })
}

async fn redis_reachable() -> bool {
    RedisBackend::with_prefix(&integration_redis_url(), "vf-int-probe:")
        .await
        .is_ok()
}

impl Scenario {
    const fn needs_redis(&self) -> bool {
        matches!(self.topology, Topology::TieredS3RedisMmap { .. })
    }

    const fn uses_s3_vector(&self) -> bool {
        matches!(self.cold, ColdKind::S3Vector)
    }
}

fn chunks_for_corpus(base: &[Vec<f32>], namespace: &str) -> Result<Vec<Chunk>, DripstoneError> {
    let mut out = Vec::with_capacity(base.len());
    for (i, v) in base.iter().enumerate() {
        let doc_id = format!("doc{i}");
        let vec_arc: std::sync::Arc<[f32]> = std::sync::Arc::from(v.clone().into_boxed_slice());
        out.push(Chunk::new(
            doc_id,
            0,
            "",
            vec_arc,
            0,
            0,
            1,
            json!({"idx": i}),
            Some(namespace.to_string()),
        ));
    }
    Ok(out)
}

fn recall_at_k(
    results: &[dripstone::SearchResult],
    gt_row: Option<&[i32]>,
    k: usize,
) -> Option<f32> {
    let gt = gt_row?;
    let mut exact: std::collections::HashSet<i32> = gt.iter().take(k).copied().collect();
    if exact.is_empty() {
        return None;
    }
    let mut hit = 0_i32;
    for r in results.iter().take(k) {
        if let Some(rest) = r.chunk.doc_id.strip_prefix("doc")
            && let Ok(idx) = rest.parse::<i32>()
            && exact.remove(&idx)
        {
            hit += 1;
        }
    }
    Some(hit as f32 / k as f32)
}

fn retrieved_doc_indices(results: &[dripstone::SearchResult], k: usize) -> Vec<i32> {
    let mut indices = Vec::new();
    for result in results.iter().take(k) {
        let Some(rest) = result.chunk.doc_id.strip_prefix("doc") else {
            continue;
        };
        let Ok(index) = rest.parse::<i32>() else {
            continue;
        };
        indices.push(index);
    }
    indices
}

fn distribution_of(samples: &mut [f32]) -> Option<super::metrics::RecallPercentileSummary> {
    if samples.is_empty() {
        None
    } else {
        Some(super::metrics::summarize_recall_samples(samples))
    }
}

async fn build_coldforge(
    aws_cfg: &IntegrationAwsConfig,
    dim: usize,
) -> Result<Arc<dyn Backend>, DripstoneError> {
    let mut c = ColdForgeConfig::default();
    c.bucket = aws_cfg.s3_bucket.clone();
    c.key_prefix = aws_cfg.s3_prefix.clone();
    c.region = aws_cfg.region.clone();
    c.vector_dimension = dim;
    c.cache_size_bytes = 64 * 1024 * 1024;
    c.num_centroids = 64;
    c.wal_batch_interval_ms = 500;
    c.wal_max_batch_size = 384;
    c.enable_compression = true;
    let b = ColdForgeBackend::new(c).await?;
    Ok(Arc::new(b))
}

async fn build_s3_vector(
    aws_cfg: &IntegrationAwsConfig,
    dim: usize,
    distance_metric: Metric,
) -> Result<Arc<dyn Backend>, DripstoneError> {
    let cfg = S3VectorBackendConfig::new(
        aws_cfg.s3_vector_bucket.clone(),
        aws_cfg.s3_vector_index.clone(),
        dim,
        None,
        None,
        None,
    )
    .with_sidecar_s3_bucket(aws_cfg.s3_vector_sidecar_bucket.clone())
    .with_distance_metric(distance_metric)
    .with_region(&aws_cfg.s3_vectors_region)
    .with_batch_size(200)
    .with_search_oversample_factor(4)
    .with_search_oversample_max_candidates(64)
    .with_search_refine_with_vectors(true);
    let b = S3VectorBackend::new(cfg).await?;
    Ok(Arc::new(b))
}

/// Tuned HNSW for real ANN corpora (aligned with `speed_benchmarks::harness::tuned_hnsw_config`).
const fn integration_hnsw_config(graph_space: HnswGraphSpace) -> HnswConfig {
    HnswConfig {
        m: 32,
        ef_construction: 400,
        ef_search: 200,
        max_nb_connection: 32,
        rerank_overfetch: 4,
        graph_space,
    }
}

const fn integration_indexed_config() -> IndexedBackendConfig {
    use dripstone::IndexUpdateMode;
    IndexedBackendConfig {
        auto_rebuild_threshold: 0,
        fallback_to_brute_force: true,
        overfetch_factor: 2,
        index_update_mode: IndexUpdateMode::Synchronous,
        scan_unindexed_tail: true,
        deferred_flush_threshold: 256,
        deferred_search_flush_threshold: 0,
        max_tail_scan_records: 0,
    }
}

fn scenario_ann_params(
    distance_metric: Metric,
    graph_space: HnswGraphSpace,
    hc: &HnswConfig,
    ic: &IndexedBackendConfig,
) -> ScenarioAnnParams {
    ScenarioAnnParams {
        search_metric: format!("{distance_metric:?}"),
        hnsw_graph_space: format!("{graph_space:?}"),
        hnsw_m: hc.m,
        hnsw_ef_construction: hc.ef_construction,
        hnsw_ef_search: hc.ef_search,
        hnsw_rerank_overfetch: hc.rerank_overfetch,
        indexed_overfetch_factor: ic.overfetch_factor,
        indexed_auto_rebuild_threshold: ic.auto_rebuild_threshold,
    }
}

fn build_indexed_mmap_hot(
    dim: usize,
    graph_space: HnswGraphSpace,
) -> Result<
    (
        Arc<dyn Backend>,
        tempfile::TempDir,
        std::path::PathBuf,
        HnswConfig,
        IndexedBackendConfig,
    ),
    DripstoneError,
> {
    let dir = tempdir().map_err(|e| DripstoneError::storage_with_source("tempdir hot mmap", e))?;
    let path = dir.path().join("hot_tier.vf1");
    let mmap = MmapBackend::new(
        MmapBackendConfig::new(path.to_string_lossy(), dim)
            .with_read_write(true)
            .with_flush_every(512),
    )?;
    let hc = integration_hnsw_config(graph_space);
    let ic = integration_indexed_config();
    let hnsw = HnswIndex::new(hc.clone(), dim);
    let hot: Arc<dyn Backend> = Arc::new(IndexedBackend::new(mmap, Box::new(hnsw), ic.clone()));
    Ok((hot, dir, path, hc, ic))
}

async fn build_backend(
    aws_cfg: &IntegrationAwsConfig,
    scenario: &Scenario,
    dim: usize,
    distance_metric: Metric,
    hnsw_graph_space: HnswGraphSpace,
) -> Result<
    (
        Arc<dyn Backend>,
        Option<tempfile::TempDir>,
        Option<std::path::PathBuf>,
        Option<ScenarioAnnParams>,
    ),
    DripstoneError,
> {
    let cold: Arc<dyn Backend> = match scenario.cold {
        ColdKind::ColdForge => build_coldforge(aws_cfg, dim).await?,
        ColdKind::S3Vector => build_s3_vector(aws_cfg, dim, distance_metric).await?,
    };

    match scenario.topology {
        Topology::ColdOnly => Ok((cold, None, None, None)),
        Topology::Tiered { write_strategy } => {
            let (hot, hot_dir, hot_path, hc, ic) = build_indexed_mmap_hot(dim, hnsw_graph_space)?;
            let ann = scenario_ann_params(distance_metric, hnsw_graph_space, &hc, &ic);
            let chain: Arc<dyn Backend> = TieredChainBuilder::new()
                .tier(cold, TierConfig::new("cold"))
                .tier(
                    hot,
                    TierConfig::new("hot_mmap_indexed")
                        .max_vectors(50_000)
                        .max_bytes(512 * 1024 * 1024)
                        .promotion_threshold(1),
                )
                .write_strategy(write_strategy)
                .background_sync(false)
                .sync_interval_secs(3600)
                .build()?;
            Ok((chain, Some(hot_dir), Some(hot_path), Some(ann)))
        }
        Topology::TieredS3RedisMmap { write_strategy } => {
            if !matches!(scenario.cold, ColdKind::S3Vector) {
                return Err(DripstoneError::Misconfiguration(
                    "TieredS3RedisMmap requires S3 Vector cold tier".to_string(),
                ));
            }
            let redis_tier: Arc<dyn Backend> = Arc::new(
                RedisBackend::with_prefix(&integration_redis_url(), "vf-int-tier:")
                    .await
                    .map_err(|e| {
                        DripstoneError::storage(format!(
                            "Redis middle tier (set REDIS_URL or start compose redis): {e}"
                        ))
                    })?,
            );
            let (hot, hot_dir, hot_path, hc, ic) = build_indexed_mmap_hot(dim, hnsw_graph_space)?;
            let ann = scenario_ann_params(distance_metric, hnsw_graph_space, &hc, &ic);
            let chain: Arc<dyn Backend> = TieredChainBuilder::new()
                .tier(cold, TierConfig::new("cold_s3v"))
                .tier(
                    redis_tier,
                    TierConfig::new("warm_redis")
                        .max_vectors(50_000)
                        .max_bytes(512 * 1024 * 1024)
                        .promotion_threshold(1),
                )
                .tier(
                    hot,
                    TierConfig::new("hot_mmap_indexed")
                        .max_vectors(50_000)
                        .max_bytes(512 * 1024 * 1024)
                        .promotion_threshold(1),
                )
                .write_strategy(write_strategy)
                .search_strategy(SearchStrategy::HotThenSourceFallback)
                .background_sync(false)
                .sync_interval_secs(3600)
                .build()?;
            Ok((chain, Some(hot_dir), Some(hot_path), Some(ann)))
        }
        Topology::TieredS3MemoryHot => {
            if !matches!(scenario.cold, ColdKind::S3Vector) {
                return Err(DripstoneError::Misconfiguration(
                    "TieredS3MemoryHot requires S3 Vector cold tier".to_string(),
                ));
            }
            let promote = memory_hot_promotion_threshold();
            errorln!(
                "  tiered_s3_memory_hot: search_strategy=HotThenSourceFallback promotion_threshold={promote} \
                 (override with DRIPSTONE_INT_MEMORY_HOT_PROMOTION_THRESHOLD)"
            );
            let mem_hot: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
            let chain: Arc<dyn Backend> = TieredChainBuilder::new()
                .tier(cold, TierConfig::new("cold_s3v"))
                .tier(
                    mem_hot,
                    TierConfig::new("memory")
                        .max_vectors(50_000)
                        .max_bytes(512 * 1024 * 1024)
                        .promotion_threshold(promote),
                )
                .write_strategy(WriteStrategy::WriteToSourceAndHot)
                .search_strategy(SearchStrategy::HotThenSourceFallback)
                .background_sync(false)
                .sync_interval_secs(3600)
                .build()?;
            Ok((chain, None, None, None))
        }
    }
}

fn untimed_prewarm_enabled() -> bool {
    std::env::var("DRIPSTONE_INT_UNTIMED_PREWARM")
        .map(|s| s == "1" || s.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Hits on the in-memory hot tier before promotion (default `1`, same as other tiered benches).
/// Set to `2` or higher to require repeated access before serving from hot tier.
fn memory_hot_promotion_threshold() -> usize {
    std::env::var("DRIPSTONE_INT_MEMORY_HOT_PROMOTION_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
}

/// Extra post-ingest settle time for S3 Vector scenarios before readiness/recall checks.
/// Set `DRIPSTONE_INT_S3VECTOR_STABILIZATION_WAIT_SECS=0` to disable or raise to `10`
/// when probing index visibility latency on freshly ingested corpora.
fn s3vector_stabilization_wait() -> Duration {
    let secs = std::env::var("DRIPSTONE_INT_S3VECTOR_STABILIZATION_WAIT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5);
    Duration::from_secs(secs)
}

async fn wait_for_s3vector_query_readiness(
    backend: &Arc<dyn Backend>,
    scenario: &Scenario,
    queries: &[Vec<f32>],
    namespace: &str,
    limit: usize,
    metric: Metric,
) -> Result<(), DripstoneError> {
    if !scenario.uses_s3_vector() || queries.is_empty() {
        return Ok(());
    }

    let sample_count = queries.len().min(3);
    let max_attempts = 12;
    let retry_delay = Duration::from_millis(500);

    for attempt in 1..=max_attempts {
        let mut hit_counts = Vec::with_capacity(sample_count);
        for query in queries.iter().take(sample_count) {
            let search = SearchQuery::new(std::sync::Arc::from(query.clone()))
                .metric(metric)
                .limit(limit)
                .namespace(namespace);
            let results = backend.search(&search).await?;
            hit_counts.push(results.len());
        }

        if hit_counts.iter().all(|&hits| hits >= limit) {
            if attempt > 1 {
                errorln!(
                    "  s3vector readiness: scenario={} stabilized after {} attempt(s) with hit_counts={:?}",
                    scenario.name,
                    attempt,
                    hit_counts
                );
            }
            return Ok(());
        }

        errorln!(
            "  s3vector readiness: scenario={} attempt={}/{} hit_counts={:?}; waiting {:?}",
            scenario.name,
            attempt,
            max_attempts,
            hit_counts,
            retry_delay
        );

        if attempt < max_attempts {
            sleep(retry_delay).await;
        }
    }

    Err(DripstoneError::storage(format!(
        "S3 Vector search readiness did not stabilize for scenario {} after {} attempts; refusing to record potentially misleading benchmark numbers",
        scenario.name, max_attempts
    )))
}

async fn run_scenario(
    aws_cfg: &IntegrationAwsConfig,
    scenario: &Scenario,
    corpus: &datasets::DatasetCorpus,
    dim: usize,
) -> Result<BenchmarkRun, DripstoneError> {
    let scenario_start = Instant::now();
    let stage_elapsed_ms = || scenario_start.elapsed().as_secs_f64() * 1000.0;
    errorln!(
        "  [stage +{:.1} ms] {} bootstrap_start",
        stage_elapsed_ms(),
        scenario.name
    );
    let ns = format!(
        "vf-{}-{}",
        scenario.name,
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let t_load = Instant::now();
    let chunks = chunks_for_corpus(&corpus.base, &ns)?;
    let dataset_load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let t_boot = Instant::now();
    let (backend, _hot_tmpdir, hot_mmap_path, scenario_ann) = build_backend(
        aws_cfg,
        scenario,
        dim,
        corpus.search_metric,
        corpus.hnsw_graph_space,
    )
    .await?;
    let bootstrap_ms = t_boot.elapsed().as_secs_f64() * 1000.0;
    errorln!(
        "  [stage +{:.1} ms] {} bootstrap_done",
        stage_elapsed_ms(),
        scenario.name
    );

    let batch = 200_usize;
    let t_ing = Instant::now();
    errorln!(
        "  [stage +{:.1} ms] {} ingest_start",
        stage_elapsed_ms(),
        scenario.name
    );
    for c in chunks.chunks(batch) {
        backend.upsert(c).await?;
    }
    let ingest_ms = t_ing.elapsed().as_secs_f64() * 1000.0;
    let vectors_ingested = corpus.base.len();
    errorln!(
        "  [stage +{:.1} ms] {} ingest_done",
        stage_elapsed_ms(),
        scenario.name
    );

    errorln!(
        "  [stage +{:.1} ms] {} flush_start",
        stage_elapsed_ms(),
        scenario.name
    );
    backend.flush().await.ok();
    errorln!(
        "  [stage +{:.1} ms] {} flush_done",
        stage_elapsed_ms(),
        scenario.name
    );

    if scenario.uses_s3_vector() {
        let stabilization_wait = s3vector_stabilization_wait();
        if !stabilization_wait.is_zero() {
            errorln!(
                "  [stage +{:.1} ms] {} stabilization_wait {:?} \
                 (override with DRIPSTONE_INT_S3VECTOR_STABILIZATION_WAIT_SECS)",
                stage_elapsed_ms(),
                scenario.name,
                stabilization_wait
            );
            sleep(stabilization_wait).await;
            errorln!(
                "  [stage +{:.1} ms] {} stabilization_wait_done",
                stage_elapsed_ms(),
                scenario.name
            );
        }
    }

    let limit = 10_usize;
    wait_for_s3vector_query_readiness(
        &backend,
        scenario,
        &corpus.queries,
        &ns,
        limit,
        corpus.search_metric,
    )
    .await?;

    let q0 = SearchQuery::new(std::sync::Arc::from(corpus.queries[0].clone()))
        .metric(corpus.search_metric)
        .limit(limit)
        .namespace(&ns);

    let prewarm = untimed_prewarm_enabled();
    if prewarm {
        let _ = backend.search(&q0).await?;
    }

    let t1 = Instant::now();
    errorln!(
        "  [stage +{:.1} ms] {} first_search_start",
        stage_elapsed_ms(),
        scenario.name
    );
    let _first = backend.search(&q0).await?;
    let first_search_ms = t1.elapsed().as_secs_f64() * 1000.0;
    errorln!(
        "  [stage +{:.1} ms] {} first_search_done",
        stage_elapsed_ms(),
        scenario.name
    );

    let mut per_query_recall: Vec<f32> = Vec::new();
    let mut per_query_answer: Vec<f32> = Vec::new();
    let mut per_query_support: Vec<f32> = Vec::new();

    let mut samples: Vec<Duration> = Vec::new();
    let iters = corpus.queries.len().saturating_mul(3).max(12);
    let nq = corpus.queries.len();
    for i in 0..iters {
        let q_idx = i % nq;
        errorln!(
            "  [query +{:.1} ms] {} iter={}/{} q_idx={} start",
            stage_elapsed_ms(),
            scenario.name,
            i + 1,
            iters,
            q_idx
        );
        let qi = &corpus.queries[q_idx];
        let q = SearchQuery::new(std::sync::Arc::from(qi.clone()))
            .metric(corpus.search_metric)
            .limit(limit)
            .namespace(&ns);
        let tq = Instant::now();
        let res = backend.search(&q).await?;
        let query_elapsed = tq.elapsed();
        samples.push(query_elapsed);
        errorln!(
            "  [query +{:.1} ms] {} iter={}/{} q_idx={} done ({:.1} ms, hits={})",
            stage_elapsed_ms(),
            scenario.name,
            i + 1,
            iters,
            q_idx,
            query_elapsed.as_secs_f64() * 1000.0,
            res.len()
        );

        // Recall vs ground truth for the ingested `base` (default: exact top-k on loaded train rows).
        if i < nq
            && let Some(gt) = corpus.ground_truth.as_ref()
            && let Some(row) = gt.get(q_idx)
            && let Some(r) = recall_at_k(&res, Some(row.as_slice()), limit)
        {
            per_query_recall.push(r);
        }
        if i < nq
            && let Some(judgments) = corpus.judgments.as_ref().and_then(|rows| rows.get(q_idx))
        {
            let retrieved = retrieved_doc_indices(&res, limit);
            let graded =
                dripstone::corpus::graded_recall_at_k(&retrieved, &judgments.neighbors, limit);
            if let Some(answer) = graded.answer_at_k {
                per_query_answer.push(answer as f32);
            }
            if let Some(support) = graded.support_at_k {
                per_query_support.push(support as f32);
            }
        }
    }

    let warmed = summarize_durations(&mut samples);
    let qps = if warmed.mean_ms > 0.0 {
        1000.0 / warmed.mean_ms
    } else {
        0.0
    };

    let recall_distribution = if per_query_recall.is_empty() {
        None
    } else {
        Some(summarize_recall_samples(&mut per_query_recall))
    };

    let recall_at_k = recall_distribution.as_ref().map(|d| d.mean);
    let answer_distribution = distribution_of(&mut per_query_answer);
    let support_distribution = distribution_of(&mut per_query_support);
    let answer_at_k = answer_distribution.as_ref().map(|d| d.mean);
    let support_at_k = support_distribution.as_ref().map(|d| d.mean);
    errorln!(
        "  [stage +{:.1} ms] {} query_loop_done",
        stage_elapsed_ms(),
        scenario.name
    );

    let mmap_snapshot_bytes = if let Some(p) = hot_mmap_path {
        std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let corpus_diagnostics = corpus.diagnostics.clone();

    errorln!(
        "  [stage +{:.1} ms] {} cleanup_start",
        stage_elapsed_ms(),
        scenario.name
    );
    backend.delete_namespace(&ns).await.map_err(|e| {
        DripstoneError::storage(format!(
            "cleanup delete_namespace failed for scenario {} namespace {}: {e}",
            scenario.name, ns
        ))
    })?;
    errorln!(
        "  [stage +{:.1} ms] {} cleanup_done",
        stage_elapsed_ms(),
        scenario.name
    );

    Ok(BenchmarkRun {
        scenario: scenario.name.to_string(),
        dataset_load_ms,
        bootstrap_ms,
        ingest_ms,
        vectors_ingested,
        first_search_ms,
        warmed_search: warmed,
        qps,
        recall_at_k,
        recall_distribution,
        answer_at_k,
        support_at_k,
        answer_distribution,
        support_distribution,
        mmap_snapshot_bytes,
        corpus_diagnostics,
        scenario_ann,
        untimed_prewarm_search: prewarm,
    })
}

pub async fn run_matrix() -> Result<(), DripstoneError> {
    let aws_cfg = aws::load_integration_aws_config().await?;

    aws::ensure_standard_bucket(&aws_cfg).await?;

    let t_ds = Instant::now();
    let corpus = datasets::load_integration_corpus().await?;
    let dim = corpus.dimension;
    errorln!(
        "dataset: {} ({:?}) train {} query {} ground_truth={:?} hdf5_neighbors_in_file={} (load wall {:.1} ms)",
        corpus.name,
        corpus.search_metric,
        corpus.base.len(),
        corpus.queries.len(),
        corpus.diagnostics.ground_truth_source,
        corpus.diagnostics.hdf5_neighbors_present,
        t_ds.elapsed().as_secs_f64() * 1000.0
    );

    let vectors_ready = prepare_s3_vectors(&aws_cfg, dim, corpus.search_metric).await?;

    let redis_ok = redis_reachable().await;
    if !redis_ok {
        errorln!(
            "dripstone integration: Redis not reachable at {} — skipping scenarios that need a Redis middle tier",
            integration_redis_url()
        );
    }

    for sc in scenario_list() {
        if sc.needs_redis() && !redis_ok {
            errorln!("--- scenario: {} --- (skipped: Redis unavailable)", sc.name);
            continue;
        }
        if sc.uses_s3_vector() && !vectors_ready {
            errorln!(
                "--- scenario: {} --- (skipped: S3 Vector index is not present; set DRIPSTONE_INT_S3_VECTORS_ENSURE=1 to create it)",
                sc.name
            );
            continue;
        }
        errorln!("--- scenario: {} ---", sc.name);
        let run = run_scenario(&aws_cfg, &sc, &corpus, dim).await?;
        errorln!(
            "  ingest: {:.1} ms ({} vectors), first_search: {:.1} ms, p50: {:.2} ms, qps: {:.1}, recall@10_mean: {:?}, answer@10_mean: {:?}, support@10_mean: {:?}, recall_dist: {:?}, mmap_snapshot: {} B, prewarm: {}",
            run.ingest_ms,
            run.vectors_ingested,
            run.first_search_ms,
            run.warmed_search.p50_ms,
            run.qps,
            run.recall_at_k,
            run.answer_at_k,
            run.support_at_k,
            run.recall_distribution,
            run.mmap_snapshot_bytes,
            run.untimed_prewarm_search
        );
        write_run_json(&run)?;
    }

    Ok(())
}
