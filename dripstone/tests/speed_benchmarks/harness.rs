//! Shared benchmark harness for `speed_benchmarks` integration tests.
//!
//! Corpus vectors and metrics come from [`super::datasets::load_integration_corpus`] (same HDF5
//! pipeline as `tests/integration/datasets.rs`: `DRIPSTONE_INT_*` env vars).

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

#[cfg(any(
    feature = "mmap-backend",
    feature = "redis-backend",
    feature = "postgres-backend",
    feature = "qdrant-backend",
    feature = "chroma-backend"
))]
macro_rules! errorln {
    () => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
    ($($arg:tt)+) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use super::common::{
    dripstone_bench_backends_wanted, dripstone_bench_long, dripstone_bench_max_points,
};
use super::datasets::DatasetCorpus;
use dripstone::backend::memory::MemoryBackend;
use dripstone::embedding::{EmbeddingRecord, SimilarityMetric};
use dripstone::index::{HnswConfig, HnswGraphSpace, HnswIndex, VectorIndex};
use dripstone::search::Filter;
use dripstone::{Chunk, DripstoneError, Metric, SearchQuery, traits::Backend};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};
use time::OffsetDateTime;

/// Percentile calculator for latency samples.
pub struct PercentileStats {
    samples: Vec<Duration>,
}

impl PercentileStats {
    pub const fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn add(&mut self, duration: Duration) {
        self.samples.push(duration);
    }

    pub fn calculate(&mut self) -> PercentileResult {
        self.samples.sort();
        let len = self.samples.len();

        if len == 0 {
            return PercentileResult::default();
        }

        let p50_idx = (len as f64 * 0.50) as usize;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;

        let sum: Duration = self.samples.iter().sum();
        let mean = sum / len as u32;

        let min = self.samples.first().copied().unwrap_or_default();
        let max = self.samples.last().copied().unwrap_or_default();

        PercentileResult {
            mean,
            p50: self.samples[p50_idx.min(len - 1)],
            p95: self.samples[p95_idx.min(len - 1)],
            p99: self.samples[p99_idx.min(len - 1)],
            min,
            max,
            count: len,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct PercentileResult {
    pub mean: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub min: Duration,
    pub max: Duration,
    pub count: usize,
}

impl PercentileResult {
    pub fn print(&self, operation: &str) {
        outputln!("\n{operation} Latency Statistics:");
        outputln!("  Samples:  {}", self.count);
        outputln!("  Mean:     {:?}", self.mean);
        outputln!("  Min:      {:?}", self.min);
        outputln!("  p50:      {:?}", self.p50);
        outputln!("  p95:      {:?}", self.p95);
        outputln!("  p99:      {:?}", self.p99);
        outputln!("  Max:      {:?}", self.max);
    }
}

/// Metadata filter shape for selectivity benchmarks.
#[derive(Debug, Clone, Copy)]
pub enum FilterSelectivity {
    None,
    /// ~50% match (`segment == "alpha"`).
    Half,
    /// ~1% match (`bucket == 0`).
    OnePercent,
}

/// One benchmark scenario (corpus shape + query workload).
#[derive(Debug, Clone)]
pub struct BenchmarkCase {
    pub name: &'static str,
    pub dimension: usize,
    pub corpus_size: usize,
    pub query_count: usize,
    pub limit: usize,
    pub metric: Metric,
    pub filter: FilterSelectivity,
}

impl BenchmarkCase {
    pub fn capped_by_max_points(&self, max_points: usize) -> Self {
        let corpus_size = self.corpus_size.min(max_points);
        let mut c = self.clone();
        c.corpus_size = corpus_size;
        c
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub subject: String,
    pub case: String,
    pub build_or_load: Duration,
    pub upsert: Option<Duration>,
    pub search: PercentileResult,
    pub qps: f64,
    pub recall_at_k: Option<f32>,
    pub index_serialized_bytes: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct HnswSweepConfig {
    pub label: String,
    pub config: HnswConfig,
}

pub fn print_summary_table(rows: &[BenchmarkSummary]) {
    outputln!("\nBenchmark summary");
    for r in rows {
        outputln!("  subject: {}", r.subject);
        outputln!("    case: {}", r.case);
        outputln!("    build_or_load: {:?}", r.build_or_load);
        outputln!("    upsert: {:?}", r.upsert);
        outputln!("    search p50: {:?}", r.search.p50);
        outputln!("    qps: {:.1}", r.qps);
        outputln!("    recall_at_k: {:?}", r.recall_at_k);
        outputln!("    index_serialized_bytes: {:?}", r.index_serialized_bytes);
    }
}

/// Intersection of the first `k` `approx` IDs with the exact top-`k` ID set, divided by `k`.
///
/// Expects `exact_ids` to be `k` **unique** IDs from a deterministic exact top-k run; duplicate
/// IDs in `exact_ids` shrink `exact_set` and trip `debug_assert` in debug builds.
pub fn recall_at_k(exact_ids: &[String], approx_ids: &[String]) -> f32 {
    let k = exact_ids.len();
    if k == 0 {
        return 0.0;
    }
    let exact_set: HashSet<&str> = exact_ids.iter().map(String::as_str).collect();
    debug_assert_eq!(
        exact_set.len(),
        k,
        "recall_at_k: exact top-k IDs must be unique (got {} distinct for k={k})",
        exact_set.len()
    );
    let hit = approx_ids
        .iter()
        .take(k)
        .filter(|id| exact_set.contains(id.as_str()))
        .count();
    hit as f32 / k as f32
}

pub fn metric_to_similarity(metric: Metric) -> Option<SimilarityMetric> {
    Some(SimilarityMetric::from(metric))
}

pub const fn tuned_hnsw_config() -> HnswConfig {
    HnswConfig {
        m: 32,
        ef_construction: 400,
        ef_search: 200,
        max_nb_connection: 32,
        rerank_overfetch: 4,
        graph_space: HnswGraphSpace::CosineCompatible,
    }
}

/// Same as [`tuned_hnsw_config`] but uses the corpus graph space (cosine vs L2) from the HDF5 URL / metric.
pub const fn tuned_hnsw_config_for_corpus(corpus: &DatasetCorpus) -> HnswConfig {
    let mut c = tuned_hnsw_config();
    c.graph_space = corpus.hnsw_graph_space;
    c
}

pub fn hnsw_sweep_configs() -> Vec<HnswSweepConfig> {
    let mut configs = Vec::new();
    for m in [16, 32] {
        for ef_search in [64, 128, 200, 400] {
            for rerank_overfetch in [2, 4, 8] {
                configs.push(HnswSweepConfig {
                    label: format!("m={m} ef_search={ef_search} overfetch={rerank_overfetch}"),
                    config: HnswConfig {
                        m,
                        ef_construction: 400,
                        ef_search,
                        max_nb_connection: m,
                        rerank_overfetch,
                        graph_space: HnswGraphSpace::CosineCompatible,
                    },
                });
            }
        }
    }
    configs
}

fn metadata_for_index(i: usize, filter: FilterSelectivity) -> Value {
    match filter {
        FilterSelectivity::None => json!({}),
        FilterSelectivity::Half => json!({
            "segment": if i.is_multiple_of(2) { "alpha" } else { "beta" },
            "ordinal": i,
        }),
        FilterSelectivity::OnePercent => json!({
            "segment": if i.is_multiple_of(2) { "alpha" } else { "beta" },
            "ordinal": i,
            "bucket": i32::from(!i.is_multiple_of(100)),
        }),
    }
}

fn corpus_query_arc(corpus: &DatasetCorpus, query_seed: usize) -> Arc<[f32]> {
    let nq = corpus.queries.len();
    let idx = if nq == 0 { 0 } else { query_seed % nq };
    Arc::from(corpus.queries[idx].clone().into_boxed_slice())
}

/// First `n` training vectors as chunks (`vec-0` ..), empty metadata (for micro-benchmarks).
pub fn corpus_train_chunks_simple(
    corpus: &DatasetCorpus,
    n: usize,
    namespace: &str,
) -> Result<Vec<Chunk>, DripstoneError> {
    if n > corpus.base.len() {
        return Err(DripstoneError::storage(format!(
            "requested {n} train rows but corpus has only {}",
            corpus.base.len()
        )));
    }
    let mut chunks = Vec::with_capacity(n);
    for i in 0..n {
        let vec_arc: Arc<[f32]> = Arc::from(corpus.base[i].clone().into_boxed_slice());
        chunks.push(Chunk::new(
            format!("vec-{i}"),
            0,
            "",
            vec_arc,
            0,
            0,
            1,
            json!({}),
            Some(namespace.to_string()),
        ));
    }
    Ok(chunks)
}

/// Build upsert rows from the integration HDF5 corpus (train slice + metadata filters).
pub fn chunks_for_case_from_corpus(
    case: &BenchmarkCase,
    namespace: &str,
    corpus: &DatasetCorpus,
) -> Result<Vec<Chunk>, DripstoneError> {
    if case.dimension != corpus.dimension {
        return Err(DripstoneError::storage(format!(
            "benchmark dimension {} != corpus dimension {}",
            case.dimension, corpus.dimension
        )));
    }
    if case.metric != corpus.search_metric {
        return Err(DripstoneError::storage(format!(
            "benchmark metric {:?} != corpus search metric {:?}",
            case.metric, corpus.search_metric
        )));
    }
    if case.corpus_size > corpus.base.len() {
        return Err(DripstoneError::storage(format!(
            "case corpus_size {} exceeds loaded train rows {}",
            case.corpus_size,
            corpus.base.len()
        )));
    }
    let mut chunks = Vec::with_capacity(case.corpus_size);
    for i in 0..case.corpus_size {
        let meta = metadata_for_index(i, case.filter);
        let vec_arc: Arc<[f32]> = Arc::from(corpus.base[i].clone().into_boxed_slice());
        chunks.push(Chunk::new(
            format!("vec-{i}"),
            0,
            "",
            vec_arc,
            0,
            0,
            1,
            meta,
            Some(namespace.to_string()),
        ));
    }
    Ok(chunks)
}

fn search_filter_for_case(case: &BenchmarkCase) -> Option<Filter> {
    match case.filter {
        FilterSelectivity::None => None,
        FilterSelectivity::Half => Some(Filter::Equal {
            field: "segment".to_string(),
            value: json!("alpha"),
        }),
        FilterSelectivity::OnePercent => Some(Filter::Equal {
            field: "bucket".to_string(),
            value: json!(0),
        }),
    }
}

fn make_search_query_from_corpus(
    case: &BenchmarkCase,
    namespace: &str,
    query_seed: usize,
    corpus: &DatasetCorpus,
) -> SearchQuery {
    let qv = corpus_query_arc(corpus, query_seed);
    let mut q = SearchQuery::new(qv)
        .metric(case.metric)
        .limit(case.limit)
        .namespace(namespace);
    if let Some(f) = search_filter_for_case(case) {
        q = q.filter(f);
    }
    q
}

/// Upsert, warm up, measure repeated searches on a [`Backend`] using integration HDF5 vectors.
pub async fn bench_backend_case_with_corpus(
    subject: &str,
    backend: Arc<dyn Backend>,
    case: &BenchmarkCase,
    namespace: &str,
    corpus: &DatasetCorpus,
) -> Result<BenchmarkSummary, DripstoneError> {
    let chunks = chunks_for_case_from_corpus(case, namespace, corpus)?;
    let upsert_start = Instant::now();
    backend.upsert(&chunks).await?;
    let upsert = upsert_start.elapsed();

    for warm in 0..5 {
        let q = make_search_query_from_corpus(case, namespace, warm * 17 + 3, corpus);
        let _ = backend.search(&q).await?;
    }

    let mut stats = PercentileStats::new();
    let search_start = Instant::now();
    for i in 0..case.query_count {
        let q = make_search_query_from_corpus(case, namespace, i * 7 + 11, corpus);
        let t0 = Instant::now();
        let _ = backend.search(&q).await?;
        stats.add(t0.elapsed());
    }
    let search_wall = search_start.elapsed();
    let search = stats.calculate();
    let qps = case.query_count as f64 / search_wall.as_secs_f64().max(1e-9);

    Ok(BenchmarkSummary {
        subject: subject.to_string(),
        case: case.name.to_string(),
        build_or_load: Duration::ZERO,
        upsert: Some(upsert),
        search,
        qps,
        recall_at_k: None,
        index_serialized_bytes: None,
    })
}

pub fn chunks_to_embedding_records(chunks: &[Chunk]) -> Vec<EmbeddingRecord> {
    chunks
        .iter()
        .map(|c| {
            let ns = c.namespace.as_deref().map(std::sync::Arc::from);
            let mut r = EmbeddingRecord::from_arc(c.id.as_str(), Arc::clone(&c.vector));
            r.namespace = ns;
            r.metadata = Some(c.metadata.clone());
            r
        })
        .collect()
}

/// Exact vs HNSW on integration corpus vectors; `hc.graph_space` is overridden from `corpus`.
///
/// Recall here compares approximate HNSW IDs to **runtime exact search** on the same upserted
/// chunks (`MemoryBackend::search`), not to HDF5 `neighbors` (integration scenarios may use
/// `DatasetCorpus::ground_truth` from a different source; see `CorpusDiagnostics::ground_truth_source`).
pub async fn bench_exact_vs_hnsw_with_corpus(
    case: &BenchmarkCase,
    namespace: &str,
    corpus: &DatasetCorpus,
    mut hc: HnswConfig,
) -> Result<(BenchmarkSummary, BenchmarkSummary, f32), DripstoneError> {
    if !matches!(case.filter, FilterSelectivity::None) {
        return Err(DripstoneError::storage(
            "bench_exact_vs_hnsw_with_corpus requires FilterSelectivity::None".to_string(),
        ));
    }

    let sim = metric_to_similarity(case.metric).ok_or_else(|| {
        DripstoneError::storage("benchmark: unsupported metric for HNSW comparison".to_string())
    })?;

    hc.graph_space = corpus.hnsw_graph_space;

    let chunks = chunks_for_case_from_corpus(case, namespace, corpus)?;
    let memory: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let upsert_start = Instant::now();
    memory.upsert(&chunks).await?;
    let upsert_mem = upsert_start.elapsed();

    let records = chunks_to_embedding_records(&chunks);
    let index = HnswIndex::new(hc, case.dimension);
    let build_start = Instant::now();
    index.build(&records).await?;
    let build_dur = build_start.elapsed();

    let ser_bytes = index.to_bytes_async().await?.len();

    let mut exact_stats = PercentileStats::new();
    let mut hnsw_stats = PercentileStats::new();
    let mut recalls = Vec::with_capacity(case.query_count);
    let mut hnsw_wall = Duration::ZERO;

    for i in 0..case.query_count {
        let q = make_search_query_from_corpus(case, namespace, i * 7 + 11, corpus);
        let qv = q.query().to_vec();

        let t0 = Instant::now();
        let exact = memory.search(&q).await?;
        exact_stats.add(t0.elapsed());
        let exact_ids: Vec<String> = exact.iter().map(|r| r.chunk.id.clone()).collect();

        let t1 = Instant::now();
        let approx = index.search(&qv, case.limit, sim).await?;
        let hnsw_elapsed = t1.elapsed();
        hnsw_wall += hnsw_elapsed;
        hnsw_stats.add(hnsw_elapsed);
        let approx_ids: Vec<String> = approx.iter().map(|r| r.record.id.to_string()).collect();
        recalls.push(recall_at_k(&exact_ids, &approx_ids));
    }

    let mean_recall = if recalls.is_empty() {
        0.0
    } else {
        recalls.iter().sum::<f32>() / recalls.len() as f32
    };

    let exact_search = exact_stats.calculate();
    let hnsw_search = hnsw_stats.calculate();
    let qps_exact = case.query_count as f64 / exact_search.mean.as_secs_f64().max(1e-12);
    let qps_hnsw = case.query_count as f64 / hnsw_wall.as_secs_f64().max(1e-9);

    let exact_summary = BenchmarkSummary {
        subject: "memory_exact".to_string(),
        case: case.name.to_string(),
        build_or_load: Duration::ZERO,
        upsert: Some(upsert_mem),
        search: exact_search,
        qps: qps_exact,
        recall_at_k: Some(1.0),
        index_serialized_bytes: None,
    };

    let hnsw_summary = BenchmarkSummary {
        subject: "hnsw".to_string(),
        case: case.name.to_string(),
        build_or_load: build_dur,
        upsert: None,
        search: hnsw_search,
        qps: qps_hnsw,
        recall_at_k: Some(mean_recall),
        index_serialized_bytes: Some(ser_bytes),
    };

    Ok((exact_summary, hnsw_summary, mean_recall))
}

/// Smoke- and long-tier cases derived from the loaded integration corpus (same caps as `DRIPSTONE_INT_TRAIN_MAX`).
pub fn benchmark_cases_from_corpus(
    corpus: &DatasetCorpus,
    max_points: usize,
    long: bool,
) -> Vec<BenchmarkCase> {
    let dim = corpus.dimension;
    let metric = corpus.search_metric;
    let n = corpus.base.len();
    let q_avail = corpus.queries.len().max(1);
    if n < 10 {
        return Vec::new();
    }
    let n_cap = n.min(max_points).max(10);

    let mut out = vec![
        BenchmarkCase {
            name: "smoke_1k_integration",
            dimension: dim,
            corpus_size: n_cap.min(1000),
            query_count: q_avail.min(50).max(1),
            limit: 10,
            metric,
            filter: FilterSelectivity::None,
        },
        BenchmarkCase {
            name: "smoke_filter_half",
            dimension: dim,
            corpus_size: n_cap.min(2000).max(20),
            query_count: q_avail.min(30).max(1),
            limit: 10,
            metric,
            filter: FilterSelectivity::Half,
        },
        BenchmarkCase {
            name: "smoke_filter_rare",
            dimension: dim,
            corpus_size: n_cap.min(5000).max(100),
            query_count: q_avail.min(25).max(1),
            limit: 10,
            metric,
            filter: FilterSelectivity::OnePercent,
        },
    ];

    if long {
        let q100 = q_avail.min(100).max(1);
        out.push(BenchmarkCase {
            name: "long_10k_integration",
            dimension: dim,
            corpus_size: n_cap.min(10_000).max(100),
            query_count: q100,
            limit: 10,
            metric,
            filter: FilterSelectivity::None,
        });
        out.push(BenchmarkCase {
            name: "long_50k_integration",
            dimension: dim,
            corpus_size: n_cap.min(50_000).max(100),
            query_count: q_avail.min(50).max(1),
            limit: 10,
            metric,
            filter: FilterSelectivity::None,
        });
        out.push(BenchmarkCase {
            name: "long_100k_integration",
            dimension: dim,
            corpus_size: n_cap.min(100_000).max(100),
            query_count: q_avail.min(40).max(1),
            limit: 10,
            metric,
            filter: FilterSelectivity::None,
        });
    }
    out
}

/// Targets 10K / 50K / 100K train rows, clamped to the loaded corpus; deduplicates when the cap is lower.
pub fn hnsw_scale_cases_from_corpus(
    corpus: &DatasetCorpus,
    max_points: usize,
) -> Vec<BenchmarkCase> {
    let dim = corpus.dimension;
    let metric = corpus.search_metric;
    let n_cap = corpus.base.len().min(max_points);
    let q_avail = corpus.queries.len().max(1);
    let q50 = q_avail.min(50).max(5);
    let q35 = q_avail.min(35).max(5);
    let q25 = q_avail.min(25).max(5);

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for (name, target, qc) in [
        ("scale_10k_integration", 10_000_usize, q50),
        ("scale_50k_integration", 50_000, q35),
        ("scale_100k_integration", 100_000, q25),
    ] {
        let size = n_cap.min(target).max(1000);
        if !seen.insert(size) {
            continue;
        }
        out.push(BenchmarkCase {
            name,
            dimension: dim,
            corpus_size: size,
            query_count: qc,
            limit: 10,
            metric,
            filter: FilterSelectivity::None,
        });
    }
    out
}

/// Applies [`super::common::dripstone_bench_max_points`] so 100K runs only when cap allows.
pub fn active_hnsw_scale_cases(corpus: &DatasetCorpus) -> Vec<BenchmarkCase> {
    let max_pts = dripstone_bench_max_points();
    hnsw_scale_cases_from_corpus(corpus, max_pts)
        .into_iter()
        .filter(|c| c.corpus_size >= 1000)
        .collect()
}

#[inline]
pub fn duration_to_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

pub fn print_ann_scale(
    case: &BenchmarkCase,
    exact: &BenchmarkSummary,
    hnsw: &BenchmarkSummary,
    recall: f32,
) {
    let brute_ms = duration_to_ms(exact.search.p50);
    let ann_ms = duration_to_ms(hnsw.search.p50);
    let speedup = if ann_ms > 1.0e-9 {
        brute_ms / ann_ms
    } else {
        f64::INFINITY
    };
    outputln!();
    outputln!("case: {}", case.name);
    outputln!("vectors: {}", case.corpus_size);
    outputln!("dimension: {}", case.dimension);
    outputln!("brute_force_search_p50_ms: {brute_ms:.3}");
    outputln!("hnsw_search_p50_ms: {ann_ms:.3}");
    outputln!("speedup_p50: {speedup:.1}");
    outputln!(
        "recall_at_{}_mean_vs_runtime_exact: {recall:.3}",
        case.limit
    );
    outputln!("hnsw_qps_hnsw_search_only: {:.1}", hnsw.qps);
    outputln!("hnsw_build: {:?}", hnsw.build_or_load);
    outputln!(
        "hnsw_index_serialized_bytes: {}",
        hnsw.index_serialized_bytes.unwrap_or(0)
    );
}

pub fn print_hnsw_sweep_row(
    case: &BenchmarkCase,
    config_label: &str,
    hnsw: &BenchmarkSummary,
    recall: f32,
) {
    outputln!();
    outputln!("case: {}", case.name);
    outputln!("config: {config_label}");
    outputln!("hnsw_search_p50_ms: {:.3}", duration_to_ms(hnsw.search.p50));
    outputln!(
        "recall_at_{}_mean_vs_runtime_exact: {recall:.3}",
        case.limit
    );
    outputln!("hnsw_qps_hnsw_search_only: {:.1}", hnsw.qps);
    outputln!("hnsw_build: {:?}", hnsw.build_or_load);
    outputln!(
        "hnsw_index_serialized_bytes: {}",
        hnsw.index_serialized_bytes.unwrap_or(0)
    );
}

pub fn active_benchmark_cases(corpus: &DatasetCorpus) -> Vec<BenchmarkCase> {
    let max_pts = dripstone_bench_max_points();
    benchmark_cases_from_corpus(corpus, max_pts, dripstone_bench_long())
        .into_iter()
        .map(|c| c.capped_by_max_points(max_pts))
        .filter(|c| c.corpus_size >= 10 && c.query_count > 0)
        .collect()
}

pub fn concurrency_task_counts() -> Vec<usize> {
    if dripstone_bench_long() {
        vec![1, 2, 4, 8, 16, 32]
    } else {
        vec![1, 2, 4, 8]
    }
}

/// Timings from [`bench_memory_vs_mmap`].
#[cfg(all(feature = "memory-backend", feature = "mmap-backend"))]
#[derive(Debug, Clone)]
pub struct MemoryMmapBenchStats {
    pub n: usize,
    pub dimension: usize,
    pub mmap_layout: String,
    pub memory_upsert: Duration,
    pub memory_search_p50: Duration,
    pub mmap_cold_open: Duration,
    pub mmap_first_search: Duration,
    pub mmap_search_p50: Duration,
    pub memory_p50_div_mmap_p50: f64,
}

#[cfg(all(feature = "memory-backend", feature = "mmap-backend"))]
async fn bench_memory_vs_mmap_inner(
    corpus: &DatasetCorpus,
    take_n: usize,
    namespace: &str,
    mmap_layout: &str,
    file_name: &str,
    encoding: Option<dripstone::backend::mmap::MmapVectorEncoding>,
) -> Result<MemoryMmapBenchStats, DripstoneError> {
    use dripstone::backend::mmap::{
        MmapBackend, MmapBackendConfig, write_dripstone_mmap, write_dripstone_mmap_with_encoding,
    };
    use tempfile::tempdir;

    let n = take_n.min(corpus.base.len()).max(1);
    let dimension = corpus.dimension;

    let chunks: Vec<Chunk> = (0..n)
        .map(|i| {
            let vec_arc: Arc<[f32]> = Arc::from(corpus.base[i].clone().into_boxed_slice());
            Chunk::new(
                format!("doc-{i}"),
                0,
                "",
                vec_arc,
                0,
                0,
                1,
                json!({}),
                Some(namespace.to_string()),
            )
        })
        .collect();

    let memory: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let mem_ins = Instant::now();
    memory.upsert(&chunks).await?;
    let mem_upsert = mem_ins.elapsed();

    let q = SearchQuery::new(corpus_query_arc(corpus, 0))
        .metric(corpus.search_metric)
        .limit(10)
        .namespace(namespace);
    for _ in 0..5 {
        let _ = memory.search(&q).await?;
    }
    let mut mem_warm = PercentileStats::new();
    for _ in 0..30 {
        let t = Instant::now();
        let _ = memory.search(&q).await?;
        mem_warm.add(t.elapsed());
    }
    let mem_p50 = mem_warm.calculate().p50;

    let dir = tempdir().map_err(|e| DripstoneError::storage(format!("tempdir: {e}")))?;
    let path = dir.path().join(file_name);
    match encoding {
        Some(encoding) => write_dripstone_mmap_with_encoding(&path, &chunks, encoding)?,
        None => write_dripstone_mmap(&path, &chunks)?,
    }

    let open_t = Instant::now();
    let mmap_backend = MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), dimension))?;
    let mmap_open = open_t.elapsed();

    let s0 = Instant::now();
    let _ = mmap_backend.search(&q).await?;
    let mmap_first = s0.elapsed();

    let mut mmap_warm = PercentileStats::new();
    for _ in 0..30 {
        let t = Instant::now();
        let _ = mmap_backend.search(&q).await?;
        mmap_warm.add(t.elapsed());
    }
    let mmap_avg = mmap_warm.calculate().p50;

    outputln!("\nMemory vs mmap (mmap layout: {mmap_layout})");
    outputln!("  vectors: {n}");
    outputln!("  dimension: {dimension}");
    outputln!("  memory_upsert: {mem_upsert:?}");
    outputln!("  memory_search_p50 (30 samples): {mem_p50:?}");
    outputln!("  mmap_cold_open: {mmap_open:?}");
    outputln!("  mmap_first_search: {mmap_first:?}");
    outputln!("  mmap_search_p50 (30 samples): {mmap_avg:?}");
    let r_warm = mem_p50.as_secs_f64() / mmap_avg.as_secs_f64().max(1e-12);
    outputln!("  memory_p50_div_mmap_p50: {r_warm:.2}");

    let stats = MemoryMmapBenchStats {
        n,
        dimension,
        mmap_layout: mmap_layout.to_string(),
        memory_upsert: mem_upsert,
        memory_search_p50: mem_p50,
        mmap_cold_open: mmap_open,
        mmap_first_search: mmap_first,
        mmap_search_p50: mmap_avg,
        memory_p50_div_mmap_p50: r_warm,
    };

    drop(mmap_backend);
    drop(dir);
    let _ = memory.delete_namespace(namespace).await;
    Ok(stats)
}

/// Run memory vs mmap on identical corpus (both features required at compile time).
#[cfg(all(feature = "memory-backend", feature = "mmap-backend"))]
pub async fn bench_memory_vs_mmap(
    corpus: &DatasetCorpus,
    take_n: usize,
    namespace: &str,
) -> Result<MemoryMmapBenchStats, DripstoneError> {
    let mmap_layout = {
        #[cfg(target_arch = "aarch64")]
        {
            "v2_f32_default".to_string()
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            "v2_f16_default".to_string()
        }
    };
    bench_memory_vs_mmap_inner(
        corpus,
        take_n,
        namespace,
        &mmap_layout,
        "bench_compare.vf1",
        None,
    )
    .await
}

/// Run memory vs mmap on identical corpus with an explicit mmap encoding.
#[cfg(all(feature = "memory-backend", feature = "mmap-backend"))]
pub async fn bench_memory_vs_mmap_with_encoding(
    corpus: &DatasetCorpus,
    take_n: usize,
    namespace: &str,
    mmap_layout: &str,
    file_name: &str,
    encoding: dripstone::backend::mmap::MmapVectorEncoding,
) -> Result<MemoryMmapBenchStats, DripstoneError> {
    bench_memory_vs_mmap_inner(
        corpus,
        take_n,
        namespace,
        mmap_layout,
        file_name,
        Some(encoding),
    )
    .await
}

/// Optional backend matrix: only runs backends listed in `DRIPSTONE_BENCH_BACKENDS`.
///
/// Returns an empty vector when the env var is unset (skipped).
pub async fn run_backend_matrix_if_configured() -> Result<Vec<BenchmarkSummary>, DripstoneError> {
    let wanted = dripstone_bench_backends_wanted();
    if wanted.is_empty() {
        outputln!("(skip backend matrix: set DRIPSTONE_BENCH_BACKENDS=memory,mmap,redis,... )");
        return Ok(Vec::new());
    }

    let corpus = super::datasets::load_integration_corpus().await?;
    let max_pts = dripstone_bench_max_points();
    let n_cap = corpus.base.len().min(max_pts).min(500).max(10);
    let case = BenchmarkCase {
        name: "matrix_small",
        dimension: corpus.dimension,
        corpus_size: n_cap,
        query_count: corpus.queries.len().min(20).max(1),
        limit: 10,
        metric: corpus.search_metric,
        filter: FilterSelectivity::None,
    };

    let ns_base = format!(
        "bench-matrix-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let mut summaries = Vec::new();
    run_backend_matrix_inner(&wanted, &case, &ns_base, &mut summaries, &corpus).await?;
    print_summary_table(&summaries);
    Ok(summaries)
}

async fn run_backend_matrix_inner(
    wanted: &HashSet<String>,
    case: &BenchmarkCase,
    ns_base: &str,
    summaries: &mut Vec<BenchmarkSummary>,
    corpus: &DatasetCorpus,
) -> Result<(), DripstoneError> {
    if wanted.contains("memory") {
        #[cfg(feature = "memory-backend")]
        {
            let ns = format!("{ns_base}-memory");
            let b: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
            let s = bench_backend_case_with_corpus("memory", b.clone(), case, &ns, corpus).await?;
            summaries.push(s);
            let _ = b.delete_namespace(&ns).await;
        }
    }

    #[cfg(all(feature = "memory-backend", feature = "mmap-backend"))]
    if wanted.contains("mmap") {
        use dripstone::backend::mmap::{
            MmapBackend, MmapBackendConfig, MmapVectorEncoding, write_dripstone_mmap,
            write_dripstone_mmap_with_encoding,
        };
        use tempfile::tempdir;

        let ns = format!("{ns_base}-mmap");
        let chunks = chunks_for_case_from_corpus(case, &ns, corpus)?;
        let variants = [
            #[cfg(target_arch = "aarch64")]
            ("mmap_v1_f32_default", "matrix_v1_default.vf1"),
            #[cfg(not(target_arch = "aarch64"))]
            ("mmap_v1_f16_default", "matrix_v1_default.vf1"),
            ("mmap_v1_f32_explicit", "matrix_v1_f32.vf1"),
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
                other => {
                    return Err(DripstoneError::storage(format!(
                        "unknown mmap benchmark variant: {other}"
                    )));
                }
            }
            let open = Instant::now();
            match MmapBackend::new(MmapBackendConfig::new(
                path.to_string_lossy(),
                case.dimension,
            )) {
                Ok(mmap_b) => {
                    let load = open.elapsed();
                    let mut stats = PercentileStats::new();
                    let wall = Instant::now();
                    for i in 0..case.query_count {
                        let q = make_search_query_from_corpus(case, &ns, i * 7 + 11, corpus);
                        let t0 = Instant::now();
                        let _ = mmap_b.search(&q).await?;
                        stats.add(t0.elapsed());
                    }
                    let w = wall.elapsed();
                    let search = stats.calculate();
                    let qps = case.query_count as f64 / w.as_secs_f64().max(1e-9);
                    summaries.push(BenchmarkSummary {
                        subject: variant.to_string(),
                        case: case.name.to_string(),
                        build_or_load: load,
                        upsert: None,
                        search,
                        qps,
                        recall_at_k: None,
                        index_serialized_bytes: None,
                    });
                    drop(mmap_b);
                }
                Err(e) => errorln!("{variant} backend matrix skip: {e}"),
            }
            drop(dir);
        }
    }

    #[cfg(feature = "redis-backend")]
    if wanted.contains("redis") || wanted.contains("valkey") {
        use dripstone::backend::redis::RedisBackend;
        if wanted.contains("redis") {
            let url =
                std::env::var("REDIS_URL").unwrap_or_else(|_| super::common::redis_url_local());
            let ns = format!("{ns_base}-redis");
            match RedisBackend::with_prefix(&url, &ns).await {
                Ok(r) => {
                    let b: Arc<dyn Backend> = Arc::new(r);
                    if let Ok(s) =
                        bench_backend_case_with_corpus("redis", b.clone(), case, &ns, corpus).await
                    {
                        summaries.push(s);
                    }
                    let _ = b.delete_namespace(&ns).await;
                }
                Err(e) => errorln!("redis matrix skip: {e}"),
            }
        }
        if wanted.contains("valkey") {
            let url =
                std::env::var("VALKEY_URL").unwrap_or_else(|_| super::common::valkey_url_local());
            let ns = format!("{ns_base}-valkey");
            match RedisBackend::with_prefix(&url, &ns).await {
                Ok(r) => {
                    let b: Arc<dyn Backend> = Arc::new(r);
                    if let Ok(s) =
                        bench_backend_case_with_corpus("valkey", b.clone(), case, &ns, corpus).await
                    {
                        summaries.push(s);
                    }
                    let _ = b.delete_namespace(&ns).await;
                }
                Err(e) => errorln!("valkey matrix skip: {e}"),
            }
        }
    }

    #[cfg(feature = "postgres-backend")]
    if wanted.contains("postgres") {
        use dripstone::backend::postgres::{PostgresBackend, PostgresBackendOptions};
        let url = super::common::postgres_url_local();
        let ns = format!("{ns_base}-pg");
        let opts = PostgresBackendOptions {
            table_name: "embeddings_speed_matrix".to_string(),
            ensure_schema: true,
            max_connections: 4,
            vector_dimension: Some(case.dimension as i32),
            hnsw_index: None,
        };
        match PostgresBackend::with_options(&url, opts).await {
            Ok(p) => {
                let b: Arc<dyn Backend> = Arc::new(p);
                if let Ok(s) =
                    bench_backend_case_with_corpus("postgres", b.clone(), case, &ns, corpus).await
                {
                    summaries.push(s);
                }
                let _ = b.delete_namespace(&ns).await;
            }
            Err(e) => errorln!("postgres matrix skip: {e}"),
        }
    }

    #[cfg(feature = "qdrant-backend")]
    if wanted.contains("qdrant") || wanted.contains("qdrant_http") {
        use dripstone::backend::qdrant::{QdrantBackendOptions, QdrantHttpBackend};
        let url = std::env::var("EMBEDDING_QDRANT_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:6333".to_string());
        let mut options = QdrantBackendOptions::new(url, case.dimension);
        options.collection_prefix = Some("speed_mx_".to_string());
        options.api_key = super::common::qdrant_api_key();
        let ns = format!("{ns_base}-qdrant");
        match QdrantHttpBackend::with_options(options).await {
            Ok(q) => {
                let b: Arc<dyn Backend> = Arc::new(q);
                if let Ok(s) =
                    bench_backend_case_with_corpus("qdrant_http", b.clone(), case, &ns, corpus)
                        .await
                {
                    summaries.push(s);
                }
                let _ = b.delete_namespace(&ns).await;
            }
            Err(e) => errorln!("qdrant_http matrix skip: {e}"),
        }
    }

    #[cfg(feature = "chroma-backend")]
    if wanted.contains("chroma") {
        use dripstone::backend::chroma::{ChromaBackend, ChromaBackendOptions};
        let url = std::env::var("EMBEDDING_CHROMA_URL")
            .unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
        let options = ChromaBackendOptions {
            base_url: url,
            api_key: super::common::chroma_api_key(),
            collection_prefix: Some("speed_mx_".to_string()),
            expected_dimension: Some(case.dimension),
            timeout: None,
            batch_size: None,
            retry_config: None,
        };
        let ns = format!("{ns_base}-chroma");
        match ChromaBackend::with_options(options).await {
            Ok(c) => {
                let b: Arc<dyn Backend> = Arc::new(c);
                if let Ok(s) =
                    bench_backend_case_with_corpus("chroma", b.clone(), case, &ns, corpus).await
                {
                    summaries.push(s);
                }
                let _ = b.delete_namespace(&ns).await;
            }
            Err(e) => errorln!("chroma matrix skip: {e}"),
        }
    }

    Ok(())
}
