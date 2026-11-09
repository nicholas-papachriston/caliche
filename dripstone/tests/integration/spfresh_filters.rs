macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dripstone::backend::indexed::{IndexedBackend, IndexedBackendConfig};
use dripstone::backend::memory::MemoryBackend;
use dripstone::index::spfresh::{CentroidIndex, SPFreshConfig};
use dripstone::search::Filter;
use dripstone::{Chunk, DripstoneError, Metric, SearchQuery, SearchResult, traits::Backend};
use serde_json::json;

use super::metrics::write_named_json;

const TOPIC_COUNT: usize = 16;
const SEGMENT_COUNT: usize = 4;
const RECORDS_PER_CLUSTER: usize = 18;
const DIMENSION: usize = TOPIC_COUNT * 2 + SEGMENT_COUNT;
const LIMIT: usize = 10;
const LATENCY_REPEATS: usize = 4;

const SEGMENTS: [&str; SEGMENT_COUNT] = ["alpha", "beta", "gamma", "delta"];

#[derive(Debug, Default, Clone)]
struct LatencySummary {
    p50: Duration,
    p95: Duration,
    mean: Duration,
}

#[derive(Debug, Clone)]
struct BenchmarkRow {
    name: &'static str,
    native_recall_mean: f32,
    fallback_recall_mean: f32,
    native_latency: LatencySummary,
    fallback_latency: LatencySummary,
    same_recall_fallback: Option<ProbeSweepMatch>,
}

#[derive(Debug, Clone)]
struct ProbeSweepMatch {
    probes: usize,
    recall_mean: f32,
    latency: LatencySummary,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkArtifact {
    kind: &'static str,
    dataset: &'static str,
    dimension: usize,
    limit: usize,
    records_per_cluster: usize,
    initial_num_probes: usize,
    rows: Vec<BenchmarkArtifactRow>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkArtifactRow {
    name: &'static str,
    native_recall_mean: f32,
    fallback_same_budget_recall_mean: f32,
    native_latency_ms: LatencySummaryMs,
    fallback_same_budget_latency_ms: LatencySummaryMs,
    same_recall_fallback: Option<ProbeSweepArtifact>,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ProbeSweepArtifact {
    probes: usize,
    recall_mean: f32,
    latency_ms: LatencySummaryMs,
    p50_latency_penalty_vs_native: f64,
}

#[derive(Debug, Clone, serde::Serialize)]
struct LatencySummaryMs {
    p50_ms: f64,
    p95_ms: f64,
    mean_ms: f64,
}

fn normalize(mut vector: Vec<f32>) -> Result<Vec<f32>, DripstoneError> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return Err(DripstoneError::storage(
            "spfresh filter benchmark generated a zero vector".to_string(),
        ));
    }
    for value in &mut vector {
        *value /= norm;
    }
    Ok(vector)
}

fn topic_center(topic_idx: usize) -> Result<Vec<f32>, DripstoneError> {
    let mut vector = vec![0.0; DIMENSION];
    let base = topic_idx * 2;
    vector[base] = 1.0;
    vector[base + 1] = 0.55;
    normalize(vector)
}

fn segment_basis(segment_idx: usize) -> Vec<f32> {
    let mut vector = vec![0.0; DIMENSION];
    vector[TOPIC_COUNT * 2 + segment_idx] = 1.0;
    vector
}

fn mix_topic_and_segment(
    topic_idx: usize,
    segment_weights: [f32; SEGMENT_COUNT],
) -> Result<Vec<f32>, DripstoneError> {
    let mut vector = topic_center(topic_idx)?;
    for (segment_idx, weight) in segment_weights.into_iter().enumerate() {
        let basis = segment_basis(segment_idx);
        for (value, basis_value) in vector.iter_mut().zip(basis) {
            *value += basis_value * weight;
        }
    }
    normalize(vector)
}

fn cluster_vector(
    topic_idx: usize,
    segment_idx: usize,
    record_idx: usize,
) -> Result<Vec<f32>, DripstoneError> {
    let mut weights = [0.0; SEGMENT_COUNT];
    weights[segment_idx] = 0.75;
    let mut vector = mix_topic_and_segment(topic_idx, weights)?;
    let jitter_axis = (topic_idx + segment_idx + record_idx) % (TOPIC_COUNT * 2);
    let jitter = ((record_idx % 7) as f32 - 3.0) * 0.0035;
    vector[jitter_axis] += jitter;
    normalize(vector)
}

fn equal_query(topic_idx: usize) -> Result<Vec<f32>, DripstoneError> {
    mix_topic_and_segment(topic_idx, [0.22, 0.92, 0.78, 0.62])
}

fn in_query(topic_idx: usize) -> Result<Vec<f32>, DripstoneError> {
    mix_topic_and_segment(topic_idx, [0.44, 0.98, 0.72, 0.84])
}

fn build_chunks(namespace: &str) -> Result<Vec<Chunk>, DripstoneError> {
    let mut chunks = Vec::with_capacity(TOPIC_COUNT * SEGMENT_COUNT * RECORDS_PER_CLUSTER);
    for topic_idx in 0..TOPIC_COUNT {
        for (segment_idx, segment) in SEGMENTS.iter().enumerate() {
            for record_idx in 0..RECORDS_PER_CLUSTER {
                let vector = cluster_vector(topic_idx, segment_idx, record_idx)?;
                let vector_arc: Arc<[f32]> = Arc::from(vector.into_boxed_slice());
                chunks.push(Chunk::new(
                    format!("topic-{topic_idx}-{segment}-{record_idx}"),
                    0,
                    "",
                    vector_arc,
                    0,
                    0,
                    1,
                    json!({
                        "topic": topic_idx,
                        "segment": segment,
                        "always_one": 1,
                    }),
                    Some(namespace.to_string()),
                ));
            }
        }
    }
    Ok(chunks)
}

fn equal_filter() -> Filter {
    Filter::Equal {
        field: "segment".to_string(),
        value: json!("alpha"),
    }
}

fn in_filter() -> Filter {
    Filter::In {
        field: "segment".to_string(),
        values: vec![json!("alpha"), json!("gamma")],
    }
}

fn fallback_equivalent(filter: Filter) -> Filter {
    Filter::And {
        left: Box::new(filter),
        right: Box::new(Filter::GreaterThanOrEqual {
            field: "always_one".to_string(),
            value: json!(1),
        }),
    }
}

fn query_for(namespace: &str, vector: Vec<f32>, filter: Filter) -> SearchQuery {
    SearchQuery::new(Arc::from(vector.into_boxed_slice()))
        .metric(Metric::Cosine)
        .limit(LIMIT)
        .namespace(namespace)
        .filter(filter)
}

const fn spfresh_config(num_probes: usize) -> SPFreshConfig {
    SPFreshConfig {
        num_centroids: TOPIC_COUNT * SEGMENT_COUNT,
        max_iterations: 40,
        convergence_threshold: 0.0001,
        num_probes,
        metric: dripstone::embedding::SimilarityMetric::Cosine,
        max_cluster_records: None,
    }
}

async fn build_spfresh_backend(
    chunks: &[Chunk],
    num_probes: usize,
    overfetch_factor: usize,
) -> Result<IndexedBackend<MemoryBackend>, DripstoneError> {
    let spfresh = CentroidIndex::new(spfresh_config(num_probes), DIMENSION);
    let indexed_backend = IndexedBackend::new(
        MemoryBackend::new(),
        Box::new(spfresh),
        IndexedBackendConfig {
            auto_rebuild_threshold: 0,
            fallback_to_brute_force: true,
            overfetch_factor,
            ..Default::default()
        },
    );
    indexed_backend.upsert(chunks).await?;
    indexed_backend.rebuild_index().await?;
    Ok(indexed_backend)
}

fn recall_at_k(exact: &[SearchResult], approx: &[SearchResult], k: usize) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let exact_ids: HashSet<&str> = exact
        .iter()
        .take(k)
        .map(|row| row.chunk.id.as_str())
        .collect();
    let hit = approx
        .iter()
        .take(k)
        .filter(|row| exact_ids.contains(row.chunk.id.as_str()))
        .count();
    hit as f32 / k as f32
}

fn summarize_latencies(samples: &mut [Duration]) -> LatencySummary {
    samples.sort_unstable();
    let count = samples.len();
    if count == 0 {
        return LatencySummary::default();
    }
    let sum: Duration = samples.iter().copied().sum();
    LatencySummary {
        p50: samples[count / 2],
        p95: samples[((count as f64 * 0.95) as usize).min(count - 1)],
        mean: sum / count as u32,
    }
}

fn latency_ms(summary: &LatencySummary) -> LatencySummaryMs {
    LatencySummaryMs {
        p50_ms: summary.p50.as_secs_f64() * 1000.0,
        p95_ms: summary.p95.as_secs_f64() * 1000.0,
        mean_ms: summary.mean.as_secs_f64() * 1000.0,
    }
}

#[derive(Debug, Clone)]
struct Measurement {
    recall_mean: f32,
    latency: LatencySummary,
}

async fn measure_backend_filter(
    exact_backend: &MemoryBackend,
    indexed_backend: &IndexedBackend<MemoryBackend>,
    namespace: &str,
    filter: Filter,
    query_builder: fn(usize) -> Result<Vec<f32>, DripstoneError>,
) -> Result<Measurement, DripstoneError> {
    let mut recalls = Vec::with_capacity(TOPIC_COUNT);
    let mut samples = Vec::with_capacity(TOPIC_COUNT * LATENCY_REPEATS);

    for warmup_topic in 0..TOPIC_COUNT.min(4) {
        let query = query_builder(warmup_topic)?;
        let benchmark_query = query_for(namespace, query, filter.clone());
        let _ = indexed_backend.search(&benchmark_query).await?;
    }

    for topic_idx in 0..TOPIC_COUNT {
        let query = query_builder(topic_idx)?;
        let exact_query = query_for(namespace, query.clone(), filter.clone());
        let exact = exact_backend.search(&exact_query).await?;

        for repeat in 0..LATENCY_REPEATS {
            let benchmark_query = query_for(namespace, query.clone(), filter.clone());
            let start = Instant::now();
            let measured = indexed_backend.search(&benchmark_query).await?;
            samples.push(start.elapsed());
            if repeat == 0 {
                recalls.push(recall_at_k(&exact, &measured, LIMIT));
            }
        }
    }

    Ok(Measurement {
        recall_mean: recalls.iter().sum::<f32>() / recalls.len() as f32,
        latency: summarize_latencies(&mut samples),
    })
}

async fn sweep_same_recall_fallback(
    exact_backend: &MemoryBackend,
    chunks: &[Chunk],
    namespace: &str,
    fallback_filter_shape: Filter,
    query_builder: fn(usize) -> Result<Vec<f32>, DripstoneError>,
    target_recall_mean: f32,
) -> Result<Option<ProbeSweepMatch>, DripstoneError> {
    let max_probes = TOPIC_COUNT * SEGMENT_COUNT;
    let full_corpus_overfetch = chunks.len().div_ceil(LIMIT).max(64);
    for probes in [2_usize, 4, 8, 16, 24, 32, 48, max_probes] {
        if probes > max_probes {
            continue;
        }
        let backend = build_spfresh_backend(chunks, probes, full_corpus_overfetch).await?;
        let measurement = measure_backend_filter(
            exact_backend,
            &backend,
            namespace,
            fallback_filter_shape.clone(),
            query_builder,
        )
        .await?;
        if measurement.recall_mean + 0.01 >= target_recall_mean {
            return Ok(Some(ProbeSweepMatch {
                probes,
                recall_mean: measurement.recall_mean,
                latency: measurement.latency,
            }));
        }
    }
    Ok(None)
}

async fn measure_case(
    exact_backend: &MemoryBackend,
    indexed_backend: &IndexedBackend<MemoryBackend>,
    chunks: &[Chunk],
    namespace: &str,
    name: &'static str,
    native_filter: Filter,
    fallback_filter_shape: Filter,
    query_builder: fn(usize) -> Result<Vec<f32>, DripstoneError>,
) -> Result<BenchmarkRow, DripstoneError> {
    let native = measure_backend_filter(
        exact_backend,
        indexed_backend,
        namespace,
        native_filter,
        query_builder,
    )
    .await?;
    let fallback_same_budget = measure_backend_filter(
        exact_backend,
        indexed_backend,
        namespace,
        fallback_filter_shape.clone(),
        query_builder,
    )
    .await?;
    let same_recall_fallback = sweep_same_recall_fallback(
        exact_backend,
        chunks,
        namespace,
        fallback_filter_shape,
        query_builder,
        native.recall_mean,
    )
    .await?;

    Ok(BenchmarkRow {
        name,
        native_recall_mean: native.recall_mean,
        fallback_recall_mean: fallback_same_budget.recall_mean,
        native_latency: native.latency,
        fallback_latency: fallback_same_budget.latency,
        same_recall_fallback,
    })
}

pub async fn run() -> Result<(), DripstoneError> {
    fastrand::seed(7);

    let namespace = format!(
        "spfresh-filter-bench-{}",
        time::OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let chunks = build_chunks(&namespace)?;

    let exact_backend = MemoryBackend::new();
    exact_backend.upsert(&chunks).await?;

    let indexed_backend = build_spfresh_backend(&chunks, 2, 4).await?;

    let equal = measure_case(
        &exact_backend,
        &indexed_backend,
        &chunks,
        &namespace,
        "equal(segment=alpha)",
        equal_filter(),
        fallback_equivalent(equal_filter()),
        equal_query,
    )
    .await?;
    let in_case = measure_case(
        &exact_backend,
        &indexed_backend,
        &chunks,
        &namespace,
        "in(segment in [alpha,gamma])",
        in_filter(),
        fallback_equivalent(in_filter()),
        in_query,
    )
    .await?;

    for row in [&equal, &in_case] {
        let speedup = row.fallback_latency.p50.as_secs_f64() / row.native_latency.p50.as_secs_f64();
        outputln!("\nSPFresh filtered benchmark: {}", row.name);
        outputln!("  native recall@{LIMIT}: {:.3}", row.native_recall_mean);
        outputln!(
            "  fallback recall@{LIMIT} (same probe budget, no preselection): {:.3}",
            row.fallback_recall_mean
        );
        outputln!(
            "  native latency p50/p95/mean: {:?} / {:?} / {:?}",
            row.native_latency.p50,
            row.native_latency.p95,
            row.native_latency.mean
        );
        outputln!(
            "  fallback latency p50/p95/mean (same probe budget, no preselection): {:?} / {:?} / {:?}",
            row.fallback_latency.p50,
            row.fallback_latency.p95,
            row.fallback_latency.mean
        );
        outputln!("  p50 speedup from cluster preselection: {speedup:.2}x");
        if let Some(same_recall) = &row.same_recall_fallback {
            let penalty =
                same_recall.latency.p50.as_secs_f64() / row.native_latency.p50.as_secs_f64();
            outputln!(
                "  fallback same-recall target reached at num_probes={} with recall@{LIMIT} {:.3}",
                same_recall.probes,
                same_recall.recall_mean
            );
            outputln!(
                "  fallback same-recall latency p50/p95/mean: {:?} / {:?} / {:?}",
                same_recall.latency.p50,
                same_recall.latency.p95,
                same_recall.latency.mean
            );
            outputln!("  same-recall p50 latency penalty: {penalty:.2}x");
        } else {
            outputln!("  fallback same-recall target was not reached within num_probes sweep");
        }
    }

    assert!(
        equal.native_recall_mean >= 0.9,
        "equal filter native recall should stay high, got {:.3}",
        equal.native_recall_mean
    );
    assert!(
        in_case.native_recall_mean >= 0.85,
        "in filter native recall should stay high, got {:.3}",
        in_case.native_recall_mean
    );
    assert!(
        equal.native_recall_mean > equal.fallback_recall_mean + 0.25,
        "equal filter should improve recall materially ({:.3} vs {:.3})",
        equal.native_recall_mean,
        equal.fallback_recall_mean
    );
    assert!(
        in_case.native_recall_mean > in_case.fallback_recall_mean + 0.20,
        "in filter should improve recall materially ({:.3} vs {:.3})",
        in_case.native_recall_mean,
        in_case.fallback_recall_mean
    );
    assert!(
        equal
            .same_recall_fallback
            .as_ref()
            .is_some_and(|row| row.probes > 2 && row.recall_mean + 0.01 >= equal.native_recall_mean),
        "equal filter fallback should need a higher probe count to match native recall"
    );
    assert!(
        in_case.same_recall_fallback.as_ref().is_some_and(
            |row| row.probes > 2 && row.recall_mean + 0.01 >= in_case.native_recall_mean
        ),
        "in filter fallback should need a higher probe count to match native recall"
    );

    let artifact = BenchmarkArtifact {
        kind: "dripstone_spfresh_filtered_benchmark",
        dataset: "synthetic_clustered_segments",
        dimension: DIMENSION,
        limit: LIMIT,
        records_per_cluster: RECORDS_PER_CLUSTER,
        initial_num_probes: 2,
        rows: [&equal, &in_case]
            .into_iter()
            .map(|row| BenchmarkArtifactRow {
                name: row.name,
                native_recall_mean: row.native_recall_mean,
                fallback_same_budget_recall_mean: row.fallback_recall_mean,
                native_latency_ms: latency_ms(&row.native_latency),
                fallback_same_budget_latency_ms: latency_ms(&row.fallback_latency),
                same_recall_fallback: row.same_recall_fallback.as_ref().map(|fallback| {
                    ProbeSweepArtifact {
                        probes: fallback.probes,
                        recall_mean: fallback.recall_mean,
                        latency_ms: latency_ms(&fallback.latency),
                        p50_latency_penalty_vs_native: fallback.latency.p50.as_secs_f64()
                            / row.native_latency.p50.as_secs_f64(),
                    }
                }),
            })
            .collect(),
    };
    write_named_json("spfresh_filtered_benchmark", &artifact)?;

    exact_backend.delete_namespace(&namespace).await?;
    indexed_backend.delete_namespace(&namespace).await?;
    Ok(())
}
