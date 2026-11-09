#![cfg(feature = "redis-backend")]

mod common;

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::env;
use std::time::{Duration, Instant};

use dripstone::backend::redis::RedisBackend;
use dripstone::search::Filter;
use dripstone::{Chunk, DripstoneError, Metric, SearchQuery, traits::Backend};
use serde_json::json;
use serial_test::serial;
use std::sync::Arc;
use time::OffsetDateTime;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn redis_backend_performance() -> Result<(), DripstoneError> {
    let url = env::var("REDIS_URL").unwrap_or_else(|_| common::redis_url_local());
    run_performance_suite(&url, "redis-perf").await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[serial]
async fn valkey_backend_performance() -> Result<(), DripstoneError> {
    let url = env::var("VALKEY_URL").unwrap_or_else(|_| common::valkey_url_local());
    run_performance_suite(&url, "valkey-perf").await
}

async fn run_performance_suite(url: &str, namespace_hint: &str) -> Result<(), DripstoneError> {
    let backend: Arc<dyn Backend> = Arc::new(RedisBackend::with_prefix(url, namespace_hint).await?);
    let namespace = format!(
        "{}-{}",
        namespace_hint,
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    // Clean up old test namespaces to prevent performance degradation
    let cleanup_start = Instant::now();
    let all_namespaces = backend.namespaces().await?;
    for ns in &all_namespaces {
        if ns.starts_with(namespace_hint) && ns != &namespace {
            let _ = backend.delete_namespace(ns).await;
        }
    }
    outputln!(
        "[{}] cleaned up {} old test namespaces in {:?}",
        namespace,
        all_namespaces
            .iter()
            .filter(|ns| ns.starts_with(namespace_hint))
            .count(),
        cleanup_start.elapsed()
    );

    let warmup_flush = Instant::now();
    backend.delete_namespace(&namespace).await?;
    outputln!(
        "[{}] initial namespace flush completed in {:?}",
        namespace,
        warmup_flush.elapsed()
    );

    let dimension = env_dimension();
    let item_count = env_item_count();
    let insert_budget = performance_budget("EMBEDDING_PERF_INSERT_MS", Duration::from_secs(10));
    let search_budget = performance_budget("EMBEDDING_PERF_SEARCH_MS", Duration::from_secs(10));
    let search_iterations = env::var("EMBEDDING_PERF_SEARCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(25);

    let mut chunks = Vec::with_capacity(item_count);
    for index in 0..item_count {
        let vector = build_vector(index, dimension);
        let segment = if index % 2 == 0 { "alpha" } else { "beta" };
        let metadata = json!({
            "segment": segment,
            "ordinal": index,
        });
        chunks.push(Chunk::new(
            format!("item-{index}"),
            0,
            "",
            vector,
            0,
            0,
            1,
            metadata,
            Some(namespace.clone()),
        ));
    }

    let bulk_start = Instant::now();
    backend.upsert(&chunks).await?;
    let bulk_span = bulk_start.elapsed();
    outputln!("[{namespace}] bulk upsert inserted {item_count} embeddings in {bulk_span:?}");

    assert!(
        bulk_span <= insert_budget,
        "bulk upsert exceeded budget: {bulk_span:?} > {insert_budget:?}"
    );

    let count_start = Instant::now();
    let embedding_total = backend.count(Some(&namespace)).await?;
    let count_span = count_start.elapsed();
    outputln!("[{namespace}] counted {embedding_total} embeddings in {count_span:?}");
    assert_eq!(
        embedding_total, item_count,
        "counted embeddings mismatch expected {item_count}"
    );

    let namespaces_start = Instant::now();
    let namespaces = backend.namespaces().await?;
    let namespaces_span = namespaces_start.elapsed();
    outputln!("[{namespace}] namespace discovery {namespaces:?} in {namespaces_span:?}");
    assert!(
        namespaces.iter().any(|value| value == &namespace),
        "namespace {namespace} missing from enumeration"
    );

    let query_vec = build_vector(item_count / 2, dimension);
    let query = SearchQuery::new(query_vec.clone())
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(&namespace);
    let mut search_span = Duration::ZERO;
    for iteration in 0..search_iterations {
        let started = Instant::now();
        let results = backend.search(&query).await?;
        let elapsed = started.elapsed();
        search_span += elapsed;
        outputln!(
            "[{}] similarity search iteration {} took {:?}",
            namespace,
            iteration + 1,
            elapsed
        );
        assert!(
            !results.is_empty(),
            "similarity search returned no results for namespace {namespace}"
        );
    }
    outputln!(
        "[{namespace}] similarity search total {search_span:?} across {search_iterations} iterations"
    );

    assert!(
        search_span <= search_budget,
        "similarity search exceeded budget: {search_span:?} > {search_budget:?}"
    );

    let filtered_start = Instant::now();
    let filter_min_score = env_min_score();
    let query_filtered = SearchQuery::new(query_vec.clone())
        .metric(Metric::Cosine)
        .limit(5)
        .namespace(&namespace)
        .filter(Filter::Equal {
            field: "segment".to_string(),
            value: json!("alpha"),
        });
    let filtered: Vec<_> = backend
        .search(&query_filtered)
        .await?
        .into_iter()
        .filter(|r| r.score >= filter_min_score)
        .collect();
    let filtered_span = filtered_start.elapsed();
    outputln!(
        "[{}] filtered cosine search returned {} results in {:?}",
        namespace,
        filtered.len(),
        filtered_span
    );
    assert!(
        filtered.iter().all(|entry| !entry.chunk.metadata.is_null()),
        "filtered search should preserve metadata"
    );
    assert!(
        filtered
            .iter()
            .all(|entry| entry.chunk.namespace.as_deref() == Some(namespace.as_str())),
        "filtered search yielded unexpected namespace"
    );

    let radius_start = Instant::now();
    let radius_distance = env_max_distance();
    let query_euclidean = SearchQuery::new(query_vec)
        .metric(Metric::Euclidean)
        .limit(10)
        .namespace(&namespace);
    let euclidean: Vec<_> = backend
        .search(&query_euclidean)
        .await?
        .into_iter()
        .filter(|r| r.score <= radius_distance)
        .collect();
    let radius_span = radius_start.elapsed();
    outputln!(
        "[{}] Euclidean radius search returned {} results in {:?}",
        namespace,
        euclidean.len(),
        radius_span
    );
    assert!(
        !euclidean.is_empty(),
        "Euclidean radius search expected at least one neighbour"
    );
    assert!(
        euclidean
            .iter()
            .all(|entry| search_options_respects_distance(entry.score, radius_distance)),
        "Euclidean radius search exceeded distance budget"
    );

    let cleanup_start = Instant::now();
    backend.delete_namespace(&namespace).await?;
    outputln!(
        "[{}] cleanup namespace flush completed in {:?}",
        namespace,
        cleanup_start.elapsed()
    );
    Ok(())
}

fn build_vector(seed: usize, dimension: usize) -> Vec<f32> {
    let mut vector = Vec::with_capacity(dimension);
    let base = (seed % dimension) as f32;
    for offset in 0..dimension {
        vector.push(((base + offset as f32) * 0.001) % 1.0);
    }
    vector
}

fn env_dimension() -> usize {
    env::var("EMBEDDING_PERF_DIMENSION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(256)
}

fn env_item_count() -> usize {
    env::var("EMBEDDING_PERF_ITEM_COUNT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(200)
}

fn performance_budget(key: &str, default: Duration) -> Duration {
    env::var(key)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map_or(default, Duration::from_millis)
}

fn search_options_respects_distance(distance: f32, max_distance: f32) -> bool {
    distance <= max_distance
}

fn env_min_score() -> f32 {
    env::var("EMBEDDING_PERF_FILTER_MIN_SCORE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(0.2)
}

fn env_max_distance() -> f32 {
    env::var("EMBEDDING_PERF_RADIUS_MAX_DISTANCE")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(2.5)
}
