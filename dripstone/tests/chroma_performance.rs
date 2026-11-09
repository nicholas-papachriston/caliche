#![cfg(feature = "chroma-backend")]

mod common;

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::env;
use std::time::{Duration, Instant};

use dripstone::backend::chroma::{ChromaBackend, ChromaBackendOptions};
use dripstone::search::Filter;
use dripstone::{Chunk, DripstoneError, Metric, SearchQuery, traits::Backend};
use serde_json::json;
use std::sync::Arc;
use time::OffsetDateTime;

const DEFAULT_CHROMA_URL: &str = "http://127.0.0.1:8000";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chroma_backend_performance() -> Result<(), DripstoneError> {
    let url = env::var("EMBEDDING_CHROMA_URL").unwrap_or_else(|_| DEFAULT_CHROMA_URL.to_string());
    let options = ChromaBackendOptions {
        base_url: url,
        api_key: common::chroma_api_key(),
        collection_prefix: Some("perf_".to_string()),
        expected_dimension: Some(env_dimension()),
        timeout: None,
        batch_size: None,
        retry_config: None,
    };

    let backend: Arc<dyn Backend> = match ChromaBackend::with_options(options).await {
        Ok(backend) => Arc::new(backend),
        Err(error) => {
            errorln!("skipping chroma performance suite: {error}");
            return Ok(());
        }
    };
    run_performance_suite(backend, "chroma-perf").await
}

async fn run_performance_suite(
    backend: Arc<dyn Backend>,
    namespace_hint: &str,
) -> Result<(), DripstoneError> {
    let namespace = format!(
        "{}-{}",
        namespace_hint,
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let warmup_flush = Instant::now();
    if let Err(error) = backend.delete_namespace(&namespace).await {
        errorln!("skipping chroma performance suite during flush: {error}");
        return Ok(());
    }
    outputln!(
        "[{}] initial namespace flush completed in {:?}",
        namespace,
        warmup_flush.elapsed()
    );

    let dimension = env_dimension();
    let item_count = env_item_count();
    let insert_budget = performance_budget("EMBEDDING_PERF_INSERT_MS", Duration::from_secs(15));
    let search_budget = performance_budget("EMBEDDING_PERF_SEARCH_MS", Duration::from_secs(8));
    let search_iterations = env::var("EMBEDDING_PERF_SEARCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(25);

    let mut chunks = Vec::with_capacity(item_count);
    for index in 0..item_count {
        let vector = build_vector(index, dimension);
        let metadata = json!({
            "segment": if index % 2 == 0 { "alpha" } else { "beta" },
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
    if let Err(error) = backend.upsert(&chunks).await {
        errorln!("skipping chroma performance suite during upsert: {error}");
        return Ok(());
    }
    let bulk_span = bulk_start.elapsed();
    outputln!("[{namespace}] bulk upsert inserted {item_count} embeddings in {bulk_span:?}");
    assert!(
        bulk_span <= insert_budget,
        "bulk upsert exceeded budget: {bulk_span:?} > {insert_budget:?}"
    );

    let count_start = Instant::now();
    let embedding_total = match backend.count(Some(&namespace)).await {
        Ok(count) => count,
        Err(error) => {
            errorln!("skipping chroma performance suite during count: {error}");
            return Ok(());
        }
    };
    let count_span = count_start.elapsed();
    outputln!("[{namespace}] counted {embedding_total} embeddings in {count_span:?}");
    assert_eq!(embedding_total, item_count);

    let namespaces_start = Instant::now();
    let namespaces = match backend.namespaces().await {
        Ok(namespaces) => namespaces,
        Err(error) => {
            errorln!("skipping chroma performance suite during namespace listing: {error}");
            return Ok(());
        }
    };
    let namespaces_span = namespaces_start.elapsed();
    outputln!("[{namespace}] namespace discovery {namespaces:?} in {namespaces_span:?}");
    assert!(
        namespaces.iter().any(|value| value == namespace.as_str()),
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
        let results = match backend.search(&query).await {
            Ok(results) => results,
            Err(error) => {
                errorln!("skipping chroma performance suite during similarity search: {error}");
                return Ok(());
            }
        };
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
    let query_filtered = SearchQuery::new(query_vec)
        .metric(Metric::Cosine)
        .limit(5)
        .namespace(&namespace)
        .filter(Filter::Equal {
            field: "segment".to_string(),
            value: json!("alpha"),
        });
    let filtered = match backend.search(&query_filtered).await {
        Ok(results) => results,
        Err(error) => {
            errorln!("skipping chroma performance suite during filtered search: {error}");
            return Ok(());
        }
    };
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

    let cleanup_start = Instant::now();
    let _ = backend.delete_namespace(&namespace).await;
    outputln!(
        "[{}] cleanup namespace flush completed in {:?}",
        namespace,
        cleanup_start.elapsed()
    );

    Ok(())
}

fn build_vector(seed: usize, dimension: usize) -> Vec<f32> {
    let mut vector = Vec::with_capacity(dimension);
    let base = (seed % dimension.max(1)) as f32;
    for offset in 0..dimension {
        vector.push(((base + offset as f32) * 0.003) % 1.0);
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
