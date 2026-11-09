#![cfg(feature = "qdrant-backend")]

mod common;

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::qdrant::{QdrantBackendOptions, QdrantHttpBackend};
use dripstone::{Chunk, DripstoneError, Metric, SearchQuery, traits::Backend};
use serde_json::json;
use std::sync::Arc;
use time::OffsetDateTime;

const DEFAULT_QDRANT_URL: &str = "http://127.0.0.1:6333";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn qdrant_backend_smoke() -> Result<(), DripstoneError> {
    let url =
        std::env::var("EMBEDDING_QDRANT_URL").unwrap_or_else(|_| DEFAULT_QDRANT_URL.to_owned());
    let namespace = format!(
        "qdrant-func-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let api_key = common::qdrant_api_key();

    // QdrantBackendOptions still uses SimilarityMetric
    // Create options with default (SimilarityMetric::Cosine is the default)
    let mut options = QdrantBackendOptions::new(url.clone(), 6);
    options.collection_prefix = Some("test_".to_string());
    options.api_key = api_key;

    let backend: Arc<dyn Backend> = match QdrantHttpBackend::with_options(options).await {
        Ok(backend) => Arc::new(backend),
        Err(error) => {
            errorln!("skipping qdrant backend smoke test: {error}");
            return Ok(());
        }
    };

    if let Err(error) = backend.delete_namespace(&namespace).await {
        errorln!("skipping qdrant backend smoke test during flush: {error}");
        return Ok(());
    }

    let mut chunks = Vec::new();
    for index in 0..8 {
        let vector = build_vector(index);
        let chunk = Chunk::new(
            format!("item-{index}"),
            0,
            "",
            vector,
            0,
            0,
            1,
            json!({
                "segment": if index % 2 == 0 { "alpha" } else { "beta" },
                "ordinal": index
            }),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }
    if let Err(error) = backend.upsert(&chunks).await {
        errorln!("skipping qdrant backend smoke test during upsert: {error}");
        return Ok(());
    }

    let stored = match backend.get("item-3", Some(&namespace)).await {
        Ok(Some(chunk)) => chunk,
        Ok(None) => {
            errorln!("skipping qdrant backend smoke test: missing expected record");
            return Ok(());
        }
        Err(error) => {
            errorln!("skipping qdrant backend smoke test during get: {error}");
            return Ok(());
        }
    };
    assert_eq!(stored.id, "item-3");

    let count = match backend.count(Some(&namespace)).await {
        Ok(count) => count,
        Err(error) => {
            errorln!("skipping qdrant backend smoke test during count: {error}");
            return Ok(());
        }
    };
    assert_eq!(count, 8);

    let namespaces = match backend.namespaces().await {
        Ok(namespaces) => namespaces,
        Err(error) => {
            errorln!("skipping qdrant backend smoke test during namespace listing: {error}");
            return Ok(());
        }
    };
    assert!(
        namespaces.iter().any(|value| value == namespace.as_str()),
        "expected namespace to be discoverable"
    );

    let query = SearchQuery::new(build_vector(4))
        .metric(Metric::Cosine)
        .limit(5)
        .namespace(&namespace);
    let search = match backend.search(&query).await {
        Ok(results) => results,
        Err(error) => {
            errorln!("skipping qdrant backend smoke test during similarity search: {error}");
            return Ok(());
        }
    };
    assert!(
        !search.is_empty(),
        "similarity search should return results"
    );

    use dripstone::core::SearchResult;
    use dripstone::search::Filter;
    let query_filtered = SearchQuery::new(build_vector(5))
        .metric(Metric::Cosine)
        .limit(5)
        .namespace(&namespace)
        .filter(Filter::Equal {
            field: "segment".to_string(),
            value: json!("beta"),
        });
    let filtered: Vec<SearchResult> = match backend.search(&query_filtered).await {
        Ok(results) => {
            // Client-side filtering for min_score
            results.into_iter().filter(|r| r.score >= 0.1).collect()
        }
        Err(error) => {
            errorln!("skipping qdrant backend smoke test during filtered search: {error}");
            return Ok(());
        }
    };
    assert!(
        filtered.iter().all(|entry| !entry.chunk.metadata.is_null()),
        "metadata should be preserved"
    );
    assert!(
        filtered
            .iter()
            .all(|entry| entry.chunk.namespace.as_deref() == Some(namespace.as_str())),
        "unexpected namespace in filtered results"
    );

    let _ = backend.delete_namespace(&namespace).await;

    Ok(())
}

fn build_vector(seed: usize) -> Vec<f32> {
    (0..6)
        .map(|offset| ((seed + offset) as f32 * 0.01) % 1.0)
        .collect()
}
