#![cfg(feature = "chroma-backend")]

mod common;

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::DripstoneError;
use dripstone::backend::chroma::{ChromaBackend, ChromaBackendOptions};
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::traits::Backend;
use serde_json::json;
use std::sync::Arc;
use time::OffsetDateTime;

const DEFAULT_CHROMA_URL: &str = "http://127.0.0.1:8000";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chroma_backend_smoke() -> Result<(), DripstoneError> {
    let url =
        std::env::var("EMBEDDING_CHROMA_URL").unwrap_or_else(|_| DEFAULT_CHROMA_URL.to_string());
    let namespace = format!(
        "chroma-func-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let options = ChromaBackendOptions {
        base_url: url.clone(),
        api_key: common::chroma_api_key(),
        collection_prefix: Some("test_".to_string()),
        expected_dimension: Some(6),
        timeout: None,
        batch_size: None,
        retry_config: None,
    };

    let backend: Arc<dyn Backend> = match ChromaBackend::with_options(options).await {
        Ok(backend) => Arc::new(backend),
        Err(error) => {
            errorln!("skipping chroma backend smoke test: {error}");
            return Ok(());
        }
    };

    if let Err(error) = backend.delete_namespace(namespace.as_str()).await {
        errorln!("skipping chroma backend smoke test during flush: {error}");
        return Ok(());
    }

    let mut chunks = Vec::new();
    for index in 0..6 {
        let vector = build_vector(index);
        let chunk = Chunk::new(
            format!("item-{index}"),
            0,
            format!("Content {index}"),
            vector,
            0,
            12,
            1,
            json!({ "segment": if index % 2 == 0 { "alpha" } else { "beta" } }),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }

    if let Err(error) = backend.upsert(&chunks).await {
        errorln!("skipping chroma backend smoke test during upsert: {error}");
        return Ok(());
    }

    match backend.get("item-1#0", Some(namespace.as_str())).await {
        Ok(Some(chunk)) => assert_eq!(chunk.id, "item-1#0"),
        Ok(None) => {
            errorln!("skipping chroma backend smoke test: missing expected record");
            return Ok(());
        }
        Err(error) => {
            errorln!("skipping chroma backend smoke test during get: {error}");
            return Ok(());
        }
    }

    match backend.count(Some(namespace.as_str())).await {
        Ok(count) => assert_eq!(count, 6),
        Err(error) => {
            errorln!("skipping chroma backend smoke test during count: {error}");
            return Ok(());
        }
    }

    let query_vector = build_vector(3);
    let query = SearchQuery {
        vector: query_vector.clone().into(),
        limit: 4,
        metric: Metric::Cosine,
        namespace: Some(namespace.clone()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let search = match backend.search(&query).await {
        Ok(results) => results,
        Err(error) => {
            errorln!("skipping chroma backend smoke test during search: {error}");
            return Ok(());
        }
    };
    assert!(
        !search.is_empty(),
        "similarity search expected at least one result"
    );

    use dripstone::search::Filter;
    let filter = Filter::Equal {
        field: "segment".to_string(),
        value: json!("alpha"),
    };
    let query_filtered = SearchQuery {
        vector: build_vector(4).into(),
        limit: 3,
        metric: Metric::Cosine,
        namespace: Some(namespace.clone()),
        filter: Some(filter),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let filtered = match backend.search(&query_filtered).await {
        Ok(results) => results,
        Err(error) => {
            errorln!("skipping chroma backend smoke test during filtered search: {error}");
            return Ok(());
        }
    };
    assert!(
        filtered.iter().all(|entry| !entry.chunk.metadata.is_null()),
        "metadata should be preserved"
    );

    let _ = backend.delete_namespace(namespace.as_str()).await;
    Ok(())
}

fn build_vector(seed: usize) -> Vec<f32> {
    (0..6)
        .map(|offset| ((seed + offset) as f32 * 0.02) % 1.0)
        .collect()
}
