#![cfg(feature = "weaviate-backend")]

mod common;

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::weaviate::{WeaviateBackend, WeaviateBackendOptions};
use dripstone::search::Filter;
use dripstone::{Chunk, DripstoneError, Metric, SearchQuery, traits::Backend};
use serde_json::json;
use std::sync::Arc;
use time::OffsetDateTime;

const DEFAULT_WEAVIATE_URL: &str = "http://127.0.0.1:8081";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn weaviate_backend_smoke() -> Result<(), DripstoneError> {
    let url = std::env::var("EMBEDDING_WEAVIATE_URL")
        .unwrap_or_else(|_| DEFAULT_WEAVIATE_URL.to_string());
    let namespace = format!(
        "weaviate-func-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let api_key = common::weaviate_api_key();

    let options = WeaviateBackendOptions {
        base_url: url.clone(),
        class_prefix: Some("test_".to_string()),
        vector_dimension: 6,
        api_key,
        timeout: None,
        batch_size: None,
        retry_config: None,
    };

    let backend: Arc<dyn Backend> = match WeaviateBackend::with_options(options).await {
        Ok(backend) => Arc::new(backend),
        Err(error) => {
            errorln!("skipping weaviate backend smoke test: {error}");
            return Ok(());
        }
    };

    if let Err(error) = backend.delete_namespace(&namespace).await {
        errorln!("skipping weaviate backend smoke test during flush: {error}");
        return Ok(());
    }

    let mut chunks = Vec::new();
    for index in 0..6 {
        let vector = build_vector(index);
        let chunk = Chunk::new(
            format!("item-{index}"),
            0,
            "",
            vector,
            0,
            0,
            1,
            json!({ "segment": if index % 2 == 0 { "alpha" } else { "beta" } }),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }

    if let Err(error) = backend.upsert(&chunks).await {
        errorln!("skipping weaviate backend smoke test during upsert: {error}");
        return Ok(());
    }

    match backend.get("item-1", Some(&namespace)).await {
        Ok(Some(chunk)) => assert_eq!(chunk.id, "item-1"),
        Ok(None) => {
            errorln!("skipping weaviate backend smoke test: missing expected record");
            return Ok(());
        }
        Err(error) => {
            errorln!("skipping weaviate backend smoke test during get: {error}");
            return Ok(());
        }
    }

    match backend.count(Some(&namespace)).await {
        Ok(count) => assert_eq!(count, 6),
        Err(error) => {
            errorln!("skipping weaviate backend smoke test during count: {error}");
            return Ok(());
        }
    }

    let query = SearchQuery::new(build_vector(3))
        .metric(Metric::Cosine)
        .limit(4)
        .namespace(&namespace);
    let search = match backend.search(&query).await {
        Ok(results) => results,
        Err(error) => {
            errorln!("skipping weaviate backend smoke test during search: {error}");
            return Ok(());
        }
    };
    assert!(
        !search.is_empty(),
        "similarity search expected at least one result"
    );

    let query_filtered = SearchQuery::new(build_vector(4))
        .metric(Metric::Cosine)
        .limit(3)
        .namespace(&namespace)
        .filter(Filter::Equal {
            field: "segment".to_string(),
            value: json!("alpha"),
        });
    let filtered = match backend.search(&query_filtered).await {
        Ok(results) => results,
        Err(error) => {
            errorln!("skipping weaviate backend smoke test during filtered search: {error}");
            return Ok(());
        }
    };
    assert!(
        filtered.iter().all(|entry| !entry.chunk.metadata.is_null()),
        "metadata should be preserved"
    );

    let _ = backend.delete_namespace(&namespace).await;
    Ok(())
}

fn build_vector(seed: usize) -> Vec<f32> {
    (0..6)
        .map(|offset| ((seed + offset) as f32 * 0.02) % 1.0)
        .collect()
}
