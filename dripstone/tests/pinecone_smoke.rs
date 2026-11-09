#![cfg(feature = "pinecone-backend")]

macro_rules! errorln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stderr().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::env;

use dripstone::backend::pinecone::{PineconeBackend, PineconeBackendOptions};
use dripstone::search::Filter;
use dripstone::{Chunk, DripstoneError, Metric, SearchQuery, traits::Backend};
use serde_json::json;
use std::sync::Arc;
use time::OffsetDateTime;

const DEFAULT_DIMENSION: usize = 6;

#[ignore = "requires EMBEDDING_PINECONE_HOST and EMBEDDING_PINECONE_API_KEY; run with --ignored"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pinecone_backend_smoke() -> Result<(), DripstoneError> {
    let host = if let Some(host) = env_var("EMBEDDING_PINECONE_HOST") {
        host
    } else {
        errorln!("skipping pinecone backend smoke test: EMBEDDING_PINECONE_HOST not set");
        return Err(DripstoneError::storage(
            "pinecone opt-in or connection failed".to_string(),
        ));
    };
    let api_key = if let Some(key) = env_var("EMBEDDING_PINECONE_API_KEY") {
        key
    } else {
        errorln!("skipping pinecone backend smoke test: EMBEDDING_PINECONE_API_KEY not set");
        return Err(DripstoneError::storage(
            "pinecone opt-in or connection failed".to_string(),
        ));
    };

    let dimension = env::var("EMBEDDING_PINECONE_DIMENSION")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_DIMENSION);

    let options = PineconeBackendOptions {
        base_url: host,
        api_key,
        dimension,
        timeout: None,
        batch_size: None,
        retry_config: None,
    };

    let backend: Arc<dyn Backend> = match PineconeBackend::with_options(options).await {
        Ok(backend) => Arc::new(backend),
        Err(error) => {
            errorln!("skipping pinecone backend smoke test: {error}");
            return Err(DripstoneError::storage(
                "pinecone opt-in or connection failed".to_string(),
            ));
        }
    };
    let namespace = format!(
        "pinecone-func-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    if let Err(error) = backend.delete_namespace(&namespace).await {
        errorln!("skipping pinecone backend smoke test during flush: {error}");
        return Err(DripstoneError::storage(
            "pinecone opt-in or connection failed".to_string(),
        ));
    }

    let mut chunks = Vec::new();
    for index in 0..dimension.max(6) {
        let vector = build_vector(index, dimension);
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
        errorln!("skipping pinecone backend smoke test during upsert: {error}");
        return Err(DripstoneError::storage(
            "pinecone opt-in or connection failed".to_string(),
        ));
    }

    match backend.get("item-1", Some(&namespace)).await {
        Ok(Some(chunk)) => assert_eq!(chunk.id, "item-1"),
        Ok(None) => {
            errorln!("skipping pinecone backend smoke test: missing expected record");
            return Err(DripstoneError::storage(
                "pinecone opt-in or connection failed".to_string(),
            ));
        }
        Err(error) => {
            errorln!("skipping pinecone backend smoke test during get: {error}");
            return Err(DripstoneError::storage(
                "pinecone opt-in or connection failed".to_string(),
            ));
        }
    }

    let count = match backend.count(Some(&namespace)).await {
        Ok(count) => count,
        Err(error) => {
            errorln!("skipping pinecone backend smoke test during count: {error}");
            return Err(DripstoneError::storage(
                "pinecone opt-in or connection failed".to_string(),
            ));
        }
    };
    assert_eq!(count, dimension.max(6));

    let query = SearchQuery::new(build_vector(2, dimension))
        .metric(Metric::Cosine)
        .limit(4)
        .namespace(&namespace);
    let search = match backend.search(&query).await {
        Ok(results) => results,
        Err(error) => {
            errorln!("skipping pinecone backend smoke test during search: {error}");
            return Err(DripstoneError::storage(
                "pinecone opt-in or connection failed".to_string(),
            ));
        }
    };
    assert!(
        !search.is_empty(),
        "pinecone similarity search returned no results"
    );

    let query_filtered = SearchQuery::new(build_vector(3, dimension))
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
            errorln!("skipping pinecone backend smoke test during filtered search: {error}");
            return Err(DripstoneError::storage(
                "pinecone opt-in or connection failed".to_string(),
            ));
        }
    };
    assert!(
        filtered.iter().all(|entry| !entry.chunk.metadata.is_null()),
        "metadata should be preserved"
    );

    let _ = backend.delete_namespace(&namespace).await;
    Ok(())
}

fn env_var(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(value) if !value.trim().is_empty() => Some(value),
        _ => None,
    }
}

fn build_vector(seed: usize, dimension: usize) -> Vec<f32> {
    (0..dimension)
        .map(|offset| ((seed + offset) as f32 * 0.01) % 1.0)
        .collect()
}
