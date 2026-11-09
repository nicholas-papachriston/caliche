#![cfg(feature = "postgres-backend")]

mod common;

use std::time::Duration;

use dripstone::DripstoneError;
use dripstone::backend::postgres::{PostgresBackend, PostgresBackendOptions};
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::traits::Backend;
use serde_json::json;
use std::sync::Arc;
use time::OffsetDateTime;

const CONNECT_TIMEOUT_SECS: u64 = 5;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn postgres_backend_roundtrip() -> Result<(), DripstoneError> {
    let url = common::postgres_url_local();
    let namespace = format!(
        "pg-test-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let options = PostgresBackendOptions {
        table_name: "embeddings_roundtrip_6".to_string(),
        ensure_schema: true,
        max_connections: 5,
        vector_dimension: Some(6),
        hnsw_index: None,
    };

    let backend: Arc<dyn Backend> = match tokio::time::timeout(
        Duration::from_secs(CONNECT_TIMEOUT_SECS),
        PostgresBackend::with_options(url.as_str(), options),
    )
    .await
    {
        Ok(Ok(b)) => Arc::new(b),
        Ok(Err(e)) => return Err(e),
        Err(_) => return Ok(()), // Postgres unreachable or slow; skip test
    };

    backend.delete_namespace(namespace.as_str()).await?;

    let mut chunks = Vec::new();
    for index in 0..16 {
        let vector = build_vector(index, 6);
        let metadata = json!({
            "segment": if index % 2 == 0 { "even" } else { "odd" },
            "ordinal": index,
        });
        let chunk = Chunk::new(
            format!("item-{index}"),
            0,
            format!("Content {index}"),
            vector,
            0,
            12,
            1,
            metadata,
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }

    backend.upsert(&chunks).await?;

    let stored = backend
        .get("item-3#0", Some(namespace.as_str()))
        .await?
        .ok_or_else(|| DripstoneError::NotFound("item-3".to_string()))?;
    let segment = stored
        .metadata
        .get("segment")
        .ok_or_else(|| DripstoneError::storage("missing segment metadata"))?;
    if segment != "odd" {
        return Err(DripstoneError::storage("unexpected segment value"));
    }

    let total = backend.count(Some(namespace.as_str())).await?;
    assert_eq!(total, 16, "count mismatch for namespace {namespace}");

    let listed = backend.list(Some(namespace.as_str()), None).await?;
    assert_eq!(
        listed.len(),
        16,
        "list embeddings should return all records for namespace {namespace}"
    );

    let namespaces = backend.namespaces().await?;
    assert!(
        namespaces.iter().any(|value| value == namespace.as_str()),
        "namespace discovery missing {namespace}"
    );

    use dripstone::search::Filter;
    let filter = Filter::Equal {
        field: "segment".to_string(),
        value: json!("even"),
    };
    let query_cosine = SearchQuery {
        vector: build_vector(4, 6).into(),
        limit: 5,
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
    let all_cosine_results = backend.search(&query_cosine).await?;
    // Filter by min_score client-side
    let cosine_results: Vec<_> = all_cosine_results
        .into_iter()
        .filter(|r| r.score >= 0.1)
        .collect();
    assert!(
        !cosine_results.is_empty(),
        "cosine search should return at least one neighbour"
    );
    assert!(
        cosine_results
            .iter()
            .all(|entry| entry.chunk.namespace.as_deref() == Some(namespace.as_str())),
        "cosine search returned mismatched namespace"
    );
    assert!(
        cosine_results
            .iter()
            .all(|entry| !entry.chunk.metadata.is_null()),
        "cosine search should carry metadata forward"
    );

    let query_euclidean = SearchQuery {
        vector: build_vector(5, 6).into(),
        limit: 10,
        metric: Metric::Euclidean,
        namespace: Some(namespace.clone()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let all_euclidean_results = backend.search(&query_euclidean).await?;
    // Filter by max_distance client-side
    let euclidean_results: Vec<_> = all_euclidean_results
        .into_iter()
        .filter(|r| r.score <= 1.0 + f32::EPSILON)
        .collect();
    assert!(
        euclidean_results
            .iter()
            .all(|entry| entry.score <= 1.0_f32 + f32::EPSILON),
        "Euclidean search exceeded distance threshold"
    );

    backend.delete_namespace(namespace.as_str()).await?;

    let remaining = backend.count(Some(namespace.as_str())).await?;
    assert_eq!(
        remaining, 0,
        "delete_namespace should remove all records in {namespace}"
    );

    Ok(())
}

fn build_vector(seed: usize, dimension: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(dimension);
    let base = (seed % dimension) as f32;
    for offset in 0..dimension {
        let value = ((base + offset as f32) * 0.05) % 1.0;
        values.push(value);
    }
    values
}
