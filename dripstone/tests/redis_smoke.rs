#![cfg(feature = "redis-backend")]

mod common;

use common::build_chunk;
use dripstone::DripstoneError;
use dripstone::backend::redis::RedisBackend;
use dripstone::core::{Metric, SearchQuery};
use dripstone::traits::Backend;
use std::sync::Arc;
use time::OffsetDateTime;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn redis_backend_smoke() -> Result<(), DripstoneError> {
    let url = std::env::var("REDIS_URL").unwrap_or_else(|_| common::redis_url_local());
    let namespace = format!(
        "redis-func-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend: Arc<dyn Backend> =
        Arc::new(RedisBackend::with_prefix(&url, namespace.as_str()).await?);

    backend.delete_namespace(&namespace).await?;

    let chunk = build_chunk("doc-1".to_string(), &namespace, 1, 6);
    backend.upsert(&[chunk]).await?;

    let fetched = backend
        .get("doc-1#0", Some(&namespace))
        .await?
        .ok_or_else(|| DripstoneError::NotFound("doc-1".to_string()))?;
    assert_eq!(fetched.id, "doc-1#0");

    let listed = backend.list(Some(&namespace), None).await?;
    assert_eq!(listed.len(), 1);

    let query = SearchQuery {
        vector: Arc::from(vec![0.001, 0.002, 0.003, 0.004, 0.005, 0.006].into_boxed_slice()),
        limit: 3,
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
    let search_results = backend.search(&query).await?;
    assert!(!search_results.is_empty());

    backend.delete_namespace(namespace.as_str()).await?;

    Ok(())
}
