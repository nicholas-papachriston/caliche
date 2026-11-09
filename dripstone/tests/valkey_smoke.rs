#![cfg(feature = "redis-backend")]

mod common;

use common::build_chunk;
use dripstone::backend::redis::RedisBackend;
use dripstone::search::Filter;
use dripstone::{DripstoneError, Metric, SearchQuery, traits::Backend};
use std::sync::Arc;
use time::OffsetDateTime;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn valkey_backend_smoke() -> Result<(), DripstoneError> {
    let url = std::env::var("VALKEY_URL").unwrap_or_else(|_| common::valkey_url_local());
    let namespace = format!(
        "valkey-func-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let backend: Arc<dyn Backend> =
        Arc::new(RedisBackend::with_prefix(url.as_str(), namespace.as_str()).await?);

    backend.delete_namespace(&namespace).await?;

    let chunk = build_chunk("doc-42".to_string(), &namespace, 2, 6);
    backend.upsert(&[chunk]).await?;

    let total = backend.count(Some(&namespace)).await?;
    assert_eq!(total, 1);

    let query = SearchQuery::new(vec![0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
        .metric(Metric::Cosine)
        .limit(5)
        .namespace(&namespace);
    let neighbours = backend.search(&query).await?;
    assert!(
        neighbours.iter().any(|entry| entry.chunk.id == "doc-42#0"),
        "expected inserted record to appear in similarity search"
    );

    let query_filtered = SearchQuery::new(vec![0.001, 0.002, 0.003, 0.004, 0.005, 0.006])
        .metric(Metric::Cosine)
        .limit(5)
        .namespace(&namespace)
        .filter(Filter::Equal {
            field: "segment".to_string(),
            value: serde_json::json!("alpha"),
        });
    let filtered = backend.search(&query_filtered).await?;
    assert_eq!(filtered.len(), 1);

    backend.delete_namespace(&namespace).await?;

    Ok(())
}
