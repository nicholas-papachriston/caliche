//! Soft-skip DynamoDB native vector integration coverage.
//!
//! Env (any of these unlocks the suite):
//! - `DRIPSTONE_DYNAMODB_TABLE` (required for live run)
//! - `DRIPSTONE_DYNAMODB_INDEX` (default `embeddings`)
//! - `AWS_REGION` / `DRIPSTONE_DYNAMODB_REGION`
//! - `DRIPSTONE_DYNAMODB_ENDPOINT` for LocalStack
//! - `DRIPSTONE_DYNAMODB_ENSURE=1` to create table/index
//! - `DRIPSTONE_DYNAMODB_FORCE=1` to attempt even when describe fails
//!
//! ```text
//! cargo test -p dripstone --test dynamodb_vector_tests \
//!   --features "dynamodb-backend,memory-backend,tokio" -- --test-threads=1
//! ```

#![cfg(all(
    feature = "dynamodb-backend",
    feature = "memory-backend",
    feature = "tokio"
))]

use std::env;
use std::sync::Arc;

use dripstone::backend::dynamodb::{DynamoDbVectorBackend, DynamoDbVectorConfig};
use dripstone::{
    Backend, Chunk, MemoryBackend, Metric, SearchQuery, TierConfig, TieredChainBuilder,
};
use serde_json::json;
use time::OffsetDateTime;

struct DdbEnv {
    table: String,
    index: String,
    region: String,
    endpoint: Option<String>,
    ensure: bool,
}

fn ddb_env() -> DdbEnv {
    let table = env::var("DRIPSTONE_DYNAMODB_TABLE")
        .expect("DRIPSTONE_DYNAMODB_TABLE must be set when this ignored test runs");
    DdbEnv {
        table,
        index: env::var("DRIPSTONE_DYNAMODB_INDEX").unwrap_or_else(|_| "embeddings".to_string()),
        region: env::var("DRIPSTONE_DYNAMODB_REGION")
            .or_else(|_| env::var("AWS_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string()),
        endpoint: env::var("DRIPSTONE_DYNAMODB_ENDPOINT").ok(),
        ensure: env::var("DRIPSTONE_DYNAMODB_ENSURE").ok().as_deref() == Some("1"),
    }
}

async fn connect(env: &DdbEnv, dimension: usize) -> Result<DynamoDbVectorBackend, String> {
    let mut config = DynamoDbVectorConfig::new(&env.table, &env.index, dimension)
        .with_distance_metric(Metric::Cosine)
        .with_region(&env.region)
        .with_ensure_table(env.ensure)
        .with_ensure_index(env.ensure);
    if let Some(ref endpoint) = env.endpoint {
        config = config.with_endpoint_url(endpoint);
    }
    DynamoDbVectorBackend::new(config)
        .await
        .map_err(|e| e.to_string())
}

fn sample_chunks(namespace: &str, dimension: usize) -> Vec<Chunk> {
    (0..5)
        .map(|i| {
            let mut vector = vec![0.01 * i as f32; dimension];
            vector[i % dimension] = 1.0;
            Chunk::new(
                format!("doc-{i}"),
                0,
                format!("content {i}"),
                vector,
                0,
                10,
                1,
                json!({ "i": i }),
                Some(namespace.to_string()),
            )
        })
        .collect()
}

#[ignore = "requires DRIPSTONE_DYNAMODB_TABLE and AWS credentials; run with --ignored"]
#[tokio::test]
async fn dynamodb_standalone_upsert_search_roundtrip() {
    let env = ddb_env();
    let dimension = 8usize;
    let backend = connect(&env, dimension).await.expect("dynamodb connect");

    let namespace = format!("vf-ddb-{}", OffsetDateTime::now_utc().unix_timestamp());
    let chunks = sample_chunks(&namespace, dimension);
    backend.upsert(&chunks).await.expect("upsert");

    let got = backend
        .get(&chunks[0].id, Some(&namespace))
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.content, chunks[0].content);

    let hits = backend
        .search(
            &SearchQuery::new(chunks[0].vector.to_vec())
                .namespace(&namespace)
                .metric(Metric::Cosine)
                .limit(3),
        )
        .await
        .expect("search");
    assert!(!hits.is_empty());
    assert_eq!(hits[0].chunk.id, chunks[0].id);

    let count = backend.count(Some(&namespace)).await.expect("count");
    assert_eq!(count, chunks.len());

    backend.delete_namespace(&namespace).await.expect("cleanup");
}

#[ignore = "requires DRIPSTONE_DYNAMODB_TABLE and AWS credentials; run with --ignored"]
#[tokio::test]
async fn dynamodb_tiered_with_memory() {
    let env = ddb_env();
    let dimension = 8usize;
    let cold = Arc::new(connect(&env, dimension).await.expect("dynamodb connect"));
    let hot = Arc::new(MemoryBackend::new());
    let backend: Arc<dyn Backend> = TieredChainBuilder::new()
        .tier(cold, TierConfig::new("dynamodb"))
        .tier(hot, TierConfig::new("memory").max_vectors(64))
        .background_sync(false)
        .build()
        .expect("tiered");

    let namespace = format!("vf-ddb-tier-{}", OffsetDateTime::now_utc().unix_timestamp());
    let chunks = sample_chunks(&namespace, dimension);
    backend.upsert(&chunks).await.expect("upsert");
    let hits = backend
        .search(
            &SearchQuery::new(chunks[1].vector.to_vec())
                .namespace(&namespace)
                .metric(Metric::Cosine)
                .limit(2),
        )
        .await
        .expect("search");
    assert!(!hits.is_empty());
    backend.delete_namespace(&namespace).await.expect("cleanup");
}
