//! DynamoDB native vectors: standalone + tiered with Memory.
//!
//! Demonstrates:
//! 1. `DynamoDbVectorBackend` as a standalone `Backend`
//! 2. The same backend as a cold tier in `TieredChainBuilder`
//! 3. Compatibility with client MaxSim fallback (`native_maxsim: false`)
//!
//! ## Prerequisites
//!
//! - AWS credentials with DynamoDB vector-index access
//! - Env:
//!   - `DRIPSTONE_DYNAMODB_TABLE` (required)
//!   - `DRIPSTONE_DYNAMODB_INDEX` (default `embeddings`)
//!   - `AWS_REGION` (default `us-east-1`)
//!   - `DRIPSTONE_DYNAMODB_ENSURE=1` to create table/index when missing
//!
//! ## Run
//!
//! ```bash
//! cargo run -p dripstone --example dynamodb_vector_tiered_example \
//!   --features "dynamodb-backend,memory-backend"
//! ```

#![cfg(all(feature = "dynamodb-backend", feature = "memory-backend"))]

macro_rules! outputln {
    () => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
    ($($arg:tt)+) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use std::env;
use std::sync::Arc;

use dripstone::backend::dynamodb::{DynamoDbVectorBackend, DynamoDbVectorConfig};
use dripstone::{
    Backend, Chunk, MemoryBackend, Metric, SearchQuery, TierConfig, TieredChainBuilder,
    search_with_maxsim_fallback,
};
use serde_json::json;
use time::OffsetDateTime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Ok(table) = env::var("DRIPSTONE_DYNAMODB_TABLE") else {
        outputln!(
            "SKIP: set DRIPSTONE_DYNAMODB_TABLE. SearchVectors uses https://search-dynamodb.{{region}}.api.aws unless DRIPSTONE_DYNAMODB_ENDPOINT or DRIPSTONE_DYNAMODB_SEARCH_ENDPOINT is set."
        );
        return Ok(());
    };
    if table.is_empty() {
        outputln!("SKIP: DRIPSTONE_DYNAMODB_TABLE is empty");
        return Ok(());
    }
    let index = env::var("DRIPSTONE_DYNAMODB_INDEX").unwrap_or_else(|_| "embeddings".to_string());
    let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let ensure = env::var("DRIPSTONE_DYNAMODB_ENSURE").ok().as_deref() == Some("1");
    let dimension = 8usize;

    let mut config = DynamoDbVectorConfig::new(&table, &index, dimension)
        .with_distance_metric(Metric::Cosine)
        .with_region(&region)
        .with_ensure_table(ensure)
        .with_ensure_index(ensure);

    if let Ok(endpoint) = env::var("DRIPSTONE_DYNAMODB_ENDPOINT") {
        config = config.with_endpoint_url(endpoint);
    }
    if let Ok(search_endpoint) = env::var("DRIPSTONE_DYNAMODB_SEARCH_ENDPOINT") {
        config = config.with_search_endpoint_url(search_endpoint);
    }

    outputln!("=== DynamoDB Vector Backend Example ===\n");
    outputln!("Table: {table}");
    outputln!("Index: {index}");
    outputln!("Region: {region}");
    outputln!(
        "SearchVectors host: https://search-dynamodb.{region}.api.aws when no endpoint override is set"
    );
    outputln!();

    let ddb: Arc<DynamoDbVectorBackend> = Arc::new(DynamoDbVectorBackend::new(config).await?);
    let namespace = format!("ddb-demo-{}", OffsetDateTime::now_utc().unix_timestamp());

    // Standalone upsert + search
    outputln!("--- Standalone Backend ---");
    let chunks = demo_chunks(&namespace, dimension);
    ddb.upsert(&chunks).await?;
    let missing = ddb.get("missing-id#0", Some(&namespace)).await?;
    outputln!("missing id get: {}", missing.is_some());
    if missing.is_some() {
        return Err("get of an id that was not written returned a chunk".into());
    }
    let query = SearchQuery::new(chunks[0].vector.to_vec())
        .namespace(&namespace)
        .metric(Metric::Cosine)
        .limit(3);
    let hits = search_until_visible(ddb.as_ref(), &query).await?;
    outputln!("standalone hits: {}", hits.len());
    for hit in &hits {
        outputln!("  {} score={:.4}", hit.chunk.id, hit.score);
    }

    // MaxSim client fallback (combinable method)
    let maxsim_hits = search_with_maxsim_fallback(ddb.as_ref(), &query).await?;
    outputln!("maxsim-fallback hits: {}", maxsim_hits.len());

    // Tiered: DynamoDB cold + Memory hot
    outputln!("\n--- TieredChainBuilder ---");
    let hot: Arc<MemoryBackend> = Arc::new(MemoryBackend::new());
    let tiered: Arc<dyn Backend> = TieredChainBuilder::new()
        .tier(ddb.clone(), TierConfig::new("dynamodb"))
        .tier(
            hot,
            TierConfig::new("memory")
                .max_vectors(32)
                .max_bytes(4 * 1024 * 1024),
        )
        .background_sync(false)
        .build()?;

    let tiered_ns = format!("{namespace}-tiered");
    tiered.upsert(&demo_chunks(&tiered_ns, dimension)).await?;
    let tiered_hits = tiered
        .search(
            &SearchQuery::new(vec![1.0; dimension])
                .namespace(&tiered_ns)
                .metric(Metric::Cosine)
                .limit(3),
        )
        .await?;
    outputln!("tiered hits: {}", tiered_hits.len());

    let _ = ddb.delete_namespace(&namespace).await;
    let _ = tiered.delete_namespace(&tiered_ns).await;
    outputln!("\nDone.");
    Ok(())
}

async fn search_until_visible(
    backend: &dyn Backend,
    query: &SearchQuery,
) -> Result<Vec<dripstone::SearchResult>, Box<dyn std::error::Error>> {
    const ATTEMPTS: usize = 8;
    let mut last = Vec::new();
    for attempt in 0..ATTEMPTS {
        last = backend.search(query).await?;
        if !last.is_empty() {
            return Ok(last);
        }
        if attempt + 1 < ATTEMPTS {
            tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        }
    }
    Err("SearchVectors stayed empty after upsert; refusing to treat that as success".into())
}

fn demo_chunks(namespace: &str, dimension: usize) -> Vec<Chunk> {
    (0..4)
        .map(|i| {
            let mut vector = vec![0.05 * i as f32; dimension];
            if let Some(slot) = vector.get_mut(i % dimension) {
                *slot = 1.0;
            }
            Chunk::new(
                format!("doc-{i}"),
                0,
                format!("chunk {i}"),
                vector,
                0,
                8,
                1,
                json!({ "i": i }),
                Some(namespace.to_string()),
            )
        })
        .collect()
}
