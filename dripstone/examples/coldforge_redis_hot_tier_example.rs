//! `ColdForge` with Redis Hot Tier Example
//!
//! Redis is a shared `get` cache. Search uses each process's SPFresh index.
//! Other processes see writes only after `flush` (or the WAL batch interval)
//! because they replay object-store segments. A hot-tier hit is ignored when
//! the namespace index does not contain that id.
//!
//! ## Architecture
//!
//! ```text
//! Search: in-process SPFresh index (id + vector), then the chunk-id body
//! Get:    local cache, then Redis, then the chunk-id body (index membership wins)
//! Durability: WAL segments on S3 after flush; envelopes at chunk-id keys
//! ```
//!
//! ## What this does not do
//!
//! - Search is not served from Redis.
//! - `upsert` returning does not mean the WAL segment is durable. This example calls `flush`.
//! - `WalCompactor` deletes a prefix only after `compact_wal` saves a membership
//!   snapshot and the dropped envelopes match chunk-id bodies.
//! - With a hot tier, `wal_refresh_interval_ms = 0` lists S3 on every get before
//!   Redis. This example keeps the default 1000 ms interval.
//!
//! Run with:
//! ```bash
//! cargo run --example coldforge_redis_hot_tier_example --features coldforge-backend,redis-backend
//! ```

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

use dripstone::backend::coldforge::{ColdForgeBackend, ColdForgeConfig};
use dripstone::backend::redis::RedisBackend;
use dripstone::storage::ObjectStoreConfig;
use dripstone::{Chunk, Metric, SearchQuery, traits::Backend};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("ColdForge with Redis Hot Tier Example\n");

    // Get Redis URL from environment
    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());

    let Ok(bucket) = std::env::var("COLDFORGE_BUCKET") else {
        outputln!(
            "SKIP: set COLDFORGE_BUCKET (for example example-bucket), AWS_REGION=us-east-1, and AWS_PROFILE=example"
        );
        return Ok(());
    };
    if bucket.is_empty() {
        outputln!("SKIP: COLDFORGE_BUCKET is empty");
        return Ok(());
    }
    let region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    outputln!("Configuration:");
    outputln!("  Hot Tier: Redis");
    outputln!("    URL: {redis_url}");
    outputln!("  Cold Tier: S3");
    outputln!("    Bucket: {bucket}");
    outputln!("    Region: {region}");
    outputln!("    wal_refresh_interval_ms: 1000");
    outputln!();

    // Create Redis backend for hot tier
    outputln!("Connecting to Redis hot tier...");
    let redis_backend: Arc<dyn Backend> =
        Arc::new(RedisBackend::with_prefix(redis_url.as_str(), "coldforge:hot:").await?);
    outputln!("Connected to Redis");

    // Configure ColdForge with Redis as hot tier
    let mut config = ColdForgeConfig::default();
    config.bucket = bucket;
    config.key_prefix = "coldforge/".to_string();
    config.region = region.clone();
    config.access_key_id = None;
    config.secret_access_key = None;
    config.session_token = None;
    config.object_store_config = Some(if std::env::var("OBJECT_STORE_ENDPOINT").is_ok() {
        ObjectStoreConfig::from_env()?
    } else {
        ObjectStoreConfig::aws_s3(Some(region), None, None, None)
    });
    config.wal_refresh_interval_ms = 1_000;
    config.cache_size_bytes = 256 * 1024 * 1024; // 256MB local cache
    config.num_centroids = 50;
    config.wal_batch_interval_ms = 1000;
    config.enable_compression = true;
    config.vector_dimension = 128;
    config.hot_tier = Some(redis_backend);

    outputln!("Creating ColdForge backend with Redis hot tier...");
    let backend: Arc<dyn Backend> = Arc::new(ColdForgeBackend::new(config).await?);
    outputln!("ColdForge backend ready\n");

    // Example 1: Insert vectors (goes to Redis hot tier + S3 cold tier)
    outputln!("--- Example 1: Insert Vectors ---");
    outputln!("Inserting 50 vectors...");

    let mut chunks = Vec::new();
    for i in 0..50 {
        let vector: Vec<f32> = (0..128).map(|_| fastrand::f32()).collect();
        let chunk = Chunk::new(
            format!("doc_{i}"),
            0,
            "",
            vector,
            0,
            0,
            1,
            serde_json::json!({
                "title": format!("Document {}", i),
                "category": if i % 2 == 0 { "even" } else { "odd" }
            }),
            Some("demo".to_string()),
        );
        chunks.push(chunk);
    }

    backend.upsert(&chunks).await?;
    backend.flush().await?;
    outputln!("Inserted 50 vectors");
    outputln!("  - Vectors are in this process index and the Redis get-cache");
    outputln!("  - flush wrote the WAL segment; that is the durable copy");

    // Example 2: Query from hot tier
    outputln!("\n--- Example 2: Query from Hot Tier ---");
    let query_vector: Vec<f32> = (0..128).map(|_| fastrand::f32()).collect();

    let query = SearchQuery::new(query_vector)
        .metric(Metric::Cosine)
        .limit(5)
        .namespace("demo");

    let start = std::time::Instant::now();
    let results = backend.search(&query).await?;
    let duration = start.elapsed();

    outputln!("Found {} results in {:?}", results.len(), duration);
    outputln!("  - Query ran on this process SPFresh index");
    outputln!("  - Redis was not the search index");

    outputln!("\n  Top 3 results:");
    for (i, result) in results.iter().take(3).enumerate() {
        outputln!(
            "    {}. {} (score: {:.4})",
            i + 1,
            result.chunk.id,
            result.score
        );
    }

    outputln!("\n--- Example 3: Get after flush ---");
    outputln!("get checks Redis only when the namespace index still has the id.");

    if let Some(_chunk) = backend.get("doc_10", Some("demo")).await? {
        outputln!("  Retrieved doc_10");
        outputln!("  Another process sees it after it replays the flushed WAL");
    }

    let count = backend.count(Some("demo")).await?;
    outputln!("\nVectors in this process index: {count}");
    outputln!("  - Count follows the index, not Redis key cardinality");

    outputln!("\nColdForge with Redis hot tier example completed.");
    outputln!("\nInvariants:");
    outputln!("  - Redis is a get cache checked against the index");
    outputln!("  - Search stays on the in-process index");
    outputln!("  - Durability is the flushed WAL segment on S3");
    outputln!(
        "  - Call refresh_shared_wal on a single-writer reader after another process flushes"
    );

    Ok(())
}
