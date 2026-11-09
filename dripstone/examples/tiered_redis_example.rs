//! Tiered chain: S3 (cold) + Redis (hot).
//!
//! Uses [`TieredChainBuilder`] to combine S3 as the cold tier (source of truth)
//! with Redis as the hot tier. Tiers are added in cold-to-hot order.
//! Redis can be a shared hot tier. The default search strategy reads every tier.
//! `HotThenSourceFallback` does not stop on a capped Redis tier: `max_vectors`
//! means that tier is a partial cache and colder tiers are still searched.
//!
//! ## Architecture
//!
//! ```text
//! Tier 0 (cold): S3 Standard Storage  (source of truth)
//! Tier 1 (hot): Redis                 (promotion based on access)
//! ```
//!
//! ## Prerequisites
//!
//! - Redis (default: <redis://127.0.0.1:6379>)
//! - AWS credentials and S3 bucket
//!
//! ## Run
//!
//! ```bash
//! cargo run --example tiered_redis_example --features redis-backend,s3-backend
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

use dripstone::backend::redis::RedisBackend;
use dripstone::backend::s3::{S3Backend, S3BackendConfig};
use dripstone::search::Filter;
use dripstone::{Chunk, Metric, SearchQuery, traits::Backend};
use dripstone::{TierConfig, TieredChainBuilder};
use serde_json::json;
use std::env;
use std::sync::Arc;
use time::OffsetDateTime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Tiered Storage with Redis Hot Tier ===\n");

    // Get configuration
    let redis_url = env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let s3_bucket = env::var("TIERED_S3_BUCKET").unwrap_or_else(|_| "my-tiered-bucket".to_string());
    let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    outputln!("Configuration:");
    outputln!("  Hot Tier: Redis");
    outputln!("    URL: {redis_url}");
    outputln!("    Prefix: tiered:hot:");
    outputln!("  Cold Tier: S3 Standard Storage");
    outputln!("    Bucket: {s3_bucket}");
    outputln!("    Region: {region}");
    outputln!();

    // Create Redis hot tier
    let hot_tier = Arc::new(RedisBackend::with_prefix(redis_url.as_str(), "tiered:hot:").await?);
    outputln!("Connected to Redis hot tier");

    // Create S3 cold tier
    let cold_config = S3BackendConfig {
        bucket: s3_bucket.clone(),
        key_prefix: "cold-tier/".to_string(),
        access_key_id: None,
        secret_access_key: None,
        session_token: None,
        region: None,
        enable_compression: true,
        batch_size: 100,
        cache_indices: false,
    };

    let cold_tier = Arc::new(S3Backend::with_config(cold_config).await?);
    outputln!("Created S3 cold tier");

    // Create tiered backend (cold -> hot)
    let backend: Arc<dyn Backend> = TieredChainBuilder::new()
        .tier(cold_tier, TierConfig::new("s3"))
        .tier(
            hot_tier,
            TierConfig::new("redis")
                .max_vectors(5000)
                .max_bytes(1024 * 1024 * 1024),
        )
        .background_sync(true)
        .sync_interval_secs(300)
        .build()?;
    outputln!("Created tiered backend\n");

    // Use a unique namespace
    let namespace = format!(
        "tiered-redis-{}",
        OffsetDateTime::now_utc().unix_timestamp()
    );

    // Example 1: Insert vectors
    outputln!("--- Example 1: Insert Vectors ---");
    insert_demo(&backend, &namespace).await?;

    // Example 2: Query performance
    outputln!("\n--- Example 2: Query Performance ---");
    query_demo(&backend, &namespace).await?;

    // Example 3: Distributed access simulation
    outputln!("\n--- Example 3: Distributed Access ---");
    distributed_access_demo(&backend, &namespace).await?;

    // Example 4: Metadata filtering
    outputln!("\n--- Example 4: Metadata Filtering ---");
    metadata_filtering_demo(&backend, &namespace).await?;

    // Cleanup
    outputln!("\n--- Cleanup ---");
    backend.delete_namespace(&namespace).await?;
    outputln!("Cleaned up test data");

    outputln!("\n=== Example Complete ===");
    outputln!("\nKey Benefits Demonstrated:");
    outputln!("  • Distributed hot cache via Redis");
    outputln!("  • Fast access to frequently used vectors");
    outputln!("  • Cost-effective cold storage");
    outputln!("  • Automatic promotion/eviction");

    Ok(())
}

async fn insert_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Inserting 100 vectors...");

    let categories = ["tech", "science", "business"];
    let mut chunks = Vec::new();
    for i in 0..100 {
        let embedding = generate_embedding(i, 384);
        let category = categories[i % categories.len()];
        chunks.push(Chunk::new(
            format!("doc-{i}"),
            0,
            "",
            embedding,
            0,
            0,
            1,
            json!({
                "title": format!("Document {}", i),
                "category": category,
                "priority": (i % 5) + 1,
                "timestamp": OffsetDateTime::now_utc().unix_timestamp(),
            }),
            Some(namespace.to_string()),
        ));
    }

    let start = std::time::Instant::now();
    backend.upsert(&chunks).await?;
    let duration = start.elapsed();

    outputln!("Inserted 100 vectors in {duration:?}");
    outputln!("  - All vectors written to cold tier (S3)");
    outputln!("  - Vectors promoted to hot tier (Redis) on access");

    let count = backend.count(Some(namespace)).await?;
    outputln!("  - Total count: {count}");

    Ok(())
}

async fn query_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let query_vec = generate_embedding(50, 384);
    let query = SearchQuery::new(query_vec)
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace);

    outputln!("Performing similarity search...");
    let start = std::time::Instant::now();

    let results = backend.search(&query).await?;

    let duration = start.elapsed();

    outputln!("Found {} results in {:?}", results.len(), duration);
    outputln!("  - Query served from appropriate tier");
    outputln!("  - Results promoted to hot tier for future access");

    outputln!("\n  Top 5 results:");
    for (i, result) in results.iter().take(5).enumerate() {
        let title = result
            .chunk
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        outputln!("    {}. {} (score: {:.4})", i + 1, title, result.score);
    }

    Ok(())
}

async fn distributed_access_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Simulating distributed access patterns...");
    outputln!("  (In a real deployment, multiple instances would share Redis)");

    // Simulate different access patterns
    let access_patterns = vec![
        ("doc-10", "Instance A"),
        ("doc-25", "Instance B"),
        ("doc-50", "Instance C"),
        ("doc-75", "Instance A"),
    ];

    for (doc_id, instance) in &access_patterns {
        outputln!("  {instance} accessing {doc_id}...");
        if let Some(chunk) = backend.get(doc_id, Some(namespace)).await? {
            let title = chunk
                .metadata
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or("N/A");
            outputln!("    Retrieved: {doc_id} ({title})");
            outputln!("    Vector now in shared Redis cache");
        }
    }

    outputln!("All accessed vectors are in shared hot tier");
    outputln!("  - Any instance can benefit from cached vectors");
    outputln!("  - Reduces load on cold tier");

    Ok(())
}

async fn metadata_filtering_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Demonstrating metadata filtering...");

    let query_vec = generate_embedding(30, 384);
    let query = SearchQuery::new(query_vec)
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace)
        .filter(Filter::Equal {
            field: "category".to_string(),
            value: json!("tech"),
        });

    let start = std::time::Instant::now();
    let results = backend.search(&query).await?;
    let duration = start.elapsed();

    outputln!("Found {} tech documents in {:?}", results.len(), duration);

    outputln!("\n  Filtered results:");
    for (i, result) in results.iter().take(5).enumerate() {
        let title = result
            .chunk
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        let category = result
            .chunk
            .metadata
            .get("category")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        outputln!(
            "    {}. {} [{}] (score: {:.4})",
            i + 1,
            title,
            category,
            result.score
        );
    }

    Ok(())
}

fn generate_embedding(seed: usize, dimension: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    let hash_seed = hasher.finish();

    let mut embedding = Vec::with_capacity(dimension);
    let mut rng_state = hash_seed;

    for _ in 0..dimension {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let value = ((rng_state >> 32) as f32 / u32::MAX as f32).mul_add(2.0, -1.0);
        embedding.push(value);
    }

    // Normalize
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for val in &mut embedding {
            *val /= magnitude;
        }
    }

    embedding
}
