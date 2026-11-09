//! Tiered chain: S3 Vector Buckets (cold) + Memory (hot).
//!
//! Uses [`TieredChainBuilder`] with S3 Vector Buckets as the cold tier and
//! in-memory as the hot tier. Tiers are added in cold-to-hot order.
//! Gives server-side search in S3 plus fast access for hot data.
//!
//! ## Prerequisites
//!
//! - AWS credentials; S3 Vector Bucket and index
//! - Env: `S3_VECTOR_BUCKET`, `S3_VECTOR_INDEX`, `AWS_REGION` (default: us-east-1)
//!
//! ## Run
//!
//! ```bash
//! cargo run --example s3_vector_tiered_example --features "s3-backend,memory-backend"
//! ```

#![cfg(all(feature = "s3-backend", feature = "memory-backend"))]

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

use dripstone::MemoryBackend;
use dripstone::backend::s3::{S3VectorBackend, S3VectorBackendConfig};
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::traits::Backend;
use dripstone::{TierConfig, TieredChainBuilder};
use serde_json::json;
use std::env;
use std::sync::Arc;
use time::OffsetDateTime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== S3 Vector Buckets + Tiered Storage Example ===\n");

    let Ok(bucket_name) = env::var("S3_VECTOR_BUCKET") else {
        outputln!(
            "SKIP: set S3_VECTOR_BUCKET and S3_VECTOR_INDEX. This example does not create a vector bucket."
        );
        return Ok(());
    };
    let Ok(index_name) = env::var("S3_VECTOR_INDEX") else {
        outputln!("SKIP: set S3_VECTOR_INDEX. This example does not create a vector index.");
        return Ok(());
    };
    if bucket_name.is_empty() || index_name.is_empty() {
        outputln!("SKIP: S3_VECTOR_BUCKET and S3_VECTOR_INDEX must be non-empty");
        return Ok(());
    }
    let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    outputln!("Configuration:");
    outputln!("  Cold Tier: S3 Vector Buckets");
    outputln!("    Bucket: {bucket_name}");
    outputln!("    Index: {index_name}");
    outputln!("    Region: {region}");
    outputln!("  Hot Tier: In-Memory");
    outputln!("    Max Vectors: 50");
    outputln!("    Max Memory: 10 MB");
    outputln!();

    // Create cold tier (S3 Vector Buckets)
    let cold_config = S3VectorBackendConfig::new(&bucket_name, &index_name, 384, None, None, None)
        .with_distance_metric(Metric::Cosine)
        .with_region(&region)
        .with_batch_size(100);

    let cold_tier: Arc<S3VectorBackend> = Arc::new(S3VectorBackend::new(cold_config).await?);
    outputln!("Connected to S3 Vector Buckets (cold tier)");

    // Create hot tier (in-memory)
    let hot_tier: Arc<MemoryBackend> = Arc::new(MemoryBackend::new());
    outputln!("Created in-memory backend (hot tier)");

    // Create tiered backend (cold -> hot)
    let backend: Arc<dyn Backend> = TieredChainBuilder::new()
        .tier(cold_tier, TierConfig::new("s3-vector"))
        .tier(
            hot_tier,
            TierConfig::new("memory")
                .max_vectors(50)
                .max_bytes(10 * 1024 * 1024),
        )
        .background_sync(true)
        .sync_interval_secs(60)
        .build()?;
    outputln!("Created tiered backend\n");

    // Use a unique namespace for this example
    let namespace = format!("tiered-demo-{}", OffsetDateTime::now_utc().unix_timestamp());

    // Example 1: Insert vectors (go to both tiers)
    outputln!("--- Example 1: Insert Vectors ---");
    insert_demo_vectors(&backend, &namespace).await?;

    // Example 2: Query vectors (uses hot tier when possible)
    outputln!("\n--- Example 2: Query from Tiered Storage ---");
    query_demo(&backend, &namespace).await?;

    // Example 3: Access patterns and promotion
    outputln!("\n--- Example 3: Access Patterns ---");
    access_pattern_demo(&backend, &namespace).await?;

    // Example 4: Hot tier eviction
    outputln!("\n--- Example 4: Hot Tier Eviction ---");
    eviction_demo(&backend, &namespace).await?;

    // Cleanup
    outputln!("\n--- Cleanup ---");
    backend.delete_namespace(&namespace).await?;
    outputln!("Cleaned up test data");

    outputln!("\n=== Example Complete ===");
    Ok(())
}

async fn insert_demo_vectors(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Inserting 20 vectors into tiered storage...");

    let categories = ["technology", "science", "business", "health"];
    let mut chunks = Vec::new();

    for i in 0..20 {
        let embedding = generate_demo_embedding(i, 384);
        let title = format!("Document {i}");
        let chunk = Chunk::new(
            format!("doc-{i}"),
            0,
            &title,
            embedding,
            0,
            title.len(),
            1,
            json!({
                "title": title,
                "category": categories[i % categories.len()],
                "priority": (i % 5) + 1,
            }),
            Some(namespace.to_string()),
        );
        chunks.push(chunk);
    }

    backend.upsert(&chunks).await?;

    outputln!("Inserted 20 vectors");
    outputln!("  - All vectors written to cold tier (S3 Vector Buckets)");
    outputln!("  - All vectors promoted to hot tier (in-memory)");

    let count = backend.count(Some(namespace)).await?;
    outputln!("  - Total count: {count}");

    Ok(())
}

async fn query_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let query_vector = generate_demo_embedding(5, 384);

    outputln!("Performing similarity search...");
    let start = std::time::Instant::now();

    let query = SearchQuery {
        vector: query_vector.into(),
        limit: 5,
        metric: Metric::Cosine,
        namespace: Some(namespace.to_string()),
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    let duration = start.elapsed();

    outputln!("Found {} results in {:?}", results.len(), duration);
    outputln!("  Note: Query served from appropriate tier");
    outputln!("        (hot tier if sufficient data, cold tier for complete search)");

    for (i, result) in results.iter().enumerate() {
        let title = result
            .chunk
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");

        outputln!("  {}. {} (score: {:.4})", i + 1, title, result.score);
    }

    Ok(())
}

async fn access_pattern_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Simulating access patterns...");

    // Access specific documents multiple times
    let frequently_accessed = ["doc-5#0", "doc-10#0", "doc-15#0"];

    for id in &frequently_accessed {
        for _ in 0..3 {
            if let Some(chunk) = backend.get(id, Some(namespace)).await? {
                let title = chunk
                    .metadata
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("N/A");
                outputln!("  Accessed: {id} ({title})");
            }
        }
    }

    outputln!("Frequently accessed vectors are now 'hot'");
    outputln!("  - Subsequent accesses will be fast (from hot tier)");
    outputln!("  - Infrequently accessed vectors remain in cold tier");

    Ok(())
}

async fn eviction_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Testing hot tier capacity limits...");

    // Insert more vectors to trigger eviction
    let mut chunks = Vec::new();
    for i in 20..80 {
        let embedding = generate_demo_embedding(i, 384);
        let title = format!("Document {i}");
        let chunk = Chunk::new(
            format!("doc-{i}"),
            0,
            &title,
            embedding,
            0,
            title.len(),
            1,
            json!({
                "title": title,
                "batch": "eviction-test",
            }),
            Some(namespace.to_string()),
        );
        chunks.push(chunk);
    }

    backend.upsert(&chunks).await?;

    outputln!("Inserted 60 additional vectors");
    outputln!("  - Hot tier capacity exceeded");
    outputln!("  - Least recently used vectors evicted from hot tier");
    outputln!("  - All vectors still available in cold tier (S3 Vector Buckets)");

    // Verify all vectors are still accessible
    let count = backend.count(Some(namespace)).await?;
    outputln!("  - Total vectors accessible: {count}");

    // Access an old vector (should be retrieved from cold tier)
    if let Some(_chunk) = backend.get("doc-1#0", Some(namespace)).await? {
        outputln!("  - Successfully retrieved 'doc-1' from cold tier");
        outputln!("  - Vector promoted back to hot tier on access");
    }

    Ok(())
}

fn generate_demo_embedding(seed: usize, dimension: usize) -> Vec<f32> {
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
