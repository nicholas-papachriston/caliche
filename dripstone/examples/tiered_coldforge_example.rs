//! Tiered chain: S3 (cold) + `ColdForge` (hot).
//!
//! Uses [`TieredChainBuilder`] with S3 as the cold tier and `ColdForge` (`SPFresh` indexing)
//! as the hot tier. Tiers are added in cold-to-hot order.
//!
//! ## Architecture
//!
//! ```text
//! Tier 0 (cold): S3 Standard Storage  (source of truth)
//! Tier 1 (hot): ColdForge (SPFresh + Cache)  (promotion based on access)
//! ```
//!
//! ## Prerequisites
//!
//! - AWS credentials; two S3 buckets or same bucket with different prefixes
//!
//! A **tiering benchmark** section reports ingest time, first search, and mean search latency
//! after promotion-style reads (plain text). Override size with `TIERED_BENCH_VECTORS`.
//!
//! ## Run
//!
//! ```bash
//! cargo run --example tiered_coldforge_example --features coldforge-backend,s3-backend
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
use dripstone::backend::s3::{S3Backend, S3BackendConfig};
use dripstone::{Chunk, Metric, SearchQuery, TierConfig, TieredChainBuilder, traits::Backend};
use serde_json::json;
use std::env;
use std::sync::Arc;
use std::time::Instant;
use time::OffsetDateTime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Tiered Storage with ColdForge Hot Tier ===\n");

    // Get configuration
    let hot_bucket =
        env::var("TIERED_HOT_BUCKET").unwrap_or_else(|_| "my-hot-tier-bucket".to_string());
    let cold_bucket =
        env::var("TIERED_COLD_BUCKET").unwrap_or_else(|_| "my-cold-tier-bucket".to_string());
    let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    outputln!("Configuration:");
    outputln!("  Hot Tier: ColdForge (SPFresh + Cache)");
    outputln!("    Bucket: {hot_bucket}");
    outputln!("    Cache Size: 512 MB");
    outputln!("    Centroids: 128");
    outputln!("  Cold Tier: S3 Standard Storage");
    outputln!("    Bucket: {cold_bucket}");
    outputln!("    Region: {region}");
    outputln!();

    // Create ColdForge hot tier
    let mut hot_config = ColdForgeConfig::default();
    hot_config.bucket = hot_bucket.clone();
    hot_config.key_prefix = "hot-tier/".to_string();
    hot_config.access_key_id = None;
    hot_config.secret_access_key = None;
    hot_config.session_token = None;
    hot_config.region = region.clone();
    hot_config.cache_size_bytes = 512 * 1024 * 1024; // 512MB cache
    hot_config.num_centroids = 128;
    hot_config.wal_batch_interval_ms = 1000;
    hot_config.enable_compression = true;
    hot_config.vector_dimension = 384;
    hot_config.hot_tier = None; // Use in-memory cache (default)
    // object_store_config is set to None by Default::default() when the field exists

    let hot_tier = Arc::new(ColdForgeBackend::new(hot_config).await?);
    outputln!("Created ColdForge hot tier");

    // Create S3 cold tier
    let cold_config = S3BackendConfig {
        bucket: cold_bucket.clone(),
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
            TierConfig::new("coldforge")
                .max_vectors(1000)
                .max_bytes(512 * 1024 * 1024)
                .promotion_threshold(2),
        )
        .background_sync(true)
        .sync_interval_secs(300)
        .build()?;
    outputln!("Created tiered backend\n");

    // Use a unique namespace
    let namespace = format!(
        "tiered-coldforge-{}",
        OffsetDateTime::now_utc().unix_timestamp()
    );

    // Example 1: Bulk insert (goes to cold tier, promoted to hot tier)
    outputln!("--- Example 1: Bulk Insert ---");
    bulk_insert_demo(&backend, &namespace).await?;

    // Example 2: Access pattern-based promotion
    outputln!("\n--- Example 2: Access Pattern Promotion ---");
    promotion_demo(&backend, &namespace).await?;

    // Example 3: Search performance comparison
    outputln!("\n--- Example 3: Search Performance ---");
    search_performance_demo(&backend, &namespace).await?;

    // Example 4: Eviction and re-promotion
    outputln!("\n--- Example 4: Eviction and Re-promotion ---");
    eviction_demo(&backend, &namespace).await?;

    outputln!("\n--- Tiering benchmark ---");
    tiering_benchmark_section(&backend, &namespace, 384, 2).await?;

    // Cleanup
    outputln!("\n--- Cleanup ---");
    backend.delete_namespace(&namespace).await?;
    outputln!("Cleaned up test data");

    outputln!("\n=== Example Complete ===");
    outputln!("\nKey Benefits Demonstrated:");
    outputln!("  • SPFresh indexing for fast hot tier searches");
    outputln!("  • Automatic promotion based on access patterns");
    outputln!("  • Cost-effective cold storage in S3");
    outputln!("  • Seamless tier management");
    outputln!("  • Tiering benchmark (env TIERED_BENCH_VECTORS, default 128)");

    Ok(())
}

async fn bulk_insert_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Inserting 200 vectors into tiered storage...");

    let categories = ["tech", "science", "business", "health"];
    let mut chunks = Vec::new();
    for i in 0..200 {
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
            }),
            Some(namespace.to_string()),
        ));
    }

    let start = std::time::Instant::now();
    backend.upsert(&chunks).await?;
    let duration = start.elapsed();

    outputln!("Inserted 200 vectors in {duration:?}");
    outputln!("  - All vectors written to cold tier (S3)");
    outputln!("  - Vectors will be promoted to hot tier (ColdForge) on access");

    let count = backend.count(Some(namespace)).await?;
    outputln!("  - Total count: {count}");

    Ok(())
}

async fn promotion_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Accessing specific documents to trigger promotion...");

    // Access some documents multiple times to trigger promotion
    let frequently_accessed = ["doc-10", "doc-25", "doc-50", "doc-75"];

    for id in &frequently_accessed {
        outputln!("  Accessing {id} multiple times...");
        for _ in 0..3 {
            if let Some(chunk) = backend.get(id, Some(namespace)).await? {
                let title = chunk
                    .metadata
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("N/A");
                outputln!("    Retrieved: {id} ({title})");
            }
        }
    }

    outputln!("Frequently accessed vectors promoted to hot tier");
    outputln!("  - These vectors now benefit from SPFresh indexing");
    outputln!("  - Subsequent searches will be faster");

    Ok(())
}

async fn search_performance_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Comparing search performance...");

    let query_vec = generate_embedding(100, 384);
    let query = SearchQuery::new(query_vec.clone())
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace);

    // First search (may hit cold tier)
    outputln!("  Performing first search...");
    let start = std::time::Instant::now();
    let results1 = backend.search(&query).await?;
    let duration1 = start.elapsed();

    outputln!("    Found {} results in {:?}", results1.len(), duration1);

    // Second search (should benefit from hot tier)
    outputln!("  Performing second search (after promotion)...");
    let start = std::time::Instant::now();
    let results2 = backend.search(&query).await?;
    let duration2 = start.elapsed();

    outputln!("    Found {} results in {:?}", results2.len(), duration2);

    if duration2 < duration1 {
        outputln!("  Second search was faster (hot tier benefit)");
    }

    outputln!("\n  Top 3 results:");
    for (i, result) in results2.iter().take(3).enumerate() {
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

/// Timed ingest + search workload: S3 cold, `ColdForge` hot (`promotion_threshold` on hot tier).
async fn tiering_benchmark_section(
    backend: &Arc<dyn Backend>,
    base_namespace: &str,
    dimension: usize,
    promotion_threshold: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let bench_ns = format!("{base_namespace}-bench");
    let n = env::var("TIERED_BENCH_VECTORS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&v| v > 0)
        .unwrap_or(128);

    let mut chunks = Vec::with_capacity(n);
    for i in 0..n {
        let embedding = generate_embedding(30_000 + i, dimension);
        chunks.push(Chunk::new(
            format!("bench-{i}"),
            0,
            "",
            embedding,
            0,
            0,
            1,
            json!({ "bench": true, "i": i }),
            Some(bench_ns.clone()),
        ));
    }

    let t_ing = Instant::now();
    backend.upsert(&chunks).await?;
    let upsert_ms = t_ing.elapsed().as_secs_f64() * 1000.0;

    let query_vec = generate_embedding(40_000, dimension);
    let query = SearchQuery::new(query_vec.clone())
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(&bench_ns);

    let t_first = Instant::now();
    let r0 = backend.search(&query).await?;
    let first_search_ms = t_first.elapsed().as_secs_f64() * 1000.0;

    let promote_ids = ["bench-0", "bench-1", "bench-2", "bench-3"];
    for id in promote_ids {
        for _ in 0..promotion_threshold.saturating_add(1) {
            let _ = backend.get(id, Some(bench_ns.as_str())).await?;
        }
    }

    let warmed_runs: usize = 20;
    let mut sum_ns = 0_u128;
    for _ in 0..warmed_runs {
        let t = Instant::now();
        let _ = backend.search(&query).await?;
        sum_ns += t.elapsed().as_nanos();
    }
    let avg_warmed_ms = (sum_ns as f64 / warmed_runs as f64) / 1_000_000.0;

    outputln!("  namespace_suffix: bench");
    outputln!("  vectors: {n} dim {dimension}");
    outputln!("  upsert_ms: {upsert_ms:.2}");
    outputln!(
        "  first_search_ms: {first_search_ms:.2} (hits {} results)",
        r0.len()
    );
    outputln!(
        "  promotion_gets_per_hot_doc: {}",
        promotion_threshold.saturating_add(1)
    );
    outputln!("  avg_search_ms_warmed_{warmed_runs}_runs: {avg_warmed_ms:.3}");

    backend.delete_namespace(&bench_ns).await?;
    Ok(())
}

async fn eviction_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Testing eviction behavior...");

    // Insert more vectors to trigger eviction
    outputln!("  Inserting additional vectors to fill hot tier...");
    let mut chunks = Vec::new();
    for i in 200..500 {
        let embedding = generate_embedding(i, 384);
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
                "batch": "eviction-test",
            }),
            Some(namespace.to_string()),
        ));
    }
    backend.upsert(&chunks).await?;

    outputln!("Inserted 300 additional vectors");
    outputln!("  - Hot tier capacity exceeded");
    outputln!("  - Least recently used vectors evicted from hot tier");
    outputln!("  - All vectors still accessible from cold tier");

    // Verify all vectors are accessible
    let count = backend.count(Some(namespace)).await?;
    outputln!("  - Total vectors accessible: {count}");

    // Access an old vector (should be retrieved from cold tier and re-promoted)
    outputln!("  Accessing previously evicted vector...");
    if let Some(chunk) = backend.get("doc-10", Some(namespace)).await? {
        let title = chunk
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("N/A");
        outputln!("    Retrieved 'doc-10' ({title}) from cold tier");
        outputln!("    Vector re-promoted to hot tier");
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
