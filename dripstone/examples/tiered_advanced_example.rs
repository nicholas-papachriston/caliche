//! Tiered chain with custom promotion: S3 (cold) + Memory (hot).
//!
//! Uses [`TieredChainBuilder`] with a custom per-tier `promotion_threshold`
//! so vectors are promoted to the hot tier only after multiple accesses.
//! Also demonstrates access-pattern analysis and performance monitoring.
//! A closing **tiering benchmark** prints ingest time, first search, and mean search
//! latency after promotion-style reads (plain text for logs/docs).
//!
//! ## Run
//!
//! ```bash
//! cargo run --example tiered_advanced_example --features s3-backend,memory-backend
//! ```

#![cfg(feature = "s3-backend")]

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

use dripstone::backend::memory::MemoryBackend;
use dripstone::backend::s3::{S3Backend, S3BackendConfig};
use dripstone::backend::shared::{AccessTracker, build_composite_key};
use dripstone::{Chunk, Metric, SearchQuery, TierConfig, TieredChainBuilder, traits::Backend};
use serde_json::json;
use std::env;
use std::sync::Arc;
use std::time::Instant;
use time::OffsetDateTime;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Advanced Tiered Storage Example ===\n");

    // Get configuration
    let s3_bucket = env::var("TIERED_S3_BUCKET").unwrap_or_else(|_| "my-tiered-bucket".to_string());
    let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

    outputln!("Configuration:");
    outputln!("  Hot Tier: In-Memory");
    outputln!("    Max Vectors: 200");
    outputln!("    Max Memory: 50 MB");
    outputln!("  Cold Tier: S3 Standard Storage");
    outputln!("    Bucket: {s3_bucket}");
    outputln!("    Region: {region}");
    outputln!();

    // Create backends
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
    let hot_tier = Arc::new(MemoryBackend::new());

    // Create tiered backend (cold -> hot) with custom promotion threshold
    let backend: Arc<dyn Backend> = TieredChainBuilder::new()
        .tier(cold_tier, TierConfig::new("s3"))
        .tier(
            hot_tier,
            TierConfig::new("memory")
                .max_vectors(200)
                .max_bytes(50 * 1024 * 1024)
                .promotion_threshold(3),
        )
        .background_sync(true)
        .sync_interval_secs(300)
        .build()?;

    // Create shared access tracker for monitoring
    let access_tracker = Arc::new(AccessTracker::new());

    outputln!("Created tiered backend with custom configuration\n");

    let namespace = format!(
        "tiered-advanced-{}",
        OffsetDateTime::now_utc().unix_timestamp()
    );

    // Example 1: Custom promotion strategy
    outputln!("--- Example 1: Custom Promotion Strategy ---");
    custom_promotion_demo(&backend, &access_tracker, &namespace).await?;

    // Example 2: Access pattern analysis
    outputln!("\n--- Example 2: Access Pattern Analysis ---");
    access_analysis_demo(&backend, &access_tracker, &namespace).await?;

    // Example 3: Performance monitoring
    outputln!("\n--- Example 3: Performance Monitoring ---");
    performance_monitoring_demo(&backend, &namespace).await?;

    // Example 4: Selective promotion
    outputln!("\n--- Example 4: Selective Promotion ---");
    selective_promotion_demo(&backend, &access_tracker, &namespace).await?;

    outputln!("\n--- Tiering benchmark ---");
    tiering_benchmark_section(&backend, &namespace, 384, 3).await?;

    // Cleanup
    outputln!("\n--- Cleanup ---");
    backend.delete_namespace(&namespace).await?;
    outputln!("Cleaned up test data");

    outputln!("\n=== Example Complete ===");
    outputln!("\nAdvanced Features Demonstrated:");
    outputln!("  • Custom promotion strategies");
    outputln!("  • Access pattern analysis");
    outputln!("  • Performance monitoring");
    outputln!("  • Selective promotion based on metadata");
    outputln!("  • Tiering benchmark (env TIERED_BENCH_VECTORS, default 128)");

    Ok(())
}

fn estimate_chunk_size(chunk: &Chunk) -> usize {
    let base_size = std::mem::size_of::<Chunk>();
    let vector_size = chunk.vector.len() * std::mem::size_of::<f32>();
    let id_size = chunk.id.len();
    let doc_id_size = chunk.doc_id.len();
    let content_size = chunk.content.len();
    let namespace_size = chunk.namespace.as_ref().map_or(0, std::string::String::len);
    let metadata_size = chunk.metadata.to_string().len();
    base_size
        + vector_size
        + id_size
        + doc_id_size
        + content_size
        + namespace_size
        + metadata_size
        + 100
}

async fn custom_promotion_demo(
    backend: &Arc<dyn Backend>,
    access_tracker: &AccessTracker,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Inserting vectors with different priorities...");

    // Insert vectors with priority metadata
    let mut chunks = Vec::new();
    for i in 0..50 {
        let embedding = generate_embedding(i, 384);
        let priority = if i % 10 == 0 { "high" } else { "normal" };
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
                "priority": priority,
                "index": i,
            }),
            Some(namespace.to_string()),
        ));
    }
    backend.upsert(&chunks).await?;

    outputln!("Inserted 50 vectors");
    outputln!("  - 5 high priority, 45 normal priority");

    // Access high priority documents more frequently
    outputln!("\n  Accessing high priority documents...");
    for i in (0..50).step_by(10) {
        let doc_id = format!("doc-{i}");
        let key = build_composite_key(Some(namespace), &doc_id);

        // Access multiple times to trigger promotion
        for _ in 0..4 {
            if let Some(chunk) = backend.get(&doc_id, Some(namespace)).await? {
                let size = estimate_chunk_size(&chunk);
                access_tracker.record_access(key.clone(), size);
            }
        }

        if access_tracker.should_promote(&key, 3) {
            outputln!("    {doc_id} promoted (access count >= 3)");
        }
    }

    Ok(())
}

async fn access_analysis_demo(
    backend: &Arc<dyn Backend>,
    access_tracker: &AccessTracker,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Analyzing access patterns...");

    // Simulate various access patterns
    let patterns = vec![
        ("doc-5", 10), // Frequently accessed
        ("doc-15", 8), // Frequently accessed
        ("doc-25", 5), // Moderately accessed
        ("doc-35", 2), // Rarely accessed
        ("doc-45", 1), // Rarely accessed
    ];

    for (doc_id, count) in &patterns {
        let key = build_composite_key(Some(namespace), doc_id);
        outputln!("  Accessing {doc_id} {count} times...");

        for _ in 0..*count {
            if let Some(chunk) = backend.get(doc_id, Some(namespace)).await? {
                let size = estimate_chunk_size(&chunk);
                access_tracker.record_access(key.clone(), size);
            }
        }
    }

    outputln!("\n  Access Statistics:");
    for (doc_id, _) in &patterns {
        let key = build_composite_key(Some(namespace), doc_id);
        if let Some(metrics) = access_tracker.get_metrics(&key) {
            outputln!(
                "    {}: {} accesses, last accessed: {:?}",
                doc_id,
                metrics.access_count,
                metrics.last_accessed
            );
        }
    }

    Ok(())
}

#[expect(
    clippy::expect_used,
    reason = "example code panics on empty benchmark samples for brevity"
)]
async fn performance_monitoring_demo(
    backend: &Arc<dyn Backend>,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Monitoring query performance...");

    let query_vec = generate_embedding(20, 384);
    let query = SearchQuery::new(query_vec.clone())
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(namespace);

    // Perform multiple queries and measure performance
    let mut durations = Vec::new();

    for i in 0..5 {
        let start = std::time::Instant::now();
        let results = backend.search(&query).await?;
        let duration = start.elapsed();
        durations.push(duration);

        outputln!(
            "  Query {}: {} results in {:?}",
            i + 1,
            results.len(),
            duration
        );
    }

    let avg_duration: f64 = durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / durations.len() as f64;

    outputln!("\n  Performance Summary:");
    outputln!("    Average query time: {avg_duration:.2}ms");
    outputln!(
        "    Min: {:?}",
        durations
            .iter()
            .min()
            .expect("durations should not be empty")
    );
    outputln!(
        "    Max: {:?}",
        durations
            .iter()
            .max()
            .expect("durations should not be empty")
    );

    Ok(())
}

/// Timed ingest + search workload to characterize cold vs warmed tiering (memory hot, S3 cold).
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
        let embedding = generate_embedding(10_000 + i, dimension);
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

    let query_vec = generate_embedding(20_000, dimension);
    let query = SearchQuery::new(query_vec.clone())
        .metric(Metric::Cosine)
        .limit(10)
        .namespace(&bench_ns);

    let t_first = Instant::now();
    let r0 = backend.search(&query).await?;
    let first_search_ms = t_first.elapsed().as_secs_f64() * 1000.0;

    // Drive promotion to hot tier (matches TierConfig promotion_threshold on memory).
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

async fn selective_promotion_demo(
    backend: &Arc<dyn Backend>,
    access_tracker: &AccessTracker,
    namespace: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("Selective promotion based on metadata...");

    // Insert vectors with different categories
    let mut chunks = Vec::new();
    for i in 50..100 {
        let embedding = generate_embedding(i, 384);
        let category = if i % 3 == 0 { "premium" } else { "standard" };
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
                "index": i,
            }),
            Some(namespace.to_string()),
        ));
    }
    backend.upsert(&chunks).await?;

    outputln!("Inserted 50 additional vectors");
    outputln!("  - Premium category: higher promotion priority");

    // Only promote premium documents
    outputln!("\n  Promoting premium documents...");
    for i in (50..100).step_by(3) {
        let doc_id = format!("doc-{i}");
        let key = build_composite_key(Some(namespace), &doc_id);

        // Access premium documents
        if let Some(chunk) = backend.get(&doc_id, Some(namespace)).await? {
            let category = chunk
                .metadata
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");

            if category == "premium" {
                let size = estimate_chunk_size(&chunk);
                access_tracker.record_access(key.clone(), size);

                // Lower threshold for premium
                if access_tracker.should_promote(&key, 2) {
                    outputln!("    {doc_id} promoted (premium, access count >= 2)");
                }
            }
        }
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
