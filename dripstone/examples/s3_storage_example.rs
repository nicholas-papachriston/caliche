//! S3 Standard backend and tiered chain (S3 + Memory).
//!
//! 1. Direct S3: standard S3 buckets with client-side linear scan.
//! 2. Tiered chain: [`TieredChainBuilder`] with S3 (cold) + Memory (hot).
//!
//! For S3 Vector Buckets (server-side search), see `s3_vector_buckets_example.rs`.

#![cfg(all(feature = "s3-backend", feature = "memory-backend"))]

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::memory::MemoryBackend;
use dripstone::backend::s3::{S3Backend, S3BackendConfig};
use dripstone::{Chunk, Metric, SearchQuery, TierConfig, TieredChainBuilder, traits::Backend};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== S3 Vector Storage Examples ===\n");

    let Ok(bucket) = std::env::var("S3_BUCKET") else {
        outputln!(
            "SKIP: set S3_BUCKET (for example example-bucket), AWS_REGION=us-east-1, and AWS_PROFILE=example"
        );
        return Ok(());
    };
    if bucket.is_empty() {
        outputln!("SKIP: S3_BUCKET is empty");
        return Ok(());
    }

    example_direct_s3(&bucket).await?;
    example_tiered_storage(&bucket).await?;

    Ok(())
}

/// Example: Direct S3 storage
async fn example_direct_s3(bucket: &str) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("--- Example 1: Direct S3 Storage ---");

    let namespace = format!(
        "documents-{}",
        time::OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let config = S3BackendConfig {
        bucket: bucket.to_string(),
        key_prefix: format!("embeddings/{namespace}/"),
        access_key_id: None,
        secret_access_key: None,
        session_token: None,
        region: std::env::var("AWS_REGION").ok(),
        enable_compression: true,
        batch_size: 100,
        cache_indices: true,
    };

    // Create S3 backend
    let backend: Arc<dyn Backend> = Arc::new(S3Backend::with_config(config).await?);

    // Create sample vectors
    outputln!("Uploading vectors to S3...");
    let mut chunks = Vec::new();
    for i in 0..10 {
        let vector: Vec<f32> = (0..384).map(|j| ((i + j) as f32 * 0.01) % 1.0).collect();
        let chunk = Chunk::new(
            format!("doc-{i}"),
            0,
            "",
            vector,
            0,
            0,
            1,
            serde_json::json!({
                "title": format!("Document {}", i),
                "category": if i % 2 == 0 { "tech" } else { "science" }
            }),
            Some(namespace.clone()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    outputln!("Uploaded 10 vectors to S3");

    // Query vectors from S3
    outputln!("\nQuerying vectors from S3...");
    let query_vector: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01) % 1.0).collect();

    let query = SearchQuery::new(query_vector)
        .metric(Metric::Cosine)
        .limit(5)
        .namespace(&namespace);
    let results = backend.search(&query).await?;
    if results.is_empty() {
        return Err(
            "direct S3 search returned 0 hits after upsert; the namespace index was not updated"
                .into(),
        );
    }

    outputln!("Found {} similar vectors:", results.len());
    for (i, result) in results.iter().enumerate() {
        outputln!(
            "  {}. {} (score: {:.4})",
            i + 1,
            result.chunk.id,
            result.score
        );
    }

    // Count vectors
    let count = backend.count(Some(&namespace)).await?;
    outputln!("\nTotal vectors in S3: {count}");
    if count == 0 {
        return Err(
            "direct S3 count stayed 0 after upsert; the namespace index was not updated".into(),
        );
    }

    Ok(())
}

/// Example: Tiered storage with S3 cold storage and in-memory hot storage
async fn example_tiered_storage(bucket: &str) -> Result<(), Box<dyn std::error::Error>> {
    outputln!("\n--- Example 2: Tiered Storage (Memory + S3) ---");

    let s3_config = S3BackendConfig {
        bucket: bucket.to_string(),
        key_prefix: format!(
            "embeddings/tiered/{}/",
            time::OffsetDateTime::now_utc().unix_timestamp_nanos()
        ),
        access_key_id: None,
        secret_access_key: None,
        session_token: None,
        region: std::env::var("AWS_REGION").ok(),
        enable_compression: true,
        batch_size: 100,
        cache_indices: true,
    };

    let cold_tier: Arc<S3Backend> = Arc::new(S3Backend::with_config(s3_config).await?);

    // Create in-memory backend for hot tier
    let hot_tier: Arc<MemoryBackend> = Arc::new(MemoryBackend::new());

    // Create tiered backend (cold -> hot)
    let backend: Arc<dyn Backend> = TieredChainBuilder::new()
        .tier(cold_tier, TierConfig::new("s3"))
        .tier(
            hot_tier,
            TierConfig::new("memory")
                .max_vectors(100)
                .max_bytes(50 * 1024 * 1024),
        )
        .background_sync(true)
        .sync_interval_secs(60)
        .build()?;

    outputln!("Uploading vectors to tiered storage...");

    // Upload vectors - they go to both hot and cold tiers
    let mut chunks = Vec::new();
    for i in 0..50 {
        let vector: Vec<f32> = (0..384).map(|j| ((i + j) as f32 * 0.01) % 1.0).collect();
        let chunk = Chunk::new(
            format!("article-{i}"),
            0,
            "",
            vector,
            0,
            0,
            1,
            serde_json::json!({
                "title": format!("Article {}", i),
                "views": i * 100
            }),
            Some("articles".to_string()),
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    outputln!("Uploaded 50 vectors to tiered storage");

    // Query - will use hot tier for speed
    outputln!("\nQuerying from tiered storage (uses hot tier when possible)...");
    let query_vector: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01) % 1.0).collect();

    let query = SearchQuery::new(query_vector)
        .metric(Metric::Cosine)
        .limit(10)
        .namespace("articles");

    let start = std::time::Instant::now();
    let results = backend.search(&query).await?;
    let duration = start.elapsed();

    outputln!("Found {} results in {:?}:", results.len(), duration);
    for (i, result) in results.iter().take(5).enumerate() {
        outputln!(
            "  {}. {} (score: {:.4})",
            i + 1,
            result.chunk.id,
            result.score
        );
    }

    // Access specific vectors (promotes to hot tier)
    outputln!("\nAccessing specific vectors (promotes to hot tier)...");
    for id in &["article-5", "article-10", "article-15"] {
        if let Some(chunk) = backend.get(id, Some("articles")).await? {
            outputln!("  Retrieved: {}", chunk.id);
        }
    }

    outputln!("\nTiered storage example complete");
    outputln!("  - Frequently accessed vectors are kept in fast memory");
    outputln!("  - All vectors are durably stored in S3");
    outputln!("  - LRU eviction manages hot tier capacity");

    Ok(())
}
