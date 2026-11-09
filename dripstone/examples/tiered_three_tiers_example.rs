//! Tiered chain with three tiers: `SQLite` (cold) -> Redis (warm) -> Memory (hot).
//!
//! Uses [`TieredChainBuilder`] to build a cold-to-hot chain. Reads hit Memory first,
//! then Redis, then `SQLite`. Frequently accessed data is promoted toward the hot tier.
//!
//! ## Architecture
//!
//! ```text
//! Tier 0 (cold):  SQLite   (source of truth, durable)
//! Tier 1 (warm):  Redis    (shared cache)
//! Tier 2 (hot):   Memory   (process-local cache)
//! ```
//!
//! ## Prerequisites
//!
//! - Redis (default: <redis://127.0.0.1:6379>)
//! - Writable path for `SQLite` DB (default: ./`data/tiered_three.db`)
//!
//! ## Run
//!
//! ```bash
//! cargo run --example tiered_three_tiers_example --features sqlite-backend,redis-backend,memory-backend
//! ```

#![cfg(all(
    feature = "sqlite-backend",
    feature = "redis-backend",
    feature = "memory-backend"
))]

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
use dripstone::backend::redis::RedisBackend;
use dripstone::backend::sqlite::{MemoryStrategy, SqliteBackend, SqliteBackendConfig};
use dripstone::traits::Backend;
use dripstone::{Chunk, Metric, SearchQuery, TierConfig, TieredChainBuilder};
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Three-Tier Chain (SQLite -> Redis -> Memory) ===\n");

    let redis_url =
        std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
    let db_path = std::env::var("TIERED_SQLITE_PATH")
        .map_or_else(|_| PathBuf::from("./data/tiered_three.db"), PathBuf::from);

    outputln!("Configuration:");
    outputln!("  Tier 0 (cold):  SQLite  at {}", db_path.display());
    outputln!("  Tier 1 (warm):  Redis   at {redis_url}");
    outputln!("  Tier 2 (hot):   Memory  (in-process)");
    outputln!();

    // Cold tier: SQLite
    if let Some(parent) = db_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let sqlite_config = SqliteBackendConfig {
        db_path: db_path.clone(),
        memory_strategy: MemoryStrategy::QueryOnly,
        max_connections: 5,
        table_name: "vectors".to_string(),
        ensure_schema: true,
    };
    let cold_tier: Arc<SqliteBackend> = Arc::new(SqliteBackend::new(sqlite_config).await?);
    outputln!("Created SQLite cold tier");

    // Warm tier: Redis
    let redis_tier = Arc::new(RedisBackend::with_prefix(&redis_url, "tiered:warm:").await?);
    outputln!("Created Redis warm tier");

    // Hot tier: Memory
    let hot_tier = Arc::new(MemoryBackend::new());
    outputln!("Created Memory hot tier");

    // Build chain (cold -> hot)
    let backend: Arc<dyn Backend> = TieredChainBuilder::new()
        .tier(cold_tier, TierConfig::new("sqlite"))
        .tier(
            redis_tier,
            TierConfig::new("redis")
                .max_vectors(10_000)
                .max_bytes(100 * 1024 * 1024),
        )
        .tier(
            hot_tier,
            TierConfig::new("memory")
                .max_vectors(2_000)
                .max_bytes(20 * 1024 * 1024),
        )
        .background_sync(true)
        .sync_interval_secs(60)
        .build()?;
    outputln!("Created three-tier backend\n");

    let namespace = "three-tier-demo";

    // Upsert a few chunks
    outputln!("--- Insert ---");
    let chunks = vec![
        Chunk::new(
            "doc-1",
            0,
            "First document",
            vec![0.1; 64],
            0,
            15,
            1,
            serde_json::json!({}),
            Some(namespace.to_string()),
        ),
        Chunk::new(
            "doc-2",
            0,
            "Second document",
            vec![0.2; 64],
            0,
            16,
            1,
            serde_json::json!({}),
            Some(namespace.to_string()),
        ),
    ];
    backend.upsert(&chunks).await?;
    outputln!("Inserted {} chunks", chunks.len());

    // Get (will promote through tiers on access)
    outputln!("\n--- Get ---");
    let one = backend.get("doc-1#0", Some(namespace)).await?;
    outputln!("get(doc-1#0): {}", one.is_some());

    // Search
    outputln!("\n--- Search ---");
    let query = SearchQuery::new(vec![0.1; 64])
        .metric(Metric::Cosine)
        .limit(5)
        .namespace(namespace);
    let results = backend.search(&query).await?;
    outputln!("Search returned {} results", results.len());

    // Cleanup
    backend.delete_namespace(namespace).await?;
    outputln!("\nCleaned up namespace");

    outputln!("\n=== Done ===");
    Ok(())
}
