//! Vector Stores Example
//!
//! This example demonstrates:
//! - Dripstone vector database operations
//! - Similarity search with different metrics
//! - Namespaces for data isolation
//! - Metadata filtering
//! - Batch operations
//! - ColdForge S3-backed storage
//! - Tiered backend (hot/cold tiers)
//! - Agent with tiered vector config (VectorBackendConfig::Tiered)
//!
//! Best practices shown:
//! - Efficient vector storage and retrieval
//! - Similarity metric selection
//! - Cost-effective storage for large-scale deployments
//!
//! Run with:
//! ```bash
//! cargo run --example vector_stores --features vector-memory,vector-store-coldforge,vector-store-s3
//! # Tiered config example (memory + redis):
//! cargo run --example vector_stores --features vector-memory,vector-store-redis
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::AgentBuilder;
use caliche::prelude::ProviderType;
use caliche::vecstore::{IndexUpdateMode, IndexedBackendConfig, TierConfig, VectorBackendConfig};
use dripstone::backend::memory::MemoryBackend;
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::traits::Backend;
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    basic_operations().await?;
    similarity_search_example().await?;
    namespace_example().await?;
    metadata_filtering_example().await?;
    Box::pin(coldforge_example()).await?;
    tiered_backend_example().await?;
    tiered_config_agent_example().await?;
    deferred_indexed_backend_example().await?;

    Ok(())
}

/// Example 1: Basic CRUD operations
async fn basic_operations() -> Result<()> {
    println!("Example 1: Basic CRUD Operations\n");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let chunk = Chunk::new(
        "doc:1",
        0,
        "Introduction to AI",
        vec![0.1, 0.2, 0.3, 0.4],
        0,
        20,
        1,
        json!({
            "title": "Introduction to AI",
            "category": "technology",
            "year": 2024
        }),
        None,
    );

    backend.upsert(&[chunk.clone()]).await?;
    println!("Inserted record: doc:1");

    let retrieved = backend.get("doc:1#0", None).await?;
    if let Some(chunk) = retrieved {
        println!(
            "Retrieved: {} (vector dim: {})\n",
            chunk.id,
            chunk.vector.len()
        );
    }

    Ok(())
}

/// Example 2: Similarity search with different metrics
async fn similarity_search_example() -> Result<()> {
    println!("Example 2: Similarity Search\n");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let documents = vec![
        ("doc:ai", vec![0.9, 0.1, 0.2], "Artificial Intelligence"),
        ("doc:ml", vec![0.85, 0.15, 0.25], "Machine Learning"),
        ("doc:dl", vec![0.88, 0.12, 0.22], "Deep Learning"),
    ];

    let mut chunks = Vec::new();
    for (id, vector, title) in documents {
        let chunk = Chunk::new(
            id,
            0,
            title,
            vector,
            0,
            title.len(),
            1,
            json!({
                "title": title
            }),
            None,
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    let query_vector: Arc<[f32]> = Arc::from([0.87f32, 0.13f32, 0.23f32]);
    let query = SearchQuery {
        vector: query_vector,
        limit: 3,
        metric: Metric::Cosine,
        namespace: None,
        filter: None,
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    println!("Cosine Similarity (top 3):");
    for (i, result) in results.iter().enumerate() {
        let title = result
            .chunk
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        println!("  {}. {} - Score: {:.4}", i + 1, title, result.score);
    }
    println!();

    Ok(())
}

/// Example 3: Working with namespaces
async fn namespace_example() -> Result<()> {
    println!("Example 3: Namespaces\n");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let chunk1 = Chunk::new(
        "doc:1",
        0,
        "Production data",
        vec![0.1, 0.2, 0.3],
        0,
        15,
        1,
        json!({"env": "prod"}),
        Some("production".to_string()),
    );

    let chunk2 = Chunk::new(
        "doc:1",
        0,
        "Staging data",
        vec![0.4, 0.5, 0.6],
        0,
        13,
        1,
        json!({"env": "staging"}),
        Some("staging".to_string()),
    );

    backend.upsert(&[chunk1, chunk2]).await?;

    let namespaces = backend.namespaces().await?;
    println!("Available namespaces: {:?}", namespaces);

    let prod_count = backend.count(Some("production")).await?;
    let staging_count = backend.count(Some("staging")).await?;

    println!(
        "Record counts - Production: {}, Staging: {}\n",
        prod_count, staging_count
    );

    Ok(())
}

/// Example 4: Metadata filtering
async fn metadata_filtering_example() -> Result<()> {
    println!("Example 4: Metadata Filtering\n");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    let documents = vec![
        (
            "paper:1",
            vec![0.1, 0.2, 0.3],
            "Attention Is All You Need",
            json!({
                "title": "Attention Is All You Need",
                "year": 2017,
                "category": "nlp"
            }),
        ),
        (
            "paper:2",
            vec![0.15, 0.25, 0.35],
            "BERT",
            json!({
                "title": "BERT",
                "year": 2018,
                "category": "nlp"
            }),
        ),
        (
            "paper:3",
            vec![0.8, 0.1, 0.2],
            "ResNet",
            json!({
                "title": "ResNet",
                "year": 2015,
                "category": "vision"
            }),
        ),
    ];

    let mut chunks = Vec::new();
    for (id, vector, content, metadata) in documents {
        let chunk = Chunk::new(id, 0, content, vector, 0, content.len(), 1, metadata, None);
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    let query_vector: Arc<[f32]> = Arc::from([0.12f32, 0.22f32, 0.32f32]);

    // Note: Filter support depends on backend capabilities
    // For now, we'll do a simple search and filter client-side
    let query = SearchQuery {
        vector: query_vector,
        limit: 5,
        metric: Metric::Cosine,
        namespace: None,
        filter: None, // Backend filter support would go here
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    // Filter client-side for category='nlp'
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| r.chunk.metadata.get("category").and_then(|v| v.as_str()) == Some("nlp"))
        .collect();

    println!("Filtered search (category='nlp'):");
    for result in filtered {
        let title = result
            .chunk
            .metadata
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown");
        println!("  - {} (score: {:.4})", title, result.score);
    }
    println!();

    Ok(())
}

/// Example 5: ColdForge backend
#[cfg_attr(
    not(feature = "vector-store-coldforge"),
    expect(
        clippy::unused_async,
        reason = "the ColdForge body needs vector-store-coldforge"
    )
)]
async fn coldforge_example() -> Result<()> {
    println!("Example 5: ColdForge S3-Backed Storage\n");

    #[cfg(feature = "vector-store-coldforge")]
    {
        let config = VectorBackendConfig::ColdForge {
            bucket: std::env::var("S3_BUCKET").unwrap_or_else(|_| "my-vectors".to_string()),
            prefix: "coldforge/".to_string(),
            region: std::env::var("AWS_REGION").ok(),
            dimension: 1536,
            compression: true,
        };

        let _backend = config.build_backend().await?;
        println!("ColdForge backend created with in-memory hot tier");

        // Use ColdForge with AgentBuilder
        let mut agent = AgentBuilder::new()
            .provider(ProviderType::OpenAI)
            .vector_backend_config(config)
            .enable_tool("semantic_search")
            .build()
            .await?;

        agent
            .chat("Index this document with ID doc1: Rust is a systems programming language.")
            .await?;

        let response = agent
            .chat("What programming language is mentioned?")
            .await?;

        println!("RAG Query Result: {}\n", response.text());
    }

    #[cfg(not(feature = "vector-store-coldforge"))]
    {
        println!("(Skipped: Enable 'vector-store-coldforge' feature)\n");
    }

    Ok(())
}

/// Example 6: Tiered backend
#[cfg_attr(
    not(all(feature = "vector-store-coldforge", feature = "vector-store-s3")),
    expect(
        clippy::unused_async,
        reason = "the tiered body needs vector-store-coldforge and vector-store-s3"
    )
)]
async fn tiered_backend_example() -> Result<()> {
    println!("Example 6: Tiered Backend (Hot/Cold)\n");

    #[cfg(all(feature = "vector-store-coldforge", feature = "vector-store-s3"))]
    {
        use dripstone::backend::coldforge::{ColdForgeBackend, ColdForgeConfig};
        use dripstone::backend::s3::{S3Backend, S3BackendConfig};
        use dripstone::{TierConfig, TieredChainBuilder};

        let mut hot_config = ColdForgeConfig::default();
        hot_config.bucket =
            std::env::var("TIERED_HOT_BUCKET").unwrap_or_else(|_| "hot-tier".to_string());
        hot_config.key_prefix = "hot/".to_string();
        hot_config.region = std::env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());
        hot_config.cache_size_bytes = 2048 * 1024 * 1024;
        hot_config.num_centroids = 256;
        hot_config.wal_batch_interval_ms = 1000;
        hot_config.enable_compression = true;
        hot_config.vector_dimension = 1536;
        hot_config.hot_tier = None;
        hot_config.access_key_id = None;
        hot_config.secret_access_key = None;
        hot_config.session_token = None;
        // object_store_config is set to None by Default::default() when the field exists
        let hot_tier = Arc::new(ColdForgeBackend::new(hot_config).await?);
        println!("Hot tier created (ColdForge)");

        let cold_config = S3BackendConfig {
            bucket: std::env::var("TIERED_COLD_BUCKET").unwrap_or_else(|_| "cold-tier".to_string()),
            key_prefix: "cold/".to_string(),
            enable_compression: true,
            batch_size: 100,
            cache_indices: true,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            region: None,
        };
        let cold_tier = Arc::new(S3Backend::with_config(cold_config).await?);
        println!("Cold tier created (S3)");

        let _tiered_backend = TieredChainBuilder::new()
            .tier(cold_tier, TierConfig::new("s3"))
            .tier(
                hot_tier,
                TierConfig::new("coldforge")
                    .max_vectors(10000)
                    .max_bytes(2 * 1024 * 1024 * 1024)
                    .promotion_threshold(2),
            )
            .build()?;

        println!("Tiered backend configured:");
        println!("  Promotion threshold: 2 accesses");
        println!("  Hot tier max vectors: 10,000");
        println!("  Hot tier max size: 2GB");
        println!("\nVectors are automatically:");
        println!("  - Promoted to hot tier after frequent access");
        println!("  - Evicted to cold tier when hot tier is full");
        println!("  - Retrieved from cold tier when needed\n");
    }

    #[cfg(not(all(feature = "vector-store-coldforge", feature = "vector-store-s3")))]
    {
        println!("(Skipped: Enable 'vector-store-coldforge' and 'vector-store-s3' features)\n");
    }

    Ok(())
}

/// Example 7: Agent with tiered vector config (VectorBackendConfig::Tiered)
///
/// Uses the config-based API: N tiers (cold -> hot) via vector_backend_config.
/// Run with: cargo run --example vector_stores --features vector-memory,vector-store-redis
async fn tiered_config_agent_example() -> Result<()> {
    println!("Example 7: Agent with Tiered Vector Config (Memory + Redis)\n");

    #[cfg(feature = "vector-store-redis")]
    {
        let redis_url =
            std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1/".to_string());
        let config = VectorBackendConfig::Tiered {
            tiers: vec![
                (
                    VectorBackendConfig::Memory {
                        ttl_secs: 3600,
                        max_entries: Some(10_000),
                    },
                    TierConfig::new("hot")
                        .max_vectors(10_000)
                        .promotion_threshold(1),
                ),
                (
                    VectorBackendConfig::Redis {
                        url: redis_url.clone(),
                        prefix: "tiered:warm:".to_string(),
                    },
                    TierConfig::new("warm")
                        .max_vectors(100_000)
                        .promotion_threshold(2),
                ),
            ],
            write_strategy: None,
            search_strategy: None,
            background_sync: true,
            sync_interval_secs: 60,
        };

        let _agent = AgentBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4o-mini")
            .vector_backend_config(config)
            .enable_tool("semantic_search")
            .build()
            .await?;

        println!("Agent built with tiered vector backend:");
        println!("  Tier 0 (cold): Memory, 10k vectors, promotion after 1 access");
        println!(
            "  Tier 1 (hot):  Redis ({}), 100k vectors, promotion after 2 accesses",
            redis_url
        );
        println!("  Background sync: every 60s\n");
    }

    #[cfg(not(feature = "vector-store-redis"))]
    {
        println!("(Skipped: Enable 'vector-store-redis' for tiered config example)\n");
    }

    Ok(())
}

/// Example 8: Deferred ANN indexing on a local indexed backend
///
/// Uses `IndexUpdateMode::Deferred` so upserts commit to storage first and HNSW
/// inserts queue until flush or tail scan on search.
async fn deferred_indexed_backend_example() -> Result<()> {
    println!("Example 8: Deferred indexed backend (Superstore pattern)\n");

    #[cfg(feature = "vector-memory")]
    {
        use dripstone::Metric;
        use dripstone::core::SearchQuery;

        let config = VectorBackendConfig::Indexed {
            inner: Box::new(VectorBackendConfig::Memory {
                ttl_secs: 3600,
                max_entries: Some(10_000),
            }),
            dimension: 4,
            hnsw_config: None,
            config: Some(IndexedBackendConfig {
                index_update_mode: IndexUpdateMode::Deferred,
                scan_unindexed_tail: true,
                deferred_flush_threshold: 128,
                deferred_search_flush_threshold: 0,
                max_tail_scan_records: 0,
                auto_rebuild_threshold: 0,
                fallback_to_brute_force: true,
                overfetch_factor: 2,
            }),
        };

        let backend = config.build_backend().await?;
        let chunk = Chunk::new(
            "doc",
            0,
            "Deferred ingest path",
            vec![1.0_f32, 0.0, 0.0, 0.0],
            0,
            20,
            1,
            json!({ "source": "example" }),
            None,
        );
        backend.upsert(&[chunk]).await?;

        let query = SearchQuery::new(vec![1.0_f32, 0.0, 0.0, 0.0])
            .metric(Metric::Cosine)
            .limit(1);
        let hits = backend.search(&query).await?;
        println!(
            "Deferred indexed search returned {} hit(s); top id={}\n",
            hits.len(),
            hits.first()
                .map(|hit| hit.chunk.id.as_str())
                .unwrap_or("<none>")
        );
    }

    #[cfg(not(feature = "vector-memory"))]
    {
        println!("(Skipped: Enable 'vector-memory' for deferred indexed example)\n");
    }

    Ok(())
}
