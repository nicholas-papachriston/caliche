//! ColdForge shards under a shared namespace WAL, with per-shard Memory hot tiers.

#![cfg(all(
    feature = "coldforge-backend",
    feature = "memory-backend",
    feature = "tokio"
))]

use std::sync::Arc;

use dripstone::backend::coldforge::ColdForgeBackend;
use dripstone::backend::memory::{InMemoryConfig, MemoryBackend};
use dripstone::{
    Backend, Chunk, ColdForgeShardOptions, MemoryObjectStore, Metric, SearchQuery,
    SharedNamespaceWal, SharedWalConfig, WalShardedBackend, build_memory_coldforge_wal_sharded,
};
use serde_json::json;

fn make_chunk(doc: &str, dim: usize, namespace: Option<&str>) -> Chunk {
    let mut vector = vec![0.0_f32; dim];
    for (i, b) in doc.bytes().enumerate() {
        vector[i % dim] += f32::from(b) / 255.0;
    }
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
    for v in &mut vector {
        *v /= norm;
    }
    Chunk::new(
        doc,
        0,
        format!("content-{doc}"),
        vector,
        0,
        8,
        1,
        json!({ "doc": doc }),
        namespace.map(str::to_string),
    )
}

#[tokio::test]
async fn coldforge_shards_under_shared_wal_search_and_balance() {
    const DIM: usize = 32;
    const N: usize = 400;
    const SHARDS: usize = 4;

    let backend = build_memory_coldforge_wal_sharded(ColdForgeShardOptions {
        shard_count: SHARDS,
        vector_dimension: DIM,
        num_centroids: 8,
        memory_hot_tier: true,
        ..ColdForgeShardOptions::default()
    })
    .expect("build coldforge sharded");

    let chunks: Vec<_> = (0..N)
        .map(|i| make_chunk(&format!("doc-{i}"), DIM, Some("giant")))
        .collect();
    for window in chunks.chunks(50) {
        backend.upsert(window).await.expect("upsert");
    }

    assert_eq!(backend.count(Some("giant")).await.expect("count"), N);
    let shard_counts = backend.shard_counts(Some("giant")).await.expect("sc");
    assert_eq!(shard_counts.iter().sum::<usize>(), N);
    assert!(
        shard_counts.iter().all(|&c| c > 0),
        "expected every ColdForge shard to hold data: {shard_counts:?}"
    );

    let probe = &chunks[42];
    let query = SearchQuery::new(probe.vector.clone())
        .limit(3)
        .metric(Metric::Cosine)
        .namespace("giant");
    let hits = backend.search(&query).await.expect("search");
    assert_eq!(hits[0].chunk.id, probe.id);
}

#[tokio::test]
async fn coldforge_hot_tier_receives_write_through() {
    let store = MemoryObjectStore::shared();
    let hot = Arc::new(MemoryBackend::with_config(InMemoryConfig {
        default_ttl_secs: Some(86_400),
        max_entries: None,
        auto_cleanup: false,
    }));
    let mut cfg = dripstone::backend::coldforge::ColdForgeConfig::default();
    cfg.bucket = "vf".to_string();
    cfg.key_prefix = "shard0/".to_string();
    cfg.vector_dimension = 8;
    cfg.num_centroids = 4;
    cfg.enable_compression = false;
    cfg.cache_size_bytes = 8 * 1024 * 1024;

    let coldforge = ColdForgeBackend::with_object_store_and_hot_tier(
        Arc::clone(&store) as Arc<dyn dripstone::storage::ObjectStore>,
        cfg,
        Arc::clone(&hot) as Arc<dyn Backend>,
    );

    let chunk = make_chunk("hot-doc", 8, Some("ns"));
    coldforge
        .upsert(std::slice::from_ref(&chunk))
        .await
        .expect("upsert");

    let from_hot = hot
        .get(&chunk.id, Some("ns"))
        .await
        .expect("hot get")
        .expect("present in hot tier");
    assert_eq!(from_hot.id, chunk.id);
}

#[tokio::test]
async fn shared_wal_recovers_coldforge_shards() {
    let store = MemoryObjectStore::shared();
    let options = ColdForgeShardOptions {
        shard_count: 3,
        vector_dimension: 16,
        num_centroids: 4,
        bucket: "vf".to_string(),
        shared_wal_prefix: "ns/wal/".to_string(),
        memory_hot_tier: true,
    };
    let backend = dripstone::build_coldforge_wal_sharded(Arc::clone(&store) as _, options.clone())
        .expect("build");

    let chunks = vec![
        make_chunk("a", 16, Some("ns")),
        make_chunk("b", 16, Some("ns")),
        make_chunk("c", 16, Some("ns")),
    ];
    backend.upsert(&chunks).await.expect("upsert");
    assert_eq!(backend.wal().len(), 3);

    // New ColdForge shard set + same durable shared WAL.
    let recovered = dripstone::build_coldforge_wal_sharded(store, options).expect("rebuild shards");
    let applied = recovered.recover().await.expect("recover");
    assert_eq!(applied, 3);
    assert_eq!(recovered.count(Some("ns")).await.expect("count"), 3);
    assert!(
        recovered
            .get(&chunks[1].id, Some("ns"))
            .await
            .expect("get")
            .is_some()
    );
}

#[tokio::test]
async fn from_shards_accepts_prebuilt_coldforge() {
    let store = MemoryObjectStore::shared();
    let mut shards = Vec::new();
    for i in 0..2 {
        let mut cfg = dripstone::backend::coldforge::ColdForgeConfig::default();
        cfg.bucket = "vf".to_string();
        cfg.key_prefix = format!("pre/{i}/");
        cfg.vector_dimension = 8;
        cfg.num_centroids = 2;
        cfg.enable_compression = false;
        shards.push(Arc::new(ColdForgeBackend::with_object_store(
            Arc::clone(&store) as _,
            cfg,
        )) as Arc<dyn Backend>);
    }
    let wal = SharedNamespaceWal::with_object_store(
        store,
        SharedWalConfig {
            bucket: "vf".to_string(),
            key_prefix: "pre-wal/".to_string(),
        },
    );
    let backend = WalShardedBackend::from_shards(shards, wal).expect("from_shards");
    assert_eq!(backend.shard_count(), 2);
    let chunk = make_chunk("prebuilt", 8, None);
    backend
        .upsert(std::slice::from_ref(&chunk))
        .await
        .expect("upsert");
    assert!(backend.get(&chunk.id, None).await.expect("get").is_some());
}
