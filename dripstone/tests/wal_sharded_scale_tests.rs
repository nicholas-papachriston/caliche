//! Scale tests for single-WAL multi-shard namespaces.
//!
//! Validates turbopuffer-inspired ownership (`hash(id) % N`), fan-out search
//! merge, durable WAL recovery via [`MemoryObjectStore`], and balance across
//! shards on a large synthetic corpus.

#![cfg(all(feature = "memory-backend", feature = "tokio"))]
#![expect(
    clippy::expect_used,
    reason = "integration tests use expect for concise failure messages"
)]

use std::collections::HashMap;
use std::sync::Arc;

use dripstone::backend::memory::{InMemoryConfig, MemoryBackend};
use dripstone::{
    Backend, Chunk, MemoryObjectStore, Metric, SearchQuery, ShardManager, SharedNamespaceWal,
    SharedWalConfig, WalShardedBackend,
};
use serde_json::json;

fn unlimited_memory() -> Arc<dyn Backend> {
    Arc::new(MemoryBackend::with_config(InMemoryConfig {
        default_ttl_secs: Some(86_400),
        max_entries: None,
        auto_cleanup: false,
    }))
}

fn make_chunk(doc: &str, dim: usize, namespace: Option<&str>) -> Chunk {
    let mut vector = vec![0.0_f32; dim];
    // Deterministic non-zero direction from doc id bytes.
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

fn build_sharded(shard_count: usize, wal: SharedNamespaceWal) -> WalShardedBackend {
    WalShardedBackend::builder()
        .shard_count(shard_count)
        .shard_factory(|_| unlimited_memory())
        .wal(wal)
        .build()
        .expect("sharded backend")
}

#[test]
fn shard_index_is_stable_and_covers_all_shards() {
    let manager = ShardManager::new(8);
    let mut counts = HashMap::new();
    for i in 0..2_000 {
        let idx = manager
            .shard_index_for_key(&format!("doc-{i}"))
            .expect("index");
        assert!(idx < 8);
        *counts.entry(idx).or_insert(0) += 1;
    }
    assert_eq!(counts.len(), 8, "expected all shards to receive keys");
    let min = *counts.values().min().expect("min");
    let max = *counts.values().max().expect("max");
    assert!(
        max - min < 400,
        "shard imbalance too high: min={min} max={max}"
    );
}

#[tokio::test]
async fn upsert_routes_to_owning_shard_and_search_merges() {
    let backend = build_sharded(4, SharedNamespaceWal::in_memory());
    let ns = Some("corpus");
    let chunks: Vec<_> = (0..40)
        .map(|i| make_chunk(&format!("doc-{i}"), 16, ns))
        .collect();
    backend.upsert(&chunks).await.expect("upsert");

    assert_eq!(backend.count(ns).await.expect("count"), 40);
    let counts = backend.shard_counts(ns).await.expect("shard counts");
    assert_eq!(counts.len(), 4);
    assert_eq!(counts.iter().sum::<usize>(), 40);
    assert!(counts.iter().all(|&c| c > 0), "empty shard: {counts:?}");

    let probe = &chunks[7];
    let got = backend
        .get(&probe.id, ns)
        .await
        .expect("get")
        .expect("present");
    assert_eq!(got.id, probe.id);

    let query = SearchQuery::new(probe.vector.clone())
        .limit(5)
        .metric(Metric::Cosine)
        .namespace("corpus");
    let results = backend.search(&query).await.expect("search");
    assert!(!results.is_empty());
    assert_eq!(results[0].chunk.id, probe.id);
}

#[tokio::test]
async fn durable_wal_recovers_after_empty_shards() {
    let store = MemoryObjectStore::shared();
    let wal = SharedNamespaceWal::with_object_store(
        store.clone(),
        SharedWalConfig {
            bucket: "vf".to_string(),
            key_prefix: "ns/large/wal/".to_string(),
        },
    );
    let backend = build_sharded(3, wal);
    let chunks = vec![
        make_chunk("alpha", 8, Some("ns")),
        make_chunk("beta", 8, Some("ns")),
        make_chunk("gamma", 8, Some("ns")),
    ];
    backend.upsert(&chunks).await.expect("upsert");
    assert_eq!(backend.wal().len(), 3);

    // Simulate process loss: new shard set + same durable WAL.
    let wal2 = SharedNamespaceWal::with_object_store(
        store,
        SharedWalConfig {
            bucket: "vf".to_string(),
            key_prefix: "ns/large/wal/".to_string(),
        },
    );
    let recovered = build_sharded(3, wal2);
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
async fn large_corpus_search_matches_monolithic_baseline() {
    const N: usize = 5_000;
    const DIM: usize = 32;
    const SHARDS: usize = 8;

    let sharded = build_sharded(SHARDS, SharedNamespaceWal::in_memory());
    let mono = unlimited_memory();

    let mut chunks = Vec::with_capacity(N);
    for i in 0..N {
        chunks.push(make_chunk(&format!("doc-{i}"), DIM, Some("giant")));
    }
    // Batch upsert in slices to keep memory pressure reasonable.
    for window in chunks.chunks(250) {
        sharded.upsert(window).await.expect("sharded upsert");
        mono.upsert(window).await.expect("mono upsert");
    }

    assert_eq!(sharded.count(Some("giant")).await.expect("c"), N);
    let shard_counts = sharded.shard_counts(Some("giant")).await.expect("sc");
    assert_eq!(shard_counts.iter().sum::<usize>(), N);
    assert!(
        shard_counts.iter().all(|&c| c > N / SHARDS / 3),
        "unexpectedly empty/skewed shards: {shard_counts:?}"
    );

    // Probe several ids: top-1 from sharded must match monolithic MemoryBackend.
    for probe_idx in [0_usize, 17, 999, 2500, 4999] {
        let probe = &chunks[probe_idx];
        let query = SearchQuery::new(probe.vector.clone())
            .limit(3)
            .metric(Metric::Cosine)
            .namespace("giant");
        let sharded_hit = sharded.search(&query).await.expect("sharded search");
        let mono_hit = mono.search(&query).await.expect("mono search");
        assert_eq!(
            sharded_hit[0].chunk.id, mono_hit[0].chunk.id,
            "top-1 mismatch at probe {probe_idx}"
        );
        assert_eq!(sharded_hit[0].chunk.id, probe.id);
    }
}

#[tokio::test]
async fn delete_namespace_is_wal_ordered_and_clears_all_shards() {
    let backend = build_sharded(4, SharedNamespaceWal::in_memory());
    let chunks: Vec<_> = (0..100)
        .map(|i| make_chunk(&format!("n-{i}"), 8, Some("ephemeral")))
        .collect();
    backend.upsert(&chunks).await.expect("upsert");
    backend
        .delete_namespace("ephemeral")
        .await
        .expect("delete ns");
    assert_eq!(backend.count(Some("ephemeral")).await.expect("count"), 0);
    let counts = backend
        .shard_counts(Some("ephemeral"))
        .await
        .expect("shard counts");
    assert!(counts.iter().all(|&c| c == 0));

    // Replay still ends empty because Clear is in the log after upserts.
    let applied = backend.replay_wal().await.expect("replay");
    assert!(applied >= 101);
    assert_eq!(backend.count(Some("ephemeral")).await.expect("count2"), 0);
}
