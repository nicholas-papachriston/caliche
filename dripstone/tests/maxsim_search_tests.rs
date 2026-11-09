//! Late-interaction MaxSim correctness and performance coverage.

#![cfg(all(feature = "memory-backend", feature = "tokio"))]

use std::sync::Arc;
use std::time::Instant;

use dripstone::backend::memory::{InMemoryConfig, MemoryBackend};
use dripstone::config::DripstoneConfig;
use dripstone::traits::Embedder;
use dripstone::{
    Backend, Chunk, Dripstone, MaxSimOptions, Metric, SearchQuery, exact_maxsim_search,
    multi_vectors_from_owned, pooled_vector_from_multi,
};
use serde_json::json;

#[cfg(feature = "coldforge-backend")]
use dripstone::{ColdForgeShardOptions, build_memory_coldforge_wal_sharded};

fn make_multi_chunk(doc: &str, tokens: Vec<Vec<f32>>, namespace: Option<&str>) -> Chunk {
    let mv = multi_vectors_from_owned(tokens);
    let pooled = pooled_vector_from_multi(&mv);
    Chunk::new(
        doc,
        0,
        format!("content-{doc}"),
        pooled,
        0,
        8,
        1,
        json!({ "doc": doc }),
        namespace.map(str::to_string),
    )
    .with_multi_vectors(mv)
}

#[tokio::test]
async fn memory_exact_maxsim_ranks_aligned_document_first() {
    let backend = MemoryBackend::with_config(InMemoryConfig {
        default_ttl_secs: Some(86_400),
        max_entries: None,
        auto_cleanup: false,
    });

    let good = make_multi_chunk("good", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns"));
    let bad = make_multi_chunk("bad", vec![vec![-1.0, 0.0], vec![0.0, -1.0]], Some("ns"));
    // Decoy: dense-similar pooled vector but poor token alignment.
    let decoy = make_multi_chunk("decoy", vec![vec![0.7, 0.7], vec![0.7, 0.7]], Some("ns"));
    backend
        .upsert(&[good.clone(), bad, decoy])
        .await
        .expect("upsert");

    let q_tokens = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let query = SearchQuery::new(pooled_vector_from_multi(&q_tokens))
        .limit(1)
        .metric(Metric::DotProduct)
        .namespace("ns")
        .with_multi_vectors(q_tokens)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct).overfetch(8));

    let hits = backend.search(&query).await.expect("search");
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].chunk.id, good.id);
    assert!(
        hits[0].score > 1.5,
        "expected strong MaxSim score, got {}",
        hits[0].score
    );
}

#[tokio::test]
async fn exact_maxsim_helper_matches_memory_backend() {
    let chunks = vec![
        make_multi_chunk("best", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns")),
        make_multi_chunk("mid", vec![vec![1.0, 0.0], vec![0.5, 0.5]], Some("ns")),
        make_multi_chunk("worst", vec![vec![-1.0, 0.0], vec![0.0, -1.0]], Some("ns")),
    ];
    let backend = MemoryBackend::new();
    backend.upsert(&chunks).await.expect("upsert");

    let q_tokens = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let query = SearchQuery::new(pooled_vector_from_multi(&q_tokens))
        .limit(2)
        .metric(Metric::DotProduct)
        .namespace("ns")
        .with_multi_vectors(Arc::clone(&q_tokens))
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct));

    let helper = exact_maxsim_search(&chunks, &query).expect("exact");
    let via_backend = backend.search(&query).await.expect("search");
    assert_eq!(helper.len(), via_backend.len());
    for (a, b) in helper.iter().zip(via_backend.iter()) {
        assert_eq!(a.chunk.id, b.chunk.id);
        assert!((a.score - b.score).abs() < 1e-5);
    }
    assert_eq!(helper[0].chunk.id, "best#0");
    assert_eq!(helper[1].chunk.id, "mid#0");
}

#[cfg(feature = "coldforge-backend")]
#[tokio::test]
async fn coldforge_wal_sharded_maxsim_reranks_after_ann() {
    const DIM: usize = 16;
    const TOKENS: usize = 4;
    const N: usize = 120;

    let backend = build_memory_coldforge_wal_sharded(ColdForgeShardOptions {
        shard_count: 4,
        vector_dimension: DIM,
        num_centroids: 8,
        memory_hot_tier: true,
        ..ColdForgeShardOptions::default()
    })
    .expect("build");

    let mut chunks = Vec::with_capacity(N);
    for i in 0..N {
        // Distractors: orthogonal / anti-aligned tokens so MaxSim cannot tie the target.
        let tokens = (0..TOKENS)
            .map(|t| {
                let mut v = vec![0.0_f32; DIM];
                v[(DIM / 2 + (i + t) % (DIM / 2)) % DIM] = 1.0;
                v
            })
            .collect();
        chunks.push(make_multi_chunk(&format!("doc-{i}"), tokens, Some("giant")));
    }
    // Unique target: exact match to the query token basis on [0..TOKENS).
    chunks[42] = make_multi_chunk(
        "doc-42",
        (0..TOKENS)
            .map(|t| {
                let mut v = vec![0.0_f32; DIM];
                v[t % DIM] = 1.0;
                v
            })
            .collect(),
        Some("giant"),
    );

    for window in chunks.chunks(40) {
        backend.upsert(window).await.expect("upsert");
    }
    backend.flush().await.expect("flush");

    let q_tokens = multi_vectors_from_owned(
        (0..TOKENS)
            .map(|t| {
                let mut v = vec![0.0_f32; DIM];
                v[t % DIM] = 1.0;
                v
            })
            .collect(),
    );
    let query = SearchQuery::new(pooled_vector_from_multi(&q_tokens))
        .limit(3)
        .metric(Metric::DotProduct)
        .namespace("giant")
        .with_multi_vectors(q_tokens)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct).overfetch(10));

    let hits = backend.search(&query).await.expect("search");
    assert!(!hits.is_empty());
    assert_eq!(hits[0].chunk.id, "doc-42#0");
    assert!(hits[0].chunk.multi_vectors.is_some());
}

/// Microbench: batched MaxSim over a mid-size corpus stays interactive.
#[test]
fn maxsim_score_throughput_is_performant() {
    const DOCS: usize = 2_000;
    const Q_TOKENS: usize = 8;
    const D_TOKENS: usize = 16;
    const DIM: usize = 32;

    let query = multi_vectors_from_owned(
        (0..Q_TOKENS)
            .map(|t| (0..DIM).map(|d| ((t + d) as f32).sin()).collect::<Vec<_>>())
            .collect(),
    );
    let docs: Vec<_> = (0..DOCS)
        .map(|i| {
            multi_vectors_from_owned(
                (0..D_TOKENS)
                    .map(|t| {
                        (0..DIM)
                            .map(|d| ((i + t + d) as f32 * 0.01).cos())
                            .collect::<Vec<_>>()
                    })
                    .collect(),
            )
        })
        .collect();

    let start = Instant::now();
    let mut checksum = 0.0_f32;
    for doc in &docs {
        checksum += dripstone::maxsim_score(&query, doc, Metric::DotProduct).expect("score");
    }
    let elapsed = start.elapsed();
    let docs_per_sec = DOCS as f64 / elapsed.as_secs_f64().max(1e-9);

    assert!(checksum.is_finite());
    // Debug builds vary; keep a conservative floor that still catches pathological O(n^3).
    assert!(
        docs_per_sec > 200.0,
        "MaxSim too slow: {docs_per_sec:.1} docs/s over {elapsed:?}"
    );
}

#[derive(Debug)]
struct TokenSensitiveEmbedder;

fn token_basis(text: &str) -> Vec<f32> {
    match text {
        "alpha" => vec![1.0, 0.0],
        "beta" => vec![0.0, 1.0],
        "decoy" => vec![0.7, 0.7],
        _ => vec![0.1, 0.1],
    }
}

#[async_trait::async_trait]
impl Embedder for TokenSensitiveEmbedder {
    async fn embed(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, dripstone::errors::DripstoneError> {
        Ok(texts.iter().copied().map(token_basis).collect())
    }

    async fn embed_query(
        &self,
        query: &str,
    ) -> Result<Vec<f32>, dripstone::errors::DripstoneError> {
        Ok(token_basis(query))
    }

    fn dimension(&self) -> usize {
        2
    }

    fn name(&self) -> &'static str {
        "token-sensitive"
    }
}

#[tokio::test]
async fn ingest_late_interaction_attaches_token_vectors_and_ranks() {
    let backend = MemoryBackend::new();
    let config = DripstoneConfig::default().with_search(|search| search.late_interaction(true));
    let vf = Dripstone::with_config(backend, TokenSensitiveEmbedder, config);

    vf.ingest("aligned", "alpha beta")
        .namespace("ns")
        .await
        .expect("ingest aligned");
    vf.ingest("decoy", "decoy decoy")
        .namespace("ns")
        .await
        .expect("ingest decoy");

    let aligned = vf
        .backend()
        .get("aligned#0", Some("ns"))
        .await
        .expect("get")
        .expect("chunk");
    let token_count = aligned
        .multi_vectors
        .as_ref()
        .map_or(0, |vectors| vectors.len());
    assert_eq!(
        token_count, 2,
        "ingest should attach one vector per unique token"
    );

    let q_tokens = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let hits = vf
        .search("alpha beta")
        .limit(1)
        .namespace("ns")
        .with_multi_vectors(q_tokens)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct).overfetch(8))
        .await
        .expect("search");
    assert_eq!(hits[0].chunk.id, "aligned#0");
}
