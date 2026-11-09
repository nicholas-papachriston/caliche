//! Client MaxSim fallback, S3/SQLite multi-vector persistence, and PLAID coverage.

#![cfg(all(feature = "memory-backend", feature = "tokio"))]

#[cfg(feature = "s3-backend")]
use std::sync::Arc;

use async_trait::async_trait;
use dripstone::backend::indexed::{IndexedBackend, IndexedBackendConfig};
use dripstone::backend::memory::{InMemoryConfig, MemoryBackend};
use dripstone::index::spfresh::{CentroidIndex, SPFreshConfig};
use dripstone::{
    Backend, Capabilities, Chunk, Dripstone, DripstoneError, Embedder,
    LateInteractionIndexedBackend, MaxSimOptions, Metric, PlaidConfig, PlaidIndex, SearchQuery,
    SearchResult, multi_vectors_from_owned, pooled_vector_from_multi, search_with_maxsim_fallback,
};
use serde_json::json;

#[cfg(feature = "sqlite-backend")]
use dripstone::backend::sqlite::{MemoryStrategy, SqliteBackend, SqliteBackendConfig};

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

/// Dense-only backend: search ignores MaxSim (reports `native_maxsim = false`).
#[derive(Debug)]
struct DenseOnlyBackend {
    inner: MemoryBackend,
    caps: Capabilities,
}

impl DenseOnlyBackend {
    fn new() -> Self {
        let mut caps = Capabilities::memory();
        caps.native_maxsim = false;
        Self {
            inner: MemoryBackend::with_config(InMemoryConfig {
                default_ttl_secs: Some(86_400),
                max_entries: None,
                auto_cleanup: false,
            }),
            caps,
        }
    }
}

#[async_trait]
impl Backend for DenseOnlyBackend {
    fn capabilities(&self) -> Capabilities {
        self.caps.clone()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        self.inner.upsert(chunks).await
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        self.inner.get(id, namespace).await
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        self.inner.delete(id, namespace).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        self.inner.delete_document(doc_id, namespace).await
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        // Force dense-only: strip multi-vector request so Memory won't MaxSim-native.
        let mut dense = query.clone();
        dense.query_multi_vectors = None;
        dense.maxsim = None;
        self.inner.search(&dense).await
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.inner.namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        self.inner.delete_namespace(namespace).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        self.inner.count(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        self.inner.list(namespace, limit).await
    }
}

#[tokio::test]
async fn client_maxsim_fallback_reranks_non_native_backend() {
    let backend = DenseOnlyBackend::new();
    let good = make_multi_chunk("good", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns"));
    let bad = make_multi_chunk("bad", vec![vec![-1.0, 0.0], vec![0.0, -1.0]], Some("ns"));
    // Dense-similar decoy (pooled near query) but poor MaxSim.
    let decoy = make_multi_chunk("decoy", vec![vec![0.8, 0.2], vec![0.2, 0.8]], Some("ns"));
    backend
        .upsert(&[good.clone(), bad, decoy])
        .await
        .expect("upsert");

    let q = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let query = SearchQuery::new(pooled_vector_from_multi(&q))
        .limit(1)
        .metric(Metric::DotProduct)
        .namespace("ns")
        .with_multi_vectors(q)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct).overfetch(8));

    // Dense-only path may prefer decoy; fallback must recover "good".
    let hits = search_with_maxsim_fallback(&backend, &query)
        .await
        .expect("fallback");
    assert_eq!(hits[0].chunk.id, good.id);
    assert!(hits[0].chunk.multi_vectors.is_some());
}

#[tokio::test]
async fn late_interaction_backend_sets_native_maxsim_and_ranks() {
    let inner = MemoryBackend::new();
    let backend = LateInteractionIndexedBackend::new(
        inner,
        dripstone::LateInteractionIndexedBackendConfig {
            plaid: PlaidConfig {
                num_centroids: 8,
                probes: 4,
                candidate_budget: 64,
                metric: Metric::DotProduct,
            },
            auto_rebuild_threshold: 1,
            rebuild_on_search: true,
        },
    );
    assert!(backend.capabilities().native_maxsim);

    let chunks = vec![
        make_multi_chunk("good", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns")),
        make_multi_chunk("bad", vec![vec![-1.0, 0.0], vec![0.0, -1.0]], Some("ns")),
    ];
    backend.upsert(&chunks).await.expect("upsert");
    backend.flush().await.expect("flush");

    let q = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let query = SearchQuery::new(pooled_vector_from_multi(&q))
        .limit(1)
        .namespace("ns")
        .with_multi_vectors(q)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct));
    let hits = backend.search(&query).await.expect("search");
    assert_eq!(hits[0].chunk.id, "good#0");
}

#[test]
fn plaid_index_candidate_generation_is_performant() {
    use std::time::Instant;

    const N: usize = 1_000;
    const TOKENS: usize = 8;
    const DIM: usize = 32;

    let mut chunks = Vec::with_capacity(N);
    for i in 0..N {
        let tokens = (0..TOKENS)
            .map(|t| {
                (0..DIM)
                    .map(|d| ((i + t + d) as f32 * 0.01).sin())
                    .collect::<Vec<_>>()
            })
            .collect();
        chunks.push(make_multi_chunk(&format!("d{i}"), tokens, Some("ns")));
    }
    let index = PlaidIndex::new(PlaidConfig {
        num_centroids: 64,
        probes: 4,
        candidate_budget: 128,
        metric: Metric::DotProduct,
    });
    let build_start = Instant::now();
    index.rebuild(&chunks).expect("rebuild");
    let build_elapsed = build_start.elapsed();

    let q = multi_vectors_from_owned(
        (0..TOKENS)
            .map(|t| {
                (0..DIM)
                    .map(|d| ((t + d) as f32 * 0.01).sin())
                    .collect::<Vec<_>>()
            })
            .collect(),
    );
    let query = SearchQuery::new(pooled_vector_from_multi(&q))
        .limit(10)
        .namespace("ns")
        .with_multi_vectors(q)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct));
    let search_start = Instant::now();
    let hits = index.search(&query).expect("search");
    let search_elapsed = search_start.elapsed();

    assert_eq!(hits.len(), 10);
    assert!(
        build_elapsed.as_secs_f64() < 5.0,
        "PLAID rebuild too slow: {build_elapsed:?}"
    );
    assert!(
        search_elapsed.as_millis() < 500,
        "PLAID search too slow: {search_elapsed:?}"
    );
}

#[cfg(feature = "sqlite-backend")]
#[tokio::test]
async fn sqlite_persists_multi_vectors_round_trip() {
    let file = tempfile::NamedTempFile::new().expect("temp db");
    let config = SqliteBackendConfig {
        db_path: file.path().to_path_buf(),
        memory_strategy: MemoryStrategy::LoadAll,
        max_connections: 2,
        table_name: "vectors".to_string(),
        ensure_schema: true,
    };
    let backend = SqliteBackend::new(config).await.expect("open");

    let chunk = make_multi_chunk(
        "doc",
        vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        Some("ns"),
    );
    backend.upsert(&[chunk.clone()]).await.expect("upsert");

    // Bypass cache: open a second handle on the same DB file.
    let backend2 = SqliteBackend::new(SqliteBackendConfig {
        db_path: file.path().to_path_buf(),
        memory_strategy: MemoryStrategy::QueryOnly,
        max_connections: 2,
        table_name: "vectors".to_string(),
        ensure_schema: false,
    })
    .await
    .expect("reopen");
    let got = backend2
        .get(&chunk.id, Some("ns"))
        .await
        .expect("get")
        .expect("present");
    assert!(got.multi_vectors.is_some());
    assert_eq!(got.multi_vectors.as_ref().unwrap().len(), 2);
    assert_eq!(
        got.multi_vectors.as_ref().unwrap()[0].as_ref(),
        &[1.0, 0.0, 0.0]
    );
}

#[cfg(feature = "s3-backend")]
#[test]
fn s3_record_v2_round_trips_multi_vectors() {
    use dripstone::embedding::EmbeddingRecord;

    // Exercise serialize/deserialize through the same helpers used by S3Backend
    // by going through a public MemoryObjectStore ColdForge path is heavier;
    // unit-test the codec via EmbeddingRecord JSON + a local round-trip of the
    // V2 magic path using the backend's private API is not available, so we
    // verify persistence via MemoryObjectStore + ColdForge upsert/get instead
    // when s3 feature is on — see coldforge path below.
    let mv = multi_vectors_from_owned(vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    let record = EmbeddingRecord::new("id", vec![0.1_f32, 0.2]).with_multi_vectors(Arc::clone(&mv));
    let json = serde_json::to_string(&record).expect("ser");
    let back: EmbeddingRecord = serde_json::from_str(&json).expect("de");
    assert_eq!(back.multi_vectors.as_ref().unwrap().len(), 2);
}

#[cfg(feature = "s3-backend")]
#[tokio::test]
async fn s3_backend_memory_store_persists_multi_vectors() {
    use dripstone::backend::s3::{S3Backend, S3BackendConfig};
    use dripstone::storage::MemoryObjectStore;

    let store = MemoryObjectStore::shared();
    let backend = S3Backend::with_object_store(
        store,
        S3BackendConfig {
            bucket: "test".into(),
            key_prefix: "mv/".into(),
            enable_compression: false,
            batch_size: 10,
            cache_indices: true,
            ..S3BackendConfig::default()
        },
    );

    let chunk = make_multi_chunk("s3doc", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns"));
    backend.upsert(&[chunk.clone()]).await.expect("upsert");
    let got = backend
        .get(&chunk.id, Some("ns"))
        .await
        .expect("get")
        .expect("present");
    assert!(got.multi_vectors.is_some());
    assert_eq!(got.multi_vectors.as_ref().unwrap()[0].as_ref(), &[1.0, 0.0]);
}

#[tokio::test]
async fn indexed_backend_maxsim_reranks_hydrated_candidates() {
    let inner = MemoryBackend::new();
    let index = CentroidIndex::new(
        SPFreshConfig {
            num_centroids: 4,
            ..SPFreshConfig::default()
        },
        2,
    );
    let backend = IndexedBackend::new(
        inner,
        Box::new(index),
        IndexedBackendConfig {
            auto_rebuild_threshold: 1,
            fallback_to_brute_force: false,
            overfetch_factor: 2,
            ..Default::default()
        },
    );

    let chunks = vec![
        make_multi_chunk("good", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns")),
        make_multi_chunk("bad", vec![vec![-1.0, 0.0], vec![0.0, -1.0]], Some("ns")),
        make_multi_chunk("mid", vec![vec![0.5, 0.5], vec![0.5, 0.5]], Some("ns")),
    ];
    backend.upsert(&chunks).await.expect("upsert");
    backend.flush().await.expect("flush");

    let q = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let query = SearchQuery::new(pooled_vector_from_multi(&q))
        .limit(1)
        .metric(Metric::DotProduct)
        .namespace("ns")
        .with_multi_vectors(q)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct).overfetch(8));
    let hits = backend.search(&query).await.expect("search");
    assert_eq!(hits[0].chunk.id, "good#0");
    assert!(hits[0].chunk.multi_vectors.is_some());
}

#[derive(Debug)]
struct FixedEmbedder;

#[async_trait]
impl Embedder for FixedEmbedder {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DripstoneError> {
        Ok(texts.iter().map(|_| vec![0.5_f32, 0.5]).collect())
    }

    async fn embed_query(&self, _query: &str) -> Result<Vec<f32>, DripstoneError> {
        Ok(vec![0.5, 0.5])
    }

    fn dimension(&self) -> usize {
        2
    }

    fn name(&self) -> &'static str {
        "fixed"
    }
}

#[tokio::test]
async fn search_builder_client_maxsim_fallback_ranks_aligned_doc() {
    let backend = DenseOnlyBackend::new();
    backend
        .upsert(&[
            make_multi_chunk("good", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns")),
            make_multi_chunk("bad", vec![vec![-1.0, 0.0], vec![0.0, -1.0]], Some("ns")),
        ])
        .await
        .expect("upsert");

    let vf = Dripstone::new(backend, FixedEmbedder);
    let q = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let hits = vf
        .search("unused-text")
        .limit(1)
        .namespace("ns")
        .with_multi_vectors(q)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct).overfetch(8))
        .await
        .expect("search");
    assert_eq!(hits[0].chunk.id, "good#0");
}

#[test]
fn plaid_remove_chunk_excludes_document_from_search() {
    let index = PlaidIndex::new(PlaidConfig {
        num_centroids: 8,
        probes: 4,
        candidate_budget: 32,
        metric: Metric::DotProduct,
    });
    let chunks = vec![
        make_multi_chunk("keep", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns")),
        make_multi_chunk("drop", vec![vec![1.0, 0.0], vec![0.0, 1.0]], Some("ns")),
    ];
    index.rebuild(&chunks).expect("rebuild");
    index.remove_chunk("drop#0");

    let q = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    let query = SearchQuery::new(pooled_vector_from_multi(&q))
        .limit(5)
        .namespace("ns")
        .with_multi_vectors(q)
        .with_maxsim(MaxSimOptions::new(Metric::DotProduct));
    let hits = index.search(&query).expect("search");
    assert!(hits.iter().all(|h| h.chunk.id != "drop#0"));
    assert!(hits.iter().any(|h| h.chunk.id == "keep#0"));
}
