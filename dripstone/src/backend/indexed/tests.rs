use super::*;
use crate::core::{Capabilities, Chunk, Metric, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::index::HnswIndex;
use crate::index::hnsw::HnswConfig;
use crate::search::Filter;
use crate::traits::Backend;
use async_trait::async_trait;
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct StubBackend {
    chunks: tokio::sync::RwLock<Vec<Chunk>>,
    get_calls: AtomicUsize,
}

impl StubBackend {
    fn new() -> Self {
        Self {
            chunks: tokio::sync::RwLock::new(Vec::new()),
            get_calls: AtomicUsize::new(0),
        }
    }

    fn get_call_count(&self) -> usize {
        self.get_calls.load(Ordering::Relaxed)
    }

    fn reset_get_call_count(&self) {
        self.get_calls.store(0, Ordering::Relaxed);
    }
}

#[async_trait]
impl Backend for StubBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities::memory()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let mut store = self.chunks.write().await;
        for c in chunks {
            store.retain(|existing| existing.id != c.id);
            store.push(c.clone());
        }
        drop(store);
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        _namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        self.get_calls.fetch_add(1, Ordering::Relaxed);
        let store = self.chunks.read().await;
        let result = store.iter().find(|c| c.id == id).cloned();
        drop(store);
        Ok(result)
    }

    async fn delete(&self, id: &str, _namespace: Option<&str>) -> Result<(), DripstoneError> {
        let mut store = self.chunks.write().await;
        store.retain(|c| c.id != id);
        drop(store);
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        _namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let mut store = self.chunks.write().await;
        let prefix = format!("{doc_id}#");
        let before = store.len();
        store.retain(|c| !c.id.starts_with(&prefix));
        Ok(before - store.len())
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let store = self.chunks.read().await;
        let mut results: Vec<SearchResult> = store
            .iter()
            .map(|c| {
                let similarity = query
                    .metric
                    .compute(query.query(), &c.vector)
                    .unwrap_or(0.0);
                SearchResult::new(c.clone(), similarity, 0)
            })
            .collect();
        drop(store);
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(query.limit);
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i;
        }
        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        Ok(vec![])
    }

    async fn delete_namespace(&self, _namespace: &str) -> Result<(), DripstoneError> {
        Ok(())
    }

    async fn count(&self, _namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let store = self.chunks.read().await;
        Ok(store.len())
    }

    async fn list(
        &self,
        _namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let store = self.chunks.read().await;
        let all: Vec<Chunk> = store.clone();
        drop(store);
        Ok(match limit {
            Some(l) => all.into_iter().take(l).collect(),
            None => all,
        })
    }
}

fn make_chunk(
    id: &str,
    vector: Vec<f32>,
    namespace: Option<&str>,
    metadata: serde_json::Value,
) -> Chunk {
    Chunk {
        id: id.to_string(),
        doc_id: "doc".to_string(),
        content: id.to_string(),
        vector: Arc::from(vector),
        start_offset: 0,
        end_offset: 0,
        chunk_index: 0,
        total_chunks: 1,
        metadata,
        namespace: namespace.map(ToOwned::to_owned),
        multi_vectors: None,
    }
}

#[tokio::test]
async fn indexed_search_returns_results() {
    let dim = 4;
    let hnsw = HnswIndex::new(HnswConfig::default(), dim);
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(hnsw),
        IndexedBackendConfig {
            auto_rebuild_threshold: 0,
            fallback_to_brute_force: true,
            overfetch_factor: 2,
            ..Default::default()
        },
    );

    let chunks = vec![
        make_chunk("a#0", vec![1.0, 0.0, 0.0, 0.0], None, json!({})),
        make_chunk("b#0", vec![0.0, 1.0, 0.0, 0.0], None, json!({})),
        make_chunk("c#0", vec![0.9, 0.1, 0.0, 0.0], None, json!({})),
    ];
    backend.upsert(&chunks).await.unwrap();

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(2)
        .metric(Metric::Cosine);
    let results = backend.search(&query).await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].chunk.id, "a#0");
}

#[tokio::test]
async fn fallback_to_brute_force_when_index_empty() {
    let dim = 4;
    let hnsw = HnswIndex::new(HnswConfig::default(), dim);
    let stub = StubBackend::new();
    {
        let chunk = make_chunk("x#0", vec![1.0, 0.0, 0.0, 0.0], None, json!({}));
        stub.upsert(&[chunk]).await.unwrap();
    }

    let backend = IndexedBackend::new(
        stub,
        Box::new(hnsw),
        IndexedBackendConfig {
            auto_rebuild_threshold: 0,
            fallback_to_brute_force: true,
            overfetch_factor: 2,
            ..Default::default()
        },
    );

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0]).limit(5);
    let results = backend.search(&query).await.unwrap();
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn delete_removes_from_index() {
    let dim = 4;
    let hnsw = HnswIndex::new(HnswConfig::default(), dim);
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(hnsw),
        IndexedBackendConfig::default(),
    );

    let chunks = vec![
        make_chunk("a#0", vec![1.0, 0.0, 0.0, 0.0], None, json!({})),
        make_chunk("b#0", vec![0.0, 1.0, 0.0, 0.0], None, json!({})),
    ];
    backend.upsert(&chunks).await.unwrap();

    backend.delete("a#0", None).await.unwrap();

    assert_eq!(backend.index().len(), 1);
}

#[tokio::test]
async fn indexed_search_uses_native_filter_to_reduce_fetches() {
    let dim = 4;
    let hnsw = HnswIndex::new(
        HnswConfig {
            ef_search: 32,
            rerank_overfetch: 4,
            ..Default::default()
        },
        dim,
    );
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(hnsw),
        IndexedBackendConfig {
            auto_rebuild_threshold: 0,
            fallback_to_brute_force: false,
            overfetch_factor: 4,
            ..Default::default()
        },
    );

    let chunks = vec![
        make_chunk(
            "match-a#0",
            vec![1.0, 0.0, 0.0, 0.0],
            Some("forge"),
            json!({"domain": "docs", "tier": 1}),
        ),
        make_chunk(
            "match-b#0",
            vec![0.99, 0.01, 0.0, 0.0],
            Some("forge"),
            json!({"domain": "docs", "tier": 1}),
        ),
        make_chunk(
            "other-a#0",
            vec![0.98, 0.02, 0.0, 0.0],
            Some("forge"),
            json!({"domain": "other", "tier": 0}),
        ),
        make_chunk(
            "other-b#0",
            vec![0.97, 0.03, 0.0, 0.0],
            Some("forge"),
            json!({"domain": "other", "tier": 0}),
        ),
        make_chunk(
            "other-c#0",
            vec![0.96, 0.04, 0.0, 0.0],
            Some("forge"),
            json!({"domain": "other", "tier": 0}),
        ),
        make_chunk(
            "other-d#0",
            vec![0.95, 0.05, 0.0, 0.0],
            Some("forge"),
            json!({"domain": "other", "tier": 0}),
        ),
    ];
    backend.upsert(&chunks).await.unwrap();

    let supported_query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(2)
        .metric(Metric::Cosine)
        .namespace("forge")
        .filter(Filter::Equal {
            field: "domain".to_string(),
            value: json!("docs"),
        });
    let supported_results = backend.search(&supported_query).await.unwrap();
    assert_eq!(supported_results.len(), 2);
    let supported_fetches = backend.inner.get_call_count();

    backend.inner.reset_get_call_count();

    let fallback_query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(2)
        .metric(Metric::Cosine)
        .namespace("forge")
        .filter(Filter::GreaterThan {
            field: "tier".to_string(),
            value: json!(0),
        });
    let fallback_results = backend.search(&fallback_query).await.unwrap();
    assert_eq!(fallback_results.len(), 2);
    let fallback_fetches = backend.inner.get_call_count();

    assert!(supported_fetches < fallback_fetches);
}

#[tokio::test]
async fn deferred_tail_scan_fetches_only_overfetch_candidates() {
    let dim = 4;
    let hnsw = HnswIndex::new(HnswConfig::default(), dim);
    let stub = StubBackend::new();
    let backend = IndexedBackend::new(
        stub,
        Box::new(hnsw),
        IndexedBackendConfig {
            auto_rebuild_threshold: 0,
            fallback_to_brute_force: false,
            overfetch_factor: 2,
            index_update_mode: IndexUpdateMode::Deferred,
            scan_unindexed_tail: true,
            deferred_flush_threshold: 0,
            deferred_search_flush_threshold: 0,
            max_tail_scan_records: 0,
        },
    );

    let chunks: Vec<Chunk> = (0..50)
        .map(|i| {
            make_chunk(
                &format!("vec-{i}"),
                vec![i as f32 / 50.0, 0.0, 0.0, 0.0],
                None,
                json!({}),
            )
        })
        .collect();
    backend.upsert(&chunks).await.unwrap();
    assert_eq!(backend.pending_index_count(), 50);

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(5)
        .metric(Metric::Cosine);
    backend.inner.reset_get_call_count();
    let results = backend.search(&query).await.unwrap();
    assert!(!results.is_empty());
    assert!(
        backend.inner.get_call_count() <= 10,
        "expected at most overfetch(limit) tail fetches, got {}",
        backend.inner.get_call_count()
    );
}

#[tokio::test]
async fn deferred_tail_scan_scores_from_record_metadata_without_prefetch() {
    let dim = 4;
    let hnsw = HnswIndex::new(HnswConfig::default(), dim);
    let stub = StubBackend::new();
    let backend = IndexedBackend::new(
        stub,
        Box::new(hnsw),
        IndexedBackendConfig {
            auto_rebuild_threshold: 0,
            fallback_to_brute_force: false,
            overfetch_factor: 2,
            index_update_mode: IndexUpdateMode::Deferred,
            scan_unindexed_tail: true,
            deferred_flush_threshold: 0,
            deferred_search_flush_threshold: 0,
            max_tail_scan_records: 0,
        },
    );

    let chunks = vec![
        make_chunk(
            "match#0",
            vec![1.0, 0.0, 0.0, 0.0],
            Some("forge"),
            json!({"domain": "docs"}),
        ),
        make_chunk(
            "skip#0",
            vec![0.99, 0.01, 0.0, 0.0],
            Some("forge"),
            json!({"domain": "other"}),
        ),
    ];
    backend.upsert(&chunks).await.unwrap();

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(1)
        .metric(Metric::Cosine)
        .namespace("forge")
        .filter(Filter::Equal {
            field: "domain".to_string(),
            value: json!("docs"),
        });
    backend.inner.reset_get_call_count();
    let results = backend.search(&query).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "match#0");
    assert_eq!(backend.inner.get_call_count(), 1);
}
