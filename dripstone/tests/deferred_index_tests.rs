//! Integration tests for Superstore-style deferred ANN indexing on [`IndexedBackend`].

use std::sync::Arc;

use async_trait::async_trait;
use dripstone::backend::indexed::{IndexUpdateMode, IndexedBackend, IndexedBackendConfig};
use dripstone::core::{Capabilities, Chunk, Metric, SearchQuery};
use dripstone::errors::DripstoneError;
use dripstone::index::HnswIndex;
use dripstone::index::hnsw::HnswConfig;
use dripstone::traits::Backend;
use serde_json::json;

#[derive(Debug)]
struct StubBackend {
    chunks: tokio::sync::RwLock<Vec<Chunk>>,
}

impl StubBackend {
    fn new() -> Self {
        Self {
            chunks: tokio::sync::RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl Backend for StubBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities::memory()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let mut store = self.chunks.write().await;
        for chunk in chunks {
            store.retain(|existing| existing.id != chunk.id);
            store.push(chunk.clone());
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        _namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let store = self.chunks.read().await;
        Ok(store.iter().find(|chunk| chunk.id == id).cloned())
    }

    async fn delete(&self, id: &str, _namespace: Option<&str>) -> Result<(), DripstoneError> {
        let mut store = self.chunks.write().await;
        store.retain(|chunk| chunk.id != id);
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
        store.retain(|chunk| !chunk.id.starts_with(&prefix));
        Ok(before - store.len())
    }

    async fn search(
        &self,
        query: &SearchQuery,
    ) -> Result<Vec<dripstone::core::SearchResult>, DripstoneError> {
        let store = self.chunks.read().await;
        let mut results: Vec<dripstone::core::SearchResult> = store
            .iter()
            .map(|chunk| {
                let score = query
                    .metric
                    .compute(query.query(), &chunk.vector)
                    .unwrap_or(0.0);
                dripstone::core::SearchResult::new(chunk.clone(), score, 0)
            })
            .collect();
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(query.limit);
        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        Ok(vec![])
    }

    async fn delete_namespace(&self, _namespace: &str) -> Result<(), DripstoneError> {
        Ok(())
    }

    async fn count(&self, _namespace: Option<&str>) -> Result<usize, DripstoneError> {
        Ok(self.chunks.read().await.len())
    }

    async fn list(
        &self,
        _namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let store = self.chunks.read().await;
        let all: Vec<Chunk> = store.clone();
        Ok(match limit {
            Some(limit) => all.into_iter().take(limit).collect(),
            None => all,
        })
    }
}

fn make_chunk(id: &str, vector: Vec<f32>) -> Chunk {
    Chunk {
        id: id.to_string(),
        doc_id: "doc".to_string(),
        content: id.to_string(),
        vector: Arc::from(vector),
        start_offset: 0,
        end_offset: 0,
        chunk_index: 0,
        total_chunks: 1,
        metadata: json!({}),
        namespace: None,
        multi_vectors: None,
    }
}

fn deferred_config(auto_flush: usize) -> IndexedBackendConfig {
    IndexedBackendConfig {
        auto_rebuild_threshold: 0,
        fallback_to_brute_force: false,
        overfetch_factor: 2,
        index_update_mode: IndexUpdateMode::Deferred,
        scan_unindexed_tail: true,
        deferred_flush_threshold: auto_flush,
        deferred_search_flush_threshold: 0,
        max_tail_scan_records: 0,
    }
}

#[tokio::test]
async fn deferred_upsert_keeps_vectors_out_of_index_until_flush() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        deferred_config(0),
    );

    let chunk = make_chunk("fresh#0", vec![1.0, 0.0, 0.0, 0.0]);
    backend.upsert(&[chunk]).await.unwrap();

    assert_eq!(backend.index().len(), 0);
    assert_eq!(backend.pending_index_count(), 1);
}

#[tokio::test]
async fn deferred_search_with_tail_scan_returns_read_your_write() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        deferred_config(0),
    );

    backend
        .upsert(&[make_chunk("fresh#0", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(1)
        .metric(Metric::Cosine);
    let results = backend.search(&query).await.unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "fresh#0");
    assert_eq!(backend.index().len(), 0);
}

#[tokio::test]
async fn deferred_search_without_tail_scan_hides_pending_vectors() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        IndexedBackendConfig {
            scan_unindexed_tail: false,
            ..deferred_config(0)
        },
    );

    backend
        .upsert(&[make_chunk("fresh#0", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(1)
        .metric(Metric::Cosine);
    let results = backend.search(&query).await.unwrap();
    assert!(results.is_empty());

    backend.flush_pending_index().await.unwrap();
    let results = backend.search(&query).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "fresh#0");
}

#[tokio::test]
async fn deferred_flush_moves_pending_into_index() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        deferred_config(0),
    );

    backend
        .upsert(&[make_chunk("fresh#0", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();
    backend.flush_pending_index().await.unwrap();

    assert_eq!(backend.pending_index_count(), 0);
    assert_eq!(backend.index().len(), 1);
}

#[tokio::test]
async fn deferred_auto_flush_threshold_drains_pending_queue() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        deferred_config(2),
    );

    backend
        .upsert(&[make_chunk("a#0", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();
    assert_eq!(backend.pending_index_count(), 1);

    backend
        .upsert(&[make_chunk("b#0", vec![0.0, 1.0, 0.0, 0.0])])
        .await
        .unwrap();

    assert_eq!(backend.pending_index_count(), 0);
    assert_eq!(backend.index().len(), 2);
}

#[tokio::test]
async fn deferred_delete_removes_pending_record() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        deferred_config(0),
    );

    backend
        .upsert(&[make_chunk("fresh#0", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();
    backend.delete("fresh#0", None).await.unwrap();

    assert_eq!(backend.pending_index_count(), 0);

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(1)
        .metric(Metric::Cosine);
    let results = backend.search(&query).await.unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn synchronous_mode_still_updates_index_on_upsert() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        IndexedBackendConfig {
            auto_rebuild_threshold: 0,
            fallback_to_brute_force: false,
            overfetch_factor: 2,
            index_update_mode: IndexUpdateMode::Synchronous,
            scan_unindexed_tail: true,
            deferred_flush_threshold: 256,
            deferred_search_flush_threshold: 0,
            max_tail_scan_records: 0,
        },
    );

    backend
        .upsert(&[make_chunk("fresh#0", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();

    assert_eq!(backend.pending_index_count(), 0);
    assert_eq!(backend.index().len(), 1);
}

#[tokio::test]
async fn deferred_close_flushes_pending_index() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        deferred_config(0),
    );

    backend
        .upsert(&[make_chunk("fresh#0", vec![1.0, 0.0, 0.0, 0.0])])
        .await
        .unwrap();
    backend.close().await.unwrap();

    assert_eq!(backend.pending_index_count(), 0);
    assert_eq!(backend.index().len(), 1);
}

#[tokio::test]
async fn deferred_search_flush_threshold_drains_pending_on_search() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        IndexedBackendConfig {
            deferred_flush_threshold: 0,
            deferred_search_flush_threshold: 3,
            ..deferred_config(0)
        },
    );

    backend
        .upsert(&[
            make_chunk("a#0", vec![1.0, 0.0, 0.0, 0.0]),
            make_chunk("b#0", vec![0.0, 1.0, 0.0, 0.0]),
            make_chunk("c#0", vec![0.0, 0.0, 1.0, 0.0]),
        ])
        .await
        .unwrap();
    assert_eq!(backend.pending_index_count(), 3);
    assert_eq!(backend.index().len(), 0);

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(1)
        .metric(Metric::Cosine);
    backend.search(&query).await.unwrap();

    assert_eq!(backend.pending_index_count(), 0);
    assert_eq!(backend.index().len(), 3);
}

#[tokio::test]
async fn max_tail_scan_records_limits_pending_visibility() {
    let backend = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        IndexedBackendConfig {
            max_tail_scan_records: 2,
            ..deferred_config(0)
        },
    );

    let chunks: Vec<Chunk> = (0..20)
        .map(|i| {
            let weight = i as f32 * 0.04;
            make_chunk(&format!("vec-{i}"), vec![1.0 - weight, weight, 0.0, 0.0])
        })
        .collect();
    backend.upsert(&chunks).await.unwrap();

    let query = SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
        .limit(1)
        .metric(Metric::Cosine);
    let capped = backend.search(&query).await.unwrap();
    assert_eq!(capped.len(), 1);
    assert_eq!(capped[0].chunk.id, "vec-18");

    let uncapped = IndexedBackend::new(
        StubBackend::new(),
        Box::new(HnswIndex::new(HnswConfig::default(), 4)),
        deferred_config(0),
    );
    uncapped.upsert(&chunks).await.unwrap();
    let full = uncapped.search(&query).await.unwrap();
    assert_eq!(full.len(), 1);
    assert_eq!(full[0].chunk.id, "vec-0");
}
