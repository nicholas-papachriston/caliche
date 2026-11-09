use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use crate::core::{Capabilities, Chunk, FullTextQuery, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::fts::{FullTextIndex, InMemoryFullTextIndex};
use crate::search::fusion::{FusionConfig, fuse_search_results};
use crate::traits::Backend;

#[derive(Debug, Clone, Default)]
pub struct FullTextIndexedBackendConfig {
    /// Rebuild the full lexical index after this many writes. `0` defers rebuilds until search/flush/close.
    pub auto_rebuild_threshold: usize,
    /// Optional path for persisting the in-memory lexical snapshot.
    pub snapshot_path: Option<String>,
    /// Rebuild the lexical index on search if writes are pending.
    pub rebuild_on_search: bool,
}

pub struct FullTextIndexedBackend<B: Backend> {
    inner: B,
    index: Arc<dyn FullTextIndex>,
    config: FullTextIndexedBackendConfig,
    dirty_count: Arc<std::sync::atomic::AtomicUsize>,
}

impl<B: Backend> fmt::Debug for FullTextIndexedBackend<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FullTextIndexedBackend")
            .field("inner", &self.inner)
            .field("config", &self.config)
            .field(
                "dirty_count",
                &self.dirty_count.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

impl<B: Backend> FullTextIndexedBackend<B> {
    pub fn new(
        inner: B,
        index: Arc<dyn FullTextIndex>,
        config: FullTextIndexedBackendConfig,
    ) -> Self {
        Self {
            inner,
            index,
            config,
            dirty_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }
    }

    pub fn with_in_memory_index(inner: B) -> Self {
        Self::new(
            inner,
            Arc::new(InMemoryFullTextIndex::new()),
            FullTextIndexedBackendConfig::default(),
        )
    }

    pub fn index(&self) -> Arc<dyn FullTextIndex> {
        Arc::clone(&self.index)
    }

    /// Backend wrapped by the lexical index.
    #[cfg(test)]
    pub(crate) fn inner(&self) -> &B {
        &self.inner
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn rebuild_index(&self) -> Result<(), DripstoneError> {
        let chunks = self.inner.list(None, None).await?;
        self.index.build(&chunks).await?;
        self.dirty_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn load_snapshot_if_configured(&self) -> Result<bool, DripstoneError> {
        let Some(path) = self.config.snapshot_path.as_deref() else {
            return Ok(false);
        };
        self.index.load_snapshot(path).await
    }

    async fn maybe_rebuild(&self) -> Result<(), DripstoneError> {
        let threshold = self.config.auto_rebuild_threshold;
        if threshold == 0 {
            return Ok(());
        }
        let dirty = self.dirty_count.load(std::sync::atomic::Ordering::Relaxed);
        if dirty >= threshold {
            self.rebuild_index().await?;
        }
        Ok(())
    }

    fn materialize_chunk(mut chunk: Chunk, include_content: bool, include_vectors: bool) -> Chunk {
        if !include_content {
            chunk.content.clear();
        }
        if !include_vectors {
            chunk.vector = Arc::from([]);
        }
        chunk
    }
}

#[async_trait]
impl<B: Backend> Backend for FullTextIndexedBackend<B> {
    fn capabilities(&self) -> Capabilities {
        let mut caps = self.inner.capabilities();
        caps.native_full_text = true;
        caps.native_hybrid_full_text = true;
        caps
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        self.inner.upsert(chunks).await?;
        for chunk in chunks {
            self.index.add_chunk(chunk.clone()).await?;
        }
        self.dirty_count
            .fetch_add(chunks.len(), std::sync::atomic::Ordering::Relaxed);
        self.maybe_rebuild().await?;
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        self.inner.get(id, namespace).await
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        self.inner.delete(id, namespace).await?;
        self.index.remove_chunk(id, namespace).await?;
        self.dirty_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.maybe_rebuild().await?;
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let chunks = self.inner.list(namespace, None).await?;
        let prefix = format!("{doc_id}#");
        let matching: Vec<(String, Option<String>)> = chunks
            .iter()
            .filter(|chunk| chunk.id.starts_with(&prefix) || chunk.id == doc_id)
            .map(|chunk| (chunk.id.clone(), chunk.namespace.clone()))
            .collect();
        let deleted = self.inner.delete_document(doc_id, namespace).await?;
        for (chunk_id, chunk_namespace) in matching {
            self.index
                .remove_chunk(&chunk_id, chunk_namespace.as_deref())
                .await?;
        }
        self.dirty_count
            .fetch_add(deleted, std::sync::atomic::Ordering::Relaxed);
        self.maybe_rebuild().await?;
        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        self.inner.search(query).await
    }

    async fn search_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        if self.index.is_empty() && self.inner.count(None).await? > 0 {
            let loaded_snapshot = self.load_snapshot_if_configured().await?;
            if !loaded_snapshot && self.inner.count(None).await? > 0 {
                self.rebuild_index().await?;
            }
        }
        if self.index.is_dirty() && self.config.rebuild_on_search {
            self.rebuild_index().await?;
        }
        let hits = self.index.search_maxscore(query).await?;
        let mut results = Vec::with_capacity(hits.len());
        for (rank, hit) in hits.into_iter().enumerate() {
            if let Some(chunk) = self
                .inner
                .get(&hit.chunk_id, hit.namespace.as_deref())
                .await?
            {
                results.push(SearchResult::new(
                    Self::materialize_chunk(chunk, query.include_content, query.include_vectors),
                    hit.score,
                    rank,
                ));
            }
        }
        Ok(results)
    }

    async fn search_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let dense_results = self.inner.search(dense).await?;
        let lexical_results = self.search_full_text(lexical).await?;
        let mut fused = fuse_search_results(dense_results, lexical_results, fusion)?;
        fused.truncate(dense.limit.min(lexical.limit));
        for (rank, result) in fused.iter_mut().enumerate() {
            result.rank = rank;
        }
        Ok(fused)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.inner.namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        self.inner.delete_namespace(namespace).await?;
        self.rebuild_index().await
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

    async fn flush(&self) -> Result<(), DripstoneError> {
        self.inner.flush().await?;
        if self.index.is_dirty() {
            self.rebuild_index().await?;
        }
        if let Some(path) = self.config.snapshot_path.as_deref() {
            self.index.flush_snapshot(path).await?;
        }
        Ok(())
    }

    async fn flush_pending_ann_index(&self) -> Result<(), DripstoneError> {
        self.inner.flush_pending_ann_index().await
    }

    fn pending_ann_index_count(&self) -> usize {
        self.inner.pending_ann_index_count()
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        if self.index.is_dirty() {
            self.rebuild_index().await?;
        }
        if let Some(path) = self.config.snapshot_path.as_deref() {
            self.index.flush_snapshot(path).await?;
        }
        self.inner.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Metric;
    use async_trait::async_trait;
    use parking_lot::RwLock;
    use tempfile::tempdir;

    #[derive(Debug, Default)]
    struct StubBackend {
        chunks: RwLock<Vec<Chunk>>,
    }

    impl StubBackend {
        fn chunk(
            doc_id: &str,
            chunk_index: usize,
            content: &str,
            vector: Vec<f32>,
            namespace: Option<&str>,
            metadata: serde_json::Value,
        ) -> Chunk {
            Chunk::new(
                doc_id,
                chunk_index,
                content,
                vector,
                0,
                content.len(),
                1,
                metadata,
                namespace.map(ToOwned::to_owned),
            )
        }
    }

    #[async_trait]
    impl Backend for StubBackend {
        fn capabilities(&self) -> Capabilities {
            Capabilities::memory()
        }

        async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
            let mut guard = self.chunks.write();
            for chunk in chunks {
                guard.retain(|existing| existing.id != chunk.id);
                guard.push(chunk.clone());
            }
            drop(guard);
            Ok(())
        }

        async fn get(
            &self,
            id: &str,
            namespace: Option<&str>,
        ) -> Result<Option<Chunk>, DripstoneError> {
            Ok(self
                .chunks
                .read()
                .iter()
                .find(|chunk| {
                    chunk.id == id
                        && namespace
                            .is_none_or(|namespace| chunk.namespace.as_deref() == Some(namespace))
                })
                .cloned())
        }

        async fn delete(&self, id: &str, _namespace: Option<&str>) -> Result<(), DripstoneError> {
            self.chunks.write().retain(|chunk| chunk.id != id);
            Ok(())
        }

        async fn delete_document(
            &self,
            doc_id: &str,
            _namespace: Option<&str>,
        ) -> Result<usize, DripstoneError> {
            let prefix = format!("{doc_id}#");
            let mut guard = self.chunks.write();
            let before = guard.len();
            guard.retain(|chunk| !(chunk.id.starts_with(&prefix) || chunk.id == doc_id));
            Ok(before.saturating_sub(guard.len()))
        }

        async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
            let mut results = self
                .chunks
                .read()
                .iter()
                .filter(|chunk| {
                    query
                        .namespace
                        .as_deref()
                        .is_none_or(|namespace| chunk.namespace.as_deref() == Some(namespace))
                })
                .filter(|chunk| {
                    query
                        .filter
                        .as_ref()
                        .is_none_or(|filter| filter.matches(&chunk.metadata))
                })
                .map(|chunk| {
                    let score = query
                        .metric
                        .compute(query.query(), &chunk.vector)
                        .unwrap_or(0.0);
                    SearchResult::new(chunk.clone(), score, 0)
                })
                .collect::<Vec<_>>();
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            results.truncate(query.limit);
            for (rank, result) in results.iter_mut().enumerate() {
                result.rank = rank;
            }
            Ok(results)
        }

        async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
            Ok(vec!["docs".to_string()])
        }

        async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
            self.chunks
                .write()
                .retain(|chunk| chunk.namespace.as_deref() != Some(namespace));
            Ok(())
        }

        async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
            Ok(self
                .chunks
                .read()
                .iter()
                .filter(|chunk| {
                    namespace.is_none_or(|namespace| chunk.namespace.as_deref() == Some(namespace))
                })
                .count())
        }

        async fn list(
            &self,
            namespace: Option<&str>,
            limit: Option<usize>,
        ) -> Result<Vec<Chunk>, DripstoneError> {
            let mut chunks = self
                .chunks
                .read()
                .iter()
                .filter(|chunk| {
                    namespace.is_none_or(|namespace| chunk.namespace.as_deref() == Some(namespace))
                })
                .cloned()
                .collect::<Vec<_>>();
            if let Some(limit) = limit {
                chunks.truncate(limit);
            }
            Ok(chunks)
        }
    }

    #[tokio::test]
    async fn full_text_indexed_backend_searches_and_hydrates() {
        let backend = FullTextIndexedBackend::with_in_memory_index(StubBackend::default());
        let chunks = vec![
            StubBackend::chunk(
                "doc-a",
                0,
                "rust vector search and bm25 scoring",
                vec![1.0, 0.0],
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
            StubBackend::chunk(
                "doc-b",
                0,
                "rust rust rust full text retrieval",
                vec![0.5, 0.5],
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
        ];
        backend
            .upsert(&chunks)
            .await
            .expect("upsert should succeed");

        let results = backend
            .search_full_text(&FullTextQuery::new("rust retrieval").limit(2))
            .await
            .expect("search_full_text should succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, "doc-b#0");
        assert_eq!(
            results[0].chunk.content,
            "rust rust rust full text retrieval"
        );
    }

    #[tokio::test]
    async fn full_text_indexed_backend_fuses_dense_and_lexical_results() {
        let backend = FullTextIndexedBackend::with_in_memory_index(StubBackend::default());
        let chunks = vec![
            StubBackend::chunk(
                "doc-a",
                0,
                "rust vector search and bm25 scoring",
                vec![1.0, 0.0],
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
            StubBackend::chunk(
                "doc-b",
                0,
                "rust rust rust full text retrieval",
                vec![0.2, 0.8],
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
        ];
        backend
            .upsert(&chunks)
            .await
            .expect("upsert should succeed");

        let dense = SearchQuery::new(vec![1.0, 0.0])
            .limit(2)
            .metric(Metric::Cosine)
            .namespace("docs");
        let lexical = FullTextQuery::new("retrieval rust")
            .limit(2)
            .namespace("docs");

        let results = backend
            .search_hybrid_full_text(
                &dense,
                &lexical,
                &FusionConfig::weighted_sum().with_alpha(0.4),
            )
            .await
            .expect("hybrid search should succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, "doc-b#0");
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn full_text_indexed_backend_supports_rrf_fusion() {
        let backend = FullTextIndexedBackend::with_in_memory_index(StubBackend::default());
        let chunks = vec![
            StubBackend::chunk(
                "doc-a",
                0,
                "rust vector search and bm25 scoring",
                vec![1.0, 0.0],
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
            StubBackend::chunk(
                "doc-b",
                0,
                "rust rust rust full text retrieval",
                vec![0.2, 0.8],
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
        ];
        backend
            .upsert(&chunks)
            .await
            .expect("upsert should succeed");

        let dense = SearchQuery::new(vec![1.0, 0.0])
            .limit(2)
            .metric(Metric::Cosine)
            .namespace("docs");
        let lexical = FullTextQuery::new("retrieval rust")
            .limit(2)
            .namespace("docs");

        let results = backend
            .search_hybrid_full_text(&dense, &lexical, &FusionConfig::reciprocal_rank())
            .await
            .expect("hybrid search should succeed");

        assert_eq!(results.len(), 2);
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn full_text_indexed_backend_defers_rebuild_until_search() {
        let backend = FullTextIndexedBackend::with_in_memory_index(StubBackend::default());
        let chunks = vec![StubBackend::chunk(
            "doc-a",
            0,
            "rust vector search and bm25 scoring",
            vec![1.0, 0.0],
            Some("docs"),
            serde_json::json!({"kind": "guide"}),
        )];
        backend
            .upsert(&chunks)
            .await
            .expect("upsert should succeed");
        assert!(
            backend.index().is_dirty(),
            "index should remain dirty after upsert"
        );

        let results = backend
            .search_full_text(&FullTextQuery::new("rust").limit(1))
            .await
            .expect("search_full_text should succeed");

        assert_eq!(results.len(), 1);
        assert!(
            !backend.index().is_dirty(),
            "search should rebuild dirty index"
        );
    }

    #[tokio::test]
    async fn full_text_indexed_backend_flushes_and_reloads_snapshot() {
        let dir = tempdir().expect("tempdir");
        let snapshot_path = dir.path().join("fts_snapshot.json");
        let config = FullTextIndexedBackendConfig {
            auto_rebuild_threshold: 0,
            snapshot_path: Some(snapshot_path.to_string_lossy().to_string()),
            rebuild_on_search: true,
        };
        let backend = FullTextIndexedBackend::new(
            StubBackend::default(),
            Arc::new(InMemoryFullTextIndex::new()),
            config.clone(),
        );
        let chunks = vec![
            StubBackend::chunk(
                "doc-a",
                0,
                "rust vector search and bm25 scoring",
                vec![1.0, 0.0],
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
            StubBackend::chunk(
                "doc-b",
                0,
                "rust rust rust full text retrieval",
                vec![0.5, 0.5],
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
        ];
        backend
            .upsert(&chunks)
            .await
            .expect("upsert should succeed");
        backend.flush().await.expect("flush should succeed");
        assert!(
            snapshot_path.exists(),
            "snapshot file should exist after flush"
        );

        let reloaded_inner = StubBackend::default();
        reloaded_inner
            .upsert(&chunks)
            .await
            .expect("reload backend upsert should succeed");
        let reloaded = FullTextIndexedBackend::new(
            reloaded_inner,
            Arc::new(InMemoryFullTextIndex::new()),
            config,
        );

        let results = reloaded
            .search_full_text(&FullTextQuery::new("rust retrieval").limit(2))
            .await
            .expect("search_full_text should succeed");

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.id, "doc-b#0");
    }
}
