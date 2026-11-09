//! ANN-accelerated backend decorator types and implementation.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;

use super::config::{DeferredIndexState, IndexUpdateMode, IndexedBackendConfig};
use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::embedding::{EmbeddingRecord, SimilarityMetric};
use crate::errors::DripstoneError;
use crate::index::{IndexSearchRequest, VectorIndex};
use crate::search::{SearchPostFilter, compile_native_filter_plan};
use crate::traits::Backend;

/// A [`Backend`] decorator that accelerates `search()` via a [`VectorIndex`].
///
/// Storage (upsert / get / delete) is delegated to the inner backend.  On
/// `upsert`, each chunk is also added to the in-memory index so that
/// subsequent searches use ANN rather than brute-force scan.
pub struct IndexedBackend<B: Backend> {
    pub(super) inner: B,
    index: Arc<dyn VectorIndex>,
    config: IndexedBackendConfig,
    dirty_count: Arc<std::sync::atomic::AtomicUsize>,
    deferred: Option<Arc<DeferredIndexState>>,
}

/// Error returned when rebuilding a replacement index fails.
///
/// The original backend is returned so callers can recover without moving
/// vector storage or rebuilding the previous index.
pub struct ReindexError<B: Backend> {
    pub backend: IndexedBackend<B>,
    pub source: DripstoneError,
}

impl<B: Backend> fmt::Debug for ReindexError<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReindexError")
            .field("backend", &self.backend)
            .field("source", &self.source)
            .finish()
    }
}

impl<B: Backend> fmt::Display for ReindexError<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to build replacement vector index")
    }
}

impl<B: Backend> std::error::Error for ReindexError<B> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

impl<B: Backend> fmt::Debug for IndexedBackend<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndexedBackend")
            .field("inner", &self.inner)
            .field("config", &self.config)
            .field(
                "dirty_count",
                &self.dirty_count.load(std::sync::atomic::Ordering::Relaxed),
            )
            .finish_non_exhaustive()
    }
}

fn merge_search_results(
    mut index_results: Vec<SearchResult>,
    tail_results: Vec<SearchResult>,
    limit: usize,
    higher_is_better: bool,
) -> Vec<SearchResult> {
    use std::collections::HashMap;

    let mut best: HashMap<String, SearchResult> =
        HashMap::with_capacity(index_results.len().saturating_add(tail_results.len()));
    for result in index_results.drain(..).chain(tail_results) {
        match best.entry(result.chunk.id.clone()) {
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(result);
            }
            std::collections::hash_map::Entry::Occupied(mut slot) => {
                let replace = if higher_is_better {
                    result.score > slot.get().score
                } else {
                    result.score < slot.get().score
                };
                if replace {
                    let _ = slot.insert(result);
                }
            }
        }
    }

    let mut merged: Vec<SearchResult> = best.into_values().collect();
    if higher_is_better {
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        merged.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    merged.truncate(limit.max(1));
    for (rank, result) in merged.iter_mut().enumerate() {
        result.rank = rank;
    }
    merged
}

impl<B: Backend> IndexedBackend<B> {
    fn from_parts(
        inner: B,
        index: Arc<dyn VectorIndex>,
        config: IndexedBackendConfig,
        dirty_count: Arc<std::sync::atomic::AtomicUsize>,
        deferred: Option<Arc<DeferredIndexState>>,
    ) -> Self {
        Self {
            inner,
            index,
            config,
            dirty_count,
            deferred,
        }
    }

    fn deferred_state(&self) -> Option<&DeferredIndexState> {
        self.deferred.as_deref()
    }

    fn is_deferred(&self) -> bool {
        self.config.index_update_mode == IndexUpdateMode::Deferred
    }

    async fn load_all_records(inner: &B) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        let chunks = inner.list(None, None).await?;
        Ok(chunks.iter().map(EmbeddingRecord::from).collect())
    }

    /// Create a new indexed backend wrapping `inner` with the given `index`.
    pub fn new(inner: B, index: Box<dyn VectorIndex>, config: IndexedBackendConfig) -> Self {
        let deferred = (config.index_update_mode == IndexUpdateMode::Deferred).then(|| {
            Arc::new(DeferredIndexState {
                pending: tokio::sync::RwLock::new(Vec::new()),
            })
        });
        Self::from_parts(
            inner,
            Arc::from(index),
            config,
            Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            deferred,
        )
    }

    /// Count of embedding records committed to the inner backend but not yet inserted
    /// into the ANN index when [`IndexUpdateMode::Deferred`] is enabled.
    pub fn pending_index_count(&self) -> usize {
        let Some(state) = self.deferred_state() else {
            return 0;
        };
        state.pending.try_read().map_or(0, |pending| pending.len())
    }

    /// Insert all pending deferred records into the ANN index.
    ///
    /// No-op when [`IndexUpdateMode::Synchronous`] is configured.
    ///
    /// # Errors
    ///
    /// Returns an error when index insertion or a periodic rebuild fails.
    pub async fn flush_pending_index(&self) -> Result<(), DripstoneError> {
        if !self.is_deferred() {
            return Ok(());
        }
        let records = {
            let Some(state) = self.deferred_state() else {
                return Ok(());
            };
            let mut pending = state.pending.write().await;
            std::mem::take(&mut *pending)
        };
        if records.is_empty() {
            return Ok(());
        }
        self.apply_index_records(&records).await
    }

    async fn apply_index_records(&self, records: &[EmbeddingRecord]) -> Result<(), DripstoneError> {
        for record in records {
            self.index.add(record.clone()).await?;
        }
        self.dirty_count
            .fetch_add(records.len(), std::sync::atomic::Ordering::Relaxed);
        self.maybe_rebuild().await
    }

    async fn queue_deferred_records(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let Some(state) = self.deferred_state() else {
            return Ok(());
        };
        let should_flush = {
            let mut pending = state.pending.write().await;
            for chunk in chunks {
                let record = EmbeddingRecord::from(chunk);
                pending.retain(|existing| existing.id != record.id);
                pending.push(record);
            }
            let threshold = self.config.deferred_flush_threshold;
            threshold > 0 && pending.len() >= threshold
        };
        if should_flush {
            self.flush_pending_index().await?;
        }
        Ok(())
    }

    async fn remove_deferred_ids<'a, I: IntoIterator<Item = &'a str>>(
        &self,
        ids: I,
    ) -> Result<(), DripstoneError> {
        let Some(state) = self.deferred_state() else {
            return Ok(());
        };
        let mut pending = state.pending.write().await;
        for id in ids {
            pending.retain(|record| record.id.as_ref() != id);
        }
        Ok(())
    }

    async fn clear_deferred_pending(&self) {
        if let Some(state) = self.deferred_state() {
            let mut pending = state.pending.write().await;
            pending.clear();
        }
    }

    async fn maybe_flush_on_search(&self) -> Result<(), DripstoneError> {
        if !self.is_deferred() {
            return Ok(());
        }
        let threshold = self.config.deferred_search_flush_threshold;
        if threshold == 0 {
            return Ok(());
        }
        if self.pending_index_count() >= threshold {
            self.flush_pending_index().await?;
        }
        Ok(())
    }

    async fn scan_unindexed_tail(
        &self,
        query: &SearchQuery,
        post_filter: &SearchPostFilter<'_>,
        fetch_limit: usize,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let Some(state) = self.deferred_state() else {
            return Ok(Vec::new());
        };
        let pending = state.pending.read().await;
        if pending.is_empty() {
            return Ok(Vec::new());
        }

        let tail_records: &[EmbeddingRecord] = {
            let max = self.config.max_tail_scan_records;
            if max > 0 {
                let start = pending.len().saturating_sub(max);
                &pending[start..]
            } else {
                &pending[..]
            }
        };

        let higher_better = query.metric.higher_is_better();
        let null_metadata = serde_json::Value::Null;
        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(tail_records.len());
        for (idx, record) in tail_records.iter().enumerate() {
            let ns = record.namespace.as_deref();
            let metadata = record.metadata.as_ref().unwrap_or(&null_metadata);
            if !post_filter.passes(ns, metadata) {
                continue;
            }
            let score = query
                .metric
                .compute(query.query(), &record.vector)
                .unwrap_or(0.0);
            scored.push((score, idx));
        }

        if higher_better {
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        scored.truncate(fetch_limit.max(1));

        let mut results = Vec::with_capacity(scored.len());
        for (score, idx) in scored {
            let record = &tail_records[idx];
            let ns = record.namespace.as_deref();
            if let Some(chunk) = self.inner.get(record.id.as_ref(), ns).await? {
                results.push(SearchResult::new(chunk, score, 0));
            }
        }
        Ok(results)
    }

    /// Consume the decorator and return the wrapped storage backend.
    pub fn into_inner(self) -> B {
        let (inner, _, _, _, _) = self.into_raw_parts();
        inner
    }

    /// Consume the decorator and return the wrapped backend, index, and config.
    pub fn into_parts(self) -> (B, Arc<dyn VectorIndex>, IndexedBackendConfig) {
        let (inner, index, config, _, _) = self.into_raw_parts();
        (inner, index, config)
    }

    fn into_raw_parts(
        self,
    ) -> (
        B,
        Arc<dyn VectorIndex>,
        IndexedBackendConfig,
        Arc<std::sync::atomic::AtomicUsize>,
        Option<Arc<DeferredIndexState>>,
    ) {
        let Self {
            inner,
            index,
            config,
            dirty_count,
            deferred,
        } = self;
        (inner, index, config, dirty_count, deferred)
    }

    /// Shared reference to the wrapped index (for pipeline stages or external use).
    pub fn index(&self) -> Arc<dyn VectorIndex> {
        Arc::clone(&self.index)
    }

    /// Rebuild the ANN index with a different index type while reusing the same
    /// stored chunks. This does not migrate mmap file formats or move vector
    /// data between backends.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn reindex(self, new_index: Box<dyn VectorIndex>) -> Result<Self, ReindexError<B>> {
        let (inner, old_index, config, dirty_count, deferred) = self.into_raw_parts();
        let records = match Self::load_all_records(&inner).await {
            Ok(records) => records,
            Err(source) => {
                return Err(ReindexError {
                    backend: Self::from_parts(inner, old_index, config, dirty_count, deferred),
                    source,
                });
            }
        };

        let new_index: Arc<dyn VectorIndex> = Arc::from(new_index);
        if let Err(source) = new_index.build(&records).await {
            return Err(ReindexError {
                backend: Self::from_parts(inner, old_index, config, dirty_count, deferred),
                source,
            });
        }

        Ok(Self::from_parts(
            inner,
            new_index,
            config,
            Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            deferred,
        ))
    }

    /// Rebuild the index from all chunks currently stored in the inner backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn rebuild_index(&self) -> Result<(), DripstoneError> {
        let records = Self::load_all_records(&self.inner).await?;
        // `VectorIndex::build` resets the index; do not call sync `clear()` here — it uses
        // `tokio::sync::RwLock::blocking_write` and panics on a Tokio worker thread.
        self.index.build(&records).await?;
        self.dirty_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        self.clear_deferred_pending().await;
        Ok(())
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
}

#[async_trait]
impl<B: Backend> Backend for IndexedBackend<B> {
    fn capabilities(&self) -> Capabilities {
        let mut caps = self.inner.capabilities();
        caps.requires_exact_filter_pass = true;
        caps
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        self.inner.upsert(chunks).await?;

        if self.is_deferred() {
            return self.queue_deferred_records(chunks).await;
        }

        let records: Vec<EmbeddingRecord> = chunks.iter().map(EmbeddingRecord::from).collect();
        self.apply_index_records(&records).await
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
        self.remove_deferred_ids([id]).await?;
        self.index.remove(id).await?;
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let chunks = self.inner.list(namespace, None).await?;
        let prefix = format!("{doc_id}#");
        let matching_ids: Vec<&str> = chunks
            .iter()
            .filter(|c| c.id.starts_with(&prefix))
            .map(|c| c.id.as_str())
            .collect();

        let count = self.inner.delete_document(doc_id, namespace).await?;

        self.remove_deferred_ids(matching_ids.iter().copied())
            .await?;
        for id in &matching_ids {
            self.index.remove(id).await?;
        }
        Ok(count)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        if self.index.is_empty() && self.config.fallback_to_brute_force {
            return crate::traits::search_with_maxsim_fallback(&self.inner, query).await;
        }

        self.maybe_flush_on_search().await?;

        let metric: SimilarityMetric = query.metric.into();
        let base_limit = if query.query_multi_vectors.is_some() {
            crate::search::overfetch_limit(query)
        } else {
            query.limit
        };
        let overfetch = base_limit
            .saturating_mul(self.config.overfetch_factor)
            .max(base_limit);
        let native_filter_plan =
            compile_native_filter_plan(query.namespace.as_deref(), query.filter.as_ref());
        let candidates = self
            .index
            .search_request(IndexSearchRequest {
                query: query.query(),
                limit: overfetch,
                metric,
                native_filter: native_filter_plan
                    .is_supported()
                    .then_some(&native_filter_plan),
            })
            .await?;

        let post_filter = SearchPostFilter {
            query_namespace: query.namespace.as_deref(),
            filter: query.filter.as_ref(),
            native_plan: &native_filter_plan,
        };
        let mut results: Vec<SearchResult> = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            let id: &str = &candidate.record.id;
            let ns = candidate.record.namespace.as_deref();
            if let Some(chunk) = self.inner.get(id, ns).await? {
                if !post_filter.passes(ns, &chunk.metadata) {
                    continue;
                }
                results.push(SearchResult::new(chunk, candidate.score, 0));
            }
        }

        if query.query_multi_vectors.is_some() {
            return crate::search::rerank_with_maxsim(results, query);
        }

        let higher_better = query.metric.higher_is_better();
        if self.is_deferred() && self.config.scan_unindexed_tail {
            let tail_fetch_limit = overfetch.max(query.limit);
            let tail_results = self
                .scan_unindexed_tail(query, &post_filter, tail_fetch_limit)
                .await?;
            results = merge_search_results(results, tail_results, query.limit, higher_better);
            return Ok(results);
        }

        results.sort_by(|a, b| {
            if higher_better {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });
        results.truncate(query.limit);
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i;
        }
        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.inner.namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let chunks = self.inner.list(Some(namespace), None).await?;
        self.inner.delete_namespace(namespace).await?;

        self.remove_deferred_ids(chunks.iter().map(|chunk| chunk.id.as_str()))
            .await?;
        for chunk in &chunks {
            self.index.remove(&chunk.id).await?;
        }
        Ok(())
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
        self.flush_pending_index().await?;
        self.inner.flush().await
    }

    async fn flush_pending_ann_index(&self) -> Result<(), DripstoneError> {
        self.flush_pending_index().await
    }

    fn pending_ann_index_count(&self) -> usize {
        self.pending_index_count()
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        self.flush_pending_index().await?;
        self.inner.close().await
    }
}
