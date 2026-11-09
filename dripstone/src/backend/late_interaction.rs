//! Backend decorator that owns a [`PlaidIndex`] for native MaxSim retrieval.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;

use crate::core::{Capabilities, Chunk, FullTextQuery, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::index::plaid::{PlaidConfig, PlaidIndex};
use crate::search::FusionConfig;
use crate::traits::Backend;

/// Configuration for [`LateInteractionIndexedBackend`].
#[derive(Debug, Clone, Default)]
pub struct LateInteractionIndexedBackendConfig {
    pub plaid: PlaidConfig,
    /// Rebuild PLAID from the inner store after this many writes (`0` = only on search/flush).
    pub auto_rebuild_threshold: usize,
    /// Rebuild on search when the index is dirty.
    pub rebuild_on_search: bool,
}

/// Wraps a storage backend with a PLAID late-interaction index.
pub struct LateInteractionIndexedBackend<B: Backend> {
    inner: B,
    index: Arc<PlaidIndex>,
    config: LateInteractionIndexedBackendConfig,
    dirty_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Tracks whether a full rebuild is needed (incremental upserts after cold start).
    needs_rebuild: Arc<RwLock<bool>>,
}

impl<B: Backend> fmt::Debug for LateInteractionIndexedBackend<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LateInteractionIndexedBackend")
            .field("inner", &self.inner)
            .field("config", &self.config)
            .field("plaid_len", &self.index.len())
            .finish_non_exhaustive()
    }
}

impl<B: Backend> LateInteractionIndexedBackend<B> {
    pub fn new(inner: B, config: LateInteractionIndexedBackendConfig) -> Self {
        let index = Arc::new(PlaidIndex::new(config.plaid.clone()));
        Self {
            inner,
            index,
            config,
            dirty_count: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            needs_rebuild: Arc::new(RwLock::new(true)),
        }
    }

    pub fn with_default_index(inner: B) -> Self {
        Self::new(inner, LateInteractionIndexedBackendConfig::default())
    }

    #[must_use]
    pub fn index(&self) -> Arc<PlaidIndex> {
        Arc::clone(&self.index)
    }

    ///
    /// # Errors
    ///
    /// Returns when listing or rebuilding fails.
    pub async fn rebuild_index(&self) -> Result<(), DripstoneError> {
        let chunks = self.inner.list(None, None).await?;
        self.index.rebuild(&chunks)?;
        self.dirty_count
            .store(0, std::sync::atomic::Ordering::Relaxed);
        *self.needs_rebuild.write() = false;
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

    async fn ensure_index_for_search(&self) -> Result<(), DripstoneError> {
        if *self.needs_rebuild.read() || self.index.is_empty() {
            let count = self.inner.count(None).await?;
            if count > 0 {
                self.rebuild_index().await?;
            }
        } else if self.config.rebuild_on_search
            && self.dirty_count.load(std::sync::atomic::Ordering::Relaxed) > 0
        {
            self.rebuild_index().await?;
        }
        Ok(())
    }
}

#[async_trait]
impl<B: Backend> Backend for LateInteractionIndexedBackend<B> {
    fn capabilities(&self) -> Capabilities {
        let mut caps = self.inner.capabilities();
        caps.native_maxsim = true;
        caps
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        self.inner.upsert(chunks).await?;
        for chunk in chunks {
            self.index.upsert_chunk(chunk)?;
        }
        self.dirty_count
            .fetch_add(chunks.len(), std::sync::atomic::Ordering::Relaxed);
        *self.needs_rebuild.write() = true;
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
        self.index.remove_chunk(id);
        self.dirty_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let chunks = self.inner.list(namespace, None).await?;
        let prefix = format!("{doc_id}#");
        for chunk in &chunks {
            if chunk.id.starts_with(&prefix) || chunk.id == doc_id {
                self.index.remove_chunk(&chunk.id);
            }
        }
        let deleted = self.inner.delete_document(doc_id, namespace).await?;
        self.dirty_count
            .fetch_add(deleted, std::sync::atomic::Ordering::Relaxed);
        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        if query.query_multi_vectors.is_some() {
            self.ensure_index_for_search().await?;
            return self.index.search(query);
        }
        self.inner.search(query).await
    }

    async fn search_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.inner.search_full_text(query).await
    }

    async fn search_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        if dense.query_multi_vectors.is_some() {
            self.ensure_index_for_search().await?;
        }
        self.inner
            .search_hybrid_full_text(dense, lexical, fusion)
            .await
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
        if self.dirty_count.load(std::sync::atomic::Ordering::Relaxed) > 0
            || *self.needs_rebuild.read()
        {
            self.rebuild_index().await?;
        }
        Ok(())
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        self.flush().await?;
        self.inner.close().await
    }
}
