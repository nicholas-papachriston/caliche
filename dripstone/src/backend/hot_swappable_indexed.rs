//! Hot-swappable ANN backend wrapper.
//!
//! This wrapper keeps an active ANN-decorated backend behind an atomic handle so
//! searches can continue while a replacement index is rebuilt from the same
//! stored chunks. It rebuilds ANN state only; mmap file-format migration remains
//! a separate storage conversion step.

use std::fmt;
use std::sync::Arc;

use arc_swap::ArcSwapAny;
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::BoxStream;

use crate::backend::indexed::{IndexedBackend, IndexedBackendConfig};
use crate::core::{Capabilities, Chunk, FullTextQuery, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::index::VectorIndex;
use crate::search::FusionConfig;
use crate::traits::Backend;

/// ANN backend wrapper that supports atomic index cutovers.
pub struct HotSwappableIndexedBackend {
    inner: Arc<dyn Backend>,
    current: ArcSwapAny<Arc<Box<dyn Backend>>>,
    config: IndexedBackendConfig,
    mutation_lock: tokio::sync::Mutex<()>,
}

impl fmt::Debug for HotSwappableIndexedBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HotSwappableIndexedBackend")
            .field("inner", &self.inner)
            .field("current", &self.current.load_full())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl HotSwappableIndexedBackend {
    /// Create a new hot-swappable ANN backend from shared storage and an index.
    pub fn new(
        inner: Arc<dyn Backend>,
        index: Box<dyn VectorIndex>,
        config: IndexedBackendConfig,
    ) -> Self {
        let current: Arc<Box<dyn Backend>> = Arc::new(Box::new(IndexedBackend::new(
            Arc::clone(&inner),
            index,
            config.clone(),
        )));
        Self {
            inner,
            current: ArcSwapAny::from(current),
            config,
            mutation_lock: tokio::sync::Mutex::new(()),
        }
    }

    fn current_backend(&self) -> Arc<Box<dyn Backend>> {
        self.current.load_full()
    }

    /// Rebuild a replacement ANN index from stored chunks and atomically swap it
    /// into service. This rebuilds index state only; it does not migrate mmap
    /// storage formats.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn reindex_and_swap(
        &self,
        new_index: Box<dyn VectorIndex>,
    ) -> Result<(), DripstoneError> {
        let _guard = self.mutation_lock.lock().await;
        let replacement =
            IndexedBackend::new(Arc::clone(&self.inner), new_index, self.config.clone());
        replacement.rebuild_index().await?;
        let replacement: Arc<Box<dyn Backend>> = Arc::new(Box::new(replacement));
        self.current.store(replacement);
        Ok(())
    }
}

#[async_trait]
impl Backend for HotSwappableIndexedBackend {
    fn capabilities(&self) -> Capabilities {
        self.current_backend().capabilities()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let _guard = self.mutation_lock.lock().await;
        self.current_backend().upsert(chunks).await
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        self.current_backend().get(id, namespace).await
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let _guard = self.mutation_lock.lock().await;
        self.current_backend().delete(id, namespace).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let _guard = self.mutation_lock.lock().await;
        self.current_backend()
            .delete_document(doc_id, namespace)
            .await
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        self.current_backend().search(query).await
    }

    async fn search_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.current_backend().search_full_text(query).await
    }

    async fn search_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.current_backend()
            .search_hybrid_full_text(dense, lexical, fusion)
            .await
    }

    fn search_stream<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxStream<'a, Result<SearchResult, DripstoneError>> {
        let backend = self.current_backend();
        let query = query.clone();
        Box::pin(
            futures::stream::once(async move { backend.search(&query).await }).flat_map(|result| {
                match result {
                    Ok(results) => futures::stream::iter(results.into_iter().map(Ok)).boxed(),
                    Err(error) => futures::stream::once(async move { Err(error) }).boxed(),
                }
            }),
        )
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.current_backend().namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let _guard = self.mutation_lock.lock().await;
        self.current_backend().delete_namespace(namespace).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        self.current_backend().count(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        self.current_backend().list(namespace, limit).await
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        self.current_backend().flush().await
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        self.current_backend().close().await
    }
}
