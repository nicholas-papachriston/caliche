//! Core traits for `Dripstone` backends and embedders.
//!
//! This module defines the fundamental traits that backends and embedders must implement:
//! - `Backend` - Vector storage and search operations
//! - `Embedder` - Text to vector embedding generation

use crate::core::{Capabilities, Chunk, FullTextQuery, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::search::FusionConfig;
use async_trait::async_trait;
use futures::stream::BoxStream;
use std::fmt::Debug;
use std::sync::Arc;

/// A vector storage backend.
///
/// Backends implement storage and retrieval of chunks (embedded document pieces).
/// `Dripstone` is generic over backends, allowing for different storage solutions
/// while maintaining a consistent API.
///
/// # Implementation Notes
///
/// - All methods are async and fallible
/// - Backends should report their capabilities via `capabilities()`
/// - `Dripstone` will adapt its behavior based on capabilities (e.g., client-side
///   hybrid search if the backend doesn't support it natively)
/// - When [`Capabilities::requires_exact_filter_pass`] is set, `native_filter` may
///   only preselect candidates; callers still rely on exact [`crate::search::Filter::matches`]
///   on hydrated chunks for full filter semantics
///
/// # Example Implementation
///
/// ```text
/// use dripstone::traits::Backend;
/// use dripstone::core::{Capabilities, Chunk, SearchQuery, SearchResult};
/// use dripstone::errors::DripstoneError;
///
/// struct MyBackend { /* ... */ }
///
/// #[async_trait]
/// impl Backend for MyBackend {
///     fn capabilities(&self) -> Capabilities {
///         Capabilities::memory()
///     }
///
///     async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
///         // Store chunks...
///         Ok(())
///     }
///
///     // ... implement other methods
/// }
/// ```
#[async_trait]
pub trait Backend: Send + Sync + Debug {
    /// Returns the capabilities of this backend.
    ///
    /// `Dripstone` uses this to determine what features need to be implemented
    /// client-side vs delegated to the backend.
    fn capabilities(&self) -> Capabilities;

    // =========================================================================
    // CRUD Operations
    // =========================================================================

    /// Insert or update chunks in the backend.
    ///
    /// If a chunk with the same ID already exists, it should be replaced.
    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError>;

    /// Retrieve a chunk by its ID.
    ///
    /// Returns `Ok(None)` if the chunk doesn't exist.
    async fn get(&self, id: &str, namespace: Option<&str>)
    -> Result<Option<Chunk>, DripstoneError>;

    /// Delete a chunk by its ID.
    ///
    /// Returns `Ok(())` even if the chunk doesn't exist (idempotent).
    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError>;

    /// Delete all chunks belonging to a document.
    ///
    /// This deletes all chunks with IDs matching the pattern `{doc_id}#*`.
    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError>;

    // =========================================================================
    // Search Operations
    // =========================================================================

    /// Execute a vector similarity search.
    ///
    /// Returns results sorted by relevance (best first).
    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError>;

    /// Execute a native BM25/MAXSCORE full-text search when supported.
    async fn search_full_text(
        &self,
        _query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        Err(DripstoneError::pipeline(
            "backend does not support native full-text search",
        ))
    }

    /// Execute a native dense + lexical hybrid search when supported.
    async fn search_hybrid_full_text(
        &self,
        _dense: &SearchQuery,
        _lexical: &FullTextQuery,
        _fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        Err(DripstoneError::pipeline(
            "backend does not support native dense + full-text hybrid search",
        ))
    }

    /// Execute a streaming vector similarity search.
    ///
    /// Results are yielded as they become available, useful for large result sets
    /// or when you want to start processing results before the full search completes.
    ///
    /// Default implementation collects all results and streams them.
    fn search_stream<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxStream<'a, Result<SearchResult, DripstoneError>> {
        use futures::StreamExt;

        // Default: collect all results then stream
        let query = query.clone();
        Box::pin(
            futures::stream::once(async move { self.search(&query).await }).flat_map(|result| {
                match result {
                    Ok(results) => futures::stream::iter(results.into_iter().map(Ok)).boxed(),
                    Err(e) => futures::stream::once(async move { Err(e) }).boxed(),
                }
            }),
        )
    }

    // =========================================================================
    // Namespace Management
    // =========================================================================

    /// List all namespaces in the backend.
    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError>;

    /// Delete an entire namespace and all its chunks.
    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError>;

    /// Count chunks in the backend, optionally filtered by namespace.
    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError>;

    /// List all chunks in the backend, optionally filtered by namespace.
    ///
    /// Returns all chunks in the specified namespace (or default namespace if None).
    /// This is useful for operations that need to iterate over all stored chunks,
    /// such as cleanup, migration, or bulk operations.
    ///
    /// Note: For large namespaces, this may be memory-intensive. Consider using
    /// pagination or streaming if available.
    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError>;

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /// Flush any pending writes to durable storage.
    ///
    /// Some backends buffer writes for performance. Call this to ensure
    /// all data is persisted.
    async fn flush(&self) -> Result<(), DripstoneError> {
        Ok(()) // Default: no-op
    }

    /// Flush deferred ANN index inserts when supported.
    ///
    /// [`IndexedBackend`](crate::backend::indexed::IndexedBackend) drains its
    /// pending queue into the in-memory graph. Other backends no-op.
    async fn flush_pending_ann_index(&self) -> Result<(), DripstoneError> {
        Ok(())
    }

    /// Count vectors committed to storage but not yet inserted into the ANN index.
    fn pending_ann_index_count(&self) -> usize {
        0
    }

    /// Close the backend and release resources.
    ///
    /// After calling this, the backend should not be used.
    async fn close(&self) -> Result<(), DripstoneError> {
        Ok(()) // Default: no-op
    }
}

/// Blanket implementation for `Box<dyn Backend>` to enable dynamic dispatch.
#[async_trait]
impl Backend for Box<dyn Backend> {
    fn capabilities(&self) -> Capabilities {
        (**self).capabilities()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        (**self).upsert(chunks).await
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        (**self).get(id, namespace).await
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        (**self).delete(id, namespace).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        (**self).delete_document(doc_id, namespace).await
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        (**self).search(query).await
    }

    async fn search_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        (**self).search_full_text(query).await
    }

    async fn search_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        (**self)
            .search_hybrid_full_text(dense, lexical, fusion)
            .await
    }

    fn search_stream<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxStream<'a, Result<SearchResult, DripstoneError>> {
        (**self).search_stream(query)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        (**self).namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        (**self).delete_namespace(namespace).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        (**self).count(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        (**self).list(namespace, limit).await
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        (**self).flush().await
    }

    async fn flush_pending_ann_index(&self) -> Result<(), DripstoneError> {
        (**self).flush_pending_ann_index().await
    }

    fn pending_ann_index_count(&self) -> usize {
        (**self).pending_ann_index_count()
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        (**self).close().await
    }
}

/// Text to vector embedding generator.
///
/// Embedders convert text into dense vector representations suitable for
/// similarity search. `Dripstone` is generic over embedders, supporting both
/// local models (ONNX, Candle) and API-based providers.
///
/// # Implementation Notes
///
/// - `embed` is for document text (may include special prefixes)
/// - `embed_query` is for search queries (may use different prompts)
/// - Implementations should be thread-safe (Send + Sync)
///
/// # Example Implementation
///
/// ```text
/// use dripstone::traits::Embedder;
/// use dripstone::errors::DripstoneError;
///
/// struct MyEmbedder { dimension: usize }
///
/// #[async_trait]
/// impl Embedder for MyEmbedder {
///     async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DripstoneError> {
///         // Generate embeddings for each text...
///         Ok(vec![vec![0.0; self.dimension]; texts.len()])
///     }
///
///     async fn embed_query(&self, query: &str) -> Result<Vec<f32>, DripstoneError> {
///         // For some models, query embedding uses a different prompt
///         self.embed(&[query])
///             .await
///             .and_then(|v| v.into_iter().next().ok_or_else(|| {
///                 DripstoneError::storage("empty embedding response")
///             }))
///     }
///
///     fn dimension(&self) -> usize {
///         self.dimension
///     }
///
///     fn name(&self) -> &str {
///         "my-embedder"
///     }
/// }
/// ```
#[async_trait]
pub trait Embedder: Send + Sync + Debug {
    /// Generate embeddings for multiple texts.
    ///
    /// For document ingestion, texts may be automatically prefixed with
    /// model-specific prompts (e.g., "passage: " for E5 models).
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DripstoneError>;

    /// Generate an embedding for a search query.
    ///
    /// Query embeddings may use different prompts than document embeddings
    /// (e.g., "query: " for E5 models).
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, DripstoneError>;

    /// Returns the dimension of the embedding vectors.
    fn dimension(&self) -> usize;

    /// Returns the model name/identifier.
    fn name(&self) -> &str;

    /// Returns the maximum sequence length the model supports.
    ///
    /// Returns `None` if there's no hard limit.
    fn max_sequence_length(&self) -> Option<usize> {
        None
    }
}

/// Blanket implementation for `Arc<dyn Backend>` to enable dynamic dispatch with shared ownership.
#[async_trait]
impl Backend for Arc<dyn Backend> {
    fn capabilities(&self) -> Capabilities {
        (**self).capabilities()
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        (**self).upsert(chunks).await
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        (**self).get(id, namespace).await
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        (**self).delete(id, namespace).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        (**self).delete_document(doc_id, namespace).await
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        (**self).search(query).await
    }

    async fn search_full_text(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        (**self).search_full_text(query).await
    }

    async fn search_hybrid_full_text(
        &self,
        dense: &SearchQuery,
        lexical: &FullTextQuery,
        fusion: &FusionConfig,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        (**self)
            .search_hybrid_full_text(dense, lexical, fusion)
            .await
    }

    fn search_stream<'a>(
        &'a self,
        query: &'a SearchQuery,
    ) -> BoxStream<'a, Result<SearchResult, DripstoneError>> {
        (**self).search_stream(query)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        (**self).namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        (**self).delete_namespace(namespace).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        (**self).count(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        (**self).list(namespace, limit).await
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        (**self).flush().await
    }

    async fn flush_pending_ann_index(&self) -> Result<(), DripstoneError> {
        (**self).flush_pending_ann_index().await
    }

    fn pending_ann_index_count(&self) -> usize {
        (**self).pending_ann_index_count()
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        (**self).close().await
    }
}

/// Blanket implementation for `Box<dyn Embedder>` to enable dynamic dispatch.
#[async_trait]
impl Embedder for Box<dyn Embedder> {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DripstoneError> {
        (**self).embed(texts).await
    }

    async fn embed_query(&self, query: &str) -> Result<Vec<f32>, DripstoneError> {
        (**self).embed_query(query).await
    }

    fn dimension(&self) -> usize {
        (**self).dimension()
    }

    fn name(&self) -> &str {
        (**self).name()
    }

    fn max_sequence_length(&self) -> Option<usize> {
        (**self).max_sequence_length()
    }
}

/// Run backend search, applying client-side MaxSim when the backend lacks
/// [`Capabilities::native_maxsim`].
///
/// Non-native backends are overfetched, candidates are hydrated via `get` when
/// `multi_vectors` are missing from the dense hit payload, then reranked with
/// [`crate::search::rerank_with_maxsim`].
///
/// # Errors
///
/// Returns when backend search, hydration, or MaxSim scoring fails.
pub async fn search_with_maxsim_fallback<B: Backend + ?Sized>(
    backend: &B,
    query: &SearchQuery,
) -> Result<Vec<SearchResult>, DripstoneError> {
    if query.query_multi_vectors.is_none() || backend.capabilities().native_maxsim {
        return backend.search(query).await;
    }

    let mut dense_query = query.clone();
    dense_query.limit = crate::search::overfetch_limit(query);
    let dense = backend.search(&dense_query).await?;
    let hydrated = hydrate_multi_vectors(backend, dense).await?;
    crate::search::rerank_with_maxsim(hydrated, query)
}

async fn hydrate_multi_vectors<B: Backend + ?Sized>(
    backend: &B,
    mut results: Vec<SearchResult>,
) -> Result<Vec<SearchResult>, DripstoneError> {
    for result in &mut results {
        if result.chunk.multi_vectors.is_some() {
            continue;
        }
        let ns = result.chunk.namespace.as_deref();
        if let Some(full) = backend.get(&result.chunk.id, ns).await?
            && full.multi_vectors.is_some()
        {
            result.chunk.multi_vectors = full.multi_vectors;
        }
    }
    Ok(results)
}

/// Type aliases for dynamic dispatch scenarios.
pub type DynBackend = Box<dyn Backend>;
pub type DynEmbedder = Box<dyn Embedder>;

#[cfg(test)]
mod tests {
    use super::*;

    // A simple mock backend for testing
    #[derive(Debug)]
    struct MockBackend;

    #[async_trait]
    impl Backend for MockBackend {
        fn capabilities(&self) -> Capabilities {
            Capabilities::memory()
        }

        async fn upsert(&self, _chunks: &[Chunk]) -> Result<(), DripstoneError> {
            Ok(())
        }

        async fn get(
            &self,
            _id: &str,
            _namespace: Option<&str>,
        ) -> Result<Option<Chunk>, DripstoneError> {
            Ok(None)
        }

        async fn delete(&self, _id: &str, _namespace: Option<&str>) -> Result<(), DripstoneError> {
            Ok(())
        }

        async fn delete_document(
            &self,
            _doc_id: &str,
            _namespace: Option<&str>,
        ) -> Result<usize, DripstoneError> {
            Ok(0)
        }

        async fn search(&self, _query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
            Ok(vec![])
        }

        async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
            Ok(vec![])
        }

        async fn delete_namespace(&self, _namespace: &str) -> Result<(), DripstoneError> {
            Ok(())
        }

        async fn count(&self, _namespace: Option<&str>) -> Result<usize, DripstoneError> {
            Ok(0)
        }

        async fn list(
            &self,
            _namespace: Option<&str>,
            _limit: Option<usize>,
        ) -> Result<Vec<Chunk>, DripstoneError> {
            Ok(vec![])
        }
    }

    // A simple mock embedder for testing
    #[derive(Debug)]
    struct MockEmbedder {
        dim: usize,
    }

    #[async_trait]
    impl Embedder for MockEmbedder {
        async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DripstoneError> {
            Ok(texts.iter().map(|_| vec![0.0; self.dim]).collect())
        }

        async fn embed_query(&self, _query: &str) -> Result<Vec<f32>, DripstoneError> {
            Ok(vec![0.0; self.dim])
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn name(&self) -> &'static str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_mock_backend() {
        let backend = MockBackend;
        assert!(backend.capabilities().supports_namespaces);

        let chunk = Chunk::new(
            "doc-1",
            0,
            "test",
            vec![0.1, 0.2],
            0,
            4,
            1,
            serde_json::json!({}),
            None,
        );
        backend
            .upsert(&[chunk])
            .await
            .expect("upsert should succeed for mock backend");

        let query = SearchQuery::new(vec![0.1, 0.2]);
        let results = backend
            .search(&query)
            .await
            .expect("search should succeed for mock backend");
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_mock_embedder() {
        let embedder = MockEmbedder { dim: 384 };
        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.name(), "mock");

        let embeddings = embedder
            .embed(&["hello", "world"])
            .await
            .expect("embed should succeed for mock embedder");
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 384);

        let query_embedding = embedder
            .embed_query("test query")
            .await
            .expect("embed_query should succeed for mock embedder");
        assert_eq!(query_embedding.len(), 384);
    }

    #[tokio::test]
    async fn test_dyn_backend() {
        let backend: DynBackend = Box::new(MockBackend);
        assert!(backend.capabilities().supports_namespaces);
    }

    #[tokio::test]
    async fn test_dyn_embedder() {
        let embedder: DynEmbedder = Box::new(MockEmbedder { dim: 256 });
        assert_eq!(embedder.dimension(), 256);
    }
}
