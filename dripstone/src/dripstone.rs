//! The main `Dripstone` entry point.
//!
//! `Dripstone<B, E>` is the unified entry point for all vector operations:
//! - Document ingestion (chunk -> embed -> store)
//! - Search (embed query -> search -> optional rerank)
//! - Management (namespaces, counts, deletion)

use crate::chunk_hash_store::ChunkHashStore;
use crate::config::DripstoneConfig;
use crate::core::{
    Chunk, Document, FullTextOptions, PreChunk, SearchMode, SearchQuery, SearchResult,
};
use crate::errors::DripstoneError;
#[cfg(feature = "local")]
use crate::local::{LocalEmbedder, OpenBackend};
#[cfg(feature = "rust-stemmers")]
use crate::rag::HybridSearchHelper;
#[cfg(feature = "embedding-ort")]
use crate::rag::rerank::{CrossEncoderConfig, CrossEncoderReranker};
use crate::traits::{Backend, Embedder};
use serde::Serialize;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

mod ingest;
mod search_builder;

/// The main entry point for `Dripstone` operations.
///
/// `Dripstone` is generic over:
/// - `B: Backend` - The vector storage backend
/// - `E: Embedder` - The embedding model
///
/// This design provides:
/// - Zero-cost abstractions via monomorphization
/// - Type-safe configuration
/// - Unified API for all operations
///
/// # Example
///
/// ```text
/// use dripstone::{EmbeddingConfig, MemoryBackend, OrtEmbedder, Dripstone};
///
/// // Requires features `memory-backend` and `embedding-ort`.
/// let backend = MemoryBackend::new();
/// let embedder = OrtEmbedder::new(
///     EmbeddingConfig::new().with_model_repo("./models/minilm"),
/// )?;
/// let vf = Dripstone::new(backend, embedder);
///
/// vf.ingest("doc-1", "Hello, world!")
///     .namespace("knowledge-base")
///     .await?;
/// let results = vf.search("greeting").limit(5).await?;
/// ```
pub struct Dripstone<B: Backend, E: Embedder> {
    backend: Arc<B>,
    embedder: Arc<E>,
    config: DripstoneConfig,
    chunk_hash_store: Option<Arc<dyn ChunkHashStore>>,
    #[cfg(feature = "rust-stemmers")]
    hybrid_helper: Option<Arc<HybridSearchHelper>>,
    #[cfg(feature = "embedding-ort")]
    reranker: Option<Arc<CrossEncoderReranker>>,
}

impl<B: Backend, E: Embedder> Debug for Dripstone<B, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("Dripstone");
        debug
            .field("backend", &"<Backend>")
            .field("embedder", &"<Embedder>")
            .field("config", &self.config)
            .field("chunk_hash_store", &self.chunk_hash_store.is_some());
        #[cfg(feature = "rust-stemmers")]
        debug.field("hybrid_helper", &self.hybrid_helper.is_some());
        #[cfg(feature = "embedding-ort")]
        debug.field("reranker", &self.reranker.is_some());
        debug.finish()
    }
}

#[cfg(feature = "local")]
impl Dripstone<OpenBackend, LocalEmbedder> {
    /// Open a directory-backed store with the default in-process embedder.
    ///
    /// The store keeps text, so [`Self::search`] ranks with dense vectors plus
    /// BM25 using reciprocal-rank fusion. Call [`SearchBuilder::dense_only`]
    /// to rank that query with dense similarity only. [`Self::new`] and
    /// [`Self::builder`] stay on dense search.
    ///
    /// The default embedder hashes tokens into a fixed vector. It does not
    /// download a model. Enable `embedding-ort` or `embedding-candle` when a
    /// neural embedder is required, and pass that embedder to [`Self::new`].
    /// The runnable form of this path is `examples/quickstart.rs`.
    ///
    /// A second `open` of the same directory fails while this handle is alive.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory is not writable, when another handle
    /// already holds the directory, or when `store.json` parses but its version,
    /// embedder, or dimension is not supported. An unsupported snapshot is left
    /// in place. The backup is not loaded and the file is not rewritten.
    ///
    /// An empty or unreadable `store.json` is renamed to
    /// `store.json.corrupt-<timestamp>` and a warning is logged. `open` then
    /// loads `store.json.bak` when that backup is a supported snapshot. The
    /// quarantined file is not overwritten. If the backup cannot restore the
    /// store, `open` returns an error. A later `open` still refuses while that
    /// quarantine file remains and no snapshot was restored.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, DripstoneError> {
        let backend = OpenBackend::open(path)?;
        let config = DripstoneConfig::default()
            .with_chunking(|chunking| chunking.strategy(crate::config::ChunkStrategy::None))
            .with_search(|search| {
                search
                    .default_mode(SearchMode::HybridDenseFullText)
                    .hybrid_full_text_fusion(crate::search::FusionConfig::reciprocal_rank())
            });
        Ok(Self::with_config(backend, LocalEmbedder::new(), config))
    }
}

impl<B: Backend, E: Embedder> Dripstone<B, E> {
    #[cfg(feature = "rust-stemmers")]
    fn hybrid_helper_for_config(config: &DripstoneConfig) -> Option<Arc<HybridSearchHelper>> {
        if config.search.hybrid_enabled {
            Some(Arc::new(HybridSearchHelper::new()))
        } else {
            None
        }
    }

    #[cfg(feature = "embedding-ort")]
    fn reranker_for_config(config: &DripstoneConfig) -> Option<Arc<CrossEncoderReranker>> {
        if config.search.rerank_enabled {
            config.search.rerank_model_path.as_ref().and_then(|path| {
                CrossEncoderReranker::new(CrossEncoderConfig::new(
                    path.to_string_lossy().to_string(),
                ))
                .ok()
                .map(Arc::new)
            })
        } else {
            None
        }
    }

    /// Create a new `Dripstone` instance with default configuration.
    pub fn new(backend: B, embedder: E) -> Self {
        Self::with_config(backend, embedder, DripstoneConfig::default())
    }

    /// Store one text document.
    ///
    /// This is the short form of [`Self::ingest`] for a single document.
    ///
    /// # Errors
    ///
    /// Returns an error when chunking, embedding, or the backend write fails.
    pub async fn add(&self, id: &str, text: &str) -> Result<IngestResult, DripstoneError> {
        self.ingest(id, text).await
    }

    /// Create a new `Dripstone` instance with custom configuration.
    pub fn with_config(backend: B, embedder: E, config: DripstoneConfig) -> Self {
        Self::with_config_and_hash_store(backend, embedder, config, None)
    }

    /// Create a new `Dripstone` instance with custom configuration and optional chunk hash store.
    pub fn with_config_and_hash_store(
        backend: B,
        embedder: E,
        config: DripstoneConfig,
        chunk_hash_store: Option<Arc<dyn ChunkHashStore>>,
    ) -> Self {
        Self {
            backend: Arc::new(backend),
            embedder: Arc::new(embedder),
            chunk_hash_store,
            #[cfg(feature = "rust-stemmers")]
            hybrid_helper: Self::hybrid_helper_for_config(&config),
            #[cfg(feature = "embedding-ort")]
            reranker: Self::reranker_for_config(&config),
            config,
        }
    }

    /// Create a builder for more complex configuration.
    #[must_use]
    pub fn builder() -> DripstoneBuilder<B, E> {
        DripstoneBuilder::new()
    }

    /// Get a reference to the configuration.
    pub const fn config(&self) -> &DripstoneConfig {
        &self.config
    }

    /// Get a reference to the backend.
    #[must_use]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Get a reference to the embedder.
    #[must_use]
    pub fn embedder(&self) -> &E {
        &self.embedder
    }

    /// Get the embedding dimension.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.embedder.dimension()
    }

    // =========================================================================
    // Ingest API
    // =========================================================================

    /// Start an ingest operation for a document.
    ///
    /// This returns an `IngestBuilder` that allows you to configure the ingestion
    /// before executing it.
    ///
    /// # Example
    ///
    /// ```text
    /// // Simple ingestion
    /// vf.ingest("doc-1", "Document content").await?;
    ///
    /// // With options
    /// vf.ingest("doc-2", content)
    ///     .namespace("papers")
    ///     .metadata(MyMeta { author: "Alice" })
    ///     .await?;
    /// ```
    #[must_use]
    pub fn ingest<'a>(&'a self, doc_id: &'a str, content: &'a str) -> IngestBuilder<'a, B, E> {
        IngestBuilder::new(self, doc_id, content)
    }

    /// Ingest a document with pre-computed chunks (skips internal chunking).
    ///
    /// Use when chunking is done externally, e.g. AST-aware chunking at
    /// function/class/object boundaries. `content` is the full document text
    /// (used for hashline line number calculation when enabled).
    #[must_use]
    pub fn ingest_chunks<'a>(
        &'a self,
        doc_id: &'a str,
        content: &'a str,
        chunks: Vec<PreChunk>,
    ) -> IngestChunksBuilder<'a, B, E> {
        IngestChunksBuilder::new(self, doc_id, content, chunks)
    }

    /// Ingest multiple documents in a batch.
    ///
    /// More efficient than individual ingestions for large document sets.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn ingest_batch(
        &self,
        documents: Vec<Document>,
    ) -> Result<IngestResult, DripstoneError> {
        let mut total_chunks = 0;
        for doc in documents {
            let result = self.ingest(&doc.id, &doc.content).await?;
            total_chunks += result.chunks_created;
        }
        Ok(IngestResult {
            chunks_created: total_chunks,
        })
    }

    // =========================================================================
    // Search API
    // =========================================================================

    /// Start a search operation.
    ///
    /// This returns a `SearchBuilder` that allows you to configure the search
    /// before executing it.
    ///
    /// # Example
    ///
    /// ```text
    /// // Simple search
    /// let results = vf.search("my query").await?;
    ///
    /// // With options
    /// let results = vf.search("my query")
    ///     .limit(20)
    ///     .namespace("papers")
    ///     .hybrid(0.7)
    ///     .rerank(true)
    ///     .await?;
    /// ```
    pub fn search<'a>(&'a self, query: &'a str) -> SearchBuilder<'a, B, E> {
        SearchBuilder::new(self, query)
    }

    // =========================================================================
    // Management API
    // =========================================================================

    /// Get a chunk by its ID.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        self.backend.get(id, namespace).await
    }

    /// Delete a chunk by its ID.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        self.backend.delete(id, namespace).await
    }

    /// Delete all chunks for a document.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let count = self.backend.delete_document(doc_id, namespace).await?;
        if let Some(ref store) = self.chunk_hash_store {
            let _ = store.delete_doc(doc_id, namespace).await;
        }
        Ok(count)
    }

    /// List all namespaces.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.backend.namespaces().await
    }

    /// Delete an entire namespace.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        self.backend.delete_namespace(namespace).await
    }

    /// Count chunks in the store.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        self.backend.count(namespace).await
    }

    /// Flush any pending writes.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn flush(&self) -> Result<(), DripstoneError> {
        self.backend.flush().await
    }
}

/// Builder for constructing `Dripstone` instances.
pub struct DripstoneBuilder<B: Backend, E: Embedder> {
    backend: Option<B>,
    embedder: Option<E>,
    config: DripstoneConfig,
    chunk_hash_store: Option<Arc<dyn ChunkHashStore>>,
    _phantom: PhantomData<(B, E)>,
}

impl<B: Backend, E: Embedder> DripstoneBuilder<B, E> {
    /// Create a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            backend: None,
            embedder: None,
            config: DripstoneConfig::default(),
            chunk_hash_store: None,
            _phantom: PhantomData,
        }
    }

    /// Set the chunk hash store for incremental indexing.
    #[must_use]
    pub fn chunk_hash_store(mut self, store: Arc<dyn ChunkHashStore>) -> Self {
        self.chunk_hash_store = Some(store);
        self
    }

    /// Set the backend.
    #[must_use]
    pub fn backend(mut self, backend: B) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set the embedder.
    #[must_use]
    pub fn embedder(mut self, embedder: E) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Set the configuration.
    #[must_use]
    pub fn config(mut self, config: DripstoneConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the `Dripstone` instance.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - backend or embedder are not set, or
    /// - `hashline.incremental_indexing` is enabled without a chunk hash
    ///   store configured via [`Self::chunk_hash_store`]. Incremental
    ///   indexing is a no-op without persistence, so silently running in
    ///   non-incremental mode would be a correctness surprise.
    pub fn build(self) -> Result<Dripstone<B, E>, DripstoneError> {
        let backend = self
            .backend
            .ok_or_else(|| DripstoneError::configuration("backend is required"))?;
        let embedder = self
            .embedder
            .ok_or_else(|| DripstoneError::configuration("embedder is required"))?;

        if self.config.hashline.enabled
            && self.config.hashline.incremental_indexing
            && self.chunk_hash_store.is_none()
        {
            return Err(DripstoneError::configuration(
                "hashline.incremental_indexing is enabled but no chunk hash store was provided; \
                 call DripstoneBuilder::chunk_hash_store(...) or disable incremental_indexing",
            ));
        }

        Ok(Dripstone::with_config_and_hash_store(
            backend,
            embedder,
            self.config,
            self.chunk_hash_store,
        ))
    }
}

impl<B: Backend, E: Embedder> Default for DripstoneBuilder<B, E> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for ingest operations with pre-computed chunks.
pub struct IngestChunksBuilder<'a, B: Backend, E: Embedder> {
    dripstone: &'a Dripstone<B, E>,
    doc_id: &'a str,
    content: &'a str,
    chunks: Vec<PreChunk>,
    namespace: Option<String>,
    metadata: serde_json::Value,
}

/// Builder for ingest operations.
pub struct IngestBuilder<'a, B: Backend, E: Embedder> {
    dripstone: &'a Dripstone<B, E>,
    doc_id: &'a str,
    content: &'a str,
    namespace: Option<String>,
    metadata: serde_json::Value,
}

/// Internal struct for chunked text before embedding.
struct ChunkText {
    text: String,
    start_offset: usize,
    end_offset: usize,
    metadata: Option<serde_json::Value>,
}

impl ChunkText {
    fn new(text: impl Into<String>, start_offset: usize, end_offset: usize) -> Self {
        Self {
            text: text.into(),
            start_offset,
            end_offset,
            metadata: None,
        }
    }

    fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

/// Internal struct for sentence data with offsets.
struct SentenceData {
    text: String,
    start_offset: usize,
    end_offset: usize,
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::cosine_similarity(a, b).unwrap_or(0.0)
}

/// Result of an ingest operation.
#[derive(Debug, Clone)]
pub struct IngestResult {
    /// Number of chunks created and stored.
    pub chunks_created: usize,
}

/// Builder for search operations.
#[must_use]
pub struct SearchBuilder<'a, B: Backend, E: Embedder> {
    dripstone: &'a Dripstone<B, E>,
    query: &'a str,
    search_mode: SearchMode,
    limit: Option<usize>,
    namespace: Option<String>,
    filter: Option<crate::search::Filter>,
    hybrid_weight: Option<f32>,
    hybrid_full_text_weight: Option<f32>,
    hybrid_full_text_fusion: Option<crate::search::FusionConfig>,
    full_text_options: FullTextOptions,
    rerank: bool,
    /// Override `config.hashline.format_results`: Some(true) = always format, Some(false) = never, None = use config
    hashline_format: Option<bool>,
    mmr_config: Option<crate::search::MmrConfig>,
    #[cfg(feature = "search-advanced")]
    temporal_decay_config: Option<crate::search::TemporalDecayConfig>,
    /// Optional late-interaction query tokens (ColBERT / MaxSim).
    query_multi_vectors: Option<crate::search::MultiVectors>,
    maxsim: Option<crate::search::MaxSimOptions>,
}

/// Type alias for `Dripstone` with dynamic dispatch (for runtime flexibility).
pub type DynDripstone = Dripstone<Box<dyn Backend>, Box<dyn Embedder>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Capabilities, FullTextQuery, SearchQuery};
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use std::sync::Arc;

    #[derive(Debug, Default)]
    struct TestBackend {
        captured: Option<Arc<Mutex<Vec<Chunk>>>>,
    }

    #[async_trait]
    impl Backend for TestBackend {
        fn capabilities(&self) -> Capabilities {
            Capabilities::memory()
        }

        async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
            if let Some(captured) = &self.captured {
                captured.lock().extend(chunks.iter().cloned());
            }
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
            Ok(Vec::new())
        }

        async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
            Ok(Vec::new())
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
            Ok(Vec::new())
        }
    }

    #[derive(Debug, Default, Clone)]
    struct RecordingState {
        dense: usize,
        full_text: usize,
        hybrid_full_text: usize,
    }

    #[derive(Debug, Clone)]
    struct RecordingBackend {
        state: Arc<Mutex<RecordingState>>,
    }

    impl RecordingBackend {
        fn new() -> (Self, Arc<Mutex<RecordingState>>) {
            let state = Arc::new(Mutex::new(RecordingState::default()));
            (
                Self {
                    state: Arc::clone(&state),
                },
                state,
            )
        }

        fn make_result(id: &str, score: f32) -> SearchResult {
            SearchResult::new(
                Chunk::new(
                    id,
                    0,
                    id,
                    vec![1.0, 0.0],
                    0,
                    id.len(),
                    1,
                    serde_json::json!({}),
                    None,
                ),
                score,
                0,
            )
        }
    }

    #[async_trait]
    impl Backend for RecordingBackend {
        fn capabilities(&self) -> Capabilities {
            Capabilities {
                native_full_text: true,
                native_hybrid_full_text: true,
                ..Capabilities::memory()
            }
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
            self.state.lock().dense += 1;
            Ok(vec![Self::make_result("dense-doc", 1.0)])
        }

        async fn search_full_text(
            &self,
            _query: &FullTextQuery,
        ) -> Result<Vec<SearchResult>, DripstoneError> {
            self.state.lock().full_text += 1;
            Ok(vec![Self::make_result("lexical-doc", 2.0)])
        }

        async fn search_hybrid_full_text(
            &self,
            _dense: &SearchQuery,
            _lexical: &FullTextQuery,
            _fusion: &crate::search::FusionConfig,
        ) -> Result<Vec<SearchResult>, DripstoneError> {
            self.state.lock().hybrid_full_text += 1;
            Ok(vec![Self::make_result("hybrid-doc", 3.0)])
        }

        async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
            Ok(Vec::new())
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
            Ok(Vec::new())
        }
    }

    #[derive(Debug)]
    struct TestEmbedder;

    #[async_trait]
    impl Embedder for TestEmbedder {
        async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, DripstoneError> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }

        async fn embed_query(&self, _query: &str) -> Result<Vec<f32>, DripstoneError> {
            Ok(vec![1.0, 0.0])
        }

        fn dimension(&self) -> usize {
            2
        }

        fn name(&self) -> &'static str {
            "test"
        }
    }

    #[test]
    fn test_config_default() {
        let config = DripstoneConfig::default();
        assert_eq!(config.chunking.target_tokens, 512);
        assert_eq!(config.search.default_limit, 10);
    }

    #[test]
    fn ingest_builder_try_metadata_returns_error_for_non_serializable_metadata() {
        #[derive(Debug)]
        struct NotSerializable;

        impl Serialize for NotSerializable {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(serde::ser::Error::custom("nope"))
            }
        }

        let vf = Dripstone::new(TestBackend::default(), TestEmbedder);
        let builder = vf.ingest("doc-1", "hello");
        let Err(err) = builder.try_metadata(NotSerializable) else {
            panic!("metadata should fail");
        };
        assert!(matches!(err, DripstoneError::Serialization(_)));
    }

    #[test]
    fn ingest_chunks_builder_try_metadata_returns_error_for_non_serializable_metadata() {
        #[derive(Debug)]
        struct NotSerializable;

        impl Serialize for NotSerializable {
            fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                Err(serde::ser::Error::custom("nope"))
            }
        }

        let vf = Dripstone::new(TestBackend::default(), TestEmbedder);
        let builder = vf.ingest_chunks("doc-1", "hello", vec![PreChunk::new("hello", 0, 5)]);
        let Err(err) = builder.try_metadata(NotSerializable) else {
            panic!("metadata should fail");
        };
        assert!(matches!(err, DripstoneError::Serialization(_)));
    }

    #[tokio::test]
    async fn search_on_new_stays_dense() {
        let (backend, state) = RecordingBackend::new();
        let vf = Dripstone::new(backend, TestEmbedder);
        let results = vf.search("rust search").await.expect("dense search");
        assert_eq!(results[0].chunk.doc_id, "dense-doc");
        let state = state.lock().clone();
        assert_eq!(state.dense, 1);
        assert_eq!(state.full_text, 0);
        assert_eq!(state.hybrid_full_text, 0);
    }

    #[tokio::test]
    async fn search_on_builder_stays_dense() {
        let (backend, state) = RecordingBackend::new();
        let vf = Dripstone::builder()
            .backend(backend)
            .embedder(TestEmbedder)
            .build()
            .expect("builder");
        let results = vf.search("rust search").await.expect("dense search");
        assert_eq!(results[0].chunk.doc_id, "dense-doc");
        assert_eq!(vf.config().search.default_mode, SearchMode::Dense);
        let state = state.lock().clone();
        assert_eq!(state.dense, 1);
        assert_eq!(state.full_text, 0);
        assert_eq!(state.hybrid_full_text, 0);
    }

    #[tokio::test]
    async fn builder_default_search_mode_is_dense() {
        let (backend, _state) = RecordingBackend::new();
        let built = Dripstone::builder()
            .backend(backend)
            .embedder(TestEmbedder)
            .build()
            .expect("builder");
        assert_eq!(built.config().search.default_mode, SearchMode::Dense);
    }

    #[tokio::test]
    async fn configured_default_mode_is_what_search_runs() {
        let (backend, state) = RecordingBackend::new();
        let config = crate::config::DripstoneConfig::default()
            .with_search(|search| search.default_mode(SearchMode::HybridDenseFullText));
        let built = Dripstone::with_config(backend, TestEmbedder, config);
        let _results = built.search("rust search").await.expect("hybrid search");
        let state = state.lock().clone();
        assert_eq!(state.hybrid_full_text, 1);
        assert_eq!(state.dense, 0);
        assert_eq!(state.full_text, 0);
    }

    #[tokio::test]
    async fn search_on_with_config_hybrid_flag_stays_dense() {
        let (backend, state) = RecordingBackend::new();
        let config = DripstoneConfig::default().with_search(|search| search.hybrid(true));
        let vf = Dripstone::with_config(backend, TestEmbedder, config);
        let results = vf.search("rust search").await.expect("dense search");
        assert_eq!(results[0].chunk.doc_id, "dense-doc");
        let state = state.lock().clone();
        assert_eq!(state.dense, 1);
        assert_eq!(state.full_text, 0);
        assert_eq!(state.hybrid_full_text, 0);
    }

    #[tokio::test]
    async fn search_builder_routes_full_text_queries_to_backend() {
        let (backend, state) = RecordingBackend::new();
        let vf = Dripstone::new(backend, TestEmbedder);

        let results = vf
            .search("rust search")
            .full_text()
            .await
            .expect("full-text search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "lexical-doc#0");

        let state = state.lock().clone();
        assert_eq!(state.full_text, 1);
        assert_eq!(state.dense, 0);
        assert_eq!(state.hybrid_full_text, 0);
    }

    #[tokio::test]
    async fn search_builder_routes_hybrid_full_text_queries_to_backend() {
        let (backend, state) = RecordingBackend::new();
        let vf = Dripstone::new(backend, TestEmbedder);

        let results = vf
            .search("rust search")
            .hybrid_full_text(0.4)
            .await
            .expect("hybrid full-text search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.id, "hybrid-doc#0");

        let state = state.lock().clone();
        assert_eq!(state.hybrid_full_text, 1);
        assert_eq!(state.dense, 0);
        assert_eq!(state.full_text, 0);
    }

    #[cfg(feature = "embedding-ort")]
    #[tokio::test]
    async fn search_builder_errors_when_rerank_requested_without_model() {
        let (backend, _state) = RecordingBackend::new();
        let vf = Dripstone::new(backend, TestEmbedder);

        let err = vf
            .search("rust search")
            .rerank(true)
            .await
            .expect_err("reranking without a configured model should fail");

        assert!(
            matches!(err, DripstoneError::Misconfiguration(message) if message.contains("no ONNX reranker is configured"))
        );
    }

    #[cfg(feature = "ast")]
    #[tokio::test]
    async fn ast_aware_ingest_records_function_symbols() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let backend = TestBackend {
            captured: Some(Arc::clone(&captured)),
        };
        let config = crate::config::DripstoneConfig::default().with_chunking(|chunking| {
            chunking
                .strategy(crate::config::ChunkStrategy::AstAware)
                .code_language("python")
        });
        let vf = Dripstone::with_config(backend, TestEmbedder, config);
        let source = "def alpha(x):\n    return x\n\ndef beta(y):\n    return y\n";
        let created = vf
            .ingest("demo", source)
            .metadata(serde_json::json!({ "path": "demo.py" }))
            .await
            .expect("ingest")
            .chunks_created;
        assert_eq!(created, 2);
        let chunks = captured.lock().clone();
        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0]
                .metadata
                .get("symbol")
                .and_then(|value| value.as_str()),
            Some("alpha")
        );
        assert_eq!(
            chunks[1]
                .metadata
                .get("symbol")
                .and_then(|value| value.as_str()),
            Some("beta")
        );
        assert_eq!(
            chunks[0]
                .metadata
                .get("language")
                .and_then(|value| value.as_str()),
            Some("python")
        );
        assert!(chunks[0].content.contains("def alpha"));
    }
}
