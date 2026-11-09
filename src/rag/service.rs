//! Core RAG service implementation
//!
//! This module provides the core RAG functionality that can be used independently
//! or integrated into tools, pipelines, builders, and workflows.

use crate::agent::tools::context_manager::{ContextManager, ContextStats};
use crate::data::storage::vector::Filter;
use crate::llm::ProviderType;
use crate::llm::embedding::EmbeddingProvider;
use crate::messages::TokenUsage;
use crate::rag::{RagFusionMode, RagHashlineOptions, RagSearchMode};
use dripstone::config::{ChunkStrategy, DripstoneConfig};
use dripstone::graph::{GraphBackend, QueryableGraphBackend};
use dripstone::metrics::{NoOpMetrics, SharedMetrics, SimpleMetrics};
use dripstone::traits::{Backend, Embedder};
use dripstone::{
    ChunkHashStore, Dripstone, FullTextIndexedBackend, LateInteractionIndexedBackend,
    MemoryChunkHashStore, SearchResult,
};
use futures_util::stream::{self, StreamExt};
use std::sync::Arc;
use std::time::Instant;

type RagSearchBuilder<'a> =
    dripstone::dripstone::SearchBuilder<'a, Arc<dyn Backend>, EmbeddingProviderWrapper>;

struct SearchBuilderParams<'a> {
    query: &'a str,
    top_k: usize,
    search_mode: RagSearchMode,
    fusion_mode: RagFusionMode,
    dense_weight: Option<f32>,
    sparse_weight: Option<f32>,
    rerank: bool,
    filter: Option<Filter>,
    warnings: &'a mut Vec<String>,
}

/// Wrapper to implement Embedder trait for EmbeddingProvider
#[derive(Debug)]
pub struct EmbeddingProviderWrapper {
    provider: Arc<dyn EmbeddingProvider>,
    dimension: usize,
    name: String,
}

impl EmbeddingProviderWrapper {
    pub fn new(provider: Arc<dyn EmbeddingProvider>, dimension: usize, name: &str) -> Self {
        Self {
            provider,
            dimension,
            name: name.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl Embedder for EmbeddingProviderWrapper {
    async fn embed(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, dripstone::errors::DripstoneError> {
        use crate::llm::EmbeddingRequest;
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let request = EmbeddingRequest {
            texts: texts_owned,
            model: None,
        };
        self.provider
            .embed(request)
            .await
            .map(|response| response.embeddings)
            .map_err(|e| dripstone::errors::DripstoneError::storage(e.to_string()))
    }

    async fn embed_query(
        &self,
        query: &str,
    ) -> Result<Vec<f32>, dripstone::errors::DripstoneError> {
        self.embed(&[query]).await.and_then(|v| {
            v.into_iter().next().ok_or_else(|| {
                dripstone::errors::DripstoneError::storage("empty embedding response")
            })
        })
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// RAG service for document indexing and retrieval
///
/// This service provides the core RAG functionality and can be used independently
/// or integrated into tools, pipelines, builders, and workflows.
///
/// Uses Dripstone with the unified Backend trait for all vector operations.
pub struct RagService {
    embedding_provider: Arc<dyn EmbeddingProvider>,
    /// Dripstone instance for all vector operations
    dripstone: Arc<Dripstone<Arc<dyn Backend>, EmbeddingProviderWrapper>>,
    /// Metrics collector for tracking operations
    metrics: SharedMetrics,
    /// Chunking strategy; AstAware triggers structure-based chunking when path is provided
    #[cfg_attr(
        not(feature = "ast"),
        expect(dead_code, reason = "only AST-aware chunking reads the strategy")
    )]
    chunking_strategy: ChunkStrategy,
    /// Optional graph backend for Graph RAG
    graph_backend: Option<Arc<dyn GraphBackend>>,
    /// Optional queryable graph backend for hybrid retrieval (superset of graph_backend)
    queryable_graph_backend: Option<Arc<dyn QueryableGraphBackend>>,
    /// Optional LLM provider for entity extraction
    llm_provider: Option<Arc<dyn crate::llm::Provider>>,
    /// Namespace for isolating documents (e.g. per-session). Default "rag".
    namespace: String,
    /// Skip re-embedding unchanged hashline chunks on re-ingest.
    incremental_indexing: bool,
    /// In-process hashes used when [`Self::incremental_indexing`] is enabled.
    chunk_hash_store: Option<Arc<MemoryChunkHashStore>>,
    /// Token multi-vectors + MaxSim / PLAID late interaction.
    late_interaction: bool,
}

impl std::fmt::Debug for RagService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RagService")
            .field("embedding_provider", &"<dyn EmbeddingProvider>")
            .field("dripstone", &"<Dripstone>")
            .field("metrics", &"<SharedMetrics>")
            .field("graph_backend", &self.graph_backend.is_some())
            .field(
                "queryable_graph_backend",
                &self.queryable_graph_backend.is_some(),
            )
            .field("llm_provider", &self.llm_provider.is_some())
            .finish()
    }
}

const MAX_INDEX_CONCURRENCY: usize = 32;
const DEFAULT_INDEX_CONCURRENCY_CAP: usize = 4;

impl RagService {
    /// Embed concurrency for workspace ingest.
    ///
    /// `CALICHE_INDEX_CONCURRENCY` overrides the default (max 32). The default
    /// leaves one core for the interactive host and caps at 4 so background
    /// indexing after deferred REPL attach cannot saturate a laptop.
    pub(crate) fn index_concurrency() -> usize {
        Self::parse_index_concurrency(
            std::env::var("CALICHE_INDEX_CONCURRENCY").ok().as_deref(),
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(2),
        )
    }

    pub(crate) fn parse_index_concurrency(
        raw: Option<&str>,
        available_parallelism: usize,
    ) -> usize {
        raw.and_then(|value| value.parse().ok())
            .filter(|&n: &usize| n > 0)
            .map_or_else(
                || Self::default_index_concurrency(available_parallelism),
                |n| n.min(MAX_INDEX_CONCURRENCY),
            )
    }

    pub(crate) fn default_index_concurrency(available_parallelism: usize) -> usize {
        available_parallelism
            .saturating_sub(1)
            .clamp(1, DEFAULT_INDEX_CONCURRENCY_CAP)
    }

    /// How many file bodies to keep resident while ingesting.
    ///
    /// Sized from concurrency (`* 2`, clamped 4..=16) so a 100KB max-file window
    /// stays around 1.6MB of source text instead of the whole tree.
    pub(crate) fn index_ingest_window_size(concurrency: usize) -> usize {
        concurrency.saturating_mul(2).clamp(4, 16)
    }

    fn format_search_hits(
        results: impl IntoIterator<Item = SearchResult>,
        min_similarity: f32,
    ) -> Vec<String> {
        results
            .into_iter()
            .filter(|result| result.score >= min_similarity)
            .map(|result| format!("[Similarity: {:.3}] {}", result.score, result.chunk.content))
            .collect()
    }

    fn resolve_dense_weight(dense_weight: Option<f32>, sparse_weight: Option<f32>) -> f32 {
        match (dense_weight, sparse_weight) {
            (Some(dense), Some(sparse)) => {
                let dense = dense.max(0.0);
                let sparse = sparse.max(0.0);
                let total = dense + sparse;
                if total > 0.0 {
                    (dense / total).clamp(0.0, 1.0)
                } else {
                    0.7
                }
            }
            (Some(dense), None) => dense.clamp(0.0, 1.0),
            (None, Some(sparse)) => (1.0 - sparse).clamp(0.0, 1.0),
            (None, None) => 0.7,
        }
    }

    fn metadata_filter_to_filter(value: &serde_json::Value) -> Option<Filter> {
        let obj = value.as_object()?;
        let filters: Vec<Filter> = obj
            .iter()
            .map(|(key, value)| Filter::Equal {
                field: key.clone(),
                value: value.clone(),
            })
            .collect();

        match filters.len() {
            0 => None,
            1 => filters.into_iter().next(),
            _ => filters.into_iter().reduce(|acc, filter| Filter::And {
                left: Box::new(acc),
                right: Box::new(filter),
            }),
        }
    }

    fn effective_search_mode(
        &self,
        requested_mode: RagSearchMode,
        warnings: &mut Vec<String>,
    ) -> RagSearchMode {
        let capabilities = self.dripstone.backend().capabilities();
        match requested_mode {
            RagSearchMode::Dense | RagSearchMode::HybridDenseSparse => requested_mode,
            RagSearchMode::FullText => {
                if capabilities.native_full_text {
                    RagSearchMode::FullText
                } else {
                    warnings.push(
                        "Requested full-text search, but the current backend does not support Dripstone native full-text search; falling back to dense retrieval.".to_string(),
                    );
                    RagSearchMode::Dense
                }
            }
            RagSearchMode::HybridDenseFullText => {
                if capabilities.native_full_text {
                    RagSearchMode::HybridDenseFullText
                } else {
                    warnings.push(
                        "Requested dense + full-text hybrid search, but the current backend does not support Dripstone native full-text search; falling back to dense + sparse hybrid search.".to_string(),
                    );
                    RagSearchMode::HybridDenseSparse
                }
            }
        }
    }

    // Keep retrieval orchestration on Dripstone's SearchBuilder for now.
    // SearchPipeline is a good fit for backend-local ANN/post-processing stages, but this
    // service also needs builder-owned mode selection (dense, sparse hybrid, full-text,
    // dense+full-text), capability fallbacks, and reranker wiring in a single place.
    fn configure_search_builder<'a>(
        &'a self,
        params: SearchBuilderParams<'a>,
    ) -> RagSearchBuilder<'a> {
        let effective_mode = self.effective_search_mode(params.search_mode, params.warnings);
        let resolved_dense_weight =
            Self::resolve_dense_weight(params.dense_weight, params.sparse_weight);

        let mut search_builder = self
            .dripstone
            .search(params.query)
            .limit(if params.rerank {
                params.top_k * 3
            } else {
                params.top_k
            })
            .namespace(&self.namespace)
            .rerank(params.rerank);

        if let Some(filter_dsl) = params.filter {
            search_builder = search_builder.filter(filter_dsl);
        }

        match effective_mode {
            RagSearchMode::Dense => search_builder,
            RagSearchMode::HybridDenseSparse => {
                if params.fusion_mode == RagFusionMode::ReciprocalRank {
                    params.warnings.push(
                        "Reciprocal-rank fusion is only available for dense + full-text hybrid search; using weighted dense + sparse hybrid retrieval instead.".to_string(),
                    );
                }
                search_builder.hybrid(resolved_dense_weight)
            }
            RagSearchMode::FullText => {
                if params.fusion_mode == RagFusionMode::ReciprocalRank {
                    params.warnings.push(
                        "Fusion mode is ignored for full-text-only search; using Dripstone full-text retrieval.".to_string(),
                    );
                }
                search_builder.full_text().query_expansion(true)
            }
            RagSearchMode::HybridDenseFullText => {
                let builder = match params.fusion_mode {
                    RagFusionMode::Weighted => {
                        search_builder.hybrid_full_text(resolved_dense_weight)
                    }
                    RagFusionMode::ReciprocalRank => search_builder.hybrid_full_text_rrf(60.0),
                };
                builder.query_expansion(true)
            }
        }
    }

    fn wrap_backend_with_full_text(backend: Arc<dyn Backend>) -> Arc<dyn Backend> {
        if backend.capabilities().native_full_text {
            backend
        } else {
            Arc::new(FullTextIndexedBackend::with_in_memory_index(backend))
        }
    }

    fn wrap_backend(backend: Arc<dyn Backend>, late_interaction: bool) -> Arc<dyn Backend> {
        let backend = if late_interaction {
            Arc::new(LateInteractionIndexedBackend::with_default_index(backend))
        } else {
            backend
        };
        Self::wrap_backend_with_full_text(backend)
    }

    async fn embed_late_interaction_query(
        &self,
        query: &str,
    ) -> anyhow::Result<Option<dripstone::MultiVectors>> {
        let max_tokens = self.dripstone.config().search.max_token_vectors.max(1);
        let tokens = dripstone::tokens_for_late_interaction(query, max_tokens);
        if tokens.is_empty() {
            return Ok(None);
        }
        let refs: Vec<&str> = tokens.iter().map(std::string::String::as_str).collect();
        let vectors = self
            .dripstone
            .embedder()
            .embed(&refs)
            .await
            .map_err(|e| anyhow::anyhow!("late-interaction query embed failed: {e}"))?;
        if vectors.len() != tokens.len() {
            return Err(anyhow::anyhow!(
                "late-interaction query embed count mismatch"
            ));
        }
        Ok(Some(dripstone::multi_vectors_from_owned(vectors)))
    }

    async fn prune_stale_incremental_chunks(&self, doc_id: &str) -> anyhow::Result<()> {
        let Some(store) = self.chunk_hash_store.as_ref() else {
            return Ok(());
        };
        let hashes = store
            .get_doc_chunk_hashes(doc_id, Some(self.namespace.as_str()))
            .await
            .map_err(|e| {
                anyhow::anyhow!("Failed to load incremental chunk hashes for '{doc_id}': {e}")
            })?;
        let live_count = hashes
            .keys()
            .max()
            .map_or(0, |max_idx| max_idx.saturating_add(1));
        let mut idx = live_count;
        loop {
            let chunk_id = format!("{doc_id}#{idx}");
            match self
                .dripstone
                .backend()
                .get(&chunk_id, Some(self.namespace.as_str()))
                .await
            {
                Ok(Some(_)) => {
                    self.dripstone
                        .delete(&chunk_id, Some(self.namespace.as_str()))
                        .await
                        .map_err(|e| {
                            anyhow::anyhow!("Failed to prune stale chunk '{chunk_id}': {e}")
                        })?;
                    idx = idx.saturating_add(1);
                }
                Ok(None) => break,
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to inspect chunk '{chunk_id}' during incremental prune: {e}"
                    ));
                }
            }
        }
        Ok(())
    }

    fn chunk_metadata(path: Option<&std::path::Path>) -> serde_json::Value {
        let mut meta = serde_json::Map::new();
        let Some(path) = path else {
            return serde_json::Value::Object(meta);
        };
        meta.insert(
            "path".to_owned(),
            serde_json::Value::String(path.to_string_lossy().into_owned()),
        );
        if let Some(name) = path.file_name() {
            meta.insert(
                "file_name".to_owned(),
                serde_json::Value::String(name.to_string_lossy().into_owned()),
            );
        }
        if let Ok(modified) = std::fs::metadata(path).and_then(|info| info.modified()) {
            meta.insert(
                "created_at".to_owned(),
                serde_json::Value::String(
                    chrono::DateTime::<chrono::Utc>::from(modified).to_rfc3339(),
                ),
            );
        }
        serde_json::Value::Object(meta)
    }

    fn path_date_pattern() -> Option<regex::Regex> {
        static PATTERN: std::sync::OnceLock<Option<regex::Regex>> = std::sync::OnceLock::new();
        PATTERN
            .get_or_init(|| regex::Regex::new(r"(\d{4}-\d{2}-\d{2})").ok())
            .clone()
    }

    async fn ingest_document(
        &self,
        document: &str,
        doc_id: &str,
        path: Option<&std::path::Path>,
    ) -> anyhow::Result<dripstone::IngestResult> {
        let metadata = Self::chunk_metadata(path);

        #[cfg(feature = "ast")]
        if matches!(self.chunking_strategy, ChunkStrategy::AstAware)
            && let Some(path_ref) = path
        {
            let boundaries = crate::ast::chunk_at_ast_boundaries(document, path_ref);
            if !boundaries.is_empty() {
                let pre_chunks: Vec<dripstone::PreChunk> = boundaries
                    .into_iter()
                    .map(|b| dripstone::PreChunk::new(b.text, b.start_offset, b.end_offset))
                    .collect();
                return self
                    .dripstone
                    .ingest_chunks(doc_id, document, pre_chunks)
                    .namespace(&self.namespace)
                    .metadata(metadata)
                    .await
                    .map_err(|e| anyhow::anyhow!("Dripstone ingest_chunks failed: {e}"));
            }
        }

        self.dripstone
            .ingest(doc_id, document)
            .namespace(&self.namespace)
            .metadata(metadata)
            .await
            .map_err(|e| anyhow::anyhow!("Dripstone ingest failed: {e}"))
    }

    /// Create a new RAG service with the given embedding provider using the default in-memory backend.
    ///
    /// The in-memory backend is fast but ephemeral - data is lost when the process ends.
    pub fn new(embedding_provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self::with_metrics(embedding_provider, false)
    }

    /// Create a new RAG service with metrics enabled
    pub fn with_metrics(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        metrics_enabled: bool,
    ) -> Self {
        use dripstone::backend::memory::MemoryBackend;

        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        Self::new_with_backend(embedding_provider, backend, metrics_enabled, false, false)
    }

    /// Create a new RAG service with hashline tagging for caliche edit compatibility.
    ///
    /// When `hashline_enabled` is true, ingested chunks get content-addressed line hashes.
    /// When `hashline_format_results` is true, search results include `line:hash| content` tags.
    pub fn with_metrics_and_hashline(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        metrics_enabled: bool,
        hashline_enabled: bool,
        hashline_format_results: bool,
    ) -> Self {
        use dripstone::backend::memory::MemoryBackend;

        let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
        Self::new_with_backend(
            embedding_provider,
            backend,
            metrics_enabled,
            hashline_enabled,
            hashline_format_results,
        )
    }

    /// Create a new RAG service with a caller-provided vector backend.
    ///
    /// This enables persistent vector stores (Qdrant, Postgres/pgvector, Redis, etc.)
    /// when paired with a backend from `VectorBackendConfig::build_backend()`.
    ///
    /// When `hashline_enabled` is true, ingested chunks get content-addressed line hashes.
    /// When `hashline_format_results` is true, search results include `line:hash| content` tags.
    pub fn new_with_backend(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        backend: Arc<dyn Backend>,
        metrics_enabled: bool,
        hashline_enabled: bool,
        hashline_format_results: bool,
    ) -> Self {
        let options = RagHashlineOptions::for_edit_tools(hashline_enabled, hashline_format_results);
        Self::new_with_backend_and_options(embedding_provider, backend, metrics_enabled, options)
    }

    /// Create a new RAG service with a caller-provided backend and hashline options.
    ///
    /// Use [`RagHashlineOptions::for_workspace`] for workspace indexing,
    /// [`RagHashlineOptions::for_edit_tools`] for edit-tool compatibility only.
    pub fn new_with_backend_and_options(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        backend: Arc<dyn Backend>,
        metrics_enabled: bool,
        options: RagHashlineOptions,
    ) -> Self {
        let metrics: SharedMetrics = if metrics_enabled {
            Arc::new(SimpleMetrics::new())
        } else {
            Arc::new(NoOpMetrics)
        };

        let embedder = EmbeddingProviderWrapper::new(
            Arc::clone(&embedding_provider),
            1536,
            "embedding-provider",
        );

        let backend = Self::wrap_backend(backend, options.late_interaction);

        let mut config = DripstoneConfig::default();
        config.search.hybrid_enabled = true;
        // Reranking requires an ONNX model path (`search.rerank_model_path`). Keep it
        // disabled by default so basic RAG query works out of the box.
        config.search.rerank_enabled = false;
        config.search.late_interaction = options.late_interaction;
        config.hashline.enabled = options.enabled;
        config.hashline.line_level = options.enabled;
        config.hashline.format_results = options.format_results;
        config.hashline.bm25_dedup = options.bm25_dedup;
        config.hashline.line_level_bm25 = options.line_level_bm25;
        // Unchanged chunks skip re-embedding. `index_document` must not delete first
        // when this is on, or skipped chunks would vanish from the backend.
        config.hashline.incremental_indexing = options.enabled;
        config.chunking.strategy = options.chunking_strategy;

        let chunk_hash_store = options
            .enabled
            .then(|| Arc::new(MemoryChunkHashStore::new()));
        let hash_store_trait = chunk_hash_store
            .as_ref()
            .map(|store| Arc::clone(store) as Arc<dyn ChunkHashStore>);

        let dripstone = Arc::new(Dripstone::with_config_and_hash_store(
            backend,
            embedder,
            config,
            hash_store_trait,
        ));

        Self {
            embedding_provider,
            dripstone,
            metrics,
            chunking_strategy: options.chunking_strategy,
            graph_backend: None,
            queryable_graph_backend: None,
            llm_provider: None,
            namespace: "rag".to_owned(),
            incremental_indexing: options.enabled,
            chunk_hash_store,
            late_interaction: options.late_interaction,
        }
    }

    /// Set the namespace for document isolation (e.g. per-session). Default is "rag".
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// Create a new RAG service with graph backend support for Graph RAG
    ///
    /// This enables entity extraction and graph-enhanced search capabilities.
    /// The graph backend stores entities and relationships extracted from documents,
    /// and the optional LLM provider is used for entity extraction.
    pub fn with_graph_backend(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        graph_backend: Arc<dyn GraphBackend>,
        llm_provider: Option<Arc<dyn crate::llm::Provider>>,
    ) -> Self {
        let mut service = Self::new(embedding_provider);
        service.graph_backend = Some(graph_backend);
        service.llm_provider = llm_provider;
        service
    }

    /// Create a new RAG service with queryable graph backend for hybrid vector+graph retrieval.
    /// Stores the backend as both graph_backend and queryable_graph_backend.
    pub fn with_queryable_graph_backend(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        graph_backend: Arc<dyn QueryableGraphBackend>,
        llm_provider: Option<Arc<dyn crate::llm::Provider>>,
    ) -> Self {
        let mut service = Self::new(embedding_provider);
        service.graph_backend = Some(Arc::clone(&graph_backend) as Arc<dyn GraphBackend>);
        service.queryable_graph_backend = Some(graph_backend);
        service.llm_provider = llm_provider;
        service
    }

    /// Run hybrid vector+graph search. Requires queryable_graph_backend to be configured.
    pub async fn hybrid_search(
        &self,
        query: crate::rag::HybridQuery,
    ) -> anyhow::Result<Vec<crate::rag::HybridResult>> {
        let graph = self
            .queryable_graph_backend
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Queryable graph backend not configured"))?;
        let hybrid = crate::rag::HybridGraphVectorSearch::new(
            Arc::clone(graph),
            Arc::clone(&self.embedding_provider),
        );
        hybrid.search(query).await
    }

    /// Index a document by splitting it into chunks and embedding each chunk
    /// Returns (chunk_count, token_usage)
    pub async fn index_document(
        &self,
        document: &str,
        doc_id: &str,
    ) -> anyhow::Result<(usize, Option<TokenUsage>)> {
        self.index_document_with_path(document, doc_id, None::<&std::path::Path>)
            .await
    }

    /// Index a document, using AST-aware chunking when path is provided and strategy is AstAware.
    pub async fn index_document_with_path(
        &self,
        document: &str,
        doc_id: &str,
        path: Option<impl AsRef<std::path::Path>>,
    ) -> anyhow::Result<(usize, Option<TokenUsage>)> {
        let start = Instant::now();

        if document.is_empty() {
            self.delete_document(doc_id).await?;
            return Ok((0, None));
        }

        if !self.incremental_indexing {
            self.delete_document(doc_id).await?;
        }

        let path = path.as_ref().map(AsRef::as_ref);
        let result = self.ingest_document(document, doc_id, path).await?;

        let duration_micros = start.elapsed().as_micros() as u64;

        self.metrics
            .record_upsert_latency(std::time::Duration::from_micros(duration_micros));
        self.metrics.record_index_size(result.chunks_created);

        let token_usage = None;

        if self.incremental_indexing {
            self.prune_stale_incremental_chunks(doc_id).await?;
        }

        if let Some(ref graph_backend) = self.graph_backend {
            use crate::rag::graph::GraphIndexer;
            let graph_indexer = GraphIndexer::new(
                self.embedding_provider.clone(),
                graph_backend.clone(),
                self.llm_provider.clone(),
            );
            let _ = graph_indexer.index_chunk(document, doc_id).await;
        }

        Ok((result.chunks_created, token_usage))
    }

    /// Batch index multiple documents
    /// Returns (total_chunks, total_documents, failures, token_usage)
    pub async fn index_batch(
        &self,
        documents: Vec<(String, String)>, // (doc_id, document)
    ) -> anyhow::Result<(usize, usize, Vec<String>, Option<TokenUsage>)> {
        let with_paths: Vec<(String, String, Option<std::path::PathBuf>)> = documents
            .into_iter()
            .map(|(id, doc)| (id, doc, None))
            .collect();
        self.index_batch_with_paths(with_paths).await
    }

    /// Batch index documents with optional paths for AST-aware chunking.
    /// When path is Some and strategy is AstAware, uses structure-based chunking.
    ///
    /// Documents are embedded concurrently in small windows so ingest yields back
    /// to the interactive runtime. Override with `CALICHE_INDEX_CONCURRENCY`
    /// (adaptive default, max 32).
    pub async fn index_batch_with_paths(
        &self,
        mut documents: Vec<(String, String, Option<std::path::PathBuf>)>, // (doc_id, document, path)
    ) -> anyhow::Result<(usize, usize, Vec<String>, Option<TokenUsage>)> {
        if documents.is_empty() {
            return Ok((0, 0, Vec::new(), None));
        }

        let concurrency = Self::index_concurrency();
        let window = Self::index_ingest_window_size(concurrency);
        let mut total_chunks = 0;
        let mut total_docs = 0;
        let mut total_usage: Option<TokenUsage> = None;
        let mut failures = Vec::new();

        while !documents.is_empty() {
            self.fail_if_embedding_stopped()?;
            let drain_end = window.min(documents.len());
            let batch: Vec<(String, String, Option<std::path::PathBuf>)> =
                documents.drain(..drain_end).collect();
            let (chunks, docs, batch_failures, usage) =
                self.index_batch_window(batch, concurrency).await?;
            self.fail_if_embedding_stopped()?;
            total_chunks += chunks;
            total_docs += docs;
            failures.extend(batch_failures);
            if let Some(u) = usage {
                total_usage = Some(total_usage.take().map_or(u, |acc| acc + u));
            }
            if !documents.is_empty() {
                tokio::task::yield_now().await;
            }
        }

        Ok((total_chunks, total_docs, failures, total_usage))
    }

    /// Stop a batch once the embedding provider stopped for the session, so
    /// the rest of the documents are not tried one by one.
    fn fail_if_embedding_stopped(&self) -> anyhow::Result<()> {
        match self.embedding_provider.stopped_reason() {
            Some(reason) => Err(crate::llm::EmbeddingStopped { reason }.into()),
            None => Ok(()),
        }
    }

    async fn index_batch_window(
        &self,
        documents: Vec<(String, String, Option<std::path::PathBuf>)>,
        concurrency: usize,
    ) -> anyhow::Result<(usize, usize, Vec<String>, Option<TokenUsage>)> {
        let input_count = documents.len();
        let outcomes: Vec<Result<(usize, Option<TokenUsage>), String>> = stream::iter(documents)
            .map(|(doc_id, document, path)| async move {
                match self
                    .index_document_with_path(&document, &doc_id, path.as_deref())
                    .await
                {
                    Ok((count, usage)) => Ok((count, usage)),
                    Err(e) => Err(format!("{doc_id}: {e}")),
                }
            })
            .buffer_unordered(concurrency)
            .collect()
            .await;

        let mut total_chunks = 0;
        let mut total_usage: Option<TokenUsage> = None;
        let mut failures = Vec::new();
        for outcome in outcomes {
            match outcome {
                Ok((count, usage)) => {
                    total_chunks += count;
                    if let Some(ref u) = usage {
                        total_usage = Some(total_usage.take().map_or(*u, |acc| acc + *u));
                    }
                }
                Err(failure) => failures.push(failure),
            }
        }

        Ok((
            total_chunks,
            input_count.saturating_sub(failures.len()),
            failures,
            total_usage,
        ))
    }

    /// Query the indexed documents for relevant chunks (basic search).
    /// Returns (results, top_similarity_for_debug, token_usage).
    /// top_similarity_for_debug is the highest similarity score even if below threshold.
    /// Safe to call concurrently; multiple parallel RAG requests are supported.
    pub async fn query(
        &self,
        query: &str,
        top_k: usize,
        min_similarity: f32,
    ) -> anyhow::Result<(Vec<String>, Option<f32>, Option<TokenUsage>)> {
        // Use Dripstone's search API
        let start = Instant::now();
        let mut perf = crate::observability::perf::span("rag.query");
        let search_results = self
            .dripstone
            .search(query)
            .limit(top_k)
            .namespace(&self.namespace)
            .await
            .map_err(|e| anyhow::anyhow!("Dripstone search failed: {}", e))?;
        let duration_micros = start.elapsed().as_micros() as u64;
        if let Some(ref mut span) = perf {
            span.set_match_count(search_results.len() as u64);
        }
        drop(perf);

        // Record metrics
        self.metrics
            .record_search_latency(std::time::Duration::from_micros(duration_micros));
        self.metrics
            .record_search_result_count(search_results.len());

        let top_similarity = search_results.first().map(|r| r.score);
        let results = Self::format_search_hits(search_results, min_similarity);

        // Dripstone doesn't track token usage, return None
        let token_usage = None;

        Ok((results, top_similarity, token_usage))
    }

    /// Query used for automatic workspace context injection.
    ///
    /// Uses Dripstone dense + full-text hybrid retrieval (weighted fusion so scores
    /// stay compatible with [`WorkspaceContextQuery::min_similarity`]), plus MMR /
    /// temporal decay when those CLI flags are set.
    pub async fn query_context(
        &self,
        query: &str,
        opts: &crate::rag::WorkspaceContextQuery,
    ) -> anyhow::Result<(Vec<String>, Option<f32>, Option<TokenUsage>)> {
        let (use_mmr, mmr_lambda, use_temporal_decay, temporal_decay_half_life_days) = opts
            .search_options
            .as_ref()
            .map_or((None, None, None, None), |options| {
                (
                    options.use_mmr,
                    options.mmr_lambda,
                    options.use_temporal_decay,
                    options.temporal_decay_half_life_days,
                )
            });

        let (results, top_similarity, token_usage, warnings, _stats) = self
            .enhanced_query(
                query,
                opts.top_k,
                opts.min_similarity,
                opts.search_mode(),
                opts.fusion_mode(),
                None,
                None,
                false,
                None,
                None,
                None,
                None,
                None,
                use_mmr,
                mmr_lambda,
                use_temporal_decay,
                temporal_decay_half_life_days,
            )
            .await?;
        for warning in warnings {
            tracing::debug!("Workspace RAG enhanced query: {warning}");
        }
        let formatted = results
            .into_iter()
            .map(|result| format!("[Similarity: {:.3}] {}", result.score, result.chunk.content))
            .collect();
        Ok((formatted, top_similarity, token_usage))
    }

    /// Enhanced query with Dripstone-backed dense, hybrid, or full-text retrieval,
    /// plus reranking, metadata filtering, MMR, and temporal decay.
    #[allow(clippy::too_many_arguments)]
    pub async fn enhanced_query(
        &self,
        query: &str,
        top_k: usize,
        min_similarity: f32,
        search_mode: RagSearchMode,
        fusion_mode: RagFusionMode,
        dense_weight: Option<f32>,
        sparse_weight: Option<f32>,
        rerank: bool,
        metadata_filter: Option<&serde_json::Value>,
        filter: Option<Filter>,
        max_context_tokens: Option<usize>,
        provider: Option<ProviderType>,
        model: Option<&str>,
        use_mmr: Option<bool>,
        mmr_lambda: Option<f64>,
        use_temporal_decay: Option<bool>,
        temporal_decay_half_life_days: Option<f64>,
    ) -> anyhow::Result<(
        Vec<SearchResult>,
        Option<f32>,
        Option<TokenUsage>,
        Vec<String>,          // warnings
        Option<ContextStats>, // context stats
    )> {
        let mut warnings = Vec::new();

        let late_interaction_tokens = if self.late_interaction {
            match self.embed_late_interaction_query(query).await {
                Ok(tokens) => tokens,
                Err(error) => {
                    warnings.push(format!("Late-interaction MaxSim skipped: {error}"));
                    None
                }
            }
        } else {
            None
        };

        let explicit_filter =
            filter.or_else(|| metadata_filter.and_then(Self::metadata_filter_to_filter));
        let mut search_builder = self.configure_search_builder(SearchBuilderParams {
            query,
            top_k,
            search_mode,
            fusion_mode,
            dense_weight,
            sparse_weight,
            rerank,
            filter: explicit_filter,
            warnings: &mut warnings,
        });

        if let Some(multi_vectors) = late_interaction_tokens {
            search_builder = search_builder
                .with_multi_vectors(multi_vectors)
                .with_maxsim(
                    dripstone::MaxSimOptions::new(dripstone::Metric::DotProduct).overfetch(5),
                );
        }

        // MMR diversity reranking (dripstone search-advanced)
        if use_mmr == Some(true) {
            let lambda = mmr_lambda.unwrap_or(0.7).clamp(0.0, 1.0);
            let mmr_config = dripstone::search::MmrConfig {
                enabled: true,
                lambda,
                similarity: dripstone::search::MmrSimilarity::Cosine,
            };
            search_builder = search_builder.mmr(mmr_config);
        }

        // Temporal decay (dripstone search-advanced)
        if use_temporal_decay == Some(true) {
            let half_life = temporal_decay_half_life_days.unwrap_or(30.0);
            let decay_config = dripstone::search::TemporalDecayConfig {
                enabled: true,
                half_life_days: half_life,
                timestamp_key: "created_at".to_owned(),
                path_date_pattern: Self::path_date_pattern(),
                evergreen_paths: vec!["MEMORY.md".into()],
            };
            search_builder = search_builder.temporal_decay(decay_config);
        }

        let start = Instant::now();
        let mut perf = crate::observability::perf::span("rag.enhanced_query");
        let search_results = search_builder
            .await
            .map_err(|e| anyhow::anyhow!("Dripstone search failed: {}", e))?;
        let duration_micros = start.elapsed().as_micros() as u64;
        if let Some(ref mut span) = perf {
            span.set_match_count(search_results.len() as u64);
        }
        drop(perf);

        // Record metrics
        self.metrics
            .record_search_latency(std::time::Duration::from_micros(duration_micros));
        self.metrics
            .record_search_result_count(search_results.len());

        // Filter by minimum similarity
        let mut results: Vec<SearchResult> = search_results
            .into_iter()
            .filter(|r| r.score >= min_similarity)
            .collect();

        // Dripstone doesn't track token usage
        let token_usage = None;

        // Limit to top_k before context management
        results.truncate(top_k * 2); // Get more candidates for context fitting

        // Apply context window management if requested
        let context_stats = if let Some(max_tokens) = max_context_tokens {
            let manager = if let Some(prov) = provider {
                ContextManager::from_provider(prov, model)
            } else {
                ContextManager::new(max_tokens, max_tokens / 10) // Reserve 10% for completion
            };

            // Estimate system prompt and query tokens
            let system_prompt_tokens = 200; // Typical system prompt
            let query_tokens = query.len() / 4; // Rough estimate

            let (fitted_chunks, stats) =
                manager.fit_search_results(results, system_prompt_tokens, query_tokens);

            if stats.truncated {
                warnings.push(format!(
                    "Context window limit reached: included {} of {} retrieved chunks ({} omitted)",
                    stats.included_chunks, stats.total_chunks, stats.omitted_chunks
                ));
            }

            results = fitted_chunks;
            Some(stats)
        } else {
            // Still limit to top_k if no context management
            results.truncate(top_k);
            None
        };

        // Final limit to top_k
        results.truncate(top_k);

        // Get top similarity for debug
        let top_similarity = results.first().map(|r| r.score);

        Ok((
            results,
            top_similarity,
            token_usage,
            warnings,
            context_stats,
        ))
    }

    /// Perform graph-enhanced search
    ///
    /// Combines vector search with graph traversal to find related entities
    /// and build richer context for RAG queries.
    pub async fn graph_search(
        &self,
        query: &str,
        top_k: usize,
        max_hops: usize,
    ) -> anyhow::Result<crate::rag::graph::GraphSearchResult> {
        let graph_backend = self
            .graph_backend
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Graph backend not configured"))?;

        use crate::rag::graph::GraphSearch;
        let graph_search = GraphSearch::new(
            Arc::clone(&self.embedding_provider),
            Arc::clone(graph_backend),
        );

        graph_search.search(query, top_k, max_hops).await
    }

    /// Get cache statistics
    pub async fn stats(&self) -> anyhow::Result<(usize, usize)> {
        // Count chunks via Dripstone backend
        let count = self
            .dripstone
            .backend()
            .count(Some(self.namespace.as_str()))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to count: {}", e))?;

        Ok((count, count))
    }

    /// Clear all indexed documents
    pub async fn clear(&self) -> anyhow::Result<()> {
        // Delete all chunks in the rag namespace
        self.dripstone
            .backend()
            .delete_namespace(self.namespace.as_str())
            .await
            .map_err(|e| anyhow::anyhow!("Failed to clear: {}", e))?;
        if let Some(store) = self.chunk_hash_store.as_ref() {
            store.clear();
        }
        Ok(())
    }

    /// Delete all chunks for one source document in this RAG namespace.
    pub async fn delete_document(&self, doc_id: &str) -> anyhow::Result<usize> {
        self.dripstone
            .delete_document(doc_id, Some(self.namespace.as_str()))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to delete document '{doc_id}': {e}"))
    }

    /// Get a reference to the embedding provider
    pub fn embedding_provider(&self) -> &Arc<dyn EmbeddingProvider> {
        &self.embedding_provider
    }

    /// Get a reference to the Dripstone instance
    pub fn dripstone(&self) -> &Arc<Dripstone<Arc<dyn Backend>, EmbeddingProviderWrapper>> {
        &self.dripstone
    }

    /// Whether this service indexes token multi-vectors and searches with MaxSim / PLAID.
    #[must_use]
    pub const fn late_interaction_enabled(&self) -> bool {
        self.late_interaction
    }

    /// Get the text for a chunk ID (for formatting results)
    ///
    /// This retrieves the chunk content from the vector store by ID.
    pub async fn get_chunk_text(&self, chunk_id: &str) -> Option<String> {
        // Try to get the chunk from the backend
        match self
            .dripstone
            .backend()
            .get(chunk_id, Some(self.namespace.as_str()))
            .await
        {
            Ok(Some(chunk)) => Some(chunk.content),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_INDEX_CONCURRENCY_CAP, MAX_INDEX_CONCURRENCY, RagService};

    #[test]
    fn default_index_concurrency_leaves_a_core_and_caps() {
        assert_eq!(RagService::default_index_concurrency(1), 1);
        assert_eq!(RagService::default_index_concurrency(2), 1);
        assert_eq!(RagService::default_index_concurrency(8), 4);
        assert_eq!(
            RagService::default_index_concurrency(32),
            DEFAULT_INDEX_CONCURRENCY_CAP
        );
    }

    #[test]
    fn parse_index_concurrency_honors_env_and_caps() {
        assert_eq!(RagService::parse_index_concurrency(None, 8), 4);
        assert_eq!(RagService::parse_index_concurrency(Some("0"), 8), 4);
        assert_eq!(RagService::parse_index_concurrency(Some("nope"), 2), 1);
        assert_eq!(RagService::parse_index_concurrency(Some("16"), 2), 16);
        assert_eq!(
            RagService::parse_index_concurrency(Some("99"), 8),
            MAX_INDEX_CONCURRENCY
        );
    }

    #[test]
    fn ingest_window_stays_small() {
        assert_eq!(RagService::index_ingest_window_size(1), 4);
        assert_eq!(RagService::index_ingest_window_size(4), 8);
        assert_eq!(RagService::index_ingest_window_size(32), 16);
    }
}
