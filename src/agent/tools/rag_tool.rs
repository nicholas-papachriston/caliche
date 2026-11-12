use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::data::storage::vector::Filter;
use crate::llm::ProviderType;
use crate::rag::{
    RagFusionMode, RagHashlineOptions, RagSearchMode, RagSearchOptions, RagService,
    filter_dto::filter_from_value,
};
use async_trait::async_trait;
use serde_json::json;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;

mod formatters;
mod types;
use formatters::{format_no_results, format_query_success};
pub use types::IndexDirectoryResult;
use types::MAX_TOP_K;

fn parse_search_mode(arguments: &serde_json::Value) -> RagSearchMode {
    match arguments["search_mode"].as_str() {
        Some("hybrid") => RagSearchMode::HybridDenseSparse,
        Some("full_text") => RagSearchMode::FullText,
        Some("dense") => RagSearchMode::Dense,
        Some("hybrid_full_text") | None => RagSearchMode::HybridDenseFullText,
        Some(_) => RagSearchMode::Dense,
    }
}

fn parse_fusion_mode(arguments: &serde_json::Value) -> RagFusionMode {
    match arguments["fusion_mode"].as_str() {
        Some("rrf") => RagFusionMode::ReciprocalRank,
        Some("weighted" | _) | None => RagFusionMode::Weighted,
    }
}

struct QueryArgs<'a> {
    top_k: usize,
    min_similarity: f32,
    search_mode: RagSearchMode,
    fusion_mode: RagFusionMode,
    dense_weight: Option<f32>,
    sparse_weight: Option<f32>,
    rerank: bool,
    metadata_filter: Option<&'a serde_json::Value>,
    max_context_tokens: Option<usize>,
    use_mmr: Option<bool>,
    mmr_lambda: Option<f64>,
    use_temporal_decay: Option<bool>,
    temporal_decay_half_life_days: Option<f64>,
}

impl<'a> QueryArgs<'a> {
    fn from_arguments(
        arguments: &'a serde_json::Value,
        defaults: Option<&RagSearchOptions>,
    ) -> Self {
        let metadata_filter = arguments.get("metadata_filter");
        Self {
            top_k: arguments["top_k"].as_u64().unwrap_or(5) as usize,
            min_similarity: arguments["min_similarity"].as_f64().unwrap_or(0.3) as f32,
            search_mode: parse_search_mode(arguments),
            fusion_mode: parse_fusion_mode(arguments),
            dense_weight: arguments["dense_weight"].as_f64().map(|w| w as f32),
            sparse_weight: arguments["sparse_weight"].as_f64().map(|w| w as f32),
            rerank: arguments["rerank"].as_bool().unwrap_or(false),
            metadata_filter,
            max_context_tokens: arguments["max_context_tokens"].as_u64().map(|v| v as usize),
            use_mmr: arguments["use_mmr"]
                .as_bool()
                .or_else(|| defaults.and_then(|d| d.use_mmr)),
            mmr_lambda: arguments["mmr_lambda"]
                .as_f64()
                .or_else(|| defaults.and_then(|d| d.mmr_lambda)),
            use_temporal_decay: arguments["use_temporal_decay"]
                .as_bool()
                .or_else(|| defaults.and_then(|d| d.use_temporal_decay)),
            temporal_decay_half_life_days: arguments["temporal_decay_half_life_days"]
                .as_f64()
                .or_else(|| defaults.and_then(|d| d.temporal_decay_half_life_days)),
        }
    }

    fn needs_enhanced(&self) -> bool {
        self.search_mode != RagSearchMode::Dense
            || self.fusion_mode != RagFusionMode::Weighted
            || self.rerank
            || self.metadata_filter.is_some()
            || self.max_context_tokens.is_some()
            || self.use_mmr == Some(true)
            || self.use_temporal_decay == Some(true)
    }
}

fn format_simple_no_results(
    query: &str,
    top_similarity: Option<f32>,
    min_similarity: f32,
    usage: Option<crate::messages::TokenUsage>,
) -> ToolResult {
    let debug_info = top_similarity
        .map(|top_sim| {
            format!(" (highest similarity was {top_sim:.3}, below threshold {min_similarity:.3})")
        })
        .unwrap_or_default();
    let mut result = json!({
        "status": "no_results",
        "message": format!("No relevant documents found for query: '{}'{}. Try lowering min_similarity or indexing more documents.", query, debug_info),
        "results_count": 0,
        "top_similarity": top_similarity
    });
    if let Some(ref u) = usage {
        result["embedding_tokens_used"] = json!(u.embedding_tokens);
    }
    match usage {
        Some(u) => ToolResult::success_with_usage(result.to_string(), u),
        None => ToolResult::success(result.to_string()),
    }
}

fn format_simple_success(
    query: &str,
    top_k: usize,
    min_similarity: f32,
    results: Vec<String>,
    usage: Option<crate::messages::TokenUsage>,
) -> ToolResult {
    let mut result = json!({
        "status": "success",
        "results": results,
        "results_count": results.len(),
        "query": query,
        "top_k": top_k,
        "min_similarity": min_similarity
    });
    if let Some(ref u) = usage {
        result["embedding_tokens_used"] = json!(u.embedding_tokens);
    }
    match usage {
        Some(u) => ToolResult::success_with_usage(result.to_string(), u),
        None => ToolResult::success(result.to_string()),
    }
}

/// Agent tool for indexing documents and retrieving context via [`RagService`].
///
/// Default construction uses an in-memory vector backend. Pass a custom backend with
/// [`RagTool::with_backend`] when you need Redis, Postgres/pgvector, Qdrant, or another
/// Dripstone store.
///
/// # Examples
///
/// ```rust,no_run
/// # use caliche::RagTool;
/// # use caliche::OpenAIProvider;
/// # use std::sync::Arc;
/// # async fn example() -> anyhow::Result<()> {
/// let embedding_provider = Arc::new(OpenAIProvider::try_new("key".to_string(), "model".to_string())?);
/// let rag_tool = RagTool::new(embedding_provider.clone(), None::<String>);
///
/// # #[cfg(feature = "vector-store-qdrant")]
/// # {
/// use dripstone::backend::qdrant::{QdrantBackendOptions, QdrantHttpBackend};
/// use dripstone::embedding::SimilarityMetric;
///
/// let qdrant_backend = QdrantHttpBackend::with_options(QdrantBackendOptions {
///     base_url: "http://localhost:6333".to_string(),
///     collection_prefix: Some("kb_".to_string()),
///     vector_dimension: 1536,
///     distance: SimilarityMetric::Cosine,
///     api_key: None,
///     timeout: None,
///     batch_size: Some(100),
///     retry_config: None,
/// }).await?;
///
/// let rag_tool = RagTool::with_backend(embedding_provider, Arc::new(qdrant_backend), None::<String>);
/// # }
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct RagTool {
    service: Arc<RagService>,
    /// Default search options applied when a tool call omits them.
    default_search_options: Option<RagSearchOptions>,
}

impl RagTool {
    /// Build a RAG tool with an in-memory vector backend.
    ///
    /// Data is process-local. Use [`RagTool::with_backend`] for a durable store.
    /// `namespace` isolates documents (default `"rag"`).
    pub fn new(
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
        namespace: Option<impl Into<String>>,
    ) -> Self {
        let mut service = RagService::new(embedding_provider);
        if let Some(ns) = namespace {
            service = service.with_namespace(ns);
        }
        Self {
            service: Arc::new(service),
            default_search_options: None,
        }
    }

    /// Create a new RAG tool with metrics enabled/disabled.
    pub fn with_metrics(
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
        metrics_enabled: bool,
    ) -> Self {
        let service = RagService::with_metrics(embedding_provider, metrics_enabled);
        Self {
            service: Arc::new(service),
            default_search_options: None,
        }
    }

    /// Create a new RAG tool with hashline tagging for `caliche` edit compatibility.
    ///
    /// When `hashline_enabled` is true, ingested chunks get content-addressed line hashes.
    /// When `hashline_format_results` is true, search results include `line:hash| content` tags
    /// usable with `hashline_replace`, `hashline_insert`, `hashline_delete` in fs op=edit.
    ///
    /// For `bm25_dedup` or `line_level_bm25`, use [`RagTool::new_with_hashline_options`].
    pub fn new_with_hashline(
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
        namespace: Option<impl Into<String>>,
        hashline_enabled: bool,
        hashline_format_results: bool,
    ) -> Self {
        Self::new_with_hashline_options(
            embedding_provider,
            namespace,
            RagHashlineOptions::for_edit_tools(hashline_enabled, hashline_format_results),
        )
    }

    /// Create a new RAG tool using a caller-provided `Dripstone` backend.
    ///
    /// Use this to plug in persistent vector stores built from `VectorBackendConfig::build_backend()`.
    pub fn with_backend(
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
        backend: Arc<dyn dripstone::Backend + Send + Sync>,
        namespace: Option<impl Into<String>>,
    ) -> Self {
        Self::with_backend_and_metrics(embedding_provider, backend, false, namespace)
    }

    /// Create a new RAG tool using a caller-provided `Dripstone` backend, with optional metrics.
    pub fn with_backend_and_metrics(
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
        backend: Arc<dyn dripstone::Backend + Send + Sync>,
        metrics_enabled: bool,
        namespace: Option<impl Into<String>>,
    ) -> Self {
        let mut service = RagService::new_with_backend(
            embedding_provider,
            backend,
            metrics_enabled,
            false,
            false,
        );
        if let Some(ns) = namespace {
            service = service.with_namespace(ns);
        }
        Self {
            service: Arc::new(service),
            default_search_options: None,
        }
    }

    /// Create a new RAG tool with backend and hashline options.
    ///
    /// Use [`RagHashlineOptions::for_workspace`] for workspace indexing, or build options
    /// explicitly. Enables `bm25_dedup`, `line_level_bm25`, and hashline tagging as configured.
    pub fn with_backend_and_hashline_options(
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
        backend: Arc<dyn dripstone::Backend + Send + Sync>,
        namespace: Option<impl Into<String>>,
        metrics_enabled: bool,
        options: RagHashlineOptions,
    ) -> Self {
        let mut service = RagService::new_with_backend_and_options(
            embedding_provider,
            backend,
            metrics_enabled,
            options,
        );
        if let Some(ns) = namespace {
            service = service.with_namespace(ns);
        }
        Self {
            service: Arc::new(service),
            default_search_options: None,
        }
    }

    /// Create a new RAG tool (in-memory) with hashline options.
    ///
    /// Use [`RagHashlineOptions::for_workspace`] for workspace indexing, or
    /// [`RagHashlineOptions::for_edit_tools`] for edit-tool compatibility only.
    pub fn new_with_hashline_options(
        embedding_provider: Arc<dyn crate::llm::EmbeddingProvider>,
        namespace: Option<impl Into<String>>,
        options: RagHashlineOptions,
    ) -> Self {
        use dripstone::backend::memory::MemoryBackend;

        let backend: Arc<dyn dripstone::Backend + Send + Sync> = Arc::new(MemoryBackend::new());
        Self::with_backend_and_hashline_options(
            embedding_provider,
            backend,
            namespace,
            false,
            options,
        )
    }

    /// Set default search options (MMR, temporal decay). When the agent calls the RAG tool
    /// without specifying these, these defaults are used. Used by the CLI and workspace config.
    #[must_use]
    pub const fn with_default_search_options(mut self, options: RagSearchOptions) -> Self {
        self.default_search_options = Some(options);
        self
    }

    /// Get a reference to the underlying RAG service.
    ///
    /// This allows sharing the same backend between `RagTool` and `RagService` instances,
    /// ensuring that content indexed via one can be queried via the other.
    #[must_use]
    pub fn service(&self) -> Arc<RagService> {
        Arc::clone(&self.service)
    }

    /// Index a document by splitting it into chunks and embedding each chunk
    /// Returns (`chunk_count`, `token_usage`)
    pub async fn index_document(
        &self,
        document: &str,
        doc_id: &str,
    ) -> anyhow::Result<(usize, Option<crate::messages::TokenUsage>)> {
        self.service.index_document(document, doc_id).await
    }

    /// Batch index multiple documents
    /// Returns (`total_chunks`, `total_documents`, failures, `token_usage`)
    pub async fn index_batch(
        &self,
        documents: Vec<(String, String)>, // (doc_id, document)
    ) -> anyhow::Result<(
        usize,
        usize,
        Vec<String>,
        Option<crate::messages::TokenUsage>,
    )> {
        self.service.index_batch(documents).await
    }

    /// Query the indexed documents for relevant chunks (basic search)
    /// Returns (results, `top_similarity_for_debug`, `token_usage`)
    /// `top_similarity_for_debug` is the highest similarity score even if below threshold
    pub async fn query(
        &self,
        query: &str,
        top_k: usize,
        min_similarity: f32,
    ) -> anyhow::Result<(
        Vec<String>,
        Option<f32>,
        Option<crate::messages::TokenUsage>,
    )> {
        self.service.query(query, top_k, min_similarity).await
    }

    /// Enhanced query with hybrid search, reranking, metadata filtering, MMR, and temporal decay
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
        Vec<dripstone::SearchResult>,
        Option<f32>,
        Option<crate::messages::TokenUsage>,
        Vec<String>,                                                // warnings
        Option<crate::agent::tools::context_manager::ContextStats>, // context stats
    )> {
        self.service
            .enhanced_query(
                query,
                top_k,
                min_similarity,
                search_mode,
                fusion_mode,
                dense_weight,
                sparse_weight,
                rerank,
                metadata_filter,
                filter,
                max_context_tokens,
                provider,
                model,
                use_mmr,
                mmr_lambda,
                use_temporal_decay,
                temporal_decay_half_life_days,
            )
            .await
    }

    /// Get cache statistics
    pub async fn stats(&self) -> anyhow::Result<(usize, usize)> {
        self.service.stats().await
    }

    /// Clear all indexed documents
    pub async fn clear(&self) -> anyhow::Result<()> {
        self.service.clear().await
    }

    /// Index a file by reading, chunking, and embedding its contents.
    /// Uses AST-aware chunking when configured and the file type is supported.
    /// Returns (`chunk_count`, `token_usage`)
    pub async fn index_file(
        &self,
        file_path: &str,
        doc_id: Option<&str>,
    ) -> anyhow::Result<(usize, Option<crate::messages::TokenUsage>)> {
        let content = read_file_for_indexing(file_path, 0).await?;
        let doc_id = doc_id.unwrap_or(file_path);
        let path = std::path::Path::new(file_path);
        self.service
            .index_document_with_path(&content, doc_id, Some(path))
            .await
    }

    /// Index all files in a directory
    /// Returns `IndexDirectoryResult` with statistics
    pub async fn index_directory(
        &self,
        dir_path: &str,
        recursive: bool,
        pattern: Option<&str>,
        max_files: Option<usize>,
    ) -> anyhow::Result<IndexDirectoryResult> {
        let files = collect_directory_files(dir_path, recursive, pattern, max_files).await?;

        if files.is_empty() {
            return Ok(IndexDirectoryResult {
                files_indexed: 0,
                total_chunks: 0,
                failures: vec!["No files found matching criteria".to_string()],
                token_usage: None,
            });
        }

        let mut documents = Vec::new();
        for file_path in &files {
            match read_file_for_indexing(file_path.to_str().unwrap_or(""), 0).await {
                Ok(content) => {
                    let doc_id = file_path.to_str().unwrap_or("");
                    documents.push((doc_id.to_string(), content));
                }
                Err(_e) => {
                    // Continue with other files even if one fails
                    continue;
                }
            }
        }

        if documents.is_empty() {
            return Ok(IndexDirectoryResult {
                files_indexed: 0,
                total_chunks: 0,
                failures: vec!["Failed to read any files".to_string()],
                token_usage: None,
            });
        }

        let (total_chunks, _total_docs, failures, token_usage) =
            self.index_batch(documents).await?;

        Ok(IndexDirectoryResult {
            files_indexed: files.len() - failures.len(),
            total_chunks,
            failures,
            token_usage,
        })
    }

    async fn execute_index(&self, arguments: &serde_json::Value) -> ToolResult {
        let document = match arguments["document"].as_str() {
            Some(doc) => doc,
            None => {
                return ToolResult::error(
                    "Missing 'document' parameter for index operation".to_string(),
                );
            }
        };
        let doc_id = match arguments["doc_id"].as_str() {
            Some(id) => id,
            None => {
                return ToolResult::error(
                    "Missing 'doc_id' parameter for index operation".to_string(),
                );
            }
        };
        match self.index_document(document, doc_id).await {
            Ok((count, usage)) => {
                let mut result = json!({
                    "status": "success",
                    "message": format!("Successfully indexed document '{}' with {} chunks", doc_id, count),
                    "chunks_indexed": count,
                    "doc_id": doc_id
                });
                if let Some(ref u) = usage {
                    result["embedding_tokens_used"] = json!(u.embedding_tokens);
                }
                match usage {
                    Some(u) => ToolResult::success_with_usage(result.to_string(), u),
                    None => ToolResult::success(result.to_string()),
                }
            }
            Err(e) => ToolResult::error(format!("Failed to index document: {e}")),
        }
    }

    async fn execute_index_batch(&self, arguments: &serde_json::Value) -> ToolResult {
        let documents = match arguments["documents"].as_array() {
            Some(arr) => {
                let mut docs = Vec::new();
                for item in arr {
                    if let (Some(doc_id), Some(document)) =
                        (item["doc_id"].as_str(), item["document"].as_str())
                    {
                        docs.push((doc_id.to_string(), document.to_string()));
                    } else {
                        return ToolResult::error(
                            "Each document in 'documents' array must have 'doc_id' and 'document' fields".to_string(),
                        );
                    }
                }
                docs
            }
            None => {
                return ToolResult::error(
                    "Missing 'documents' parameter for index_batch operation".to_string(),
                );
            }
        };
        match self.index_batch(documents).await {
            Ok((total_chunks, total_docs, failures, usage)) => {
                let mut result = json!({
                    "status": "success",
                    "message": format!("Successfully indexed {} documents with {} total chunks", total_docs, total_chunks),
                    "total_chunks": total_chunks,
                    "total_documents": total_docs,
                    "failures": failures.len()
                });
                if !failures.is_empty() {
                    result["failure_details"] = json!(failures);
                }
                if let Some(ref u) = usage {
                    result["embedding_tokens_used"] = json!(u.embedding_tokens);
                }
                match usage {
                    Some(u) => ToolResult::success_with_usage(result.to_string(), u),
                    None => ToolResult::success(result.to_string()),
                }
            }
            Err(e) => ToolResult::error(format!("Failed to index batch: {e}")),
        }
    }

    async fn execute_index_file(&self, arguments: &serde_json::Value) -> ToolResult {
        let path = match params::extract_string(arguments, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };
        let doc_id = arguments["doc_id"].as_str();
        match self.index_file(&path, doc_id).await {
            Ok((count, usage)) => {
                let doc_id_used = doc_id.unwrap_or(&path);
                let mut result = json!({
                    "status": "success",
                    "message": format!("Successfully indexed file '{}' with {} chunks", path, count),
                    "chunks_indexed": count,
                    "file_path": path,
                    "doc_id": doc_id_used
                });
                if let Some(ref u) = usage {
                    result["embedding_tokens_used"] = json!(u.embedding_tokens);
                }
                match usage {
                    Some(u) => ToolResult::success_with_usage(result.to_string(), u),
                    None => ToolResult::success(result.to_string()),
                }
            }
            Err(e) => ToolResult::error(format!("Failed to index file '{path}': {e}")),
        }
    }

    async fn execute_index_directory(&self, arguments: &serde_json::Value) -> ToolResult {
        let path = match params::extract_string(arguments, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };
        let recursive = params::extract_bool(arguments, "recursive", false);
        let pattern = arguments["pattern"].as_str();
        let max_files = arguments["max_files"].as_u64().map(|n| n as usize);
        match self
            .index_directory(&path, recursive, pattern, max_files)
            .await
        {
            Ok(result) => {
                let mut json_result = json!({
                    "status": "success",
                    "message": format!("Successfully indexed {} files from '{}' with {} total chunks", result.files_indexed, path, result.total_chunks),
                    "files_indexed": result.files_indexed,
                    "total_chunks": result.total_chunks,
                    "failures": result.failures.len()
                });
                if !result.failures.is_empty() {
                    json_result["failure_details"] = json!(result.failures);
                }
                if let Some(ref u) = result.token_usage {
                    json_result["embedding_tokens_used"] = json!(u.embedding_tokens);
                }
                match result.token_usage {
                    Some(u) => ToolResult::success_with_usage(json_result.to_string(), u),
                    None => ToolResult::success(json_result.to_string()),
                }
            }
            Err(e) => ToolResult::error(format!("Failed to index directory '{path}': {e}")),
        }
    }

    async fn execute_query(&self, arguments: &serde_json::Value) -> ToolResult {
        let query = match arguments["query"].as_str() {
            Some(q) => q,
            None => {
                return ToolResult::error(
                    "Missing 'query' parameter for query operation".to_string(),
                );
            }
        };
        let opts = QueryArgs::from_arguments(arguments, self.default_search_options.as_ref());
        if opts.needs_enhanced() {
            self.execute_enhanced_query(query, &opts).await
        } else {
            self.execute_simple_query(query, opts.top_k.min(MAX_TOP_K), opts.min_similarity)
                .await
        }
    }

    async fn execute_enhanced_query(&self, query: &str, opts: &QueryArgs<'_>) -> ToolResult {
        let top_k = opts.top_k.min(MAX_TOP_K);
        match self
            .enhanced_query(
                query,
                top_k,
                opts.min_similarity,
                opts.search_mode,
                opts.fusion_mode,
                opts.dense_weight,
                opts.sparse_weight,
                opts.rerank,
                opts.metadata_filter,
                opts.metadata_filter.and_then(|v| filter_from_value(v).ok()),
                opts.max_context_tokens,
                None,
                None,
                opts.use_mmr,
                opts.mmr_lambda,
                opts.use_temporal_decay,
                opts.temporal_decay_half_life_days,
            )
            .await
        {
            Ok((results, top_similarity, usage, warnings, context_stats)) => {
                let formatted_results: Vec<String> = results
                    .iter()
                    .map(|r| format!("[Similarity: {:.3}] {}", r.score, r.chunk.content))
                    .collect();
                if formatted_results.is_empty() {
                    format_no_results(
                        query,
                        top_similarity,
                        opts.min_similarity,
                        "Try lowering min_similarity or indexing more documents.",
                        usage,
                        warnings,
                        context_stats.as_ref(),
                        &json!({}),
                    )
                } else {
                    format_query_success(
                        formatted_results,
                        query,
                        top_k,
                        opts.min_similarity,
                        usage,
                        warnings,
                        context_stats.as_ref(),
                        &json!({
                            "search_mode": opts.search_mode.as_str(),
                            "fusion_mode": opts.fusion_mode.as_str(),
                            "rerank": opts.rerank
                        }),
                    )
                }
            }
            Err(e) => ToolResult::error(format!("Query failed: {e}")),
        }
    }

    async fn execute_simple_query(
        &self,
        query: &str,
        top_k: usize,
        min_similarity: f32,
    ) -> ToolResult {
        match self.query(query, top_k, min_similarity).await {
            Ok((results, top_similarity, usage)) => {
                if results.is_empty() {
                    format_simple_no_results(query, top_similarity, min_similarity, usage)
                } else {
                    format_simple_success(query, top_k, min_similarity, results, usage)
                }
            }
            Err(e) => ToolResult::error(format!("Query failed: {e}")),
        }
    }

    async fn execute_advanced_query(&self, arguments: &serde_json::Value) -> ToolResult {
        let query = match arguments["query"].as_str() {
            Some(q) => q,
            None => {
                return ToolResult::error(
                    "Missing 'query' parameter for advanced_query operation".to_string(),
                );
            }
        };
        let top_k = arguments["top_k"].as_u64().unwrap_or(5) as usize;
        let top_k = top_k.min(MAX_TOP_K);
        let min_similarity = arguments["min_similarity"].as_f64().unwrap_or(0.3) as f32;
        let search_mode = parse_search_mode(arguments);
        let fusion_mode = parse_fusion_mode(arguments);
        let dense_weight = arguments["dense_weight"].as_f64().map(|w| w as f32);
        let sparse_weight = arguments["sparse_weight"].as_f64().map(|w| w as f32);
        let rerank = arguments["rerank"].as_bool().unwrap_or(false);
        let metadata_filter = arguments.get("metadata_filter");
        let use_diversity = arguments["use_diversity"].as_bool().unwrap_or(false);
        let max_context_tokens = arguments["max_context_tokens"].as_u64().map(|v| v as usize);
        let filter: Option<Filter> = metadata_filter.and_then(|v| filter_from_value(v).ok());
        let defaults = self.default_search_options.as_ref();
        let use_mmr = arguments["use_mmr"]
            .as_bool()
            .or_else(|| defaults.and_then(|d| d.use_mmr));
        let mmr_lambda = arguments["mmr_lambda"]
            .as_f64()
            .or_else(|| defaults.and_then(|d| d.mmr_lambda));
        let use_temporal_decay = arguments["use_temporal_decay"]
            .as_bool()
            .or_else(|| defaults.and_then(|d| d.use_temporal_decay));
        let temporal_decay_half_life_days = arguments["temporal_decay_half_life_days"]
            .as_f64()
            .or_else(|| defaults.and_then(|d| d.temporal_decay_half_life_days));

        match self
            .enhanced_query(
                query,
                top_k,
                min_similarity,
                search_mode,
                fusion_mode,
                dense_weight,
                sparse_weight,
                rerank,
                metadata_filter,
                filter,
                max_context_tokens,
                None,
                None,
                use_mmr,
                mmr_lambda,
                use_temporal_decay,
                temporal_decay_half_life_days,
            )
            .await
        {
            Ok((results, top_similarity, usage, warnings, context_stats)) => {
                let formatted_results: Vec<String> = results
                    .iter()
                    .map(|r| format!("[Similarity: {:.3}] {}", r.score, r.chunk.content))
                    .collect();
                if formatted_results.is_empty() {
                    format_no_results(
                        query,
                        top_similarity,
                        min_similarity,
                        "Try lowering min_similarity or adjusting filters.",
                        usage,
                        warnings,
                        context_stats.as_ref(),
                        &json!({
                            "search_mode": search_mode.as_str(),
                            "fusion_mode": fusion_mode.as_str(),
                            "rerank": rerank,
                            "use_diversity": use_diversity
                        }),
                    )
                } else {
                    format_query_success(
                        formatted_results,
                        query,
                        top_k,
                        min_similarity,
                        usage,
                        warnings,
                        context_stats.as_ref(),
                        &json!({
                            "search_mode": search_mode.as_str(),
                            "fusion_mode": fusion_mode.as_str(),
                            "rerank": rerank,
                            "use_diversity": use_diversity
                        }),
                    )
                }
            }
            Err(e) => ToolResult::error(format!("Advanced query failed: {e}")),
        }
    }

    async fn execute_graph_search(&self, arguments: &serde_json::Value) -> ToolResult {
        let query = match arguments["query"].as_str() {
            Some(q) => q,
            None => {
                return ToolResult::error(
                    "Missing 'query' parameter for graph_search operation".to_string(),
                );
            }
        };
        let top_k = arguments["top_k"].as_u64().unwrap_or(5) as usize;
        let max_hops = arguments["max_hops"].as_u64().unwrap_or(1) as usize;
        match self.service.graph_search(query, top_k, max_hops).await {
            Ok(graph_result) => {
                let mut result = json!({
                    "status": "success",
                    "query": query,
                    "entry_nodes": graph_result.entry_nodes.len(),
                    "expanded_nodes": graph_result.expanded_nodes.len(),
                    "expanded_edges": graph_result.expanded_edges.len(),
                    "context": graph_result.context,
                    "message": format!(
                        "Found {} entry nodes, expanded to {} nodes with {} relationships",
                        graph_result.entry_nodes.len(),
                        graph_result.expanded_nodes.len(),
                        graph_result.expanded_edges.len()
                    )
                });
                let node_details: Vec<serde_json::Value> = graph_result
                    .expanded_nodes
                    .iter()
                    .take(10)
                    .map(|node| {
                        json!({
                            "id": node.id,
                            "type": node.node_type,
                            "name": node.properties.get("name").and_then(|v| v.as_str()).unwrap_or(&node.id)
                        })
                    })
                    .collect();
                result["sample_nodes"] = json!(node_details);
                ToolResult::success(result.to_string())
            }
            Err(e) => ToolResult::error(format!(
                "Graph search failed: {e}. Make sure graph backend is configured."
            )),
        }
    }

    async fn execute_hybrid_vector_graph_search(
        &self,
        arguments: &serde_json::Value,
    ) -> ToolResult {
        let query_text = match arguments["query"].as_str().filter(|s| !s.is_empty()) {
            Some(q) => q.to_string(),
            None => {
                return ToolResult::error(
                    "Missing required 'query' parameter for hybrid vector+graph search",
                );
            }
        };
        let anchor_node_id = arguments["anchor_node_id"].as_str().map(String::from);
        let max_hops = arguments["max_hops"].as_u64().unwrap_or(3) as usize;
        let top_k = arguments["top_k"].as_u64().unwrap_or(10) as usize;
        let strategy_str = arguments["strategy"].as_str().unwrap_or("auto");
        let strategy = match strategy_str {
            "intersection" => crate::rag::HybridStrategy::Intersection,
            "vector_then_graph" => crate::rag::HybridStrategy::VectorThenGraph,
            "graph_then_vector" => crate::rag::HybridStrategy::GraphThenVector,
            "union" => crate::rag::HybridStrategy::Union {
                vector_weight: arguments["vector_weight"].as_f64().unwrap_or(0.6) as f32,
                graph_weight: arguments["graph_weight"].as_f64().unwrap_or(0.4) as f32,
            },
            _ => crate::rag::HybridStrategy::Auto,
        };
        let hybrid_query = crate::rag::HybridQuery {
            text_query: query_text,
            anchor_node_id,
            max_hops,
            edge_type_filter: None,
            node_type_filter: None,
            vector_top_k: top_k,
            graph_top_k: top_k,
            strategy,
        };
        match self.service.hybrid_search(hybrid_query).await {
            Ok(results) => {
                let result_list: Vec<serde_json::Value> = results
                    .iter()
                    .map(|r| {
                        json!({
                            "node_id": r.node.id,
                            "node_type": r.node.node_type,
                            "vector_score": r.vector_score,
                            "graph_distance": r.graph_distance,
                            "combined_score": r.combined_score,
                            "provenance": format!("{:?}", r.provenance)
                        })
                    })
                    .collect();
                let result = json!({
                    "status": "success",
                    "result_count": results.len(),
                    "results": result_list
                });
                ToolResult::success(result.to_string())
            }
            Err(e) => ToolResult::error(format!(
                "Hybrid search failed: {e}. Ensure queryable graph backend is configured."
            )),
        }
    }

    async fn execute_stats(&self) -> ToolResult {
        match self.stats().await {
            Ok((embedding_count, text_count)) => {
                let result = json!({
                    "status": "success",
                    "embeddings_stored": embedding_count,
                    "texts_stored": text_count,
                    "message": format!("RAG cache contains {} embeddings and {} text chunks", embedding_count, text_count)
                });
                ToolResult::success(result.to_string())
            }
            Err(e) => ToolResult::error(format!("Failed to get stats: {e}")),
        }
    }

    async fn execute_clear(&self) -> ToolResult {
        match self.clear().await {
            Ok(()) => {
                let result = json!({
                    "status": "success",
                    "message": "Successfully cleared all indexed documents from RAG cache"
                });
                ToolResult::success(result.to_string())
            }
            Err(e) => ToolResult::error(format!("Failed to clear cache: {e}")),
        }
    }
}

/// Read file with size limit and return content
async fn read_file_for_indexing(path_str: &str, max_size: usize) -> anyhow::Result<String> {
    const DEFAULT_MAX_SIZE: usize = 10_485_760; // 10MB default

    let max_size = if max_size > 0 {
        max_size
    } else {
        DEFAULT_MAX_SIZE
    };

    // Validate and resolve path
    let normalized_path = path::validate_and_resolve_path(path_str, Some(true))
        .await
        .map_err(|e| anyhow::anyhow!("Path validation failed: {e}"))?;

    // Check file size
    path::validate_file_size(&normalized_path, max_size as u64)
        .map_err(|e| anyhow::anyhow!("File size validation failed: {e}"))?;

    let bytes = path::read_canonical_bytes(&normalized_path)
        .await
        .map_err(|e| anyhow::anyhow!("Failed to read file '{path_str}': {e}"))?;

    #[cfg(feature = "document-loaders")]
    {
        let processor = crate::data::loaders::DocumentProcessor::default();
        let file_name = normalized_path.file_name().and_then(|name| name.to_str());
        if let Ok(processed) = processor.process_bytes(&bytes, "", file_name).await
            && !processed.text.trim().is_empty()
        {
            return Ok(processed.text);
        }
    }

    // Fallback: plain UTF-8 text
    String::from_utf8(bytes).map_err(|e| anyhow::anyhow!("Failed to read file '{path_str}': {e}"))
}

/// Simple pattern matching for file names (supports basic glob patterns like *.rs)
fn matches_pattern(file_name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(stripped) = pattern.strip_prefix("*.") {
        // Pattern like "*.rs"
        return file_name.ends_with(&format!(".{stripped}"));
    }

    if pattern.starts_with('*') && pattern.ends_with('*') {
        // Pattern like "*test*"
        let inner = &pattern[1..pattern.len() - 1];
        return file_name.contains(inner);
    }

    if let Some(suffix) = pattern.strip_prefix("*") {
        // Pattern like "*test"
        return file_name.ends_with(suffix);
    }

    if let Some(prefix) = pattern.strip_suffix("*") {
        // Pattern like "test*"
        return file_name.starts_with(prefix);
    }

    // Exact match
    file_name == pattern
}

/// Walk directory and collect file paths matching criteria
async fn collect_directory_files(
    path_str: &str,
    recursive: bool,
    pattern: Option<&str>,
    max_files: Option<usize>,
) -> anyhow::Result<Vec<PathBuf>> {
    let normalized_path = path::validate_and_resolve_path(path_str, Some(false))
        .await
        .map_err(|e| anyhow::anyhow!("Path validation failed: {e}"))?;

    let mut files = Vec::new();
    let mut to_process = vec![normalized_path];

    while let Some(current_path) = to_process.pop() {
        let mut entries = fs::read_dir(&current_path).await.map_err(|e| {
            anyhow::anyhow!(
                "Failed to read directory '{}': {}",
                current_path.display(),
                e
            )
        })?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read directory entry: {e}"))?
        {
            let entry_path = entry.path();
            let metadata = entry
                .metadata()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to get metadata: {e}"))?;

            if metadata.is_file() {
                let file_name = entry_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");

                // Check pattern match if provided
                if let Some(pat) = pattern
                    && !matches_pattern(file_name, pat)
                {
                    continue;
                }

                files.push(entry_path);

                // Check max_files limit
                if let Some(max) = max_files
                    && files.len() >= max
                {
                    return Ok(files);
                }
            } else if metadata.is_dir() && recursive {
                to_process.push(entry_path);
            }
        }
    }

    Ok(files)
}

const NAME: &str = "rag";
const DESCRIPTION: &str = "Index documents and retrieve relevant context using Dripstone-backed retrieval (RAG - Retrieval Augmented Generation). \
Supports multiple vector store backends for persistent storage (Qdrant, Pinecone, PostgreSQL, Redis, Chroma, S3). \
Operations: index (store a document), index_batch (index multiple documents), index_file (read and index a single file), \
index_directory (read and index all files in a directory), query (search for relevant chunks), \
advanced_query (search with hybrid, full-text, reranking, and filtering), graph_search (graph-enhanced search with entity relationships), hybrid_vector_graph_search (vector+graph hybrid retrieval), \
stats (get cache info), clear (remove all documents). \
Advanced features: dense search, dense+sparse hybrid retrieval, full-text search, dense+full-text hybrid retrieval, cross-encoder reranking, metadata filtering, batch indexing, \
graph RAG (entity extraction and relationship traversal), and adjustable similarity thresholds. \
File operations (index_file, index_directory) automatically read files, chunk them, and store embeddings for semantic retrieval. \
This prevents context window overflow by storing large file outputs in vector DB instead of sending to LLM. \
Full-text search requires a Dripstone full-text capable backend. Graph search requires graph backend configuration.";

impl ToolMeta for RagTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["index", "index_batch", "index_file", "index_directory", "query", "advanced_query", "graph_search", "hybrid_vector_graph_search", "stats", "clear"],
                    "description": "Operation to perform: 'index' to store a document, 'index_batch' to index multiple documents, 'index_file' to read and index a single file, 'index_directory' to read and index all files in a directory, 'query' for basic search, 'advanced_query' for search with dense/hybrid/full-text modes, reranking, and filtering, 'graph_search' for graph-enhanced search with entity relationships, 'hybrid_vector_graph_search' for vector+graph hybrid retrieval, 'stats' to view cache info, 'clear' to remove all documents"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path to index (required for 'index_file' and 'index_directory' operations). Supports local paths (relative or absolute) and S3 URIs (s3://bucket-name/path/to/file.txt)"
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.rs', '*.py', '*test*'). Only used with 'index_directory' operation. Supports basic patterns: '*.ext' for file extensions, '*text*' for substring matching, 'text*' for prefix, '*text' for suffix"
                },
                "max_files": {
                    "type": "integer",
                    "description": "Maximum number of files to index (default: unlimited, valid for 'index_directory' operation)",
                    "minimum": 1
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to recursively index subdirectories (default: false, valid for 'index_directory' operation)",
                    "default": false
                },
                "document": {
                    "type": "string",
                    "description": "Document text to index (required for 'index' operation)"
                },
                "doc_id": {
                    "type": "string",
                    "description": "Unique identifier for the document (required for 'index', optional for 'advanced_query' as filter)"
                },
                "query": {
                    "type": "string",
                    "description": "Natural language query to search for (required for 'query' and 'advanced_query' operations)"
                },
                "search_mode": {
                    "type": "string",
                    "enum": ["dense", "hybrid", "full_text", "hybrid_full_text"],
                    "description": "Dripstone retrieval mode for 'query' and 'advanced_query'. 'dense' = semantic embeddings only, 'hybrid' = dense+sparse hybrid, 'full_text' = lexical full-text, 'hybrid_full_text' = dense + full-text fusion. Defaults to 'hybrid_full_text'."
                },
                "use_maxsim": {
                    "type": "boolean",
                    "description": "Request ColBERT-style MaxSim late interaction on dense retrieval. Requires the index to have been built with late-interaction token vectors (RagHashlineOptions.late_interaction or CALICHE_RAG_LATE_INTERACTION=true). When the service was built that way, MaxSim is already applied; this flag is informational.",
                    "default": false
                },
                "fusion_mode": {
                    "type": "string",
                    "enum": ["weighted", "rrf"],
                    "description": "Fusion strategy for hybrid retrieval. 'weighted' uses dense_weight. 'rrf' uses reciprocal-rank fusion and is currently honored for 'hybrid_full_text'. Defaults to 'weighted'."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, valid for 'query' and 'advanced_query')",
                    "default": 5
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Minimum similarity threshold 0.0-1.0 (default: 0.3, valid for 'query' and 'advanced_query'). Lower values return more results but may be less relevant.",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "use_diversity": {
                    "type": "boolean",
                    "description": "Enable diversity reranking to avoid similar chunks (default: false, valid for 'advanced_query')",
                    "default": false
                },
                "use_mmr": {
                    "type": "boolean",
                    "description": "Enable MMR (Maximal Marginal Relevance) diversity reranking (default: false, valid for 'query' and 'advanced_query')",
                    "default": false
                },
                "mmr_lambda": {
                    "type": "number",
                    "description": "MMR lambda 0.0-1.0: 0 = max diversity, 1 = max relevance (default: 0.7, valid when use_mmr=true)",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "use_temporal_decay": {
                    "type": "boolean",
                    "description": "Down-weight older chunks by creation time (default: false, valid for 'query' and 'advanced_query'; chunks need created_at metadata)",
                    "default": false
                },
                "temporal_decay_half_life_days": {
                    "type": "number",
                    "description": "Half-life in days for temporal decay (default: 30, valid when use_temporal_decay=true)",
                    "default": 30,
                    "minimum": 0.1
                },
                "dense_weight": {
                    "type": "number",
                    "description": "Weight for the dense side of weighted hybrid retrieval (default: 0.7, valid for 'hybrid' and 'hybrid_full_text' with fusion_mode='weighted')",
                    "default": 0.7,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "sparse_weight": {
                    "type": "number",
                    "description": "Compatibility weight for the sparse side of hybrid retrieval. When both dense_weight and sparse_weight are provided, RagTool normalizes them to a dense_weight for Dripstone.",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "rerank": {
                    "type": "boolean",
                    "description": "Enable cross-encoder reranking for better relevance (default: false, valid for 'query' and 'advanced_query', requires 'rag-reranking' feature)",
                    "default": false
                },
                "metadata_filter": {
                    "type": "object",
                    "description": "Metadata filter for 'query' and 'advanced_query'. Supports two formats: 1) Simple key-value pairs: {\"category\": \"tech\", \"year\": 2024}. 2) Advanced Filter DSL for complex queries: {\"type\": \"and\", \"left\": {\"type\": \"equal\", \"field\": \"category\", \"value\": \"tech\"}, \"right\": {\"type\": \"greater_than\", \"field\": \"year\", \"value\": 2023}}. Filter DSL supports: equal, not_equal, in, not_in, greater_than, less_than, greater_than_or_equal, less_than_or_equal, and, or, not. Example: {\"type\": \"or\", \"left\": {\"type\": \"equal\", \"field\": \"category\", \"value\": \"tech\"}, \"right\": {\"type\": \"equal\", \"field\": \"category\", \"value\": \"science\"}}",
                    "additionalProperties": true
                },
                "max_context_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to use for retrieved context (optional, auto-detects from provider if not specified, valid for 'query' and 'advanced_query')",
                    "minimum": 100
                },
                "documents": {
                    "type": "array",
                    "description": "Array of documents to index in batch format: [{\"doc_id\": \"id1\", \"document\": \"text1\"}, ...] (required for 'index_batch' operation)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_id": {"type": "string"},
                            "document": {"type": "string"}
                        },
                        "required": ["doc_id", "document"]
                    }
                },
                "max_hops": {
                    "type": "integer",
                    "description": "Maximum number of hops for graph traversal (default: 1, valid for 'graph_search' and 'hybrid_vector_graph_search')",
                    "default": 1,
                    "minimum": 1,
                    "maximum": 5
                },
                "anchor_node_id": {
                    "type": "string",
                    "description": "Graph anchor node for hybrid search (measure graph distance from this node)"
                },
                "strategy": {
                    "type": "string",
                    "enum": ["auto", "intersection", "vector_then_graph", "graph_then_vector", "union"],
                    "description": "Hybrid search strategy. 'auto' (default) selects based on query: anchor+text->intersection, anchor only->graph_then_vector, text only->vector_then_graph",
                    "default": "auto"
                },
                "vector_weight": {
                    "type": "number",
                    "default": 0.6,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Weight for vector score in union strategy"
                },
                "graph_weight": {
                    "type": "number",
                    "default": 0.4,
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Weight for graph score in union strategy"
                }
            },
            "required": ["operation"]
        })
    }
}

#[async_trait]
impl Tool for RagTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let operation = match params::extract_operation(&arguments) {
            Ok(op) => op,
            Err(e) => return ToolResult::error(e),
        };
        match operation.as_str() {
            "index" => self.execute_index(&arguments).await,
            "index_batch" => self.execute_index_batch(&arguments).await,
            "index_file" => self.execute_index_file(&arguments).await,
            "index_directory" => self.execute_index_directory(&arguments).await,
            "query" => self.execute_query(&arguments).await,
            "advanced_query" => self.execute_advanced_query(&arguments).await,
            "graph_search" => self.execute_graph_search(&arguments).await,
            "hybrid_vector_graph_search" => {
                self.execute_hybrid_vector_graph_search(&arguments).await
            }
            "stats" => self.execute_stats().await,
            "clear" => self.execute_clear().await,
            _ => ToolResult::error(format!("Unknown operation: {operation}")),
        }
    }
}

#[cfg(test)]
mod search_mode_tests {
    use super::{RagSearchMode, parse_search_mode};
    use serde_json::json;

    #[test]
    fn omitted_search_mode_defaults_to_hybrid_full_text() {
        assert_eq!(
            parse_search_mode(&json!({})),
            RagSearchMode::HybridDenseFullText
        );
        assert_eq!(
            parse_search_mode(&json!({"search_mode": "hybrid_full_text"})),
            RagSearchMode::HybridDenseFullText
        );
    }

    #[test]
    fn explicit_dense_and_sparse_hybrid_modes_are_preserved() {
        assert_eq!(
            parse_search_mode(&json!({"search_mode": "dense"})),
            RagSearchMode::Dense
        );
        assert_eq!(
            parse_search_mode(&json!({"search_mode": "hybrid"})),
            RagSearchMode::HybridDenseSparse
        );
        assert_eq!(
            parse_search_mode(&json!({"search_mode": "full_text"})),
            RagSearchMode::FullText
        );
    }
}
