//! JavaScript bindings for Caliche RAG Service

use caliche::agent::service::ProviderFactory;
use caliche::agent::tools::RagTool as RustRagTool;
use caliche::rag::RagService as RustRagService;
use caliche::rag::{RagFusionMode, RagSearchMode};
use caliche::traits::credentials::EnvironmentCredentialProvider;
use napi_derive::napi;
use std::sync::Arc;

/// JavaScript wrapper for RagService
#[napi]
pub struct RagService {
    inner: Arc<RustRagService>,
}

#[napi]
impl RagService {
    /// Create a new RagService with default in-memory backend
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(provider: String, model: String) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let credential_provider = Arc::new(EnvironmentCredentialProvider::new());
        let embedding_provider =
            ProviderFactory::create_embedding_provider(provider_type, model, credential_provider)
                .await
                .map_err(|e| {
                    napi::Error::from_reason(format!("Failed to create embedding provider: {}", e))
                })?
                .ok_or_else(|| napi::Error::from_reason("Provider does not support embeddings"))?;

        let rag_service = RustRagService::new(embedding_provider);
        Ok(Self {
            inner: Arc::new(rag_service),
        })
    }

    /// Index a document
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn index_document(
        &self,
        content: String,
        doc_id: String,
    ) -> napi::Result<IndexDocumentResult> {
        let result = self.inner.index_document(&content, &doc_id).await;
        match result {
            Ok((chunk_count, usage)) => Ok(IndexDocumentResult {
                chunk_count: chunk_count as u32,
                usage: usage.map(Into::into),
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Failed to index document: {}",
                e
            ))),
        }
    }

    /// Query for relevant chunks
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn query(
        &self,
        query: String,
        top_k: u32,
        min_similarity: Option<f64>,
    ) -> napi::Result<QueryResult> {
        let min_sim = min_similarity.unwrap_or(0.3) as f32;
        let result = self.inner.query(&query, top_k as usize, min_sim).await;
        match result {
            Ok((results, top_similarity, usage)) => Ok(QueryResult {
                results,
                top_similarity: top_similarity.map(|v| v as f64),
                usage: usage.map(Into::into),
            }),
            Err(e) => Err(napi::Error::from_reason(format!("Failed to query: {}", e))),
        }
    }

    /// Enhanced query with hybrid search, reranking, and filtering
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn enhanced_query(
        &self,
        query: String,
        top_k: u32,
        min_similarity: Option<f64>,
        search_mode: Option<String>,
        dense_weight: Option<f64>,
        sparse_weight: Option<f64>,
        rerank: Option<bool>,
        metadata_filter: Option<serde_json::Value>,
        max_context_tokens: Option<u32>,
    ) -> napi::Result<EnhancedQueryResult> {
        let min_sim = min_similarity.unwrap_or(0.3) as f32;
        let rerank = rerank.unwrap_or(false);
        let search_mode = match search_mode.as_deref() {
            Some("hybrid") => RagSearchMode::HybridDenseSparse,
            Some("full_text") => RagSearchMode::FullText,
            Some("hybrid_full_text") => RagSearchMode::HybridDenseFullText,
            Some("dense") | Some(_) | None => RagSearchMode::Dense,
        };
        let fusion_mode = RagFusionMode::Weighted;

        let query_result = self
            .inner
            .enhanced_query(
                &query,
                top_k as usize,
                min_sim,
                search_mode,
                fusion_mode,
                dense_weight.map(|w| w as f32),
                sparse_weight.map(|w| w as f32),
                rerank,
                metadata_filter.as_ref(),
                None, // Filter DSL not exposed yet
                max_context_tokens.map(|t| t as usize),
                None, // Provider type
                None, // Model
                None, // use_mmr
                None, // mmr_lambda
                None, // use_temporal_decay
                None, // temporal_decay_half_life_days
            )
            .await;

        match query_result {
            Ok((results, top_sim, usage, warnings, stats)) => {
                // Get chunk texts for results
                let mut texts = Vec::new();
                for result in results.iter() {
                    let text = self.inner.get_chunk_text(&result.chunk.id).await;
                    texts.push(text);
                }

                let result_items: Vec<QueryResultItem> = results
                    .iter()
                    .zip(texts.iter())
                    .map(|(result, text)| QueryResultItem {
                        id: result.chunk.id.clone(),
                        score: result.score as f64,
                        text: text.clone().unwrap_or_default(),
                        metadata: Some(result.chunk.metadata.clone()),
                    })
                    .collect();

                Ok(EnhancedQueryResult {
                    results: result_items,
                    top_similarity: top_sim.map(|v| v as f64),
                    usage: usage.map(Into::into),
                    warnings,
                    context_stats: stats.map(Into::into),
                })
            }
            Err(e) => Err(napi::Error::from_reason(format!(
                "Failed to perform enhanced query: {}",
                e
            ))),
        }
    }

    /// Graph search (requires graph backend)
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn graph_search(
        &self,
        query: String,
        top_k: u32,
        max_hops: Option<u32>,
    ) -> napi::Result<GraphSearchResult> {
        let max_hops = max_hops.unwrap_or(2) as usize;
        let result = self
            .inner
            .graph_search(&query, top_k as usize, max_hops)
            .await;
        match result {
            Ok(graph_result) => Ok(GraphSearchResult {
                entry_nodes: graph_result
                    .entry_nodes
                    .iter()
                    .map(|n| GraphNode {
                        id: n.id.clone(),
                        r#type: n.node_type.clone(),
                    })
                    .collect(),
                expanded_nodes: graph_result
                    .expanded_nodes
                    .iter()
                    .map(|n| GraphNode {
                        id: n.id.clone(),
                        r#type: n.node_type.clone(),
                    })
                    .collect(),
                expanded_edges: graph_result
                    .expanded_edges
                    .iter()
                    .map(|e| GraphEdge {
                        source: e.source_id.clone(),
                        target: e.target_id.clone(),
                        relation: e.edge_type.clone(),
                    })
                    .collect(),
                context: graph_result.context,
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Failed to perform graph search: {}",
                e
            ))),
        }
    }

    /// Get statistics
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn stats(&self) -> napi::Result<RagStats> {
        let result = self.inner.stats().await;
        match result {
            Ok((embedding_count, text_count)) => Ok(RagStats {
                embedding_count: embedding_count as u32,
                text_count: text_count as u32,
            }),
            Err(e) => Err(napi::Error::from_reason(format!(
                "Failed to get stats: {}",
                e
            ))),
        }
    }

    /// Clear all indexed documents
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn clear(&self) -> napi::Result<()> {
        self.inner
            .clear()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to clear: {}", e)))
    }
}

#[napi(object)]
pub struct IndexDocumentResult {
    pub chunk_count: u32,
    pub usage: Option<crate::TokenUsageJs>,
}

#[napi(object)]
pub struct QueryResult {
    pub results: Vec<String>,
    pub top_similarity: Option<f64>,
    pub usage: Option<crate::TokenUsageJs>,
}

#[napi(object)]
pub struct QueryResultItem {
    pub id: String,
    pub score: f64,
    pub text: String,
    pub metadata: Option<serde_json::Value>,
}

#[napi(object)]
pub struct EnhancedQueryResult {
    pub results: Vec<QueryResultItem>,
    pub top_similarity: Option<f64>,
    pub usage: Option<crate::TokenUsageJs>,
    pub warnings: Vec<String>,
    pub context_stats: Option<ContextStats>,
}

#[napi(object)]
pub struct ContextStats {
    pub included_chunks: u32,
    pub omitted_chunks: u32,
    pub total_chunks: u32,
    pub truncated: bool,
}

impl From<caliche::agent::tools::context_manager::ContextStats> for ContextStats {
    fn from(stats: caliche::agent::tools::context_manager::ContextStats) -> Self {
        Self {
            included_chunks: stats.included_chunks as u32,
            omitted_chunks: stats.omitted_chunks as u32,
            total_chunks: stats.total_chunks as u32,
            truncated: stats.truncated,
        }
    }
}

#[napi(object)]
pub struct GraphNode {
    pub id: String,
    pub r#type: String,
}

#[napi(object)]
pub struct GraphEdge {
    pub source: String,
    pub target: String,
    pub relation: String,
}

#[napi(object)]
pub struct GraphSearchResult {
    pub entry_nodes: Vec<GraphNode>,
    pub expanded_nodes: Vec<GraphNode>,
    pub expanded_edges: Vec<GraphEdge>,
    pub context: String,
}

#[napi(object)]
pub struct RagStats {
    pub embedding_count: u32,
    pub text_count: u32,
}

/// JavaScript wrapper for RagTool
#[napi]
pub struct RagTool {
    pub(crate) inner: Arc<RustRagTool>,
}

#[napi]
impl RagTool {
    /// Create a new RagTool with default in-memory backend
    #[napi(factory)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn new(provider: String, model: String) -> napi::Result<Self> {
        let provider_type = crate::parse_provider_type(&provider)?;
        let credential_provider = Arc::new(EnvironmentCredentialProvider::new());
        let embedding_provider =
            ProviderFactory::create_embedding_provider(provider_type, model, credential_provider)
                .await
                .map_err(|e| {
                    napi::Error::from_reason(format!("Failed to create embedding provider: {}", e))
                })?
                .ok_or_else(|| napi::Error::from_reason("Provider does not support embeddings"))?;

        let rag_tool = RustRagTool::new(embedding_provider, None::<String>);
        Ok(Self {
            inner: Arc::new(rag_tool),
        })
    }

    /// Get the underlying RAG service from a RagTool
    #[napi(catch_unwind)]
    pub fn service(&self) -> RagService {
        RagService {
            inner: self.inner.service(),
        }
    }
}
