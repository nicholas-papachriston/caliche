//! RAG (Retrieval Augmented Generation) API
//!
//! This module provides a reusable RAG service that can be used independently
//! or integrated into pipelines, builders, workflows, and tools.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```rust,no_run
//! use caliche::rag::RagService;
//! use caliche::llm::OpenAIProvider;
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let embedding_provider = Arc::new(
//!     OpenAIProvider::try_new("key".to_string(), "model".to_string())?
//! );
//! let rag = RagService::new(embedding_provider);
//!
//! // Index a document
//! let (chunks, usage) = rag.index_document("Python is a programming language...", "python_intro").await?;
//!
//! // Query
//! let (results, top_sim, usage) = rag.query("What is Python?", 5, 0.3).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Usage with Hybrid Search and Reranking
//!
//! ```rust,no_run
//! use caliche::rag::RagService;
//! use caliche::llm::ProviderType;
//! use serde_json::json;
//!
//! # async fn example() -> anyhow::Result<()> {
//! use caliche::llm::OpenAIProvider;
//! use std::sync::Arc;
//! let embedding_provider = Arc::new(
//!     OpenAIProvider::try_new("key".to_string(), "model".to_string())?
//! );
//! let rag = RagService::new(embedding_provider);
//! // Advanced query with hybrid search, reranking, and metadata filtering
//! let (results, top_sim, usage, warnings, context_stats) = rag.enhanced_query(
//!     "machine learning",
//!     10,
//!     0.3,
//!     caliche::rag::RagSearchMode::HybridDenseSparse,
//!     caliche::rag::RagFusionMode::Weighted,
//!     Some(0.7),  // dense_weight
//!     Some(0.3),  // sparse_weight
//!     true,  // rerank
//!     Some(&json!({"category": "tech"})),  // metadata_filter
//!     None,  // filter (Filter DSL, takes precedence over metadata_filter)
//!     Some(4000),  // max_context_tokens
//!     Some(ProviderType::OpenAI),
//!     Some("gpt-4o"),
//!     Some(true),   // use_mmr
//!     Some(0.7),    // mmr_lambda
//!     None,         // use_temporal_decay
//!     None,         // temporal_decay_half_life_days
//! ).await?;
//! # Ok(())
//! # }
//! ```

/// Default options for RAG search (MMR, temporal decay). Used by the RAG tool
/// when the LLM does not specify them; can be set via CLI or workspace config.
#[derive(Debug, Clone, Default)]
pub struct RagSearchOptions {
    /// Enable MMR diversity reranking by default.
    pub use_mmr: Option<bool>,
    /// MMR lambda (0.0–1.0) when use_mmr is true.
    pub mmr_lambda: Option<f64>,
    /// Enable temporal decay by default.
    pub use_temporal_decay: Option<bool>,
    /// Half-life in days when use_temporal_decay is true.
    pub temporal_decay_half_life_days: Option<f64>,
}

/// Retrieval knobs for automatic workspace context injection.
///
/// Injected context always uses Dripstone dense + full-text hybrid search
/// ([`RagSearchMode::HybridDenseFullText`] with weighted fusion). MMR and
/// temporal decay still apply when set on [`Self::search_options`].
#[derive(Debug, Clone)]
pub struct WorkspaceContextQuery {
    /// Number of chunks to retrieve.
    pub top_k: usize,
    /// Minimum similarity threshold (0.0–1.0).
    pub min_similarity: f32,
    /// Kept for CLI/workspace ingest wiring (`--line-level-bm25`). Injected
    /// retrieval uses full-text hybrid rather than sparse BM25.
    pub line_level_bm25: bool,
    /// Optional MMR / temporal-decay defaults from CLI/workspace config.
    pub search_options: Option<RagSearchOptions>,
}

impl Default for WorkspaceContextQuery {
    fn default() -> Self {
        Self {
            top_k: 12,
            min_similarity: 0.25,
            line_level_bm25: false,
            search_options: None,
        }
    }
}

impl WorkspaceContextQuery {
    /// Injected context always uses Dripstone enhanced retrieval.
    #[must_use]
    pub const fn uses_enhanced_retrieval(&self) -> bool {
        true
    }

    /// Preferred injected search mode: dense fused with Dripstone FTS.
    #[must_use]
    pub const fn search_mode(&self) -> RagSearchMode {
        RagSearchMode::HybridDenseFullText
    }

    /// Weighted fusion keeps fused scores in a similarity-like range so
    /// [`Self::min_similarity`] still applies. Reciprocal-rank scores are ~1/k.
    #[must_use]
    pub const fn fusion_mode(&self) -> RagFusionMode {
        RagFusionMode::Weighted
    }
}

#[cfg(test)]
mod workspace_context_query_tests {
    use super::{RagFusionMode, RagSearchMode, RagSearchOptions, WorkspaceContextQuery};

    #[test]
    fn injected_context_always_uses_enhanced_full_text_hybrid() {
        let query = WorkspaceContextQuery::default();
        assert!(query.uses_enhanced_retrieval());
        assert_eq!(query.search_mode(), RagSearchMode::HybridDenseFullText);
        assert_eq!(query.fusion_mode(), RagFusionMode::Weighted);
    }

    #[test]
    fn line_level_bm25_does_not_downgrade_injected_fts_hybrid() {
        let query = WorkspaceContextQuery {
            line_level_bm25: true,
            ..WorkspaceContextQuery::default()
        };
        assert_eq!(query.search_mode(), RagSearchMode::HybridDenseFullText);
    }

    #[test]
    fn mmr_is_still_carried_on_search_options() {
        let query = WorkspaceContextQuery {
            search_options: Some(RagSearchOptions {
                use_mmr: Some(true),
                ..RagSearchOptions::default()
            }),
            ..WorkspaceContextQuery::default()
        };
        assert!(query.uses_enhanced_retrieval());
        assert_eq!(
            query
                .search_options
                .as_ref()
                .and_then(|options| options.use_mmr),
            Some(true)
        );
    }
}

/// Search modes exposed by Caliche while delegating execution to Dripstone.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RagSearchMode {
    /// Semantic dense retrieval using embeddings only.
    #[default]
    Dense,
    /// Dense retrieval fused with Dripstone's sparse/BM25-style hybrid search path.
    HybridDenseSparse,
    /// Lexical full-text search using a Dripstone full-text indexed backend.
    FullText,
    /// Dense retrieval fused with Dripstone full-text retrieval.
    HybridDenseFullText,
}

impl RagSearchMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::HybridDenseSparse => "hybrid",
            Self::FullText => "full_text",
            Self::HybridDenseFullText => "hybrid_full_text",
        }
    }
}

/// Fusion modes supported by the Dripstone-backed RAG query path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RagFusionMode {
    /// Weighted score fusion using the provided dense weight.
    #[default]
    Weighted,
    /// Reciprocal-rank fusion, currently supported for dense + full-text hybrid search.
    ReciprocalRank,
}

impl RagFusionMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Weighted => "weighted",
            Self::ReciprocalRank => "rrf",
        }
    }
}

pub mod filter_dto;
pub mod graph;
pub mod hashline_options;
pub mod hybrid;
pub mod service;

pub use filter_dto::FilterDto;
pub use graph::{ExtractedTriplet, GraphIndexer, GraphSearch, GraphSearchResult};
/// Options for hashline tagging and BM25 in RAG. Use [`RagHashlineOptions::for_workspace`]
/// for workspace indexing or [`RagHashlineOptions::for_edit_tools`] for edit compatibility.
pub use hashline_options::RagHashlineOptions;
pub use hybrid::{
    HybridGraphVectorSearch, HybridQuery, HybridResult, HybridStrategy, ResultProvenance,
};
pub use service::{EmbeddingProviderWrapper, RagService};
