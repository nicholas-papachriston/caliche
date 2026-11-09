//! Hashline options for RAG indexing and search.
//!
//! Controls content-addressed hashline tagging, BM25 deduplication, line-level BM25,
//! chunking strategy (e.g. AST-aware for code/JSON), and optional MaxSim late interaction.

use dripstone::config::ChunkStrategy;

/// Options for hashline tagging, BM25 behavior, and chunking in RAG indexing.
#[derive(Debug, Clone)]
pub struct RagHashlineOptions {
    /// Enable hashline tagging during ingestion (line hashes for edit compatibility).
    pub enabled: bool,
    /// Format search results with `line:hash| content` tags for hashline_replace/edit tools.
    pub format_results: bool,
    /// Skip duplicate chunk texts (by content hash) when fitting BM25.
    pub bm25_dedup: bool,
    /// Use line-level BM25: index each line, aggregate scores by chunk.
    pub line_level_bm25: bool,
    /// Chunking strategy. AstAware uses AST boundaries when path is known and language is supported.
    pub chunking_strategy: ChunkStrategy,
    /// Embed token multi-vectors at ingest and search with MaxSim / PLAID.
    pub late_interaction: bool,
}

impl Default for RagHashlineOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            format_results: false,
            bm25_dedup: false,
            line_level_bm25: false,
            chunking_strategy: ChunkStrategy::Sentence,
            late_interaction: false,
        }
    }
}

impl RagHashlineOptions {
    /// All options disabled (default for non-workspace RAG).
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Options for edit-tool compatibility only (hashline tagging, no BM25 changes).
    pub fn for_edit_tools(enabled: bool, format_results: bool) -> Self {
        Self {
            enabled,
            format_results,
            bm25_dedup: false,
            line_level_bm25: false,
            chunking_strategy: ChunkStrategy::Sentence,
            late_interaction: false,
        }
    }

    /// Options for workspace indexing: bm25_dedup always true, hashline when edit tools present,
    /// AST-aware chunking for code and structured files.
    pub fn for_workspace(has_edit_tools: bool, line_level_bm25: bool) -> Self {
        Self {
            enabled: has_edit_tools,
            format_results: has_edit_tools,
            bm25_dedup: true,
            line_level_bm25,
            chunking_strategy: ChunkStrategy::AstAware,
            late_interaction: false,
        }
    }

    /// Enable hashline tagging.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Format search results with line:hash tags for edit tools.
    pub fn format_results(mut self, format: bool) -> Self {
        self.format_results = format;
        self
    }

    /// Skip duplicate chunk texts when fitting BM25.
    pub fn bm25_dedup(mut self, dedup: bool) -> Self {
        self.bm25_dedup = dedup;
        self
    }

    /// Use line-level BM25 (higher precision, more memory).
    pub fn line_level_bm25(mut self, enabled: bool) -> Self {
        self.line_level_bm25 = enabled;
        self
    }

    /// Set chunking strategy. AstAware uses AST boundaries when path is provided.
    pub fn chunking_strategy(mut self, strategy: ChunkStrategy) -> Self {
        self.chunking_strategy = strategy;
        self
    }

    /// Enable ColBERT-style MaxSim (token multi-vectors + optional PLAID wrap).
    pub const fn late_interaction(mut self, enabled: bool) -> Self {
        self.late_interaction = enabled;
        self
    }
}
