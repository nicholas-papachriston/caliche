//! Configuration for `Dripstone`.
//!
//! This module provides a consolidated configuration hierarchy for all `Dripstone` features.
//! Instead of scattered config types, everything is under `DripstoneConfig`.

use crate::core::{FullTextOptions, Metric, SearchMode};
use crate::search::FusionConfig;
use serde::{Deserialize, Serialize};
#[cfg(feature = "embedding-ort")]
use std::path::PathBuf;

/// Unified configuration for `Dripstone`.
///
/// # Example
///
/// ```rust
/// use dripstone::config::DripstoneConfig;
///
/// // Use defaults
/// let config = DripstoneConfig::default();
///
/// // Or customize
/// let config = DripstoneConfig::default()
///     .with_chunking(|c| c.target_tokens(1024).overlap_tokens(100))
///     .with_search(|s| s.default_limit(20));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[must_use]
pub struct DripstoneConfig {
    /// Chunking configuration
    pub chunking: ChunkingConfig,
    /// Search configuration
    pub search: SearchConfig,
    /// Index configuration
    pub index: IndexConfig,
    /// Hashline configuration (content-addressed tagging for RAG)
    pub hashline: HashlineConfig,
}

impl DripstoneConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Modify hashline configuration.
    pub fn with_hashline<F>(mut self, f: F) -> Self
    where
        F: FnOnce(HashlineConfig) -> HashlineConfig,
    {
        self.hashline = f(self.hashline);
        self
    }

    /// Modify chunking configuration.
    pub fn with_chunking<F>(mut self, f: F) -> Self
    where
        F: FnOnce(ChunkingConfig) -> ChunkingConfig,
    {
        self.chunking = f(self.chunking);
        self
    }

    /// Modify search configuration.
    pub fn with_search<F>(mut self, f: F) -> Self
    where
        F: FnOnce(SearchConfig) -> SearchConfig,
    {
        self.search = f(self.search);
        self
    }

    /// Modify index configuration.
    pub fn with_index<F>(mut self, f: F) -> Self
    where
        F: FnOnce(IndexConfig) -> IndexConfig,
    {
        self.index = f(self.index);
        self
    }
}

/// Configuration for content-addressed hashline tagging.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
#[expect(
    clippy::struct_excessive_bools,
    reason = "public serde configuration keeps independent feature toggles as compatible fields"
)]
pub struct HashlineConfig {
    /// Enable hashline tagging during ingestion
    pub enabled: bool,
    /// Number of hash bytes (1 = 2 hex chars, 2 = 4 hex chars). Default 1.
    pub hash_bytes: u8,
    /// Normalize trailing whitespace before hashing
    pub normalize_trailing_ws: bool,
    /// Include per-line hashes (vs chunk-level only). More metadata, enables line-anchored output.
    pub line_level: bool,
    /// Format search result content with hashline tags when returning. Enables line-tagged edit tools.
    pub format_results: bool,
    /// Enable incremental indexing: skip re-embedding chunks whose content hash is unchanged.
    pub incremental_indexing: bool,
    /// Skip duplicate chunk texts (by content hash) when fitting BM25. Improves IDF accuracy for corpora with repeated chunks.
    pub bm25_dedup: bool,
    /// Use line-level BM25: index each line, aggregate scores by chunk. Higher precision for line-specific matches.
    pub line_level_bm25: bool,
}

impl Default for HashlineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hash_bytes: 1,
            normalize_trailing_ws: true,
            line_level: true,
            format_results: false,
            incremental_indexing: false,
            bm25_dedup: false,
            line_level_bm25: false,
        }
    }
}

impl HashlineConfig {
    /// Enable hashline tagging.
    pub const fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set `format_results` to true for line-tagged edit tools.
    pub const fn format_results(mut self, format: bool) -> Self {
        self.format_results = format;
        self
    }

    /// Skip duplicate chunk texts (by content hash) when fitting BM25.
    pub const fn bm25_dedup(mut self, dedup: bool) -> Self {
        self.bm25_dedup = dedup;
        self
    }

    /// Use line-level BM25 for hybrid search.
    pub const fn line_level_bm25(mut self, enabled: bool) -> Self {
        self.line_level_bm25 = enabled;
        self
    }

    /// Enable incremental indexing. Requires a chunk hash store to be
    /// configured via [`crate::DripstoneBuilder::chunk_hash_store`]; builder
    /// construction will fail otherwise.
    pub const fn incremental_indexing(mut self, enabled: bool) -> Self {
        self.incremental_indexing = enabled;
        self
    }

    /// Include per-line hashes in addition to chunk-level tags.
    pub const fn line_level(mut self, enabled: bool) -> Self {
        self.line_level = enabled;
        self
    }

    /// Number of hash bytes to emit per chunk (1 = 2 hex chars, 2 = 4 hex
    /// chars, ...).
    pub const fn hash_bytes(mut self, bytes: u8) -> Self {
        self.hash_bytes = bytes;
        self
    }

    /// Normalize trailing whitespace before hashing so trivial newline
    /// differences don't invalidate the cache.
    pub const fn normalize_trailing_ws(mut self, normalize: bool) -> Self {
        self.normalize_trailing_ws = normalize;
        self
    }

    /// Convert to [`crate::hashline::HashlineOptions`] for the hashline module.
    #[must_use]
    pub const fn to_options(&self) -> crate::hashline::HashlineOptions {
        crate::hashline::HashlineOptions {
            hash_bytes: self.hash_bytes,
            normalize_trailing_ws: self.normalize_trailing_ws,
        }
    }
}

/// Chunking strategy for document ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ChunkStrategy {
    /// Sentence-aware chunking (default) - respects sentence boundaries
    #[default]
    Sentence,
    /// Fixed-size token chunks
    FixedTokens,
    /// Paragraph-based chunking
    Paragraph,
    /// Semantic chunking based on embedding similarity
    Semantic,
    /// AST-aware chunking. With the `ast` feature (enabled by `corpus`), splits
    /// at function and method boundaries using tree-sitter. Set
    /// [`ChunkingConfig::code_language`] or pass `language` / `path` metadata.
    /// Falls back to sentence chunking when parsing is unavailable.
    AstAware,
    /// No chunking - treat entire document as one chunk
    None,
}

/// Configuration for document chunking.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct ChunkingConfig {
    /// Chunking strategy to use
    pub strategy: ChunkStrategy,
    /// Target number of tokens per chunk
    pub target_tokens: usize,
    /// Number of overlap tokens between chunks
    pub overlap_tokens: usize,
    /// Minimum chunk size in tokens (chunks smaller than this are merged)
    pub min_tokens: usize,
    /// Maximum chunk size in tokens (hard limit)
    pub max_tokens: usize,
    /// Semantic chunking: similarity threshold for chunk boundaries (0.0-1.0)
    pub semantic_threshold: f32,
    /// Language tag for [`ChunkStrategy::AstAware`] when ingest metadata has none.
    ///
    /// Examples: `python`, `rust`, `javascript`, `go`, `java`. A `path` metadata
    /// field is used when this is unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code_language: Option<String>,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            strategy: ChunkStrategy::Sentence,
            target_tokens: 512,
            overlap_tokens: 50,
            min_tokens: 50,
            max_tokens: 1024,
            semantic_threshold: 0.7,
            code_language: None,
        }
    }
}

impl ChunkingConfig {
    /// Set the chunking strategy.
    pub const fn strategy(mut self, strategy: ChunkStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the target tokens per chunk.
    pub const fn target_tokens(mut self, tokens: usize) -> Self {
        self.target_tokens = tokens;
        self
    }

    /// Set the overlap tokens between chunks.
    pub const fn overlap_tokens(mut self, tokens: usize) -> Self {
        self.overlap_tokens = tokens;
        self
    }

    /// Set the minimum tokens per chunk.
    pub const fn min_tokens(mut self, tokens: usize) -> Self {
        self.min_tokens = tokens;
        self
    }

    /// Set the maximum tokens per chunk.
    pub const fn max_tokens(mut self, tokens: usize) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set the semantic threshold (for semantic chunking).
    pub const fn semantic_threshold(mut self, threshold: f32) -> Self {
        self.semantic_threshold = threshold;
        self
    }

    /// Set the language tag used by [`ChunkStrategy::AstAware`].
    pub fn code_language(mut self, language: impl Into<String>) -> Self {
        self.code_language = Some(language.into());
        self
    }

    /// Create a config for no chunking (whole document as one chunk).
    pub fn no_chunking() -> Self {
        Self {
            strategy: ChunkStrategy::None,
            ..Default::default()
        }
    }
}

/// Configuration for search operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct SearchConfig {
    /// Default number of results to return
    pub default_limit: usize,
    /// Default similarity metric
    pub default_metric: Metric,
    /// Enable hybrid search by default (dense + sparse)
    pub hybrid_enabled: bool,
    /// Weight for dense vectors in hybrid search (0.0-1.0)
    /// 1.0 = pure dense, 0.0 = pure sparse
    pub hybrid_dense_weight: f32,
    /// Enable reranking by default
    pub rerank_enabled: bool,
    /// Number of candidates to fetch before reranking (multiplier of limit)
    pub rerank_candidates_multiplier: usize,
    /// Default query-time options for full-text retrieval.
    pub full_text_options: FullTextOptions,
    /// Default dense weight for dense + full-text hybrid search.
    pub hybrid_full_text_dense_weight: f32,
    /// Default fusion strategy for dense + full-text hybrid search.
    pub hybrid_full_text_fusion: FusionConfig,
    /// Mode used by string search when the caller does not pick one.
    ///
    /// [`crate::Dripstone::new`] and the builder leave this at [`SearchMode::Dense`].
    /// [`crate::Dripstone::open`] sets [`SearchMode::HybridDenseFullText`].
    #[serde(default)]
    pub default_mode: SearchMode,
    /// Path to the ONNX model file for cross-encoder reranking
    #[cfg(feature = "embedding-ort")]
    pub rerank_model_path: Option<PathBuf>,
    /// When true (and hashline enabled), format chunk content with line:hash tags in search results
    pub hashline_in_results: bool,
    /// Embed per-token multi-vectors at ingest for MaxSim / PLAID late interaction.
    #[serde(default)]
    pub late_interaction: bool,
    /// Cap on token vectors attached to each chunk (and used when tokenizing queries).
    #[serde(default = "default_max_token_vectors")]
    pub max_token_vectors: usize,
}

const fn default_max_token_vectors() -> usize {
    crate::search::DEFAULT_MAX_TOKEN_VECTORS
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 10,
            default_metric: Metric::Cosine,
            hybrid_enabled: false,
            hybrid_dense_weight: 0.7,
            rerank_enabled: false,
            rerank_candidates_multiplier: 3,
            full_text_options: FullTextOptions::default(),
            hybrid_full_text_dense_weight: 0.5,
            hybrid_full_text_fusion: FusionConfig::weighted_sum().with_alpha(0.5),
            default_mode: SearchMode::Dense,
            #[cfg(feature = "embedding-ort")]
            rerank_model_path: None,
            hashline_in_results: false,
            late_interaction: false,
            max_token_vectors: crate::search::DEFAULT_MAX_TOKEN_VECTORS,
        }
    }
}

impl SearchConfig {
    /// Set the default limit.
    pub const fn default_limit(mut self, limit: usize) -> Self {
        self.default_limit = limit;
        self
    }

    /// Set the default metric.
    pub const fn default_metric(mut self, metric: Metric) -> Self {
        self.default_metric = metric;
        self
    }

    /// Enable or disable hybrid search.
    pub const fn hybrid(mut self, enabled: bool) -> Self {
        self.hybrid_enabled = enabled;
        self
    }

    /// Set the hybrid dense weight.
    pub const fn hybrid_weight(mut self, weight: f32) -> Self {
        self.hybrid_dense_weight = weight.clamp(0.0, 1.0);
        self
    }

    /// Override the default lexical query options.
    pub const fn full_text_options(mut self, options: FullTextOptions) -> Self {
        self.full_text_options = options;
        self
    }

    /// Set the default dense weight for dense + full-text hybrid search.
    pub const fn hybrid_full_text_weight(mut self, weight: f32) -> Self {
        self.hybrid_full_text_dense_weight = weight.clamp(0.0, 1.0);
        self.hybrid_full_text_fusion = FusionConfig::weighted_sum().with_alpha(weight);
        self
    }

    /// Set the default fusion config for dense + full-text hybrid search.
    pub const fn hybrid_full_text_fusion(mut self, fusion: FusionConfig) -> Self {
        self.hybrid_full_text_fusion = fusion;
        self
    }

    /// Set the mode used when a string search does not pick one.
    pub const fn default_mode(mut self, mode: SearchMode) -> Self {
        self.default_mode = mode;
        self
    }

    /// Enable or disable reranking.
    pub const fn rerank(mut self, enabled: bool) -> Self {
        self.rerank_enabled = enabled;
        self
    }

    /// Enable hashline formatting in search results (`line:hash| content`).
    pub const fn hashline_in_results(mut self, enabled: bool) -> Self {
        self.hashline_in_results = enabled;
        self
    }

    /// Embed token multi-vectors at ingest for MaxSim / PLAID.
    pub const fn late_interaction(mut self, enabled: bool) -> Self {
        self.late_interaction = enabled;
        self
    }

    /// Cap token vectors stored per chunk.
    pub fn max_token_vectors(mut self, max_tokens: usize) -> Self {
        self.max_token_vectors = max_tokens.max(1);
        self
    }
}

/// Index strategy for vector storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum IndexStrategy {
    /// HNSW index (default) - fast approximate nearest neighbor
    #[default]
    HNSW,
    /// Flat/brute-force index - exact search, slower but accurate
    Flat,
    /// IVF index - inverted file index for large datasets
    IVF,
}

/// Configuration for vector indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct IndexConfig {
    /// Index strategy to use
    pub strategy: IndexStrategy,
    /// HNSW: Number of bi-directional links per node
    pub hnsw_m: usize,
    /// HNSW: Size of the dynamic candidate list during construction
    pub hnsw_ef_construction: usize,
    /// HNSW: Size of the dynamic candidate list during search
    pub hnsw_ef_search: usize,
    /// IVF: Number of clusters/centroids
    pub ivf_nlist: usize,
    /// IVF: Number of clusters to probe during search
    pub ivf_nprobe: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            strategy: IndexStrategy::HNSW,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            ivf_nlist: 100,
            ivf_nprobe: 10,
        }
    }
}

impl IndexConfig {
    /// Set the index strategy.
    pub const fn strategy(mut self, strategy: IndexStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the `HNSW` `M` parameter.
    pub const fn hnsw_m(mut self, m: usize) -> Self {
        self.hnsw_m = m;
        self
    }

    /// Set the `HNSW` `ef_construction` parameter.
    pub const fn hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.hnsw_ef_construction = ef;
        self
    }

    /// Set the `HNSW` `ef_search` parameter.
    pub const fn hnsw_ef_search(mut self, ef: usize) -> Self {
        self.hnsw_ef_search = ef;
        self
    }

    /// Create a high-recall HNSW config (slower builds, better search).
    pub fn hnsw_high_recall() -> Self {
        Self {
            strategy: IndexStrategy::HNSW,
            hnsw_m: 32,
            hnsw_ef_construction: 400,
            hnsw_ef_search: 100,
            ..Default::default()
        }
    }

    /// Create a fast HNSW config (faster builds, lower recall).
    pub fn hnsw_fast() -> Self {
        Self {
            strategy: IndexStrategy::HNSW,
            hnsw_m: 8,
            hnsw_ef_construction: 100,
            hnsw_ef_search: 30,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DripstoneConfig::default();
        assert_eq!(config.chunking.target_tokens, 512);
        assert_eq!(config.search.default_limit, 10);
        assert_eq!(config.index.hnsw_m, 16);
    }

    #[test]
    fn test_config_builder() {
        let config = DripstoneConfig::default()
            .with_chunking(|c| c.target_tokens(1024))
            .with_search(|s| s.default_limit(20).hybrid(true))
            .with_index(|i| i.hnsw_m(32));

        assert_eq!(config.chunking.target_tokens, 1024);
        assert_eq!(config.search.default_limit, 20);
        assert!(config.search.hybrid_enabled);
        assert_eq!(config.index.hnsw_m, 32);
    }

    #[test]
    fn test_no_chunking() {
        let config = ChunkingConfig::no_chunking();
        assert_eq!(config.strategy, ChunkStrategy::None);
    }

    #[test]
    fn missing_default_mode_deserializes_as_dense() {
        let mut value = serde_json::to_value(SearchConfig::default()).expect("serialize");
        let removed = value
            .as_object_mut()
            .expect("object")
            .remove("default_mode");
        assert!(removed.is_some());
        let parsed: SearchConfig = serde_json::from_value(value).expect("deserialize");
        assert_eq!(parsed.default_mode, SearchMode::Dense);
    }
}
