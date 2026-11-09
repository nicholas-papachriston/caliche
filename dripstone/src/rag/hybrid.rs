//! Hybrid search utilities for combining dense and sparse vector search.
//!
//! This module provides a helper for generating sparse vectors using BM25
//! for hybrid search capabilities.

#[cfg(feature = "rust-stemmers")]
use crate::search::{Bm25Scorer, SimpleTokenizer};
use crate::sparse::SparseVector;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

/// Helper for generating sparse vectors using BM25 for hybrid search.
///
/// This encapsulates BM25 scorer and tokenizer configuration, providing
/// a simple interface for fitting on a corpus and generating sparse vectors
/// for queries.
#[cfg(feature = "rust-stemmers")]
pub struct HybridSearchHelper {
    bm25: Arc<Mutex<Bm25Scorer>>,
    tokenizer: Arc<SimpleTokenizer>,
}

#[cfg(feature = "rust-stemmers")]
impl std::fmt::Debug for HybridSearchHelper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HybridSearchHelper")
            .field("is_fitted", &self.is_fitted())
            .finish()
    }
}

#[cfg(feature = "rust-stemmers")]
impl HybridSearchHelper {
    /// Create a new hybrid search helper with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bm25: Arc::new(Mutex::new(Bm25Scorer::new())),
            tokenizer: Arc::new(SimpleTokenizer::new().with_english_stopwords()),
        }
    }

    /// Create a new hybrid search helper with a custom tokenizer.
    #[must_use]
    pub fn with_tokenizer(tokenizer: SimpleTokenizer) -> Self {
        Self {
            bm25: Arc::new(Mutex::new(Bm25Scorer::new())),
            tokenizer: Arc::new(tokenizer),
        }
    }

    /// Fit the BM25 model on a corpus of documents.
    ///
    /// This builds the vocabulary and computes document frequencies (IDF statistics).
    /// Should be called after indexing documents to enable sparse vector generation.
    pub fn fit(&self, documents: &[String]) {
        self.fit_dedup(documents, false);
    }

    /// Fit the BM25 model on a corpus of documents, optionally skipping duplicates by content hash.
    ///
    /// When `dedup` is true, chunk texts with the same content hash are excluded from
    /// vocabulary/IDF computation after the first occurrence.
    pub fn fit_dedup(&self, documents: &[String], dedup: bool) {
        if documents.is_empty() {
            return;
        }

        let mut scorer = self.bm25.lock();
        if dedup {
            scorer.fit_with_dedup(documents, &*self.tokenizer);
        } else {
            scorer.fit(documents, &*self.tokenizer);
        }
    }

    /// Fit the BM25 model on a line-level corpus.
    ///
    /// Each entry is (`chunk_id`, lines). Enables `score_chunks_by_lines` for line-aware scoring.
    pub fn fit_lines(&self, chunk_lines: &[(String, Vec<String>)]) {
        if chunk_lines.is_empty() {
            return;
        }

        let mut scorer = self.bm25.lock();
        scorer.fit_lines(chunk_lines, &*self.tokenizer);
    }

    /// Score chunks using line-level BM25 and aggregate by `chunk_id`.
    ///
    /// Returns `HashMap`<`chunk_id`, score>. Only available when `fit_lines` was used.
    #[must_use]
    pub fn score_chunks_by_lines(&self, query: &str) -> HashMap<String, f32> {
        let scorer = self.bm25.lock();
        let per_line = scorer.score_line_documents(query, &*self.tokenizer);
        drop(scorer);
        let mut by_chunk: HashMap<String, f32> = HashMap::new();
        for (chunk_id, score) in per_line {
            *by_chunk.entry(chunk_id).or_insert(0.0) += score;
        }
        by_chunk
    }

    /// Check if line-level BM25 corpus is available (`fit_lines` was called).
    #[must_use]
    pub fn has_line_corpus(&self) -> bool {
        let scorer = self.bm25.lock();
        scorer.has_line_corpus()
    }

    /// Generate a sparse vector for a query string.
    ///
    /// Returns `None` if the BM25 model hasn't been fitted yet (vocabulary is empty).
    #[must_use]
    pub fn generate_sparse_vector(&self, query: &str) -> Option<SparseVector> {
        let scorer = self.bm25.lock();

        // Check if model has been fitted
        if scorer.vocab_size() == 0 {
            return None;
        }

        Some(scorer.transform(query, &*self.tokenizer))
    }

    /// Check if the BM25 model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        let scorer = self.bm25.lock();
        scorer.vocab_size() > 0
    }
}

#[cfg(feature = "rust-stemmers")]
impl Default for HybridSearchHelper {
    fn default() -> Self {
        Self::new()
    }
}

/// Stub implementation when rust-stemmers feature is not enabled.
#[cfg(not(feature = "rust-stemmers"))]
pub struct HybridSearchHelper;

#[cfg(not(feature = "rust-stemmers"))]
impl HybridSearchHelper {
    pub fn new() -> Self {
        Self
    }

    pub fn with_tokenizer(_tokenizer: crate::search::SimpleTokenizer) -> Self {
        Self
    }

    pub fn fit(&self, _documents: &[String]) {
        // No-op when feature is disabled
    }

    pub fn fit_dedup(&self, _documents: &[String], _dedup: bool) {
        // No-op when feature is disabled
    }

    pub fn fit_lines(&self, _chunk_lines: &[(String, Vec<String>)]) {
        // No-op when feature is disabled
    }

    pub fn score_chunks_by_lines(&self, _query: &str) -> HashMap<String, f32> {
        HashMap::new()
    }

    pub fn has_line_corpus(&self) -> bool {
        false
    }

    pub fn generate_sparse_vector(&self, _query: &str) -> Option<SparseVector> {
        None
    }

    pub fn is_fitted(&self) -> bool {
        false
    }
}

#[cfg(not(feature = "rust-stemmers"))]
impl Default for HybridSearchHelper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "rust-stemmers")]
    #[test]
    fn test_hybrid_search_helper() {
        let helper = HybridSearchHelper::new();
        assert!(!helper.is_fitted());

        let documents = vec![
            "the quick brown fox".to_string(),
            "quick brown dog".to_string(),
            "the lazy dog".to_string(),
        ];

        helper.fit(&documents);
        assert!(helper.is_fitted());

        let sparse = helper.generate_sparse_vector("quick fox");
        assert!(sparse.is_some());
        let sparse = sparse.expect("sparse vector should be generated after fitting");
        assert!(sparse.nnz() > 0);
    }

    #[cfg(feature = "rust-stemmers")]
    #[test]
    fn test_hybrid_search_helper_not_fitted() {
        let helper = HybridSearchHelper::new();
        assert!(!helper.is_fitted());

        let sparse = helper.generate_sparse_vector("test query");
        assert!(sparse.is_none());
    }
}
