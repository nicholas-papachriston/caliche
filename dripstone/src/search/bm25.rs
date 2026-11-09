//! BM25 scoring algorithm for text-based retrieval.

use crate::search::tokenizer::Tokenizer;
use crate::sparse::SparseVector;
use crate::utils::numeric::usize_to_f32;
use std::collections::{HashMap, HashSet};

/// Configuration for BM25 scoring.
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// Controls term frequency saturation (typically 1.2-2.0).
    pub k1: f32,
    /// Controls document length normalization (typically 0.75).
    pub b: f32,
}

impl Bm25Config {
    /// Creates a new BM25 configuration with standard parameters.
    #[must_use]
    pub const fn new() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }

    #[must_use]
    pub const fn with_k1(mut self, k1: f32) -> Self {
        self.k1 = k1;
        self
    }

    #[must_use]
    pub const fn with_b(mut self, b: f32) -> Self {
        self.b = b;
        self
    }
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self::new()
    }
}

/// BM25 scorer for converting text into sparse vectors and computing relevance scores.
pub struct Bm25Scorer {
    config: Bm25Config,
    /// Number of documents in the corpus
    num_docs: usize,
    /// Average document length in the corpus
    avg_doc_length: f32,
    /// Document frequency for each term (term -> number of docs containing it)
    doc_frequencies: HashMap<String, usize>,
    /// Vocabulary mapping terms to indices
    vocabulary: HashMap<String, u32>,
    /// Line-level corpus for `score_chunks_by_lines`: (`chunk_id`, `line_text`) per line.
    /// Set when `fit_lines` is used; cleared when `fit/fit_with_dedup` is used.
    line_corpus: Option<Vec<(String, String)>>,
}

impl Bm25Scorer {
    /// Creates a new BM25 scorer with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(Bm25Config::default())
    }

    /// Creates a BM25 scorer with custom configuration.
    #[must_use]
    pub fn with_config(config: Bm25Config) -> Self {
        Self {
            config,
            num_docs: 0,
            avg_doc_length: 0.0,
            doc_frequencies: HashMap::new(),
            vocabulary: HashMap::new(),
            line_corpus: None,
        }
    }

    /// Fits the BM25 model on a corpus of documents.
    ///
    /// This builds the vocabulary and computes document frequencies (IDF statistics).
    /// Clears any line-level corpus from a previous `fit_lines` call.
    pub fn fit<T: Tokenizer>(&mut self, documents: &[String], tokenizer: &T) {
        self.line_corpus = None;
        self.fit_impl(documents.iter().map(std::string::String::as_str), tokenizer);
    }

    /// Fits the BM25 model with content-hash deduplication.
    ///
    /// Skips duplicate documents (by hashline chunk hash) so vocabulary and IDF
    /// are not inflated by repeated content. First occurrence of each hash is included.
    /// Clears any line-level corpus from a previous `fit_lines` call.
    pub fn fit_with_dedup<T: Tokenizer>(&mut self, documents: &[String], tokenizer: &T) {
        self.line_corpus = None;
        let options = crate::hashline::HashlineOptions::default();
        let mut seen = HashSet::new();
        let deduped: Vec<&str> = documents
            .iter()
            .filter_map(|doc| {
                let hash = crate::hashline::compute_chunk_hash(doc, &options);
                if seen.insert(hash) {
                    Some(doc.as_str())
                } else {
                    None
                }
            })
            .collect();
        self.fit_impl(deduped.iter().copied(), tokenizer);
    }

    /// Fits the BM25 model on a line-level corpus.
    ///
    /// Each line is treated as a "document" for vocabulary and IDF computation.
    /// Stores the line corpus for later use by `score_line_documents`.
    /// Each entry is (`chunk_id`, lines) where `chunk_id` identifies the chunk for aggregation.
    pub fn fit_lines<T: Tokenizer>(
        &mut self,
        chunk_lines: &[(String, Vec<String>)],
        tokenizer: &T,
    ) {
        let mut line_corpus: Vec<(String, String)> = Vec::new();
        let mut line_texts: Vec<&str> = Vec::new();
        for (chunk_id, lines) in chunk_lines {
            for line in lines {
                if !line.is_empty() {
                    line_corpus.push((chunk_id.clone(), line.clone()));
                    line_texts.push(line.as_str());
                }
            }
        }
        self.line_corpus = Some(line_corpus);
        self.fit_impl(line_texts, tokenizer);
    }

    fn fit_impl<'a, T: Tokenizer, I>(&mut self, documents: I, tokenizer: &T)
    where
        I: IntoIterator<Item = &'a str>,
    {
        // Always reset prior state so repeated fit() calls are idempotent and
        // fit(&[]) produces a clean, empty corpus rather than keeping stale
        // vocab and IDF from a previous fit.
        self.vocabulary.clear();
        self.doc_frequencies.clear();
        self.avg_doc_length = 0.0;

        let docs: Vec<&str> = documents.into_iter().collect();
        self.num_docs = docs.len();
        if self.num_docs == 0 {
            return;
        }

        let mut total_length = 0_usize;
        let mut term_doc_sets: HashMap<String, HashSet<usize>> = HashMap::new();

        for (doc_idx, doc) in docs.iter().enumerate() {
            let tokens = tokenizer.tokenize(doc);
            total_length = total_length.saturating_add(tokens.len());

            let mut seen_terms = HashSet::new();
            for token in tokens {
                if seen_terms.insert(token.clone()) {
                    term_doc_sets.entry(token).or_default().insert(doc_idx);
                }
            }
        }

        self.avg_doc_length = usize_to_f32(total_length) / usize_to_f32(self.num_docs);

        for (next_idx, (term, doc_set)) in term_doc_sets.into_iter().enumerate() {
            let df = doc_set.len();
            self.doc_frequencies.insert(term.clone(), df);
            let Ok(next_idx) = u32::try_from(next_idx) else {
                break;
            };
            self.vocabulary.insert(term, next_idx);
        }
    }

    /// Converts text into a BM25-weighted sparse vector.
    pub fn transform<T: Tokenizer>(&self, text: &str, tokenizer: &T) -> SparseVector {
        let tokens = tokenizer.tokenize(text);
        let doc_length = usize_to_f32(tokens.len());

        // Count term frequencies in this document
        let mut term_freqs: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *term_freqs.entry(token).or_insert(0) += 1;
        }

        let mut sparse = SparseVector::new();

        for (term, tf) in &term_freqs {
            if let Some(&term_idx) = self.vocabulary.get(term) {
                let score = self.compute_term_score(term, *tf, doc_length);
                sparse.set(term_idx, score);
            }
        }

        sparse
    }

    /// Computes BM25 score for a term in a document.
    fn compute_term_score(&self, term: &str, tf: usize, doc_length: f32) -> f32 {
        let df = self.doc_frequencies.get(term).copied().unwrap_or(1);

        // IDF component: log((N - df + 0.5) / (df + 0.5) + 1)
        let idf = (usize_to_f32(self.num_docs - df) + 0.5) / (usize_to_f32(df) + 0.5) + 1.0;
        let idf = idf.ln();

        // TF component with length normalization
        let normalized_tf = usize_to_f32(tf)
            / self
                .config
                .b
                .mul_add(doc_length / self.avg_doc_length, 1.0 - self.config.b);

        // BM25 formula

        idf * (normalized_tf * (self.config.k1 + 1.0)) / (normalized_tf + self.config.k1)
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocabulary.len()
    }

    /// Returns the term index for a given term, if it exists in the vocabulary.
    #[must_use]
    pub fn term_index(&self, term: &str) -> Option<u32> {
        self.vocabulary.get(term).copied()
    }

    /// Computes BM25 score for a single document (e.g. a line) given a query.
    fn compute_document_score<T: Tokenizer>(&self, doc: &str, query: &str, tokenizer: &T) -> f32 {
        let doc_tokens = tokenizer.tokenize(doc);
        if doc_tokens.is_empty() {
            return 0.0;
        }
        let doc_length = usize_to_f32(doc_tokens.len());

        let mut doc_tf: HashMap<String, usize> = HashMap::new();
        for token in doc_tokens {
            *doc_tf.entry(token).or_insert(0) += 1;
        }

        let query_tokens = tokenizer.tokenize(query);
        let mut score = 0.0;
        for term in &query_tokens {
            if self.vocabulary.contains_key(term)
                && let Some(&tf) = doc_tf.get(term)
            {
                score += self.compute_term_score(term, tf, doc_length);
            }
        }
        score
    }

    /// Returns true if a line-level corpus is stored (from `fit_lines`).
    #[must_use]
    pub const fn has_line_corpus(&self) -> bool {
        self.line_corpus.is_some()
    }

    /// Scores each line in the stored line corpus against the query.
    /// Returns (`chunk_id`, score) per line. Caller aggregates by `chunk_id`.
    pub fn score_line_documents<T: Tokenizer>(
        &self,
        query: &str,
        tokenizer: &T,
    ) -> Vec<(String, f32)> {
        let Some(ref corpus) = self.line_corpus else {
            return Vec::new();
        };
        corpus
            .iter()
            .map(|(chunk_id, line)| {
                let score = self.compute_document_score(line, query, tokenizer);
                (chunk_id.clone(), score)
            })
            .collect()
    }
}

impl Default for Bm25Scorer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::tokenizer::SimpleTokenizer;

    #[test]
    fn test_bm25_basic() {
        let documents = vec![
            "the quick brown fox".to_string(),
            "quick brown dog".to_string(),
            "the lazy dog".to_string(),
        ];

        let tokenizer = SimpleTokenizer::new();
        let mut scorer = Bm25Scorer::new();
        scorer.fit(&documents, &tokenizer);

        assert!(scorer.vocab_size() > 0);

        let sparse = scorer.transform("quick fox", &tokenizer);
        assert!(sparse.nnz() > 0);
    }

    #[test]
    fn test_bm25_config() {
        let config = Bm25Config::new().with_k1(2.0).with_b(0.5);

        assert!((config.k1 - 2.0).abs() <= f32::EPSILON);
        assert!((config.b - 0.5).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_bm25_empty_corpus() {
        let documents: Vec<String> = vec![];
        let tokenizer = SimpleTokenizer::new();
        let mut scorer = Bm25Scorer::new();
        scorer.fit(&documents, &tokenizer);

        assert_eq!(scorer.vocab_size(), 0);
    }

    #[test]
    fn test_bm25_fit_empty_after_fit_clears_vocabulary() {
        let tokenizer = SimpleTokenizer::new();
        let mut scorer = Bm25Scorer::new();
        let documents = vec!["the quick brown fox".to_string()];
        scorer.fit(&documents, &tokenizer);
        assert!(scorer.vocab_size() > 0);

        let empty: Vec<String> = Vec::new();
        scorer.fit(&empty, &tokenizer);
        assert_eq!(
            scorer.vocab_size(),
            0,
            "fit(&[]) must clear vocab from prior fit"
        );

        let sparse = scorer.transform("quick fox", &tokenizer);
        assert_eq!(
            sparse.nnz(),
            0,
            "transform after fit(empty) must produce an empty sparse vector"
        );
    }

    #[test]
    fn test_bm25_fit_with_dedup() {
        let duplicate = "the quick brown fox".to_string();
        let documents = vec![
            duplicate.clone(),
            "quick brown dog".to_string(),
            duplicate.clone(),
            "the lazy dog".to_string(),
            duplicate,
        ];

        let tokenizer = SimpleTokenizer::new();

        let mut scorer_no_dedup = Bm25Scorer::new();
        scorer_no_dedup.fit(&documents, &tokenizer);
        let vocab_no_dedup = scorer_no_dedup.vocab_size();

        let mut scorer_dedup = Bm25Scorer::new();
        scorer_dedup.fit_with_dedup(&documents, &tokenizer);
        let vocab_dedup = scorer_dedup.vocab_size();

        assert!(vocab_dedup <= vocab_no_dedup);
        assert!(scorer_dedup.vocab_size() > 0);

        let sparse = scorer_dedup.transform("quick fox", &tokenizer);
        assert!(sparse.nnz() > 0);
    }

    #[test]
    fn test_bm25_fit_lines_and_score() {
        let tokenizer = SimpleTokenizer::new();
        let mut scorer = Bm25Scorer::new();
        let chunk_lines = vec![
            (
                "doc1#0".to_string(),
                vec!["hello world".to_string(), "foo bar".to_string()],
            ),
            (
                "doc1#1".to_string(),
                vec!["world peace".to_string(), "bar baz".to_string()],
            ),
        ];
        scorer.fit_lines(&chunk_lines, &tokenizer);
        assert!(scorer.vocab_size() > 0);
        assert!(scorer.has_line_corpus());

        let per_line = scorer.score_line_documents("world bar", &tokenizer);
        assert!(!per_line.is_empty());
        let chunk_scores: std::collections::HashMap<String, f32> =
            per_line
                .into_iter()
                .fold(std::collections::HashMap::new(), |mut acc, (id, s)| {
                    *acc.entry(id).or_insert(0.0) += s;
                    acc
                });
        assert!(chunk_scores.get("doc1#0").copied().unwrap_or(0.0) > 0.0);
        assert!(chunk_scores.get("doc1#1").copied().unwrap_or(0.0) > 0.0);
    }
}
