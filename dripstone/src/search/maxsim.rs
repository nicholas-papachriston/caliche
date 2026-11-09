//! Late-interaction (ColBERT-style) MaxSim scoring and reranking.

use std::collections::HashSet;
use std::sync::Arc;

use crate::core::{Chunk, Metric, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::search::tokenizer::{SimpleTokenizer, Tokenizer};
use crate::simd::DenseMetric;

/// Default cap on token vectors per chunk or query for late interaction.
pub const DEFAULT_MAX_TOKEN_VECTORS: usize = 32;

/// Tokenize text for ColBERT-style MaxSim (unique, first-seen, length-capped).
///
/// Uses [`SimpleTokenizer`] without stopword removal so content tokens stay.
/// Empty input after tokenize falls back to the trimmed original string.
#[must_use]
pub fn tokens_for_late_interaction(text: &str, max_tokens: usize) -> Vec<String> {
    let cap = max_tokens.max(1);
    let tokenizer = SimpleTokenizer::new();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for token in tokenizer.tokenize(text) {
        if token.is_empty() || !seen.insert(token.clone()) {
            continue;
        }
        out.push(token);
        if out.len() >= cap {
            break;
        }
    }
    if out.is_empty() {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_ascii_lowercase());
        }
    }
    out
}

/// Token / passage multi-vectors for late interaction.
///
/// Outer length = number of tokens; each inner slice is the token embedding dim.
pub type MultiVectors = Arc<[Arc<[f32]>]>;

/// Options controlling MaxSim scoring during search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxSimOptions {
    /// Pairwise similarity used inside MaxSim (`Cosine` or `DotProduct`).
    pub metric: Metric,
    /// ANN/dense overfetch multiplier before MaxSim (`candidates = limit * overfetch`).
    pub overfetch: usize,
    /// When true, fuse dense score with MaxSim (`dense_weight` on dense).
    pub fuse_with_dense: bool,
    /// Dense weight in `[0, 1]` when [`Self::fuse_with_dense`] is set.
    pub dense_weight: f32,
}

impl Default for MaxSimOptions {
    fn default() -> Self {
        Self {
            metric: Metric::DotProduct,
            overfetch: 5,
            fuse_with_dense: false,
            dense_weight: 0.3,
        }
    }
}

impl MaxSimOptions {
    #[must_use]
    pub const fn new(metric: Metric) -> Self {
        Self {
            metric,
            overfetch: 5,
            fuse_with_dense: false,
            dense_weight: 0.3,
        }
    }

    #[must_use]
    pub fn overfetch(mut self, overfetch: usize) -> Self {
        self.overfetch = overfetch.max(1);
        self
    }

    #[must_use]
    pub const fn fuse_with_dense(mut self, dense_weight: f32) -> Self {
        self.fuse_with_dense = true;
        self.dense_weight = dense_weight;
        self
    }
}

fn dense_metric(metric: Metric) -> Result<DenseMetric, DripstoneError> {
    match metric {
        Metric::Cosine => Ok(DenseMetric::Cosine),
        Metric::DotProduct => Ok(DenseMetric::Dot),
        Metric::Euclidean | Metric::Manhattan => Err(DripstoneError::pipeline(
            "MaxSim only supports Cosine or DotProduct pairwise metrics",
        )),
    }
}

/// `MaxSim(Q, D) = sum_q max_d sim(q, d)`.
///
/// # Errors
///
/// Returns when the metric is unsupported or pairwise scoring fails.
pub fn maxsim_score(
    query_tokens: &[Arc<[f32]>],
    doc_tokens: &[Arc<[f32]>],
    metric: Metric,
) -> Result<f32, DripstoneError> {
    if query_tokens.is_empty() || doc_tokens.is_empty() {
        return Ok(0.0);
    }
    let dense = dense_metric(metric)?;
    let doc_refs: Vec<&[f32]> = doc_tokens.iter().map(std::convert::AsRef::as_ref).collect();
    let mut total = 0.0_f32;
    for q in query_tokens {
        let prepared = crate::simd::PreparedQueryScorer::new(q.as_ref(), dense)?;
        let scores = prepared.score_dense_batch(&doc_refs)?;
        let best = scores.into_iter().fold(f32::NEG_INFINITY, f32::max);
        if best.is_finite() {
            total += best;
        }
    }
    Ok(total)
}

/// Build [`MultiVectors`] from owned token vectors.
#[must_use]
pub fn multi_vectors_from_owned(tokens: Vec<Vec<f32>>) -> MultiVectors {
    let arcs: Vec<Arc<[f32]>> = tokens.into_iter().map(|v| Arc::from(v)).collect();
    Arc::from(arcs)
}

/// Mean-pool token vectors into a single routing vector (ANN first stage).
#[must_use]
pub fn pooled_vector_from_multi(tokens: &MultiVectors) -> Arc<[f32]> {
    if tokens.is_empty() {
        return Arc::from([]);
    }
    let dim = tokens[0].len();
    let mut acc = vec![0.0_f32; dim];
    let mut n = 0_f32;
    for token in tokens.iter() {
        if token.len() != dim {
            continue;
        }
        for (a, b) in acc.iter_mut().zip(token.iter()) {
            *a += *b;
        }
        n += 1.0;
    }
    if n > 0.0 {
        for a in &mut acc {
            *a /= n;
        }
    }
    Arc::from(acc)
}

/// Candidate limit for dense/ANN stage before MaxSim rerank.
#[must_use]
pub fn overfetch_limit(query: &SearchQuery) -> usize {
    let overfetch = query.maxsim.map_or(1, |o| o.overfetch.max(1));
    query.limit.max(1).saturating_mul(overfetch)
}

/// Rerank dense candidates with late-interaction MaxSim when requested.
///
/// # Errors
///
/// Returns when late-interaction scoring fails.
pub fn rerank_with_maxsim(
    mut dense_results: Vec<SearchResult>,
    query: &SearchQuery,
) -> Result<Vec<SearchResult>, DripstoneError> {
    let Some(query_tokens) = query.query_multi_vectors.as_ref() else {
        return Ok(dense_results);
    };
    let opts = query.maxsim.unwrap_or_default();
    let dense_weight = opts.dense_weight.clamp(0.0, 1.0);

    for result in &mut dense_results {
        let Some(doc_tokens) = result.chunk.multi_vectors.as_ref() else {
            if !opts.fuse_with_dense {
                result.score = f32::NEG_INFINITY;
            }
            continue;
        };
        let maxsim = maxsim_score(query_tokens, doc_tokens, opts.metric)?;
        if opts.fuse_with_dense {
            result.score = dense_weight * result.score + (1.0 - dense_weight) * maxsim;
        } else {
            result.score = maxsim;
        }
    }

    dense_results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    dense_results.truncate(query.limit.max(1));
    for (rank, result) in dense_results.iter_mut().enumerate() {
        result.rank = rank;
    }
    Ok(dense_results)
}

/// Exact late-interaction search over in-memory chunks (correctness / microbench path).
///
/// # Errors
///
/// Returns when scoring fails.
pub fn exact_maxsim_search(
    chunks: &[Chunk],
    query: &SearchQuery,
) -> Result<Vec<SearchResult>, DripstoneError> {
    let Some(query_tokens) = query.query_multi_vectors.as_ref() else {
        return Err(DripstoneError::pipeline(
            "exact_maxsim_search requires query_multi_vectors",
        ));
    };
    let opts = query.maxsim.unwrap_or_default();
    let mut results = Vec::new();
    for chunk in chunks {
        if let Some(ref ns) = query.namespace
            && chunk.namespace.as_deref() != Some(ns.as_str())
        {
            continue;
        }
        let Some(doc_tokens) = chunk.multi_vectors.as_ref() else {
            continue;
        };
        let score = maxsim_score(query_tokens, doc_tokens, opts.metric)?;
        results.push(SearchResult::new(chunk.clone(), score, 0));
    }
    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(query.limit.max(1));
    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maxsim_prefers_aligned_token_sets() {
        let q = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let good = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let bad = multi_vectors_from_owned(vec![vec![-1.0, 0.0], vec![0.0, -1.0]]);
        let s_good = maxsim_score(&q, &good, Metric::DotProduct).expect("score");
        let s_bad = maxsim_score(&q, &bad, Metric::DotProduct).expect("score");
        assert!(s_good > s_bad);
    }

    #[test]
    fn pooled_vector_means_tokens() {
        let mv = multi_vectors_from_owned(vec![vec![2.0, 0.0], vec![0.0, 2.0]]);
        let pooled = pooled_vector_from_multi(&mv);
        assert!((pooled[0] - 1.0).abs() < 1e-5);
        assert!((pooled[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn tokens_for_late_interaction_dedupes_and_caps() {
        let tokens = tokens_for_late_interaction("Alpha alpha BETA beta gamma", 3);
        assert_eq!(tokens, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn tokens_for_late_interaction_falls_back_to_trimmed_text() {
        let tokens = tokens_for_late_interaction("   ", 8);
        assert!(tokens.is_empty());
        let tokens = tokens_for_late_interaction("!!!", 8);
        assert_eq!(tokens, vec!["!!!"]);
    }
}
