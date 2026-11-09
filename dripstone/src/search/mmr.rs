//! Maximal Marginal Relevance (MMR) diversity reranking for search results.
//!
//! Balances relevance with diversity by iteratively selecting results that
//! maximize: lambda * relevance - (1 - lambda) * `max_similarity_to_selected`.

use crate::core::SearchResult;
use crate::simd;
use crate::utils::numeric::usize_to_f64;
use std::collections::HashSet;
use std::hash::BuildHasher;

/// Similarity function used for MMR diversity computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MmrSimilarity {
    /// Jaccard token-set similarity (fast, no embeddings needed).
    Jaccard,
    /// Cosine similarity on embedding vectors (more accurate, requires vectors).
    Cosine,
}

/// Configuration for MMR re-ranking.
#[derive(Debug, Clone)]
pub struct MmrConfig {
    /// Whether MMR is enabled.
    pub enabled: bool,
    /// Lambda in [0.0, 1.0]: 0.0 = max diversity, 1.0 = max relevance. Default: 0.7.
    pub lambda: f64,
    /// Similarity function for diversity.
    pub similarity: MmrSimilarity,
}

impl Default for MmrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            lambda: 0.7,
            similarity: MmrSimilarity::Jaccard,
        }
    }
}

/// Tokenize text for Jaccard similarity: alphanumeric tokens, lowercased.
#[must_use]
pub fn tokenize(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Jaccard similarity between two token sets. Returns a value in [0.0, 1.0].
#[must_use]
pub fn jaccard_similarity<S: BuildHasher>(a: &HashSet<String, S>, b: &HashSet<String, S>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    usize_to_f64(intersection) / usize_to_f64(union)
}

/// Cosine similarity between two embedding vectors. Returns a value in [-1.0, 1.0].
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    simd::cosine_similarity(a, b).map_or(0.0, |score| f64::from(score).clamp(-1.0, 1.0))
}

/// MMR score: lambda * relevance - (1 - lambda) * `max_similarity_to_selected`.
#[must_use]
pub fn mmr_score(relevance: f64, max_similarity: f64, lambda: f64) -> f64 {
    lambda.mul_add(relevance, -((1.0 - lambda) * max_similarity))
}

/// Re-rank search results using MMR. Iteratively selects items balancing
/// relevance with diversity. Returns a new vec of `SearchResult` with updated ranks.
pub fn mmr_rerank(results: &[SearchResult], config: &MmrConfig) -> Vec<SearchResult> {
    if !config.enabled || results.is_empty() {
        return results.to_vec();
    }

    let lambda = config.lambda.clamp(0.0, 1.0);

    // Normalize scores to [0.0, 1.0] using min-max
    let scores: Vec<f64> = {
        let min_s = results
            .iter()
            .map(|r| r.score)
            .fold(f32::INFINITY, f32::min);
        let max_s = results
            .iter()
            .map(|r| r.score)
            .fold(f32::NEG_INFINITY, f32::max);
        let range = max_s - min_s;
        if range < 1e-10f32 {
            results.iter().map(|_| 1.0f64).collect()
        } else {
            results
                .iter()
                .map(|r| f64::from((r.score - min_s) / range))
                .collect()
        }
    };

    match config.similarity {
        MmrSimilarity::Jaccard => mmr_rerank_jaccard(results, &scores, lambda),
        MmrSimilarity::Cosine => mmr_rerank_cosine(results, &scores, lambda),
    }
}

fn mmr_rerank_jaccard(results: &[SearchResult], scores: &[f64], lambda: f64) -> Vec<SearchResult> {
    let token_sets: Vec<HashSet<String>> =
        results.iter().map(|r| tokenize(&r.chunk.content)).collect();

    let mut selected = Vec::with_capacity(results.len());
    let mut remaining: Vec<usize> = (0..results.len()).collect();

    while !remaining.is_empty() {
        let mut best_idx = remaining[0];
        let mut best_mmr = {
            let relevance = scores[best_idx];
            let max_sim = selected
                .iter()
                .map(|&s_idx| jaccard_similarity(&token_sets[s_idx], &token_sets[best_idx]))
                .fold(0.0f64, f64::max);
            mmr_score(relevance, max_sim, lambda)
        };
        for &idx in remaining.iter().skip(1) {
            let relevance = scores[idx];
            let max_sim = selected
                .iter()
                .map(|&s_idx| jaccard_similarity(&token_sets[s_idx], &token_sets[idx]))
                .fold(0.0f64, f64::max);
            let mmr = mmr_score(relevance, max_sim, lambda);
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = idx;
            }
        }

        remaining.retain(|&i| i != best_idx);
        selected.push(best_idx);
    }

    selected
        .into_iter()
        .enumerate()
        .map(|(rank, idx)| {
            let mut r = results[idx].clone();
            r.rank = rank;
            r
        })
        .collect()
}

fn mmr_rerank_cosine(results: &[SearchResult], scores: &[f64], lambda: f64) -> Vec<SearchResult> {
    let mut vectors = Vec::with_capacity(results.len());
    for result in results {
        vectors.push(result.chunk.vector.as_ref());
    }
    let dim = vectors.first().map_or(0, |v| v.len());
    if dim == 0 {
        return results.to_vec();
    }

    let mut selected = Vec::with_capacity(results.len());
    let mut remaining: Vec<usize> = (0..results.len()).collect();

    while !remaining.is_empty() {
        let mut best_idx = remaining[0];
        let mut best_mmr = {
            let relevance = scores[best_idx];
            let mut max_sim = 0.0_f64;
            for &selected_idx in &selected {
                max_sim = max_sim.max(cosine_similarity(vectors[selected_idx], vectors[best_idx]));
            }
            mmr_score(relevance, max_sim, lambda)
        };
        for &idx in &remaining[1..] {
            let relevance = scores[idx];
            let mut max_sim = 0.0_f64;
            for &selected_idx in &selected {
                max_sim = max_sim.max(cosine_similarity(vectors[selected_idx], vectors[idx]));
            }
            let mmr = mmr_score(relevance, max_sim, lambda);
            if mmr > best_mmr {
                best_mmr = mmr;
                best_idx = idx;
            }
        }

        remaining.retain(|&i| i != best_idx);
        selected.push(best_idx);
    }

    selected
        .into_iter()
        .enumerate()
        .map(|(rank, idx)| {
            let mut r = results[idx].clone();
            r.rank = rank;
            r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize() {
        let t = tokenize("Hello, world! Hello again.");
        assert!(t.contains("hello"));
        assert!(t.contains("world"));
        assert!(t.contains("again"));
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn test_jaccard() {
        let a: HashSet<String> = ["a", "b", "c"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        let b: HashSet<String> = ["b", "c", "d"]
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_cosine() {
        let a: Vec<f32> = vec![1.0, 0.0, 0.0];
        let b: Vec<f32> = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
        let c: Vec<f32> = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_mmr_score() {
        assert!((mmr_score(1.0, 0.0, 0.7) - 0.7).abs() < 1e-6);
        assert!(mmr_score(0.5, 0.8, 0.7) < 0.7);
    }
}
