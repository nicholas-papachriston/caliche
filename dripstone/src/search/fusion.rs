//! Score fusion strategies for combining dense and sparse retrieval results.

use crate::core::SearchResult;
use crate::errors::DripstoneError;
use crate::simd;
use crate::utils::numeric::usize_to_f32;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

/// Strategy for fusing dense and sparse search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion
    ReciprocalRank,
    /// Weighted sum of normalized scores
    WeightedSum,
}

/// Configuration for fusion operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    pub strategy: FusionStrategy,
    /// Weight for dense results (0.0-1.0), sparse gets (1.0 - alpha)
    pub alpha: f32,
    /// RRF k parameter (typically 60)
    pub rrf_k: f32,
}

impl FusionConfig {
    #[must_use]
    pub const fn new(strategy: FusionStrategy) -> Self {
        Self {
            strategy,
            alpha: 0.5,
            rrf_k: 60.0,
        }
    }

    #[must_use]
    pub const fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.clamp(0.0, 1.0);
        self
    }

    #[must_use]
    pub const fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k;
        self
    }

    /// Creates a config for RRF with default k=60.
    #[must_use]
    pub const fn reciprocal_rank() -> Self {
        Self::new(FusionStrategy::ReciprocalRank)
    }

    /// Creates a config for weighted sum with equal weights.
    #[must_use]
    pub const fn weighted_sum() -> Self {
        Self::new(FusionStrategy::WeightedSum)
    }
}

/// Fuses dense and sparse search results according to the specified strategy.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn fuse_search_results(
    dense_results: Vec<SearchResult>,
    sparse_results: Vec<SearchResult>,
    config: &FusionConfig,
) -> Result<Vec<SearchResult>, DripstoneError> {
    match config.strategy {
        FusionStrategy::ReciprocalRank => {
            reciprocal_rank_fusion(dense_results, sparse_results, config.rrf_k)
        }
        FusionStrategy::WeightedSum => weighted_fusion(dense_results, sparse_results, config.alpha),
    }
}

/// Fuses dense results with sparse scores given as a `chunk_id` -> score map.
/// Only chunks present in `dense_results` are included; sparse scores boost their ranking.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn fuse_dense_with_sparse_scores<S: BuildHasher>(
    dense_results: Vec<SearchResult>,
    sparse_scores: &HashMap<String, f32, S>,
    alpha: f32,
) -> Result<Vec<SearchResult>, DripstoneError> {
    let dense_normalized = normalize_scores(dense_results);

    if sparse_scores.is_empty() {
        return Ok(dense_normalized);
    }

    let sparse_scores_vec: Vec<f32> = sparse_scores.values().copied().collect();
    let (min_sparse, max_sparse) = simd::find_min_max(&sparse_scores_vec);
    let sparse_denom = if (max_sparse - min_sparse).abs() < f32::EPSILON {
        1.0
    } else {
        max_sparse - min_sparse
    };

    let sparse_weight = 1.0 - alpha;
    let mut results: Vec<SearchResult> = dense_normalized
        .into_iter()
        .map(|mut result| {
            let combined = if let Some(&s) = sparse_scores.get(&result.chunk.id) {
                let sparse_norm = (s - min_sparse) / sparse_denom;
                alpha.mul_add(result.score, sparse_weight * sparse_norm)
            } else {
                alpha * result.score
            };
            result.score = combined;
            result
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }

    Ok(results)
}

/// Reciprocal Rank Fusion (RRF) algorithm.
///
/// Combines rankings by computing: score = sum(1 / (k + rank)) for each result.
/// Each list contributes the first occurrence of a chunk id. Equal scores break
/// toward the smaller chunk id.
///
/// # Errors
///
/// Returns an error when `k` is not finite and greater than zero.
pub fn reciprocal_rank_fusion(
    dense_results: Vec<SearchResult>,
    sparse_results: Vec<SearchResult>,
    k: f32,
) -> Result<Vec<SearchResult>, DripstoneError> {
    if !k.is_finite() || k <= 0.0 {
        return Err(DripstoneError::configuration(
            "reciprocal rank fusion k must be finite and positive",
        ));
    }
    let mut scores: HashMap<String, (f32, SearchResult)> = HashMap::new();
    add_rrf_list(&mut scores, dense_results, k);
    add_rrf_list(&mut scores, sparse_results, k);
    Ok(rank_rrf_scores(scores))
}

fn add_rrf_list(
    scores: &mut HashMap<String, (f32, SearchResult)>,
    results: Vec<SearchResult>,
    k: f32,
) {
    let mut seen = HashSet::new();
    for (offset, result) in results.into_iter().enumerate() {
        if !seen.insert(result.chunk.id.clone()) {
            continue;
        }
        let rrf_score = 1.0 / (k + usize_to_f32(offset.saturating_add(1)));
        scores
            .entry(result.chunk.id.clone())
            .and_modify(|(score, _)| *score += rrf_score)
            .or_insert((rrf_score, result));
    }
}

fn rank_rrf_scores(scores: HashMap<String, (f32, SearchResult)>) -> Vec<SearchResult> {
    let mut results: Vec<_> = scores
        .into_iter()
        .map(|(_, (score, mut result))| {
            result.score = score;
            result
        })
        .collect();
    results.sort_by(by_rrf_rank);
    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }
    results
}

fn by_rrf_rank(left: &SearchResult, right: &SearchResult) -> std::cmp::Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.chunk.id.cmp(&right.chunk.id))
}

/// Weighted sum fusion with score normalization.
///
/// Combines normalized scores: final = alpha * `dense_norm` + (1-alpha) * `sparse_norm`
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn weighted_fusion(
    dense_results: Vec<SearchResult>,
    sparse_results: Vec<SearchResult>,
    alpha: f32,
) -> Result<Vec<SearchResult>, DripstoneError> {
    // Normalize scores to [0, 1]
    let dense_normalized = normalize_scores(dense_results);
    let sparse_normalized = normalize_scores(sparse_results);

    let mut scores: HashMap<String, (f32, SearchResult)> = HashMap::new();

    // Add weighted dense scores
    for result in dense_normalized {
        let weighted_score = alpha * result.score;
        scores.insert(result.chunk.id.clone(), (weighted_score, result));
    }

    // Add weighted sparse scores
    let sparse_weight = 1.0 - alpha;
    for result in sparse_normalized {
        let weighted_score = sparse_weight * result.score;
        scores
            .entry(result.chunk.id.clone())
            .and_modify(|(score, _)| *score += weighted_score)
            .or_insert((weighted_score, result));
    }

    // Convert to results and sort
    let mut results: Vec<_> = scores
        .into_iter()
        .map(|(_, (score, mut result))| {
            result.score = score;
            result
        })
        .collect();

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Update ranks
    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }

    Ok(results)
}

/// Normalizes scores to [0, 1] range using min-max normalization.
/// Uses SIMD-accelerated min/max finding when available.
fn normalize_scores(mut results: Vec<SearchResult>) -> Vec<SearchResult> {
    if results.is_empty() {
        return results;
    }

    // Extract scores into a vector for SIMD processing
    let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
    let (min_score, max_score) = simd::find_min_max(&scores);

    if (max_score - min_score).abs() < f32::EPSILON {
        // All scores are the same, set to 1.0
        for result in &mut results {
            result.score = 1.0;
        }
    } else {
        for result in &mut results {
            result.score = (result.score - min_score) / (max_score - min_score);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Chunk;

    fn create_result(id: &str, score: f32, rank: usize) -> SearchResult {
        SearchResult::new(
            Chunk::new(
                id.to_string(),
                0,
                "test".to_string(),
                vec![1.0],
                0,
                4,
                1,
                serde_json::json!({}),
                None,
            ),
            score,
            rank,
        )
    }

    #[test]
    fn test_rrf_basic() {
        let dense = vec![create_result("a", 0.9, 0), create_result("b", 0.8, 1)];
        let sparse = vec![create_result("b", 0.95, 0), create_result("c", 0.85, 1)];

        let results = reciprocal_rank_fusion(dense, sparse, 60.0)
            .expect("Failed to perform reciprocal rank fusion");

        // "b" should rank highest as it appears in both (id is formatted as doc_id#chunk_index)
        assert_eq!(results[0].chunk.doc_id, "b");
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_weighted_fusion() {
        let dense = vec![create_result("a", 0.9, 0), create_result("b", 0.5, 1)];
        let sparse = vec![create_result("b", 0.8, 0), create_result("c", 0.6, 1)];

        let results =
            weighted_fusion(dense, sparse, 0.7).expect("Failed to perform weighted fusion");

        assert!(!results.is_empty());
        // All scores should be in [0, 1] after normalization and weighting
        for result in &results {
            assert!(result.score >= 0.0 && result.score <= 1.0);
        }
    }

    #[test]
    fn test_normalize_scores() {
        let results = vec![
            create_result("a", 0.5, 0),
            create_result("b", 1.0, 1),
            create_result("c", 0.0, 2),
        ];

        let normalized = normalize_scores(results);

        assert!((normalized[0].score - 0.5).abs() <= f32::EPSILON);
        assert!((normalized[1].score - 1.0).abs() <= f32::EPSILON);
        assert!(normalized[2].score.abs() <= f32::EPSILON);
    }

    #[test]
    fn test_fusion_config() {
        let config = FusionConfig::reciprocal_rank().with_rrf_k(30.0);

        assert_eq!(config.strategy, FusionStrategy::ReciprocalRank);
        assert!((config.rrf_k - 30.0).abs() <= f32::EPSILON);
    }

    #[test]
    fn test_fuse_dense_with_sparse_scores() {
        let dense = vec![
            create_result("a", 0.9, 0),
            create_result("b", 0.7, 1),
            create_result("c", 0.5, 2),
        ];
        let mut sparse_scores = HashMap::new();
        sparse_scores.insert("a#0".to_string(), 0.2);
        sparse_scores.insert("b#0".to_string(), 1.0);
        sparse_scores.insert("c#0".to_string(), 0.5);

        let results = fuse_dense_with_sparse_scores(dense, &sparse_scores, 0.5)
            .expect("fusion should succeed");
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.score >= 0.0 && r.score <= 1.0);
        }
    }

    #[test]
    fn reciprocal_rank_uses_k_60() {
        let config = FusionConfig::reciprocal_rank();
        assert_eq!(config.rrf_k, 60.0);
        let results =
            reciprocal_rank_fusion(vec![create_result("a", 0.9, 0)], Vec::new(), config.rrf_k)
                .expect("rrf");
        assert_eq!(results[0].score, 1.0 / 61.0);
    }

    #[test]
    fn reciprocal_rank_rejects_non_positive_k() {
        for k in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let error = reciprocal_rank_fusion(vec![create_result("a", 0.9, 0)], Vec::new(), k)
                .expect_err("k must be positive");
            assert!(error.to_string().contains("k"), "{error}");
        }
    }

    #[test]
    fn reciprocal_rank_breaks_ties_by_chunk_id() {
        let dense = vec![create_result("b", 0.2, 0)];
        let sparse = vec![create_result("a", 0.9, 0)];
        let results = reciprocal_rank_fusion(dense, sparse, 60.0).expect("rrf");
        assert_eq!(results[0].score, results[1].score);
        assert_eq!(results[0].chunk.id, "a#0");
        assert_eq!(results[1].chunk.id, "b#0");
    }

    #[test]
    fn reciprocal_rank_counts_the_first_duplicate_on_each_list() {
        let dense = vec![
            create_result("a", 0.9, 0),
            create_result("a", 0.1, 1),
            create_result("b", 0.2, 2),
        ];
        let sparse = vec![create_result("a", 0.4, 0), create_result("a", 0.3, 1)];
        let forward = reciprocal_rank_fusion(dense.clone(), sparse.clone(), 60.0).expect("forward");
        let backward = reciprocal_rank_fusion(sparse, dense, 60.0).expect("backward");
        let forward_a = forward
            .iter()
            .find(|result| result.chunk.doc_id == "a")
            .expect("a");
        let backward_a = backward
            .iter()
            .find(|result| result.chunk.doc_id == "a")
            .expect("a");
        assert_eq!(forward_a.score, 2.0 / 61.0);
        assert_eq!(backward_a.score, forward_a.score);
    }

    #[test]
    fn fuse_search_results_uses_the_configured_rrf_k() {
        let config = FusionConfig::reciprocal_rank().with_rrf_k(10.0);
        let results = fuse_search_results(vec![create_result("a", 0.9, 0)], Vec::new(), &config)
            .expect("fuse");
        assert_eq!(results[0].score, 1.0 / 11.0);

        let infinite = FusionConfig::reciprocal_rank().with_rrf_k(f32::INFINITY);
        let error = fuse_search_results(vec![create_result("a", 0.9, 0)], Vec::new(), &infinite)
            .expect_err("infinite k");
        assert!(error.to_string().contains("finite"), "{error}");
    }

    #[test]
    fn reciprocal_rank_tie_order_is_independent_of_hash_order() {
        // At 2^30 the f32 ulp is 128, so every rank in this list shares one score.
        let k = (1_u32 << 30) as f32;
        assert_eq!(k + 8.0, k);
        let dense: Vec<_> = ["m", "c", "z", "a", "q", "b", "k", "d"]
            .into_iter()
            .map(|id| create_result(id, 0.2, 0))
            .collect();
        let results = reciprocal_rank_fusion(dense, Vec::new(), k).expect("rrf");
        let ids: Vec<_> = results
            .iter()
            .map(|result| result.chunk.id.as_str())
            .collect();
        assert_eq!(
            ids,
            ["a#0", "b#0", "c#0", "d#0", "k#0", "m#0", "q#0", "z#0"]
        );
        assert!(
            results
                .windows(2)
                .all(|pair| pair[0].score == pair[1].score)
        );
    }
}
