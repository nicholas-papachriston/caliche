//! Search-time centroid probing and candidate scoring.
//!
//! Filtered ANN intersects [`super::ClusterPostingState`] cluster bitmaps, probes
//! centroids, then scores records in those clusters. This scan is not a document
//! store: it does not widen into per-chunk postings or chunk envelopes.

use super::{CentroidIndex, ClusterPostingState, dense_metric, score_to_distance};
use crate::embedding::{EmbeddingRecord, SimilarityMetric, SimilarityResult};
use crate::errors::DripstoneError;
use crate::index::IndexSearchRequest;
use crate::simd;
use roaring::RoaringBitmap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

impl CentroidIndex {
    /// Find the k nearest centroids to a query vector under the given metric.
    pub(crate) fn find_nearest_centroids(
        &self,
        query: &[f32],
        k: usize,
        allowed_clusters: Option<&RoaringBitmap>,
        metric: SimilarityMetric,
    ) -> Result<Vec<usize>, DripstoneError> {
        let centroids = self.centroids.read();
        if centroids.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let prepared = simd::PreparedQueryScorer::new(query, dense_metric(metric))?;
        let mut centroid_refs = Vec::with_capacity(centroids.len());
        for centroid in centroids.iter() {
            centroid_refs.push(centroid.as_slice());
        }
        let raw_scores = prepared.score_dense_batch(&centroid_refs)?;
        let mut distances = Vec::with_capacity(raw_scores.len());
        for (idx, raw) in raw_scores.into_iter().enumerate() {
            if let Some(allowed_clusters) = allowed_clusters
                && !allowed_clusters.contains(ClusterPostingState::cluster_index(idx)?.as_u32())
            {
                continue;
            }
            distances.push((idx, score_to_distance(metric, raw)));
        }
        drop(centroids);

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(distances.into_iter().take(k).map(|(idx, _)| idx).collect())
    }
}

/// Nearest centroid for incremental `add` routing (uses configured cluster metric).
pub fn nearest_centroid_for_vector(
    index: &CentroidIndex,
    vector: &[f32],
) -> Result<usize, DripstoneError> {
    let centroids = index.centroids.read();
    let metric = index.config.metric;
    let prepared = simd::PreparedQueryScorer::new(vector, dense_metric(metric))?;
    let mut centroid_refs = Vec::with_capacity(centroids.len());
    for centroid in centroids.iter() {
        centroid_refs.push(centroid.as_slice());
    }
    let raw_scores = prepared.score_dense_batch(&centroid_refs)?;
    drop(centroids);
    let mut best_idx = 0;
    let mut best_distance = f32::INFINITY;
    for (idx, raw) in raw_scores.into_iter().enumerate() {
        let distance = score_to_distance(metric, raw);
        if distance
            .partial_cmp(&best_distance)
            .unwrap_or(std::cmp::Ordering::Greater)
            .is_lt()
        {
            best_distance = distance;
            best_idx = idx;
        }
    }
    Ok(best_idx)
}

pub fn search_request(
    index: &CentroidIndex,
    request: IndexSearchRequest<'_>,
) -> Result<Vec<SimilarityResult>, DripstoneError> {
    let query = request.query;
    let limit = request.limit;
    let metric = request.metric;
    let records_empty = {
        let records = index.records.read();
        records.is_empty()
    };
    if records_empty {
        return Ok(Vec::new());
    }

    let num_probes = {
        let centroids = index.centroids.read();
        index
            .query_num_probes
            .load(Ordering::Relaxed)
            .min(centroids.len())
    };
    let allowed_clusters = request
        .native_filter
        .filter(|plan| plan.is_supported())
        .map(|plan| {
            let postings = index.cluster_postings.read();
            CentroidIndex::allowed_clusters_from_plan(&postings, plan)
        })
        .transpose()?
        .flatten();
    if allowed_clusters
        .as_ref()
        .is_some_and(RoaringBitmap::is_empty)
    {
        return Ok(Vec::new());
    }
    let nearest_centroid_indices =
        index.find_nearest_centroids(query, num_probes, allowed_clusters.as_ref(), metric)?;

    let candidates: Vec<Arc<EmbeddingRecord>> = {
        let records = index.records.read();
        let clusters = index.clusters.read();
        let mut out: Vec<Arc<EmbeddingRecord>> = Vec::new();
        for centroid_idx in nearest_centroid_indices {
            if let Some(cluster) = clusters.get(centroid_idx) {
                for record_id in &cluster.record_ids {
                    if let Some(record_arc) = records.get(record_id) {
                        out.push(Arc::clone(record_arc));
                    }
                }
            }
        }
        out
    };

    let prepared = simd::PreparedQueryScorer::new(query, dense_metric(metric))?;
    let mut candidate_vectors = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        candidate_vectors.push(candidate.vector.as_ref());
    }
    let raw_scores = prepared.score_dense_batch(&candidate_vectors)?;

    let mut results = Vec::with_capacity(candidates.len());
    for (record_arc, raw_score) in candidates.into_iter().zip(raw_scores) {
        let score = match metric {
            SimilarityMetric::Cosine | SimilarityMetric::Dot => raw_score,
            SimilarityMetric::Euclidean
            | SimilarityMetric::Manhattan
            | SimilarityMetric::Chebyshev => 1.0 / (1.0 + raw_score),
        };
        results.push(SimilarityResult::new(record_arc, score));
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results.truncate(limit);
    Ok(results)
}
