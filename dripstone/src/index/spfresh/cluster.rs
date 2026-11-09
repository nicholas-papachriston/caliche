//! K-means clustering and centroid maintenance for `SPFresh`.
//!
//! [`rebuild_record_to_cluster`] is a pure function of `clusters[].record_ids`.
//! [`CentroidIndex::split_cluster`] and [`CentroidIndex::merge_clusters`] move
//! those chunk ids, then call this helper and rebuild cluster-index bitmaps.
//! They do not rekey bodies, and they do not address a record by cluster index.

use super::postings::ClusterPostingState;
use super::{CentroidIndex, Cluster, ClusterFilterSummary, dense_metric, score_to_distance};
use crate::embedding::{EmbeddingRecord, SimilarityMetric};
use crate::errors::DripstoneError;
use crate::simd;
use crate::utils::numeric::usize_to_f32;
use std::collections::HashMap;
use std::sync::Arc;

pub fn rebuild_record_to_cluster(clusters: &[Cluster]) -> HashMap<String, usize> {
    let mut record_to_cluster = HashMap::new();
    for (cluster_idx, cluster) in clusters.iter().enumerate() {
        for record_id in &cluster.record_ids {
            record_to_cluster.insert(record_id.clone(), cluster_idx);
        }
    }
    record_to_cluster
}

impl CentroidIndex {
    /// Initialize centroids using K-means++ under the configured metric.
    pub(crate) fn initialize_centroids(
        &self,
        vectors: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> Result<Vec<Vec<f32>>, DripstoneError> {
        if vectors.is_empty() {
            return Ok(Vec::new());
        }

        let k = self.config.num_centroids.min(vectors.len());
        let mut centroids = Vec::with_capacity(k);

        centroids.push(vectors[0].clone());

        let dense = dense_metric(metric);
        for _ in 1..k {
            let mut distances = Vec::with_capacity(vectors.len());
            for vector in vectors {
                let mut min_distance = f32::INFINITY;
                for centroid in &centroids {
                    let prepared = simd::PreparedQueryScorer::new(vector, dense)?;
                    let raw = prepared.score_dense(centroid).unwrap_or(f32::INFINITY);
                    let d = score_to_distance(metric, raw);
                    min_distance = min_distance.min(d * d);
                }
                distances.push(min_distance.max(0.0));
            }

            let sum: f32 = distances.iter().sum();
            if sum > 0.0 {
                for d in &mut distances {
                    *d /= sum;
                }
            }

            let mut cumsum = 0.0;
            let random = fastrand::f32();
            let mut chosen = None;
            for (i, &prob) in distances.iter().enumerate() {
                cumsum += prob;
                if random <= cumsum {
                    chosen = Some(i);
                    break;
                }
            }
            let chosen = chosen.unwrap_or(vectors.len() - 1);
            centroids.push(vectors[chosen].clone());
        }

        Ok(centroids)
    }

    /// Assign each vector to the nearest centroid under the given metric.
    pub(crate) fn assign_to_clusters(
        vectors: &[Vec<f32>],
        centroids: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> Result<Vec<usize>, DripstoneError> {
        let dense = dense_metric(metric);
        let mut assignments = Vec::with_capacity(vectors.len());
        for vector in vectors {
            let prepared = simd::PreparedQueryScorer::new(vector, dense)?;
            let mut best_idx = 0;
            let mut best_dist = f32::INFINITY;
            for (idx, centroid) in centroids.iter().enumerate() {
                let raw = prepared.score_dense(centroid).unwrap_or(f32::INFINITY);
                let dist = score_to_distance(metric, raw);
                if dist
                    .partial_cmp(&best_dist)
                    .unwrap_or(std::cmp::Ordering::Greater)
                    .is_lt()
                {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            assignments.push(best_idx);
        }
        Ok(assignments)
    }

    /// Update centroids based on cluster assignments. Only Cosine centroids
    /// are re-normalized to unit length; other metrics keep the raw mean.
    pub(crate) fn update_centroids(
        &self,
        vectors: &[Vec<f32>],
        assignments: &[usize],
        num_centroids: usize,
        metric: SimilarityMetric,
        previous_centroids: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let mut new_centroids = vec![vec![0.0; self.dimension]; num_centroids];
        let mut counts = vec![0_usize; num_centroids];

        for i in 0..vectors.len() {
            let vector = &vectors[i];
            let cluster_idx = assignments[i];
            for dim in 0..vector.len() {
                new_centroids[cluster_idx][dim] += vector[dim];
            }
            counts[cluster_idx] += 1;
        }

        for cluster_idx in 0..new_centroids.len() {
            let count = counts[cluster_idx];
            if count > 0 {
                let centroid = &mut new_centroids[cluster_idx];
                for value in centroid.iter_mut() {
                    *value /= usize_to_f32(count);
                }
                if matches!(metric, SimilarityMetric::Cosine) {
                    let mut squared_sum = 0.0_f32;
                    for &value in centroid.iter() {
                        squared_sum += value * value;
                    }
                    let norm = squared_sum.sqrt();
                    if norm > 0.0 {
                        for value in centroid.iter_mut() {
                            *value /= norm;
                        }
                    }
                }
            }
            if count == 0
                && let Some(previous) = previous_centroids.get(cluster_idx)
            {
                new_centroids[cluster_idx].clone_from(previous);
            }
        }

        new_centroids
    }

    /// Check if centroids have converged (under the given metric).
    pub(crate) fn has_converged(
        &self,
        old_centroids: &[Vec<f32>],
        new_centroids: &[Vec<f32>],
        metric: SimilarityMetric,
    ) -> bool {
        let dense = dense_metric(metric);
        for i in 0..old_centroids.len() {
            let Ok(prepared) = simd::PreparedQueryScorer::new(&old_centroids[i], dense) else {
                return false;
            };
            let raw = prepared
                .score_dense(&new_centroids[i])
                .unwrap_or(f32::INFINITY);
            let dist = match metric {
                SimilarityMetric::Cosine => 1.0 - raw,
                SimilarityMetric::Dot => -raw,
                _ => raw,
            };
            if dist >= self.config.convergence_threshold {
                return false;
            }
        }
        true
    }

    /// Run K-means clustering under the configured metric.
    pub(crate) fn kmeans_clustering(
        &self,
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<Vec<f32>>, Vec<usize>), DripstoneError> {
        let metric = self.config.metric;
        let k = self.config.num_centroids.min(vectors.len());
        let mut centroids = self.initialize_centroids(vectors, metric)?;
        let mut assignments = vec![0; vectors.len()];

        for _ in 0..self.config.max_iterations {
            let new_assignments = Self::assign_to_clusters(vectors, &centroids, metric)?;
            let new_centroids =
                self.update_centroids(vectors, &new_assignments, k, metric, &centroids);

            if self.has_converged(&centroids, &new_centroids, metric) {
                return Ok((new_centroids, new_assignments));
            }

            centroids = new_centroids;
            assignments = new_assignments;
        }

        Ok((centroids, assignments))
    }
}

/// Membership moved from one cluster index to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitOutcome {
    /// Cluster that kept the ids not listed in [`Self::moved_chunk_ids`].
    pub source: usize,
    /// New cluster index appended for the moved ids.
    pub created: usize,
    /// Chunk ids whose cluster membership changed. Object keys are unchanged.
    pub moved_chunk_ids: Vec<String>,
}

/// Membership moved into `destination` before `removed` was dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeOutcome {
    /// Cluster index after the source cluster was removed.
    pub destination: usize,
    /// Cluster index that was removed.
    pub removed: usize,
    /// Chunk ids that changed cluster. Object keys are unchanged.
    pub moved_chunk_ids: Vec<String>,
}

impl CentroidIndex {
    pub(super) fn rebalance_if_needed(&self, cluster_idx: usize) -> Result<(), DripstoneError> {
        let Some(limit) = self.config.max_cluster_records else {
            return Ok(());
        };
        let limit = limit.max(1);
        let mut steps = 0usize;
        loop {
            let size = self
                .clusters
                .read()
                .get(cluster_idx)
                .map(|cluster| cluster.record_ids.len())
                .unwrap_or(0);
            if size <= limit {
                return Ok(());
            }
            if steps >= size {
                return Err(DripstoneError::storage(
                    "SPFresh split did not shrink the cluster",
                ));
            }
            self.split_cluster(cluster_idx)?;
            steps += 1;
        }
    }

    /// Split one cluster into two by moving chunk ids.
    ///
    /// `records` stays keyed by chunk id. Summaries for the two clusters are
    /// rebuilt from scoring records, then cluster-index bitmaps are rebuilt
    /// from every summary. This does not write object storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the cluster has fewer than two members, a scoring
    /// vector is missing, or a centroid cannot be scored. Nothing is moved
    /// when that happens.
    pub fn split_cluster(&self, cluster_idx: usize) -> Result<SplitOutcome, DripstoneError> {
        let ids = {
            let clusters = self.clusters.read();
            let cluster = clusters.get(cluster_idx).ok_or_else(|| {
                DripstoneError::storage(format!("SPFresh split cluster {cluster_idx} is missing"))
            })?;
            if cluster.record_ids.len() < 2 {
                return Err(DripstoneError::storage(
                    "SPFresh split needs at least two members; membership was not changed",
                ));
            }
            cluster.record_ids.clone()
        };
        let (scoring, vectors) = self.scoring_vectors(&ids)?;
        let (stay, leave) = partition_membership(&vectors, self.config.metric)?;
        let stay_ids = indices_to_ids(&ids, &stay);
        let leave_ids = indices_to_ids(&ids, &leave);
        let stay_summary = summary_for_records(&refs_at(&scoring, &stay))?;
        let leave_summary = summary_for_records(&refs_at(&scoring, &leave))?;
        let stay_centroid = mean_centroid(&vectors_at(&vectors, &stay), self.config.metric);
        let leave_centroid = mean_centroid(&vectors_at(&vectors, &leave), self.config.metric);

        let created = {
            let mut centroids = self.centroids.write();
            let mut clusters = self.clusters.write();
            let mut summaries = self.cluster_summaries.write();
            if centroids.len() != clusters.len() || summaries.len() != clusters.len() {
                return Err(DripstoneError::storage(
                    "SPFresh cluster, centroid, and summary lengths differ; split was not applied",
                ));
            }
            let mut next_summaries = summaries.clone();
            next_summaries[cluster_idx] = stay_summary.clone();
            next_summaries.push(leave_summary.clone());
            let postings = ClusterPostingState::rebuild_cluster_postings(&next_summaries)?;
            clusters[cluster_idx].record_ids.clone_from(&stay_ids);
            clusters[cluster_idx].centroid.clone_from(&stay_centroid);
            centroids[cluster_idx].clone_from(&stay_centroid);
            summaries[cluster_idx] = stay_summary;
            clusters.push(Cluster {
                centroid: leave_centroid.clone(),
                record_ids: leave_ids.clone(),
            });
            centroids.push(leave_centroid);
            summaries.push(leave_summary);
            let created = clusters.len() - 1;
            let record_to_cluster = rebuild_record_to_cluster(&clusters);
            drop(summaries);
            drop(clusters);
            drop(centroids);
            *self.record_to_cluster.write() = record_to_cluster;
            *self.cluster_postings.write() = postings;
            created
        };

        Ok(SplitOutcome {
            source: cluster_idx,
            created,
            moved_chunk_ids: leave_ids,
        })
    }

    /// Move every chunk id from `source` into `destination` and drop `source`.
    ///
    /// Cluster indexes above `source` shift down. `record_to_cluster` and the
    /// cluster-index bitmaps are rebuilt. Chunk ids and `records` keys do not
    /// change.
    ///
    /// # Errors
    ///
    /// Returns an error when an index is missing, the two indexes are the same,
    /// or a scoring vector is missing. Nothing is moved when that happens.
    pub fn merge_clusters(
        &self,
        destination: usize,
        source: usize,
    ) -> Result<MergeOutcome, DripstoneError> {
        if destination == source {
            return Err(DripstoneError::storage(
                "SPFresh merge needs two distinct clusters",
            ));
        }
        let (destination_ids, source_ids) = {
            let clusters = self.clusters.read();
            let destination_ids = clusters.get(destination).ok_or_else(|| {
                DripstoneError::storage(format!(
                    "SPFresh merge destination {destination} is missing"
                ))
            })?;
            let source_ids = clusters.get(source).ok_or_else(|| {
                DripstoneError::storage(format!("SPFresh merge source {source} is missing"))
            })?;
            (
                destination_ids.record_ids.clone(),
                source_ids.record_ids.clone(),
            )
        };
        let mut combined_ids = destination_ids;
        combined_ids.extend(source_ids.iter().cloned());
        let (scoring, vectors) = self.scoring_vectors(&combined_ids)?;
        let summary = summary_for_records(
            &scoring
                .iter()
                .map(std::convert::AsRef::as_ref)
                .collect::<Vec<_>>(),
        )?;
        let centroid = mean_centroid(&vectors, self.config.metric);
        let new_destination = if destination > source {
            destination - 1
        } else {
            destination
        };

        {
            let mut centroids = self.centroids.write();
            let mut clusters = self.clusters.write();
            let mut summaries = self.cluster_summaries.write();
            if centroids.len() != clusters.len() || summaries.len() != clusters.len() {
                return Err(DripstoneError::storage(
                    "SPFresh cluster, centroid, and summary lengths differ; merge was not applied",
                ));
            }
            let mut next_summaries = summaries.clone();
            next_summaries[destination] = summary.clone();
            next_summaries.remove(source);
            let postings = ClusterPostingState::rebuild_cluster_postings(&next_summaries)?;
            clusters[destination].record_ids.clone_from(&combined_ids);
            clusters[destination].centroid.clone_from(&centroid);
            centroids[destination].clone_from(&centroid);
            summaries[destination] = summary;
            clusters.remove(source);
            centroids.remove(source);
            summaries.remove(source);
            let record_to_cluster = rebuild_record_to_cluster(&clusters);
            drop(summaries);
            drop(clusters);
            drop(centroids);
            *self.record_to_cluster.write() = record_to_cluster;
            *self.cluster_postings.write() = postings;
        }

        Ok(MergeOutcome {
            destination: new_destination,
            removed: source,
            moved_chunk_ids: source_ids,
        })
    }

    fn scoring_vectors(
        &self,
        ids: &[String],
    ) -> Result<(Vec<Arc<EmbeddingRecord>>, Vec<Vec<f32>>), DripstoneError> {
        let records = self.records.read();
        let mut scoring = Vec::with_capacity(ids.len());
        let mut vectors = Vec::with_capacity(ids.len());
        for id in ids {
            let Some(record) = records.get(id) else {
                return Err(DripstoneError::storage(format!(
                    "SPFresh rebalance needs the scoring vector for {id}; membership was not changed"
                )));
            };
            vectors.push(record.vector.to_vec());
            scoring.push(Arc::clone(record));
        }
        Ok((scoring, vectors))
    }
}

fn indices_to_ids(ids: &[String], indexes: &[usize]) -> Vec<String> {
    indexes.iter().map(|&index| ids[index].clone()).collect()
}

fn refs_at<'a>(records: &'a [Arc<EmbeddingRecord>], indexes: &[usize]) -> Vec<&'a EmbeddingRecord> {
    indexes
        .iter()
        .map(|&index| records[index].as_ref())
        .collect()
}

fn vectors_at(vectors: &[Vec<f32>], indexes: &[usize]) -> Vec<Vec<f32>> {
    indexes
        .iter()
        .map(|&index| vectors[index].clone())
        .collect()
}

fn summary_for_records(
    records: &[&EmbeddingRecord],
) -> Result<ClusterFilterSummary, DripstoneError> {
    let mut summary = ClusterFilterSummary::default();
    let mut postings = ClusterPostingState::default();
    for record in records {
        CentroidIndex::insert_record_filter_terms(&mut summary, &mut postings, 0, record)?;
    }
    Ok(summary)
}

fn mean_centroid(vectors: &[Vec<f32>], metric: SimilarityMetric) -> Vec<f32> {
    if vectors.is_empty() {
        return Vec::new();
    }
    let mut centroid = vec![0.0_f32; vectors[0].len()];
    for vector in vectors {
        for (slot, value) in centroid.iter_mut().zip(vector.iter()) {
            *slot += *value;
        }
    }
    let count = usize_to_f32(vectors.len());
    for value in &mut centroid {
        *value /= count;
    }
    if matches!(metric, SimilarityMetric::Cosine) {
        let mut squared = 0.0_f32;
        for value in &centroid {
            squared += value * value;
        }
        let norm = squared.sqrt();
        if norm > 0.0 {
            for value in &mut centroid {
                *value /= norm;
            }
        }
    }
    centroid
}

fn partition_membership(
    vectors: &[Vec<f32>],
    metric: SimilarityMetric,
) -> Result<(Vec<usize>, Vec<usize>), DripstoneError> {
    let dense = dense_metric(metric);
    let prepared = simd::PreparedQueryScorer::new(&vectors[0], dense)?;
    let mut far = 1usize;
    let mut far_distance = f32::NEG_INFINITY;
    for (index, vector) in vectors.iter().enumerate().skip(1) {
        let raw = prepared.score_dense(vector).unwrap_or(f32::INFINITY);
        let distance = score_to_distance(metric, raw);
        if distance > far_distance {
            far_distance = distance;
            far = index;
        }
    }
    let mut stay = Vec::new();
    let mut leave = Vec::new();
    for (index, vector) in vectors.iter().enumerate() {
        let toward_first = vector_distance(vector, &vectors[0], metric)?;
        let toward_far = vector_distance(vector, &vectors[far], metric)?;
        if toward_first <= toward_far {
            stay.push(index);
        } else {
            leave.push(index);
        }
    }
    if leave.is_empty() {
        if let Some(moved) = stay.pop() {
            leave.push(moved);
        }
    }
    if stay.is_empty() {
        if let Some(moved) = leave.pop() {
            stay.push(moved);
        }
    }
    if stay.is_empty() || leave.is_empty() {
        return Err(DripstoneError::storage(
            "SPFresh split could not place members on both sides",
        ));
    }
    Ok((stay, leave))
}

fn vector_distance(
    left: &[f32],
    right: &[f32],
    metric: SimilarityMetric,
) -> Result<f32, DripstoneError> {
    let prepared = simd::PreparedQueryScorer::new(left, dense_metric(metric))?;
    let raw = prepared.score_dense(right).unwrap_or(f32::INFINITY);
    Ok(score_to_distance(metric, raw))
}
