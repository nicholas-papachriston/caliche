//! `SPFresh`: Centroid-based approximate nearest neighbor index
//!
//! Optimized for object storage with minimal roundtrips and write amplification.
//! Uses K-means clustering to partition vectors into clusters, enabling fast
//! coarse search followed by fine-grained search within relevant clusters.
//!
//! # Membership and bodies
//!
//! `records`, `clusters[].record_ids`, and `record_to_cluster` are keyed by
//! chunk id (`{doc_id}#{chunk_index}`), never by cluster index or a local slot.
//! `records` keeps the pooled vector and `multi_vectors` for MaxSim. A chunk
//! envelope's `content` is not stored on that map.
//!
//! [`CentroidIndex::split_cluster`] and [`CentroidIndex::merge_clusters`] move
//! chunk ids in `record_ids` / `record_to_cluster`, update summary counts, and
//! rebuild [`ClusterPostingState`] from those summaries. They do not rekey
//! `records` or any object store. Incremental `add` places a new id on the
//! nearest centroid. When [`SPFreshConfig::max_cluster_records`] is set, `add`
//! may split that cluster's membership. Bodies stay keyed by chunk id.
//! [`cluster::rebuild_record_to_cluster`] stays a pure function of `record_ids`.

mod cluster;
mod id_plan;
mod postings;
mod search;

use cluster::rebuild_record_to_cluster;
use postings::ClusterPostingState;

use crate::backend::chunk_envelope::{KEY_CONTENT, KEY_ORIGINAL_METADATA};
use crate::embedding::{EmbeddingRecord, SimilarityMetric, SimilarityResult};
use crate::errors::DripstoneError;
use crate::index::{IndexSearchRequest, VectorIndex};
use crate::simd;
use async_trait::async_trait;
use parking_lot::RwLock;
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Configuration for `SPFresh` index
#[derive(Debug, Clone)]
pub struct SPFreshConfig {
    /// Number of centroids (clusters) to create
    pub num_centroids: usize,
    /// Maximum K-means iterations
    pub max_iterations: usize,
    /// Convergence threshold for K-means
    pub convergence_threshold: f32,
    /// Number of centroids to probe during search
    pub num_probes: usize,
    /// Similarity metric used for clustering and incremental routing.
    /// Search-time requests may override this per-query; the clustering
    /// itself uses this metric so the cluster shapes match the intended
    /// score function.
    pub metric: SimilarityMetric,
    /// Split a cluster after `add` when it grows past this many chunk ids.
    /// `None` leaves membership where incremental insert placed it.
    /// The split moves chunk ids only.
    pub max_cluster_records: Option<usize>,
}

impl Default for SPFreshConfig {
    fn default() -> Self {
        Self {
            num_centroids: 256,
            max_iterations: 100,
            convergence_threshold: 0.001,
            num_probes: 10,
            metric: SimilarityMetric::Cosine,
            max_cluster_records: None,
        }
    }
}

/// A cluster of vectors with their centroid
#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug))]
pub struct Cluster {
    /// Centroid vector
    pub centroid: Vec<f32>,
    /// Record IDs in this cluster
    pub record_ids: Vec<String>,
}

#[derive(Debug, Clone, Default, Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug))]
pub struct ClusterFilterSummary {
    pub namespace_counts: HashMap<String, u32>,
    pub field_value_counts: HashMap<crate::search::NativeFilterKey, u32>,
}

/// `SPFresh` centroid-based index
///
/// # Memory Optimization
///
/// Records are stored as `Arc<EmbeddingRecord>` to enable cheap cloning
/// when returning search results. This avoids copying ~6KB vectors per result.
#[derive(Debug)]
pub struct CentroidIndex {
    pub(crate) config: SPFreshConfig,
    pub(crate) query_num_probes: AtomicUsize,
    pub(crate) centroids: Arc<RwLock<Vec<Vec<f32>>>>,
    pub(crate) clusters: Arc<RwLock<Vec<Cluster>>>,
    pub(crate) records: Arc<RwLock<HashMap<String, Arc<EmbeddingRecord>>>>,
    pub(crate) record_to_cluster: Arc<RwLock<HashMap<String, usize>>>,
    pub(crate) cluster_summaries: Arc<RwLock<Vec<ClusterFilterSummary>>>,
    pub(crate) cluster_postings: Arc<RwLock<ClusterPostingState>>,
    pub(crate) dimension: usize,
}

pub(crate) const fn dense_metric(metric: SimilarityMetric) -> simd::DenseMetric {
    match metric {
        SimilarityMetric::Cosine => simd::DenseMetric::Cosine,
        SimilarityMetric::Dot => simd::DenseMetric::Dot,
        SimilarityMetric::Euclidean => simd::DenseMetric::Euclidean,
        SimilarityMetric::Manhattan => simd::DenseMetric::Manhattan,
        SimilarityMetric::Chebyshev => simd::DenseMetric::Chebyshev,
    }
}

/// Convert a raw score produced by `score_dense*` into a non-negative,
/// monotonic "distance" (lower means closer) for cluster routing.
#[inline]
pub(crate) fn score_to_distance(metric: SimilarityMetric, score: f32) -> f32 {
    match metric {
        SimilarityMetric::Cosine => (1.0 - score).max(0.0),
        SimilarityMetric::Dot => -score,
        SimilarityMetric::Euclidean | SimilarityMetric::Manhattan | SimilarityMetric::Chebyshev => {
            score
        }
    }
}

/// What [`CentroidIndex::records`] keeps from an inserted record.
enum ScoringMetadata {
    /// Not a chunk envelope. Leave metadata as the caller provided it.
    Keep,
    /// Drop `content` and the rest of the chunk envelope.
    Replace(Option<serde_json::Value>),
}

fn scoring_metadata(metadata: &serde_json::Value) -> ScoringMetadata {
    let Some(object) = metadata.as_object() else {
        return ScoringMetadata::Keep;
    };
    if object.contains_key(KEY_CONTENT) && object.contains_key(KEY_ORIGINAL_METADATA) {
        let original = object
            .get(KEY_ORIGINAL_METADATA)
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let kept = match original {
            serde_json::Value::Null => None,
            serde_json::Value::Object(map) if map.is_empty() => None,
            other => Some(other),
        };
        return ScoringMetadata::Replace(kept);
    }
    if object.contains_key(KEY_CONTENT) {
        let mut kept = object.clone();
        kept.remove(KEY_CONTENT);
        if kept.is_empty() {
            return ScoringMetadata::Replace(None);
        }
        return ScoringMetadata::Replace(Some(serde_json::Value::Object(kept)));
    }
    ScoringMetadata::Keep
}

/// Pooled vector and MaxSim tokens, without chunk `content`.
///
/// A chunk envelope stores document metadata (for cluster filter summaries)
/// and drops `content`, offsets, and `doc_id`. Other metadata is unchanged
/// except a bare `content` field, which is removed.
fn index_scoring_record(mut record: EmbeddingRecord) -> EmbeddingRecord {
    let Some(metadata) = record.metadata.clone() else {
        return record;
    };
    match scoring_metadata(&metadata) {
        ScoringMetadata::Keep => record,
        ScoringMetadata::Replace(metadata) => {
            record.metadata = metadata;
            record
        }
    }
}

impl CentroidIndex {
    /// Create a new `SPFresh` index
    #[must_use]
    pub fn new(config: SPFreshConfig, dimension: usize) -> Self {
        Self {
            query_num_probes: AtomicUsize::new(config.num_probes.max(1)),
            config,
            centroids: Arc::new(RwLock::new(Vec::new())),
            clusters: Arc::new(RwLock::new(Vec::new())),
            records: Arc::new(RwLock::new(HashMap::new())),
            record_to_cluster: Arc::new(RwLock::new(HashMap::new())),
            cluster_summaries: Arc::new(RwLock::new(Vec::new())),
            cluster_postings: Arc::new(RwLock::new(ClusterPostingState::default())),
            dimension,
        }
    }

    /// Serialize index to bytes for storage
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DripstoneError> {
        let centroids = self.centroids.read().clone();
        let clusters = self.clusters.read().clone();
        let cluster_summaries = self.cluster_summaries.read().clone();

        let data = IndexData {
            centroids,
            clusters,
            cluster_summaries,
            dimension: self.dimension,
        };

        rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .map(rkyv::util::AlignedVec::into_vec)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize index: {e}")))
    }

    /// Deserialize index from bytes
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn from_bytes(bytes: &[u8], config: SPFreshConfig) -> Result<Self, DripstoneError> {
        let data: IndexData = rkyv::from_bytes::<_, rkyv::rancor::Error>(bytes)
            .map_err(|e| DripstoneError::storage(format!("Failed to deserialize index: {e}")))?;
        let record_to_cluster = rebuild_record_to_cluster(&data.clusters);
        let cluster_postings =
            ClusterPostingState::rebuild_cluster_postings(&data.cluster_summaries)?;

        Ok(Self {
            query_num_probes: AtomicUsize::new(config.num_probes.max(1)),
            config,
            centroids: Arc::new(RwLock::new(data.centroids)),
            clusters: Arc::new(RwLock::new(data.clusters)),
            records: Arc::new(RwLock::new(HashMap::new())),
            record_to_cluster: Arc::new(RwLock::new(record_to_cluster)),
            cluster_summaries: Arc::new(RwLock::new(data.cluster_summaries)),
            cluster_postings: Arc::new(RwLock::new(cluster_postings)),
            dimension: data.dimension,
        })
    }

    /// Chunk ids stored in a membership snapshot.
    ///
    /// The snapshot does not contain envelopes. Duplicate ids are an error so
    /// compaction cannot treat a corrupt checkpoint as durable.
    ///
    /// # Errors
    ///
    /// Returns an error when the bytes are not an index snapshot or a chunk id
    /// is listed in more than one cluster.
    pub fn snapshot_membership(
        bytes: &[u8],
    ) -> Result<std::collections::BTreeSet<String>, DripstoneError> {
        let data: IndexData = rkyv::from_bytes::<_, rkyv::rancor::Error>(bytes)
            .map_err(|e| DripstoneError::storage(format!("Failed to deserialize index: {e}")))?;
        let mut ids = std::collections::BTreeSet::new();
        let mut count = 0usize;
        for cluster in &data.clusters {
            for id in &cluster.record_ids {
                count += 1;
                if !ids.insert(id.clone()) {
                    return Err(DripstoneError::storage(format!(
                        "snapshot lists chunk id {id} in more than one cluster"
                    )));
                }
            }
        }
        if ids.len() != count {
            return Err(DripstoneError::storage(
                "snapshot membership count does not match unique chunk ids",
            ));
        }
        Ok(ids)
    }

    /// Scoring record stored for `id`, if this process has inserted it.
    ///
    /// ColdForge uses this as an existence check. The chunk envelope, including
    /// `content`, lives in the chunk-id body store. Without the ColdForge
    /// feature the method has no production caller outside the tests, so the
    /// dead-code lint is expected there.
    #[must_use]
    #[cfg_attr(
        all(not(test), not(feature = "coldforge-backend")),
        expect(
            dead_code,
            reason = "only the coldforge backend and the tests read records by id"
        )
    )]
    pub(crate) fn get_record(&self, id: &str) -> Option<Arc<EmbeddingRecord>> {
        self.records.read().get(id).map(Arc::clone)
    }

    /// Get index statistics
    pub fn stats(&self) -> IndexStats {
        let centroids = self.centroids.read();
        let clusters = self.clusters.read();
        let records = self.records.read();

        IndexStats {
            num_centroids: centroids.len(),
            num_clusters: clusters.len(),
            num_records: records.len(),
            dimension: self.dimension,
        }
    }

    /// Update the number of centroids probed during search.
    pub fn set_num_probes(&self, num_probes: usize) {
        self.query_num_probes
            .store(num_probes.max(1), Ordering::Relaxed);
    }
}

/// Serializable index data
#[derive(Archive, Serialize, Deserialize)]
#[rkyv(derive(Debug))]
struct IndexData {
    centroids: Vec<Vec<f32>>,
    clusters: Vec<Cluster>,
    cluster_summaries: Vec<ClusterFilterSummary>,
    dimension: usize,
}

/// Index statistics
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_centroids: usize,
    pub num_clusters: usize,
    pub num_records: usize,
    pub dimension: usize,
}

#[async_trait]
impl VectorIndex for CentroidIndex {
    async fn build(&self, records: &[EmbeddingRecord]) -> Result<(), DripstoneError> {
        if records.is_empty() {
            self.clear();
            return Ok(());
        }

        let stored: Vec<EmbeddingRecord> =
            records.iter().cloned().map(index_scoring_record).collect();
        let vectors: Vec<Vec<f32>> = stored.iter().map(|r| r.vector.to_vec()).collect();

        let (centroids, assignments) = self.kmeans_clustering(&vectors)?;

        let centroids_for_storage = centroids.clone();

        let mut clusters: Vec<Cluster> = centroids
            .into_iter()
            .map(|centroid| Cluster {
                centroid,
                record_ids: Vec::new(),
            })
            .collect();
        let mut cluster_summaries = vec![ClusterFilterSummary::default(); clusters.len()];
        let mut cluster_postings = ClusterPostingState::default();
        let mut record_to_cluster = HashMap::with_capacity(records.len());

        for (record, &cluster_idx) in stored.iter().zip(assignments.iter()) {
            clusters[cluster_idx].record_ids.push(record.id.to_string());
            record_to_cluster.insert(record.id.to_string(), cluster_idx);
            Self::insert_record_filter_terms(
                &mut cluster_summaries[cluster_idx],
                &mut cluster_postings,
                cluster_idx,
                record,
            )?;
        }

        {
            let mut records_map = self.records.write();
            records_map.clear();
            for record in stored {
                let record_id = record.id.to_string();
                records_map.insert(record_id, Arc::new(record));
            }
        }

        {
            let mut centroids_guard = self.centroids.write();
            *centroids_guard = centroids_for_storage;
        }
        {
            let mut clusters_guard = self.clusters.write();
            *clusters_guard = clusters;
        }
        {
            let mut record_to_cluster_guard = self.record_to_cluster.write();
            *record_to_cluster_guard = record_to_cluster;
        }
        {
            let mut summaries_guard = self.cluster_summaries.write();
            *summaries_guard = cluster_summaries;
        }
        {
            let mut postings_guard = self.cluster_postings.write();
            *postings_guard = cluster_postings;
        }

        Ok(())
    }

    async fn search_request(
        &self,
        request: IndexSearchRequest<'_>,
    ) -> Result<Vec<SimilarityResult>, DripstoneError> {
        search::search_request(self, request)
    }

    async fn add(&self, record: EmbeddingRecord) -> Result<(), DripstoneError> {
        let record = index_scoring_record(record);
        let record_id = record.id.to_string();
        if self.records.read().contains_key(&record_id) {
            self.remove(&record_id).await?;
        } else if self.record_to_cluster.read().contains_key(&record_id) {
            // `from_bytes` restores membership without record bodies. Attaching
            // the envelope must not append the id onto `record_ids` again.
            self.records.write().insert(record_id, Arc::new(record));
            return Ok(());
        }
        let record_arc = Arc::new(record);

        {
            let mut records = self.records.write();
            records.insert(record_id.clone(), Arc::clone(&record_arc));
        }

        let need_rebuild = {
            let centroids = self.centroids.read();
            centroids.is_empty()
        };
        if need_rebuild {
            let all_records: Vec<EmbeddingRecord> = {
                let records = self.records.read();
                records.values().map(|arc| (**arc).clone()).collect()
            };
            return self.build(&all_records).await;
        }

        let nearest_centroid_idx =
            search::nearest_centroid_for_vector(self, record_arc.vector.as_ref())?;

        {
            let mut clusters = self.clusters.write();
            if let Some(cluster) = clusters.get_mut(nearest_centroid_idx) {
                cluster.record_ids.push(record_id.clone());
            }
        }
        {
            let mut record_to_cluster = self.record_to_cluster.write();
            record_to_cluster.insert(record_id, nearest_centroid_idx);
        }
        {
            let mut summaries = self.cluster_summaries.write();
            let mut postings = self.cluster_postings.write();
            if let Some(summary) = summaries.get_mut(nearest_centroid_idx) {
                Self::insert_record_filter_terms(
                    summary,
                    &mut postings,
                    nearest_centroid_idx,
                    record_arc.as_ref(),
                )?;
            }
        }

        self.rebalance_if_needed(nearest_centroid_idx)?;
        Ok(())
    }

    async fn remove(&self, id: &str) -> Result<(), DripstoneError> {
        let removed_record = {
            let mut records = self.records.write();
            records.remove(id)
        };
        let cluster_idx = {
            let mut record_to_cluster = self.record_to_cluster.write();
            record_to_cluster.remove(id)
        };

        if let Some(cluster_idx) = cluster_idx {
            let mut clusters = self.clusters.write();
            if let Some(cluster) = clusters.get_mut(cluster_idx) {
                cluster.record_ids.retain(|record_id| record_id != id);
            }
            drop(clusters);
            if let Some(record_arc) = removed_record {
                let mut summaries = self.cluster_summaries.write();
                let mut postings = self.cluster_postings.write();
                if let Some(summary) = summaries.get_mut(cluster_idx) {
                    Self::remove_record_filter_terms(
                        summary,
                        &mut postings,
                        cluster_idx,
                        record_arc.as_ref(),
                    )?;
                }
            }
        }

        Ok(())
    }

    fn len(&self) -> usize {
        self.records.read().len()
    }

    fn set_ef_search(&self, ef: usize) {
        self.set_num_probes(ef);
    }

    fn clear(&self) {
        self.records.write().clear();
        self.record_to_cluster.write().clear();
        self.centroids.write().clear();
        self.clusters.write().clear();
        self.cluster_summaries.write().clear();
        *self.cluster_postings.write() = ClusterPostingState::default();
    }

    async fn iter_records(&self) -> Result<Vec<Arc<EmbeddingRecord>>, DripstoneError> {
        let records = self.records.read();
        Ok(records.values().map(Arc::clone).collect())
    }

    async fn iter_records_by_namespace(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<Arc<EmbeddingRecord>>, DripstoneError> {
        let records = self.records.read();
        Ok(records
            .values()
            .filter(|record| match (namespace, record.namespace.as_deref()) {
                (Some(ns), Some(record_ns)) => ns == record_ns,
                (None, None) => true,
                _ => false,
            })
            .map(Arc::clone)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_records(count: usize, dimension: usize) -> Vec<EmbeddingRecord> {
        (0..count)
            .map(|i| {
                let vector: Vec<f32> = (0..dimension)
                    .map(|_| fastrand::f32().mul_add(2.0, -1.0))
                    .collect();
                EmbeddingRecord::new(format!("record_{i}"), vector)
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_spfresh_build() {
        let config = SPFreshConfig {
            num_centroids: 10,
            ..Default::default()
        };
        let index = CentroidIndex::new(config, 128);

        let records = create_test_records(100, 128);
        index
            .build(&records)
            .await
            .expect("Failed to build SPFresh index");

        assert_eq!(index.len(), 100);
        let stats = index.stats();
        assert!(stats.num_centroids > 0);
    }

    #[tokio::test]
    async fn test_spfresh_search() {
        let config = SPFreshConfig {
            num_centroids: 10,
            num_probes: 3,
            ..Default::default()
        };
        let index = CentroidIndex::new(config, 128);

        let records = create_test_records(100, 128);
        index
            .build(&records)
            .await
            .expect("Failed to build SPFresh index");

        let query: Vec<f32> = (0..128)
            .map(|_| fastrand::f32().mul_add(2.0, -1.0))
            .collect();
        let results = index
            .search(&query, 10, SimilarityMetric::Cosine)
            .await
            .expect("Failed to search SPFresh index");

        assert!(!results.is_empty());
        assert!(results.len() <= 10);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_spfresh_add_remove() {
        let config = SPFreshConfig::default();
        let index = CentroidIndex::new(config, 128);

        let records = create_test_records(50, 128);
        index
            .build(&records)
            .await
            .expect("Failed to build SPFresh index");

        let new_record = EmbeddingRecord::new(
            "new_record",
            (0..128).map(|_| fastrand::f32()).collect::<Vec<_>>(),
        );
        index
            .add(new_record)
            .await
            .expect("Failed to add record to SPFresh index");

        assert_eq!(index.len(), 51);

        index
            .remove("new_record")
            .await
            .expect("Failed to remove record from SPFresh index");

        assert_eq!(index.len(), 50);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_spfresh_serialization() {
        let config = SPFreshConfig::default();
        let index = CentroidIndex::new(config.clone(), 128);

        let records = create_test_records(50, 128);
        index
            .build(&records)
            .await
            .expect("Failed to build SPFresh index");

        let bytes = index.to_bytes().expect("Index serialization failed");

        let restored =
            CentroidIndex::from_bytes(&bytes, config).expect("Failed to deserialize index");

        let index_stats = index.stats();
        let restored_stats = restored.stats();

        assert_eq!(index_stats.num_centroids, restored_stats.num_centroids);
        assert_eq!(index_stats.dimension, restored_stats.dimension);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn snapshot_replay_does_not_duplicate_membership() {
        let config = SPFreshConfig {
            num_centroids: 2,
            ..SPFreshConfig::default()
        };
        let index = CentroidIndex::new(config.clone(), 8);
        let records = create_test_records(6, 8);
        index.build(&records).await.expect("build snapshot source");
        let sample_id = records[0].id.to_string();
        let bytes = index.to_bytes().expect("serialize snapshot");
        let restored = CentroidIndex::from_bytes(&bytes, config).expect("deserialize snapshot");

        restored
            .add(records[0].clone())
            .await
            .expect("hydrate existing id");
        restored
            .add(records[0].clone())
            .await
            .expect("replay same id again");

        let membership = restored
            .clusters
            .read()
            .iter()
            .map(|cluster| {
                cluster
                    .record_ids
                    .iter()
                    .filter(|id| id.as_str() == sample_id)
                    .count()
            })
            .sum::<usize>();
        assert_eq!(
            membership, 1,
            "snapshot plus WAL replay duplicated record id"
        );
        assert!(restored.records.read().contains_key(&sample_id));
    }

    #[tokio::test]
    async fn records_keep_vectors_and_drop_chunk_envelopes() {
        let config = SPFreshConfig {
            num_centroids: 2,
            ..SPFreshConfig::default()
        };
        let index = CentroidIndex::new(config, 2);
        let tokens = crate::search::multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let record = EmbeddingRecord::new("doc#0", vec![1.0_f32, 0.0])
            .with_namespace("demo")
            .with_multi_vectors(std::sync::Arc::clone(&tokens))
            .with_metadata(serde_json::json!({
                "doc_id": "doc",
                "content": "hello body",
                "start_offset": 0,
                "end_offset": 10,
                "chunk_index": 0,
                "total_chunks": 1,
                "original_metadata": {"category": "greeting"},
            }));
        index.add(record).await.expect("add envelope");

        let stored = index.get_record("doc#0").expect("scoring record");
        assert!(stored.metadata.as_ref().is_none_or(|metadata| {
            metadata.get(KEY_CONTENT).is_none() && metadata.get("doc_id").is_none()
        }));
        assert_eq!(
            stored
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.get("category"))
                .and_then(serde_json::Value::as_str),
            Some("greeting")
        );
        assert_eq!(stored.vector.as_ref(), &[1.0, 0.0]);
        assert_eq!(
            stored
                .multi_vectors
                .as_ref()
                .map(std::convert::AsRef::as_ref),
            Some(tokens.as_ref())
        );
        assert!(index.records.read().contains_key("doc#0"));
        assert!(index.record_to_cluster.read().contains_key("doc#0"));
        let postings = format!("{:?}", index.cluster_postings.read());
        assert!(
            !postings.contains("hello body"),
            "cluster postings stored the chunk envelope"
        );
        let cluster_idx = index.record_to_cluster.read().get("doc#0").copied();
        assert!(cluster_idx.is_some());
        let body_key = format!("bodies/demo/{}", stored.id);
        assert!(body_key.ends_with("doc#0"));
        assert!(!body_key.contains("cluster"));
    }

    #[test]
    fn membership_move_does_not_rekey_chunk_identity() {
        let chunk_id = "doc#3";
        let mut clusters = vec![
            Cluster {
                centroid: vec![1.0],
                record_ids: vec![chunk_id.to_string()],
            },
            Cluster {
                centroid: vec![0.0],
                record_ids: Vec::new(),
            },
        ];
        let before = rebuild_record_to_cluster(&clusters);
        assert_eq!(before.get(chunk_id).copied(), Some(0));

        clusters[0].record_ids.clear();
        clusters[1].record_ids.push(chunk_id.to_string());
        let after = rebuild_record_to_cluster(&clusters);
        assert_eq!(after.get(chunk_id).copied(), Some(1));
        assert_eq!(after.len(), 1);
        assert!(
            !after
                .keys()
                .any(|id| id.chars().all(|ch| ch.is_ascii_digit()))
        );
        assert_eq!(clusters[1].record_ids.as_slice(), [chunk_id]);
    }

    fn two_vector_index() -> CentroidIndex {
        CentroidIndex::new(
            SPFreshConfig {
                num_centroids: 1,
                max_cluster_records: None,
                ..SPFreshConfig::default()
            },
            2,
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn split_and_merge_move_membership_without_rekeying_records() {
        let index = two_vector_index();
        let left = EmbeddingRecord::new("doc#0", vec![1.0, 0.0]);
        let right = EmbeddingRecord::new("doc#1", vec![0.0, 1.0]);
        VectorIndex::add(&index, left).await.expect("add left");
        VectorIndex::add(&index, right).await.expect("add right");
        assert_eq!(index.clusters.read().len(), 1);

        let body_key = format!("bodies/demo/doc#0");
        let shards = crate::distributed::sharding::ShardManager::new(4);
        let shard_before = shards.shard_index_for_key("doc#0");
        let split = index.split_cluster(0).expect("split");
        assert_eq!(split.source, 0);
        assert_eq!(split.created, 1);
        assert!(!split.moved_chunk_ids.is_empty());
        assert_eq!(index.clusters.read().len(), 2);
        assert_eq!(index.cluster_summaries.read().len(), 2);
        for id in &split.moved_chunk_ids {
            assert_eq!(index.record_to_cluster.read().get(id).copied(), Some(1));
            assert!(index.records.read().contains_key(id));
        }
        assert_eq!(
            index
                .records
                .read()
                .get("doc#0")
                .expect("record")
                .vector
                .as_ref(),
            &[1.0, 0.0]
        );
        assert_eq!(body_key, "bodies/demo/doc#0");
        assert!(!body_key.contains("cluster"));
        assert_eq!(shards.shard_index_for_key("doc#0"), shard_before);

        let merged = index.merge_clusters(0, 1).expect("merge");
        assert_eq!(merged.removed, 1);
        assert_eq!(index.clusters.read().len(), 1);
        assert_eq!(index.record_to_cluster.read().len(), 2);
        assert!(index.records.read().contains_key("doc#0"));
        assert!(index.records.read().contains_key("doc#1"));
        assert_eq!(
            index
                .records
                .read()
                .get("doc#1")
                .expect("record")
                .vector
                .as_ref(),
            &[0.0, 1.0]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn split_of_one_member_leaves_membership_unchanged() {
        let index = two_vector_index();
        VectorIndex::add(&index, EmbeddingRecord::new("doc#0", vec![1.0, 0.0]))
            .await
            .expect("add");
        let err = index.split_cluster(0).expect_err("one member");
        assert!(err.to_string().contains("was not changed"));
        assert_eq!(index.clusters.read().len(), 1);
        assert_eq!(
            index.clusters.read()[0].record_ids.as_slice(),
            ["doc#0".to_string()]
        );
        assert!(index.records.read().contains_key("doc#0"));
    }
}
