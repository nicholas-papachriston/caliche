//! Native-filter postings and per-cluster attribute summaries.
//!
//! [`ClusterPostingState`] is a bitmap of [`ClusterIndex`] values. It is the
//! filtered-ANN path: intersect cluster bitmaps, probe centroids, score the
//! records in those clusters. It does not store chunk envelopes, `content`,
//! or per-chunk posting lists. Full-text and PLAID stay outside
//! [`super::CentroidIndex`]. A non-ANN plan that needs its own block size
//! uses [`super::id_plan::IdPostingPlan`], which addresses chunk ids or dense
//! doc ids and does not change [`super::SPFreshConfig::num_centroids`].

use super::{CentroidIndex, ClusterFilterSummary};
use crate::embedding::EmbeddingRecord;
use crate::errors::DripstoneError;
use crate::search::{
    EqualityTerm, NativeFilterKey, NativeFilterPlan, collect_filter_keys, json_key,
};
use roaring::RoaringBitmap;
use std::collections::HashMap;

/// Cluster slot stored in [`ClusterPostingState`] bitmaps.
///
/// This is not a chunk id, a dense doc id, or a cluster-local document slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ClusterIndex(u32);

impl ClusterIndex {
    pub(crate) fn from_usize(cluster_idx: usize) -> Result<Self, DripstoneError> {
        u32::try_from(cluster_idx)
            .map(Self)
            .map_err(|_| DripstoneError::storage("SPFresh cluster id exceeds u32 range"))
    }

    pub(crate) const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Cluster-index bitmaps for filtered ANN.
///
/// Insert paths take a [`ClusterIndex`]. There is no API for a document body,
/// `content`, or a per-chunk posting list.
#[derive(Debug, Default)]
pub struct ClusterPostingState {
    pub(crate) namespace_postings: HashMap<String, RoaringBitmap>,
    pub(crate) field_value_postings: HashMap<NativeFilterKey, RoaringBitmap>,
}

impl ClusterPostingState {
    pub(crate) fn cluster_index(cluster_idx: usize) -> Result<ClusterIndex, DripstoneError> {
        ClusterIndex::from_usize(cluster_idx)
    }

    fn insert_cluster(bitmap: &mut RoaringBitmap, cluster: ClusterIndex) {
        bitmap.insert(cluster.as_u32());
    }

    pub(crate) fn rebuild_cluster_postings(
        cluster_summaries: &[ClusterFilterSummary],
    ) -> Result<Self, DripstoneError> {
        let mut postings = Self::default();
        for (cluster_idx, summary) in cluster_summaries.iter().enumerate() {
            let cluster_id = Self::cluster_index(cluster_idx)?;
            for namespace in summary.namespace_counts.keys() {
                let bitmap = postings
                    .namespace_postings
                    .entry(namespace.clone())
                    .or_default();
                Self::insert_cluster(bitmap, cluster_id);
            }
            for key in summary.field_value_counts.keys() {
                let bitmap = postings
                    .field_value_postings
                    .entry(key.clone())
                    .or_default();
                Self::insert_cluster(bitmap, cluster_id);
            }
        }
        Ok(postings)
    }
}

impl CentroidIndex {
    pub(crate) fn insert_record_filter_terms(
        summary: &mut ClusterFilterSummary,
        cluster_postings: &mut ClusterPostingState,
        cluster_idx: usize,
        record: &EmbeddingRecord,
    ) -> Result<(), DripstoneError> {
        let cluster_id = ClusterPostingState::cluster_index(cluster_idx)?;
        let namespace = record.namespace().to_string();
        let namespace_count = summary
            .namespace_counts
            .entry(namespace.clone())
            .or_insert(0);
        *namespace_count = namespace_count
            .checked_add(1)
            .ok_or_else(|| DripstoneError::storage("SPFresh namespace count overflow"))?;
        if *namespace_count == 1 {
            let bitmap = cluster_postings
                .namespace_postings
                .entry(namespace)
                .or_default();
            ClusterPostingState::insert_cluster(bitmap, cluster_id);
        }

        if let Some(metadata) = &record.metadata {
            let mut keys = Vec::new();
            collect_filter_keys(metadata, None, &mut keys)?;
            for key in keys {
                let count = summary.field_value_counts.entry(key.clone()).or_insert(0);
                *count = count
                    .checked_add(1)
                    .ok_or_else(|| DripstoneError::storage("SPFresh field count overflow"))?;
                if *count == 1 {
                    let bitmap = cluster_postings
                        .field_value_postings
                        .entry(key)
                        .or_default();
                    ClusterPostingState::insert_cluster(bitmap, cluster_id);
                }
            }
        }

        Ok(())
    }

    pub(crate) fn remove_record_filter_terms(
        summary: &mut ClusterFilterSummary,
        cluster_postings: &mut ClusterPostingState,
        cluster_idx: usize,
        record: &EmbeddingRecord,
    ) -> Result<(), DripstoneError> {
        let cluster_id = ClusterPostingState::cluster_index(cluster_idx)?.as_u32();
        let namespace = record.namespace().to_string();
        if let Some(count) = summary.namespace_counts.get_mut(&namespace) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                summary.namespace_counts.remove(&namespace);
                if let Some(bitmap) = cluster_postings.namespace_postings.get_mut(&namespace) {
                    bitmap.remove(cluster_id);
                    if bitmap.is_empty() {
                        cluster_postings.namespace_postings.remove(&namespace);
                    }
                }
            }
        }

        if let Some(metadata) = &record.metadata {
            let mut keys = Vec::new();
            collect_filter_keys(metadata, None, &mut keys)?;
            for key in keys {
                if let Some(count) = summary.field_value_counts.get_mut(&key) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        summary.field_value_counts.remove(&key);
                        if let Some(bitmap) = cluster_postings.field_value_postings.get_mut(&key) {
                            bitmap.remove(cluster_id);
                            if bitmap.is_empty() {
                                cluster_postings.field_value_postings.remove(&key);
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn allowed_clusters_from_plan(
        cluster_postings: &ClusterPostingState,
        plan: &NativeFilterPlan,
    ) -> Result<Option<RoaringBitmap>, DripstoneError> {
        match plan {
            NativeFilterPlan::MatchAll | NativeFilterPlan::Unsupported => Ok(None),
            NativeFilterPlan::EqualityConjunction(plan) => Ok(Some(
                Self::allowed_clusters_for_conjunction(cluster_postings, plan)?,
            )),
            NativeFilterPlan::Disjunction(conjunctions) => {
                let mut union = RoaringBitmap::new();
                for conjunction in conjunctions {
                    union |= Self::allowed_clusters_for_conjunction(cluster_postings, conjunction)?;
                }
                Ok(Some(union))
            }
        }
    }

    pub(crate) fn allowed_clusters_for_conjunction(
        cluster_postings: &ClusterPostingState,
        plan: &crate::search::EqualityConjunctionPlan,
    ) -> Result<RoaringBitmap, DripstoneError> {
        let mut allowed = plan.namespace.as_deref().map(|namespace| {
            cluster_postings
                .namespace_postings
                .get(namespace)
                .cloned()
                .unwrap_or_default()
        });

        for term in &plan.terms {
            let posting = match term {
                EqualityTerm::Equal { field, value } => cluster_postings
                    .field_value_postings
                    .get(&NativeFilterKey {
                        field: field.clone(),
                        value_json: json_key(value)?,
                    })
                    .cloned()
                    .unwrap_or_default(),
                EqualityTerm::In { field, values } => {
                    let mut union = RoaringBitmap::new();
                    for value in values {
                        if let Some(bitmap) =
                            cluster_postings.field_value_postings.get(&NativeFilterKey {
                                field: field.clone(),
                                value_json: json_key(value)?,
                            })
                        {
                            union |= bitmap;
                        }
                    }
                    union
                }
            };

            match allowed.as_mut() {
                Some(existing) => *existing &= posting,
                None => allowed = Some(posting),
            }
        }

        Ok(allowed.unwrap_or_default())
    }
}

#[cfg(test)]
mod tests {
    use super::{CentroidIndex, ClusterFilterSummary, ClusterPostingState};
    use crate::embedding::EmbeddingRecord;

    #[test]
    fn cluster_postings_store_cluster_indices_only() {
        let mut summary = ClusterFilterSummary::default();
        let mut postings = ClusterPostingState::default();
        let record = EmbeddingRecord::new("doc#0", vec![1.0_f32])
            .with_namespace("demo")
            .with_metadata(serde_json::json!({
                "category": "greeting",
            }));
        CentroidIndex::insert_record_filter_terms(&mut summary, &mut postings, 3, &record)
            .expect("insert cluster index");

        let mut clusters = Vec::new();
        for bitmap in postings.namespace_postings.values() {
            clusters.extend(bitmap.iter());
        }
        for bitmap in postings.field_value_postings.values() {
            clusters.extend(bitmap.iter());
        }
        clusters.sort_unstable();
        clusters.dedup();
        assert_eq!(clusters, vec![3]);

        let rendered = format!("{postings:?}");
        assert!(!rendered.contains("doc#0"));
    }
}
