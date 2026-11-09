//! Non-ANN posting plans for chunk ids or dense doc ids.
//!
//! [`super::postings::ClusterPostingState`] stays a bitmap of cluster indices
//! used by filtered ANN. This plan is a separate type: its block size is not
//! derived from [`super::SPFreshConfig::num_centroids`] or from a ColdForge
//! centroid count. Full-text (`FullTextIndexConfig.block_size`) and PLAID
//! (`candidate_budget`) stay outside [`super::CentroidIndex`] and follow the
//! same rule.

#![allow(
    dead_code,
    reason = "non-ANN boundary type; unit tests lock its block size against num_centroids, and filtered ANN does not construct it"
)]

/// Who a non-ANN posting list points at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PostingIdentity {
    /// `{doc_id}#{chunk_index}`.
    ChunkId,
    /// Dense document id, as in full-text `DocId` to `TextDocumentMeta.chunk_id`.
    DenseDocId,
}

/// Block layout for a plan that does not scan SPFresh clusters.
///
/// Changing [`Self::block_size`] does not change the centroid count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct IdPostingPlan {
    identity: PostingIdentity,
    block_size: usize,
}

impl IdPostingPlan {
    /// Chunk-id postings with an explicit block size. Zero is raised to 1.
    #[must_use]
    pub(crate) const fn chunk_id(block_size: usize) -> Self {
        Self {
            identity: PostingIdentity::ChunkId,
            block_size: if block_size == 0 { 1 } else { block_size },
        }
    }

    /// Dense doc-id postings with an explicit block size. Zero is raised to 1.
    #[must_use]
    pub(crate) const fn dense_doc_id(block_size: usize) -> Self {
        Self {
            identity: PostingIdentity::DenseDocId,
            block_size: if block_size == 0 { 1 } else { block_size },
        }
    }

    #[must_use]
    pub(crate) const fn identity(self) -> PostingIdentity {
        self.identity
    }

    #[must_use]
    pub(crate) const fn block_size(self) -> usize {
        self.block_size
    }
}

impl Default for IdPostingPlan {
    fn default() -> Self {
        // Same default grain as full-text blocks, not `SPFreshConfig::num_centroids`.
        Self::chunk_id(32)
    }
}

#[cfg(test)]
mod tests {
    use super::IdPostingPlan;
    use crate::fts::FullTextIndexConfig;
    use crate::index::plaid::PlaidConfig;
    use crate::index::spfresh::SPFreshConfig;

    #[test]
    fn block_size_does_not_change_centroid_count() {
        let plan = IdPostingPlan::default();
        assert_eq!(plan.identity(), super::PostingIdentity::ChunkId);
        assert_eq!(plan.block_size(), 32);

        let centroids = SPFreshConfig::default();
        assert_eq!(centroids.num_centroids, 256);
        assert_ne!(plan.block_size(), centroids.num_centroids);

        let dense = IdPostingPlan::dense_doc_id(7);
        assert_eq!(dense.block_size(), 7);
        assert_eq!(dense.identity(), super::PostingIdentity::DenseDocId);
        assert_eq!(centroids.num_centroids, 256);
        assert_ne!(dense.block_size(), centroids.num_centroids);

        assert_eq!(FullTextIndexConfig::default().block_size, 32);
        assert_ne!(
            FullTextIndexConfig::default().block_size,
            SPFreshConfig::default().num_centroids
        );
        assert_eq!(PlaidConfig::default().candidate_budget, 256);
        assert_ne!(
            PlaidConfig::default().candidate_budget,
            PlaidConfig::default().num_centroids
        );
        assert_eq!(IdPostingPlan::chunk_id(0).block_size(), 1);
        assert_eq!(SPFreshConfig::default().num_centroids, 256);
    }
}
