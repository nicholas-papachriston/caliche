//! Configuration types for [`super::IndexedBackend`].

use crate::embedding::EmbeddingRecord;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexUpdateMode {
    /// Write to the inner backend and update the ANN index in the same `upsert` call.
    #[default]
    Synchronous,
    /// Commit vectors to the inner backend first and queue ANN inserts separately.
    ///
    /// Search can merge index hits with a brute-force scan of the pending tail when
    /// [`IndexedBackendConfig::scan_unindexed_tail`] is enabled (read-your-write).
    /// Call [`IndexedBackend::flush_pending_index`] or rely on
    /// [`IndexedBackendConfig::deferred_flush_threshold`] to move work off the hot path.
    Deferred,
}

/// Configuration for an [`IndexedBackend`].
#[derive(Debug, Clone)]
pub struct IndexedBackendConfig {
    /// Rebuild the full index after this many incremental upserts.
    /// Set to `0` to disable periodic rebuilds (incremental-only).
    pub auto_rebuild_threshold: usize,
    /// When the index is empty, fall back to the inner backend's brute-force
    /// `search()` instead of returning an empty result set.
    pub fallback_to_brute_force: bool,
    /// Over-fetch multiplier: the index retrieves `limit * overfetch` candidates
    /// so that downstream namespace/metadata filtering still returns enough hits.
    pub overfetch_factor: usize,
    /// When to update the ANN index relative to inner storage writes.
    pub index_update_mode: IndexUpdateMode,
    /// When [`Self::index_update_mode`] is [`IndexUpdateMode::Deferred`], scan vectors
    /// that are committed but not yet indexed on each search so recent upserts stay
    /// visible. Disable only when callers accept bounded visibility lag.
    pub scan_unindexed_tail: bool,
    /// When [`Self::index_update_mode`] is [`IndexUpdateMode::Deferred`], flush the
    /// pending queue into the ANN index after this many queued records. `0` disables
    /// automatic flush (use explicit [`IndexedBackend::flush_pending_index`]).
    pub deferred_flush_threshold: usize,
    /// When [`Self::index_update_mode`] is [`IndexUpdateMode::Deferred`], flush the
    /// pending queue at the start of `search()` when pending count reaches this threshold.
    /// `0` disables search-triggered flush.
    pub deferred_search_flush_threshold: usize,
    /// When [`Self::index_update_mode`] is [`IndexUpdateMode::Deferred`] and tail scan is
    /// enabled, score at most the last N pending records. `0` means scan the full queue.
    pub max_tail_scan_records: usize,
}

impl Default for IndexedBackendConfig {
    fn default() -> Self {
        Self {
            auto_rebuild_threshold: 1000,
            fallback_to_brute_force: true,
            overfetch_factor: 2,
            index_update_mode: IndexUpdateMode::Synchronous,
            scan_unindexed_tail: true,
            deferred_flush_threshold: 256,
            deferred_search_flush_threshold: 0,
            max_tail_scan_records: 0,
        }
    }
}

pub(crate) struct DeferredIndexState {
    pub(crate) pending: tokio::sync::RwLock<Vec<EmbeddingRecord>>,
}
