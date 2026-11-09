//! HNSW (Hierarchical Navigable Small World) index implementation.
//!
//! Provides state-of-the-art approximate nearest neighbor search using
//! hierarchical graph structures for fast similarity search.

use crate::embedding::{EmbeddingRecord, SimilarityMetric, SimilarityResult};
use crate::errors::DripstoneError;
use crate::index::{IndexSearchRequest, VectorIndex};
use crate::search::{
    EqualityTerm, NativeFilterKey, NativeFilterPlan, collect_filter_keys, json_key,
};
use crate::simd;
use async_trait::async_trait;
use hnsw_rs::prelude::*;
use roaring::RoaringBitmap;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::RwLock;

#[derive(Debug, Default)]
struct NativeFilterState {
    namespace_postings: HashMap<String, RoaringBitmap>,
    field_value_postings: HashMap<NativeFilterKey, RoaringBitmap>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct HnswIndexData {
    id_to_index: HashMap<String, usize>,
    index_to_id: HashMap<usize, String>,
    records: Vec<(String, EmbeddingRecord)>,
    dimension: usize,
    config: HnswConfig,
}

/// How vectors are represented inside the HNSW graph.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum HnswGraphSpace {
    /// Insert raw vectors directly into the L2 graph.
    RawL2,
    /// Normalize vectors first so L2 traversal matches cosine neighborhoods.
    CosineCompatible,
}

/// Configuration for HNSW index
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswConfig {
    /// Maximum number of connections for each node (M parameter)
    pub m: usize,
    /// Size of the dynamic candidate list (`ef_construction` parameter)
    pub ef_construction: usize,
    /// Size of the dynamic candidate list for search (`ef_search` parameter)
    pub ef_search: usize,
    /// Maximum number of layers
    pub max_nb_connection: usize,
    /// Candidate multiplier before exact reranking.
    pub rerank_overfetch: usize,
    /// How vectors are represented inside the graph.
    pub graph_space: HnswGraphSpace,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            max_nb_connection: 16,
            rerank_overfetch: 4,
            graph_space: HnswGraphSpace::RawL2,
        }
    }
}

/// HNSW index implementation
///
/// # Memory Optimization
///
/// Records are stored as `Arc<EmbeddingRecord>` to enable cheap cloning
/// when returning search results. This avoids copying ~6KB vectors per result.
pub struct HnswIndex {
    /// HNSW graph structure
    hnsw: Arc<RwLock<Hnsw<'static, f32, DistL2>>>,
    /// Mapping from record ID to internal index
    id_to_index: Arc<RwLock<HashMap<String, usize>>>,
    /// Mapping from internal index to record ID
    index_to_id: Arc<RwLock<HashMap<usize, String>>>,
    /// All records indexed by ID - stored as Arc for cheap cloning in search results
    records: Arc<RwLock<HashMap<String, Arc<EmbeddingRecord>>>>,
    /// Sidecar postings used to compile metadata filters into HNSW id constraints.
    native_filters: Arc<RwLock<NativeFilterState>>,
    /// Cached record count so `len()` can be non-blocking when called from async context
    length: AtomicUsize,
    /// Monotonic allocator for internal ids so incremental add/remove does not reuse live ids.
    next_index: AtomicUsize,
    /// Vector dimension
    dimension: usize,
    /// Configuration
    config: HnswConfig,
    /// Query-time `ef_search` (may be updated after build via [`Self::set_ef_search`]).
    ef_search_live: AtomicUsize,
}

impl HnswIndex {
    /// Create a new HNSW index
    #[must_use]
    pub fn new(config: HnswConfig, dimension: usize) -> Self {
        let dist = DistL2 {};
        let hnsw = Hnsw::new(
            config.max_nb_connection,
            config.m,
            config.ef_construction,
            dimension,
            dist,
        );

        let ef_search = config.ef_search.max(1);
        Self {
            hnsw: Arc::new(RwLock::new(hnsw)),
            id_to_index: Arc::new(RwLock::new(HashMap::new())),
            index_to_id: Arc::new(RwLock::new(HashMap::new())),
            records: Arc::new(RwLock::new(HashMap::new())),
            native_filters: Arc::new(RwLock::new(NativeFilterState::default())),
            length: AtomicUsize::new(0),
            next_index: AtomicUsize::new(0),
            dimension,
            config,
            ef_search_live: AtomicUsize::new(ef_search),
        }
    }

    /// Serialize index to bytes for storage (sync; do not call from async context).
    ///
    /// Note: This serializes metadata only. The HNSW graph will be rebuilt on deserialization.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DripstoneError> {
        let id_to_index = self.id_to_index.blocking_read().clone();
        let index_to_id = self.index_to_id.blocking_read().clone();
        // Dereference Arc to get owned EmbeddingRecord for serialization
        let records: Vec<(String, EmbeddingRecord)> = self
            .records
            .blocking_read()
            .iter()
            .map(|(k, v)| (k.clone(), (**v).clone()))
            .collect();

        let data = HnswIndexData {
            id_to_index,
            index_to_id,
            records,
            dimension: self.dimension,
            config: self.config.clone(),
        };

        serde_json::to_vec(&data)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize HNSW index: {e}")))
    }

    /// Serialize index to bytes (async; safe to call from async context).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn to_bytes_async(&self) -> Result<Vec<u8>, DripstoneError> {
        let id_to_index = self.id_to_index.read().await.clone();
        let index_to_id = self.index_to_id.read().await.clone();
        let records: Vec<(String, EmbeddingRecord)> = self
            .records
            .read()
            .await
            .iter()
            .map(|(k, v)| (k.clone(), (**v).clone()))
            .collect();

        let data = HnswIndexData {
            id_to_index,
            index_to_id,
            records,
            dimension: self.dimension,
            config: self.config.clone(),
        };

        serde_json::to_vec(&data)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize HNSW index: {e}")))
    }

    /// Deserialize index from bytes (sync; do not call from async context).
    ///
    /// Note: The HNSW graph will be rebuilt from the stored records.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, DripstoneError> {
        let data: HnswIndexData = serde_json::from_slice(bytes).map_err(|e| {
            DripstoneError::storage(format!("Failed to deserialize HNSW index: {e}"))
        })?;

        let index = Self::new(data.config.clone(), data.dimension);

        // Rebuild the HNSW graph from records
        {
            let hnsw = index.hnsw.blocking_write();
            let mut id_to_index = index.id_to_index.blocking_write();
            let mut index_to_id = index.index_to_id.blocking_write();
            let mut records = index.records.blocking_write();
            let mut native_filters = index.native_filters.blocking_write();

            id_to_index.clone_from(&data.id_to_index);
            index_to_id.clone_from(&data.index_to_id);
            native_filters.namespace_postings.clear();
            native_filters.field_value_postings.clear();

            // Rebuild records map and graph
            for (id, record) in data.records {
                let internal_idx = id_to_index.get(&id).copied().ok_or_else(|| {
                    DripstoneError::storage(format!(
                        "HNSW snapshot corrupt: record id {id} missing from id_to_index"
                    ))
                })?;
                // Wrap record in Arc for efficient sharing
                let record_arc = Arc::new(record);
                let graph_vector = index.graph_vector_for_record(&record_arc.vector)?;
                Self::insert_record_filter_terms(&mut native_filters, internal_idx, &record_arc)?;
                records.insert(id.clone(), Arc::clone(&record_arc));

                // Insert into HNSW graph
                hnsw.insert((graph_vector.as_ref(), internal_idx));
            }
            let record_count = records.len();
            let next_index = id_to_index
                .values()
                .copied()
                .max()
                .map_or(0, |max_idx| max_idx.saturating_add(1));
            drop((native_filters, records, index_to_id, id_to_index, hnsw));
            index.length.store(record_count, Ordering::Relaxed);
            index.next_index.store(next_index, Ordering::Relaxed);
        }

        Ok(index)
    }

    /// Deserialize index from bytes (async; safe to call from async context).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn from_bytes_async(bytes: &[u8]) -> Result<Self, DripstoneError> {
        let data: HnswIndexData = serde_json::from_slice(bytes).map_err(|e| {
            DripstoneError::storage(format!("Failed to deserialize HNSW index: {e}"))
        })?;

        let index = Self::new(data.config.clone(), data.dimension);

        {
            let hnsw = index.hnsw.write().await;
            let mut id_to_index = index.id_to_index.write().await;
            let mut index_to_id = index.index_to_id.write().await;
            let mut records = index.records.write().await;
            let mut native_filters = index.native_filters.write().await;

            id_to_index.clone_from(&data.id_to_index);
            index_to_id.clone_from(&data.index_to_id);
            native_filters.namespace_postings.clear();
            native_filters.field_value_postings.clear();

            for (id, record) in data.records {
                let internal_idx = id_to_index.get(&id).copied().ok_or_else(|| {
                    DripstoneError::storage(format!(
                        "HNSW snapshot corrupt: record id {id} missing from id_to_index"
                    ))
                })?;
                let record_arc = Arc::new(record);
                let graph_vector = index.graph_vector_for_record(&record_arc.vector)?;
                Self::insert_record_filter_terms(&mut native_filters, internal_idx, &record_arc)?;
                records.insert(id.clone(), Arc::clone(&record_arc));
                hnsw.insert((graph_vector.as_ref(), internal_idx));
            }
            let record_count = records.len();
            let next_index = id_to_index
                .values()
                .copied()
                .max()
                .map_or(0, |max_idx| max_idx.saturating_add(1));
            drop((native_filters, records, index_to_id, id_to_index, hnsw));
            index.length.store(record_count, Ordering::Relaxed);
            index.next_index.store(next_index, Ordering::Relaxed);
        }

        Ok(index)
    }

    /// Get index statistics (non-blocking; safe to call from async context)
    pub fn stats(&self) -> HnswIndexStats {
        let len = self.length.load(Ordering::Relaxed);
        HnswIndexStats {
            num_records: len,
            dimension: self.dimension,
            num_indexed: len,
        }
    }

    fn normalize_for_hnsw(vector: &[f32]) -> Result<Vec<f32>, DripstoneError> {
        let mut normalized = vector.to_vec();
        simd::l2_normalize(&mut normalized)?;
        Ok(normalized)
    }

    fn graph_vector_for_record<'a>(
        &self,
        record_vector: &'a Arc<[f32]>,
    ) -> Result<Cow<'a, [f32]>, DripstoneError> {
        match self.config.graph_space {
            HnswGraphSpace::RawL2 => Ok(Cow::Borrowed(record_vector.as_ref())),
            HnswGraphSpace::CosineCompatible => {
                Ok(Cow::Owned(Self::normalize_for_hnsw(record_vector)?))
            }
        }
    }

    fn graph_query_vector<'a>(
        &self,
        query: &'a [f32],
        metric: SimilarityMetric,
    ) -> Result<Cow<'a, [f32]>, DripstoneError> {
        if matches!(self.config.graph_space, HnswGraphSpace::CosineCompatible)
            && matches!(metric, SimilarityMetric::Cosine)
        {
            return Ok(Cow::Owned(Self::normalize_for_hnsw(query)?));
        }

        Ok(Cow::Borrowed(query))
    }

    fn candidate_limit(&self, limit: usize, total_records: usize) -> usize {
        let overfetch = self.config.rerank_overfetch.max(1);
        limit
            .saturating_mul(overfetch)
            .max(limit)
            .min(total_records)
    }

    fn insert_record_filter_terms(
        native_filters: &mut NativeFilterState,
        internal_idx: usize,
        record: &EmbeddingRecord,
    ) -> Result<(), DripstoneError> {
        let internal_idx = u32::try_from(internal_idx)
            .map_err(|_| DripstoneError::storage("Internal HNSW id exceeds u32 range"))?;
        native_filters
            .namespace_postings
            .entry(record.namespace().to_string())
            .or_default()
            .insert(internal_idx);

        if let Some(metadata) = &record.metadata {
            let mut keys = Vec::new();
            collect_filter_keys(metadata, None, &mut keys)?;
            for key in keys {
                native_filters
                    .field_value_postings
                    .entry(key)
                    .or_default()
                    .insert(internal_idx);
            }
        }

        Ok(())
    }

    fn remove_record_filter_terms(
        native_filters: &mut NativeFilterState,
        internal_idx: usize,
        record: &EmbeddingRecord,
    ) -> Result<(), DripstoneError> {
        let internal_idx = u32::try_from(internal_idx)
            .map_err(|_| DripstoneError::storage("Internal HNSW id exceeds u32 range"))?;
        let namespace_key = record.namespace().to_string();
        if let Some(bitmap) = native_filters.namespace_postings.get_mut(&namespace_key) {
            bitmap.remove(internal_idx);
            if bitmap.is_empty() {
                native_filters.namespace_postings.remove(&namespace_key);
            }
        }

        if let Some(metadata) = &record.metadata {
            let mut keys = Vec::new();
            collect_filter_keys(metadata, None, &mut keys)?;
            for key in keys {
                if let Some(bitmap) = native_filters.field_value_postings.get_mut(&key) {
                    bitmap.remove(internal_idx);
                    if bitmap.is_empty() {
                        native_filters.field_value_postings.remove(&key);
                    }
                }
            }
        }

        Ok(())
    }

    fn allowed_ids_from_plan(
        native_filters: &NativeFilterState,
        plan: &NativeFilterPlan,
    ) -> Result<Option<Vec<usize>>, DripstoneError> {
        let allowed = match plan {
            NativeFilterPlan::MatchAll | NativeFilterPlan::Unsupported => None,
            NativeFilterPlan::EqualityConjunction(plan) => {
                Some(Self::allowed_ids_for_conjunction(native_filters, plan)?)
            }
            NativeFilterPlan::Disjunction(conjunctions) => {
                let mut union = RoaringBitmap::new();
                for conjunction in conjunctions {
                    union |= Self::allowed_ids_for_conjunction(native_filters, conjunction)?;
                }
                Some(union)
            }
        };

        let allowed = allowed.map(|bitmap| {
            let mut ids = bitmap
                .into_iter()
                .filter_map(|id| usize::try_from(id).ok())
                .collect::<Vec<_>>();
            // `hnsw_rs::FilterT` for `Vec<usize>` binary-searches this list.
            ids.sort_unstable();
            ids
        });
        Ok(allowed)
    }

    fn score_indexed_record(
        idx: usize,
        index_to_id: &HashMap<usize, String>,
        records: &HashMap<String, Arc<EmbeddingRecord>>,
        prepared: &simd::PreparedQueryScorer<'_>,
    ) -> Option<SimilarityResult> {
        let id = index_to_id.get(&idx)?;
        let record_arc = records.get(id)?;
        let score = match prepared.score_dense(&record_arc.vector) {
            Ok(score) => score,
            Err(error) => {
                tracing::debug!(
                    id = %id,
                    error = %error,
                    "HNSW rerank: metric comparison failed for candidate"
                );
                return None;
            }
        };
        Some(SimilarityResult::new(Arc::clone(record_arc), score))
    }

    fn allowed_ids_for_conjunction(
        native_filters: &NativeFilterState,
        plan: &crate::search::EqualityConjunctionPlan,
    ) -> Result<RoaringBitmap, DripstoneError> {
        let mut allowed = plan.namespace.as_deref().map(|namespace| {
            native_filters
                .namespace_postings
                .get(namespace)
                .cloned()
                .unwrap_or_default()
        });

        for term in &plan.terms {
            let posting = match term {
                EqualityTerm::Equal { field, value } => native_filters
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
                            native_filters.field_value_postings.get(&NativeFilterKey {
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

/// Index statistics
#[derive(Debug, Clone)]
pub struct HnswIndexStats {
    pub num_records: usize,
    pub dimension: usize,
    pub num_indexed: usize,
}

impl HnswIndex {
    /// Reset maps and graph using async locks (safe on Tokio worker threads).
    async fn reset_state(&self) -> Result<(), DripstoneError> {
        let mut id_to_index = self.id_to_index.write().await;
        let mut index_to_id = self.index_to_id.write().await;
        let mut records_map = self.records.write().await;
        let mut native_filters = self.native_filters.write().await;
        let mut hnsw = self.hnsw.write().await;

        id_to_index.clear();
        index_to_id.clear();
        records_map.clear();
        native_filters.namespace_postings.clear();
        native_filters.field_value_postings.clear();
        self.length.store(0, Ordering::Relaxed);
        self.next_index.store(0, Ordering::Relaxed);
        self.ef_search_live
            .store(self.config.ef_search.max(1), Ordering::Relaxed);

        let dist = DistL2 {};
        *hnsw = Hnsw::new(
            self.config.max_nb_connection,
            self.config.m,
            self.config.ef_construction,
            self.dimension,
            dist,
        );
        drop((hnsw, native_filters, records_map, index_to_id, id_to_index));
        Ok(())
    }

    fn drive_index_future<F>(fut: F) -> Result<(), DripstoneError>
    where
        F: std::future::Future<Output = Result<(), DripstoneError>>,
    {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.block_on(fut)
        } else {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    DripstoneError::storage(format!(
                        "failed to create runtime for HNSW index clear: {e}"
                    ))
                })?;
            rt.block_on(fut)
        }
    }

    fn block_on_index_future<F>(fut: F)
    where
        F: std::future::Future<Output = Result<(), DripstoneError>>,
    {
        let run = || {
            if let Err(err) = Self::drive_index_future(fut) {
                tracing::warn!(error = %err, "HNSW index clear failed");
            }
        };

        if let Ok(handle) = tokio::runtime::Handle::try_current()
            && matches!(
                handle.runtime_flavor(),
                tokio::runtime::RuntimeFlavor::MultiThread
            )
        {
            tokio::task::block_in_place(run);
        } else {
            run();
        }
    }
}

#[async_trait]
impl VectorIndex for HnswIndex {
    async fn build(&self, records: &[EmbeddingRecord]) -> Result<(), DripstoneError> {
        self.reset_state().await?;

        if records.is_empty() {
            return Ok(());
        }

        for record in records {
            self.add(record.clone()).await?;
        }

        Ok(())
    }

    async fn search_request(
        &self,
        request: IndexSearchRequest<'_>,
    ) -> Result<Vec<SimilarityResult>, DripstoneError> {
        let query = request.query;
        let limit = request.limit;
        let metric = request.metric;
        if limit == 0 {
            return Ok(Vec::new());
        }

        if query.len() != self.dimension {
            return Err(DripstoneError::InvalidVectorLength {
                expected: self.dimension,
                actual: query.len(),
            });
        }

        // Keep lock acquisition order consistent with `add()` to avoid reader/writer deadlocks
        // between foreground search and background promotion writes.
        let index_to_id = self.index_to_id.read().await;
        let records = self.records.read().await;
        let native_filters = self.native_filters.read().await;
        if records.is_empty() {
            return Ok(Vec::new());
        }

        let prepared_metric = metric.prepare(query)?;
        let candidate_limit = self.candidate_limit(limit, records.len());
        let allowed_ids = request
            .native_filter
            .filter(|plan| plan.is_supported())
            .map(|plan| Self::allowed_ids_from_plan(&native_filters, plan))
            .transpose()?
            .flatten();
        // `hnsw_rs` draws node layers from OS entropy. Filtered `search_layer`
        // also keeps the entry point until a later neighbor passes the filter,
        // so a tiny allowed set can miss an id that the postings already have.
        // Scoring that set directly is exact and does not depend on the graph.
        let mut candidates = match allowed_ids.as_ref() {
            Some(ids) if ids.len() <= candidate_limit => ids
                .iter()
                .filter_map(|idx| {
                    Self::score_indexed_record(*idx, &index_to_id, &records, &prepared_metric)
                })
                .collect::<Vec<_>>(),
            filtered => {
                let graph_query = self.graph_query_vector(query, metric)?;
                let hnsw = self.hnsw.read().await;
                let filter_ref = filtered.map(|ids| ids as &dyn hnsw_rs::prelude::FilterT);
                let ef = self.ef_search_live.load(Ordering::Relaxed);
                let results =
                    hnsw.search_filter(graph_query.as_ref(), candidate_limit, ef, filter_ref);
                results
                    .into_iter()
                    .filter_map(|result| {
                        let idx = result.d_id;
                        if filtered.is_some_and(|allowed| allowed.binary_search(&idx).is_err()) {
                            return None;
                        }
                        Self::score_indexed_record(idx, &index_to_id, &records, &prepared_metric)
                    })
                    .collect::<Vec<_>>()
            }
        };
        drop((native_filters, records, index_to_id));

        // Sort by score
        if metric.prefers_lower_scores() {
            candidates.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            candidates.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Limit results
        candidates.truncate(limit);
        Ok(candidates)
    }

    async fn add(&self, record: EmbeddingRecord) -> Result<(), DripstoneError> {
        if record.vector.len() != self.dimension {
            return Err(DripstoneError::InvalidVectorLength {
                expected: self.dimension,
                actual: record.vector.len(),
            });
        }

        let record_id = record.id.to_string();
        let mut id_to_index = self.id_to_index.write().await;
        let mut index_to_id = self.index_to_id.write().await;
        let mut records = self.records.write().await;
        let mut native_filters = self.native_filters.write().await;
        let hnsw = self.hnsw.write().await;

        // Check if record already exists
        let internal_idx = if let Some(&existing_idx) = id_to_index.get(&record_id) {
            if let Some(existing_record) = records.get(&record_id) {
                Self::remove_record_filter_terms(
                    &mut native_filters,
                    existing_idx,
                    existing_record,
                )?;
            }
            existing_idx
        } else {
            let new_idx = self.next_index.fetch_add(1, Ordering::Relaxed);
            id_to_index.insert(record_id.clone(), new_idx);
            index_to_id.insert(new_idx, record_id.clone());
            self.length.fetch_add(1, Ordering::Relaxed);
            new_idx
        };

        // Wrap record in Arc for efficient storage and retrieval
        let record_arc = Arc::new(record);
        let graph_vector = self.graph_vector_for_record(&record_arc.vector)?;

        hnsw.insert((graph_vector.as_ref(), internal_idx));
        Self::insert_record_filter_terms(&mut native_filters, internal_idx, &record_arc)?;

        records.insert(record_id, record_arc);
        drop((hnsw, native_filters, records, index_to_id, id_to_index));

        Ok(())
    }

    async fn remove(&self, id: &str) -> Result<(), DripstoneError> {
        let mut id_to_index = self.id_to_index.write().await;
        let mut index_to_id = self.index_to_id.write().await;
        let mut records = self.records.write().await;
        let mut native_filters = self.native_filters.write().await;

        if let Some(&internal_idx) = id_to_index.get(id) {
            if let Some(record) = records.get(id) {
                Self::remove_record_filter_terms(&mut native_filters, internal_idx, record)?;
            }
            id_to_index.remove(id);
            index_to_id.remove(&internal_idx);
            records.remove(id);
            self.length.fetch_sub(1, Ordering::Relaxed);
            // Note: HNSW doesn't support efficient deletion, so we mark as removed
            // A full rebuild would be needed for true deletion
        }
        drop((native_filters, records, index_to_id, id_to_index));

        Ok(())
    }

    fn len(&self) -> usize {
        self.length.load(Ordering::Relaxed)
    }

    fn clear(&self) {
        Self::block_on_index_future(self.reset_state());
    }

    fn set_ef_search(&self, ef: usize) {
        self.ef_search_live.store(ef.max(1), Ordering::Relaxed);
    }

    async fn iter_records(&self) -> Result<Vec<Arc<EmbeddingRecord>>, DripstoneError> {
        let records = self.records.read().await;
        Ok(records.values().map(Arc::clone).collect())
    }

    async fn iter_records_by_namespace(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<Arc<EmbeddingRecord>>, DripstoneError> {
        let records = self.records.read().await;
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
mod tests;
