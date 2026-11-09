//! Vector indexing algorithms for efficient similarity search
//!
//! This module provides pluggable indexing strategies optimized for different
//! storage backends and performance characteristics.

#[cfg(feature = "hnsw")]
pub mod hnsw;
pub mod plaid;
pub mod spfresh;

#[cfg(feature = "hnsw")]
pub use hnsw::{HnswConfig, HnswGraphSpace, HnswIndex, HnswIndexStats};
pub use plaid::{PlaidConfig, PlaidIndex};

use std::sync::Arc;

use crate::core::{Chunk, Metric};
use crate::embedding::{EmbeddingRecord, SimilarityMetric, SimilarityResult};
use crate::errors::DripstoneError;
use crate::search::NativeFilterPlan;
use async_trait::async_trait;

/// Strategy for vector indexing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IndexStrategy {
    /// Brute force linear search - accurate but slow for large datasets
    #[default]
    BruteForce,
    /// `SPFresh` centroid-based indexing - optimized for object storage
    SPFresh,
    /// HNSW graph-based indexing - fast approximate nearest neighbor search
    HNSW,
}

impl From<&Chunk> for EmbeddingRecord {
    fn from(chunk: &Chunk) -> Self {
        let mut record = Self::from_arc(&*chunk.id, Arc::clone(&chunk.vector));
        if let Some(ns) = chunk.namespace.as_deref() {
            record = record.with_namespace(ns);
        }
        if chunk.metadata != serde_json::Value::Null {
            record = record.with_metadata(chunk.metadata.clone());
        }
        if let Some(ref mv) = chunk.multi_vectors {
            record = record.with_multi_vectors(Arc::clone(mv));
        }
        record
    }
}

/// Maps core search metrics to index-layer similarity metrics.
///
/// `Chebyshev` exists only on [`SimilarityMetric`]; backends that need it (Qdrant,
/// Postgres, Weaviate) handle fallbacks when mapping `SimilarityMetric` to native APIs.
impl From<Metric> for SimilarityMetric {
    fn from(m: Metric) -> Self {
        match m {
            Metric::Cosine => Self::Cosine,
            Metric::DotProduct => Self::Dot,
            Metric::Euclidean => Self::Euclidean,
            Metric::Manhattan => Self::Manhattan,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IndexSearchRequest<'a> {
    pub query: &'a [f32],
    pub limit: usize,
    pub metric: SimilarityMetric,
    pub native_filter: Option<&'a NativeFilterPlan>,
}

/// Common interface for vector index implementations
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// Build or update the index with the given records
    async fn build(&self, records: &[EmbeddingRecord]) -> Result<(), DripstoneError>;

    /// Search the index for similar vectors
    async fn search_request(
        &self,
        request: IndexSearchRequest<'_>,
    ) -> Result<Vec<SimilarityResult>, DripstoneError>;

    /// Search the index for similar vectors
    async fn search(
        &self,
        query: &[f32],
        limit: usize,
        metric: SimilarityMetric,
    ) -> Result<Vec<SimilarityResult>, DripstoneError> {
        self.search_request(IndexSearchRequest {
            query,
            limit,
            metric,
            native_filter: None,
        })
        .await
    }

    /// Add a single record to the index incrementally
    async fn add(&self, record: EmbeddingRecord) -> Result<(), DripstoneError>;

    /// Remove a record from the index
    async fn remove(&self, id: &str) -> Result<(), DripstoneError>;

    /// Get the number of vectors in the index
    fn len(&self) -> usize;

    /// Check if the index is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all data from the index (synchronous).
    ///
    /// Prefer [`Self::build`] with an empty slice from async code (for example
    /// [`crate::backend::indexed::IndexedBackend::rebuild_index`]). [`HnswIndex`]
    /// implements `clear` by resetting through `build(&[])` so Tokio worker threads
    /// never call `RwLock::blocking_write`.
    fn clear(&self);

    /// Update HNSW `ef_search` at query time (no-op for indexes that do not use it).
    fn set_ef_search(&self, _ef: usize) {}

    /// Stream all stored records. Used by backends that need to enumerate rather
    /// than perform an approximate search (e.g. `list` / `delete_document`).
    ///
    /// Implementations should return every currently-indexed record exactly once.
    /// Namespace filtering is handled by [`Self::iter_records_by_namespace`].
    async fn iter_records(&self) -> Result<Vec<Arc<EmbeddingRecord>>, DripstoneError>;

    /// Like [`Self::iter_records`] but filters by namespace.
    ///
    /// Pass `None` to include records whose namespace is unset.
    /// Pass `Some(ns)` to include only records whose namespace matches `ns`.
    async fn iter_records_by_namespace(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<Arc<EmbeddingRecord>>, DripstoneError>;
}
