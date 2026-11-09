use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

use crate::errors::DripstoneError;
use crate::search::Filter;
use crate::simd;
use crate::sparse::SparseVector;

/// Embedding vector type using Arc for cheap cloning.
///
/// Vectors are typically 1536 floats (~6KB), so cloning them is expensive.
/// By wrapping in Arc, cloning becomes just an atomic reference count increment.
pub type EmbeddingVector = Arc<[f32]>;

pub type Metadata = serde_json::Value;

pub const DEFAULT_NAMESPACE: &str = "default";

/// Get namespace as a reference, avoiding allocation for default
#[inline]
#[must_use]
pub fn get_namespace_ref(namespace: Option<&str>) -> &str {
    namespace.unwrap_or(DEFAULT_NAMESPACE)
}

/// Embedding record with vector data and metadata
///
/// # Memory Optimization
///
/// Uses `Arc<str>` for `id` and `namespace` fields to enable efficient sharing of
/// string data across many records. This is particularly beneficial when:
///
/// - Many records share the same namespace (common case)
/// - Records are cloned frequently (just increments Arc refcount)
/// - Multiple references to the same record exist in memory
///
/// The `Arc<str>` wrapper is transparent in the API - you can pass regular strings
/// to constructors and they will be automatically converted.
///
/// Internal backends use optimized string handling (`beef::lean::Cow<str>`, thread-local
/// buffers) to minimize allocations in hot paths.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct EmbeddingRecord {
    #[serde(
        serialize_with = "serialize_arc_str",
        deserialize_with = "deserialize_arc_str"
    )]
    pub id: Arc<str>,
    /// Dense embedding vector wrapped in Arc for cheap cloning.
    /// Cloning an `EmbeddingRecord` now only increments reference counts.
    #[serde(
        serialize_with = "serialize_arc_slice",
        deserialize_with = "deserialize_arc_slice"
    )]
    pub vector: EmbeddingVector,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[serde(
        serialize_with = "serialize_option_arc_str",
        deserialize_with = "deserialize_option_arc_str"
    )]
    pub namespace: Option<Arc<str>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sparse_vector: Option<SparseVector>,
    /// Optional late-interaction token vectors (ColBERT-style MaxSim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_vectors: Option<crate::search::MultiVectors>,
    pub created_at: OffsetDateTime,
    pub updated_at: OffsetDateTime,
}

// Serde helpers for Arc<str>
fn serialize_arc_str<S>(arc: &Arc<str>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(arc)
}

fn deserialize_arc_str<'de, D>(deserializer: D) -> Result<Arc<str>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(Arc::from(s))
}

#[expect(
    clippy::ref_option,
    reason = "serde serialize_with requires a shared reference to the field's exact Option type"
)]
fn serialize_option_arc_str<S>(opt: &Option<Arc<str>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match opt {
        Some(arc) => serializer.serialize_some(arc.as_ref()),
        None => serializer.serialize_none(),
    }
}

fn deserialize_option_arc_str<'de, D>(deserializer: D) -> Result<Option<Arc<str>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<String>::deserialize(deserializer)?;
    Ok(opt.map(Arc::from))
}

// Serde helpers for Arc<[f32]>
fn serialize_arc_slice<S>(arc: &Arc<[f32]>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    let mut seq = serializer.serialize_seq(Some(arc.len()))?;
    for element in arc.iter() {
        seq.serialize_element(element)?;
    }
    seq.end()
}

fn deserialize_arc_slice<'de, D>(deserializer: D) -> Result<Arc<[f32]>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let vec = Vec::<f32>::deserialize(deserializer)?;
    Ok(Arc::from(vec))
}

impl EmbeddingRecord {
    /// Create a new embedding record from a vector.
    ///
    /// Accepts either `Vec<f32>` or `Arc<[f32]>` for the vector parameter.
    /// The vector is stored as `Arc<[f32]>` for cheap cloning.
    pub fn new(id: impl AsRef<str>, vector: impl Into<Arc<[f32]>>) -> Self {
        let timestamp = OffsetDateTime::now_utc();
        Self {
            id: Arc::from(id.as_ref()),
            vector: vector.into(),
            namespace: None,
            metadata: None,
            sparse_vector: None,
            multi_vectors: None,
            created_at: timestamp,
            updated_at: timestamp,
        }
    }

    /// Create a new embedding record from an already-wrapped `Arc` vector.
    /// This is useful when you already have an `Arc<[f32]>` and want to avoid conversion.
    pub fn from_arc(id: impl AsRef<str>, vector: Arc<[f32]>) -> Self {
        let timestamp = OffsetDateTime::now_utc();
        Self {
            id: Arc::from(id.as_ref()),
            vector,
            namespace: None,
            metadata: None,
            sparse_vector: None,
            multi_vectors: None,
            created_at: timestamp,
            updated_at: timestamp,
        }
    }

    /// Sets the optional namespace.
    pub fn with_namespace(mut self, namespace: impl AsRef<str>) -> Self {
        self.namespace = Some(Arc::from(namespace.as_ref()));
        self
    }

    /// Sets the optional metadata payload.
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Sets the sparse vector used for hybrid retrieval.
    pub fn with_sparse_vector(mut self, sparse_vector: SparseVector) -> Self {
        self.sparse_vector = Some(sparse_vector);
        self
    }

    /// Sets late-interaction token vectors for MaxSim scoring.
    pub fn with_multi_vectors(mut self, multi_vectors: crate::search::MultiVectors) -> Self {
        self.multi_vectors = Some(multi_vectors);
        self
    }

    /// Returns true if this record has both dense and sparse vectors (hybrid).
    #[must_use]
    pub const fn is_hybrid(&self) -> bool {
        self.sparse_vector.is_some()
    }

    /// Returns true when late-interaction token vectors are present.
    #[must_use]
    pub const fn is_multi_vector(&self) -> bool {
        self.multi_vectors.is_some()
    }

    /// Refreshes the record's update timestamp.
    pub fn touch(&mut self) {
        self.updated_at = OffsetDateTime::now_utc();
    }

    /// Returns the record namespace or [`DEFAULT_NAMESPACE`].
    #[must_use]
    pub fn namespace(&self) -> &str {
        self.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE)
    }

    /// Formats a timestamp as RFC 3339.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] when the timestamp cannot be represented.
    pub fn format_timestamp(timestamp: &OffsetDateTime) -> Result<String, DripstoneError> {
        timestamp
            .format(&Rfc3339)
            .map_err(|err| DripstoneError::storage_with_source("failed to format timestamp", err))
    }

    /// Parses an RFC 3339 timestamp.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] when `value` is not a valid timestamp.
    pub fn parse_timestamp(value: &str) -> Result<OffsetDateTime, DripstoneError> {
        OffsetDateTime::parse(value, &Rfc3339)
            .map_err(|err| DripstoneError::storage_with_source("failed to parse timestamp", err))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimilarityMetric {
    Cosine,
    Dot,
    Euclidean,
    Manhattan, // L1 distance - sum of absolute differences
    Chebyshev, // L∞ distance - maximum absolute difference
}

#[derive(Debug)]
#[must_use]
pub struct SimilarityResult {
    pub record: Arc<EmbeddingRecord>,
    pub score: f32,
}

impl SimilarityResult {
    /// Creates a scored embedding result.
    pub const fn new(record: Arc<EmbeddingRecord>, score: f32) -> Self {
        Self { record, score }
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct SearchOptions {
    pub metric: SimilarityMetric,
    pub limit: usize,
    pub namespace: Option<String>,
    pub min_score: Option<f32>,
    pub max_distance: Option<f32>,
    /// Advanced metadata filter using DSL
    pub filter: Option<Filter>,
}

impl SearchOptions {
    /// Creates search options for a metric and result limit.
    pub const fn new(metric: SimilarityMetric, limit: usize) -> Self {
        Self {
            metric,
            limit,
            namespace: None,
            min_score: None,
            max_distance: None,
            filter: None,
        }
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub const fn with_min_score(mut self, min_score: f32) -> Self {
        self.min_score = Some(min_score);
        self
    }

    pub const fn with_max_distance(mut self, max_distance: f32) -> Self {
        self.max_distance = Some(max_distance);
        self
    }

    /// Set the advanced metadata filter
    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Returns the optional namespace as a borrowed string.
    #[must_use]
    pub fn namespace_ref(&self) -> Option<&str> {
        self.namespace.as_deref()
    }

    /// Returns whether `score` satisfies the configured thresholds.
    #[must_use]
    pub fn passes_threshold(&self, metric: SimilarityMetric, score: f32) -> bool {
        if metric.prefers_lower_scores() {
            if let Some(max_distance) = self.max_distance
                && score > max_distance
            {
                return false;
            }
            if let Some(min_score) = self.min_score
                && score < min_score
            {
                return false;
            }
        } else {
            if let Some(min_score) = self.min_score
                && score < min_score
            {
                return false;
            }
            if let Some(max_distance) = self.max_distance
                && score > max_distance
            {
                return false;
            }
        }
        true
    }
}

impl SimilarityMetric {
    /// Compares a query vector with a candidate.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] when the vector lengths differ or scoring fails.
    pub fn compare(self, query: &[f32], candidate: &[f32]) -> Result<f32, DripstoneError> {
        if query.len() != candidate.len() {
            return Err(DripstoneError::InvalidVectorLength {
                expected: query.len(),
                actual: candidate.len(),
            });
        }

        match self {
            Self::Cosine => simd::cosine_similarity(query, candidate),
            Self::Dot => simd::dot_product(query, candidate),
            Self::Euclidean => simd::euclidean_distance(query, candidate),
            Self::Manhattan => simd::manhattan_distance(query, candidate),
            Self::Chebyshev => simd::chebyshev_distance(query, candidate),
        }
    }

    /// Prepares a scorer for repeated comparisons against `query`.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] when the query is invalid.
    pub fn prepare(self, query: &[f32]) -> Result<simd::PreparedQueryScorer<'_>, DripstoneError> {
        simd::PreparedQueryScorer::new(query, self.as_dense_metric())
    }

    #[must_use]
    pub const fn prefers_lower_scores(self) -> bool {
        matches!(self, Self::Euclidean | Self::Manhattan | Self::Chebyshev)
    }

    const fn as_dense_metric(self) -> simd::DenseMetric {
        match self {
            Self::Cosine => simd::DenseMetric::Cosine,
            Self::Dot => simd::DenseMetric::Dot,
            Self::Euclidean => simd::DenseMetric::Euclidean,
            Self::Manhattan => simd::DenseMetric::Manhattan,
            Self::Chebyshev => simd::DenseMetric::Chebyshev,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SearchOptions, SimilarityMetric};
    use crate::search::Filter;
    use serde_json::json;

    #[test]
    fn search_options_store_filter_dsl() {
        let options = SearchOptions::new(SimilarityMetric::Cosine, 5).with_filter(Filter::Equal {
            field: "category".to_string(),
            value: json!("docs"),
        });

        assert!(matches!(
            options.filter,
            Some(Filter::Equal { ref field, ref value })
            if field == "category" && value == &json!("docs")
        ));
    }
}

#[cfg(test)]
mod multi_vector_serde_smoke {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn multi_vectors_json_round_trip() {
        let mv: crate::search::MultiVectors = Arc::from(
            vec![
                Arc::from(vec![1.0_f32, 0.0].into_boxed_slice()),
                Arc::from(vec![0.0_f32, 1.0].into_boxed_slice()),
            ]
            .into_boxed_slice(),
        );
        let record = EmbeddingRecord::new("id", vec![0.5_f32, 0.5]).with_multi_vectors(mv);
        let json = serde_json::to_string(&record).expect("ser");
        let back: EmbeddingRecord = serde_json::from_str(&json).expect("de");
        assert!(back.multi_vectors.is_some());
        assert_eq!(back.multi_vectors.as_ref().unwrap().len(), 2);
        assert_eq!(
            back.multi_vectors.as_ref().unwrap()[0].as_ref(),
            &[1.0, 0.0]
        );
    }
}
