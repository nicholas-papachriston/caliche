//! Core types for `Dripstone`.
//!
//! This module defines the fundamental types used throughout the library:
//! - `Document` - Input type for document ingestion
//! - `Chunk` - Internal representation of chunked and embedded content
//! - `SearchResult` - Output type from search operations
//! - `Metric` - Similarity metrics for vector comparison

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::errors::DripstoneError;
use crate::simd;

/// Similarity metrics for vector comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Metric {
    /// Cosine similarity (default) - angle between vectors, [-1, 1] normalized to [0, 1]
    #[default]
    Cosine,
    /// Euclidean distance (L2) - straight-line distance between vectors
    Euclidean,
    /// Dot product - raw inner product, useful for normalized vectors
    DotProduct,
    /// Manhattan distance (L1) - sum of absolute differences
    Manhattan,
}

impl Metric {
    /// Compute the similarity/distance between two vectors.
    ///
    /// For `Cosine` and `DotProduct`, higher is more similar.
    /// For `Euclidean` and `Manhattan`, lower is more similar.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] when the vectors have incompatible lengths
    /// or cannot be scored.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> Result<f32, DripstoneError> {
        match self {
            Self::Cosine => simd::cosine_similarity(a, b),
            Self::Euclidean => simd::euclidean_distance(a, b),
            Self::DotProduct => simd::dot_product(a, b),
            Self::Manhattan => simd::manhattan_distance(a, b),
        }
    }

    /// Prepares a query-scoped scorer for repeated comparisons.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] when the query vector is invalid.
    pub fn prepare<'a>(
        &self,
        query: &'a [f32],
    ) -> Result<simd::PreparedQueryScorer<'a>, DripstoneError> {
        simd::PreparedQueryScorer::new(query, self.as_dense_metric())
    }

    /// Returns true if higher scores indicate more similarity.
    #[must_use]
    pub const fn higher_is_better(&self) -> bool {
        matches!(self, Self::Cosine | Self::DotProduct)
    }

    const fn as_dense_metric(self) -> simd::DenseMetric {
        match self {
            Self::Cosine => simd::DenseMetric::Cosine,
            Self::Euclidean => simd::DenseMetric::Euclidean,
            Self::DotProduct => simd::DenseMetric::Dot,
            Self::Manhattan => simd::DenseMetric::Manhattan,
        }
    }
}

/// A pre-computed chunk for `ingest_chunks`. Use when chunking is done externally
/// (e.g. AST-aware chunking at function/class boundaries).
#[derive(Debug, Clone)]
#[must_use]
pub struct PreChunk {
    /// Text content of the chunk
    pub text: String,
    /// Start byte offset in the original document
    pub start_offset: usize,
    /// End byte offset in the original document
    pub end_offset: usize,
}

impl PreChunk {
    /// Creates a pre-computed chunk with byte offsets into its source document.
    pub fn new(text: impl Into<String>, start_offset: usize, end_offset: usize) -> Self {
        Self {
            text: text.into(),
            start_offset,
            end_offset,
        }
    }
}

/// A document to be ingested into `Dripstone`.
///
/// Documents are the input type for the ingest pipeline. They get chunked,
/// embedded, and stored as `Chunk`s.
///
/// # Type Parameters
///
/// * `M` - Metadata type. Must implement `Serialize`. Defaults to `()` for no metadata.
///
/// # Example
///
/// ```rust
/// use dripstone::core::Document;
/// use serde::Serialize;
///
/// // Simple document with no metadata
/// let doc = Document::new("doc-1", "Hello, world!");
///
/// // Document with typed metadata
/// #[derive(Serialize)]
/// struct Meta {
///     author: String,
///     category: String,
/// }
///
/// let doc = Document::with_metadata(
///     "doc-2",
///     "Document content here",
///     Meta { author: "Alice".into(), category: "tech".into() }
/// );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Document<M = ()> {
    /// Unique identifier for the document
    pub id: String,
    /// Raw text content
    pub content: String,
    /// Optional metadata (type-safe via Serialize)
    pub metadata: M,
}

impl Document<()> {
    /// Create a new document with no metadata.
    pub fn new(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata: (),
        }
    }
}

impl<M: Serialize> Document<M> {
    /// Create a new document with metadata.
    pub fn with_metadata(id: impl Into<String>, content: impl Into<String>, metadata: M) -> Self {
        Self {
            id: id.into(),
            content: content.into(),
            metadata,
        }
    }
}

/// A chunk is a piece of a document that has been processed and embedded.
///
/// Chunks are created during ingestion when documents are split according
/// to the chunking strategy. Each chunk maintains references back to its
/// source document.
///
/// # Memory Optimization
///
/// The `vector` field uses `Arc<[f32]>` for cheap cloning. Cloning a Chunk
/// only increments reference counts rather than copying the ~6KB vector data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct Chunk {
    /// Unique identifier for this chunk (format: `{doc_id}#{chunk_idx}`)
    pub id: String,
    /// ID of the source document
    pub doc_id: String,
    /// Text content of this chunk
    pub content: String,
    /// Embedding vector wrapped in Arc for cheap cloning
    #[serde(
        serialize_with = "serialize_arc_slice",
        deserialize_with = "deserialize_arc_slice"
    )]
    pub vector: Arc<[f32]>,
    /// Character offset where this chunk starts in the original document
    pub start_offset: usize,
    /// Character offset where this chunk ends in the original document
    pub end_offset: usize,
    /// Zero-based index of this chunk within the document
    pub chunk_index: usize,
    /// Total number of chunks the document was split into
    pub total_chunks: usize,
    /// Document metadata as JSON (serialized from Document.metadata)
    pub metadata: serde_json::Value,
    /// Optional namespace for multi-tenant separation
    pub namespace: Option<String>,
    /// Optional late-interaction token vectors (ColBERT-style MaxSim).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_vectors: Option<crate::search::MultiVectors>,
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

/// Magic prefix for rkyv [`ChunkStoredV1`] SQLite blobs.
#[cfg(feature = "sqlite-backend")]
pub(crate) const CHUNK_BLOB_V1_MAGIC: &[u8; 4] = b"CKV1";

/// Durable row layout for [`Chunk`] using [rkyv](https://github.com/rkyv/rkyv).
///
/// Used for `SQLite` blobs and other binary persistence.
/// Validated loads use `rkyv::from_bytes` / `rkyv::access` (requires `bytecheck` in `rkyv`).
#[derive(Archive, RkyvSerialize, RkyvDeserialize, Debug, Clone)]
#[rkyv(derive(Debug))]
pub(crate) struct ChunkStoredV1 {
    pub id: String,
    pub doc_id: String,
    pub content: String,
    pub vector: Vec<f32>,
    pub start_offset: usize,
    pub end_offset: usize,
    pub chunk_index: usize,
    pub total_chunks: usize,
    pub metadata_json: String,
    pub namespace: Option<String>,
    pub multi_vectors: Option<Vec<Vec<f32>>>,
}

impl TryFrom<&Chunk> for ChunkStoredV1 {
    type Error = DripstoneError;

    fn try_from(c: &Chunk) -> Result<Self, Self::Error> {
        Ok(Self {
            id: c.id.clone(),
            doc_id: c.doc_id.clone(),
            content: c.content.clone(),
            vector: c.vector.iter().copied().collect(),
            start_offset: c.start_offset,
            end_offset: c.end_offset,
            chunk_index: c.chunk_index,
            total_chunks: c.total_chunks,
            metadata_json: serde_json::to_string(&c.metadata).map_err(|e| {
                DripstoneError::storage(format!("Failed to serialize chunk metadata: {e}"))
            })?,
            namespace: c.namespace.clone(),
            multi_vectors: c
                .multi_vectors
                .as_ref()
                .map(|mv| mv.iter().map(|t| t.to_vec()).collect()),
        })
    }
}

impl TryFrom<ChunkStoredV1> for Chunk {
    type Error = DripstoneError;

    fn try_from(s: ChunkStoredV1) -> Result<Self, Self::Error> {
        let metadata = serde_json::from_str(&s.metadata_json)
            .map_err(|e| DripstoneError::storage(format!("Failed to parse chunk metadata: {e}")))?;
        Ok(Self {
            id: s.id,
            doc_id: s.doc_id,
            content: s.content,
            vector: Arc::from(s.vector),
            start_offset: s.start_offset,
            end_offset: s.end_offset,
            chunk_index: s.chunk_index,
            total_chunks: s.total_chunks,
            metadata,
            namespace: s.namespace,
            multi_vectors: s.multi_vectors.map(crate::search::multi_vectors_from_owned),
        })
    }
}

impl Chunk {
    /// Create a new chunk.
    ///
    /// The vector is automatically converted to `Arc<[f32]>` for cheap cloning.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        doc_id: impl Into<String>,
        chunk_index: usize,
        content: impl Into<String>,
        vector: impl Into<Arc<[f32]>>,
        start_offset: usize,
        end_offset: usize,
        total_chunks: usize,
        metadata: serde_json::Value,
        namespace: Option<String>,
    ) -> Self {
        let doc_id = doc_id.into();
        let id = format!("{doc_id}#{chunk_index}");
        Self {
            id,
            doc_id,
            content: content.into(),
            vector: vector.into(),
            start_offset,
            end_offset,
            chunk_index,
            total_chunks,
            metadata,
            namespace,
            multi_vectors: None,
        }
    }

    /// Attach late-interaction token vectors (pooled `vector` remains the ANN key).
    #[must_use]
    pub fn with_multi_vectors(mut self, multi_vectors: crate::search::MultiVectors) -> Self {
        self.multi_vectors = Some(multi_vectors);
        self
    }

    /// Returns true if this is the first chunk of the document.
    #[must_use]
    pub const fn is_first(&self) -> bool {
        self.chunk_index == 0
    }

    /// Returns true if this is the last chunk of the document.
    #[must_use]
    pub const fn is_last(&self) -> bool {
        self.chunk_index + 1 == self.total_chunks
    }

    /// Returns true if this is the only chunk (document wasn't split).
    #[must_use]
    pub const fn is_single(&self) -> bool {
        self.total_chunks == 1
    }

    /// Returns the dimension of the embedding vector.
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.vector.len()
    }

    #[cfg(feature = "sqlite-backend")]
    pub(crate) fn encode_for_sqlite_blob(&self) -> Result<Vec<u8>, DripstoneError> {
        let stored = ChunkStoredV1::try_from(self)?;
        let body = rkyv::to_bytes::<rkyv::rancor::Error>(&stored)
            .map(rkyv::util::AlignedVec::into_vec)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize chunk: {e}")))?;
        let mut out = Vec::with_capacity(CHUNK_BLOB_V1_MAGIC.len() + body.len());
        out.extend_from_slice(CHUNK_BLOB_V1_MAGIC);
        out.extend_from_slice(&body);
        Ok(out)
    }

    #[cfg(feature = "sqlite-backend")]
    pub(crate) fn decode_from_sqlite_blob(bytes: &[u8]) -> Result<Self, DripstoneError> {
        if !bytes.starts_with(CHUNK_BLOB_V1_MAGIC) {
            return Err(DripstoneError::storage(
                "SQLite chunk blob is not a supported Dripstone record (expected magic CKV1)"
                    .to_string(),
            ));
        }
        let body = &bytes[CHUNK_BLOB_V1_MAGIC.len()..];
        // Magic prefix may leave the payload unaligned for rkyv; copy into AlignedVec.
        let mut aligned = rkyv::util::AlignedVec::<16>::new();
        aligned.extend_from_slice(body);
        let stored: ChunkStoredV1 = rkyv::from_bytes::<_, rkyv::rancor::Error>(aligned.as_ref())
            .map_err(|e| DripstoneError::storage(format!("Failed to deserialize chunk: {e}")))?;
        Self::try_from(stored)
    }
}

/// A search result containing a chunk and its relevance score.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[must_use]
pub struct SearchResult {
    /// The matched chunk
    pub chunk: Chunk,
    /// Similarity/relevance score (interpretation depends on metric)
    pub score: f32,
    /// Rank in the result set (0-indexed)
    pub rank: usize,
    /// Whether `score` was produced by the backend's scoring path or is a
    /// synthesized placeholder (e.g. when the backend API does not return one).
    /// Consumers that rerank should treat non-authoritative scores as hints.
    #[serde(default = "score_authoritative_default")]
    pub score_is_authoritative: bool,
}

#[inline]
const fn score_authoritative_default() -> bool {
    true
}

impl SearchResult {
    /// Create a new search result (score assumed to be authoritative).
    pub const fn new(chunk: Chunk, score: f32, rank: usize) -> Self {
        Self {
            chunk,
            score,
            rank,
            score_is_authoritative: true,
        }
    }

    /// Create a search result whose score is a synthesized placeholder.
    pub const fn new_unscored(chunk: Chunk, placeholder_score: f32, rank: usize) -> Self {
        Self {
            chunk,
            score: placeholder_score,
            rank,
            score_is_authoritative: false,
        }
    }
}

/// High-level retrieval mode for string-based search entrypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SearchMode {
    #[default]
    Dense,
    FullText,
    HybridDenseFullText,
}

/// Query-time options for full-text retrieval.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[must_use]
pub struct FullTextOptions {
    /// BM25 term-frequency saturation parameter.
    pub bm25_k1: f32,
    /// BM25 length normalization parameter.
    pub bm25_b: f32,
    /// Posting block size used for MAXSCORE block metadata.
    pub block_size: usize,
    /// Whether to expand the query with extracted keywords before tokenization.
    pub query_expansion: bool,
    /// Optional cap on the number of terms retained from the lexical query.
    pub max_query_terms: Option<usize>,
}

impl Default for FullTextOptions {
    fn default() -> Self {
        Self {
            bm25_k1: 1.5,
            bm25_b: 0.75,
            block_size: 32,
            query_expansion: false,
            max_query_terms: None,
        }
    }
}

impl FullTextOptions {
    /// Enables or disables lexical query expansion.
    pub const fn with_query_expansion(mut self, enabled: bool) -> Self {
        self.query_expansion = enabled;
        self
    }

    /// Sets the maximum number of terms retained from the lexical query.
    pub const fn with_max_query_terms(mut self, max_query_terms: usize) -> Self {
        self.max_query_terms = Some(max_query_terms);
        self
    }
}

/// A first-class lexical query for BM25/MAXSCORE search.
#[derive(Debug, Clone)]
#[must_use]
pub struct FullTextQuery {
    /// Raw user query text.
    pub text: String,
    /// Maximum number of results to return.
    pub limit: usize,
    /// Optional namespace filter.
    pub namespace: Option<String>,
    /// Optional metadata filter.
    pub filter: Option<crate::search::Filter>,
    /// Whether to include content in results.
    pub include_content: bool,
    /// Whether to include vectors in results.
    pub include_vectors: bool,
    /// Query-time lexical options.
    pub options: FullTextOptions,
}

impl Default for FullTextQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            limit: 10,
            namespace: None,
            filter: None,
            include_content: true,
            include_vectors: false,
            options: FullTextOptions::default(),
        }
    }
}

impl FullTextQuery {
    /// Creates a lexical query from raw text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    /// Sets the maximum number of results.
    pub const fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Limits the query to a namespace.
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Applies a metadata filter.
    pub fn filter(mut self, filter: crate::search::Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Replaces the lexical query options.
    pub const fn options(mut self, options: FullTextOptions) -> Self {
        self.options = options;
        self
    }
}

/// A search query with all parameters.
///
/// Uses `Arc<[f32]>` for the query vector to allow cheap cloning when
/// the same query is used for multiple search operations (e.g., hybrid search).
#[derive(Debug, Clone)]
#[must_use]
pub struct SearchQuery {
    /// The query vector wrapped in Arc for cheap cloning
    pub vector: Arc<[f32]>,
    /// Maximum number of results to return
    pub limit: usize,
    /// Similarity metric to use
    pub metric: Metric,
    /// Optional namespace filter
    pub namespace: Option<String>,
    /// Optional metadata filter
    pub filter: Option<crate::search::Filter>,
    /// Optional hybrid search weight (0.0 = pure sparse, 1.0 = pure dense)
    pub hybrid_weight: Option<f32>,
    /// Optional sparse vector for hybrid search
    pub sparse_vector: Option<crate::sparse::SparseVector>,
    /// Optional late-interaction query token vectors for MaxSim.
    pub query_multi_vectors: Option<crate::search::MultiVectors>,
    /// MaxSim scoring options (used when `query_multi_vectors` is set).
    pub maxsim: Option<crate::search::MaxSimOptions>,
    /// Whether to include content in results
    pub include_content: bool,
    /// Whether to include vectors in results
    pub include_vectors: bool,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            vector: Arc::from([]),
            limit: 10,
            metric: Metric::Cosine,
            namespace: None,
            filter: None,
            hybrid_weight: None,
            sparse_vector: None,
            query_multi_vectors: None,
            maxsim: None,
            include_content: true,
            include_vectors: false,
        }
    }
}

impl SearchQuery {
    /// Create a new search query with just a vector.
    ///
    /// The vector is automatically converted to `Arc<[f32]>` for cheap cloning.
    pub fn new(vector: impl Into<Arc<[f32]>>) -> Self {
        Self {
            vector: vector.into(),
            ..Default::default()
        }
    }

    /// Get a reference to the query vector as a slice.
    #[must_use]
    pub fn query(&self) -> &[f32] {
        &self.vector
    }

    /// Set the limit.
    pub const fn limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Set the metric.
    pub const fn metric(mut self, metric: Metric) -> Self {
        self.metric = metric;
        self
    }

    /// Set the namespace.
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    /// Set the filter.
    pub fn filter(mut self, filter: crate::search::Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Attach late-interaction query token vectors.
    pub fn with_multi_vectors(mut self, tokens: crate::search::MultiVectors) -> Self {
        self.query_multi_vectors = Some(tokens);
        self
    }

    /// Configure MaxSim scoring (typically used with [`Self::with_multi_vectors`]).
    pub const fn with_maxsim(mut self, opts: crate::search::MaxSimOptions) -> Self {
        self.maxsim = Some(opts);
        self
    }
}

/// Backend capabilities for feature detection.
///
/// Backends report their capabilities so `Dripstone` can adapt its behavior.
/// For example, if a backend doesn't support native hybrid search,
/// `Dripstone` can implement it client-side.
#[derive(Debug, Clone, Default)]
#[must_use]
#[expect(
    clippy::struct_excessive_bools,
    reason = "public capability flags are independent and changing their fields would break API compatibility"
)]
pub struct Capabilities {
    /// Whether the backend supports native hybrid (dense + sparse) search
    pub native_hybrid: bool,
    /// Whether the backend supports native BM25/MAXSCORE full-text search.
    pub native_full_text: bool,
    /// Whether the backend supports a native dense + full-text hybrid search path.
    pub native_hybrid_full_text: bool,
    /// Whether the backend supports native metadata filtering
    pub native_filter: bool,
    /// Whether search may apply a native metadata prefilter and still run an exact
    /// [`crate::search::Filter::matches`] pass on hydrated results.
    ///
    /// When `true`, callers should not assume `native_filter` alone enforces arbitrary
    /// filter expressions (for example range or negation). Indexed ANN paths and
    /// `ColdForge` compile supported equality terms into index constraints, then verify
    /// each candidate with `Filter::matches`.
    pub requires_exact_filter_pass: bool,
    /// Whether the backend supports namespaces/collections
    pub supports_namespaces: bool,
    /// Whether the backend can score late-interaction multi-vectors (MaxSim).
    pub native_maxsim: bool,
    /// Maximum vector dimension (None = unlimited)
    pub max_dimension: Option<usize>,
    /// Maximum batch size for upserts (None = unlimited)
    pub max_batch: Option<usize>,
    /// Whether the backend supports streaming results
    pub supports_streaming: bool,
}

impl Capabilities {
    /// Create capabilities for a fully-featured backend.
    pub const fn full() -> Self {
        Self {
            native_hybrid: true,
            native_full_text: true,
            native_hybrid_full_text: true,
            native_filter: true,
            requires_exact_filter_pass: false,
            supports_namespaces: true,
            native_maxsim: true,
            max_dimension: None,
            max_batch: None,
            supports_streaming: true,
        }
    }

    /// Create capabilities for a basic in-memory backend.
    pub const fn memory() -> Self {
        Self {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: false,
            supports_namespaces: true,
            native_maxsim: true,
            max_dimension: None,
            max_batch: None,
            supports_streaming: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_cosine() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(
            (Metric::Cosine
                .compute(&a, &b)
                .expect("cosine computation should succeed")
                - 1.0)
                .abs()
                < 1e-6
        );

        let c = vec![0.0, 1.0, 0.0];
        assert!(
            Metric::Cosine
                .compute(&a, &c)
                .expect("cosine computation should succeed")
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn test_metric_euclidean() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!(
            (Metric::Euclidean
                .compute(&a, &b)
                .expect("euclidean computation should succeed")
                - 1.0)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn test_document_creation() {
        let doc = Document::new("test-1", "Hello world");
        assert_eq!(doc.id, "test-1");
        assert_eq!(doc.content, "Hello world");
    }

    #[test]
    fn test_chunk_creation() {
        let chunk = Chunk::new(
            "doc-1",
            0,
            "chunk content",
            vec![0.1, 0.2, 0.3],
            0,
            13,
            1,
            serde_json::json!({}),
            None,
        );
        assert_eq!(chunk.id, "doc-1#0");
        assert!(chunk.is_first());
        assert!(chunk.is_last());
        assert!(chunk.is_single());
        assert_eq!(chunk.dimension(), 3);
    }

    #[cfg(feature = "sqlite-backend")]
    #[test]
    fn sqlite_blob_v1_preserves_multi_vectors() {
        let mv = crate::search::multi_vectors_from_owned(vec![
            vec![1.0_f32, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ]);
        let chunk = Chunk::new(
            "doc",
            0,
            "content",
            vec![0.5_f32, 0.5, 0.0],
            0,
            7,
            1,
            serde_json::json!({}),
            Some("ns".into()),
        )
        .with_multi_vectors(mv);

        let encoded = chunk.encode_for_sqlite_blob().expect("encode v1");
        assert!(encoded.starts_with(CHUNK_BLOB_V1_MAGIC));
        let decoded = Chunk::decode_from_sqlite_blob(&encoded).expect("decode v1");
        assert_eq!(
            decoded.multi_vectors.as_ref().unwrap()[0].as_ref(),
            &[1.0, 0.0, 0.0]
        );
        assert_eq!(decoded.id, chunk.id);
    }

    #[cfg(feature = "sqlite-backend")]
    #[test]
    fn sqlite_blob_rejects_unknown_magic() {
        let err = Chunk::decode_from_sqlite_blob(b"XXXX").expect_err("unknown magic");
        let message = err.to_string();
        assert!(message.contains("CKV1"), "unexpected error: {message}");
    }
}
