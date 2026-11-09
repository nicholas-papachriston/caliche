//! S3 backend for embedding storage
//!
//! This module provides two S3-based backends:
//!
//! ## Standard S3 Backend (`S3EmbeddingBackend`)
//! Cold storage for embeddings in standard S3 buckets with:
//! - Efficient binary serialization with [rkyv](https://github.com/rkyv/rkyv)
//! - Namespace-based S3 key organization  
//! - Batch operations with compression
//! - Metadata indexing for fast listing
//! - Direct querying from S3 (slower but cost-effective)
//!
//! ## S3 Vector Backend (`S3VectorBackend`)
//! Native AWS S3 Vector Buckets support (preview feature) with:
//! - Server-side vector indexing and similarity search
//! - Native vector APIs (`PutVectors`, `QueryVectors`, etc.)
//! - Support for 1-4096 dimensional vectors
//! - Cosine and Euclidean distance metrics
//! - Built-in metadata filtering
//! - Optimized for performance and cost
//!
//! See: <https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html>

pub mod layout;
pub mod vector_backend;
pub mod vectors;
pub mod wal;

use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::RwLock;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::embedding::EmbeddingRecord;
use crate::errors::DripstoneError;
use crate::storage::{ObjectStore, ObjectStoreConfig, create_object_store};

// Re-export S3 Vector Backend types
pub use vector_backend::{S3VectorBackend, S3VectorBackendConfig};

/// Magic prefix for rkyv [`S3EmbeddingRecord`] blobs.
const S3_RECORD_V1_MAGIC: &[u8; 4] = b"S3V1";

/// Serializable format for storing embeddings in S3.
#[derive(Debug, Clone, Archive, RkyvSerialize, RkyvDeserialize)]
#[rkyv(derive(Debug))]
struct S3EmbeddingRecord {
    id: String,
    namespace: Option<String>,
    vector: Vec<f32>,
    sparse_indices: Option<HashMap<u32, f32>>,
    metadata_json: Option<String>,
    updated_at: i128,
    multi_vectors: Option<Vec<Vec<f32>>>,
}

impl From<EmbeddingRecord> for S3EmbeddingRecord {
    fn from(record: EmbeddingRecord) -> Self {
        Self {
            id: record.id.to_string(),
            namespace: record.namespace.map(|ns| ns.to_string()),
            vector: record.vector.to_vec(),
            sparse_indices: record.sparse_vector.map(|sv| sv.indices),
            metadata_json: record
                .metadata
                .map(|m| serde_json::to_string(&m).unwrap_or_default()),
            updated_at: record.updated_at.unix_timestamp_nanos(),
            multi_vectors: record
                .multi_vectors
                .map(|mv| mv.iter().map(|token| token.to_vec()).collect()),
        }
    }
}

fn embedding_from_s3_parts(
    id: String,
    namespace: Option<String>,
    vector: Vec<f32>,
    sparse_indices: Option<HashMap<u32, f32>>,
    metadata_json: Option<String>,
    updated_at: i128,
    multi_vectors: Option<Vec<Vec<f32>>>,
) -> EmbeddingRecord {
    use time::OffsetDateTime;

    let sparse_vector = sparse_indices.map(|indices| {
        let mut sv = crate::sparse::SparseVector::new();
        sv.indices = indices;
        sv
    });

    let metadata = metadata_json.and_then(|json_str| serde_json::from_str(&json_str).ok());
    let multi_vectors = multi_vectors.map(crate::search::multi_vectors_from_owned);
    let ts = OffsetDateTime::from_unix_timestamp_nanos(updated_at)
        .unwrap_or_else(|_| OffsetDateTime::now_utc());

    EmbeddingRecord {
        id: std::sync::Arc::from(id),
        namespace: namespace.map(std::sync::Arc::from),
        vector: vector.into(),
        sparse_vector,
        multi_vectors,
        metadata,
        updated_at: ts,
        created_at: ts,
    }
}

impl From<S3EmbeddingRecord> for EmbeddingRecord {
    fn from(s3_record: S3EmbeddingRecord) -> Self {
        embedding_from_s3_parts(
            s3_record.id,
            s3_record.namespace,
            s3_record.vector,
            s3_record.sparse_indices,
            s3_record.metadata_json,
            s3_record.updated_at,
            s3_record.multi_vectors,
        )
    }
}

/// Namespace index stored in S3
#[derive(Debug, Clone, Serialize, Deserialize)]
struct NamespaceIndex {
    namespace: String,
    record_ids: Vec<String>,
    total_vectors: usize,
    dimension: Option<usize>,
    last_updated: i128,
}

/// Configuration for S3 backend
#[derive(Debug, Clone)]
pub struct S3BackendConfig {
    /// S3 bucket name
    pub bucket: String,
    /// Prefix for all keys (e.g., "embeddings/")
    pub key_prefix: String,
    /// Enable compression for stored vectors
    pub enable_compression: bool,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// Cache namespace indices in memory
    pub cache_indices: bool,
    /// AWS access key ID (optional, uses default credentials if not set)
    pub access_key_id: Option<String>,
    /// AWS secret access key (optional, uses default credentials if not set)
    pub secret_access_key: Option<String>,
    /// AWS session token for temporary credentials (optional)
    pub session_token: Option<String>,
    /// AWS region (optional, uses default if not set)
    pub region: Option<String>,
}

impl Default for S3BackendConfig {
    fn default() -> Self {
        Self {
            bucket: String::from("embeddings"),
            key_prefix: String::from("vectors/"),
            enable_compression: true,
            batch_size: 100,
            cache_indices: true,
            access_key_id: None,
            secret_access_key: None,
            session_token: None,
            region: None,
        }
    }
}

/// S3-based embedding storage backend
pub struct S3EmbeddingBackend {
    store: Arc<dyn ObjectStore>,
    config: S3BackendConfig,
    index_cache: Arc<RwLock<HashMap<String, NamespaceIndex>>>,
}

impl S3EmbeddingBackend {
    /// Create a new S3 backend with optional AWS credentials
    ///
    /// Credentials can be provided via:
    /// 1. Config fields (takes precedence)
    /// 2. Default AWS credentials chain (if not provided)
    ///
    /// Note: For S3-compatible services (like `RustFS`, `MinIO`), use `with_object_store()` instead.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(config: S3BackendConfig) -> Result<Self, DripstoneError> {
        // Use credentials from config if provided, otherwise use default AWS credentials chain
        let store_config = if let (Some(access_key), Some(secret_key)) = (
            config.access_key_id.as_ref(),
            config.secret_access_key.as_ref(),
        ) {
            // Create AWS S3 config with explicit credentials
            ObjectStoreConfig::aws_s3(
                config.region.clone(),
                Some(access_key.clone()),
                Some(secret_key.clone()),
                config.session_token.clone(),
            )
        } else {
            // Use default AWS credentials chain (no env vars needed)
            ObjectStoreConfig::aws_s3(config.region.clone(), None, None, None)
        };
        let store = create_object_store(store_config).await?;
        Ok(Self {
            store,
            config,
            index_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Create from `ObjectStore`
    pub fn with_object_store(store: Arc<dyn ObjectStore>, config: S3BackendConfig) -> Self {
        Self {
            store,
            config,
            index_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate S3 key for a record
    fn record_key(&self, id: &str, namespace: Option<&str>) -> String {
        let ns = namespace.unwrap_or("default");
        format!("{}{}/{}.bin", self.config.key_prefix, ns, id)
    }

    /// Generate S3 key for namespace index
    fn index_key(&self, namespace: &str) -> String {
        format!("{}{}/_index.json", self.config.key_prefix, namespace)
    }

    /// Serialize a record to bytes (V1 magic + rkyv, includes multi-vectors).
    fn serialize_record(record: &EmbeddingRecord) -> Result<Vec<u8>, DripstoneError> {
        let s3_record = S3EmbeddingRecord::from(record.clone());
        let body = rkyv::to_bytes::<rkyv::rancor::Error>(&s3_record)
            .map(rkyv::util::AlignedVec::into_vec)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize record: {e}")))?;
        let mut out = Vec::with_capacity(S3_RECORD_V1_MAGIC.len() + body.len());
        out.extend_from_slice(S3_RECORD_V1_MAGIC);
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Deserialize a record from bytes (V1 magic + rkyv).
    fn deserialize_record(bytes: &[u8]) -> Result<EmbeddingRecord, DripstoneError> {
        if !bytes.starts_with(S3_RECORD_V1_MAGIC) {
            return Err(DripstoneError::storage(
                "S3 record blob is not a supported Dripstone record (expected magic S3V1)"
                    .to_string(),
            ));
        }
        let body = &bytes[S3_RECORD_V1_MAGIC.len()..];
        // Magic prefix may leave the payload unaligned for rkyv; copy into AlignedVec.
        let mut aligned = rkyv::util::AlignedVec::<16>::new();
        aligned.extend_from_slice(body);
        let s3_record: S3EmbeddingRecord =
            rkyv::from_bytes::<_, rkyv::rancor::Error>(aligned.as_ref()).map_err(|e| {
                DripstoneError::storage(format!("Failed to deserialize S3 record: {e}"))
            })?;
        Ok(EmbeddingRecord::from(s3_record))
    }

    /// Store record in object storage
    async fn put_record(&self, record: &EmbeddingRecord) -> Result<(), DripstoneError> {
        let key = self.record_key(&record.id, record.namespace.as_deref());
        let bytes = Self::serialize_record(record)?;

        self.store
            .put_object(
                &self.config.bucket,
                &key,
                Bytes::from(bytes),
                Some("application/octet-stream"),
            )
            .await?;

        Ok(())
    }

    /// Get record from object storage
    async fn get_record(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<EmbeddingRecord>, DripstoneError> {
        let key = self.record_key(id, namespace);

        match self.store.get_object(&self.config.bucket, &key).await {
            Ok(bytes) => {
                let record = Self::deserialize_record(&bytes)?;
                Ok(Some(record))
            }
            Err(e) if e.is_not_found() => Ok(None),
            Err(e) => Err(DripstoneError::storage(format!(
                "Failed to get object from storage: {e}"
            ))),
        }
    }

    /// Delete record from object storage
    async fn delete_record(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let key = self.record_key(id, namespace);

        self.store
            .delete_object(&self.config.bucket, &key)
            .await
            .map_err(|e| {
                DripstoneError::storage(format!("Failed to delete object from storage: {e}"))
            })?;

        Ok(())
    }

    /// Load or create namespace index
    async fn get_namespace_index(&self, namespace: &str) -> Result<NamespaceIndex, DripstoneError> {
        // Check cache first
        if self.config.cache_indices {
            let cache = self.index_cache.read();
            if let Some(index) = cache.get(namespace) {
                return Ok(index.clone());
            }
        }

        // Load from object storage
        let key = self.index_key(namespace);

        let index = match self.store.get_object(&self.config.bucket, &key).await {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .map_err(|e| DripstoneError::storage(format!("Failed to parse index: {e}")))?,
            // Only treat "not found" as an empty index; propagate other
            // transport/auth errors so callers don't see a phantom empty
            // namespace when S3 is misconfigured or unreachable.
            Err(e) if e.is_not_found() => NamespaceIndex {
                namespace: namespace.to_string(),
                record_ids: Vec::new(),
                total_vectors: 0,
                dimension: None,
                last_updated: time::OffsetDateTime::now_utc().unix_timestamp_nanos(),
            },
            Err(e) => return Err(e),
        };

        // Update cache
        if self.config.cache_indices {
            let mut cache = self.index_cache.write();
            cache.insert(namespace.to_string(), index.clone());
        }

        Ok(index)
    }

    /// Save namespace index
    async fn save_namespace_index(&self, index: &NamespaceIndex) -> Result<(), DripstoneError> {
        let key = self.index_key(&index.namespace);
        let json = serde_json::to_vec(index)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize index: {e}")))?;

        self.store
            .put_object(
                &self.config.bucket,
                &key,
                Bytes::from(json),
                Some("application/json"),
            )
            .await
            .map_err(|e| DripstoneError::storage(format!("Failed to save index: {e}")))?;

        // Update cache
        if self.config.cache_indices {
            let mut cache = self.index_cache.write();
            cache.insert(index.namespace.clone(), index.clone());
        }

        Ok(())
    }

    /// Update index after upsert
    async fn update_index_after_upsert(
        &self,
        record: &EmbeddingRecord,
    ) -> Result<(), DripstoneError> {
        let namespace = record.namespace.as_deref().unwrap_or("default");
        let mut index = self.get_namespace_index(namespace).await?;

        let id_str = record.id.to_string();
        if !index.record_ids.contains(&id_str) {
            index.record_ids.push(id_str);
            index.total_vectors = index.record_ids.len();
        }

        if index.dimension.is_none() {
            index.dimension = Some(record.vector.len());
        }

        index.last_updated = time::OffsetDateTime::now_utc().unix_timestamp_nanos();

        self.save_namespace_index(&index).await
    }

    /// Update index after delete
    async fn update_index_after_delete(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<(), DripstoneError> {
        let ns = namespace.unwrap_or("default");
        let mut index = self.get_namespace_index(ns).await?;

        index.record_ids.retain(|rid| rid != id);
        index.total_vectors = index.record_ids.len();
        index.last_updated = time::OffsetDateTime::now_utc().unix_timestamp_nanos();

        self.save_namespace_index(&index).await
    }

    /// List all records in a namespace (loads all from S3)
    async fn list_all_records(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        let ns = namespace.unwrap_or("default");
        let index = self.get_namespace_index(ns).await?;

        let mut records = Vec::new();

        for id in &index.record_ids {
            if let Some(record) = self.get_record(id, Some(ns)).await? {
                records.push(record);
            }
        }

        Ok(records)
    }
}

// =============================================================================
// Backend trait implementation
// =============================================================================

use crate::backend::chunk_envelope::{self, doc_id_from_envelope};
use crate::core::{Capabilities, Chunk, Metric, SearchQuery, SearchResult};
use crate::embedding::DEFAULT_NAMESPACE;
use crate::traits::Backend;
use std::fmt;

impl fmt::Debug for S3EmbeddingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3Backend")
            .field("bucket", &self.config.bucket)
            .field("key_prefix", &self.config.key_prefix)
            .finish_non_exhaustive()
    }
}

/// S3 backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct S3Backend {
    inner: S3EmbeddingBackend,
}

impl S3Backend {
    /// Create a new S3 backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_config(config: S3BackendConfig) -> Result<Self, DripstoneError> {
        let inner = S3EmbeddingBackend::new(config).await?;
        Ok(Self { inner })
    }

    /// Create from an already-constructed [`ObjectStore`] (tests, MinIO, shared stores).
    #[must_use]
    pub fn with_object_store(store: Arc<dyn ObjectStore>, config: S3BackendConfig) -> Self {
        Self {
            inner: S3EmbeddingBackend::with_object_store(store, config),
        }
    }
}

#[async_trait]
impl Backend for S3Backend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: false,
            requires_exact_filter_pass: false,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: None,
            max_batch: Some(self.inner.config.batch_size),
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        for chunk in chunks {
            let record = chunk_envelope::embedding_record_from_chunk(chunk);
            self.inner.put_record(&record).await?;
            self.inner.update_index_after_upsert(&record).await?;
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let record = self.inner.get_record(id, namespace).await?;

        match record {
            Some(r) => Ok(Some(chunk_envelope::decode_chunk(&r, None)?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        self.inner.delete_record(id, namespace).await?;
        self.inner.update_index_after_delete(id, namespace).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        // List all records in namespace and filter by doc_id
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.list_all_records(Some(namespace)).await?;

        let matching: Vec<_> = records
            .iter()
            .filter(|r| {
                r.metadata
                    .as_ref()
                    .and_then(doc_id_from_envelope)
                    .is_some_and(|d| d == doc_id)
            })
            .collect();

        for record in &matching {
            self.inner
                .delete_record(&record.id, Some(namespace))
                .await?;
            self.inner
                .update_index_after_delete(&record.id, Some(namespace))
                .await?;
        }

        Ok(matching.len())
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let namespace = query.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.list_all_records(Some(namespace)).await?;

        let mut results = Vec::new();
        for (i, record) in records.iter().enumerate() {
            // Compute similarity
            let score = match query.metric {
                Metric::Cosine => {
                    let dot: f32 = record
                        .vector
                        .iter()
                        .zip(query.vector.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    let norm_a: f32 = record.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                    let norm_b: f32 = query.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if norm_a > 0.0 && norm_b > 0.0 {
                        dot / (norm_a * norm_b)
                    } else {
                        0.0
                    }
                }
                Metric::Euclidean => {
                    let dist: f32 = record
                        .vector
                        .iter()
                        .zip(query.vector.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        .sqrt();
                    1.0 / (1.0 + dist)
                }
                Metric::DotProduct => record
                    .vector
                    .iter()
                    .zip(query.vector.iter())
                    .map(|(a, b)| a * b)
                    .sum(),
                Metric::Manhattan => {
                    let dist: f32 = record
                        .vector
                        .iter()
                        .zip(query.vector.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum();
                    1.0 / (1.0 + dist)
                }
            };

            let chunk = chunk_envelope::decode_chunk(record, Some(namespace))?;
            results.push(SearchResult::new(chunk, score, i));
        }

        // Sort by score and limit
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(query.limit);

        // Re-rank
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        // S3 doesn't have a direct way to list namespaces, return empty for now
        // Could be enhanced to scan bucket prefixes
        Ok(Vec::new())
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let records = self.inner.list_all_records(Some(namespace)).await?;
        for record in records {
            self.inner
                .delete_record(&record.id, Some(namespace))
                .await?;
        }
        let empty = NamespaceIndex {
            namespace: namespace.to_string(),
            record_ids: Vec::new(),
            total_vectors: 0,
            dimension: None,
            last_updated: time::OffsetDateTime::now_utc().unix_timestamp_nanos(),
        };
        self.inner.save_namespace_index(&empty).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.list_all_records(Some(namespace)).await?;
        Ok(records.len())
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.list_all_records(Some(namespace)).await?;

        let mut chunks = Vec::with_capacity(records.len());
        for record in records {
            chunks.push(chunk_envelope::decode_chunk(&record, None)?);
        }

        if let Some(limit) = limit {
            chunks.truncate(limit);
        }

        Ok(chunks)
    }
}

#[cfg(test)]
mod codec_tests {
    use super::*;
    use crate::search::multi_vectors_from_owned;

    #[test]
    fn s3_v1_round_trips_multi_vectors() {
        let record = EmbeddingRecord::new("id", vec![0.25_f32, 0.75])
            .with_namespace("ns")
            .with_multi_vectors(multi_vectors_from_owned(vec![
                vec![1.0, 0.0],
                vec![0.0, 1.0],
            ]));
        let bytes = S3EmbeddingBackend::serialize_record(&record).expect("ser");
        assert!(bytes.starts_with(S3_RECORD_V1_MAGIC));
        let back = S3EmbeddingBackend::deserialize_record(&bytes).expect("de");
        assert_eq!(back.id.as_ref(), "id");
        assert_eq!(back.multi_vectors.as_ref().unwrap().len(), 2);
        assert_eq!(
            back.multi_vectors.as_ref().unwrap()[1].as_ref(),
            &[0.0, 1.0]
        );
    }

    #[tokio::test]
    async fn upsert_makes_search_and_count_see_the_vector() {
        use crate::core::{Chunk, Metric, SearchQuery};
        use crate::storage::MemoryObjectStore;
        use crate::traits::Backend;

        let store = MemoryObjectStore::shared();
        let backend = S3Backend::with_object_store(
            store,
            S3BackendConfig {
                bucket: "bucket".to_string(),
                key_prefix: "embeddings/".to_string(),
                enable_compression: false,
                ..S3BackendConfig::default()
            },
        );
        let chunk = Chunk::new(
            "doc",
            0,
            "body",
            vec![1.0, 0.0, 0.0, 0.0],
            0,
            4,
            1,
            serde_json::json!({ "title": "doc" }),
            Some("documents".to_string()),
        );
        backend
            .upsert(std::slice::from_ref(&chunk))
            .await
            .expect("upsert");
        assert_eq!(backend.count(Some("documents")).await.expect("count"), 1);
        let hits = backend
            .search(
                &SearchQuery::new(vec![1.0, 0.0, 0.0, 0.0])
                    .metric(Metric::Cosine)
                    .limit(5)
                    .namespace("documents"),
            )
            .await
            .expect("search");
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk.id, chunk.id);
        backend
            .delete(&chunk.id, Some("documents"))
            .await
            .expect("delete");
        assert_eq!(
            backend
                .count(Some("documents"))
                .await
                .expect("count after delete"),
            0
        );
    }

    #[test]
    fn s3_rejects_unknown_magic() {
        let err = S3EmbeddingBackend::deserialize_record(b"XXXX").expect_err("unknown magic");
        let message = err.to_string();
        assert!(message.contains("S3V1"), "unexpected error: {message}");
    }
}
