//! S3 Vector Backend implementation using S3 Vector Buckets API
//!
//! This backend uses AWS S3 Vector Buckets (preview feature) for native vector storage
//! and server-side similarity search.

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use super::vectors::{
    QueryResult, S3VectorClient, VectorConfig, VectorData, VectorDistanceMetric, VectorInsert,
};
use crate::core::{Capabilities, Chunk, Metric, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::search::Filter;
use crate::traits::Backend;
use crate::utils::build_composite_key;

const INTERNAL_NAMESPACE_METADATA_KEY: &str = "__dripstone_namespace";

/// Configuration for S3 Vector Backend
#[derive(Debug, Clone)]
pub struct S3VectorBackendConfig {
    /// Vector bucket name
    pub bucket_name: String,
    /// Vector index name
    pub index_name: String,
    /// Vector dimension (1-4096)
    pub dimension: usize,
    /// Distance metric
    pub distance_metric: Metric,
    /// AWS region
    pub region: String,
    /// Batch size for bulk operations
    pub batch_size: usize,
    /// AWS access key ID (optional, uses default credentials if not set)
    pub access_key_id: Option<String>,
    /// AWS secret access key (optional, uses default credentials if not set)
    pub secret_access_key: Option<String>,
    /// AWS session token for temporary credentials (optional)
    pub session_token: Option<String>,
    /// Standard S3 bucket for `__dripstone_indexes/.../_keys.json` (vector buckets are not object stores).
    pub sidecar_s3_bucket: Option<String>,
    /// Oversample factor for `QueryVectors` when client-side refinement is enabled.
    pub search_oversample_factor: usize,
    /// Hard upper bound on the number of S3 candidates fetched for client-side refinement.
    pub search_oversample_max_candidates: usize,
    /// Whether to fetch candidate vectors and recompute exact scores locally.
    pub search_refine_with_vectors: bool,
}

impl S3VectorBackendConfig {
    pub fn new(
        bucket_name: impl Into<String>,
        index_name: impl Into<String>,
        dimension: usize,
        access_key_id: Option<String>,
        secret_access_key: Option<String>,
        session_token: Option<String>,
    ) -> Self {
        Self {
            bucket_name: bucket_name.into(),
            index_name: index_name.into(),
            dimension,
            distance_metric: Metric::Cosine,
            region: "us-east-1".to_string(),
            batch_size: 100,
            access_key_id,
            secret_access_key,
            session_token,
            sidecar_s3_bucket: None,
            search_oversample_factor: 4,
            search_oversample_max_candidates: 64,
            search_refine_with_vectors: true,
        }
    }

    #[must_use]
    pub fn with_sidecar_s3_bucket(mut self, bucket: impl Into<String>) -> Self {
        self.sidecar_s3_bucket = Some(bucket.into());
        self
    }

    #[must_use]
    pub const fn with_distance_metric(mut self, metric: Metric) -> Self {
        self.distance_metric = metric;
        self
    }

    #[must_use]
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        self.region = region.into();
        self
    }

    #[must_use]
    pub const fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    #[must_use]
    pub fn with_search_oversample_factor(mut self, factor: usize) -> Self {
        self.search_oversample_factor = factor.max(1);
        self
    }

    #[must_use]
    pub fn with_search_oversample_max_candidates(mut self, max_candidates: usize) -> Self {
        self.search_oversample_max_candidates = max_candidates.max(1);
        self
    }

    #[must_use]
    pub const fn with_search_refine_with_vectors(mut self, enabled: bool) -> Self {
        self.search_refine_with_vectors = enabled;
        self
    }
}

/// S3 Vector Backend using native S3 Vector Buckets
#[derive(Debug)]
pub struct S3VectorBackend {
    client: Arc<S3VectorClient>,
    config: S3VectorBackendConfig,
    /// Track vector keys per namespace for listing
    namespace_index: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl S3VectorBackend {
    fn requested_candidate_count(&self, limit: usize) -> usize {
        if limit == 0 {
            return 0;
        }

        if !self.config.search_refine_with_vectors {
            return limit;
        }

        let oversampled = limit.saturating_mul(self.config.search_oversample_factor.max(1));
        oversampled
            .max(limit)
            .min(self.config.search_oversample_max_candidates.max(limit))
    }

    /// Create a new S3 Vector Backend with optional AWS credentials
    ///
    /// Credentials can be provided via:
    /// 1. Config fields (takes precedence)
    /// 2. Environment variables (fallback)
    /// 3. Default AWS credentials chain (if neither provided)
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(config: S3VectorBackendConfig) -> Result<Self, DripstoneError> {
        use crate::aws::{AwsSdkConfigOpts, load_sdk_config};

        let aws_config = load_sdk_config(AwsSdkConfigOpts {
            region: Some(config.region.as_str()),
            endpoint_url: None,
            access_key_id: config.access_key_id.as_deref(),
            secret_access_key: config.secret_access_key.as_deref(),
            session_token: config.session_token.as_deref(),
            credentials_provider_name: "s3-vector-backend",
        })
        .await?;
        let aws_client = aws_sdk_s3::Client::new(&aws_config);

        Self::with_aws_client(config, aws_config, aws_client)
    }

    /// Create with an existing AWS S3 client and the `SdkConfig` used to build it.
    ///
    /// Pass the same `SdkConfig` so SigV4 signing for `s3vectors` uses the same
    /// credentials as sidecar S3 object operations.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn with_aws_client(
        config: S3VectorBackendConfig,
        aws_config: aws_config::SdkConfig,
        aws_client: aws_sdk_s3::Client,
    ) -> Result<Self, DripstoneError> {
        let vector_config = VectorConfig {
            bucket_name: config.bucket_name.clone(),
            index_name: config.index_name.clone(),
            dimension: config.dimension,
            distance_metric: VectorDistanceMetric::try_from_core_metric(config.distance_metric)?,
            region: config.region.clone(),
            sidecar_s3_bucket: config.sidecar_s3_bucket.clone(),
        };

        let client = S3VectorClient::new(vector_config, aws_config, aws_client)?;

        Ok(Self {
            client: Arc::new(client),
            config,
            namespace_index: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Generate a namespaced key for S3 Vectors
    fn namespaced_key(id: &str, namespace: Option<&str>) -> String {
        build_composite_key(namespace, id).into_owned()
    }

    /// Parse namespace from a namespaced key
    fn parse_key(key: &str) -> (Option<String>, String) {
        if let Some((ns, id)) = key.split_once(':') {
            if ns == "default" {
                (None, id.to_string())
            } else {
                (Some(ns.to_string()), id.to_string())
            }
        } else {
            (None, key.to_string())
        }
    }

    fn metadata_namespace(namespace: Option<&str>) -> &str {
        namespace.unwrap_or("default")
    }

    /// Convert Chunk to `VectorInsert`
    fn chunk_to_insert(&self, chunk: &Chunk) -> Result<VectorInsert, DripstoneError> {
        if chunk.vector.len() != self.config.dimension {
            return Err(DripstoneError::storage(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.config.dimension,
                chunk.vector.len()
            )));
        }

        let key = Self::namespaced_key(&chunk.id, chunk.namespace.as_deref());

        // Store namespace in filterable metadata so direct S3 queries stay namespace-scoped.
        let mut metadata: HashMap<String, serde_json::Value> = chunk
            .metadata
            .as_object()
            .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        metadata.insert(
            INTERNAL_NAMESPACE_METADATA_KEY.to_string(),
            serde_json::json!(Self::metadata_namespace(chunk.namespace.as_deref())),
        );

        Ok(VectorInsert {
            key,
            data: VectorData {
                // Convert Arc<[f32]> to Vec<f32> for S3 vector API
                float32: chunk.vector.to_vec(),
            },
            metadata: Some(metadata),
        })
    }

    fn metadata_from_s3(
        metadata: Option<&HashMap<String, serde_json::Value>>,
    ) -> serde_json::Value {
        let mut object: serde_json::Map<String, serde_json::Value> = metadata
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        object.remove(INTERNAL_NAMESPACE_METADATA_KEY);
        serde_json::Value::Object(object)
    }

    fn namespace_scoped_filter(query: &SearchQuery) -> Option<HashMap<String, serde_json::Value>> {
        let mut filter = query
            .filter
            .as_ref()
            .and_then(Self::filter_to_s3_filter)
            .unwrap_or_default();

        if let Some(namespace) = query.namespace.as_deref() {
            filter.insert(
                INTERNAL_NAMESPACE_METADATA_KEY.to_string(),
                serde_json::json!(Self::metadata_namespace(Some(namespace))),
            );
        }

        if filter.is_empty() {
            None
        } else {
            Some(filter)
        }
    }

    fn query_result_to_chunk_with_vector(
        &self,
        result: &QueryResult,
        vector: Option<&[f32]>,
        include_vector: bool,
    ) -> Chunk {
        let (namespace, id) = Self::parse_key(&result.key);

        // Parse doc_id and chunk_index from id (format: "doc-id#chunk-index")
        let (doc_id, chunk_index) = if let Some((doc_id, idx_str)) = id.split_once('#') {
            let idx = idx_str.parse::<usize>().unwrap_or(0);
            (doc_id.to_string(), idx)
        } else {
            (id, 0)
        };

        let metadata = Self::metadata_from_s3(result.metadata.as_ref());
        let vector = if include_vector {
            vector.map_or_else(
                || vec![0.0; self.config.dimension],
                std::borrow::ToOwned::to_owned,
            )
        } else {
            vec![0.0; self.config.dimension]
        };

        // We don't have content, start_offset, end_offset, total_chunks from S3 Vectors
        // Use defaults - in practice, you might want to store these in metadata
        Chunk::new(
            doc_id,
            chunk_index,
            "", // Content not available from S3 Vectors
            vector,
            0, // start_offset
            0, // end_offset
            1, // total_chunks (assume 1 if not known)
            metadata,
            namespace,
        )
    }

    async fn fetch_query_result_vectors(
        &self,
        results: &[QueryResult],
    ) -> Result<HashMap<String, Vec<f32>>, DripstoneError> {
        let mut vectors = HashMap::with_capacity(results.len());
        if results.is_empty() {
            return Ok(vectors);
        }

        let batch_size = self.config.batch_size.max(1);
        let keys: Vec<String> = results.iter().map(|result| result.key.clone()).collect();
        for batch in keys.chunks(batch_size) {
            let retrieved = self.client.get_vectors(batch.to_vec()).await?;
            for retrieved_vec in retrieved {
                vectors.insert(retrieved_vec.key, retrieved_vec.data.float32);
            }
        }

        Ok(vectors)
    }

    fn score_query_result(
        query: &SearchQuery,
        s3_metric: VectorDistanceMetric,
        query_result: &QueryResult,
        vector: Option<&[f32]>,
        prefer_exact_vector: bool,
    ) -> Result<f32, DripstoneError> {
        if prefer_exact_vector && let Some(vector) = vector {
            return query.metric.compute(&query.vector, vector);
        }

        match s3_metric {
            VectorDistanceMetric::Cosine => query_result.distance.map_or_else(
                || {
                    vector.map_or(Ok(0.0), |vector| {
                        query.metric.compute(&query.vector, vector)
                    })
                },
                |distance| Ok((1.0 - distance).clamp(0.0, 1.0)),
            ),
            VectorDistanceMetric::Euclidean => query_result.distance.map_or_else(
                || {
                    vector.map_or(Ok(f32::INFINITY), |vector| {
                        query.metric.compute(&query.vector, vector)
                    })
                },
                Ok,
            ),
        }
    }

    fn sort_results(query: &SearchQuery, results: &mut Vec<SearchResult>) {
        if query.metric.higher_is_better() {
            results.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            results.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        results.truncate(query.limit);
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }
    }

    /// Convert S3 Vector `RetrievedVector` to Chunk
    fn retrieved_vector_to_chunk(retrieved: &super::vectors::RetrievedVector) -> Chunk {
        let (namespace, id) = Self::parse_key(&retrieved.key);

        // Parse doc_id and chunk_index from id
        let (doc_id, chunk_index) = if let Some((doc_id, idx_str)) = id.split_once('#') {
            let idx = idx_str.parse::<usize>().unwrap_or(0);
            (doc_id.to_string(), idx)
        } else {
            (id, 0)
        };

        let metadata = Self::metadata_from_s3(retrieved.metadata.as_ref());

        Chunk::new(
            doc_id,
            chunk_index,
            "", // Content not available
            retrieved.data.float32.clone(),
            0,
            0,
            1,
            metadata,
            namespace,
        )
    }

    /// Convert Filter to S3 Vector filter format
    fn filter_to_s3_filter(filter: &Filter) -> Option<HashMap<String, serde_json::Value>> {
        match filter {
            Filter::Equal { field, value } => {
                let mut map = HashMap::new();
                map.insert(field.clone(), value.clone());
                Some(map)
            }
            // S3 Vectors may not support all filter types, so we return None for unsupported ones
            _ => None,
        }
    }

    /// Update namespace index after insert
    fn add_to_namespace_index(&self, key: &str, namespace: Option<&str>) {
        let ns = namespace.unwrap_or("default");
        let mut index = self.namespace_index.write();
        let entry = index.entry(ns.to_string()).or_default();

        if !entry.contains(&key.to_string()) {
            entry.push(key.to_string());
        }
        drop(index);
    }

    /// Remove from namespace index
    fn remove_from_namespace_index(&self, key: &str, namespace: Option<&str>) {
        let ns = namespace.unwrap_or("default");
        let mut index = self.namespace_index.write();

        if let Some(entry) = index.get_mut(ns) {
            entry.retain(|item| item != key);
        }
    }

    /// Get all keys in a namespace
    async fn get_namespace_keys(
        &self,
        namespace: Option<&str>,
    ) -> Result<Vec<String>, DripstoneError> {
        let ns = namespace.unwrap_or("default");
        let keys = self.client.list_vectors_in_namespace(ns).await?;

        // Cache locally (best-effort)
        {
            let mut index = self.namespace_index.write();
            index.insert(ns.to_string(), keys.clone());
        }

        Ok(keys)
    }
}

#[async_trait]
impl Backend for S3VectorBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false, // S3 Vectors doesn't support sparse vectors natively
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true, // S3 Vectors supports metadata filtering
            requires_exact_filter_pass: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: Some(4096), // S3 Vectors limit
            max_batch: Some(self.config.batch_size),
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Convert chunks to VectorInsert format
        let mut inserts = Vec::with_capacity(chunks.len());
        let mut keys_by_namespace: HashMap<String, Vec<String>> = HashMap::new();
        for chunk in chunks {
            let insert = self.chunk_to_insert(chunk)?;
            keys_by_namespace
                .entry(chunk.namespace.as_deref().unwrap_or("default").to_string())
                .or_default()
                .push(insert.key.clone());
            inserts.push(insert);
        }

        // Batch inserts
        for batch in inserts.chunks(self.config.batch_size) {
            self.client.put_vectors(batch.to_vec()).await?;
        }

        // Update durable + in-memory indexes after successful write
        for (ns, keys) in keys_by_namespace {
            self.client.index_add_keys(&ns, keys.clone()).await?;
            for key in keys {
                // For default namespace, the key is still "default:<id>"
                self.add_to_namespace_index(&key, Some(&ns));
            }
        }

        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let key = Self::namespaced_key(id, namespace);
        let retrieved = match self.client.get_vectors(vec![key]).await {
            Ok(retrieved) => retrieved,
            Err(err) if absent_vector_key(&err) => return Ok(None),
            Err(err) => return Err(err),
        };

        Ok(retrieved.first().map(Self::retrieved_vector_to_chunk))
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let key = Self::namespaced_key(id, namespace);
        self.client.delete_vectors(vec![key]).await?;
        let ns = namespace.unwrap_or("default");
        self.client
            .index_remove_keys(ns, vec![Self::namespaced_key(id, namespace)])
            .await?;
        self.remove_from_namespace_index(&Self::namespaced_key(id, namespace), Some(ns));
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        // Get all chunks for this document from namespace index
        let keys = self.get_namespace_keys(namespace).await?;
        let doc_keys: Vec<String> = keys
            .into_iter()
            .filter(|k| {
                let (_, id) = Self::parse_key(k);
                id.starts_with(&format!("{doc_id}#")) || id == doc_id
            })
            .collect();

        if !doc_keys.is_empty() {
            // Delete in batches
            for batch in doc_keys.chunks(self.config.batch_size) {
                self.client.delete_vectors(batch.to_vec()).await?;
            }

            // Remove from durable + in-memory index
            let ns = namespace.unwrap_or("default");
            self.client.index_remove_keys(ns, doc_keys.clone()).await?;
            for key in &doc_keys {
                self.remove_from_namespace_index(key, Some(ns));
            }
        }

        Ok(doc_keys.len())
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        if query.limit == 0 {
            return Ok(Vec::new());
        }

        // Validate metric - S3 Vectors only supports Cosine and Euclidean
        let s3_metric = VectorDistanceMetric::try_from_core_metric(query.metric)?;

        let s3_filter = Self::namespace_scoped_filter(query);
        let requested_candidates = self.requested_candidate_count(query.limit);

        // Query S3 Vectors - convert Arc<[f32]> to Vec<f32> for S3 API
        let query_results = self
            .client
            .query_vectors(query.vector.to_vec(), requested_candidates, s3_filter)
            .await?;

        let should_refine =
            self.config.search_refine_with_vectors && query_results.len() > query.limit;
        let prefetched_vectors = if should_refine || query.include_vectors {
            self.fetch_query_result_vectors(&query_results).await?
        } else {
            HashMap::new()
        };

        // Convert to SearchResults
        let mut results: Vec<SearchResult> = Vec::with_capacity(query_results.len());
        for (rank, query_result) in query_results.iter().enumerate() {
            let vector = prefetched_vectors.get(&query_result.key).map(Vec::as_slice);
            let chunk =
                self.query_result_to_chunk_with_vector(query_result, vector, query.include_vectors);
            let score =
                Self::score_query_result(query, s3_metric, query_result, vector, should_refine)?;

            results.push(SearchResult::new(chunk, score, rank));
        }

        Self::sort_results(query, &mut results);

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.client.list_namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        // Get all keys in namespace
        let keys = self.get_namespace_keys(Some(namespace)).await?;

        if !keys.is_empty() {
            // Delete in batches
            for batch in keys.chunks(self.config.batch_size) {
                self.client.delete_vectors(batch.to_vec()).await?;
            }
        }

        // Delete durable index file and clear local cache
        self.client.delete_namespace_index(namespace).await?;
        let mut index = self.namespace_index.write();
        index.remove(namespace);
        drop(index);

        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let keys = self.get_namespace_keys(namespace).await?;
        Ok(keys.len())
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let keys = self.get_namespace_keys(namespace).await?;

        let keys_to_fetch = if let Some(limit) = limit {
            keys.into_iter().take(limit).collect()
        } else {
            keys
        };

        if keys_to_fetch.is_empty() {
            return Ok(Vec::new());
        }

        // Fetch chunks in batches
        let mut all_chunks = Vec::new();
        for batch in keys_to_fetch.chunks(self.config.batch_size) {
            let retrieved = self.client.get_vectors(batch.to_vec()).await?;
            for retrieved_vec in retrieved {
                let chunk = Self::retrieved_vector_to_chunk(&retrieved_vec);
                all_chunks.push(chunk);
            }
        }

        Ok(all_chunks)
    }

    async fn flush(&self) -> Result<(), DripstoneError> {
        // S3 Vectors API operations are synchronous, so flush is a no-op
        Ok(())
    }

    async fn close(&self) -> Result<(), DripstoneError> {
        // No resources to close
        Ok(())
    }
}

/// A missing vector key is `Ok(None)`. A missing index or bucket stays an error.
fn absent_vector_key(err: &DripstoneError) -> bool {
    if !err.is_not_found() {
        return false;
    }
    let text = err.to_string().to_ascii_lowercase();
    if text.contains("index") || text.contains("bucket") || text.contains("nosuchbucket") {
        return false;
    }
    text.contains("vector") || text.contains("nosuchkey") || text.contains("specified key")
}

#[cfg(test)]
mod absent_key_tests {
    use super::absent_vector_key;
    use crate::errors::DripstoneError;

    #[test]
    fn missing_vector_key_is_absent_and_missing_index_is_not() {
        let key =
            DripstoneError::storage("GetVectors: NoSuchKey: The specified key does not exist.");
        assert!(absent_vector_key(&key));
        let index = DripstoneError::storage("the vector index was not found");
        assert!(!absent_vector_key(&index));
        let denied = DripstoneError::storage("AccessDenied");
        assert!(!absent_vector_key(&denied));
    }
}
