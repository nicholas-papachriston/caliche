use async_trait::async_trait;
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, DeletePointsBuilder, Distance, GetPointsBuilder, PointStruct,
    ScrollPointsBuilder, SearchPointsBuilder, UpsertPointsBuilder, VectorParams, VectorsConfig,
    vectors_output::VectorsOptions as OutputVectorsOptions,
};
use std::collections::HashMap;
use uuid::{NoContext, Timestamp, Uuid};

use crate::backend::chunk_envelope::{self, doc_id_from_envelope};
use crate::embedding::{DEFAULT_NAMESPACE, EmbeddingRecord, SimilarityMetric, SimilarityResult};
use crate::errors::DripstoneError;

use super::helpers::{ORIGINAL_ID_KEY, ORIGINAL_NORM_KEY, normalize_prefix};

const DEFAULT_BATCH_SIZE: usize = 500;

/// Helper to extract string from Qdrant's Value type
fn extract_string_from_value(value: &qdrant_client::qdrant::Value) -> Option<String> {
    value.kind.as_ref().and_then(|kind| match kind {
        qdrant_client::qdrant::value::Kind::StringValue(s) => Some(s.clone()),
        _ => None,
    })
}

pub struct QdrantGrpcBackend {
    client: Qdrant,
    collection_prefix: String,
    vector_dimension: usize,
    distance: SimilarityMetric,
    #[expect(dead_code, reason = "stored from the config but not read yet")]
    batch_size: usize,
}

#[derive(Clone, Debug)]
pub struct QdrantGrpcBackendOptions {
    pub grpc_url: String,
    pub collection_prefix: Option<String>,
    pub vector_dimension: usize,
    pub distance: SimilarityMetric,
    pub api_key: Option<String>,
    pub batch_size: Option<usize>,
}

impl QdrantGrpcBackendOptions {
    pub fn new(grpc_url: impl Into<String>, vector_dimension: usize) -> Self {
        Self {
            grpc_url: grpc_url.into(),
            collection_prefix: None,
            vector_dimension,
            distance: SimilarityMetric::Cosine,
            api_key: None,
            batch_size: None,
        }
    }
}

impl QdrantGrpcBackend {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(options: QdrantGrpcBackendOptions) -> Result<Self, DripstoneError> {
        if options.vector_dimension == 0 {
            return Err(DripstoneError::Misconfiguration(
                "vector dimension must be greater than zero".to_string(),
            ));
        }

        let mut client_builder = Qdrant::from_url(&options.grpc_url);
        if let Some(ref api_key) = options.api_key {
            client_builder = client_builder.api_key(api_key.clone());
        }

        let client = client_builder.build().map_err(|e| {
            DripstoneError::storage_with_source("failed to create qdrant grpc client", e)
        })?;

        let prefix = normalize_prefix(options.collection_prefix);
        let batch_size = options.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);

        let backend = Self {
            client,
            collection_prefix: prefix,
            vector_dimension: options.vector_dimension,
            distance: options.distance,
            batch_size,
        };

        backend.health_check().await?;
        Ok(backend)
    }

    fn collection_name(&self, namespace: &str) -> String {
        format!("{}{}", self.collection_prefix, namespace)
    }

    async fn health_check(&self) -> Result<(), DripstoneError> {
        self.client.health_check().await.map_err(|e| {
            DripstoneError::storage_with_source("qdrant grpc health check failed", e)
        })?;
        Ok(())
    }

    async fn ensure_collection(&self, namespace: &str) -> Result<(), DripstoneError> {
        let collection = self.collection_name(namespace);

        let exists = self
            .client
            .collection_exists(&collection)
            .await
            .map_err(|e| {
                DripstoneError::storage_with_source("failed to check collection existence", e)
            })?;

        if !exists {
            let distance = match self.distance {
                SimilarityMetric::Cosine => Distance::Cosine,
                SimilarityMetric::Dot => Distance::Dot,
                // Qdrant doesn't support Manhattan/Chebyshev natively, use Euclidean as fallback.
                SimilarityMetric::Euclidean
                | SimilarityMetric::Manhattan
                | SimilarityMetric::Chebyshev => Distance::Euclid,
            };

            let vector_params = VectorParams {
                size: self.vector_dimension as u64,
                distance: distance.into(),
                ..Default::default()
            };
            let vectors_config = VectorsConfig {
                config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                    vector_params,
                )),
            };

            self.client
                .create_collection(
                    CreateCollectionBuilder::new(&collection).vectors_config(vectors_config),
                )
                .await
                .map_err(|e| {
                    DripstoneError::storage_with_source("failed to create collection", e)
                })?;
        }

        Ok(())
    }

    fn record_to_point(record: &EmbeddingRecord) -> Result<PointStruct, DripstoneError> {
        use qdrant_client::qdrant::Value as QdrantValue;

        let seconds = u64::try_from(record.created_at.unix_timestamp()).map_err(|_| {
            DripstoneError::storage("qdrant point timestamp predates the Unix epoch")
        })?;
        let subsec_nanos = record.created_at.nanosecond();
        let id = Uuid::new_v7(Timestamp::from_unix(NoContext, seconds, subsec_nanos)).to_string();
        let mut payload: HashMap<String, QdrantValue> = HashMap::new();
        payload.insert(ORIGINAL_ID_KEY.to_string(), record.id.to_string().into());
        payload.insert(
            "created_at".to_string(),
            EmbeddingRecord::format_timestamp(&record.created_at)?.into(),
        );
        payload.insert(
            "updated_at".to_string(),
            EmbeddingRecord::format_timestamp(&record.updated_at)?.into(),
        );

        if let Some(metadata) = &record.metadata {
            let qdrant_value = serde_json::to_string(metadata).map_err(|e| {
                DripstoneError::storage_with_source("failed to serialize metadata", e)
            })?;
            payload.insert("metadata".to_string(), qdrant_value.into());
        }

        let magnitude: f32 = record.vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        payload.insert(ORIGINAL_NORM_KEY.to_string(), magnitude.into());

        // Convert Arc<[f32]> to Vec<f32> for Qdrant API
        Ok(PointStruct::new(id, record.vector.to_vec(), payload))
    }

    fn scored_point_to_record(
        point: qdrant_client::qdrant::ScoredPoint,
    ) -> Result<EmbeddingRecord, DripstoneError> {
        let payload = point.payload;

        let id = payload
            .get(ORIGINAL_ID_KEY)
            .and_then(extract_string_from_value)
            .ok_or_else(|| DripstoneError::storage("missing original id in point payload"))?;

        let vector = point
            .vectors
            .ok_or_else(|| DripstoneError::storage("missing vectors in point"))?
            .vectors_options
            .ok_or_else(|| DripstoneError::storage("missing vector options"))?;

        let vector = match vector {
            OutputVectorsOptions::Vector(v) => {
                use qdrant_client::qdrant::vector_output::Vector;
                match v.vector {
                    Some(Vector::Dense(dense)) => dense.data,
                    _ => {
                        return Err(DripstoneError::storage(
                            "unexpected vector format: expected dense vector",
                        ));
                    }
                }
            }
            OutputVectorsOptions::Vectors(_) => {
                return Err(DripstoneError::storage("unexpected vector format"));
            }
        };

        let created_at = payload
            .get("created_at")
            .and_then(extract_string_from_value)
            .ok_or_else(|| DripstoneError::storage("missing created_at"))?;
        let created_at = EmbeddingRecord::parse_timestamp(&created_at)?;

        let updated_at = payload
            .get("updated_at")
            .and_then(extract_string_from_value)
            .ok_or_else(|| DripstoneError::storage("missing updated_at"))?;
        let updated_at = EmbeddingRecord::parse_timestamp(&updated_at)?;

        let metadata = payload
            .get("metadata")
            .and_then(extract_string_from_value)
            .and_then(|s| serde_json::from_str(&s).ok());

        Ok(EmbeddingRecord {
            id: std::sync::Arc::from(id),
            vector: vector.into(),
            namespace: None,
            metadata,
            sparse_vector: None,
            multi_vectors: None,
            created_at,
            updated_at,
        })
    }

    fn retrieved_point_to_record(
        point: qdrant_client::qdrant::RetrievedPoint,
    ) -> Result<EmbeddingRecord, DripstoneError> {
        let payload = point.payload;

        let id = payload
            .get(ORIGINAL_ID_KEY)
            .and_then(extract_string_from_value)
            .ok_or_else(|| DripstoneError::storage("missing original id in point payload"))?;

        let vector = point
            .vectors
            .ok_or_else(|| DripstoneError::storage("missing vectors in point"))?
            .vectors_options
            .ok_or_else(|| DripstoneError::storage("missing vector options"))?;

        let vector = match vector {
            OutputVectorsOptions::Vector(v) => {
                use qdrant_client::qdrant::vector_output::Vector;
                match v.vector {
                    Some(Vector::Dense(dense)) => dense.data,
                    _ => {
                        return Err(DripstoneError::storage(
                            "unexpected vector format: expected dense vector",
                        ));
                    }
                }
            }
            OutputVectorsOptions::Vectors(_) => {
                return Err(DripstoneError::storage("unexpected vector format"));
            }
        };

        let created_at = payload
            .get("created_at")
            .and_then(extract_string_from_value)
            .ok_or_else(|| DripstoneError::storage("missing created_at"))?;
        let created_at = EmbeddingRecord::parse_timestamp(&created_at)?;

        let updated_at = payload
            .get("updated_at")
            .and_then(extract_string_from_value)
            .ok_or_else(|| DripstoneError::storage("missing updated_at"))?;
        let updated_at = EmbeddingRecord::parse_timestamp(&updated_at)?;

        let metadata = payload
            .get("metadata")
            .and_then(extract_string_from_value)
            .and_then(|s| serde_json::from_str(&s).ok());

        Ok(EmbeddingRecord {
            id: std::sync::Arc::from(id),
            vector: vector.into(),
            namespace: None,
            metadata,
            sparse_vector: None,
            multi_vectors: None,
            created_at,
            updated_at,
        })
    }

    // Helper methods used by QdrantBackend for delegation

    /// Find point ID by original ID stored in payload
    async fn find_point_id_by_original_id(
        &self,
        collection: &str,
        original_id: &str,
    ) -> Result<Option<qdrant_client::qdrant::PointId>, DripstoneError> {
        // Scroll through points to find the one with matching original_id
        let mut offset = None;
        loop {
            let mut scroll_builder = ScrollPointsBuilder::new(collection)
                .with_payload(true)
                .with_vectors(false)
                .limit(100);

            if let Some(off) = offset {
                scroll_builder = scroll_builder.offset(off);
            }

            let scroll_result =
                self.client.scroll(scroll_builder).await.map_err(|e| {
                    DripstoneError::storage_with_source("failed to scroll points", e)
                })?;

            let points = scroll_result.result;
            let next_offset = scroll_result.next_page_offset;

            for point in points {
                let payload = point.payload;
                if let Some(id) = payload
                    .get(ORIGINAL_ID_KEY)
                    .and_then(extract_string_from_value)
                    && id == original_id
                {
                    // Found it - return the point ID
                    return Ok(point.id);
                }
            }

            offset = next_offset;
            if offset.is_none() {
                break;
            }
        }
        Ok(None)
    }

    pub(crate) async fn upsert_embedding(
        &self,
        record: EmbeddingRecord,
    ) -> Result<(), DripstoneError> {
        let namespace = record.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);
        self.ensure_collection(namespace).await?;

        let collection = self.collection_name(namespace);
        let point = Self::record_to_point(&record)?;

        self.client
            .upsert_points(UpsertPointsBuilder::new(&collection, vec![point]))
            .await
            .map_err(|e| DripstoneError::storage_with_source("failed to upsert point", e))?;

        Ok(())
    }

    pub(crate) async fn get_embedding(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<EmbeddingRecord>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.collection_name(namespace);

        // Find the point ID by searching for the original_id in payload
        let point_id = self.find_point_id_by_original_id(&collection, id).await?;

        if let Some(pid) = point_id {
            // Retrieve the point by ID
            let points = self
                .client
                .get_points(
                    GetPointsBuilder::new(&collection, vec![pid])
                        .with_payload(true)
                        .with_vectors(true),
                )
                .await
                .map_err(|e| DripstoneError::storage_with_source("failed to get point", e))?;

            if let Some(point) = points.result.first() {
                let record = Self::retrieved_point_to_record(point.clone())?;
                return Ok(Some(record));
            }
        }

        Ok(None)
    }

    pub(crate) async fn delete_embedding(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<(), DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.collection_name(namespace);

        // Find the point ID by searching for the original_id in payload
        let point_id = self.find_point_id_by_original_id(&collection, id).await?;

        if let Some(pid) = point_id {
            use qdrant_client::qdrant::PointsIdsList;
            self.client
                .delete_points(
                    DeletePointsBuilder::new(&collection)
                        .points(PointsIdsList { ids: vec![pid] })
                        .wait(true),
                )
                .await
                .map_err(|e| DripstoneError::storage_with_source("failed to delete point", e))?;
        }

        Ok(())
    }

    pub(crate) async fn similarity_search(
        &self,
        vector: &[f32],
        limit: usize,
        metric: SimilarityMetric,
        namespace: Option<&str>,
    ) -> Result<Vec<SimilarityResult>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.ensure_collection(namespace).await?;

        let collection = self.collection_name(namespace);

        let search_result = self
            .client
            .search_points(
                SearchPointsBuilder::new(&collection, vector.to_vec(), limit as u64)
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await
            .map_err(|e| DripstoneError::storage_with_source("failed to search points", e))?;

        let mut results = Vec::new();
        for scored_point in search_result.result {
            let record = Self::scored_point_to_record(scored_point.clone())?;
            // Convert Qdrant score to similarity score
            // Qdrant returns distance for Euclidean, similarity for Cosine/Dot
            let score = if metric.prefers_lower_scores() {
                // For distance metrics, convert to similarity (lower distance = higher similarity)
                1.0 / (1.0 + scored_point.score)
            } else {
                // For similarity metrics, use score directly
                scored_point.score
            };

            results.push(SimilarityResult::new(std::sync::Arc::new(record), score));
        }

        Ok(results)
    }

    pub(crate) async fn list_namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let collections =
            self.client.list_collections().await.map_err(|e| {
                DripstoneError::storage_with_source("failed to list collections", e)
            })?;

        let mut namespaces = Vec::new();
        for collection in collections.collections {
            let name = collection.name;
            // Remove collection prefix to get namespace
            if let Some(namespace) = name.strip_prefix(&self.collection_prefix) {
                namespaces.push(namespace.to_string());
            }
        }

        Ok(namespaces)
    }

    pub(crate) async fn flush_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let collection = self.collection_name(namespace);

        // Delete all points in the collection by scrolling and deleting
        let mut offset = None;
        loop {
            let mut scroll_builder = ScrollPointsBuilder::new(&collection)
                .with_payload(false)
                .with_vectors(false)
                .limit(1000);

            if let Some(off) = offset {
                scroll_builder = scroll_builder.offset(off);
            }

            let Ok(scroll_result) = self.client.scroll(scroll_builder).await else {
                break; // Collection might not exist
            };

            let points = scroll_result.result;
            let next_offset = scroll_result.next_page_offset;

            if points.is_empty() {
                break;
            }

            // Extract point IDs
            let point_ids: Vec<qdrant_client::qdrant::PointId> =
                points.into_iter().filter_map(|p| p.id).collect();

            if !point_ids.is_empty() {
                // Delete points
                use qdrant_client::qdrant::PointsIdsList;
                self.client
                    .delete_points(
                        DeletePointsBuilder::new(&collection)
                            .points(PointsIdsList { ids: point_ids })
                            .wait(true),
                    )
                    .await
                    .map_err(|e| {
                        DripstoneError::storage_with_source("failed to delete points", e)
                    })?;
            }

            offset = next_offset;
            if offset.is_none() {
                break;
            }
        }

        Ok(())
    }

    pub(crate) async fn count_embeddings(
        &self,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.collection_name(namespace);

        let collection_info = self
            .client
            .collection_info(&collection)
            .await
            .map_err(|e| DripstoneError::storage_with_source("failed to get collection info", e))?;

        let count = collection_info
            .result
            .and_then(|info| info.points_count)
            .unwrap_or(0);

        usize::try_from(count)
            .map_err(|_| DripstoneError::storage("qdrant count exceeds addressable size"))
    }

    pub(crate) async fn list_embeddings(
        &self,
        namespace: &str,
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        let collection = self.collection_name(namespace);
        let mut records = Vec::new();
        let mut offset = None;

        loop {
            let mut scroll_builder = ScrollPointsBuilder::new(&collection)
                .with_payload(true)
                .with_vectors(true)
                .limit(100);

            if let Some(off) = offset {
                scroll_builder = scroll_builder.offset(off);
            }

            let scroll_result =
                self.client.scroll(scroll_builder).await.map_err(|e| {
                    DripstoneError::storage_with_source("failed to scroll points", e)
                })?;

            let points = scroll_result.result;
            let next_offset = scroll_result.next_page_offset;

            for point in points {
                let record = Self::retrieved_point_to_record(point)?;
                records.push(record);
            }

            offset = next_offset;
            if offset.is_none() {
                break;
            }
        }

        Ok(records)
    }
}

// =============================================================================
// NEW API (Dripstone 0.2) - Backend trait implementation
// =============================================================================

use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::traits::Backend;
use std::fmt;

impl fmt::Debug for QdrantGrpcBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QdrantBackend")
            .field("collection_prefix", &self.collection_prefix)
            .field("vector_dimension", &self.vector_dimension)
            .finish_non_exhaustive()
    }
}

/// Qdrant backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct QdrantBackend {
    inner: QdrantGrpcBackend,
}

impl QdrantBackend {
    /// Create a new Qdrant backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn connect(grpc_url: &str, vector_dimension: usize) -> Result<Self, DripstoneError> {
        let options = QdrantGrpcBackendOptions::new(grpc_url, vector_dimension);
        let inner = QdrantGrpcBackend::new(options).await?;
        Ok(Self { inner })
    }

    /// Create a new Qdrant backend with options.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_options(options: QdrantGrpcBackendOptions) -> Result<Self, DripstoneError> {
        let inner = QdrantGrpcBackend::new(options).await?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl Backend for QdrantBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: None,
            max_batch: Some(500),
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        // Convert chunks to EmbeddingRecords for the inner record-based API
        for chunk in chunks {
            let record = chunk_envelope::embedding_record_from_chunk(chunk);
            self.inner.upsert_embedding(record).await?;
        }
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let record = self.inner.get_embedding(id, namespace).await?;
        match record {
            Some(r) => Ok(Some(chunk_envelope::decode_chunk(&r, None)?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        self.inner.delete_embedding(id, namespace).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        use qdrant_client::qdrant::PointsIdsList;

        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.inner.ensure_collection(namespace).await?;
        let collection = self.inner.collection_name(namespace);

        let mut deleted: usize = 0;
        let mut offset = None;
        let mut to_delete: Vec<qdrant_client::qdrant::PointId> = Vec::new();

        loop {
            let mut scroll_builder = ScrollPointsBuilder::new(&collection)
                .with_payload(true)
                .with_vectors(false)
                .limit(1000);

            if let Some(off) = offset {
                scroll_builder = scroll_builder.offset(off);
            }

            let Ok(scroll_result) = self.inner.client.scroll(scroll_builder).await else {
                break;
            };

            let points = scroll_result.result;
            let next_offset = scroll_result.next_page_offset;

            if points.is_empty() {
                break;
            }

            for point in points {
                let payload = point.payload;
                let metadata = payload
                    .get("metadata")
                    .and_then(extract_string_from_value)
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok());

                let matches_doc = metadata
                    .as_ref()
                    .and_then(doc_id_from_envelope)
                    .is_some_and(|v| v == doc_id);

                if matches_doc && let Some(pid) = point.id {
                    to_delete.push(pid);
                }

                if to_delete.len() >= 1000 {
                    let batch = std::mem::take(&mut to_delete);
                    let batch_len = batch.len();
                    self.inner
                        .client
                        .delete_points(
                            DeletePointsBuilder::new(&collection)
                                .points(PointsIdsList { ids: batch })
                                .wait(true),
                        )
                        .await
                        .map_err(|e| {
                            DripstoneError::storage_with_source("failed to delete points", e)
                        })?;
                    deleted += batch_len;
                }
            }

            offset = next_offset;
            if offset.is_none() {
                break;
            }
        }

        if !to_delete.is_empty() {
            let batch = std::mem::take(&mut to_delete);
            let batch_len = batch.len();
            self.inner
                .client
                .delete_points(
                    DeletePointsBuilder::new(&collection)
                        .points(PointsIdsList { ids: batch })
                        .wait(true),
                )
                .await
                .map_err(|e| DripstoneError::storage_with_source("failed to delete points", e))?;
            deleted += batch_len;
        }

        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let metric = SimilarityMetric::from(query.metric);

        let inner_results = self
            .inner
            .similarity_search(
                &query.vector,
                query.limit,
                metric,
                query.namespace.as_deref(),
            )
            .await?;

        let mut results = Vec::with_capacity(inner_results.len());
        for (i, r) in inner_results.into_iter().enumerate() {
            let chunk = chunk_envelope::decode_chunk(&r.record, None)?;
            results.push(SearchResult::new(chunk, r.score, i));
        }
        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.inner.list_namespaces().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        self.inner.flush_namespace(namespace).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        self.inner.count_embeddings(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.list_embeddings(namespace).await?;

        let mut chunks = Vec::with_capacity(records.len());
        for record in records {
            chunks.push(chunk_envelope::decode_chunk(&record, Some(namespace))?);
        }

        if let Some(limit) = limit {
            chunks.truncate(limit);
        }

        Ok(chunks)
    }
}
