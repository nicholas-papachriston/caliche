use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::Duration;
use time::OffsetDateTime;

use crate::backend::http::HttpClient;
use crate::embedding::{DEFAULT_NAMESPACE, EmbeddingRecord};
use crate::errors::DripstoneError;
use crate::utils::numeric::checked_f64_to_f32;

const DEFAULT_COLLECTION_PREFIX: &str = "dripstone_";
const DEFAULT_BATCH_SIZE: usize = 1000; // Milvus supports larger batches
const CREATED_KEY: &str = "_dripstone_created_at";
const UPDATED_KEY: &str = "_dripstone_updated_at";

/// Policy for Milvus write flushes after upsert/delete.
///
/// Milvus persists mutations asynchronously unless an explicit `flush` is
/// triggered. Select the policy matching your durability requirements.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum FlushPolicy {
    /// Do not call `flush`; rely on Milvus' own flush cadence.
    Never,
    /// Call `flush`; log at warn and continue if it fails. Default.
    #[default]
    BestEffort,
    /// Call `flush`; propagate any error. Use when durability is required.
    Required,
}

pub struct MilvusEmbeddingBackend {
    http: HttpClient,
    collection_prefix: String,
    vector_dimension: usize,
    index_type: String,
    metric_type: String,
    nlist: u32,
    batch_size: usize,
    flush_policy: FlushPolicy,
}

#[derive(Debug)]
pub struct MilvusBackendOptions {
    pub endpoint: String,
    pub collection_prefix: Option<String>,
    pub vector_dimension: usize,
    pub index_type: Option<String>,
    pub metric_type: Option<String>,
    pub nlist: Option<u32>,
    pub timeout: Option<Duration>,
    pub batch_size: Option<usize>,
    pub retry_config: Option<crate::backend::http::RetryConfig>,
    /// Optional flush policy override. Defaults to `FlushPolicy::BestEffort`.
    pub flush_policy: Option<FlushPolicy>,
}

impl MilvusBackendOptions {
    pub fn new(endpoint: impl Into<String>, vector_dimension: usize) -> Self {
        Self {
            endpoint: endpoint.into(),
            collection_prefix: None,
            vector_dimension,
            index_type: None,
            metric_type: None,
            nlist: None,
            timeout: None,
            batch_size: None,
            retry_config: None,
            flush_policy: None,
        }
    }

    /// Override the flush policy applied after upserts and deletes.
    #[must_use]
    pub const fn with_flush_policy(mut self, policy: FlushPolicy) -> Self {
        self.flush_policy = Some(policy);
        self
    }
}

impl MilvusEmbeddingBackend {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(options: MilvusBackendOptions) -> Result<Self, DripstoneError> {
        if options.vector_dimension == 0 {
            return Err(DripstoneError::Misconfiguration(
                "milvus vector dimension must be greater than zero".to_string(),
            ));
        }

        let mut http_config = crate::backend::http::HttpClientConfig {
            timeout: options.timeout,
            default_headers: None,
            ..Default::default()
        };
        if let Some(retry) = options.retry_config {
            http_config.retry_config = Some(retry);
        }
        let http = HttpClient::with_config(options.endpoint, http_config)?;
        let prefix = options
            .collection_prefix
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_COLLECTION_PREFIX.to_string());
        let batch_size = options.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
        let index_type = options.index_type.unwrap_or_else(|| "IVF_FLAT".to_string());
        let metric_type = options.metric_type.unwrap_or_else(|| "COSINE".to_string());
        let nlist = options.nlist.unwrap_or(128);

        let backend = Self {
            http,
            collection_prefix: prefix,
            vector_dimension: options.vector_dimension,
            index_type,
            metric_type,
            nlist,
            batch_size,
            flush_policy: options.flush_policy.unwrap_or_default(),
        };
        backend.health_check().await?;
        Ok(backend)
    }

    /// Run a Milvus `flush` request subject to the configured policy.
    /// Errors from the flush itself are surfaced only when the policy
    /// requires durability.
    async fn maybe_flush(&self, collection: &str) -> Result<(), DripstoneError> {
        if matches!(self.flush_policy, FlushPolicy::Never) {
            return Ok(());
        }
        let flush_path = "/api/v1/collection/flush";
        let flush_payload = json!({ "collection_name": collection });
        let response = self.http.post_json(flush_path, &flush_payload).await;
        let propagate = matches!(self.flush_policy, FlushPolicy::Required);
        match response {
            Ok(response) => {
                let status = response.status();
                if status.is_success() {
                    return Ok(());
                }
                let body = response.text().await.unwrap_or_default();
                let err = DripstoneError::storage(format!(
                    "milvus flush for collection {collection} failed with status {status}: {body}"
                ));
                if propagate {
                    Err(err)
                } else {
                    tracing::warn!(error = %err, "milvus flush failed; policy=best_effort so it is ignored");
                    Ok(())
                }
            }
            Err(e) => {
                if propagate {
                    Err(e)
                } else {
                    tracing::warn!(error = %e, "milvus flush request failed; policy=best_effort so it is ignored");
                    Ok(())
                }
            }
        }
    }

    fn collection_name(&self, namespace: &str) -> String {
        format!("{}{}", self.collection_prefix, namespace)
    }

    async fn health_check(&self) -> Result<(), DripstoneError> {
        // Metrics `/healthz` listens on Milvus :9091. Compose publishes the
        // gRPC and REST API on :19530, where GET /healthz is HTTP 404.
        // Milvus 2.5 REST liveness is POST /v2/vectordb/collections/list.
        if let Ok(response) = self.http.get("/healthz").await
            && response.status().is_success()
        {
            return Ok(());
        }
        match self
            .http
            .post_json(
                "/v2/vectordb/collections/list",
                &json!({ "dbName": "default" }),
            )
            .await
        {
            Ok(response) if response.status().is_success() => Ok(()),
            Ok(response) => Err(DripstoneError::storage(format!(
                "milvus health check failed with status {} (tried GET /healthz and POST /v2/vectordb/collections/list). GET /healthz is on :9091; REST v2 is on :19530. This client still uses legacy /api/v1 for reads and writes.",
                response.status()
            ))),
            Err(error) => Err(DripstoneError::storage(format!(
                "milvus health check failed: {error}. GET /healthz is on :9091; REST v2 is POST /v2/vectordb/collections/list on :19530. This client still uses legacy /api/v1 for reads and writes."
            ))),
        }
    }

    async fn ensure_collection(&self, namespace: &str) -> Result<(), DripstoneError> {
        let collection = self.collection_name(namespace);

        if self.collection_exists(&collection).await? {
            return Ok(());
        }

        // Create collection schema
        let schema = json!({
            "collection_name": collection,
            "description": format!("Dripstone embedding collection for namespace: {}", namespace),
            "fields": [
                {
                    "field_id": 100,
                    "name": "id",
                    "is_primary": true,
                    "data_type": "VarChar",
                    "max_length": 65535
                },
                {
                    "field_id": 101,
                    "name": "vector",
                    "is_primary": false,
                    "data_type": "FloatVector",
                    "dim": self.vector_dimension
                },
                {
                    "field_id": 102,
                    "name": "metadata",
                    "is_primary": false,
                    "data_type": "JSON"
                },
                {
                    "field_id": 103,
                    "name": CREATED_KEY,
                    "is_primary": false,
                    "data_type": "VarChar",
                    "max_length": 100
                },
                {
                    "field_id": 104,
                    "name": UPDATED_KEY,
                    "is_primary": false,
                    "data_type": "VarChar",
                    "max_length": 100
                }
            ]
        });

        let create_path = "/api/v1/collection";
        let create_response = self.http.post_json(create_path, &schema).await?;
        let create_status = create_response.status();
        let create_status_code = create_status.as_u16();
        if !create_status.is_success() {
            let body = create_response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to create milvus collection {collection}: status {create_status_code} body {body}"
            )));
        }

        // Create index on vector field
        let index_params = json!({
            "collection_name": collection,
            "field_name": "vector",
            "index_name": format!("{}_vector_idx", collection),
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": {
                "nlist": self.nlist
            }
        });

        let index_path = "/api/v1/index";
        let index_response = self.http.post_json(index_path, &index_params).await?;
        let index_status = index_response.status();
        let index_status_code = index_status.as_u16();
        if !index_status.is_success() {
            let body = index_response.text().await.unwrap_or_default();
            // Index creation might fail if it already exists, which is OK
            if !body.contains("already exists") {
                return Err(DripstoneError::storage(format!(
                    "failed to create milvus index: status {index_status_code} body {body}"
                )));
            }
        }

        // Load collection
        let load_path = "/api/v1/collection/load";
        let load_payload = json!({ "collection_name": collection });
        let load_response = self.http.post_json(load_path, &load_payload).await?;
        let load_status = load_response.status();
        let load_status_code = load_status.as_u16();
        if !load_status.is_success() {
            let body = load_response.text().await.unwrap_or_default();
            // Collection might already be loaded
            if !body.contains("already loaded") {
                return Err(DripstoneError::storage(format!(
                    "failed to load milvus collection: status {load_status_code} body {body}"
                )));
            }
        }

        Ok(())
    }

    async fn collection_exists(&self, collection: &str) -> Result<bool, DripstoneError> {
        let payload = json!({ "collection_name": collection });
        let response = self
            .http
            .post_json("/api/v1/collection/existence", &payload)
            .await?;
        if !response.status().is_success() {
            return Ok(false);
        }
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        Ok(body.get("value").and_then(Value::as_bool) == Some(true))
    }

    async fn upsert_entities(
        &self,
        namespace: &str,
        records: Vec<EmbeddingRecord>,
    ) -> Result<(), DripstoneError> {
        if records.is_empty() {
            return Ok(());
        }

        self.ensure_collection(namespace).await?;
        let collection = self.collection_name(namespace);

        let mut ids = Vec::with_capacity(records.len());
        let mut vectors = Vec::with_capacity(records.len());
        let mut metadatas = Vec::with_capacity(records.len());
        let mut created_ats = Vec::with_capacity(records.len());
        let mut updated_ats = Vec::with_capacity(records.len());

        for mut record in records {
            if record.vector.len() != self.vector_dimension {
                return Err(DripstoneError::InvalidVectorLength {
                    expected: self.vector_dimension,
                    actual: record.vector.len(),
                });
            }
            record.touch();

            ids.push(record.id.to_string());
            vectors.push(record.vector);

            let metadata_json = record.metadata.map_or_else(
                || "{}".to_string(),
                |meta| serde_json::to_string(&meta).unwrap_or_else(|_| "{}".to_string()),
            );
            metadatas.push(metadata_json);

            created_ats.push(EmbeddingRecord::format_timestamp(&record.created_at)?);
            updated_ats.push(EmbeddingRecord::format_timestamp(&record.updated_at)?);
        }

        let payload = json!({
            "collection_name": collection,
            "data": [
                ids,
                vectors,
                metadatas,
                created_ats,
                updated_ats
            ]
        });

        let insert_path = "/api/v1/entities";
        let response = self.http.post_json(insert_path, &payload).await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to upsert milvus entities: status {status} body {body}"
            )));
        }

        self.maybe_flush(&collection).await?;

        Ok(())
    }

    async fn query_entity(
        &self,
        namespace: &str,
        id: &str,
    ) -> Result<Option<EmbeddingRecord>, DripstoneError> {
        self.ensure_collection(namespace).await?;
        let collection = self.collection_name(namespace);

        let payload = json!({
            "collection_name": collection,
            "expr": format!("id == \"{}\"", id),
            "output_fields": ["id", "vector", "metadata", CREATED_KEY, UPDATED_KEY],
            "limit": 1
        });

        let query_path = "/api/v1/entities";
        let response = self.http.post_json(query_path, &payload).await?;
        let status = response.status();

        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to query milvus entity: status {status} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let empty_vec: Vec<Value> = Vec::new();
        let entities = body
            .get("data")
            .and_then(Value::as_array)
            .unwrap_or(&empty_vec);

        if entities.is_empty() {
            return Ok(None);
        }

        extract_record(
            namespace,
            entities.first().ok_or_else(|| {
                DripstoneError::storage("Expected at least one entity in response")
            })?,
        )
    }

    async fn delete_entity(&self, namespace: &str, id: &str) -> Result<(), DripstoneError> {
        self.ensure_collection(namespace).await?;
        let collection = self.collection_name(namespace);

        let payload = json!({
            "collection_name": collection,
            "expr": format!("id == \"{}\"", id)
        });

        let delete_path = "/api/v1/entities";
        let response = self.http.post_json(delete_path, &payload).await?;
        let status = response.status();

        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to delete milvus entity: status {status} body {body}"
            )));
        }

        self.maybe_flush(&collection).await?;

        Ok(())
    }

    #[expect(dead_code, reason = "nothing calls it yet")]
    async fn search_collection(
        &self,
        namespace: &str,
        vector: &[f32],
        limit: usize,
        filter: Option<Value>,
    ) -> Result<Value, DripstoneError> {
        self.ensure_collection(namespace).await?;
        let collection = self.collection_name(namespace);

        let expr = if let Some(filter_obj) = filter {
            build_filter_expr(&filter_obj)?
        } else {
            String::new()
        };

        let payload = json!({
            "collection_name": collection,
            "vector": vector,
            "limit": limit,
            "output_fields": ["id", "vector", "metadata", CREATED_KEY, UPDATED_KEY],
            "expr": expr,
            "metric_type": self.metric_type
        });

        let search_path = "/api/v1/entities/search";
        let response = self.http.post_json(search_path, &payload).await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to search milvus collection: status {status} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        Ok(body)
    }

    #[expect(dead_code, reason = "nothing calls it yet")]
    async fn list_collections(&self) -> Result<Vec<String>, DripstoneError> {
        let response = self.http.get("/api/v1/collections").await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to list milvus collections: status {status} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let empty_vec: Vec<Value> = Vec::new();
        let collections = body
            .get("collection_names")
            .and_then(Value::as_array)
            .unwrap_or(&empty_vec);

        let mut namespaces = Vec::new();
        for collection in collections {
            if let Some(name) = collection.as_str()
                && let Some(stripped) = name.strip_prefix(&self.collection_prefix)
            {
                namespaces.push(stripped.to_string());
            }
        }
        Ok(namespaces)
    }

    #[expect(dead_code, reason = "nothing calls it yet")]
    async fn count_collection(&self, namespace: &str) -> Result<usize, DripstoneError> {
        self.ensure_collection(namespace).await?;
        let collection = self.collection_name(namespace);

        let payload = json!({
            "collection_name": collection,
            "expr": ""
        });

        let query_path = "/api/v1/entities";
        let response = self.http.post_json(query_path, &payload).await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to count milvus collection: status {status} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let count = body
            .get("num_entities")
            .and_then(Value::as_u64)
            .unwrap_or(0);

        usize::try_from(count)
            .map_err(|_| DripstoneError::storage("milvus count exceeds addressable size"))
    }

    async fn list_collection_entities(
        &self,
        namespace: &str,
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        self.ensure_collection(namespace).await?;
        let collection = self.collection_name(namespace);

        let payload = json!({
            "collection_name": collection,
            "expr": "",
            "output_fields": ["id", "vector", "metadata", CREATED_KEY, UPDATED_KEY],
            "limit": 10000
        });

        let query_path = "/api/v1/entities";
        let response = self.http.post_json(query_path, &payload).await?;
        let status = response.status();

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "failed to list milvus entities: status {status} body {body}"
            )));
        }

        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let empty_vec: Vec<Value> = Vec::new();
        let entities = body
            .get("data")
            .and_then(Value::as_array)
            .unwrap_or(&empty_vec);

        let mut records = Vec::new();
        for entity in entities {
            if let Ok(Some(record)) = extract_record(namespace, entity) {
                records.push(record);
            }
        }
        Ok(records)
    }
}

fn extract_record(
    namespace: &str,
    entity: &Value,
) -> Result<Option<EmbeddingRecord>, DripstoneError> {
    let id = entity
        .get("id")
        .or_else(|| entity.get("data").and_then(|d| d.get("id")))
        .and_then(Value::as_str)
        .ok_or_else(|| DripstoneError::storage("milvus entity missing id"))?;

    let vector = match entity
        .get("vector")
        .or_else(|| entity.get("data").and_then(|d| d.get("vector")))
    {
        Some(v) => parse_vector_value(v)?,
        None => return Err(DripstoneError::storage("milvus entity missing vector")),
    };

    let metadata_str = entity
        .get("metadata")
        .or_else(|| entity.get("data").and_then(|d| d.get("metadata")))
        .and_then(Value::as_str);
    let metadata = metadata_str
        .and_then(|s| serde_json::from_str::<Value>(s).ok())
        .filter(|v| {
            !(v.is_null()
                || (v.is_object() && v.as_object().is_some_and(serde_json::Map::is_empty)))
        });

    let created_at_str = entity
        .get(CREATED_KEY)
        .or_else(|| entity.get("data").and_then(|d| d.get(CREATED_KEY)))
        .and_then(Value::as_str);
    let updated_at_str = entity
        .get(UPDATED_KEY)
        .or_else(|| entity.get("data").and_then(|d| d.get(UPDATED_KEY)))
        .and_then(Value::as_str);

    let created_at = created_at_str
        .map(EmbeddingRecord::parse_timestamp)
        .transpose()?
        .unwrap_or_else(OffsetDateTime::now_utc);
    let updated_at = updated_at_str
        .map(EmbeddingRecord::parse_timestamp)
        .transpose()?
        .unwrap_or(created_at);

    Ok(Some(EmbeddingRecord {
        id: std::sync::Arc::from(id),
        vector: vector.into(),
        namespace: Some(std::sync::Arc::from(namespace)),
        metadata,
        sparse_vector: None,
        multi_vectors: None,
        created_at,
        updated_at,
    }))
}

fn parse_vector_value(value: &Value) -> Result<Vec<f32>, DripstoneError> {
    if let Some(array) = value.as_array() {
        let mut vector = Vec::with_capacity(array.len());
        for item in array {
            let number = item
                .as_f64()
                .ok_or_else(|| DripstoneError::storage("vector element was not numeric"))?;
            vector.push(checked_f64_to_f32(number).ok_or_else(|| {
                DripstoneError::storage("milvus vector element exceeds finite f32 range")
            })?);
        }
        return Ok(vector);
    }
    Err(DripstoneError::storage(
        "expected array of vector components from milvus",
    ))
}

fn build_filter_expr(filter: &Value) -> Result<String, DripstoneError> {
    // Build Milvus filter expression
    let Some(obj) = filter.as_object() else {
        return Err(DripstoneError::storage("invalid filter format for milvus"));
    };
    let mut exprs = Vec::new();
    for (key, value) in obj {
        let value_str = value.to_string().trim_matches('"').to_string();
        exprs.push(format!(
            "JSON_EXTRACT(metadata, \"$.{key}\") == \"{value_str}\""
        ));
    }
    Ok(exprs.join(" && "))
}

// =============================================================================
// Backend trait implementation
// =============================================================================

use crate::backend::chunk_envelope;
use crate::core::{Capabilities, Chunk, Metric, SearchQuery, SearchResult};
use crate::traits::Backend;
use std::fmt;

impl fmt::Debug for MilvusEmbeddingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MilvusBackend")
            .field("collection_prefix", &self.collection_prefix)
            .field("vector_dimension", &self.vector_dimension)
            .finish_non_exhaustive()
    }
}

/// Milvus backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct MilvusBackend {
    inner: MilvusEmbeddingBackend,
}

impl MilvusBackend {
    /// Create a new Milvus backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_options(options: MilvusBackendOptions) -> Result<Self, DripstoneError> {
        let inner = MilvusEmbeddingBackend::new(options).await?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl Backend for MilvusBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: Some(self.inner.vector_dimension),
            max_batch: Some(self.inner.batch_size),
            supports_streaming: false,
        }
    }

    async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        if chunks.is_empty() {
            return Ok(());
        }

        // Group chunks by namespace
        let mut namespace_chunks: HashMap<String, Vec<&Chunk>> = HashMap::new();
        for chunk in chunks {
            let namespace = chunk
                .namespace
                .as_deref()
                .unwrap_or(DEFAULT_NAMESPACE)
                .to_string();
            namespace_chunks.entry(namespace).or_default().push(chunk);
        }

        // Process each namespace
        for (namespace, chunk_group) in namespace_chunks {
            let mut records = Vec::with_capacity(chunk_group.len());
            for chunk in chunk_group {
                records.push(chunk_envelope::embedding_record_from_chunk(chunk));
            }
            self.inner.upsert_entities(&namespace, records).await?;
        }

        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let record = self.inner.query_entity(namespace, id).await?;

        match record {
            Some(r) => Ok(Some(chunk_envelope::decode_chunk(&r, Some(namespace))?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.inner.delete_entity(namespace, id).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        // Query for all chunks with this doc_id, then delete them
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.inner.collection_name(namespace);
        let expr = format!("metadata['doc_id'] == \"{doc_id}\"");

        let query_payload = json!({
            "collection_name": collection,
            "expr": expr,
            "output_fields": ["id"],
            "limit": 10000
        });

        let response = self
            .inner
            .http
            .post_json("/api/v1/entities", &query_payload)
            .await?;
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let empty: Vec<Value> = Vec::new();
        let entities = body.get("data").and_then(Value::as_array).unwrap_or(&empty);

        if entities.is_empty() {
            return Ok(0);
        }

        let ids: Vec<String> = entities
            .iter()
            .filter_map(|e| {
                e.get(0)
                    .and_then(Value::as_str)
                    .map(std::string::ToString::to_string)
            })
            .collect();

        for id in &ids {
            self.inner.delete_entity(namespace, id).await?;
        }

        Ok(ids.len())
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let namespace = query.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.inner.collection_name(namespace);

        let metric = match query.metric {
            Metric::Cosine => "COSINE",
            Metric::Euclidean => "L2",
            Metric::DotProduct => "IP",
            Metric::Manhattan => "L1",
        };

        let mut search_payload = json!({
            "collection_name": collection,
            "vector": query.vector,
            "limit": query.limit,
            "metric_type": metric,
            "output_fields": ["id", "vector", "metadata", CREATED_KEY, UPDATED_KEY],
        });

        if let Some(ref filter) = query.filter {
            let expr = crate::search::filter_to_milvus::filter_to_milvus(filter)?;
            search_payload["expr"] = json!(expr);
        }

        let response = self
            .inner
            .http
            .post_json("/api/v1/entities/search", &search_payload)
            .await?;
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let results_array = body
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| DripstoneError::storage("milvus search response missing data"))?;

        let mut results = Vec::new();
        for (i, result_obj) in results_array.iter().enumerate() {
            let distance = result_obj
                .get("distance")
                .and_then(Value::as_f64)
                .map(|distance| {
                    checked_f64_to_f32(distance).ok_or_else(|| {
                        DripstoneError::storage("milvus distance exceeds finite f32 range")
                    })
                })
                .transpose()?
                .unwrap_or(0.0);
            // Convert distance to similarity score (lower distance = higher similarity)
            let score = 1.0 / (1.0 + distance);

            let entity = result_obj
                .get("entity")
                .ok_or_else(|| DripstoneError::storage("milvus result missing entity"))?;

            let record = extract_record(namespace, entity)?
                .ok_or_else(|| DripstoneError::storage("milvus entity extraction failed"))?;
            let chunk = chunk_envelope::decode_chunk(&record, Some(namespace))?;
            results.push(SearchResult::new(chunk, score, i));
        }

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        // Milvus doesn't have a direct way to list collections, return empty for now
        // Could be enhanced to query collections
        Ok(Vec::new())
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let collection = self.inner.collection_name(namespace);
        let path = format!("/api/v1/collection/delete?collection_name={collection}");
        let response = self.inner.http.delete(path.as_str()).await?;
        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::NOT_FOUND {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "milvus delete namespace failed: status {status} body {body}"
            )));
        }
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.inner.collection_name(namespace);
        let payload = json!({
            "collection_name": collection
        });
        let response = self
            .inner
            .http
            .post_json("/api/v1/collection/stats", &payload)
            .await?;
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let row_count = body
            .get("data")
            .and_then(Value::as_object)
            .and_then(|obj| obj.get("row_count"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        usize::try_from(row_count)
            .map_err(|_| DripstoneError::storage("milvus row count exceeds addressable size"))
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.list_collection_entities(namespace).await?;

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
