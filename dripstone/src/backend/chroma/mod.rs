use async_trait::async_trait;
use serde_json::{Map as JsonMap, Value, json};
use std::collections::HashMap;
use std::time::Duration;
use time::OffsetDateTime;

use crate::backend::http::HttpClient;
use crate::embedding::{DEFAULT_NAMESPACE, EmbeddingRecord};
use crate::errors::DripstoneError;
use crate::search::filter_to_chroma::filter_to_chroma;
use crate::utils::numeric::checked_f64_to_f32;
use reqwest::header::{HeaderMap, HeaderValue};

const DEFAULT_COLLECTION_PREFIX: &str = "dripstone_";
const DEFAULT_BATCH_SIZE: usize = 500;
const CREATED_KEY: &str = "_dripstone_created_at";
const UPDATED_KEY: &str = "_dripstone_updated_at";
const CHROMA_TENANT: &str = "default_tenant";
const CHROMA_DATABASE: &str = "default_database";

/// Chroma HTTP API generation. Compose `chromadb/chroma:1.0.15` answers
/// `/api/v2/heartbeat` and returns HTTP 410 for `/api/v1/*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChromaApi {
    V1,
    V2,
}

pub struct ChromaEmbeddingBackend {
    http: HttpClient,
    collection_prefix: String,
    expected_dimension: Option<usize>,
    batch_size: usize,
    api: ChromaApi,
    /// v2 record routes key by collection UUID. v1 uses the collection name.
    collection_ids: std::sync::Mutex<HashMap<String, String>>,
}

fn chroma_collection_root(api: ChromaApi) -> String {
    match api {
        ChromaApi::V1 => "/api/v1/collections".to_string(),
        ChromaApi::V2 => {
            format!("/api/v2/tenants/{CHROMA_TENANT}/databases/{CHROMA_DATABASE}/collections")
        }
    }
}

fn chroma_records_path(api: ChromaApi, collection_key: &str, action: &str) -> String {
    format!("{}/{collection_key}/{action}", chroma_collection_root(api))
}

#[derive(Debug)]
pub struct ChromaBackendOptions {
    pub base_url: String,
    pub collection_prefix: Option<String>,
    pub expected_dimension: Option<usize>,
    pub timeout: Option<Duration>,
    pub batch_size: Option<usize>,
    pub api_key: Option<String>,
    pub retry_config: Option<crate::backend::http::RetryConfig>,
}

impl ChromaBackendOptions {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            collection_prefix: None,
            expected_dimension: None,
            timeout: None,
            batch_size: None,
            api_key: None,
            retry_config: None,
        }
    }
}

impl ChromaEmbeddingBackend {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(options: ChromaBackendOptions) -> Result<Self, DripstoneError> {
        let headers = options
            .api_key
            .as_ref()
            .map(|key| build_chroma_headers(key))
            .transpose()?;

        let mut http_config = crate::backend::http::HttpClientConfig {
            timeout: options.timeout,
            default_headers: headers,
            ..Default::default()
        };
        if let Some(retry) = options.retry_config {
            http_config.retry_config = Some(retry);
        }
        let http = HttpClient::with_config(options.base_url, http_config)?;
        let api = detect_chroma_api(&http).await?;
        let prefix = options
            .collection_prefix
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_COLLECTION_PREFIX.to_string());
        let batch_size = options.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
        Ok(Self {
            http,
            collection_prefix: prefix,
            expected_dimension: options.expected_dimension,
            batch_size,
            api,
            collection_ids: std::sync::Mutex::new(HashMap::new()),
        })
    }

    fn collection_name(&self, namespace: &str) -> String {
        format!("{}{}", self.collection_prefix, namespace)
    }

    fn cached_collection_id(&self, name: &str) -> Option<String> {
        let guard = self
            .collection_ids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.get(name).cloned()
    }

    fn remember_collection_id(&self, name: &str, id: &str) {
        let mut guard = self
            .collection_ids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.insert(name.to_string(), id.to_string());
    }

    fn forget_collection_id(&self, name: &str) {
        let mut guard = self
            .collection_ids
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.remove(name);
    }

    async fn ensure_collection(&self, namespace: &str) -> Result<String, DripstoneError> {
        let name = self.collection_name(namespace);
        match self.api {
            ChromaApi::V1 => {
                self.ensure_v1_collection(&name).await?;
                Ok(name)
            }
            ChromaApi::V2 => self.ensure_v2_collection(&name).await,
        }
    }

    async fn ensure_v1_collection(&self, collection: &str) -> Result<(), DripstoneError> {
        let get_path = format!("{}/{collection}", chroma_collection_root(ChromaApi::V1));
        match self.http.get(get_path.as_str()).await {
            Ok(response) => {
                if response.status() == reqwest::StatusCode::OK {
                    return Ok(());
                }
                if response.status() != reqwest::StatusCode::NOT_FOUND {
                    return Err(DripstoneError::storage(format!(
                        "failed to inspect chroma collection {collection}: status {}",
                        response.status()
                    )));
                }
            }
            Err(error) if error.is_not_found() => {}
            Err(error) => return Err(error),
        }
        let payload = json!({ "name": collection });
        let create_path = chroma_collection_root(ChromaApi::V1);
        let create_response = self.http.post_json(create_path.as_str(), &payload).await?;
        if !create_response.status().is_success() {
            return Err(DripstoneError::storage(format!(
                "failed to create chroma collection {collection}: status {}",
                create_response.status()
            )));
        }
        Ok(())
    }

    async fn ensure_v2_collection(&self, name: &str) -> Result<String, DripstoneError> {
        if let Some(id) = self.cached_collection_id(name) {
            return Ok(id);
        }
        if let Some(id) = self.lookup_v2_collection(name).await? {
            self.remember_collection_id(name, &id);
            return Ok(id);
        }
        let id = self.create_v2_collection(name).await?;
        self.remember_collection_id(name, &id);
        Ok(id)
    }

    async fn lookup_v2_collection(&self, name: &str) -> Result<Option<String>, DripstoneError> {
        let path = format!("{}/{name}", chroma_collection_root(ChromaApi::V2));
        match self.http.get(path.as_str()).await {
            Ok(response) => {
                if response.status() == reqwest::StatusCode::NOT_FOUND {
                    return Ok(None);
                }
                if !response.status().is_success() {
                    return Err(DripstoneError::storage(format!(
                        "failed to inspect chroma collection {name}: status {}",
                        response.status()
                    )));
                }
                let body: Value = response.json().await.map_err(DripstoneError::from)?;
                Ok(collection_id_from_body(&body))
            }
            Err(error) if error.is_not_found() => Ok(None),
            Err(error) => Err(error),
        }
    }

    async fn create_v2_collection(&self, name: &str) -> Result<String, DripstoneError> {
        let path = chroma_collection_root(ChromaApi::V2);
        let payload = json!({
            "name": name,
            "get_or_create": true,
        });
        let response = match self.http.post_json(path.as_str(), &payload).await {
            Ok(response) => response,
            Err(error) => {
                if let Some(id) = self.lookup_v2_collection(name).await? {
                    return Ok(id);
                }
                return Err(error);
            }
        };
        if !response.status().is_success() {
            if let Some(id) = self.lookup_v2_collection(name).await? {
                return Ok(id);
            }
            return Err(DripstoneError::storage(format!(
                "failed to create chroma collection {name}: status {}",
                response.status()
            )));
        }
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        collection_id_from_body(&body).ok_or_else(|| {
            DripstoneError::storage(format!(
                "chroma create collection {name} response missing id"
            ))
        })
    }

    async fn upsert_records(
        &self,
        namespace: &str,
        records: Vec<EmbeddingRecord>,
    ) -> Result<(), DripstoneError> {
        if records.is_empty() {
            return Ok(());
        }

        let collection = self.ensure_collection(namespace).await?;
        let path = chroma_records_path(self.api, &collection, "add");

        let mut ids = Vec::with_capacity(records.len());
        let mut embeddings = Vec::with_capacity(records.len());
        let mut metadatas = Vec::with_capacity(records.len());
        for mut record in records {
            if let Some(dimension) = self.expected_dimension
                && record.vector.len() != dimension
            {
                return Err(DripstoneError::InvalidVectorLength {
                    expected: dimension,
                    actual: record.vector.len(),
                });
            }
            record.touch();
            ids.push(record.id);
            embeddings.push(record.vector);
            metadatas.push(coerce_chroma_metadata(encode_metadata(
                record.metadata.take(),
                record.created_at,
                record.updated_at,
            )?));
        }

        let payload = json!({
            "ids": ids,
            "embeddings": embeddings,
            "metadatas": metadatas,
        });

        let response = self.http.post_json(path.as_str(), &payload).await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_else(|_| String::new());
            return Err(DripstoneError::storage(format!(
                "failed to upsert chroma embeddings: status {status} body {body}"
            )));
        }
        Ok(())
    }

    async fn get_records(
        &self,
        namespace: &str,
        ids: Option<Vec<String>>,
        filter: Option<Value>,
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        let collection = self.ensure_collection(namespace).await?;
        let path = chroma_records_path(self.api, &collection, "get");
        let payload = json!({
            "ids": ids,
            "where": filter,
            "include": ["embeddings", "metadatas"],
        });
        let response = self.http.post_json(path.as_str(), &payload).await?;
        let status = response.status();
        if !status.is_success() {
            if status == reqwest::StatusCode::NOT_FOUND {
                return Ok(Vec::new());
            }
            let body = response.text().await.unwrap_or_else(|_| String::new());
            return Err(DripstoneError::storage(format!(
                "failed to fetch chroma records: status {status} body {body}"
            )));
        }
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        extract_records(namespace, &body)
    }

    async fn delete_records(
        &self,
        namespace: &str,
        ids: Vec<String>,
    ) -> Result<(), DripstoneError> {
        let collection = self.ensure_collection(namespace).await?;
        let path = chroma_records_path(self.api, &collection, "delete");
        let payload = json!({ "ids": ids });
        let response = self.http.post_json(path.as_str(), &payload).await?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_else(|_| String::new());
            return Err(DripstoneError::storage(format!(
                "failed to delete chroma records: status {status} body {body}"
            )));
        }
        Ok(())
    }

    async fn query_collection(
        &self,
        namespace: &str,
        vector: &[f32],
        limit: usize,
        filter: Option<Value>,
    ) -> Result<Value, DripstoneError> {
        let collection = self.ensure_collection(namespace).await?;
        let path = chroma_records_path(self.api, &collection, "query");
        let include = match self.api {
            ChromaApi::V1 => json!(["embeddings", "metadatas", "distances", "ids"]),
            // v2 rejects `ids` in `include`; ids are always returned.
            ChromaApi::V2 => json!(["embeddings", "metadatas", "distances"]),
        };
        let payload = json!({
            "query_embeddings": [vector],
            "n_results": limit,
            "where": filter,
            "include": include,
        });
        let response = self.http.post_json(path.as_str(), &payload).await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_else(|_| String::new());
            return Err(DripstoneError::storage(format!(
                "failed to query chroma collection: status {status} body {body}"
            )));
        }
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        Ok(body)
    }
}

fn encode_metadata(
    metadata: Option<Value>,
    created_at: OffsetDateTime,
    updated_at: OffsetDateTime,
) -> Result<Value, DripstoneError> {
    let mut map = match metadata {
        Some(Value::Object(map)) => map,
        Some(other) => {
            let mut map = JsonMap::new();
            map.insert("value".to_string(), other);
            map
        }
        None => JsonMap::new(),
    };

    let created = EmbeddingRecord::format_timestamp(&created_at)?;
    let updated = EmbeddingRecord::format_timestamp(&updated_at)?;
    map.insert(CREATED_KEY.to_string(), Value::from(created));
    map.insert(UPDATED_KEY.to_string(), Value::from(updated));

    Ok(Value::Object(map))
}

fn decode_metadata(
    value: Value,
) -> Result<(Option<Value>, OffsetDateTime, OffsetDateTime), DripstoneError> {
    match value {
        Value::Object(mut map) => {
            let created = map
                .remove(CREATED_KEY)
                .map(parse_timestamp_node)
                .transpose()?
                .unwrap_or_else(OffsetDateTime::now_utc);
            let updated = map
                .remove(UPDATED_KEY)
                .map(parse_timestamp_node)
                .transpose()?
                .unwrap_or(created);

            if let Some(Value::String(raw)) = map.get("original_metadata").cloned()
                && let Ok(parsed) = serde_json::from_str::<Value>(&raw)
            {
                map.insert("original_metadata".to_string(), parsed);
            }
            let metadata = if map.is_empty() {
                None
            } else {
                Some(Value::Object(map))
            };
            Ok((metadata, created, updated))
        }
        other => Ok((
            Some(other),
            OffsetDateTime::now_utc(),
            OffsetDateTime::now_utc(),
        )),
    }
}

fn parse_timestamp_node(node: Value) -> Result<OffsetDateTime, DripstoneError> {
    match node {
        Value::String(string) => EmbeddingRecord::parse_timestamp(string.as_str()),
        other => EmbeddingRecord::parse_timestamp(other.to_string().as_str()),
    }
}

fn stringify_identifier(value: &Value) -> Result<String, DripstoneError> {
    if let Some(string) = value.as_str() {
        return Ok(string.to_string());
    }
    if let Some(int) = value.as_i64() {
        return Ok(int.to_string());
    }
    if let Some(uint) = value.as_u64() {
        return Ok(uint.to_string());
    }
    if let Some(float) = value.as_f64() {
        return Ok(float.to_string());
    }
    Err(DripstoneError::storage(
        "chroma identifier must be string or numeric",
    ))
}

fn extract_records(namespace: &str, body: &Value) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
    let empty = Vec::new();
    let ids = body.get("ids").and_then(Value::as_array).unwrap_or(&empty);
    let embeddings = body
        .get("embeddings")
        .and_then(Value::as_array)
        .unwrap_or(&empty);
    let metadatas = body
        .get("metadatas")
        .and_then(Value::as_array)
        .unwrap_or(&empty);

    let mut records = Vec::new();
    for (index, id_value) in ids.iter().enumerate() {
        let id = stringify_identifier(id_value)?;

        let vector = embeddings
            .get(index)
            .map(parse_vector_value)
            .transpose()?
            .unwrap_or_default();

        let metadata_value = metadatas.get(index).cloned().unwrap_or(Value::Null);
        let (metadata, created_at, updated_at) = decode_metadata(metadata_value)?;

        records.push(EmbeddingRecord {
            id: std::sync::Arc::from(id),
            vector: vector.into(),
            namespace: Some(std::sync::Arc::from(namespace)),
            metadata,
            sparse_vector: None,
            multi_vectors: None,
            created_at,
            updated_at,
        });
    }
    Ok(records)
}

fn parse_vector_value(value: &Value) -> Result<Vec<f32>, DripstoneError> {
    if let Some(array) = value.as_array() {
        let mut vector = Vec::with_capacity(array.len());
        for item in array {
            let number = item
                .as_f64()
                .ok_or_else(|| DripstoneError::storage("vector element was not numeric"))?;
            vector.push(checked_f64_to_f32(number).ok_or_else(|| {
                DripstoneError::storage("chroma vector element exceeds finite f32 range")
            })?);
        }
        return Ok(vector);
    }
    Err(DripstoneError::storage(
        "expected list of vector components from chroma",
    ))
}

fn build_chroma_headers(api_key: &str) -> Result<HeaderMap, DripstoneError> {
    let trimmed = api_key.trim();
    if trimmed.is_empty() {
        return Err(DripstoneError::Misconfiguration(
            "chroma api key cannot be empty".to_string(),
        ));
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        "X-Chroma-Token",
        HeaderValue::from_str(trimmed).map_err(|_| {
            DripstoneError::Misconfiguration("invalid characters in chroma api key".to_string())
        })?,
    );
    Ok(headers)
}

// =============================================================================
// Backend trait implementation
// =============================================================================

use crate::backend::chunk_envelope::{self, decode_chunk_from_envelope};
use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::traits::Backend;
use std::fmt;

impl fmt::Debug for ChromaEmbeddingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ChromaBackend")
            .field("collection_prefix", &self.collection_prefix)
            .finish_non_exhaustive()
    }
}

/// Chroma backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct ChromaBackend {
    inner: ChromaEmbeddingBackend,
}

impl ChromaBackend {
    /// Create a new Chroma backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_options(options: ChromaBackendOptions) -> Result<Self, DripstoneError> {
        let inner = ChromaEmbeddingBackend::new(options).await?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl Backend for ChromaBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: false,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: self.inner.expected_dimension,
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
            self.inner.upsert_records(&namespace, records).await?;
        }

        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self
            .inner
            .get_records(namespace, Some(vec![id.to_string()]), None)
            .await?;

        match records.first() {
            Some(r) => Ok(Some(chunk_envelope::decode_chunk(r, Some(namespace))?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.inner
            .delete_records(namespace, vec![id.to_string()])
            .await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        // Chroma supports filtering in get_records, so we can query then delete
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let filter = json!({
            "doc_id": { "$eq": doc_id }
        });

        let records = self
            .inner
            .get_records(namespace, None, Some(filter))
            .await?;
        if records.is_empty() {
            return Ok(0);
        }

        let ids: Vec<String> = records.iter().map(|r| r.id.to_string()).collect();
        self.inner.delete_records(namespace, ids.clone()).await?;
        Ok(ids.len())
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let namespace = query.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);

        let filter = query.filter.as_ref().map(filter_to_chroma).transpose()?;

        let query_result = self
            .inner
            .query_collection(namespace, &query.vector, query.limit, filter)
            .await?;

        let empty: Vec<Value> = Vec::new();
        let distances = query_result
            .get("distances")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(Value::as_array)
            .unwrap_or(&empty);

        let ids = query_result
            .get("ids")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(Value::as_array)
            .unwrap_or(&empty);

        let embeddings = query_result
            .get("embeddings")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(Value::as_array)
            .unwrap_or(&empty);

        let metadatas = query_result
            .get("metadatas")
            .and_then(Value::as_array)
            .and_then(|arr| arr.first())
            .and_then(Value::as_array)
            .unwrap_or(&empty);

        let mut results = Vec::new();
        for (i, id_value) in ids.iter().enumerate() {
            let id = stringify_identifier(id_value)?;
            let distance = distances
                .get(i)
                .and_then(Value::as_f64)
                .map(|distance| {
                    checked_f64_to_f32(distance).ok_or_else(|| {
                        DripstoneError::storage("chroma distance exceeds finite f32 range")
                    })
                })
                .transpose()?
                .unwrap_or(0.0);
            // Chroma returns distances (lower is better), convert to similarity score
            let score = 1.0 - distance.min(1.0);

            let vector = embeddings
                .get(i)
                .map(parse_vector_value)
                .transpose()?
                .unwrap_or_default();

            let metadata_value = metadatas.get(i).cloned().unwrap_or(Value::Null);
            let (metadata, _, _) = decode_metadata(metadata_value)?;
            let envelope = metadata.unwrap_or_else(|| serde_json::json!({}));
            let chunk = decode_chunk_from_envelope(
                id,
                vector.into(),
                &envelope,
                Some(namespace.to_string()),
            );
            results.push(SearchResult::new(chunk, score, i));
        }

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let path = chroma_collection_root(self.inner.api);
        let response = self.inner.http.get(path.as_str()).await?;
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        let collections = chroma_collection_list(&body);

        let mut namespaces = Vec::new();
        let prefix_len = self.inner.collection_prefix.len();
        for collection in collections {
            if let Some(name) = collection.get("name").and_then(Value::as_str)
                && name.starts_with(&self.inner.collection_prefix)
            {
                let namespace = name[prefix_len..].to_string();
                if !namespace.is_empty() {
                    namespaces.push(namespace);
                }
            }
        }

        Ok(namespaces)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let collection = self.inner.collection_name(namespace);
        let path = format!("{}/{collection}", chroma_collection_root(self.inner.api));
        match self.inner.http.delete(path.as_str()).await {
            Ok(response) => {
                let status = response.status();
                if !status.is_success() && status != reqwest::StatusCode::NOT_FOUND {
                    let body = response.text().await.unwrap_or_default();
                    return Err(DripstoneError::storage(format!(
                        "chroma delete namespace failed: status {status} body {body}"
                    )));
                }
            }
            Err(error) if error.is_not_found() => {}
            Err(error) => return Err(error),
        }
        self.inner.forget_collection_id(&collection);
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.get_records(namespace, None, None).await?;
        Ok(records.len())
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.get_records(namespace, None, None).await?;

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

async fn detect_chroma_api(http: &HttpClient) -> Result<ChromaApi, DripstoneError> {
    let v2_error = match http.get("/api/v2/heartbeat").await {
        Ok(response) if response.status().is_success() => return Ok(ChromaApi::V2),
        Ok(response) => DripstoneError::storage(format!(
            "chroma v2 heartbeat failed with status {}",
            response.status()
        )),
        Err(error) => error,
    };
    if !v2_error.is_not_found() {
        if let Ok(response) = http.get("/api/v1/heartbeat").await
            && response.status().is_success()
        {
            return Ok(ChromaApi::V1);
        }
        return Err(v2_error);
    }
    match http.get("/api/v1/heartbeat").await {
        Ok(response) if response.status().is_success() => Ok(ChromaApi::V1),
        Ok(response) => Err(DripstoneError::storage(format!(
            "chroma health check failed with status {}",
            response.status()
        ))),
        Err(error) => Err(error),
    }
}

fn collection_id_from_body(body: &Value) -> Option<String> {
    body.get("id")
        .or_else(|| body.get("data").and_then(|data| data.get("id")))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn chroma_collection_list(body: &Value) -> Vec<&Value> {
    if let Some(list) = body.as_array() {
        return list.iter().collect();
    }
    body.get("data")
        .and_then(Value::as_array)
        .map(|list| list.iter().collect())
        .unwrap_or_default()
}

/// Chroma metadata values must be bool, number, or string. Nested envelope
/// objects (notably `original_metadata`) are stored as JSON strings.
fn coerce_chroma_metadata(value: Value) -> Value {
    let Value::Object(map) = value else {
        return value;
    };
    let mut out = JsonMap::new();
    for (key, item) in map {
        match item {
            Value::Null => {}
            Value::Bool(flag) => {
                out.insert(key, Value::Bool(flag));
            }
            Value::Number(number) => {
                out.insert(key, Value::Number(number));
            }
            Value::String(text) => {
                out.insert(key, Value::String(text));
            }
            other => {
                out.insert(key, Value::String(other.to_string()));
            }
        }
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests {
    use super::{
        ChromaApi, chroma_collection_list, chroma_records_path, coerce_chroma_metadata,
        decode_metadata,
    };
    use serde_json::json;

    #[test]
    fn v2_record_path_uses_collection_id() {
        let path = chroma_records_path(ChromaApi::V2, "abc-id", "add");
        assert_eq!(
            path,
            "/api/v2/tenants/default_tenant/databases/default_database/collections/abc-id/add"
        );
    }

    #[test]
    fn v1_record_path_keeps_collection_name() {
        let path = chroma_records_path(ChromaApi::V1, "dripstone_ns", "query");
        assert_eq!(path, "/api/v1/collections/dripstone_ns/query");
    }

    #[test]
    fn v2_list_body_is_a_bare_array() {
        let body = json!([{"name": "dripstone_ns", "id": "abc"}]);
        let list = chroma_collection_list(&body);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["name"], "dripstone_ns");
    }

    #[test]
    fn nested_envelope_metadata_round_trips_as_string() {
        let coerced = coerce_chroma_metadata(json!({
            "doc_id": "item-3",
            "original_metadata": {"segment": "alpha"},
            "chunk_index": 0
        }));
        assert_eq!(coerced["doc_id"], "item-3");
        assert!(coerced["original_metadata"].is_string());
        let (metadata, _, _) = decode_metadata(coerced).expect("decode");
        let metadata = metadata.expect("metadata");
        assert_eq!(metadata["original_metadata"]["segment"], "alpha");
    }

    #[test]
    fn decode_metadata_keeps_timestamps() {
        let coerced = coerce_chroma_metadata(json!({
            "_dripstone_created_at": "2020-01-01T00:00:00Z",
            "_dripstone_updated_at": "2020-01-01T00:00:00Z",
            "doc_id": "a"
        }));
        let (metadata, _, _) = decode_metadata(coerced).expect("decode");
        assert_eq!(metadata.expect("meta")["doc_id"], "a");
    }
}
