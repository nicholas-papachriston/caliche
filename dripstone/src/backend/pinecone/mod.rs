use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Map as JsonMap, Value, json};
use std::collections::HashMap;
use std::time::Duration;
use time::OffsetDateTime;

use crate::backend::http::HttpClient;
use crate::embedding::{DEFAULT_NAMESPACE, EmbeddingRecord};
use crate::errors::DripstoneError;
use crate::utils::numeric::checked_f64_to_f32;

const DEFAULT_BATCH_SIZE: usize = 500;
const CREATED_KEY: &str = "_dripstone_created_at";
const UPDATED_KEY: &str = "_dripstone_updated_at";

pub struct PineconeEmbeddingBackend {
    http: HttpClient,
    dimension: usize,
    batch_size: usize,
}

#[derive(Debug)]
pub struct PineconeBackendOptions {
    pub base_url: String,
    pub api_key: String,
    pub dimension: usize,
    pub timeout: Option<Duration>,
    pub batch_size: Option<usize>,
    pub retry_config: Option<crate::backend::http::RetryConfig>,
}

impl PineconeEmbeddingBackend {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(options: PineconeBackendOptions) -> Result<Self, DripstoneError> {
        if options.dimension == 0 {
            return Err(DripstoneError::Misconfiguration(
                "pinecone vector dimension must be greater than zero".to_string(),
            ));
        }

        if options.api_key.trim().is_empty() {
            return Err(DripstoneError::Misconfiguration(
                "pinecone api key cannot be empty".to_string(),
            ));
        }

        let mut headers = HeaderMap::new();
        headers.insert(
            "Api-Key",
            HeaderValue::from_str(options.api_key.trim()).map_err(|_| {
                DripstoneError::Misconfiguration("invalid pinecone api key header".to_string())
            })?,
        );
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );

        let mut http_config = crate::backend::http::HttpClientConfig {
            timeout: options.timeout,
            default_headers: Some(headers),
            ..Default::default()
        };
        if let Some(retry) = options.retry_config {
            http_config.retry_config = Some(retry);
        }
        let http = HttpClient::with_config(options.base_url, http_config)?;
        let batch_size = options.batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
        let backend = Self {
            http,
            dimension: options.dimension,
            batch_size,
        };
        backend.health_check().await?;
        Ok(backend)
    }

    async fn health_check(&self) -> Result<(), DripstoneError> {
        let response = self.http.get("/").await?;
        if response.status().is_success() {
            return Ok(());
        }
        Err(DripstoneError::storage(format!(
            "pinecone health check failed with status {}",
            response.status()
        )))
    }

    async fn upsert_vectors(
        &self,
        namespace: &str,
        records: Vec<EmbeddingRecord>,
    ) -> Result<(), DripstoneError> {
        if records.is_empty() {
            return Ok(());
        }

        let mut vectors = Vec::with_capacity(records.len());
        for mut record in records {
            if record.vector.len() != self.dimension {
                return Err(DripstoneError::InvalidVectorLength {
                    expected: self.dimension,
                    actual: record.vector.len(),
                });
            }
            record.touch();
            vectors.push(json!({
                "id": record.id,
                "values": record.vector,
                "metadata": encode_metadata(record.metadata.take(), record.created_at, record.updated_at)?,
            }));
        }

        let payload = json!({
            "vectors": vectors,
            "namespace": namespace,
        });

        let response = self.http.post_json("/vectors/upsert", &payload).await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "pinecone upsert failed: status {status} body {body}"
            )));
        }
        Ok(())
    }

    async fn fetch_vector(
        &self,
        namespace: &str,
        id: &str,
    ) -> Result<Option<EmbeddingRecord>, DripstoneError> {
        let payload = json!({
            "ids": [id],
            "namespace": namespace,
        });

        let response = self.http.post_json("/vectors/fetch", &payload).await?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "pinecone fetch failed: status {status} body {body}"
            )));
        }
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        if let Some(vectors) = body.get("vectors").and_then(Value::as_object)
            && let Some(entry) = vectors.get(id)
        {
            let record = decode_record(namespace, entry, Some(id.to_string()))?;
            return Ok(Some(record));
        }
        Ok(None)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        let payload = json!({
            "deleteAll": true,
            "namespace": namespace,
        });

        let response = self.http.post_json("/vectors/delete", &payload).await?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "pinecone delete namespace failed: status {status} body {body}"
            )));
        }
        Ok(())
    }

    async fn describe_index_stats(&self) -> Result<Value, DripstoneError> {
        let response = self
            .http
            .post_json("/describe_index_stats", &json!({}))
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "pinecone describe stats failed: status {status} body {body}"
            )));
        }
        let body: Value = response.json().await.map_err(DripstoneError::from)?;
        Ok(body)
    }

    async fn query(
        &self,
        namespace: &str,
        vector: &[f32],
        limit: usize,
        filter: Option<Value>,
    ) -> Result<Value, DripstoneError> {
        let mut payload = json!({
            "vector": vector,
            "topK": limit,
            "namespace": namespace,
            "includeValues": true,
            "includeMetadata": true,
        });
        if let Some(filter) = filter {
            payload["filter"] = filter;
        }
        let response = self.http.post_json("/query", &payload).await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "pinecone query failed: status {status} body {body}"
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
            let mut wrapper = JsonMap::new();
            wrapper.insert("value".to_string(), other);
            wrapper
        }
        None => JsonMap::new(),
    };
    map.insert(
        CREATED_KEY.to_string(),
        Value::from(EmbeddingRecord::format_timestamp(&created_at)?),
    );
    map.insert(
        UPDATED_KEY.to_string(),
        Value::from(EmbeddingRecord::format_timestamp(&updated_at)?),
    );
    Ok(Value::Object(map))
}

fn decode_record(
    namespace: &str,
    value: &Value,
    id_override: Option<String>,
) -> Result<EmbeddingRecord, DripstoneError> {
    let id = match id_override {
        Some(id) => id,
        None => value
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| DripstoneError::storage("pinecone vector missing id"))?
            .to_string(),
    };
    let values = value
        .get("values")
        .and_then(Value::as_array)
        .ok_or_else(|| DripstoneError::storage("pinecone vector missing values"))?;
    let mut vector = Vec::with_capacity(values.len());
    for entry in values {
        let number = entry
            .as_f64()
            .ok_or_else(|| DripstoneError::storage("pinecone vector values not numeric"))?;
        vector.push(checked_f64_to_f32(number).ok_or_else(|| {
            DripstoneError::storage("pinecone vector element exceeds finite f32 range")
        })?);
    }
    let metadata_node = value.get("metadata").cloned().unwrap_or(Value::Null);
    let (metadata, created_at, updated_at) = decode_metadata(metadata_node)?;
    Ok(EmbeddingRecord {
        id: std::sync::Arc::from(id),
        vector: vector.into(),
        namespace: Some(std::sync::Arc::from(namespace)),
        metadata,
        sparse_vector: None,
        multi_vectors: None,
        created_at,
        updated_at,
    })
}

fn decode_metadata(
    value: Value,
) -> Result<(Option<Value>, OffsetDateTime, OffsetDateTime), DripstoneError> {
    match value {
        Value::Object(mut map) => {
            let created_at = map
                .remove(CREATED_KEY)
                .map(parse_timestamp)
                .transpose()?
                .unwrap_or_else(OffsetDateTime::now_utc);
            let updated_at = map
                .remove(UPDATED_KEY)
                .map(parse_timestamp)
                .transpose()?
                .unwrap_or(created_at);
            let metadata = if map.is_empty() {
                None
            } else {
                Some(Value::Object(map))
            };
            Ok((metadata, created_at, updated_at))
        }
        other => Ok((
            Some(other),
            OffsetDateTime::now_utc(),
            OffsetDateTime::now_utc(),
        )),
    }
}

fn parse_timestamp(node: Value) -> Result<OffsetDateTime, DripstoneError> {
    match node {
        Value::String(string) => EmbeddingRecord::parse_timestamp(string.as_str()),
        other => EmbeddingRecord::parse_timestamp(other.to_string().as_str()),
    }
}

impl PineconeEmbeddingBackend {
    async fn list_namespace_embeddings(
        &self,
        namespace: &str,
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        let mut records = Vec::new();
        let mut pagination: Option<String> = None;
        loop {
            let mut payload = json!({
                "namespace": namespace,
                "includeValues": true,
                "includeMetadata": true,
                "limit": 100,
            });
            if let Some(token) = &pagination {
                payload["paginationToken"] = Value::from(token.clone());
            }
            let response = self.http.post_json("/vectors/list", &payload).await?;
            let status = response.status();
            if status == reqwest::StatusCode::NOT_FOUND {
                break;
            }
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(DripstoneError::storage(format!(
                    "pinecone list namespace failed: status {status} body {body}"
                )));
            }
            let body: Value = response.json().await.map_err(DripstoneError::from)?;
            let vectors = body
                .get("vectors")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for entry in vectors {
                let record = decode_record(
                    namespace,
                    &entry,
                    entry
                        .get("id")
                        .and_then(Value::as_str)
                        .map(std::string::ToString::to_string),
                )?;
                records.push(record);
            }
            pagination = body
                .get("pagination")
                .and_then(Value::as_object)
                .and_then(|pagination| pagination.get("next"))
                .and_then(Value::as_str)
                .map(std::string::ToString::to_string);
            if pagination.is_none() {
                break;
            }
        }
        Ok(records)
    }
}

// =============================================================================
// Backend trait implementation
// =============================================================================

use crate::backend::chunk_envelope::{self, decode_chunk_from_envelope};
use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::traits::Backend;
use std::fmt;

impl fmt::Debug for PineconeEmbeddingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PineconeBackend")
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

/// Pinecone backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct PineconeBackend {
    inner: PineconeEmbeddingBackend,
}

impl PineconeBackend {
    /// Create a new Pinecone backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_options(options: PineconeBackendOptions) -> Result<Self, DripstoneError> {
        let inner = PineconeEmbeddingBackend::new(options).await?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl Backend for PineconeBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            requires_exact_filter_pass: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: Some(self.inner.dimension),
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
            self.inner.upsert_vectors(&namespace, records).await?;
        }

        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let record = self.inner.fetch_vector(namespace, id).await?;

        match record {
            Some(r) => Ok(Some(chunk_envelope::decode_chunk(&r, Some(namespace))?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let payload = json!({
            "ids": [id],
            "namespace": namespace,
        });
        let response = self
            .inner
            .http
            .post_json("/vectors/delete", &payload)
            .await?;
        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::NOT_FOUND {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "pinecone delete failed: status {status} body {body}"
            )));
        }
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        // Pinecone doesn't support filtering by metadata in delete, so we need to query first
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let filter = json!({
            "doc_id": { "$eq": doc_id }
        });

        // Query to find all matching vectors
        let query_result = self
            .inner
            .query(
                namespace,
                &vec![0.0; self.inner.dimension],
                10000,
                Some(filter),
            )
            .await?;
        let empty: Vec<Value> = Vec::new();
        let matches = query_result
            .get("matches")
            .and_then(Value::as_array)
            .unwrap_or(&empty);

        if matches.is_empty() {
            return Ok(0);
        }

        let ids: Vec<String> = matches
            .iter()
            .filter_map(|m| {
                m.get("id")
                    .and_then(Value::as_str)
                    .map(std::string::ToString::to_string)
            })
            .collect();

        if !ids.is_empty() {
            let payload = json!({
                "ids": ids,
                "namespace": namespace,
            });
            let response = self
                .inner
                .http
                .post_json("/vectors/delete", &payload)
                .await?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                return Err(DripstoneError::storage(format!(
                    "pinecone delete document failed: status {status} body {body}"
                )));
            }
        }

        Ok(ids.len())
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let namespace = query.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);

        let filter = query.filter.as_ref().map(|_f| {
            // Convert SearchQuery filter to Pinecone filter format
            // This is a simplified conversion - may need enhancement
            json!({})
        });

        let query_result = self
            .inner
            .query(namespace, &query.vector, query.limit, filter)
            .await?;
        let matches = query_result
            .get("matches")
            .and_then(Value::as_array)
            .ok_or_else(|| DripstoneError::storage("pinecone query response missing matches"))?;

        let mut results = Vec::with_capacity(matches.len());
        for (i, match_obj) in matches.iter().enumerate() {
            let score = match_obj
                .get("score")
                .and_then(Value::as_f64)
                .map(|score| {
                    checked_f64_to_f32(score).ok_or_else(|| {
                        DripstoneError::storage("pinecone score exceeds finite f32 range")
                    })
                })
                .transpose()?
                .unwrap_or(0.0);

            let id = match_obj
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| DripstoneError::storage("pinecone match missing id"))?;

            let values = match_obj
                .get("values")
                .and_then(Value::as_array)
                .ok_or_else(|| DripstoneError::storage("pinecone match missing values"))?;

            let vector: Vec<f32> = values
                .iter()
                .map(|value| {
                    let value = value.as_f64().ok_or_else(|| {
                        DripstoneError::storage("pinecone match vector element was not numeric")
                    })?;
                    checked_f64_to_f32(value).ok_or_else(|| {
                        DripstoneError::storage(
                            "pinecone match vector element exceeds finite f32 range",
                        )
                    })
                })
                .collect::<Result<_, _>>()?;

            let envelope = match_obj
                .get("metadata")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
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
        let stats = self.inner.describe_index_stats().await?;
        let namespaces = stats
            .get("namespaces")
            .and_then(Value::as_object)
            .map_or_else(Vec::new, |obj| obj.keys().cloned().collect());
        Ok(namespaces)
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        self.inner.delete_namespace(namespace).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let stats = self.inner.describe_index_stats().await?;
        if let Some(namespace) = namespace {
            let namespace_stats = stats
                .get("namespaces")
                .and_then(Value::as_object)
                .and_then(|obj| obj.get(namespace))
                .and_then(Value::as_object);

            if let Some(ns_stats) = namespace_stats {
                let count = ns_stats
                    .get("vector_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                return usize::try_from(count).map_err(|_| {
                    DripstoneError::storage("pinecone namespace count exceeds addressable size")
                });
            }
            return Ok(0);
        }

        // Total count across all namespaces
        let total = stats
            .get("totalVectorCount")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        usize::try_from(total)
            .map_err(|_| DripstoneError::storage("pinecone count exceeds addressable size"))
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.list_namespace_embeddings(namespace).await?;

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
