use super::client::{WeaviateBackendOptions, WeaviateEmbeddingBackend};
use super::convert::extract_record;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

use crate::backend::chunk_envelope;
use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::embedding::{DEFAULT_NAMESPACE, EmbeddingRecord};
use crate::errors::DripstoneError;
use crate::search::filter_to_weaviate::filter_to_weaviate;
use crate::traits::Backend;
use crate::utils::numeric::{checked_f64_to_f32, usize_to_f32};

// Backend trait implementation
// =============================================================================

impl fmt::Debug for WeaviateEmbeddingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeaviateBackend")
            .field("class_prefix", &self.class_prefix)
            .field("vector_dimension", &self.vector_dimension)
            .finish_non_exhaustive()
    }
}

/// Weaviate backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct WeaviateBackend {
    inner: WeaviateEmbeddingBackend,
}

impl WeaviateBackend {
    /// Create a new Weaviate backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_options(options: WeaviateBackendOptions) -> Result<Self, DripstoneError> {
        let inner = WeaviateEmbeddingBackend::new(options).await?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl Backend for WeaviateBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: Some(self.inner.vector_dimension),
            max_batch: Some(self.inner.batch_size),
            supports_streaming: false,
            requires_exact_filter_pass: true,
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
            self.inner.upsert_objects(&namespace, records).await?;
        }

        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let record = self.inner.get_object(namespace, id).await?;

        match record {
            Some(r) => Ok(Some(chunk_envelope::decode_chunk(&r, Some(namespace))?)),
            None => Ok(None),
        }
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.inner.delete_object(namespace, id).await
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        // Query for all chunks with this doc_id, then delete them
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        // Best-effort: metadata is stored as JSON text; match doc_id substring in the blob.
        let escaped = doc_id.replace('\\', "\\\\").replace('"', "\\\"");
        let where_filter = Some(format!(
            r#"path: ["metadata"], operator: Like, valueText: "*\"doc_id\":\"{escaped}\""#
        ));

        // Use query_class to find matching objects, then extract and delete
        let query_result = self
            .inner
            .query_class(
                namespace,
                &vec![0.0; self.inner.vector_dimension],
                10000,
                where_filter,
            )
            .await?;
        let class_name = self.inner.class_name(namespace);
        let empty_vec: Vec<Value> = Vec::new();
        let objects = query_result
            .get("data")
            .and_then(|d| d.get("Get"))
            .and_then(|g| g.get(class_name))
            .and_then(Value::as_array)
            .unwrap_or(&empty_vec);

        if objects.is_empty() {
            return Ok(0);
        }

        // Delete each object individually (Weaviate doesn't have batch delete by filter)
        let mut deleted = 0;
        for obj in objects {
            if let Ok(Some(record)) = extract_record(namespace, obj) {
                self.inner.delete_object(namespace, &record.id).await?;
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        const MAX_OVERSAMPLE: usize = 1024;
        let namespace = query.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);

        let (where_filter, client_filter_fallback) =
            query.filter.as_ref().map_or((None, false), |filter| {
                filter_to_weaviate(filter).map_or((None, true), |value| (Some(value), false))
            });

        let fetch_limit = if client_filter_fallback {
            query
                .limit
                .saturating_mul(2)
                .min(MAX_OVERSAMPLE)
                .max(query.limit)
        } else {
            query.limit
        };

        let query_result = self
            .inner
            .query_class(namespace, &query.vector, fetch_limit, where_filter)
            .await?;

        // Extract records from GraphQL query result
        let class_name = self.inner.class_name(namespace);
        let empty_vec: Vec<Value> = Vec::new();
        let objects = query_result
            .get("data")
            .and_then(|d| d.get("Get"))
            .and_then(|g| g.get(class_name))
            .and_then(Value::as_array)
            .unwrap_or(&empty_vec);

        // Extract (record, optional Weaviate-returned distance/certainty) pairs
        // so the per-result score reflects Weaviate's scoring rather than a
        // rank-based placeholder. When `_additional.distance` is present we
        // derive `score = 1 - distance` (cosine), otherwise we fall back to
        // `_additional.certainty`. If neither is present we keep a monotonic
        // placeholder but flag it as non-authoritative via `new_unscored`.
        let mut records_with_score: Vec<(EmbeddingRecord, Option<f32>)> = Vec::new();
        for obj in objects {
            if let Ok(Some(record)) = extract_record(namespace, obj) {
                let additional = obj.get("_additional");
                let distance = additional
                    .and_then(|a| a.get("distance"))
                    .and_then(Value::as_f64)
                    .map(|distance| {
                        checked_f64_to_f32(distance).ok_or_else(|| {
                            DripstoneError::storage("weaviate distance exceeds finite f32 range")
                        })
                    })
                    .transpose()?;
                let certainty = additional
                    .and_then(|a| a.get("certainty"))
                    .and_then(Value::as_f64)
                    .map(|certainty| {
                        checked_f64_to_f32(certainty).ok_or_else(|| {
                            DripstoneError::storage("weaviate certainty exceeds finite f32 range")
                        })
                    })
                    .transpose()?;
                let score = distance.map(|d| 1.0 - d).or(certainty);
                records_with_score.push((record, score));
            }
        }

        let total = usize_to_f32(records_with_score.len().max(1));
        let mut results = Vec::with_capacity(records_with_score.len());
        for (i, (record, score_opt)) in records_with_score.into_iter().enumerate() {
            let chunk = chunk_envelope::decode_chunk(&record, Some(namespace))?;

            if client_filter_fallback
                && let Some(ref f) = query.filter
                && !f.matches(&chunk.metadata)
            {
                continue;
            }

            if let Some(score) = score_opt {
                results.push(SearchResult::new(chunk, score, results.len()));
            } else {
                let placeholder = 1.0 - (usize_to_f32(i) / total);
                results.push(SearchResult::new_unscored(
                    chunk,
                    placeholder,
                    results.len(),
                ));
            }

            if results.len() >= query.limit {
                break;
            }
        }

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        self.inner.list_classes().await
    }

    async fn delete_namespace(&self, namespace: &str) -> Result<(), DripstoneError> {
        // One Weaviate class per namespace; drop the class (and all objects).
        let class = self.inner.class_name(namespace);
        let path = format!("/v1/schema/{class}");
        let response = self.inner.http.delete(path.as_str()).await?;
        let status = response.status();
        if !status.is_success() && status != reqwest::StatusCode::NOT_FOUND {
            let body = response.text().await.unwrap_or_default();
            return Err(DripstoneError::storage(format!(
                "weaviate delete namespace failed: status {status} body {body}"
            )));
        }
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        self.inner.count_class(namespace).await
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let records = self.inner.list_class_objects(namespace).await?;

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
