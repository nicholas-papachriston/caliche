use super::client::{QdrantBackendOptions, QdrantEmbeddingBackend};
use super::helpers::{SCROLL_PAGE_SIZE, count_namespace};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fmt;

use crate::backend::chunk_envelope;
use crate::core::{Capabilities, Chunk, SearchQuery, SearchResult};
use crate::embedding::{DEFAULT_NAMESPACE, SimilarityMetric};
use crate::errors::DripstoneError;
use crate::search::filter_to_qdrant::filter_to_qdrant;
use crate::traits::Backend;
use crate::utils::numeric::checked_f64_to_f32;

impl fmt::Debug for QdrantEmbeddingBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QdrantHttpBackend")
            .field("collection_prefix", &self.collection_prefix)
            .field("vector_dimension", &self.vector_dimension)
            .finish_non_exhaustive()
    }
}

/// Qdrant HTTP backend for the new `Dripstone` API.
#[derive(Debug)]
pub struct QdrantHttpBackend {
    inner: QdrantEmbeddingBackend,
}

impl QdrantHttpBackend {
    /// Create a new Qdrant HTTP backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn with_options(options: QdrantBackendOptions) -> Result<Self, DripstoneError> {
        let inner = QdrantEmbeddingBackend::new(options).await?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl Backend for QdrantHttpBackend {
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            native_hybrid: false,
            native_full_text: false,
            native_hybrid_full_text: false,
            native_filter: true,
            supports_namespaces: true,
            native_maxsim: false,
            max_dimension: None,
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
            let mut points = Vec::with_capacity(chunk_group.len());
            for chunk in chunk_group {
                let record = chunk_envelope::embedding_record_from_chunk(chunk);
                let (_, point) = self.inner.point_from_record(record)?;
                points.push(point);
            }
            self.inner.upsert_points(&namespace, points).await?;
        }

        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.inner.collection_name(namespace);

        let record = if QdrantEmbeddingBackend::qdrant_id_is_native_point_key(id) {
            let path = format!("/collections/{collection}/points/{id}");
            match self.inner.http.get(path.as_str()).await {
                Ok(response) => {
                    if response.status() == reqwest::StatusCode::NOT_FOUND {
                        None
                    } else {
                        let body = self.inner.expect_ok(response).await?;
                        let result = body.get("result").ok_or_else(|| {
                            DripstoneError::storage("qdrant get response missing result")
                        })?;
                        Some(self.inner.record_from_point(namespace, result)?)
                    }
                }
                // Preserve the historical semantics where "not found" is surfaced as
                // `None`, but do not silently swallow genuine transport / server
                // failures: those still propagate as errors so the caller can
                // distinguish "missing document" from "qdrant unreachable".
                Err(e) if e.is_not_found() => None,
                Err(e) => return Err(e),
            }
        } else {
            None
        };

        let record = match record {
            Some(r) => Some(r),
            None => {
                self.inner
                    .get_record_by_original_id_scroll(&collection, namespace, id)
                    .await?
            }
        };

        let Some(record) = record else {
            return Ok(None);
        };

        Ok(Some(chunk_envelope::decode_chunk(
            &record,
            Some(namespace),
        )?))
    }

    async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.inner.collection_name(namespace);
        let path = format!("/collections/{collection}/points/delete?wait=true");

        let Some(point_id) = self
            .inner
            .qdrant_delete_point_value(&collection, namespace, id)
            .await?
        else {
            return Ok(());
        };

        let payload = json!({
            "points": [point_id]
        });

        let response = self.inner.http.post_json(path.as_str(), &payload).await?;
        self.inner.expect_operation_finished(response).await?;
        Ok(())
    }

    async fn delete_document(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        // Search for all chunks with this doc_id and delete them
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        let collection = self.inner.collection_name(namespace);
        let path = format!("/collections/{collection}/points/scroll");

        let mut request = json!({
            "limit": SCROLL_PAGE_SIZE,
            "filter": {
                "must": [{
                    "key": "metadata.doc_id",
                    "match": { "value": doc_id }
                }]
            },
            "with_payload": false,
            "with_vector": false,
        });

        let mut deleted = 0;
        loop {
            let Ok(response) = self.inner.http.post_json(path.as_str(), &request).await else {
                break; // Collection might not exist
            };

            if response.status() == reqwest::StatusCode::NOT_FOUND {
                break;
            }

            let body = self.inner.expect_ok(response).await?;
            let result = body
                .get("result")
                .and_then(Value::as_object)
                .ok_or_else(|| DripstoneError::storage("scroll response missing result"))?;

            let empty: Vec<Value> = Vec::new();
            let points = result
                .get("points")
                .and_then(Value::as_array)
                .unwrap_or(&empty);

            if points.is_empty() {
                break;
            }

            let ids: Vec<Value> = points.iter().filter_map(|p| p.get("id").cloned()).collect();

            if !ids.is_empty() {
                let delete_path = format!("/collections/{collection}/points/delete?wait=true");
                let delete_payload = json!({ "points": ids });
                let delete_response = self
                    .inner
                    .http
                    .post_json(delete_path.as_str(), &delete_payload)
                    .await?;
                self.inner
                    .expect_operation_finished(delete_response)
                    .await?;
                deleted += ids.len();
            }

            match result.get("next_page_offset") {
                Some(offset) if !offset.is_null() => {
                    request["offset"] = offset.clone();
                }
                _ => break,
            }
        }

        Ok(deleted)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        const MAX_OVERSAMPLE: usize = 1024;
        let namespace = query.namespace.as_deref().unwrap_or(DEFAULT_NAMESPACE);
        let metric = SimilarityMetric::from(query.metric);
        self.inner.validate_metric(metric)?;

        let collection = self.inner.collection_name(namespace);
        let path = format!("/collections/{collection}/points/search");

        // Decide whether the filter can be pushed down to Qdrant. If not, we
        // oversample and apply `Filter::matches` client-side to preserve
        // correctness (mirrors the postgres backend behaviour so empty
        // `In([])` doesn't silently return unrelated rows, and empty
        // `NotIn([])` doesn't get turned into a hard-fail).
        let (pushdown_filter, client_filter_fallback) =
            query.filter.as_ref().map_or((None, false), |filter| {
                filter_to_qdrant(filter).map_or((None, true), |value| (Some(value), false))
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

        // Always request vectors: `record_from_point` needs them to build `Chunk`. Callers can use
        // `SearchQuery::include_vectors` only for other backends; stripping vectors here would
        // require a placeholder of the correct dimension for every hit.
        let mut search_request = json!({
            "vector": query.vector,
            "limit": fetch_limit,
            "with_payload": true,
            "with_vector": true,
        });

        if let Some(qdrant_filter) = pushdown_filter {
            search_request["filter"] = qdrant_filter;
        }

        let response = self
            .inner
            .http
            .post_json(path.as_str(), &search_request)
            .await?;
        let body = self.inner.expect_ok(response).await?;
        let result = body
            .get("result")
            .and_then(Value::as_array)
            .ok_or_else(|| DripstoneError::storage("search response missing result array"))?;

        let mut results = Vec::with_capacity(result.len());
        for point in result {
            let score = point
                .get("score")
                .and_then(Value::as_f64)
                .map(|score| {
                    checked_f64_to_f32(score).ok_or_else(|| {
                        DripstoneError::storage("qdrant score exceeds finite f32 range")
                    })
                })
                .transpose()?
                .unwrap_or(0.0);

            let record = self.inner.record_from_point(namespace, point)?;
            let chunk = chunk_envelope::decode_chunk(&record, Some(namespace))?;

            if client_filter_fallback
                && let Some(ref f) = query.filter
                && !f.matches(&chunk.metadata)
            {
                continue;
            }

            results.push(SearchResult::new(chunk, score, results.len()));
            if results.len() >= query.limit {
                break;
            }
        }

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let path = "/collections";
        let response = self.inner.http.get(path).await?;
        let body = self.inner.expect_ok(response).await?;
        let result = body
            .get("result")
            .and_then(Value::as_object)
            .and_then(|o| o.get("collections"))
            .and_then(Value::as_array)
            .ok_or_else(|| DripstoneError::storage("collections response missing result"))?;

        let mut namespaces = Vec::new();
        let prefix_len = self.inner.collection_prefix.len();
        for collection in result {
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
        self.inner.delete_collection(namespace).await
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        let namespace = namespace.unwrap_or(DEFAULT_NAMESPACE);
        count_namespace(&self.inner.http, &self.inner.collection_name(namespace)).await
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
            chunks.push(chunk_envelope::decode_chunk(&record, None)?);
        }

        if let Some(limit) = limit {
            chunks.truncate(limit);
        }

        Ok(chunks)
    }
}
