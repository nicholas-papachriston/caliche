use super::convert::{PointPayload, UpsertRequest};
use super::helpers::{ORIGINAL_ID_KEY, SCROLL_PAGE_SIZE, normalize_prefix, qdrant_distance};
use reqwest::header::{HeaderMap, HeaderValue};
use serde_json::{Value, json};
use std::time::Duration;

use crate::backend::http::HttpClient;
use crate::embedding::{EmbeddingRecord, SimilarityMetric};
use crate::errors::DripstoneError;

const DEFAULT_BATCH_SIZE: usize = 500;
#[cfg(feature = "qdrant-backend")]
const UPSERT_RETRY_ATTEMPTS: usize = 3;
#[cfg(feature = "qdrant-backend")]
const UPSERT_RETRY_BASE_DELAY_MS: u64 = 10;

#[cfg(feature = "qdrant-backend")]
pub struct QdrantEmbeddingBackend {
    pub(crate) http: HttpClient,
    pub(crate) collection_prefix: String,
    pub(crate) vector_dimension: usize,
    pub(crate) distance: SimilarityMetric,
    pub(crate) batch_size: usize,
}

#[cfg(feature = "qdrant-backend")]
#[derive(Debug)]
pub struct QdrantBackendOptions {
    pub base_url: String,
    pub collection_prefix: Option<String>,
    pub vector_dimension: usize,
    pub distance: SimilarityMetric,
    pub timeout: Option<Duration>,
    pub api_key: Option<String>,
    pub batch_size: Option<usize>,
    pub retry_config: Option<crate::backend::http::RetryConfig>,
}

#[cfg(feature = "qdrant-backend")]
impl QdrantBackendOptions {
    pub fn new(base_url: impl Into<String>, vector_dimension: usize) -> Self {
        Self {
            base_url: base_url.into(),
            collection_prefix: None,
            vector_dimension,
            distance: SimilarityMetric::Cosine,
            timeout: None,
            api_key: None,
            batch_size: None,
            retry_config: None,
        }
    }
}

#[cfg(feature = "qdrant-backend")]
impl QdrantEmbeddingBackend {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn new(options: QdrantBackendOptions) -> Result<Self, DripstoneError> {
        if options.vector_dimension == 0 {
            return Err(DripstoneError::Misconfiguration(
                "vector dimension must be greater than zero".to_string(),
            ));
        }

        let QdrantBackendOptions {
            base_url,
            collection_prefix,
            vector_dimension,
            distance,
            timeout,
            api_key,
            batch_size,
            retry_config,
        } = options;

        let headers = api_key
            .as_ref()
            .map(|key| build_qdrant_headers(key))
            .transpose()?;

        let mut http_config = crate::backend::http::HttpClientConfig {
            timeout,
            default_headers: headers,
            ..Default::default()
        };
        if let Some(retry) = retry_config {
            http_config.retry_config = Some(retry);
        }
        let http = HttpClient::with_config(base_url, http_config)?;
        let prefix = normalize_prefix(collection_prefix);
        let batch_size = batch_size.unwrap_or(DEFAULT_BATCH_SIZE).max(1);
        let backend = Self {
            http,
            collection_prefix: prefix,
            vector_dimension,
            distance,
            batch_size,
        };
        backend.health_check().await?;
        Ok(backend)
    }

    pub(crate) fn collection_name(&self, namespace: &str) -> String {
        format!("{}{}", self.collection_prefix, namespace)
    }

    pub(crate) async fn health_check(&self) -> Result<(), DripstoneError> {
        let response = self.http.get("/healthz").await?;
        if response.status().is_success() {
            return Ok(());
        }
        Err(DripstoneError::storage(format!(
            "qdrant health check failed with status {}",
            response.status()
        )))
    }

    pub(crate) async fn ensure_collection(&self, namespace: &str) -> Result<(), DripstoneError> {
        let collection = self.collection_name(namespace);
        let path = format!("/collections/{collection}");
        let response = match self.http.get(path.as_str()).await {
            Ok(response) => response,
            Err(e) => {
                // Check if the error is a 404 (collection doesn't exist)
                let error_msg = format!("{e}");
                if error_msg.contains("404")
                    || error_msg.contains("Not found")
                    || error_msg.contains("doesn't exist")
                {
                    // Collection doesn't exist, create it
                } else {
                    return Err(e);
                }
                // Fall through to create collection
                return self.create_collection(namespace).await;
            }
        };
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return self.create_collection(namespace).await;
        }
        self.expect_ok(response).await?;
        Ok(())
    }

    pub(crate) async fn create_collection(&self, namespace: &str) -> Result<(), DripstoneError> {
        let collection = self.collection_name(namespace);
        let payload = json!({
            "vectors": {
                "size": self.vector_dimension,
                "distance": qdrant_distance(self.distance),
            }
        });
        let create_path = format!("/collections/{collection}?wait=true");
        let response = match self.http.put_json(create_path.as_str(), &payload).await {
            Ok(response) => response,
            Err(e) => {
                // Check if the error is a CONFLICT (collection already exists)
                let error_msg = format!("{e}");
                if error_msg.contains("409")
                    || error_msg.contains("CONFLICT")
                    || error_msg.contains("already exists")
                {
                    return Ok(());
                }
                return Err(e);
            }
        };
        if response.status() == reqwest::StatusCode::CONFLICT {
            return Ok(());
        }
        self.expect_operation_finished(response).await?;
        Ok(())
    }

    pub(crate) async fn expect_ok(
        &self,
        response: reqwest::Response,
    ) -> Result<Value, DripstoneError> {
        let status = response.status();
        let bytes = response.bytes().await.map_err(DripstoneError::from)?;
        if !status.is_success() {
            let preview = String::from_utf8_lossy(&bytes);
            let message = preview.chars().take(512).collect::<String>();
            return Err(DripstoneError::storage(format!(
                "qdrant request failed with status {status}: {message}"
            )));
        }
        let body: Value = serde_json::from_slice(&bytes).map_err(DripstoneError::from)?;
        let status = body
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("error");
        if status != "ok" {
            return Err(DripstoneError::storage(format!(
                "qdrant returned non-ok status: {body}"
            )));
        }
        Ok(body)
    }

    pub(crate) async fn expect_operation_finished(
        &self,
        response: reqwest::Response,
    ) -> Result<Value, DripstoneError> {
        let body = self.expect_ok(response).await?;
        match body.get("result") {
            Some(Value::Bool(_)) => Ok(body),
            Some(Value::Object(map)) => {
                let status = map.get("status").and_then(Value::as_str);
                match status {
                    Some("completed" | "acknowledged") => Ok(body),
                    _ => Err(DripstoneError::storage(format!(
                        "qdrant operation did not complete: {body}"
                    ))),
                }
            }
            _ => Err(DripstoneError::storage(format!(
                "unexpected qdrant operation response: {body}"
            ))),
        }
    }

    pub(crate) async fn upsert_points(
        &self,
        namespace: &str,
        points: Vec<PointPayload>,
    ) -> Result<(), DripstoneError> {
        if points.is_empty() {
            return Ok(());
        }
        let path = format!(
            "/collections/{}/points?wait=true",
            self.collection_name(namespace)
        );
        let payload = UpsertRequest { points };
        let mut attempt = 0usize;
        loop {
            self.ensure_collection(namespace).await?;
            let response = self.http.put_json(path.as_str(), &payload).await?;
            let status = response.status();
            if status.is_success() {
                self.expect_operation_finished(response).await?;
                return Ok(());
            }
            if matches!(
                status,
                reqwest::StatusCode::CONFLICT | reqwest::StatusCode::INTERNAL_SERVER_ERROR
            ) && attempt < UPSERT_RETRY_ATTEMPTS
            {
                let delay = UPSERT_RETRY_BASE_DELAY_MS.saturating_mul(1_u64 << attempt);
                attempt += 1;
                #[cfg(feature = "tokio")]
                tokio::time::sleep(Duration::from_millis(delay.max(UPSERT_RETRY_BASE_DELAY_MS)))
                    .await;
                #[cfg(not(feature = "tokio"))]
                std::thread::sleep(Duration::from_millis(delay.max(UPSERT_RETRY_BASE_DELAY_MS)));
                continue;
            }
            self.expect_operation_finished(response).await?;
            return Ok(());
        }
    }

    #[expect(dead_code, reason = "nothing calls it yet")]
    pub(crate) async fn get_collection_points(
        &self,
        namespace: &str,
        request: Value,
        endpoint: &str,
    ) -> Result<Value, DripstoneError> {
        let path = format!(
            "/collections/{}/{}",
            self.collection_name(namespace),
            endpoint
        );
        let response = self.http.post_json(path.as_str(), &request).await?;
        self.expect_ok(response).await
    }

    pub(crate) fn validate_metric(&self, metric: SimilarityMetric) -> Result<(), DripstoneError> {
        if metric != self.distance {
            return Err(DripstoneError::Misconfiguration(format!(
                "requested metric {:?} differs from collection metric {:?}",
                metric, self.distance
            )));
        }
        Ok(())
    }

    pub(crate) async fn list_namespace_embeddings(
        &self,
        namespace: &str,
    ) -> Result<Vec<EmbeddingRecord>, DripstoneError> {
        let collection = self.collection_name(namespace);
        let path = format!("/collections/{collection}/points/scroll");
        let mut request = json!({
            "limit": SCROLL_PAGE_SIZE,
            "with_payload": true,
            "with_vector": true,
        });
        let mut records = Vec::new();

        loop {
            let response = match self.http.post_json(path.as_str(), &request).await {
                Ok(response) => {
                    if response.status() == reqwest::StatusCode::NOT_FOUND {
                        return Ok(Vec::new());
                    }
                    response
                }
                Err(e) => {
                    // Check if the error is a 404 (collection doesn't exist)
                    let error_msg = format!("{e}");
                    if error_msg.contains("404")
                        || error_msg.contains("Not found")
                        || error_msg.contains("doesn't exist")
                    {
                        return Ok(Vec::new());
                    }
                    return Err(e);
                }
            };
            let body = self.expect_ok(response).await?;
            let result = body
                .get("result")
                .and_then(Value::as_object)
                .ok_or_else(|| DripstoneError::storage("scroll response missing result object"))?;
            let points = result
                .get("points")
                .and_then(Value::as_array)
                .ok_or_else(|| DripstoneError::storage("scroll response missing points array"))?;
            for point in points {
                let record = self.record_from_point(namespace, point)?;
                records.push(record);
            }
            match result.get("next_page_offset") {
                Some(offset) if !offset.is_null() => {
                    request["offset"] = offset.clone();
                }
                _ => break,
            }
        }
        Ok(records)
    }

    pub(crate) async fn delete_collection(&self, namespace: &str) -> Result<(), DripstoneError> {
        let path = format!("/collections/{}?wait=true", self.collection_name(namespace));
        let response = match self.http.delete(path.as_str()).await {
            Ok(response) => {
                if response.status() == reqwest::StatusCode::NOT_FOUND {
                    return Ok(());
                }
                response
            }
            Err(e) => {
                // Check if the error is a 404 (collection doesn't exist)
                let error_msg = format!("{e}");
                if error_msg.contains("404")
                    || error_msg.contains("Not found")
                    || error_msg.contains("doesn't exist")
                {
                    return Ok(());
                }
                return Err(e);
            }
        };
        self.expect_operation_finished(response).await?;
        Ok(())
    }

    /// When false, the point is stored under a generated id and [`ORIGINAL_ID_KEY`] in payload.
    pub(crate) fn qdrant_id_is_native_point_key(id: &str) -> bool {
        use uuid::Uuid;
        id.parse::<u64>().is_ok() || Uuid::parse_str(id).is_ok()
    }

    pub(crate) async fn get_record_by_original_id_scroll(
        &self,
        collection: &str,
        namespace: &str,
        original_id: &str,
    ) -> Result<Option<EmbeddingRecord>, DripstoneError> {
        let path = format!("/collections/{collection}/points/scroll");
        let request = json!({
            "limit": 2,
            "filter": {
                "must": [{
                    "key": ORIGINAL_ID_KEY,
                    "match": { "value": original_id }
                }]
            },
            "with_payload": true,
            "with_vector": true,
        });

        let response = match self.http.post_json(path.as_str(), &request).await {
            Ok(r) => r,
            Err(e) if e.is_not_found() => return Ok(None),
            Err(e) => return Err(e),
        };

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }

        let body = match self.expect_ok(response).await {
            Ok(b) => b,
            Err(e) if e.is_not_found() => return Ok(None),
            Err(e) => return Err(e),
        };

        let points = body
            .get("result")
            .and_then(|r| r.as_object())
            .and_then(|r| r.get("points"))
            .and_then(|p| p.as_array());

        let Some(points) = points else {
            return Ok(None);
        };

        if points.is_empty() {
            return Ok(None);
        }

        if points.len() > 1 {
            return Err(DripstoneError::storage(
                "multiple qdrant points share the same original id",
            ));
        }

        self.record_from_point(namespace, &points[0]).map(Some)
    }

    /// Value to pass in a delete request's `points` array (numeric id, UUID string, etc.).
    pub(crate) async fn qdrant_delete_point_value(
        &self,
        collection: &str,
        _namespace: &str,
        id: &str,
    ) -> Result<Option<Value>, DripstoneError> {
        if Self::qdrant_id_is_native_point_key(id) {
            if let Ok(numeric) = id.parse::<u64>() {
                return Ok(Some(Value::from(numeric)));
            }
            return Ok(Some(Value::String(id.to_owned())));
        }
        let path = format!("/collections/{collection}/points/scroll");
        let request = json!({
            "limit": 2,
            "filter": {
                "must": [{
                    "key": ORIGINAL_ID_KEY,
                    "match": { "value": id }
                }]
            },
            "with_payload": false,
            "with_vector": false,
        });
        let response = match self.http.post_json(path.as_str(), &request).await {
            Ok(r) => r,
            Err(e) if e.is_not_found() => return Ok(None),
            Err(e) => return Err(e),
        };
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let body = match self.expect_ok(response).await {
            Ok(b) => b,
            Err(e) if e.is_not_found() => return Ok(None),
            Err(e) => return Err(e),
        };
        let points = body
            .get("result")
            .and_then(|r| r.as_object())
            .and_then(|r| r.get("points"))
            .and_then(|p| p.as_array());
        let Some(points) = points else {
            return Ok(None);
        };
        if points.is_empty() {
            return Ok(None);
        }
        if points.len() > 1 {
            return Err(DripstoneError::storage(
                "multiple qdrant points share the same original id",
            ));
        }
        points[0]
            .get("id")
            .cloned()
            .ok_or_else(|| DripstoneError::storage("qdrant scroll result missing point id"))
            .map(Some)
    }
}

#[cfg(feature = "qdrant-backend")]
pub fn build_qdrant_headers(api_key: &str) -> Result<HeaderMap, DripstoneError> {
    let trimmed = api_key.trim();
    if trimmed.is_empty() {
        return Err(DripstoneError::Misconfiguration(
            "qdrant api key cannot be empty".to_string(),
        ));
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        "api-key",
        HeaderValue::from_str(trimmed).map_err(|_| {
            DripstoneError::Misconfiguration("invalid characters in qdrant api key".to_string())
        })?,
    );
    Ok(headers)
}

// =============================================================================
