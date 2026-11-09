//! S3 Vector Buckets API implementation
//!
//! This module implements direct API calls to AWS S3 Vector Buckets (preview feature).
//! See: <https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors-getting-started.html>
//!
//! The default HTTPS host is `s3vectors.{region}.api.aws` (same as `aws-sdk-s3vectors`), not `*.amazonaws.com`.

use aws_config::SdkConfig;
use bytes::Bytes;
use parking_lot::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::aws::{send_signed_json, sign_json_http_request};
use crate::errors::DripstoneError;

/// Configuration for S3 Vector operations
#[derive(Debug, Clone)]
pub struct VectorConfig {
    /// Vector bucket name
    pub bucket_name: String,
    /// Vector index name within the bucket
    pub index_name: String,
    /// Vector dimension (1-4096)
    pub dimension: usize,
    /// Distance metric (Cosine or Euclidean)
    pub distance_metric: VectorDistanceMetric,
    /// AWS region
    pub region: String,
    /// Standard S3 bucket for namespace key lists (`__dripstone_indexes/.../_keys.json`).
    /// Vector buckets do not support `GetObject` / `ListObjectsV2`; set this for durable listing.
    /// If `None`, key lists are kept in memory for the lifetime of the client.
    pub sidecar_s3_bucket: Option<String>,
}

/// Distance metrics supported by S3 Vectors
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum VectorDistanceMetric {
    Cosine,
    Euclidean,
}

impl From<crate::embedding::SimilarityMetric> for VectorDistanceMetric {
    fn from(metric: crate::embedding::SimilarityMetric) -> Self {
        match metric {
            crate::embedding::SimilarityMetric::Euclidean => Self::Euclidean,
            crate::embedding::SimilarityMetric::Cosine
            | crate::embedding::SimilarityMetric::Dot
            | crate::embedding::SimilarityMetric::Manhattan
            | crate::embedding::SimilarityMetric::Chebyshev => Self::Cosine,
        }
    }
}

impl VectorDistanceMetric {
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn try_from_core_metric(
        metric: crate::core::Metric,
    ) -> Result<Self, crate::errors::DripstoneError> {
        match metric {
            crate::core::Metric::Cosine => Ok(Self::Cosine),
            crate::core::Metric::Euclidean => Ok(Self::Euclidean),
            crate::core::Metric::DotProduct | crate::core::Metric::Manhattan => {
                Err(crate::errors::DripstoneError::configuration(format!(
                    "S3 Vector Buckets do not support {metric:?}; use Cosine or Euclidean"
                )))
            }
        }
    }
}

impl From<crate::core::Metric> for VectorDistanceMetric {
    fn from(metric: crate::core::Metric) -> Self {
        match metric {
            crate::core::Metric::Cosine | crate::core::Metric::DotProduct => Self::Cosine,
            crate::core::Metric::Euclidean | crate::core::Metric::Manhattan => Self::Euclidean,
        }
    }
}

/// Vector data in float32 format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorData {
    pub float32: Vec<f32>,
}

/// A single vector to insert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorInsert {
    pub key: String,
    pub data: VectorData,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Request body for `PutVectors` API (`POST /PutVectors`).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PutVectorsRequest {
    vector_bucket_name: String,
    index_name: String,
    vectors: Vec<VectorInsert>,
}

/// Query vector request
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct QueryVectorsRequest {
    vector_bucket_name: String,
    index_name: String,
    query_vector: VectorData,
    top_k: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    filter: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_distance: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_metadata: Option<bool>,
}

/// Query result from S3 Vectors
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryResult {
    pub key: String,
    #[serde(default)]
    pub distance: Option<f32>,
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Response from `QueryVectors` API
#[derive(Debug, Deserialize)]
struct QueryVectorsResponse {
    vectors: Vec<QueryResult>,
}

/// Request to get specific vectors by key (`POST /GetVectors`).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GetVectorsRequest {
    vector_bucket_name: String,
    index_name: String,
    keys: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_data: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_metadata: Option<bool>,
}

/// A retrieved vector
#[derive(Debug, Deserialize)]
pub struct RetrievedVector {
    pub key: String,
    pub data: VectorData,
    #[serde(default)]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Response from `GetVectors` API
#[derive(Debug, Deserialize)]
struct GetVectorsResponse {
    vectors: Vec<RetrievedVector>,
}

/// Request to delete vectors (`POST /DeleteVectors`).
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DeleteVectorsRequest {
    vector_bucket_name: String,
    index_name: String,
    keys: Vec<String>,
}

/// Client for S3 Vector operations
#[derive(Debug)]
pub struct S3VectorClient {
    http_client: Client,
    config: VectorConfig,
    endpoint: String,
    aws_config: SdkConfig,
    s3_client: aws_sdk_s3::Client,
    /// Used when `sidecar_s3_bucket` is `None`.
    local_namespace_keys: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl S3VectorClient {
    /// Create a new S3 Vector client.
    ///
    /// `aws_config` must be the same config used to build `aws_client` so SigV4
    /// signing honors configured static credentials (not only the default chain).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn new(
        config: VectorConfig,
        aws_config: SdkConfig,
        aws_client: aws_sdk_s3::Client,
    ) -> Result<Self, DripstoneError> {
        let http_client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| DripstoneError::storage(format!("Failed to create HTTP client: {e}")))?;

        // Standard AWS partition endpoint (matches aws-sdk-s3vectors dual-stack resolver).
        // Not `*.amazonaws.com` — that host does not exist for this service.
        let endpoint = std::env::var("DRIPSTONE_S3_VECTORS_ENDPOINT")
            .unwrap_or_else(|_| format!("https://s3vectors.{}.api.aws", config.region));

        Ok(Self {
            http_client,
            config,
            endpoint,
            aws_config,
            s3_client: aws_client,
            local_namespace_keys: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Full URL for a control-plane operation (e.g. `PutVectors` -> `.../PutVectors`).
    fn operation_url(&self, operation: &str) -> String {
        let base = self.endpoint.trim_end_matches('/');
        let op = operation.trim_start_matches('/');
        format!("{base}/{op}")
    }

    /// Sign and send a request to S3 Vectors API using AWS `SigV4`; parse JSON body on success.
    async fn send_signed_request<B: Serialize + Sync, R: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        url: &str,
        body: &B,
    ) -> Result<R, DripstoneError> {
        let body_json = serde_json::to_string(body)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize request: {e}")))?;
        let signed = sign_json_http_request(
            method,
            url,
            &self.config.region,
            "s3vectors",
            &self.aws_config,
            &[("content-type", "application/json")],
            &body_json,
        )
        .await?;
        send_signed_json::<R>(&self.http_client, signed, false)
            .await?
            .ok_or_else(|| DripstoneError::storage(format!("Unexpected empty response for {url}")))
    }

    /// Same as [`Self::send_signed_request`] but successful responses have an empty body per AWS API.
    async fn send_signed_request_empty_body<B: Serialize + Sync>(
        &self,
        method: &str,
        url: &str,
        body: &B,
    ) -> Result<(), DripstoneError> {
        let body_json = serde_json::to_string(body)
            .map_err(|e| DripstoneError::storage(format!("Failed to serialize request: {e}")))?;
        let signed = sign_json_http_request(
            method,
            url,
            &self.config.region,
            "s3vectors",
            &self.aws_config,
            &[("content-type", "application/json")],
            &body_json,
        )
        .await?;
        let _: Option<serde_json::Value> =
            send_signed_json(&self.http_client, signed, true).await?;
        Ok(())
    }

    /// Insert vectors into the index
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn put_vectors(&self, vectors: Vec<VectorInsert>) -> Result<(), DripstoneError> {
        if vectors.is_empty() {
            return Ok(());
        }

        let url = self.operation_url("PutVectors");
        let request = PutVectorsRequest {
            vector_bucket_name: self.config.bucket_name.clone(),
            index_name: self.config.index_name.clone(),
            vectors,
        };

        self.send_signed_request_empty_body("POST", &url, &request)
            .await
    }

    /// Query vectors for similarity search
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn query_vectors(
        &self,
        query_vector: Vec<f32>,
        top_k: usize,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<QueryResult>, DripstoneError> {
        let url = self.operation_url("QueryVectors");

        let request = QueryVectorsRequest {
            vector_bucket_name: self.config.bucket_name.clone(),
            index_name: self.config.index_name.clone(),
            query_vector: VectorData {
                float32: query_vector,
            },
            top_k,
            filter,
            return_distance: Some(true),
            return_metadata: Some(true),
        };

        let response: QueryVectorsResponse =
            self.send_signed_request("POST", &url, &request).await?;

        Ok(response.vectors)
    }

    /// Get specific vectors by keys
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn get_vectors(
        &self,
        keys: Vec<String>,
    ) -> Result<Vec<RetrievedVector>, DripstoneError> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let url = self.operation_url("GetVectors");

        let request = GetVectorsRequest {
            vector_bucket_name: self.config.bucket_name.clone(),
            index_name: self.config.index_name.clone(),
            keys,
            return_data: Some(true),
            return_metadata: Some(true),
        };

        let response: GetVectorsResponse = self.send_signed_request("POST", &url, &request).await?;

        Ok(response.vectors)
    }

    /// Delete vectors by keys
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn delete_vectors(&self, keys: Vec<String>) -> Result<(), DripstoneError> {
        if keys.is_empty() {
            return Ok(());
        }

        let url = self.operation_url("DeleteVectors");

        let request = DeleteVectorsRequest {
            vector_bucket_name: self.config.bucket_name.clone(),
            index_name: self.config.index_name.clone(),
            keys,
        };

        self.send_signed_request_empty_body("POST", &url, &request)
            .await
    }

    fn index_prefix(&self) -> String {
        format!("__dripstone_indexes/{}/", self.config.index_name)
    }

    fn index_key(&self, namespace: &str) -> String {
        format!("{}{}{}", self.index_prefix(), namespace, "/_keys.json")
    }

    async fn read_namespace_index(&self, namespace: &str) -> Result<Vec<String>, DripstoneError> {
        let key = self.index_key(namespace);

        if let Some(ref bucket) = self.config.sidecar_s3_bucket {
            let result = self
                .s3_client
                .get_object()
                .bucket(bucket)
                .key(&key)
                .send()
                .await;

            match result {
                Ok(output) => {
                    let bytes = output
                        .body
                        .collect()
                        .await
                        .map_err(|e| {
                            DripstoneError::storage(format!(
                                "Failed to read namespace index body from S3: {e}"
                            ))
                        })?
                        .into_bytes();

                    let parsed: Vec<String> = serde_json::from_slice(&bytes).map_err(|e| {
                        DripstoneError::storage(format!(
                            "Failed to parse namespace index JSON: {e}"
                        ))
                    })?;
                    Ok(parsed)
                }
                Err(e) => {
                    if crate::storage::s3_store::sdk_error_is_missing_object(&e) {
                        Ok(Vec::new())
                    } else {
                        Err(DripstoneError::storage(format!(
                            "Failed to read namespace index from S3 (bucket={bucket} key={key}): {e:?}"
                        )))
                    }
                }
            }
        } else {
            Ok(self
                .local_namespace_keys
                .read()
                .get(namespace)
                .cloned()
                .unwrap_or_default())
        }
    }

    async fn write_namespace_index(
        &self,
        namespace: &str,
        keys: &[String],
    ) -> Result<(), DripstoneError> {
        let key = self.index_key(namespace);
        let body = serde_json::to_vec(keys).map_err(|e| {
            DripstoneError::storage(format!("Failed to serialize namespace index: {e}"))
        })?;

        if let Some(ref bucket) = self.config.sidecar_s3_bucket {
            self.s3_client
                .put_object()
                .bucket(bucket)
                .key(&key)
                .content_type("application/json")
                .body(Bytes::from(body).into())
                .send()
                .await
                .map_err(|e| {
                    DripstoneError::storage(format!(
                        "Failed to write namespace index to S3 (bucket={bucket} key={key}): {e:?}"
                    ))
                })?;
        } else {
            self.local_namespace_keys
                .write()
                .insert(namespace.to_string(), keys.to_vec());
        }

        Ok(())
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn delete_namespace_index(&self, namespace: &str) -> Result<(), DripstoneError> {
        let key = self.index_key(namespace);

        if let Some(ref bucket) = self.config.sidecar_s3_bucket {
            let result = self
                .s3_client
                .delete_object()
                .bucket(bucket)
                .key(&key)
                .send()
                .await;

            match result {
                Ok(_) => Ok(()),
                Err(e) if crate::storage::s3_store::sdk_error_is_missing_object(&e) => Ok(()),
                Err(e) => Err(DripstoneError::storage(format!(
                    "Failed to delete namespace index from S3 (bucket={bucket} key={key}): {e:?}"
                ))),
            }
        } else {
            self.local_namespace_keys.write().remove(namespace);
            Ok(())
        }
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn list_vectors_in_namespace(
        &self,
        namespace: &str,
    ) -> Result<Vec<String>, DripstoneError> {
        self.read_namespace_index(namespace).await
    }

    /// List all vector keys across namespaces (best-effort).
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn list_vectors(&self) -> Result<Vec<String>, DripstoneError> {
        let Some(bucket) = self.config.sidecar_s3_bucket.as_ref() else {
            let mut keys: Vec<String> = {
                let map = self.local_namespace_keys.read();
                map.values().flatten().cloned().collect()
            };
            keys.sort();
            keys.dedup();
            return Ok(keys);
        };

        let prefix = self.index_prefix();

        let mut continuation_token: Option<String> = None;
        let mut keys = Vec::new();

        loop {
            let mut request = self
                .s3_client
                .list_objects_v2()
                .bucket(bucket)
                .prefix(&prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request.send().await.map_err(|e| {
                DripstoneError::storage(format!(
                    "Failed to list namespace index objects (bucket={bucket} prefix={prefix}): {e:?}"
                ))
            })?;

            for obj in response.contents() {
                if let Some(key) = obj.key() {
                    // Only load our index files
                    if key.ends_with("/_keys.json") {
                        // Extract namespace segment: __dripstone_indexes/<index_name>/<ns>/_keys.json
                        let ns = key
                            .strip_prefix(&prefix)
                            .and_then(|rest| rest.split('/').next())
                            .unwrap_or("default");
                        let mut ns_keys = self.read_namespace_index(ns).await?;
                        keys.append(&mut ns_keys);
                    }
                }
            }

            if response.is_truncated().unwrap_or(false) {
                continuation_token = response
                    .next_continuation_token()
                    .map(std::string::ToString::to_string);
                if continuation_token.is_none() {
                    break;
                }
            } else {
                break;
            }
        }

        Ok(keys)
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn list_namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        let Some(bucket) = self.config.sidecar_s3_bucket.as_ref() else {
            let mut namespaces: Vec<String> =
                self.local_namespace_keys.read().keys().cloned().collect();
            namespaces.sort();
            namespaces.dedup();
            return Ok(namespaces);
        };

        let prefix = self.index_prefix();

        let mut continuation_token: Option<String> = None;
        let mut namespaces = Vec::new();

        loop {
            let mut request = self
                .s3_client
                .list_objects_v2()
                .bucket(bucket)
                .prefix(&prefix);

            if let Some(token) = continuation_token.take() {
                request = request.continuation_token(token);
            }

            let response = request.send().await.map_err(|e| {
                DripstoneError::storage(format!(
                    "Failed to list namespace index objects (bucket={bucket} prefix={prefix}): {e:?}"
                ))
            })?;

            for obj in response.contents() {
                if let Some(key) = obj.key()
                    && key.ends_with("/_keys.json")
                    && let Some(ns) = key.strip_prefix(&prefix).and_then(|rest| {
                        let mut it = rest.split('/');
                        it.next()
                    })
                {
                    namespaces.push(ns.to_string());
                }
            }

            if response.is_truncated().unwrap_or(false) {
                continuation_token = response
                    .next_continuation_token()
                    .map(std::string::ToString::to_string);
                if continuation_token.is_none() {
                    break;
                }
            } else {
                break;
            }
        }

        namespaces.sort();
        namespaces.dedup();
        Ok(namespaces)
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn index_add_keys(
        &self,
        namespace: &str,
        new_keys: impl IntoIterator<Item = String>,
    ) -> Result<(), DripstoneError> {
        let mut keys = self.read_namespace_index(namespace).await?;
        keys.extend(new_keys);
        keys.sort();
        keys.dedup();
        self.write_namespace_index(namespace, &keys).await
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn index_remove_keys(
        &self,
        namespace: &str,
        remove_keys: impl IntoIterator<Item = String>,
    ) -> Result<(), DripstoneError> {
        let mut keys = self.read_namespace_index(namespace).await?;
        let remove: std::collections::HashSet<String> = remove_keys.into_iter().collect();
        keys.retain(|k| !remove.contains(k));
        self.write_namespace_index(namespace, &keys).await
    }
}
