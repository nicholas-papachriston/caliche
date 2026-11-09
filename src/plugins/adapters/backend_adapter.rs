//! Schema-defined vector backend adapter: implements dripstone::Backend via HTTP endpoint mappings.
//!
//! Supports upsert, search, delete, and count operations via configured endpoint mappings.
//! Operations not covered by the schema (get, list, namespaces, delete_document,
//! delete_namespace) return explicit "not supported" errors.
//!
//! Gated on `vector-memory` so dripstone types are available.

#[cfg(feature = "vector-memory")]
mod impl_ {
    use crate::plugins::schema::{BackendPluginSpec, ConnectionSpec, PluginDefinition, PluginSpec};
    use async_trait::async_trait;
    use dripstone::Backend;
    use dripstone::core::{Capabilities, Chunk, SearchQuery, SearchResult};
    use dripstone::errors::DripstoneError;
    use std::sync::Arc;

    #[derive(Debug)]
    pub struct SchemaBackendAdapter {
        pub definition: Arc<PluginDefinition>,
        pub spec: BackendPluginSpec,
        #[expect(
            dead_code,
            reason = "constructor retains the shared client; requests use a DNS-pinned client"
        )]
        http_client: Arc<reqwest::Client>,
        base_url: String,
    }

    impl SchemaBackendAdapter {
        pub fn new(
            definition: Arc<PluginDefinition>,
            http_client: Arc<reqwest::Client>,
        ) -> Option<Self> {
            let spec = match &definition.spec {
                PluginSpec::Backend(s) => s.clone(),
                _ => return None,
            };
            let base_url = match &spec.connection {
                ConnectionSpec::Inline { url } => url.trim_end_matches('/').to_string(),
                ConnectionSpec::Named { .. } => return None,
            };
            Some(Self {
                definition,
                spec,
                http_client,
                base_url,
            })
        }

        fn to_capabilities(c: &crate::plugins::schema::BackendCapabilities) -> Capabilities {
            Capabilities {
                native_hybrid: c.native_hybrid,
                native_full_text: c.native_full_text,
                native_hybrid_full_text: c.native_hybrid_full_text,
                native_filter: c.native_filter,
                supports_namespaces: c.supports_namespaces,
                native_maxsim: c.native_maxsim,
                max_dimension: c.max_dimension,
                max_batch: c.max_batch,
                supports_streaming: c.supports_streaming,
                requires_exact_filter_pass: false,
            }
        }

        fn endpoint_url(&self, path: &str) -> String {
            format!("{}/{}", self.base_url, path.trim_start_matches('/'))
        }

        fn storage_err(msg: impl Into<String>) -> DripstoneError {
            DripstoneError::storage(msg.into())
        }

        fn map_http_err(e: reqwest::Error) -> DripstoneError {
            DripstoneError::storage(format!("http request failed: {e}"))
        }

        fn map_json_err(e: serde_json::Error) -> DripstoneError {
            DripstoneError::storage(format!("json error: {e}"))
        }

        async fn post_json(
            &self,
            path: &str,
            body: &serde_json::Value,
        ) -> Result<serde_json::Value, DripstoneError> {
            let url = self.endpoint_url(path);
            let resp = crate::plugins::adapters::pinned_request("POST", &url)
                .map_err(|e| Self::storage_err(e.to_string()))?
                .json(body)
                .send()
                .await
                .map_err(Self::map_http_err)?;
            let status = resp.status();
            let text = resp.text().await.map_err(Self::map_http_err)?;
            if !status.is_success() {
                return Err(Self::storage_err(format!(
                    "backend http error {status}: {text}"
                )));
            }
            serde_json::from_str(&text).map_err(Self::map_json_err)
        }

        async fn delete_json(
            &self,
            path: &str,
            body: &serde_json::Value,
        ) -> Result<(), DripstoneError> {
            let url = self.endpoint_url(path);
            let resp = crate::plugins::adapters::pinned_request("DELETE", &url)
                .map_err(|e| Self::storage_err(e.to_string()))?
                .json(body)
                .send()
                .await
                .map_err(Self::map_http_err)?;
            let status = resp.status();
            if !status.is_success() {
                let text = resp.text().await.map_err(Self::map_http_err)?;
                return Err(Self::storage_err(format!(
                    "backend delete http error {status}: {text}"
                )));
            }
            Ok(())
        }

        async fn get_json(
            &self,
            path: &str,
            params: &[(&str, String)],
        ) -> Result<serde_json::Value, DripstoneError> {
            let url = self.endpoint_url(path);
            let resp = crate::plugins::adapters::pinned_request("GET", &url)
                .map_err(|e| Self::storage_err(e.to_string()))?
                .query(params)
                .send()
                .await
                .map_err(Self::map_http_err)?;
            let status = resp.status();
            let text = resp.text().await.map_err(Self::map_http_err)?;
            if !status.is_success() {
                return Err(Self::storage_err(format!(
                    "backend http error {status}: {text}"
                )));
            }
            serde_json::from_str(&text).map_err(Self::map_json_err)
        }

        fn extract_search_results(
            &self,
            value: &serde_json::Value,
        ) -> Result<Vec<SearchResult>, DripstoneError> {
            let id_path = self.spec.result_mapping.id_path.as_deref().unwrap_or("id");
            let score_path = self
                .spec
                .result_mapping
                .score_path
                .as_deref()
                .unwrap_or("score");
            let content_path = self
                .spec
                .result_mapping
                .content_path
                .as_deref()
                .unwrap_or("content");
            let metadata_path = self
                .spec
                .result_mapping
                .metadata_path
                .as_deref()
                .unwrap_or("metadata");

            let arr = value
                .as_array()
                .ok_or_else(|| Self::storage_err("search response must be a JSON array"))?;

            let mut results = Vec::with_capacity(arr.len());
            for (rank, item) in arr.iter().enumerate() {
                let id = item
                    .get(id_path)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let score = item.get(score_path).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let content = item
                    .get(content_path)
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string();
                let metadata = item
                    .get(metadata_path)
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);

                // Derive a stable doc_id from the chunk ID (strip chunk index suffix if present)
                let doc_id = id
                    .rsplit_once('#')
                    .map(|(d, _)| d)
                    .unwrap_or(&id)
                    .to_string();

                let chunk = Chunk {
                    id: id.clone(),
                    doc_id,
                    content,
                    vector: std::sync::Arc::from([]),
                    start_offset: 0,
                    end_offset: 0,
                    chunk_index: rank,
                    total_chunks: arr.len(),
                    metadata,
                    namespace: None,
                    multi_vectors: None,
                };

                results.push(SearchResult::new(chunk, score, rank));
            }
            Ok(results)
        }
    }

    #[async_trait]
    impl Backend for SchemaBackendAdapter {
        fn capabilities(&self) -> Capabilities {
            Self::to_capabilities(&self.spec.capabilities)
        }

        async fn upsert(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
            let ep = self.spec.endpoints.upsert.as_ref().ok_or_else(|| {
                Self::storage_err("schema backend upsert endpoint not configured")
            })?;
            let body = serde_json::to_value(chunks).map_err(Self::map_json_err)?;
            self.post_json(&ep.path, &body).await?;
            Ok(())
        }

        async fn get(
            &self,
            _id: &str,
            _namespace: Option<&str>,
        ) -> Result<Option<Chunk>, DripstoneError> {
            Err(Self::storage_err(
                "schema backend does not support get by ID; use search instead",
            ))
        }

        async fn delete(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
            let ep = self.spec.endpoints.delete.as_ref().ok_or_else(|| {
                Self::storage_err("schema backend delete endpoint not configured")
            })?;
            let mut body = serde_json::json!({ "id": id });
            if let Some(ns) = namespace {
                body["namespace"] = serde_json::json!(ns);
            }
            self.delete_json(&ep.path, &body).await
        }

        async fn delete_document(
            &self,
            _doc_id: &str,
            _namespace: Option<&str>,
        ) -> Result<usize, DripstoneError> {
            Err(Self::storage_err(
                "schema backend does not support delete_document; use delete per chunk ID",
            ))
        }

        async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
            let ep = self.spec.endpoints.search.as_ref().ok_or_else(|| {
                Self::storage_err("schema backend search endpoint not configured")
            })?;
            let body = serde_json::json!({
                "vector": query.vector.as_ref(),
                "limit": query.limit,
                "namespace": query.namespace,
            });
            let response = self.post_json(&ep.path, &body).await?;
            let results_value = response.get("results").unwrap_or(&response);
            self.extract_search_results(results_value)
        }

        async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
            Err(Self::storage_err(
                "schema backend does not support namespace listing",
            ))
        }

        async fn delete_namespace(&self, _namespace: &str) -> Result<(), DripstoneError> {
            Err(Self::storage_err(
                "schema backend does not support delete_namespace",
            ))
        }

        async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
            let ep =
                self.spec.endpoints.count.as_ref().ok_or_else(|| {
                    Self::storage_err("schema backend count endpoint not configured")
                })?;
            let params: Vec<(&str, String)> = namespace
                .map(|ns| vec![("namespace", ns.to_string())])
                .unwrap_or_default();
            let response = self.get_json(&ep.path, &params).await?;
            let count = response
                .get("count")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Self::storage_err("count response missing 'count' field"))?;
            Ok(count as usize)
        }

        async fn list(
            &self,
            _namespace: Option<&str>,
            _limit: Option<usize>,
        ) -> Result<Vec<Chunk>, DripstoneError> {
            Err(Self::storage_err(
                "schema backend does not support list; use search with a broad query",
            ))
        }
    }
}

#[cfg(feature = "vector-memory")]
pub use impl_::SchemaBackendAdapter;
