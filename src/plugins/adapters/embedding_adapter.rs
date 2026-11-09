//! Schema-defined embedding provider adapter: implements EmbeddingProvider via HTTP and mapping.

use crate::llm::EmbeddingProvider;
use crate::llm::embedding::{EmbeddingRequest, EmbeddingResponse};
use crate::plugins::runtime::resolve_path;
use crate::plugins::schema::{EmbeddingPluginSpec, PluginDefinition, PluginSpec};
use anyhow::anyhow;
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug)]
pub struct SchemaEmbeddingAdapter {
    definition: Arc<PluginDefinition>,
    spec: EmbeddingPluginSpec,
    #[expect(
        dead_code,
        reason = "constructor retains the shared client; requests use a DNS-pinned client"
    )]
    http_client: Arc<reqwest::Client>,
}

impl SchemaEmbeddingAdapter {
    pub fn new(
        definition: Arc<PluginDefinition>,
        http_client: Arc<reqwest::Client>,
    ) -> Option<Self> {
        let spec = match &definition.spec {
            PluginSpec::Embedding(s) => s.clone(),
            _ => return None,
        };
        Some(Self {
            definition,
            spec,
            http_client,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for SchemaEmbeddingAdapter {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        let url = format!(
            "{}/{}",
            self.spec.base_url.trim_end_matches('/'),
            self.spec.embed_endpoint.path.trim_start_matches('/')
        );
        let mut body = serde_json::Map::new();
        body.insert(
            self.spec.request_mapping.texts_path.clone(),
            serde_json::to_value(&request.texts).map_err(|e| anyhow!("serialize texts: {}", e))?,
        );
        if let Some(ref p) = self.spec.request_mapping.model_path
            && let Some(ref m) = request.model
        {
            body.insert(p.clone(), serde_json::json!(m));
        }
        let method = self
            .spec
            .embed_endpoint
            .method
            .as_deref()
            .unwrap_or("POST")
            .to_uppercase();
        let req_builder = crate::plugins::adapters::pinned_request(method.as_str(), &url)?;
        let resp = req_builder
            .json(&body)
            .send()
            .await
            .map_err(|e| anyhow!("http request: {}", e))?;
        let status = resp.status();
        let text = resp.text().await.map_err(|e| anyhow!("read body: {}", e))?;
        if !status.is_success() {
            return Err(anyhow!("embedding api error {}: {}", status, text));
        }
        let value: serde_json::Value =
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
        let embeddings_value = resolve_path(&value, &self.spec.response_mapping.embeddings_path)
            .map_err(|e| anyhow!("response mapping embeddings: {}", e))?;
        let embeddings: Vec<Vec<f32>> = serde_json::from_value(embeddings_value)
            .map_err(|e| anyhow!("parse embeddings: {}", e))?;
        let usage = self
            .spec
            .response_mapping
            .usage_path
            .as_ref()
            .and_then(|p| resolve_path(&value, p).ok())
            .and_then(|v| serde_json::from_value(v).ok());
        let model = self
            .spec
            .response_mapping
            .model_path
            .as_ref()
            .and_then(|p| resolve_path(&value, p).ok())
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| self.definition.name.clone());
        Ok(EmbeddingResponse {
            embeddings,
            usage,
            model,
        })
    }

    fn name(&self) -> &str {
        &self.definition.name
    }
}
