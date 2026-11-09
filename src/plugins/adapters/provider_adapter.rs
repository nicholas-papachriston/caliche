//! Schema-defined LLM provider adapter: implements Provider via HTTP and request/response mapping.

use crate::llm::provider::Provider;
use crate::messages::{CompletionContent, CompletionRequest, CompletionResponse, StreamEvent};
use crate::plugins::schema::{PluginDefinition, ProviderPluginSpec};
use anyhow::anyhow;
use async_stream::stream;
use async_trait::async_trait;
use std::sync::Arc;

#[derive(Debug)]
pub struct SchemaProviderAdapter {
    definition: Arc<PluginDefinition>,
    spec: ProviderPluginSpec,
    #[expect(
        dead_code,
        reason = "constructor retains the shared client; requests use a DNS-pinned client"
    )]
    http_client: Arc<reqwest::Client>,
}

impl SchemaProviderAdapter {
    pub fn new(
        definition: Arc<PluginDefinition>,
        http_client: Arc<reqwest::Client>,
    ) -> Option<Self> {
        let spec = match &definition.spec {
            crate::plugins::schema::PluginSpec::Provider(s) => s.clone(),
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
impl Provider for SchemaProviderAdapter {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        let url = format!(
            "{}/{}",
            self.spec.base_url.trim_end_matches('/'),
            self.spec.completion_endpoint.path.trim_start_matches('/')
        );
        let mut body = serde_json::Map::new();
        if let Some(p) = &self.spec.request_mapping.messages_path {
            body.insert(
                p.clone(),
                serde_json::to_value(request.messages.as_ref())
                    .map_err(|e| anyhow!("serialize messages: {}", e))?,
            );
        }
        if let Some(p) = &self.spec.request_mapping.max_tokens_path
            && let Some(v) = request.max_tokens
        {
            body.insert(p.clone(), serde_json::json!(v));
        }
        if let Some(p) = &self.spec.request_mapping.temperature_path
            && let Some(v) = request.temperature
        {
            body.insert(p.clone(), serde_json::json!(v));
        }
        if let Some(p) = &self.spec.request_mapping.tools_path
            && let Some(t) = &request.tools
        {
            body.insert(
                p.clone(),
                serde_json::to_value(t.as_ref()).map_err(|e| anyhow!("serialize tools: {}", e))?,
            );
        }
        let method = self
            .spec
            .completion_endpoint
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
            return Err(anyhow!("provider api error {}: {}", status, text));
        }
        let value: serde_json::Value =
            serde_json::from_str(&text).unwrap_or(serde_json::Value::String(text));
        let content_str =
            crate::plugins::runtime::resolve_path(&value, &self.spec.response_mapping.content_path)
                .map_err(|e| anyhow!("response mapping: {}", e))?
                .as_str()
                .map(std::string::ToString::to_string)
                .unwrap_or_default();
        let content = CompletionContent::Text {
            content: content_str,
        };
        let finish_reason = self
            .spec
            .response_mapping
            .finish_reason_path
            .as_ref()
            .and_then(|p| crate::plugins::runtime::resolve_path(&value, p).ok())
            .and_then(|v| v.as_str().map(std::string::ToString::to_string));
        let usage = self
            .spec
            .response_mapping
            .usage_path
            .as_ref()
            .and_then(|p| crate::plugins::runtime::resolve_path(&value, p).ok())
            .and_then(|v| serde_json::from_value(v).ok());
        Ok(CompletionResponse {
            content,
            finish_reason: finish_reason.map(beef::lean::Cow::owned),
            usage,
            reasoning_content: None,
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> std::pin::Pin<Box<dyn futures_util::Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>>
    {
        tracing::warn!(
            provider = %self.definition.name,
            "schema provider does not support native streaming; falling back to buffered completion"
        );
        Box::pin(stream! {
            let response = match self.complete(request).await {
                Ok(r) => r,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            let text = response.content.extract_text();
            yield Ok(StreamEvent::TextChunk { content: text });
            yield Ok(StreamEvent::Metadata {
                usage: response.usage,
                finish_reason: response.finish_reason,
                metrics: None,
            });
        })
    }

    fn name(&self) -> &str {
        &self.definition.name
    }

    fn supports_native_tools(&self) -> bool {
        self.spec.supports_tools
    }

    fn supports_streaming(&self) -> bool {
        self.spec.streaming
    }

    fn supports_vision(&self) -> bool {
        self.spec.supports_vision
    }
}
