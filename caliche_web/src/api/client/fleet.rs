//! `fleet` API client methods.

use anyhow::{Context, Result};
use caliche_contracts::rpc::methods as rpc_methods;
use reqwest::Client;

use super::ApiClient;
use crate::api::types::{
    ChatRequest, ChatResponse, CreateEndpointRequest, EndpointInfo, EndpointListResponse,
    ToolMetricsResponse,
};

impl ApiClient {
    // ── Endpoint management ───────────────────────────────────────────────────

    /// List all managed endpoints.
    pub async fn list_endpoints(&self) -> Result<Vec<EndpointInfo>> {
        let resp: EndpointListResponse = self
            .rpc_call("endpoints.list", &serde_json::json!({}))
            .await?;
        Ok(resp.endpoints)
    }

    /// Create (and optionally start) a new managed endpoint.
    pub async fn create_endpoint(&self, req: CreateEndpointRequest) -> Result<EndpointInfo> {
        self.rpc_call("endpoints.create", &req).await
    }

    /// Start a stopped endpoint.
    pub async fn start_endpoint(&self, id: &str) -> Result<EndpointInfo> {
        self.rpc_call("endpoints.start", &serde_json::json!({ "id": id }))
            .await
    }

    /// Stop a running endpoint gracefully.
    pub async fn stop_endpoint(&self, id: &str) -> Result<EndpointInfo> {
        self.rpc_call("endpoints.stop", &serde_json::json!({ "id": id }))
            .await
    }

    /// Delete an endpoint (stops it first if running).
    pub async fn delete_endpoint(&self, id: &str) -> Result<()> {
        let _: serde_json::Value = self
            .rpc_call("endpoints.delete", &serde_json::json!({ "id": id }))
            .await?;
        Ok(())
    }

    // ── Fleet broadcast / inline chat ─────────────────────────────────────────

    /// Create a client that targets a different endpoint URL (e.g. a fleet child server).
    pub fn for_endpoint(url: impl Into<String>) -> Self {
        Self::new(url)
    }

    /// Send a single non-streaming chat message to the server at this client's
    /// base URL and return the assistant's response text.
    ///
    /// Used for fleet broadcast and per-endpoint inline chat.  Works in both
    /// native and WASM builds because it uses a plain HTTP POST (no SSE).
    pub async fn chat_simple(
        &self,
        message: &str,
        provider: Option<String>,
        model: Option<String>,
    ) -> Result<String> {
        let req = ChatRequest {
            message: message.to_string(),
            provider,
            model,
            context_window: None,
            tools: None,
            messages: None,
            mode: None,
        };
        let resp: ChatResponse = self.rpc_call(rpc_methods::CHAT_SEND, &req).await?;
        Ok(resp.response)
    }
    // ── Fleet health & metrics ───────────────────────────────────────────────

    /// Ping an arbitrary endpoint URL (GET /health) and return latency in ms.
    /// Uses a fresh client targeting that URL rather than self.base_url.
    pub async fn ping_url(url: &str) -> Option<u64> {
        let client = Client::new();
        let target = format!("{}/health", url.trim_end_matches('/'));
        let start = std::time::Instant::now();
        match client.get(&target).send().await {
            Ok(resp) if resp.status().is_success() => Some(start.elapsed().as_millis() as u64),
            _ => None,
        }
    }

    /// Fetch tool metrics from an arbitrary running endpoint URL.
    pub async fn endpoint_metrics(url: &str) -> Result<ToolMetricsResponse> {
        let client = Client::new();
        let target = format!("{}/rpc", url.trim_end_matches('/'));
        let rpc_request = serde_json::json!({
            "jsonrpc": caliche_contracts::rpc::JSONRPC_VERSION,
            "id": 1,
            "method": "metrics.listTools",
            "params": {}
        });
        let response = client
            .post(&target)
            .json(&rpc_request)
            .send()
            .await
            .context("Failed to fetch endpoint metrics")?;
        if !response.status().is_success() {
            anyhow::bail!("Endpoint metrics returned {}", response.status());
        }
        let rpc_response: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse endpoint metrics RPC response")?;
        if let Some(err) = rpc_response.get("error") {
            anyhow::bail!("Endpoint metrics RPC error: {}", err);
        }
        let result = rpc_response
            .get("result")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Missing result in endpoint metrics RPC response"))?;
        serde_json::from_value(result).context("Failed to parse endpoint metrics")
    }
}
