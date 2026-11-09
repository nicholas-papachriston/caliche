//! `catalog` API client methods.

use anyhow::Result;
use caliche_contracts::rpc::methods as rpc_methods;

use super::ApiClient;
use crate::api::types::{
    AgentSettings, CacheMetricsResponse, FileBrowsingStatus, ListDirResponse,
    ModelsCatalogResponse, ReadFileResponse, ToolInfo, ToolMetricsResponse, ToolsResponse,
};

impl ApiClient {
    /// Get available tools
    pub async fn get_tools(&self) -> Result<Vec<ToolInfo>> {
        let tools_response: ToolsResponse = self
            .rpc_call(rpc_methods::TOOLS_LIST, &serde_json::json!({}))
            .await?;
        Ok(tools_response.tools)
    }

    /// Get available models
    pub async fn get_model_catalog(&self) -> Result<ModelsCatalogResponse> {
        self.rpc_call(rpc_methods::MODELS_LIST, &serde_json::json!({}))
            .await
    }

    /// Fetch current agent settings via JSON-RPC.
    /// Returns an error when the server is in standalone mode (no shared service).
    pub async fn get_settings(&self) -> Result<AgentSettings> {
        self.rpc_call(rpc_methods::SETTINGS_GET, &serde_json::json!({}))
            .await
    }

    /// Apply settings via JSON-RPC.
    /// Returns the updated settings as confirmed by the server.
    pub async fn put_settings(&self, settings: &AgentSettings) -> Result<AgentSettings> {
        // Server-side ProviderType deserialization expects lowercase variants
        // (e.g. "openrouter"), while UI provider labels may be title-cased.
        let mut normalized = settings.clone();
        normalized.provider = normalized.provider.trim().to_ascii_lowercase();

        self.rpc_call(rpc_methods::SETTINGS_UPDATE, &normalized)
            .await
    }

    // ========================================================================
    // File Browser Methods
    // ========================================================================

    /// Check if file browsing is enabled on the server
    pub async fn is_file_browsing_enabled(&self) -> Result<FileBrowsingStatus> {
        self.rpc_call("files.enabled", &serde_json::json!({})).await
    }

    /// List directory contents
    pub async fn list_directory(&self, path: Option<&str>) -> Result<ListDirResponse> {
        self.rpc_call("files.list", &serde_json::json!({ "path": path }))
            .await
    }

    /// Read file contents
    pub async fn read_file(
        &self,
        path: &str,
        max_lines: Option<usize>,
    ) -> Result<ReadFileResponse> {
        self.rpc_call(
            "files.read",
            &serde_json::json!({ "path": path, "max_lines": max_lines }),
        )
        .await
    }
    /// Fetch tool metrics via JSON-RPC.
    pub async fn get_tool_metrics(&self) -> Result<ToolMetricsResponse> {
        self.rpc_call("metrics.listTools", &serde_json::json!({}))
            .await
    }
    // -------------------------------------------------------------------------
    // Cache Metrics
    // -------------------------------------------------------------------------

    pub async fn get_cache_metrics(&self) -> Result<CacheMetricsResponse> {
        self.rpc_call("metrics.getCache", &serde_json::json!({}))
            .await
    }
}
