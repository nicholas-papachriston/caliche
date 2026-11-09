//! `plugins` API client methods.

use anyhow::Result;
use caliche_contracts::rpc::methods as rpc_methods;

use super::ApiClient;
use crate::api::types::{
    McpConfigResponse, McpConfigTestResponse, McpServerConfig, PluginCatalogResponse,
    PluginExportResponse, PluginListResponse, PluginSaveResponse, PluginValidationResponse,
};

impl ApiClient {
    pub async fn get_plugins(&self) -> Result<PluginListResponse> {
        self.rpc_call("plugins.list", &serde_json::json!({})).await
    }

    pub async fn get_plugin_catalog(&self) -> Result<PluginCatalogResponse> {
        self.rpc_call("plugins.getCatalog", &serde_json::json!({}))
            .await
    }

    pub async fn get_plugin(&self, name: &str) -> Result<serde_json::Value> {
        self.rpc_call("plugins.get", &serde_json::json!({ "name": name }))
            .await
    }

    pub async fn validate_plugin(
        &self,
        definition: serde_json::Value,
    ) -> Result<PluginValidationResponse> {
        self.rpc_call(
            "plugins.validate",
            &serde_json::json!({ "definition": definition }),
        )
        .await
    }

    pub async fn save_plugin(&self, definition: serde_json::Value) -> Result<PluginSaveResponse> {
        self.rpc_call(
            "plugins.save",
            &serde_json::json!({ "definition": definition }),
        )
        .await
    }

    pub async fn delete_plugin(&self, name: &str) -> Result<()> {
        let _resp: serde_json::Value = self
            .rpc_call("plugins.delete", &serde_json::json!({ "name": name }))
            .await?;
        Ok(())
    }

    pub async fn export_plugin(&self, name: &str) -> Result<PluginExportResponse> {
        self.rpc_call("plugins.export", &serde_json::json!({ "name": name }))
            .await
    }

    pub async fn get_mcp_config(&self) -> Result<McpConfigResponse> {
        self.rpc_call(rpc_methods::MCP_GET_CONFIG, &serde_json::json!({}))
            .await
    }

    pub async fn put_mcp_config(
        &self,
        servers: Vec<McpServerConfig>,
        allow_empty: bool,
    ) -> Result<McpConfigResponse> {
        let mut body = serde_json::json!({ "servers": servers });
        if allow_empty {
            body["allow_empty"] = serde_json::Value::Bool(true);
        }
        self.rpc_call(rpc_methods::MCP_UPDATE_CONFIG, &body).await
    }

    pub async fn test_mcp_server(&self, server: McpServerConfig) -> Result<McpConfigTestResponse> {
        self.rpc_call(rpc_methods::MCP_TEST_CONFIG, &server).await
    }
}
