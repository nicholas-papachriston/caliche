//! Managed endpoint and tool metrics types.

use serde::{Deserialize, Serialize};

/// Per-tool metrics entry
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMetricsDto {
    pub total_calls: u64,
    pub success_rate: f64,
    pub avg_duration_ms: f64,
    #[serde(default)]
    pub p95_duration_ms: Option<u64>,
    #[serde(default)]
    pub error_types: std::collections::HashMap<String, u64>,
    #[serde(default)]
    pub last_executed: Option<String>,
}

/// Tool metrics response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolMetricsResponse {
    pub tools: std::collections::HashMap<String, ToolMetricsDto>,
}

/// Request to create and optionally start a managed endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEndpointRequest {
    pub name: String,
    pub workspace: String,
    pub port: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default)]
    pub auto_start: bool,
}

/// Full endpoint status returned by GET /api/endpoints/:id.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EndpointInfo {
    pub id: String,
    pub name: String,
    pub workspace: String,
    pub port: u16,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub auto_start: bool,
    /// "stopped" | "starting" | "running" | "failed"
    pub status: String,
    pub pid: Option<u32>,
    /// http://localhost:{port}
    pub url: String,
    pub created_at: String,
    pub started_at: Option<String>,
    pub stopped_at: Option<String>,
    pub error: Option<String>,
}

/// Response from GET /api/endpoints.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EndpointListResponse {
    pub endpoints: Vec<EndpointInfo>,
}
