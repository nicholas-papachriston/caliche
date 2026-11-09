use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpServerConfigDto {
    pub name: String,
    pub transport: String,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default = "default_mcp_enabled")]
    pub enabled: bool,
}

const fn default_mcp_enabled() -> bool {
    true
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpConfigResponseDto {
    #[serde(default)]
    pub servers: Vec<McpServerConfigDto>,
    /// When true, an empty `servers` list may replace an existing `mcp.json`.
    ///
    /// Settings Save leaves this false. An empty list then does not wipe a
    /// file that already has servers. A deliberate clear sets it true.
    #[serde(default, skip_serializing_if = "is_false")]
    pub allow_empty: bool,
}

const fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpConfigTestResponseDto {
    pub ok: bool,
    #[serde(default)]
    pub tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
