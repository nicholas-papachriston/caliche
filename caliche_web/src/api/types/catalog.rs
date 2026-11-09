//! Model catalog, provider, tool, settings, and MCP configuration types.

pub use caliche_contracts::mcp::{
    McpConfigResponseDto as McpConfigResponse, McpConfigTestResponseDto as McpConfigTestResponse,
    McpServerConfigDto as McpServerConfig,
};
pub use caliche_contracts::models::{
    ModelCatalogEntryDto as ModelInfo, ModelsCatalogResponseDto as ModelsCatalogResponse,
    ProviderCatalogEntryDto as ProviderCatalogEntry,
};
pub use caliche_contracts::settings::{
    AgentSettingsDto as AgentSettings, InteractionModeDto as InteractionMode,
};
pub use caliche_contracts::tools::{ToolInfoDto as ToolInfo, ToolsResponseDto as ToolsResponse};
