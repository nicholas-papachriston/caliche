//! HTTP API handlers for MCP server capabilities discovery.

use crate::AppState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use caliche::mcp::protocol::LATEST_PROTOCOL_VERSION;
use caliche::mcp::server::{McpServer, McpServerBuilder};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct McpCapabilitiesResponse {
    pub server_info: ServerInfo,
    pub protocol_version: String,
    pub capabilities: CapabilitiesInfo,
}

#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CapabilitiesInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompts: Option<PromptsCapability>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourcesCapability>,
}

#[derive(Debug, Serialize)]
pub struct ToolsCapability {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct PromptsCapability {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ResourcesCapability {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct McpToolsResponse {
    pub tools: Vec<McpToolInfo>,
}

#[derive(Debug, Serialize)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct McpPromptsResponse {
    pub prompts: Vec<McpPromptInfo>,
}

#[derive(Debug, Serialize)]
pub struct McpPromptInfo {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Vec<caliche::mcp::protocol::PromptArgument>>,
}

#[derive(Debug, Serialize)]
pub struct McpResourcesResponse {
    pub resources: Vec<McpResourceInfo>,
}

#[derive(Debug, Serialize)]
pub struct McpResourceInfo {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Create a default MCP server instance from AppState
async fn create_mcp_server(state: &AppState) -> Result<McpServer, (StatusCode, String)> {
    use crate::get_or_create_tool_registry;

    let tool_registry = get_or_create_tool_registry(state, state.default_provider, None)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get tool registry: {}", e),
            )
        })?;

    let server = McpServerBuilder::new()
        .name("caliche-mcp-server")
        .version("0.1.0".to_string())
        .enable_tools(true)
        .enable_prompts(false)
        .enable_resources(false)
        .tool_registry(tool_registry)
        .build()
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to build MCP server: {}", e),
            )
        })?;

    Ok(server)
}

/// GET /api/mcp/capabilities - Get MCP server capabilities
pub async fn capabilities_handler(
    State(state): State<AppState>,
) -> Result<Json<McpCapabilitiesResponse>, (StatusCode, String)> {
    let server = create_mcp_server(&state).await?;
    let server_info = server.get_server_info();
    let capabilities = server.get_capabilities();

    Ok(Json(McpCapabilitiesResponse {
        server_info: ServerInfo {
            name: server_info.name.into_owned(),
            version: server_info.version.map(|v| v.into_owned()),
        },
        protocol_version: LATEST_PROTOCOL_VERSION.to_string(),
        capabilities: CapabilitiesInfo {
            tools: capabilities.tools.map(|t| ToolsCapability {
                list_changed: t.list_changed,
            }),
            prompts: capabilities.prompts.map(|p| PromptsCapability {
                list_changed: p.list_changed,
            }),
            resources: capabilities.resources.map(|r| ResourcesCapability {
                subscribe: r.subscribe,
                list_changed: r.list_changed,
            }),
        },
    }))
}

/// GET /api/mcp/tools - List available MCP tools
pub async fn tools_handler(
    State(state): State<AppState>,
) -> Result<Json<McpToolsResponse>, (StatusCode, String)> {
    let server = create_mcp_server(&state).await?;
    let tool_registry = server.tool_registry();

    use caliche::mcp::capabilities::tools::handle_tools_list;

    let tools_result = handle_tools_list(tool_registry, None).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to retrieve tools list: {}", e),
        )
    })?;

    let tools: Vec<McpToolInfo> = tools_result
        .tools
        .into_iter()
        .map(|tool| McpToolInfo {
            name: tool.name.into_owned(),
            description: tool.description.into_owned(),
            input_schema: tool.input_schema,
        })
        .collect();

    Ok(Json(McpToolsResponse { tools }))
}

/// GET /api/mcp/prompts - List available MCP prompts
pub async fn prompts_handler(
    State(state): State<AppState>,
) -> Result<Json<McpPromptsResponse>, (StatusCode, String)> {
    let server = create_mcp_server(&state).await?;
    let template_registry = server.template_registry();

    use caliche::mcp::capabilities::prompts::handle_prompts_list;

    let prompts_result = handle_prompts_list(template_registry, None)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to retrieve prompts list: {}", e),
            )
        })?;

    let prompts: Vec<McpPromptInfo> = prompts_result
        .prompts
        .into_iter()
        .map(|prompt| McpPromptInfo {
            name: prompt.name,
            description: prompt.description,
            arguments: prompt.arguments,
        })
        .collect();

    Ok(Json(McpPromptsResponse { prompts }))
}

/// GET /api/mcp/resources - List available MCP resources
pub async fn resources_handler(
    State(state): State<AppState>,
) -> Result<Json<McpResourcesResponse>, (StatusCode, String)> {
    let server = create_mcp_server(&state).await?;
    let document_collections = server.document_collections();

    use caliche::mcp::capabilities::resources::handle_resources_list;

    let resources_result = handle_resources_list(document_collections, None)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to retrieve resources list: {}", e),
            )
        })?;

    let resources: Vec<McpResourceInfo> = resources_result
        .resources
        .into_iter()
        .map(|resource| McpResourceInfo {
            uri: resource.uri,
            name: resource.name,
            description: resource.description,
            mime_type: resource.mime_type,
        })
        .collect();

    Ok(Json(McpResourcesResponse { resources }))
}
