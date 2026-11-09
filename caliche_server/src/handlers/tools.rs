use crate::{AppState, get_or_create_provider, get_or_create_tool_registry};
use axum::extract::State;
use axum::response::Json;
use caliche_contracts::tools::{ToolInfoDto as ToolInfo, ToolsResponseDto as ToolsResponse};

/// GET /api/tools - List available tools
/// Uses default provider/model to create a service instance for listing tools
#[tracing::instrument(name = "api_tools_list", skip_all)]
pub async fn tools_handler(
    State(state): State<AppState>,
) -> Result<Json<ToolsResponse>, (axum::http::StatusCode, String)> {
    // Get or create cached provider and tool registry using defaults
    let provider =
        get_or_create_provider(&state, state.default_provider, state.default_model.clone())
            .await
            .map_err(|e| {
                (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to get or create provider: {}", e),
                )
            })?;

    let tool_registry = get_or_create_tool_registry(&state, state.default_provider, None)
        .await
        .map_err(|e| {
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to get or create tool registry: {}", e),
            )
        })?;

    // Create fresh service instance to list tools
    let service = caliche::agent::service::AgentService::from_components(
        provider,
        tool_registry,
        state.default_provider,
        state.default_model.clone(),
        None,
    )
    .map_err(|e| {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create agent service: {e}"),
        )
    })?;

    let tools: Vec<ToolInfo> = service
        .list_tools()
        .into_iter()
        .map(|(name, description, params)| ToolInfo {
            name,
            description,
            parameters: params,
        })
        .collect();

    Ok(Json(ToolsResponse { tools }))
}
