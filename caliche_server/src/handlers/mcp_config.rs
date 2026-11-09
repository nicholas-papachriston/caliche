use crate::AppState;
use crate::handlers::mcp_policy::validate_mcp_remote_url;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use caliche::agent::tools::RemoteMcpTool;
use caliche::mcp::{McpClient, McpClientConfig, McpClientTransportConfig};
pub use caliche_contracts::mcp::{
    McpConfigResponseDto as McpConfigResponse, McpConfigTestResponseDto as McpConfigTestResponse,
    McpServerConfigDto as McpServerConfigPayload,
};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

type ApiResult<T> = Result<Json<T>, (StatusCode, String)>;

/// Validate stdio MCP command against `CALICHE_MCP_COMMAND_ALLOWLIST`.
///
/// Bare names match allowlist names. Commands that contain a path separator
/// must be listed exactly; a basename of `npx` does not authorize `/tmp/npx`.
pub fn validate_stdio_command(command: &str) -> anyhow::Result<()> {
    caliche::mcp::validate_stdio_command_required(command)
}

fn to_client_config(server: &McpServerConfigPayload) -> anyhow::Result<McpClientConfig> {
    let transport = match server.transport.as_str() {
        "stdio" => {
            let command = server
                .command
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| anyhow::anyhow!("stdio transport requires command"))?;
            validate_stdio_command(command)?;
            McpClientTransportConfig::Stdio {
                command: command.to_string(),
                args: server.args.clone(),
                env: Vec::new(),
            }
        }
        "streamable-http" | "streamable_http" | "http" => {
            let url = server
                .url
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| anyhow::anyhow!("streamable-http transport requires url"))?;
            validate_mcp_remote_url(url)?;
            McpClientTransportConfig::StreamableHttp {
                url: url.to_string(),
            }
        }
        other => anyhow::bail!("unsupported MCP transport: {other}"),
    };
    Ok(McpClientConfig {
        name: server.name.clone(),
        transport,
        skip_stdio_allowlist: false,
    })
}

pub async fn load_mcp_configs(path: &std::path::Path) -> Vec<McpServerConfigPayload> {
    let Ok(content) = tokio::fs::read_to_string(path).await else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<Value>(&content) else {
        return Vec::new();
    };
    let Some(servers) = value
        .get("mcpServers")
        .or_else(|| value.get("mcp_servers"))
        .or_else(|| value.get("servers"))
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for (name, server) in servers {
        let Some(obj) = server.as_object() else {
            continue;
        };
        if let Some(command) = obj.get("command").and_then(Value::as_str) {
            out.push(McpServerConfigPayload {
                name: name.clone(),
                transport: "stdio".to_string(),
                command: Some(command.to_string()),
                args: obj
                    .get("args")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .filter_map(Value::as_str)
                            .map(ToString::to_string)
                            .collect()
                    })
                    .unwrap_or_default(),
                url: None,
                enabled: obj.get("enabled").and_then(Value::as_bool).unwrap_or(true),
            });
        } else if let Some(url) = obj
            .get("url")
            .or_else(|| obj.get("sse_url"))
            .and_then(Value::as_str)
        {
            out.push(McpServerConfigPayload {
                name: name.clone(),
                transport: "streamable-http".to_string(),
                command: None,
                args: Vec::new(),
                url: Some(url.to_string()),
                enabled: obj.get("enabled").and_then(Value::as_bool).unwrap_or(true),
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

async fn persist_mcp_configs(
    path: &std::path::Path,
    servers: &[McpServerConfigPayload],
) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let mut server_map = serde_json::Map::new();
    for server in servers {
        let entry = if server.transport == "stdio" {
            json!({
                "command": server.command.clone().unwrap_or_default(),
                "args": server.args,
                "enabled": server.enabled,
            })
        } else {
            json!({
                "url": server.url.clone().unwrap_or_default(),
                "enabled": server.enabled,
            })
        };
        server_map.insert(server.name.clone(), entry);
    }

    let payload = json!({ "mcpServers": server_map });
    let content = serde_json::to_string_pretty(&payload)?;
    tokio::fs::write(path, content).await?;
    Ok(())
}

fn remove_existing_remote_mcp_tools(
    service: &mut caliche::AgentService,
    previous: &[McpServerConfigPayload],
) {
    let prefixes: Vec<String> = previous
        .iter()
        .filter(|server| server.enabled)
        .map(|server| format!("{}__", server.name))
        .collect();
    if prefixes.is_empty() {
        return;
    }
    let tool_names: Vec<String> = service
        .list_tools()
        .into_iter()
        .map(|(name, _, _)| name)
        .collect();
    for name in tool_names {
        if prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            service.tool_registry().unregister(&name);
        }
    }
}

async fn apply_live_mcp_configs(
    state: &AppState,
    previous: &[McpServerConfigPayload],
    next: &[McpServerConfigPayload],
) -> anyhow::Result<()> {
    let Some(shared) = state.shared_service.as_ref() else {
        return Ok(());
    };
    let mut service = shared.write().await;
    remove_existing_remote_mcp_tools(&mut service, previous);

    for server in next.iter().filter(|server| server.enabled) {
        let config = to_client_config(server)?;
        let client = Arc::new(McpClient::new(config));
        let tools = tokio::time::timeout(Duration::from_secs(10), client.list_tools()).await??;
        for tool in tools {
            let wrapped_name = format!("{}__{}", server.name, tool.name);
            service.add_tool(Arc::new(RemoteMcpTool::new(
                wrapped_name,
                tool.description.to_string(),
                tool.input_schema,
                tool.name.to_string(),
                Arc::clone(&client),
            )));
        }
    }

    Ok(())
}

pub async fn get_mcp_config_handler(State(state): State<AppState>) -> ApiResult<McpConfigResponse> {
    let servers = state.mcp_servers.read().await.clone();
    Ok(Json(McpConfigResponse {
        servers,
        allow_empty: false,
    }))
}

/// True when an empty MCP payload must leave the existing server list in place.
#[must_use]
pub fn should_keep_existing_mcp(
    incoming_empty: bool,
    allow_empty: bool,
    existing_nonempty: bool,
) -> bool {
    incoming_empty && !allow_empty && existing_nonempty
}

async fn mcp_file_has_servers(path: &std::path::Path) -> bool {
    let Ok(content) = tokio::fs::read_to_string(path).await else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<Value>(&content) else {
        return !content.trim().is_empty();
    };
    value
        .get("mcpServers")
        .or_else(|| value.get("mcp_servers"))
        .or_else(|| value.get("servers"))
        .and_then(Value::as_object)
        .is_some_and(|servers| !servers.is_empty())
}

pub async fn put_mcp_config_handler(
    State(state): State<AppState>,
    Json(payload): Json<McpConfigResponse>,
) -> ApiResult<McpConfigResponse> {
    let existing_nonempty = mcp_file_has_servers(&state.mcp_config_path).await
        || !state.mcp_servers.read().await.is_empty();
    if should_keep_existing_mcp(
        payload.servers.is_empty(),
        payload.allow_empty,
        existing_nonempty,
    ) {
        if state.mcp_servers.read().await.is_empty() {
            let disk = load_mcp_configs(&state.mcp_config_path).await;
            if !disk.is_empty() {
                *state.mcp_servers.write().await = disk;
            }
        }
        let servers = state.mcp_servers.read().await.clone();
        return Ok(Json(McpConfigResponse {
            servers,
            allow_empty: false,
        }));
    }

    for server in &payload.servers {
        if server.name.trim().is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                "MCP server name must be non-empty".to_string(),
            ));
        }
        to_client_config(server).map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
    }

    let previous = state.mcp_servers.read().await.clone();
    apply_live_mcp_configs(&state, &previous, &payload.servers)
        .await
        .map_err(|err| {
            (
                StatusCode::BAD_REQUEST,
                format!("Failed to apply MCP config: {}", err),
            )
        })?;
    persist_mcp_configs(&state.mcp_config_path, &payload.servers)
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to persist MCP config: {}", err),
            )
        })?;
    *state.mcp_servers.write().await = payload.servers.clone();

    Ok(Json(McpConfigResponse {
        servers: payload.servers,
        allow_empty: false,
    }))
}

pub async fn test_mcp_config_handler(
    State(_state): State<AppState>,
    Json(server): Json<McpServerConfigPayload>,
) -> ApiResult<McpConfigTestResponse> {
    if !server.enabled {
        return Ok(Json(McpConfigTestResponse {
            ok: true,
            tools: Vec::new(),
            error: None,
        }));
    }

    let config = to_client_config(&server).map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid MCP config: {}", err),
        )
    })?;
    let client = McpClient::new(config);
    match tokio::time::timeout(Duration::from_secs(10), client.list_tools()).await {
        Ok(Ok(tools)) => Ok(Json(McpConfigTestResponse {
            ok: true,
            tools: tools
                .into_iter()
                .map(|tool| tool.name.to_string())
                .collect(),
            error: None,
        })),
        Ok(Err(err)) => Ok(Json(McpConfigTestResponse {
            ok: false,
            tools: Vec::new(),
            error: Some(err.to_string()),
        })),
        Err(_) => Ok(Json(McpConfigTestResponse {
            ok: false,
            tools: Vec::new(),
            error: Some("Timed out connecting to MCP server".to_string()),
        })),
    }
}

#[cfg(test)]
mod tests {
    use super::{mcp_file_has_servers, should_keep_existing_mcp};

    #[test]
    fn empty_payload_keeps_existing_servers_unless_clear_is_deliberate() {
        assert!(should_keep_existing_mcp(true, false, true));
        assert!(!should_keep_existing_mcp(true, true, true));
        assert!(!should_keep_existing_mcp(true, false, false));
        assert!(!should_keep_existing_mcp(false, false, true));
    }

    #[tokio::test]
    async fn mcp_file_has_servers_reads_nonempty_map() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mcp.json");
        tokio::fs::write(&path, r#"{"mcpServers":{"demo":{"command":"npx"}}}"#)
            .await
            .expect("write");
        assert!(mcp_file_has_servers(&path).await);
        tokio::fs::write(&path, r#"{"mcpServers":{}}"#)
            .await
            .expect("write empty");
        assert!(!mcp_file_has_servers(&path).await);
    }
}
