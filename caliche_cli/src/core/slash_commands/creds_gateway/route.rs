use super::super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;
use std::path::Path;

fn format_route_rows(rows: Vec<crate::gateway_runtime::GatewayRouteInfo>) -> String {
    let mut lines = vec!["Gateway Route Assignments".to_string(), String::new()];
    for row in rows {
        lines.push(format!("  Route Key: {}", row.route_key));
        lines.push(format!("    Target Agent: {}", row.target_agent_id));
        lines.push(format!("    Updated At: {}", row.updated_at));
        lines.push(String::new());
    }
    lines.join("\n")
}

async fn list_routes(dir: Option<&Path>) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::list_gateway_routes(dir).await {
        Ok(rows) => {
            if rows.is_empty() {
                return Ok(SlashCommandResult::StatusMessage(
                    "No gateway route assignments are currently pinned.".to_string(),
                ));
            }
            Ok(SlashCommandResult::ShowPopup {
                title: "Gateway Routes".to_string(),
                content: format_route_rows(rows),
            })
        }
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to list gateway routes: {err}"
        ))),
    }
}

async fn pin_route(
    dir: Option<&Path>,
    route_key: Option<String>,
    target_agent_id: Option<String>,
) -> Result<SlashCommandResult> {
    let Some(route_key) = route_key else {
        return Ok(SlashCommandResult::Error(
            "Usage: /gateway route pin <route_key> --agent <agent_id> [--dir <path>]".to_string(),
        ));
    };
    let Some(target_agent_id) = target_agent_id else {
        return Ok(SlashCommandResult::Error(
            "Usage: /gateway route pin <route_key> --agent <agent_id> [--dir <path>]".to_string(),
        ));
    };
    match crate::gateway_runtime::pin_gateway_route(dir, &route_key, &target_agent_id).await {
        Ok(()) => Ok(SlashCommandResult::StatusMessage(format!(
            "Pinned route '{}' to agent '{}'.",
            route_key, target_agent_id
        ))),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to pin gateway route: {err}"
        ))),
    }
}

async fn unpin_route(dir: Option<&Path>, route_key: Option<String>) -> Result<SlashCommandResult> {
    let Some(route_key) = route_key else {
        return Ok(SlashCommandResult::Error(
            "Usage: /gateway route unpin <route_key> [--dir <path>]".to_string(),
        ));
    };
    match crate::gateway_runtime::unpin_gateway_route(dir, &route_key).await {
        Ok(()) => Ok(SlashCommandResult::StatusMessage(format!(
            "Unpinned route '{}'.",
            route_key
        ))),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to unpin gateway route: {err}"
        ))),
    }
}

impl SlashCommandExecutor {
    pub(crate) async fn handle_gateway_route(args: &[String]) -> Result<SlashCommandResult> {
        let route_action = args.get(2).map(String::as_str).unwrap_or("list");
        let (route_key, target_agent_id, route_dir) =
            Self::parse_gateway_route_options(args.get(3..).unwrap_or(&[]));
        match route_action {
            "list" => list_routes(route_dir.as_deref()).await,
            "pin" => pin_route(route_dir.as_deref(), route_key, target_agent_id).await,
            "unpin" => unpin_route(route_dir.as_deref(), route_key).await,
            other => Ok(SlashCommandResult::Error(format!(
                "Unknown /gateway route subcommand: '{other}'\n\nAvailable: list, pin, unpin\nUsage: /gateway route pin <route_key> --agent <agent_id> [--dir <path>]"
            ))),
        }
    }
}
