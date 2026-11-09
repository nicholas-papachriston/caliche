use super::super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;
use std::path::Path;

fn format_forward(verb: &str, forward: &crate::gateway_runtime::ForwardStatus) -> String {
    format!(
        "Gateway forwarding {verb} in {} via {} on local port {} (pid {}, url: {}, log: {}).",
        forward.cwd,
        forward.provider.as_str(),
        forward.local_port,
        forward.tunnel_pid,
        forward.public_url.as_deref().unwrap_or("pending"),
        forward.log_path
    )
}

fn format_forward_status(forward: &crate::gateway_runtime::ForwardStatus) -> String {
    format!(
        "Gateway forwarding: {} via {} on local port {} (pid {}, running: {}, url: {}, log: {}).",
        forward.cwd,
        forward.provider.as_str(),
        forward.local_port,
        forward.tunnel_pid,
        forward.running,
        forward.public_url.as_deref().unwrap_or("pending"),
        forward.log_path
    )
}

fn format_forward_rows(rows: Vec<crate::gateway_runtime::ForwardStatus>) -> String {
    let mut lines = vec!["Gateway Forwarding Sessions".to_string(), String::new()];
    for row in rows {
        lines.push(format!("  {}", row.cwd));
        lines.push(format!("    Provider: {}", row.provider.as_str()));
        lines.push(format!("    Local Port: {}", row.local_port));
        lines.push(format!("    PID: {}", row.tunnel_pid));
        lines.push(format!("    Running: {}", row.running));
        lines.push(format!(
            "    URL: {}",
            row.public_url.as_deref().unwrap_or("pending")
        ));
        lines.push(format!("    Log: {}", row.log_path));
        lines.push(String::new());
    }
    lines.join("\n")
}

async fn start_forward(
    dir: Option<&Path>,
    port: Option<u16>,
    provider: Option<crate::gateway_runtime::ForwardProviderKind>,
) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::start_gateway_forwarding(dir, port, provider).await {
        Ok(forward) => Ok(SlashCommandResult::StatusMessage(format_forward(
            "active", &forward,
        ))),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to start gateway forwarding: {err}"
        ))),
    }
}

async fn stop_forward(dir: Option<&Path>) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::stop_gateway_forwarding(dir).await {
        Ok(true) => Ok(SlashCommandResult::StatusMessage(
            "Gateway forwarding stopped.".to_string(),
        )),
        Ok(false) => Ok(SlashCommandResult::StatusMessage(
            "No running forwarding session found for that directory.".to_string(),
        )),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to stop gateway forwarding: {err}"
        ))),
    }
}

async fn restart_forward(
    dir: Option<&Path>,
    port: Option<u16>,
    provider: Option<crate::gateway_runtime::ForwardProviderKind>,
) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::restart_gateway_forwarding(dir, port, provider).await {
        Ok(forward) => Ok(SlashCommandResult::StatusMessage(format_forward(
            "restarted",
            &forward,
        ))),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to restart gateway forwarding: {err}"
        ))),
    }
}

async fn status_forward(dir: Option<&Path>) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::status_gateway_forwarding(dir).await {
        Ok(Some(forward)) => Ok(SlashCommandResult::StatusMessage(format_forward_status(
            &forward,
        ))),
        Ok(None) => Ok(SlashCommandResult::StatusMessage(
            "No forwarding session is registered for that directory.".to_string(),
        )),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to read forwarding status: {err}"
        ))),
    }
}

async fn list_forwards() -> Result<SlashCommandResult> {
    match crate::gateway_runtime::list_gateway_forwardings().await {
        Ok(rows) => {
            if rows.is_empty() {
                return Ok(SlashCommandResult::StatusMessage(
                    "No gateway forwarding sessions are currently registered.".to_string(),
                ));
            }
            Ok(SlashCommandResult::ShowPopup {
                title: "Gateway Forwarding".to_string(),
                content: format_forward_rows(rows),
            })
        }
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to list gateway forwarding sessions: {err}"
        ))),
    }
}

impl SlashCommandExecutor {
    pub(crate) async fn handle_gateway_forward(args: &[String]) -> Result<SlashCommandResult> {
        let forward_action = args.get(2).map(String::as_str).unwrap_or("status");
        let (provider, forward_port, forward_dir) =
            Self::parse_gateway_forward_options(args.get(3..).unwrap_or(&[]));
        match forward_action {
            "start" => start_forward(forward_dir.as_deref(), forward_port, provider).await,
            "stop" => stop_forward(forward_dir.as_deref()).await,
            "restart" => restart_forward(forward_dir.as_deref(), forward_port, provider).await,
            "status" => status_forward(forward_dir.as_deref()).await,
            "list" => list_forwards().await,
            other => Ok(SlashCommandResult::Error(format!(
                "Unknown /gateway forward subcommand: '{other}'\n\nAvailable: start, stop, restart, status, list\nOptional: --provider cloudflare --port <n> --dir <path>"
            ))),
        }
    }
}
