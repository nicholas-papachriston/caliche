use super::super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;
use std::path::Path;

fn format_runtime_line(
    verb: &str,
    runtime: &crate::gateway_runtime::GatewayRuntimeStatus,
) -> String {
    format!(
        "Gateway runtime ({}) {verb} in {} on port {} (pid {}, log: {}).",
        runtime.entry.role,
        runtime.entry.cwd,
        runtime.entry.port,
        runtime.entry.pid,
        runtime.entry.log_path
    )
}

fn format_runtime_status(runtime: &crate::gateway_runtime::GatewayRuntimeStatus) -> String {
    format!(
        "Gateway runtime ({}): {} on port {} (pid {}, running: {}, log: {}).",
        runtime.entry.role,
        runtime.entry.cwd,
        runtime.entry.port,
        runtime.entry.pid,
        runtime.running,
        runtime.entry.log_path
    )
}

fn format_runtime_rows(rows: Vec<crate::gateway_runtime::GatewayRuntimeStatus>) -> String {
    let mut lines = vec!["Gateway Runtimes".to_string(), String::new()];
    for row in rows {
        lines.push(format!("  {}", row.entry.cwd));
        lines.push(format!("    Role: {}", row.entry.role));
        lines.push(format!("    Agent ID: {}", row.entry.agent_id));
        lines.push(format!("    Port: {}", row.entry.port));
        lines.push(format!("    PID: {}", row.entry.pid));
        lines.push(format!("    Running: {}", row.running));
        lines.push(format!("    Log: {}", row.entry.log_path));
        if let Some(forwarding) = &row.entry.forwarding {
            lines.push(format!(
                "    Forwarding: {} (pid {}, url: {})",
                forwarding.provider,
                forwarding.tunnel_pid,
                forwarding.public_url.as_deref().unwrap_or("pending")
            ));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

async fn stop_all_runtimes() -> Result<SlashCommandResult> {
    match crate::gateway_runtime::stop_all_gateway_runtimes().await {
        Ok(0) => Ok(SlashCommandResult::StatusMessage(
            "No gateway runtimes are currently registered.".to_string(),
        )),
        Ok(count) => Ok(SlashCommandResult::StatusMessage(format!(
            "Stopped {} gateway runtime(s).",
            count
        ))),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to stop all gateway runtimes: {err}"
        ))),
    }
}

async fn stop_one_runtime(dir: Option<&Path>) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::stop_gateway_runtime(dir).await {
        Ok(true) => Ok(SlashCommandResult::StatusMessage(
            "Gateway runtime stopped.".to_string(),
        )),
        Ok(false) => Ok(SlashCommandResult::StatusMessage(
            "No running gateway runtime found for that directory.".to_string(),
        )),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to stop gateway runtime: {err}"
        ))),
    }
}

impl SlashCommandExecutor {
    pub(crate) async fn handle_gateway_start(
        dir: Option<&Path>,
        port: Option<u16>,
    ) -> Result<SlashCommandResult> {
        match crate::gateway_runtime::start_gateway_runtime(dir, port).await {
            Ok(runtime) => Ok(SlashCommandResult::StatusMessage(format_runtime_line(
                "running", &runtime,
            ))),
            Err(err) => Ok(SlashCommandResult::Error(format!(
                "Failed to start gateway runtime: {err:#}"
            ))),
        }
    }

    pub(crate) async fn handle_gateway_stop(
        dir: Option<&Path>,
        stop_all: bool,
    ) -> Result<SlashCommandResult> {
        if stop_all {
            stop_all_runtimes().await
        } else {
            stop_one_runtime(dir).await
        }
    }

    pub(crate) async fn handle_gateway_restart(
        dir: Option<&Path>,
        port: Option<u16>,
    ) -> Result<SlashCommandResult> {
        match crate::gateway_runtime::restart_gateway_runtime(dir, port).await {
            Ok(runtime) => Ok(SlashCommandResult::StatusMessage(format_runtime_line(
                "restarted",
                &runtime,
            ))),
            Err(err) => Ok(SlashCommandResult::Error(format!(
                "Failed to restart gateway runtime: {err:#}"
            ))),
        }
    }

    pub(crate) async fn handle_gateway_status(dir: Option<&Path>) -> Result<SlashCommandResult> {
        match crate::gateway_runtime::status_gateway_runtime(dir).await {
            Ok(Some(runtime)) => Ok(SlashCommandResult::StatusMessage(format_runtime_status(
                &runtime,
            ))),
            Ok(None) => Ok(SlashCommandResult::StatusMessage(
                "No gateway runtime registered for that directory.".to_string(),
            )),
            Err(err) => Ok(SlashCommandResult::Error(format!(
                "Failed to read gateway runtime status: {err}"
            ))),
        }
    }

    pub(crate) async fn handle_gateway_list() -> Result<SlashCommandResult> {
        match crate::gateway_runtime::list_gateway_runtimes().await {
            Ok(rows) => {
                if rows.is_empty() {
                    return Ok(SlashCommandResult::StatusMessage(
                        "No gateway runtimes are currently registered.".to_string(),
                    ));
                }
                Ok(SlashCommandResult::ShowPopup {
                    title: "Gateway Runtimes".to_string(),
                    content: format_runtime_rows(rows),
                })
            }
            Err(err) => Ok(SlashCommandResult::Error(format!(
                "Failed to list gateway runtimes: {err}"
            ))),
        }
    }
}
