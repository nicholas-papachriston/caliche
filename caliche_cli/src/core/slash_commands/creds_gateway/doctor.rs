use super::super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;

fn format_doctor_rows(rows: Vec<crate::gateway_runtime::GatewayDoctorRecord>) -> String {
    let mut lines = vec!["Gateway Doctor".to_string(), String::new()];
    for row in rows {
        lines.push(format!("  {}", row.cwd));
        lines.push(format!("    Role: {}", row.role));
        lines.push(format!("    Port: {}", row.port));
        lines.push(format!(
            "    Recorded PID: {} (running: {})",
            row.recorded_pid, row.recorded_running
        ));
        lines.push(format!(
            "    Port Listener PID: {}",
            row.listener_pid
                .map(|pid| pid.to_string())
                .unwrap_or_else(|| "<none>".to_string())
        ));
        lines.push(format!(
            "    Listener Matches Recorded: {}",
            row.listener_matches_recorded
                .map(|matched| matched.to_string())
                .unwrap_or_else(|| "<unknown>".to_string())
        ));
        if let Some(pid) = row.forwarding_pid {
            lines.push(format!(
                "    Forwarding PID: {} (running: {})",
                pid,
                row.forwarding_running.unwrap_or(false)
            ));
        } else {
            lines.push("    Forwarding PID: <none>".to_string());
        }
        lines.push(format!(
            "    Forwarding URL: {}",
            row.forwarding_url.unwrap_or_else(|| "pending".to_string())
        ));
        lines.push(format!("    Runtime Log: {}", row.runtime_log_path));
        if let Some(path) = row.forwarding_log_path {
            lines.push(format!("    Forwarding Log: {}", path));
        }
        lines.push(String::new());
    }
    lines.join("\n")
}

impl SlashCommandExecutor {
    pub(crate) async fn handle_gateway_doctor(args: &[String]) -> Result<SlashCommandResult> {
        let (doctor_dir, include_all) =
            Self::parse_gateway_doctor_options(args.get(2..).unwrap_or(&[]));
        match crate::gateway_runtime::doctor_gateway(doctor_dir.as_deref(), include_all).await {
            Ok(rows) => {
                if rows.is_empty() {
                    return Ok(SlashCommandResult::StatusMessage(
                        "No gateway runtimes are currently registered.".to_string(),
                    ));
                }
                Ok(SlashCommandResult::ShowPopup {
                    title: "Gateway Doctor".to_string(),
                    content: format_doctor_rows(rows),
                })
            }
            Err(err) => Ok(SlashCommandResult::Error(format!(
                "Failed to run gateway doctor: {err}"
            ))),
        }
    }
}
