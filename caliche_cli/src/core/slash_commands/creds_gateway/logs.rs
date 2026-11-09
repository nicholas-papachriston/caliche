use super::super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;

fn format_log_snapshot(
    snapshot: crate::gateway_runtime::GatewayLogsSnapshot,
    include_forwarding: bool,
) -> String {
    let mut content = vec![
        format!("Directory: {}", snapshot.cwd),
        format!("Gateway Log: {}", snapshot.runtime_log_path),
        String::new(),
        "----- gateway log tail -----".to_string(),
        snapshot.runtime_log_tail,
    ];
    if include_forwarding {
        match (snapshot.forwarding_log_path, snapshot.forwarding_log_tail) {
            (Some(path), Some(tail)) => {
                content.push(String::new());
                content.push(format!("Forwarding Log: {}", path));
                content.push(String::new());
                content.push("----- forwarding log tail -----".to_string());
                content.push(tail);
            }
            _ => {
                content.push(String::new());
                content.push("No forwarding session is currently attached.".to_string());
            }
        }
    }
    content.join("\n")
}

async fn read_logs_with_fallback(
    dir: Option<&std::path::Path>,
    lines: usize,
    include_forwarding: bool,
) -> Result<crate::gateway_runtime::GatewayLogsSnapshot, anyhow::Error> {
    match crate::gateway_runtime::read_gateway_logs(dir, lines, include_forwarding).await {
        Ok(snapshot) => Ok(snapshot),
        Err(primary_err) => {
            match crate::gateway_runtime::read_gateway_logs_fallback(lines, include_forwarding)
                .await
            {
                Ok(snapshot) => Ok(snapshot),
                Err(_) => Err(primary_err),
            }
        }
    }
}

impl SlashCommandExecutor {
    pub(crate) async fn handle_gateway_logs(args: &[String]) -> Result<SlashCommandResult> {
        let (logs_dir, lines, include_forwarding) =
            Self::parse_gateway_logs_options(args.get(2..).unwrap_or(&[]));
        match read_logs_with_fallback(logs_dir.as_deref(), lines, include_forwarding).await {
            Ok(snapshot) => Ok(SlashCommandResult::ShowPopup {
                title: "Gateway Logs".to_string(),
                content: format_log_snapshot(snapshot, include_forwarding),
            }),
            Err(primary_err) => Ok(SlashCommandResult::Error(format!(
                "Failed to read gateway logs: {primary_err}"
            ))),
        }
    }
}
