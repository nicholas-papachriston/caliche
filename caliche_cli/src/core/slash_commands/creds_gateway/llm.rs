use super::super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;
use std::path::Path;

fn format_llm_status(status: crate::gateway_runtime::GatewayLlmConfigStatus) -> String {
    [
        "Gateway Model Settings".to_string(),
        String::new(),
        format!("Directory: {}", status.cwd),
        format!(
            "Configured Provider: {}",
            status.configured_provider.as_deref().unwrap_or("<inherit>")
        ),
        format!(
            "Configured Model: {}",
            status.configured_model.as_deref().unwrap_or("<inherit>")
        ),
        String::new(),
        format!(
            "Effective Provider: {}",
            status.effective_provider.as_deref().unwrap_or("<unset>")
        ),
        format!(
            "Effective Model: {}",
            status.effective_model.as_deref().unwrap_or("<unset>")
        ),
        String::new(),
        "Tip: run '/gateway restart' (or '--dir <path>') after changing settings.".to_string(),
    ]
    .join("\n")
}

async fn restart_note_after_llm_set(dir: Option<&Path>) -> String {
    match crate::gateway_runtime::status_gateway_runtime(dir).await {
        Ok(Some(_)) => match crate::gateway_runtime::restart_gateway_runtime(dir, None).await {
            Ok(runtime) => format!(
                " Gateway runtime auto-restarted on port {} (pid {}).",
                runtime.entry.port, runtime.entry.pid
            ),
            Err(err) => format!(" Saved, but failed to auto-restart runtime: {err}"),
        },
        Ok(None) => " No running gateway runtime was found to restart.".to_string(),
        Err(err) => format!(" Saved, but could not determine runtime status: {err}"),
    }
}

async fn llm_status(dir: Option<&Path>) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::gateway_llm_config_status(dir).await {
        Ok(status) => Ok(SlashCommandResult::ShowPopup {
            title: "Gateway model".to_string(),
            content: format_llm_status(status),
        }),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to read gateway llm settings: {err}"
        ))),
    }
}

async fn llm_set(
    dir: Option<&Path>,
    provider: Option<String>,
    model: Option<String>,
) -> Result<SlashCommandResult> {
    if provider.is_none() && model.is_none() {
        return Ok(SlashCommandResult::Error(
            "Usage: /gateway llm set [--provider <provider>] [--model <model>] [--dir <path>]"
                .to_string(),
        ));
    }
    match crate::gateway_runtime::set_gateway_llm_config(dir, provider.as_deref(), model.as_deref())
        .await
    {
        Ok(status) => {
            let restart_note = restart_note_after_llm_set(dir).await;
            Ok(SlashCommandResult::StatusMessage(format!(
                "Gateway model settings saved for {}. Effective provider/model: {}/{}.{}",
                status.cwd,
                status.effective_provider.as_deref().unwrap_or("<unset>"),
                status.effective_model.as_deref().unwrap_or("<unset>"),
                restart_note
            )))
        }
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to save gateway llm settings: {err}"
        ))),
    }
}

async fn llm_clear(dir: Option<&Path>) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::clear_gateway_llm_config(dir).await {
        Ok(status) => Ok(SlashCommandResult::StatusMessage(format!(
            "Gateway model overrides cleared for {}. Effective provider/model now resolve from scoped defaults/env: {}/{}.",
            status.cwd,
            status.effective_provider.as_deref().unwrap_or("<unset>"),
            status.effective_model.as_deref().unwrap_or("<unset>"),
        ))),
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Failed to clear gateway llm settings: {err}"
        ))),
    }
}

impl SlashCommandExecutor {
    pub(crate) async fn handle_gateway_llm(args: &[String]) -> Result<SlashCommandResult> {
        let llm_action = args.get(2).map(String::as_str).unwrap_or("status");
        let (gateway_provider, gateway_model, gateway_dir) =
            Self::parse_gateway_llm_options(args.get(3..).unwrap_or(&[]));
        match llm_action {
            "status" => llm_status(gateway_dir.as_deref()).await,
            "set" => llm_set(gateway_dir.as_deref(), gateway_provider, gateway_model).await,
            "clear" => llm_clear(gateway_dir.as_deref()).await,
            _ => Ok(SlashCommandResult::Error(
                "Unknown /gateway llm subcommand.\n\nUsage:\n  /gateway llm status [--dir <path>]\n  /gateway llm set [--provider <provider>] [--model <model>] [--dir <path>]\n  /gateway llm clear [--dir <path>]".to_string(),
            )),
        }
    }
}
