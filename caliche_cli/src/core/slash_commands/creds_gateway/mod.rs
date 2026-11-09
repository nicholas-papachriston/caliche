use super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;

mod doctor;
mod forward;
mod lifecycle;
mod llm;
mod logs;
mod route;
mod setup;
mod twilio;

impl SlashCommandExecutor {
    pub(crate) async fn handle_creds_gateway(args: &[String]) -> Result<SlashCommandResult> {
        let action = args.get(1).map(String::as_str).unwrap_or("status");
        let rest = args.get(2..).unwrap_or(&[]);
        let (port, dir) = Self::parse_gateway_runtime_options(rest);
        let stop_all = Self::parse_gateway_all_option(rest);

        match action {
            "setup" => Self::handle_gateway_setup(args).await,
            "start" => Self::handle_gateway_start(dir.as_deref(), port).await,
            "stop" => Self::handle_gateway_stop(dir.as_deref(), stop_all).await,
            "restart" => Self::handle_gateway_restart(dir.as_deref(), port).await,
            "status" => Self::handle_gateway_status(dir.as_deref()).await,
            "list" => Self::handle_gateway_list().await,
            "forward" => Self::handle_gateway_forward(args).await,
            "route" => Self::handle_gateway_route(args).await,
            "llm" => Self::handle_gateway_llm(args).await,
            "twilio" => Self::handle_gateway_twilio(args).await,
            "logs" => Self::handle_gateway_logs(args).await,
            "doctor" => Self::handle_gateway_doctor(args).await,
            other => Ok(SlashCommandResult::Error(format!(
                "Unknown /gateway subcommand: '{other}'\n\nAvailable: setup, start, stop, restart, status, list, forward, route, llm, twilio, logs, doctor\nOptional: --port <n> --dir <path>"
            ))),
        }
    }
}
