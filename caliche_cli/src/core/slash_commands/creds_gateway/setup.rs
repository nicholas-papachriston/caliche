use super::super::{SlashCommandExecutor, SlashCommandResult};
use anyhow::Result;
use std::path::Path;

fn format_runtime_started(runtime: &crate::gateway_runtime::GatewayRuntimeStatus) -> String {
    format!(
        "Gateway credentials configured.\nGateway runtime running in {} on port {} (pid {}, log: {}).",
        runtime.entry.cwd, runtime.entry.port, runtime.entry.pid, runtime.entry.log_path
    )
}

async fn append_forwarding_note(
    message: &mut String,
    dir: Option<&Path>,
    port: u16,
    provider: Option<crate::gateway_runtime::ForwardProviderKind>,
) {
    match crate::gateway_runtime::start_gateway_forwarding(dir, Some(port), provider).await {
        Ok(forward) => {
            if let Some(url) = forward.public_url {
                message.push_str(&format!(
                    "\nForwarding enabled via {}: {} (pid {}).",
                    forward.provider.as_str(),
                    url,
                    forward.tunnel_pid
                ));
            } else {
                message.push_str(&format!(
                    "\nForwarding enabled via {} (pid {}, URL pending; check {}).",
                    forward.provider.as_str(),
                    forward.tunnel_pid,
                    forward.log_path
                ));
            }
        }
        Err(err) => {
            message.push_str(&format!(
                "\nForwarding requested but failed to start: {err}"
            ));
        }
    }
}

async fn start_runtime_after_setup(
    dir: Option<&Path>,
    port: Option<u16>,
    forward_after_setup: bool,
    forward_provider: Option<crate::gateway_runtime::ForwardProviderKind>,
) -> Result<SlashCommandResult> {
    match crate::gateway_runtime::start_gateway_runtime(dir, port).await {
        Ok(runtime) => {
            let mut message = format_runtime_started(&runtime);
            if forward_after_setup {
                append_forwarding_note(&mut message, dir, runtime.entry.port, forward_provider)
                    .await;
            }
            Ok(SlashCommandResult::StatusMessage(message))
        }
        Err(err) => Ok(SlashCommandResult::Error(format!(
            "Gateway credentials configured, but failed to start runtime: {err:#}"
        ))),
    }
}

impl SlashCommandExecutor {
    pub(crate) async fn handle_gateway_setup(args: &[String]) -> Result<SlashCommandResult> {
        let (forward_after_setup, forward_provider, setup_port, setup_dir) =
            Self::parse_gateway_setup_options(args.get(2..).unwrap_or(&[]));
        match crate::credentials::interactive_setup_for(&[
            crate::credentials::CredentialSetupKind::Gateway,
        ])
        .await
        {
            Ok(report) if report.outcome == crate::credentials::SetupOutcome::Cancelled => Ok(
                SlashCommandResult::StatusMessage(crate::credentials::SETUP_CANCELLED.to_string()),
            ),
            Ok(_) => {
                start_runtime_after_setup(
                    setup_dir.as_deref(),
                    setup_port,
                    forward_after_setup,
                    forward_provider,
                )
                .await
            }
            Err(e) => Ok(SlashCommandResult::Error(format!("Setup failed: {e}"))),
        }
    }
}
