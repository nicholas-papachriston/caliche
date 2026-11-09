//! Top-level `daemon` CLI commands for OS user-service install.

use crate::commands::DaemonAction;
use crate::daemon::{
    default_daemon_spec, install_user_daemon, start_user_daemon, status_user_daemon,
    stop_user_daemon, uninstall_user_daemon,
};
use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use async_trait::async_trait;
use std::io::Write;
use std::path::PathBuf;

/// Handler for `caliche daemon …`.
pub struct DaemonHandler;

#[async_trait]
impl CommandHandler for DaemonHandler {
    type Args = DaemonAction;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        match args {
            DaemonAction::Install { port, label, cwd } => install_daemon(port, label, cwd),
            DaemonAction::Uninstall { label } => uninstall_daemon(label),
            DaemonAction::Start { label, port } => start_daemon(label, port),
            DaemonAction::Stop { label } => stop_daemon(label),
            DaemonAction::Status { label } => status_daemon(label),
        }
    }

    fn name(&self) -> &'static str {
        "daemon"
    }

    fn description(&self) -> &'static str {
        "Install and control a user-level Caliche server daemon"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

fn command_error(error: impl ToString) -> CliError {
    CliError::Command(error.to_string())
}

fn install_daemon(
    port: u16,
    label: Option<String>,
    cwd: Option<String>,
) -> CliResult<CommandResult> {
    let cwd_path = cwd.as_ref().map(PathBuf::from);
    let spec =
        default_daemon_spec(port, label.as_deref(), cwd_path.as_deref()).map_err(command_error)?;
    let path = install_user_daemon(&spec).map_err(command_error)?;
    let mut out = std::io::stdout().lock();
    writeln!(out, "Installed daemon unit: {}", path.display()).map_err(command_error)?;
    writeln!(
        out,
        "Label: {}  Port: {}  Cwd: {}",
        spec.label,
        spec.port,
        spec.cwd.display()
    )
    .map_err(command_error)?;
    writeln!(
        out,
        "The unit runs: {} --server --port {}",
        spec.exe.display(),
        spec.port
    )
    .map_err(command_error)?;
    writeln!(
        out,
        "List or stop recorded servers with: caliche server list"
    )
    .map_err(command_error)?;
    Ok(CommandResult::Success(Some(path.display().to_string())))
}

fn uninstall_daemon(label: Option<String>) -> CliResult<CommandResult> {
    let spec = default_daemon_spec(3000, label.as_deref(), None).map_err(command_error)?;
    uninstall_user_daemon(&spec).map_err(command_error)?;
    println!("Uninstalled daemon unit for {}", spec.label);
    Ok(CommandResult::Success(None))
}

fn start_daemon(label: Option<String>, port: Option<u16>) -> CliResult<CommandResult> {
    let spec =
        default_daemon_spec(port.unwrap_or(3000), label.as_deref(), None).map_err(command_error)?;
    start_user_daemon(&spec).map_err(command_error)?;
    println!("Started daemon {}", spec.label);
    Ok(CommandResult::Success(None))
}

fn stop_daemon(label: Option<String>) -> CliResult<CommandResult> {
    let spec = default_daemon_spec(3000, label.as_deref(), None).map_err(command_error)?;
    stop_user_daemon(&spec).map_err(command_error)?;
    println!("Stopped daemon {}", spec.label);
    Ok(CommandResult::Success(None))
}

fn status_daemon(label: Option<String>) -> CliResult<CommandResult> {
    let spec = default_daemon_spec(3000, label.as_deref(), None).map_err(command_error)?;
    let status = status_user_daemon(&spec).map_err(command_error)?;
    let mut out = std::io::stdout().lock();
    writeln!(out, "Label: {}", status.label).map_err(command_error)?;
    writeln!(out, "Unit: {}", status.unit_path.display()).map_err(command_error)?;
    writeln!(
        out,
        "Installed: {}  Running: {}",
        status.installed, status.running
    )
    .map_err(command_error)?;
    if !status.detail.is_empty() {
        writeln!(out, "Detail: {}", status.detail).map_err(command_error)?;
    }
    Ok(CommandResult::Success(None))
}
