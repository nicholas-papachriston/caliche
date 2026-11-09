//! `caliche server list|status|stop` for recorded HTTP servers.

use crate::client::ServerLock;
use crate::commands::ServerAction;
use crate::errors::{CliError, CliResult};
use crate::server_registry::{self, ServerSelector};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use async_trait::async_trait;
use std::io::Write;

/// Handler for `caliche server …`.
pub struct ServerHandler;

#[async_trait]
impl CommandHandler for ServerHandler {
    type Args = ServerAction;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        run(&args).await?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "server"
    }

    fn description(&self) -> &'static str {
        "List and stop recorded Caliche HTTP servers"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

/// Run a server inventory command without opening the credential store.
pub async fn run(action: &ServerAction) -> CliResult<()> {
    match action {
        ServerAction::List => print_records(false, None, None),
        ServerAction::Status { id, port } => print_records(true, id.as_deref(), *port),
        ServerAction::Stop { id, port } => stop(id.as_deref(), *port).await,
    }
}

fn print_records(detailed: bool, id: Option<&str>, port: Option<u16>) -> CliResult<()> {
    let dir = server_registry::default_registry_dir();
    let rows = server_registry::list_servers(&dir).map_err(command_error)?;
    let rows = filter_rows(rows, id, port)?;
    let lock = ServerLock::read_from(&ServerLock::default_path()).map_err(command_error)?;
    let text = if detailed {
        server_registry::format_status(&dir, &rows, lock.as_ref())
    } else {
        server_registry::format_list(&dir, &rows, lock.as_ref())
    };
    let mut out = std::io::stdout().lock();
    write!(out, "{text}").map_err(|err| CliError::Command(err.to_string()))?;
    Ok(())
}

async fn stop(id: Option<&str>, port: Option<u16>) -> CliResult<()> {
    let selector = server_registry::selector_from_cli(id, port).map_err(command_error)?;
    let report = server_registry::stop_servers(
        &server_registry::default_registry_dir(),
        &ServerLock::default_path(),
        selector,
    )
    .await
    .map_err(command_error)?;
    if report.failed {
        return Err(CliError::Command(report.lines.join("\n")));
    }
    let mut out = std::io::stdout().lock();
    for line in &report.lines {
        writeln!(out, "{line}").map_err(|err| CliError::Command(err.to_string()))?;
    }
    Ok(())
}

fn filter_rows(
    rows: Vec<server_registry::ListedServer>,
    id: Option<&str>,
    port: Option<u16>,
) -> CliResult<Vec<server_registry::ListedServer>> {
    let Some(selector) = status_selector(id, port).map_err(command_error)? else {
        return Ok(rows);
    };
    let filtered: Vec<_> = rows
        .into_iter()
        .filter(|row| selector.matches(&row.record))
        .collect();
    if filtered.is_empty() {
        return Err(CliError::Command(format!(
            "no Caliche server record matches {selector}"
        )));
    }
    Ok(filtered)
}

fn status_selector(id: Option<&str>, port: Option<u16>) -> anyhow::Result<Option<ServerSelector>> {
    match (id.map(str::trim).filter(|value| !value.is_empty()), port) {
        (None, None) => Ok(None),
        (id, port) => server_registry::selector_from_cli(id, port).map(Some),
    }
}

fn command_error(err: impl ToString) -> CliError {
    CliError::Command(err.to_string())
}
