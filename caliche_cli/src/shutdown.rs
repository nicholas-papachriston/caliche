//! Process exit for the CLI. Every way out stops the task ledger first:
//! a normal return through [`run_with_task_ledger`], and every early exit
//! (pipe output, a second Ctrl+C, SIGTERM, one-shot errors) through
//! [`exit`].

use anyhow::Result;
use caliche::persistence::task_ledger;
use std::future::Future;
use std::path::PathBuf;
use tokio::sync::OnceCell;

/// Result of the one ledger stop that every exit path waits on.
static LEDGER_STOPPED: OnceCell<Result<(), String>> = OnceCell::const_new();

/// Directory holding `tasks.db`: `~/.caliche`, beside `caliche.db`.
/// Fails when the home directory cannot be resolved.
pub fn task_ledger_dir() -> Result<PathBuf> {
    Ok(crate::storage::cli_home_dir()?.join(".caliche"))
}

/// Install the task ledger in `data_dir`, run `body`, then stop the ledger.
///
/// The ledger writer runs on the current runtime, so the stop must finish
/// before the runtime goes away; it commits queued writes, or saves them for
/// the next start. A missing `data_dir` or a ledger that cannot open is
/// logged and `body` runs without it. When both `body` and the stop fail,
/// `body`'s error is returned.
pub async fn run_with_task_ledger<T>(
    data_dir: Result<PathBuf>,
    body: impl Future<Output = Result<T>>,
) -> Result<T> {
    let installed = match data_dir {
        Ok(dir) => task_ledger::install(&dir).await,
        Err(error) => Err(error),
    };
    if let Err(error) = installed {
        tracing::error!(
            error = %format!("{error:#}"),
            "task ledger unavailable; tasks are not recorded"
        );
    }
    let result = body.await;
    let stopped = stop_task_ledger().await;
    match (result, stopped) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) => Err(anyhow::anyhow!("task ledger shutdown: {error}")),
        (Err(error), _) => Err(error),
    }
}

/// Stop the task ledger once. A caller that arrives while another stop is
/// running waits for it, so a forced exit cannot cut a normal stop short.
/// A failure is logged here, once.
async fn stop_task_ledger() -> Result<(), String> {
    LEDGER_STOPPED
        .get_or_init(|| async {
            task_ledger::shutdown().await.map_err(|error| {
                let message = format!("{error:#}");
                tracing::error!(error = %message, "task ledger shutdown failed");
                message
            })
        })
        .await
        .clone()
}

/// Stop the task ledger, then exit the process with `code`.
///
/// Outside a runtime (argument parsing) no ledger is installed, so this
/// exits at once. Inside one it blocks this worker until the stop is done;
/// the CLI runtime is multi-threaded, so the ledger writer keeps running.
#[expect(
    clippy::exit,
    reason = "the CLI's single process exit; stops the task ledger first"
)]
pub fn exit(code: i32) -> ! {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        let _logged = tokio::task::block_in_place(|| handle.block_on(stop_task_ledger()));
    }
    std::process::exit(code)
}
