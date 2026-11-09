//! Graceful stop for `caliche --server`.
//!
//! SIGINT and SIGTERM both start [`caliche_server::shutdown::select_server_or_drain`].

use caliche_server::shutdown::{SERVER_SHUTDOWN_DRAIN, ServeStop, select_server_or_drain};
use std::future::Future;
use tokio::sync::watch;

pub(super) async fn select_server<F>(
    server: F,
    shutdown: watch::Receiver<bool>,
) -> anyhow::Result<()>
where
    F: Future<Output = std::io::Result<()>>,
{
    match select_server_or_drain(server, shutdown, SERVER_SHUTDOWN_DRAIN).await {
        ServeStop::Finished(result) => result.map_err(|err| anyhow::anyhow!("Server error: {err}")),
        ServeStop::DrainElapsed => Ok(()),
    }
}
