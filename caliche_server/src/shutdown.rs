//! Process shutdown shared by the API binary and `caliche --server`.
//!
//! SIGINT and SIGTERM take the same path: stop accepting work, drain in-flight
//! connections, and stop when [`SERVER_SHUTDOWN_DRAIN`] elapses. The task
//! ledger stops after that, in [`run_with_task_ledger`].

use caliche::persistence::task_ledger;
use std::future::Future;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::watch;

/// How long a server may wait for in-flight work after a shutdown signal.
pub const SERVER_SHUTDOWN_DRAIN: Duration = Duration::from_secs(2);

/// Which process signal asked the server to stop.
///
/// Both variants start the same drain. The name is only for the log line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessSignal {
    /// Ctrl-C / SIGINT.
    Interrupt,
    /// SIGTERM.
    Terminate,
}

impl ProcessSignal {
    /// Stable label for logs.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Interrupt => "SIGINT",
            Self::Terminate => "SIGTERM",
        }
    }
}

/// First of `interrupt` or `terminate` to complete.
pub async fn first_process_signal(
    interrupt: impl Future<Output = ()>,
    terminate: impl Future<Output = ()>,
) -> ProcessSignal {
    tokio::select! {
        biased;
        _ = interrupt => ProcessSignal::Interrupt,
        _ = terminate => ProcessSignal::Terminate,
    }
}

/// Wait until this process receives SIGINT or SIGTERM.
pub async fn recv_process_signal() -> ProcessSignal {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let interrupt = async {
            match signal(SignalKind::interrupt()) {
                Ok(mut handle) => {
                    handle.recv().await;
                }
                Err(err) => {
                    tracing::warn!(error = %err, "failed to listen for SIGINT");
                    std::future::pending::<()>().await;
                }
            }
        };
        let terminate = async {
            match signal(SignalKind::terminate()) {
                Ok(mut handle) => {
                    handle.recv().await;
                }
                Err(err) => {
                    tracing::warn!(error = %err, "failed to listen for SIGTERM");
                    std::future::pending::<()>().await;
                }
            }
        };
        first_process_signal(interrupt, terminate).await
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        ProcessSignal::Interrupt
    }
}

/// Resolves when `shutdown` becomes true.
pub async fn wait_for_shutdown(mut shutdown: watch::Receiver<bool>) {
    if *shutdown.borrow() {
        return;
    }
    loop {
        if shutdown.changed().await.is_err() {
            return;
        }
        if *shutdown.borrow() {
            return;
        }
    }
}

/// Directory holding `tasks.db`: the `caliche` data directory, beside
/// the plugin store. Fails when the platform has no data directory.
pub fn task_ledger_dir() -> anyhow::Result<PathBuf> {
    crate::paths::task_ledger_dir()
        .ok_or_else(|| anyhow::anyhow!("no data directory for the task ledger"))
}

/// Install the task ledger in `data_dir`, run `body`, then stop the ledger.
///
/// The ledger writer runs on the current runtime, so the stop must finish
/// before the runtime goes away; it commits queued writes, or saves them for
/// the next start. Run the whole server as `body` so the stop comes after
/// the drain. A missing `data_dir` or a ledger that cannot open is logged
/// and `body` runs without it. When both `body` and the stop fail, `body`'s
/// error is returned.
pub async fn run_with_task_ledger<T>(
    data_dir: anyhow::Result<PathBuf>,
    body: impl Future<Output = anyhow::Result<T>>,
) -> anyhow::Result<T> {
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
    let stopped = task_ledger::shutdown().await;
    match (result, stopped) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) => Err(error.context("task ledger shutdown")),
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(ledger_error)) => {
            tracing::error!(
                error = %format!("{ledger_error:#}"),
                "task ledger shutdown failed"
            );
            Err(error)
        }
    }
}

/// Why [`select_server_or_drain`] returned.
#[derive(Debug)]
pub enum ServeStop<T> {
    /// The server future finished on its own.
    Finished(T),
    /// The drain bound elapsed after a shutdown signal.
    DrainElapsed,
}

/// Run `server` until it finishes, or until `shutdown` then `drain`.
///
/// `shutdown` should be the same watch the server's graceful-shutdown future
/// is waiting on, so the drain starts when new connections stop being accepted.
pub async fn select_server_or_drain<F>(
    server: F,
    shutdown: watch::Receiver<bool>,
    drain: Duration,
) -> ServeStop<F::Output>
where
    F: Future,
{
    tokio::pin!(server);
    let wait = wait_for_shutdown(shutdown);
    tokio::pin!(wait);
    let sleep = tokio::time::sleep(Duration::from_secs(86_400));
    tokio::pin!(sleep);
    let mut draining = false;
    loop {
        tokio::select! {
            biased;
            result = &mut server => return ServeStop::Finished(result),
            _ = &mut wait, if !draining => {
                draining = true;
                sleep.as_mut().reset(tokio::time::Instant::now() + drain);
            }
            _ = &mut sleep, if draining => {
                tracing::info!(
                    seconds = drain.as_secs(),
                    "shutdown drain elapsed; stopping"
                );
                return ServeStop::DrainElapsed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ProcessSignal, ServeStop, first_process_signal, select_server_or_drain, wait_for_shutdown,
    };
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use tokio::sync::watch;

    #[tokio::test]
    async fn interrupt_and_terminate_both_start_shutdown() {
        let interrupt = first_process_signal(async {}, std::future::pending()).await;
        assert_eq!(interrupt, ProcessSignal::Interrupt);
        let terminate = first_process_signal(std::future::pending(), async {}).await;
        assert_eq!(terminate, ProcessSignal::Terminate);
        assert_ne!(interrupt.as_str(), terminate.as_str());
    }

    #[tokio::test]
    async fn drain_bound_ends_a_held_connection() {
        let started_handler = Arc::new(tokio::sync::Notify::new());
        let notify = Arc::clone(&started_handler);
        let app = axum::Router::new().route(
            "/hold",
            axum::routing::get(move || {
                let notify = Arc::clone(&notify);
                async move {
                    notify.notify_one();
                    std::future::pending::<()>().await;
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("addr");
        let (tx, rx) = watch::channel(false);
        let server = axum::serve(listener, app)
            .with_graceful_shutdown(wait_for_shutdown(rx.clone()))
            .into_future();
        let handle = tokio::spawn(select_server_or_drain(
            server,
            rx,
            Duration::from_millis(200),
        ));
        let request = tokio::spawn(async move {
            let _ = reqwest::Client::new()
                .get(format!("http://{addr}/hold"))
                .send()
                .await;
        });
        tokio::time::timeout(Duration::from_secs(10), started_handler.notified())
            .await
            .expect("handler should accept the request");
        tx.send(true).expect("signal shutdown");
        let started = Instant::now();
        let stop = tokio::time::timeout(Duration::from_secs(10), handle)
            .await
            .expect("drain should be bounded")
            .expect("join");
        assert!(
            matches!(stop, ServeStop::DrainElapsed),
            "a handler that never returns must hit the drain bound, got {stop:?}"
        );
        assert!(started.elapsed() < Duration::from_secs(10));
        request.abort();
    }
}
