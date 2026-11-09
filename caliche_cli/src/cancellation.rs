//! Cancellation support for graceful shutdown of agent operations

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::Notify;

/// Cancellation token for agent operations
#[derive(Clone, Debug)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
    exit_requested: Arc<AtomicBool>,
    last_cancel_at: Arc<Mutex<Option<Instant>>>,
    notify: Arc<Notify>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            exit_requested: Arc::new(AtomicBool::new(false)),
            last_cancel_at: Arc::new(Mutex::new(None)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Shared cancel flag for cross-crate cancellation-aware operations.
    pub fn shared_cancelled_flag(&self) -> Arc<AtomicBool> {
        self.cancelled.clone()
    }

    /// Returns true when a second Ctrl+C requested immediate REPL exit.
    pub fn exit_requested(&self) -> bool {
        self.exit_requested.load(Ordering::Relaxed)
    }

    /// Request cancellation
    pub fn cancel(&self) {
        let was_cancelled = self.cancelled.swap(true, Ordering::SeqCst);
        let now = Instant::now();

        // Ctrl+C can be observed by both reedline and OS signal handlers.
        // Treat near-simultaneous duplicates as one interrupt so a single
        // keypress does not trigger "second interrupt" exit behavior.
        let mut last_cancel_guard = match self.last_cancel_at.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let is_near_duplicate = last_cancel_guard
            .as_ref()
            .is_some_and(|previous| now.duration_since(*previous) < Duration::from_millis(250));
        *last_cancel_guard = Some(now);

        if was_cancelled && !is_near_duplicate {
            self.exit_requested.store(true, Ordering::SeqCst);
        }
        self.notify.notify_waiters();
    }

    /// Reset the token so it can be reused for the next turn.
    ///
    /// Call this at the start of each new agent turn so that a previous
    /// cancellation does not immediately abort the new turn.
    pub fn reset(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
        self.exit_requested.store(false, Ordering::SeqCst);
        if let Ok(mut guard) = self.last_cancel_at.lock() {
            *guard = None;
        }
    }

    /// Wait for cancellation to be requested
    pub async fn cancelled(&self) {
        if !self.is_cancelled() {
            self.notify.notified().await;
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// Set up signal handlers for graceful shutdown
/// Returns a cancellation token that will be triggered on SIGINT/SIGTERM
pub fn setup_signal_handler() -> CancellationToken {
    let token = CancellationToken::new();
    let token_clone = token.clone();

    #[cfg(unix)]
    {
        tokio::spawn(run_unix_signal_handler_with_restart(token_clone));
    }

    #[cfg(not(unix))]
    {
        tokio::spawn(run_ctrlc_handler_with_restart(token_clone));
    }

    token
}

/// How long SIGTERM waits for the normal return before exiting: one server
/// drain plus a second for the server records and the task ledger.
const TERMINATE_GRACE: Duration =
    caliche_server::shutdown::SERVER_SHUTDOWN_DRAIN.saturating_add(Duration::from_secs(1));

/// Exit status for SIGTERM: 128 + 15.
const EXIT_TERMINATED: i32 = 143;

fn maybe_force_exit_on_second_interrupt(token: &CancellationToken) {
    if token.exit_requested() {
        eprintln!("\nInterrupted twice. Exiting.");
        crate::shutdown::exit(130);
    }
}

#[cfg(unix)]
async fn run_unix_signal_handler_once(token: CancellationToken) {
    use tokio::signal::unix::{SignalKind, signal};
    let mut sigint = signal(SignalKind::interrupt()).ok();
    let mut sigterm = signal(SignalKind::terminate()).ok();

    loop {
        tokio::select! {
            _ = async {
                if let Some(ref mut s) = sigint {
                    s.recv().await
                } else {
                    std::future::pending().await
                }
            } => {
                token.cancel();
                maybe_force_exit_on_second_interrupt(&token);
            }
            _ = async {
                if let Some(ref mut s) = sigterm {
                    s.recv().await
                } else {
                    std::future::pending().await
                }
            } => {
                token.cancel();
                // A turn or a server drain that ends in time exits through the
                // normal return; an idle REPL never returns on its own.
                tokio::time::sleep(TERMINATE_GRACE).await;
                crate::shutdown::exit(EXIT_TERMINATED);
            }
        }
    }
}

#[cfg(unix)]
async fn run_unix_signal_handler_with_restart(token: CancellationToken) {
    loop {
        let token_for_task = token.clone();
        let handle = tokio::spawn(async move {
            run_unix_signal_handler_once(token_for_task).await;
        });
        match handle.await {
            Ok(()) => break,
            Err(err) => {
                tracing::error!(
                    target: "cli::cancellation",
                    "unix signal handler task failed: {}. restarting",
                    err
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

#[cfg(not(unix))]
async fn run_ctrlc_handler_once(token: CancellationToken) {
    loop {
        if tokio::signal::ctrl_c().await.is_ok() {
            token.cancel();
            maybe_force_exit_on_second_interrupt(&token);
        }
    }
}

#[cfg(not(unix))]
async fn run_ctrlc_handler_with_restart(token: CancellationToken) {
    loop {
        let token_for_task = token.clone();
        let handle = tokio::spawn(async move {
            run_ctrlc_handler_once(token_for_task).await;
        });
        match handle.await {
            Ok(()) => break,
            Err(err) => {
                tracing::error!(
                    target: "cli::cancellation",
                    "ctrl-c handler task failed: {}. restarting",
                    err
                );
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

/// Helper to check cancellation and return an error if cancelled
pub fn check_cancellation(token: &CancellationToken) -> anyhow::Result<()> {
    if token.is_cancelled() {
        anyhow::bail!("Operation cancelled by user");
    }
    Ok(())
}
