//! Helpers for waiting on background workspace indexing before the first RAG query.
//!
//! When workspace embedding is enabled with `build_code_graph` in `WorkspaceIndexerConfig`,
//! the indexer can optionally build an AST-based code graph during indexing.
//! The graph is then available to `GraphExplorerTool` for structural code exploration.
//!
//! Workspace indexing has three states: **none** (no indexing), **partial** (in progress),
//! **complete** (done). The REPL/CLI agent should attempt to use workspace context when
//! state is partial or complete.

#[cfg(feature = "tools-rag")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "tools-rag")]
use std::time::Duration;
#[cfg(feature = "tools-rag")]
use tokio::sync::Notify;

/// Workspace indexing state. Agent uses workspace context when Partial or Complete.
#[cfg(feature = "tools-rag")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceIndexingState {
    /// No workspace indexing (no handle or RAG disabled).
    None,
    /// Indexing in progress; context may be partial.
    Partial,
    /// Indexing finished.
    Complete,
}

/// Tracks workspace indexing completion. Provides non-blocking check for background indexing.
#[cfg(feature = "tools-rag")]
pub struct WorkspaceIndexingReady {
    notify: Notify,
    done: AtomicBool,
    waited: AtomicBool,
}

#[cfg(feature = "tools-rag")]
impl Default for WorkspaceIndexingReady {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkspaceIndexingReady {
    #[must_use]
    pub fn new() -> Self {
        Self {
            notify: Notify::new(),
            done: AtomicBool::new(false),
            waited: AtomicBool::new(false),
        }
    }

    /// Signal that indexing has completed. Call from the background indexing task.
    pub fn notify_done(&self) {
        self.done.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    /// Check if indexing is done without blocking.
    /// Returns true if indexing completed, false if still in progress.
    pub fn is_ready(&self) -> bool {
        self.done.load(Ordering::Acquire)
    }

    /// Wait until indexing completes or `timeout` elapses.
    ///
    /// Safe to call repeatedly. Subscribes to the notify before re-checking `done`
    /// so a completion that races the wait is not missed.
    pub async fn wait_for_ready(&self, timeout: Duration) {
        if self.done.load(Ordering::Acquire) || timeout.is_zero() {
            return;
        }
        let notified = self.notify.notified();
        if self.done.load(Ordering::Acquire) {
            return;
        }
        let _ = tokio::time::timeout(timeout, notified).await;
    }

    /// Wait for indexing to complete, at most once per instance, with timeout.
    /// Used when the first workspace context fetch may benefit from waiting briefly for indexing.
    /// Later turns in the same task skip the wait so they are not blocked by background ingest.
    pub async fn wait_once_with_timeout(&self, timeout: Duration) {
        if self.done.load(Ordering::Acquire) {
            return;
        }
        if self.waited.swap(true, Ordering::AcqRel) {
            return;
        }
        self.wait_for_ready(timeout).await;
    }

    /// Current state: Partial (in progress) or Complete. Caller uses None when no handle.
    pub fn state(&self) -> WorkspaceIndexingState {
        if self.done.load(Ordering::Acquire) {
            WorkspaceIndexingState::Complete
        } else {
            WorkspaceIndexingState::Partial
        }
    }
}

#[cfg(all(test, feature = "tools-rag"))]
mod tests {
    use super::WorkspaceIndexingReady;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn wait_once_returns_immediately_when_already_done() {
        let ready = WorkspaceIndexingReady::new();
        ready.notify_done();
        let start = Instant::now();
        ready.wait_once_with_timeout(Duration::from_secs(5)).await;
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "completed index must not wait out the timeout"
        );
    }

    #[tokio::test]
    async fn wait_once_does_not_block_later_task_turns() {
        let ready = WorkspaceIndexingReady::new();
        ready.wait_once_with_timeout(Duration::ZERO).await;
        let start = Instant::now();
        ready.wait_once_with_timeout(Duration::from_secs(5)).await;
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "second wait_once must not re-block the task loop"
        );
    }

    #[tokio::test]
    async fn wait_for_ready_observes_notify_without_missing_it() {
        let ready = WorkspaceIndexingReady::new();
        ready.notify_done();
        let start = Instant::now();
        ready.wait_for_ready(Duration::from_secs(5)).await;
        assert!(start.elapsed() < Duration::from_secs(2));
    }
}
