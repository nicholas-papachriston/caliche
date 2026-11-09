//! Serialized stdout access for REPL terminal output.
//!
//! When the split view is active, multiple async tasks can write to the terminal:
//! the stream processor (render_chunk, tool output), the FollowUpReader (render_bar
//! on keypress), and refresh_layout. Without serialization, concurrent writes
//! can interleave and corrupt the display. This module provides a global lock
//! so all terminal output is serialized.

use parking_lot::Mutex;
use std::io::{self, stdout};

static STDOUT_LOCK: Mutex<()> = Mutex::new(());

/// Run a closure with exclusive access to stdout. All terminal output during
/// the split view should go through this to prevent overlapping/corruption.
/// Passes the concrete StdoutLock so crossterm's queue!/execute! macros work.
pub fn with_stdout<F, R>(f: F) -> R
where
    F: FnOnce(&mut io::StdoutLock<'_>) -> R,
{
    let _guard = STDOUT_LOCK.lock();
    let mut out = stdout().lock();
    f(&mut out)
}

/// Write a single newline through the serialized stdout path.
pub fn write_newline() {
    with_stdout(|out| {
        use std::io::Write;
        if let Err(err) = out.write_all(b"\n") {
            tracing::warn!(target: "repl::stdout_sync", "failed to write newline: {}", err);
            return;
        }
        if let Err(err) = out.flush() {
            tracing::warn!(target: "repl::stdout_sync", "failed to flush newline: {}", err);
        }
    });
}
