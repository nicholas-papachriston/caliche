//! Terminal capability helpers used by the REPL.
//!
//! Previously this module also hosted an experimental split-terminal view
//! (`SplitView`) with a fixed input bar in the last row.  That split-view
//! implementation has been removed because the production REPL renders output
//! directly via reedline + stdout and never wired the split path into the hot
//! loop.  Only the lightweight capability queries are kept here.

use std::io::IsTerminal;

/// Minimum terminal rows required to report a useful size to callers.
const MIN_ROWS: u16 = 4;

/// Returns `true` when stdout is attached to an ANSI-capable TTY.
///
/// Used by the renderer to decide whether to emit cursor/clear sequences.
pub(crate) fn supports_ansi_terminal_control() -> bool {
    if !std::io::stdout().is_terminal() {
        return false;
    }
    !std::env::var("TERM")
        .map(|t| t.eq_ignore_ascii_case("dumb"))
        .unwrap_or(false)
}

/// Returns `(cols, rows)` for stdout, or `None` if not a TTY / unsupported /
/// too small to be useful.
pub(crate) fn terminal_size() -> Option<(u16, u16)> {
    if !supports_ansi_terminal_control() {
        return None;
    }
    crossterm::terminal::size()
        .ok()
        .filter(|(cols, rows)| *rows >= MIN_ROWS && *cols > 0)
}
