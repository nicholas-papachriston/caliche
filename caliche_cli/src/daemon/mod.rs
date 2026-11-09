//! User-level OS daemon install for the Caliche HTTP server.
//!
//! Writes launchd (macOS) or systemd --user (Linux) units that exec
//! `caliche --server --port <N>`, then controls them via launchctl/systemctl.

mod service;
mod units;

pub use service::{
    DaemonStatus, default_daemon_spec, install_user_daemon, start_user_daemon, status_user_daemon,
    stop_user_daemon, uninstall_user_daemon,
};
pub use units::{DaemonUnitSpec, render_launchd_plist, render_systemd_user_unit};
