//! Install and control user-level Caliche daemon units.

use super::units::{DaemonUnitSpec, render_launchd_plist, render_systemd_user_unit};
use anyhow::{Context, Result, bail};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default launchd label / systemd unit basename.
pub const DEFAULT_DAEMON_LABEL: &str = "com.caliche.server";
/// Default systemd-friendly label (without domain prefix).
pub const DEFAULT_SYSTEMD_LABEL: &str = "caliche";

/// Observed daemon control state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DaemonStatus {
    pub installed: bool,
    pub running: bool,
    pub unit_path: PathBuf,
    pub label: String,
    pub detail: String,
}

/// Build a default unit spec for the current executable and cwd.
pub fn default_daemon_spec(
    port: u16,
    label: Option<&str>,
    cwd: Option<&Path>,
) -> Result<DaemonUnitSpec> {
    let exe = std::env::current_exe().context("resolve current executable")?;
    let cwd = match cwd {
        Some(path) => path.to_path_buf(),
        None => std::env::current_dir().context("resolve current directory")?,
    };
    let home = dirs::home_dir().context("resolve home directory")?;
    let log_dir = home.join(".caliche").join("logs");
    let label = label
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default_label_for_platform())
        .to_string();
    Ok(DaemonUnitSpec {
        label,
        exe,
        port,
        cwd,
        log_dir,
    })
}

fn default_label_for_platform() -> &'static str {
    if cfg!(target_os = "macos") {
        DEFAULT_DAEMON_LABEL
    } else {
        DEFAULT_SYSTEMD_LABEL
    }
}

/// Install (write) the user unit file and enable it with the OS service manager.
pub fn install_user_daemon(spec: &DaemonUnitSpec) -> Result<PathBuf> {
    fs::create_dir_all(&spec.log_dir)
        .with_context(|| format!("create log directory {}", spec.log_dir.display()))?;

    if cfg!(target_os = "macos") {
        install_launchd(spec)
    } else if cfg!(target_os = "linux") {
        install_systemd(spec)
    } else {
        bail!("daemon install is only supported on macOS and Linux")
    }
}

/// Remove the user unit and disable it.
pub fn uninstall_user_daemon(spec: &DaemonUnitSpec) -> Result<()> {
    if cfg!(target_os = "macos") {
        uninstall_launchd(spec)
    } else if cfg!(target_os = "linux") {
        uninstall_systemd(spec)
    } else {
        bail!("daemon uninstall is only supported on macOS and Linux")
    }
}

/// Start the installed user daemon.
pub fn start_user_daemon(spec: &DaemonUnitSpec) -> Result<()> {
    if cfg!(target_os = "macos") {
        run_checked(
            "launchctl",
            &["load", "-w", &unit_path(spec)?.display().to_string()],
        )
    } else if cfg!(target_os = "linux") {
        run_checked(
            "systemctl",
            &["--user", "enable", "--now", &spec.systemd_unit_name()],
        )
    } else {
        bail!("daemon start is only supported on macOS and Linux")
    }
}

/// Stop the installed user daemon.
pub fn stop_user_daemon(spec: &DaemonUnitSpec) -> Result<()> {
    if cfg!(target_os = "macos") {
        let path = unit_path(spec)?;
        if path.exists() {
            let _ = run_checked("launchctl", &["unload", "-w", &path.display().to_string()]);
        }
        Ok(())
    } else if cfg!(target_os = "linux") {
        run_checked(
            "systemctl",
            &["--user", "disable", "--now", &spec.systemd_unit_name()],
        )
    } else {
        bail!("daemon stop is only supported on macOS and Linux")
    }
}

/// Report whether the unit file exists and whether the service appears running.
pub fn status_user_daemon(spec: &DaemonUnitSpec) -> Result<DaemonStatus> {
    let unit_path = unit_path(spec)?;
    let installed = unit_path.exists();
    let (running, detail) = if cfg!(target_os = "macos") {
        launchd_status(&spec.label)
    } else if cfg!(target_os = "linux") {
        systemd_status(&spec.systemd_unit_name())
    } else {
        (false, "unsupported platform".to_string())
    };
    Ok(DaemonStatus {
        installed,
        running,
        unit_path,
        label: spec.label.clone(),
        detail,
    })
}

fn unit_path(spec: &DaemonUnitSpec) -> Result<PathBuf> {
    let home = dirs::home_dir().context("resolve home directory")?;
    if cfg!(target_os = "macos") {
        Ok(home
            .join("Library")
            .join("LaunchAgents")
            .join(spec.launchd_plist_name()))
    } else if cfg!(target_os = "linux") {
        Ok(home
            .join(".config")
            .join("systemd")
            .join("user")
            .join(spec.systemd_unit_name()))
    } else {
        bail!("daemon units are only supported on macOS and Linux")
    }
}

fn install_launchd(spec: &DaemonUnitSpec) -> Result<PathBuf> {
    let path = unit_path(spec)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create LaunchAgents directory {}", parent.display()))?;
    }
    let body = render_launchd_plist(spec);
    fs::write(&path, body).with_context(|| format!("write plist {}", path.display()))?;
    // Best-effort bootstrap; `daemon start` also loads.
    let _ = run_checked("launchctl", &["load", "-w", &path.display().to_string()]);
    Ok(path)
}

fn uninstall_launchd(spec: &DaemonUnitSpec) -> Result<()> {
    let path = unit_path(spec)?;
    if path.exists() {
        let _ = run_checked("launchctl", &["unload", "-w", &path.display().to_string()]);
        fs::remove_file(&path).with_context(|| format!("remove plist {}", path.display()))?;
    }
    Ok(())
}

fn install_systemd(spec: &DaemonUnitSpec) -> Result<PathBuf> {
    let path = unit_path(spec)?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create systemd user directory {}", parent.display()))?;
    }
    let body = render_systemd_user_unit(spec);
    fs::write(&path, body).with_context(|| format!("write unit {}", path.display()))?;
    run_checked("systemctl", &["--user", "daemon-reload"])?;
    run_checked(
        "systemctl",
        &["--user", "enable", "--now", &spec.systemd_unit_name()],
    )?;
    Ok(path)
}

fn uninstall_systemd(spec: &DaemonUnitSpec) -> Result<()> {
    let _ = run_checked(
        "systemctl",
        &["--user", "disable", "--now", &spec.systemd_unit_name()],
    );
    let path = unit_path(spec)?;
    if path.exists() {
        fs::remove_file(&path).with_context(|| format!("remove unit {}", path.display()))?;
    }
    let _ = run_checked("systemctl", &["--user", "daemon-reload"]);
    Ok(())
}

fn launchd_status(label: &str) -> (bool, String) {
    match Command::new("launchctl").arg("list").arg(label).output() {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            // `launchctl list <label>` prints "PID Status Label" when loaded.
            let running = stdout
                .lines()
                .next()
                .and_then(|line| line.split_whitespace().next())
                .is_some_and(|pid| pid != "-" && pid.parse::<i32>().is_ok());
            (running, stdout)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            (
                false,
                if stderr.is_empty() {
                    "not loaded".into()
                } else {
                    stderr
                },
            )
        }
        Err(error) => (false, format!("launchctl unavailable: {error}")),
    }
}

fn systemd_status(unit_name: &str) -> (bool, String) {
    match Command::new("systemctl")
        .args(["--user", "is-active", unit_name])
        .output()
    {
        Ok(output) => {
            let detail = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let running = detail == "active";
            (
                running,
                if detail.is_empty() {
                    String::from_utf8_lossy(&output.stderr).trim().to_string()
                } else {
                    detail
                },
            )
        }
        Err(error) => (false, format!("systemctl unavailable: {error}")),
    }
}

fn run_checked(program: &str, args: &[&str]) -> Result<()> {
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("run {program}"))?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("exit {}", output.status)
    };
    bail!("{program} {} failed: {detail}", args.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_label_is_platform_specific() {
        let label = default_label_for_platform();
        if cfg!(target_os = "macos") {
            assert_eq!(label, DEFAULT_DAEMON_LABEL);
        } else {
            assert_eq!(label, DEFAULT_SYSTEMD_LABEL);
        }
    }
}
