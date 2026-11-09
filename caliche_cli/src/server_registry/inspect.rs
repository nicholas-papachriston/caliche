//! Prove a recorded pid is still the Caliche server that wrote the record.
//!
//! Stop signals a process only after every check succeeds. A dead pid is stale.
//! A live pid whose executable, start token, `--server` argument, or listen port
//! does not match is not signaled.

use super::{SERVER_EXE_NAME, ServerOrigin, ServerRecord};
#[cfg(not(target_os = "windows"))]
use crate::process_liveness::compare_process_birth;
#[cfg(target_os = "windows")]
use crate::process_liveness::windows_process_match;
use crate::process_liveness::{ProcessMatch, REASON_TOKEN_MISMATCH};
#[cfg(all(unix, not(target_os = "linux")))]
use caliche::security::SystemBinary;
#[cfg(target_os = "linux")]
use std::collections::HashSet;
#[cfg(target_os = "linux")]
use std::fs;
#[cfg(target_os = "linux")]
use std::path::Path;

/// The flag a foreground or daemon server process is started with.
const SERVER_FLAG: &str = "--server";
pub(super) const REASON_RECORDED_EXE: &str = "record executable is not caliche";
pub(super) const REASON_EXE: &str = "process executable is not caliche";
pub(super) const REASON_CMDLINE: &str = "process was not started with --server";
pub(super) const REASON_PORT: &str = "pid is not listening on the recorded port";
pub(super) const REASON_PORT_ZERO: &str = "record port is not stoppable";

/// Result of checking one record against the live process table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Inspection {
    /// Pid is gone. The record may be removed. Do not signal.
    Stale,
    /// Pid is still the recorded Caliche server.
    Owned { token: String },
    /// Pid is alive and proven not to be that server. Do not signal. Keep the record.
    Foreign { reason: String },
    /// The pid is alive and its creation token is not the recorded one.
    ///
    /// Stop removes the record and does not signal. The `reason` text is for
    /// the operator. Deletion matches this variant, so rewording the text
    /// does not change which records are removed. Only Linux and Windows
    /// tokens prove reuse, so other Unix systems never build this variant.
    #[cfg_attr(
        all(unix, not(target_os = "linux")),
        expect(dead_code, reason = "a ps start-time mismatch does not prove reuse")
    )]
    Recycled { reason: String },
    /// Ownership could not be proven. Do not signal and keep the record.
    Unproven { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Liveness {
    Alive,
    Dead,
    /// Exists, but this user cannot signal it.
    Foreign,
    Unreadable {
        reason: String,
    },
}

pub(super) fn inspect_live(record: &ServerRecord) -> Inspection {
    if record.pid < 2 {
        return Inspection::Foreign {
            reason: "pid is not a stoppable process".to_string(),
        };
    }
    if record.port == 0 {
        return Inspection::Foreign {
            reason: REASON_PORT_ZERO.to_string(),
        };
    }
    if let Err(inspection) = platform_same_process(record) {
        return inspection;
    }
    if !recorded_exe_is_server(&record.exe) {
        return Inspection::Foreign {
            reason: REASON_RECORDED_EXE.to_string(),
        };
    }
    match exe_file_name(record.pid) {
        Ok(name) if exe_name_is_server(&name) => {}
        Ok(_) => {
            return Inspection::Foreign {
                reason: REASON_EXE.to_string(),
            };
        }
        Err(reason) => return Inspection::Unproven { reason },
    }
    if matches!(
        record.origin,
        ServerOrigin::Foreground | ServerOrigin::Daemon
    ) {
        match command_has_server_flag(record.pid) {
            Ok(true) => {}
            Ok(false) => {
                return Inspection::Foreign {
                    reason: REASON_CMDLINE.to_string(),
                };
            }
            Err(reason) => return Inspection::Unproven { reason },
        }
    }
    match pid_listens_on(record.pid, record.port) {
        Ok(true) => Inspection::Owned {
            token: record.start_token.clone(),
        },
        Ok(false) => Inspection::Unproven {
            reason: REASON_PORT.to_string(),
        },
        Err(reason) => Inspection::Unproven { reason },
    }
}

pub(super) fn current_start_token() -> Result<String, String> {
    read_start_token(std::process::id())
}

#[cfg(target_os = "windows")]
fn platform_same_process(record: &ServerRecord) -> Result<(), Inspection> {
    match windows_process_match(record.pid, &record.start_token) {
        ProcessMatch::AliveOriginal => Ok(()),
        ProcessMatch::Dead => Err(Inspection::Stale),
        ProcessMatch::Recycled => Err(token_mismatch()),
        ProcessMatch::Unreadable { reason } => Err(Inspection::Unproven { reason }),
    }
}

#[cfg(not(target_os = "windows"))]
fn platform_same_process(record: &ServerRecord) -> Result<(), Inspection> {
    match liveness(record.pid) {
        Liveness::Dead => return Err(Inspection::Stale),
        Liveness::Foreign => {
            return Err(Inspection::Foreign {
                reason: format!("pid {} is not signalable by this user", record.pid),
            });
        }
        Liveness::Unreadable { reason } => return Err(Inspection::Unproven { reason }),
        Liveness::Alive => {}
    }
    let live_token = match read_start_token(record.pid) {
        Ok(token) => token,
        Err(reason) => return Err(Inspection::Unproven { reason }),
    };
    match compare_process_birth(&live_token, &record.start_token) {
        ProcessMatch::AliveOriginal => Ok(()),
        ProcessMatch::Recycled => Err(token_mismatch()),
        ProcessMatch::Dead => Err(Inspection::Stale),
        ProcessMatch::Unreadable { reason } => Err(Inspection::Unproven { reason }),
    }
}

/// A Linux `starttime` and a Windows creation `FILETIME` are fixed for the
/// life of a process, so a different value proves the pid was reused.
#[cfg(not(all(unix, not(target_os = "linux"))))]
fn token_mismatch() -> Inspection {
    Inspection::Recycled {
        reason: REASON_TOKEN_MISMATCH.to_string(),
    }
}

/// A `ps` start time is text. A record written by an older build holds it in
/// that caller's time zone and locale, so a mismatch does not prove the pid
/// was reused. Keep the record and do not signal.
#[cfg(all(unix, not(target_os = "linux")))]
fn token_mismatch() -> Inspection {
    Inspection::Foreign {
        reason: REASON_TOKEN_MISMATCH.to_string(),
    }
}

fn recorded_exe_is_server(exe: &str) -> bool {
    exe_name_is_server(exe)
}

/// True for `caliche` and `caliche.exe`, ignoring ASCII case.
///
/// Separators are `/` and `\`, so a Windows path compares the same on Linux.
pub(super) fn exe_name_is_server(exe: &str) -> bool {
    let file = exe.rsplit(['/', '\\']).next().unwrap_or(exe);
    let stem = match file.rsplit_once('.') {
        Some((stem, ext)) if ext.eq_ignore_ascii_case("exe") => stem,
        _ => file,
    };
    stem.eq_ignore_ascii_case(SERVER_EXE_NAME)
}

#[cfg(unix)]
fn liveness(pid: u32) -> Liveness {
    if pid < 2 {
        return Liveness::Dead;
    }
    // Signal 0 only checks that the process exists.
    let Err(err) = crate::process_liveness::signal_pid(pid, 0) else {
        return Liveness::Alive;
    };
    match err.raw_os_error() {
        Some(code) if code == libc::ESRCH => Liveness::Dead,
        Some(code) if code == libc::EPERM => Liveness::Foreign,
        _ => Liveness::Unreadable {
            reason: format!("could not check pid {pid}: {err}"),
        },
    }
}

#[cfg(all(not(unix), not(target_os = "windows")))]
fn liveness(pid: u32) -> Liveness {
    let _ = pid;
    Liveness::Unreadable {
        reason: "process checks are unsupported on this platform".to_string(),
    }
}

#[cfg(target_os = "linux")]
fn read_start_token(pid: u32) -> Result<String, String> {
    let path = format!("/proc/{pid}/stat");
    let stat = fs::read_to_string(&path).map_err(|err| format!("read {path}: {err}"))?;
    parse_start_token(&stat).ok_or_else(|| format!("pid {pid} stat has no start token"))
}

#[cfg(target_os = "linux")]
pub(super) fn parse_start_token(stat: &str) -> Option<String> {
    let end = stat.rfind(')')?;
    // `/proc/pid/stat` field 22 is starttime. Field 2 is comm, so the index
    // after the closing parenthesis is 19.
    stat[end + 1..]
        .split_whitespace()
        .nth(19)
        .map(str::to_owned)
}

/// The system `ps`, found by absolute path so a planted `ps` on `PATH`
/// never runs.
#[cfg(all(unix, not(target_os = "linux")))]
fn system_ps() -> Result<std::process::Command, String> {
    SystemBinary::Ps
        .path()
        .map(std::process::Command::new)
        .ok_or_else(|| "ps is not in a system directory".to_string())
}

/// `ps -o lstart=` in the C locale and UTC, so a daemon with no locale and a
/// shell with one read the same token for the same pid.
#[cfg(all(unix, not(target_os = "linux")))]
pub(super) fn start_token_command(pid: u32) -> Result<std::process::Command, String> {
    let mut command = system_ps()?;
    command
        .args(["-p", &pid.to_string(), "-o", "lstart="])
        .env("LC_ALL", "C")
        .env("TZ", "UTC");
    Ok(command)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn read_start_token(pid: u32) -> Result<String, String> {
    let output = start_token_command(pid)?
        .output()
        .map_err(|err| format!("ps failed: {err}"))?;
    if !output.status.success() {
        return Err(format!("ps could not read pid {pid}"));
    }
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if token.is_empty() {
        return Err(format!("pid {pid} has no start time"));
    }
    Ok(token)
}

#[cfg(target_os = "windows")]
fn read_start_token(pid: u32) -> Result<String, String> {
    crate::process_liveness::creation_token(pid)
}

#[cfg(all(not(unix), not(target_os = "windows")))]
fn read_start_token(pid: u32) -> Result<String, String> {
    let _ = pid;
    Err("process start tokens are unsupported on this platform".to_string())
}

#[cfg(target_os = "linux")]
fn exe_file_name(pid: u32) -> Result<String, String> {
    let link = fs::read_link(format!("/proc/{pid}/exe"))
        .map_err(|err| format!("read exe for pid {pid}: {err}"))?;
    let text = link.to_string_lossy();
    let trimmed = text
        .strip_suffix(" (deleted)")
        .unwrap_or_else(|| text.as_ref());
    Path::new(trimmed)
        .file_name()
        .and_then(|name| name.to_str())
        .map(str::to_owned)
        .ok_or_else(|| format!("pid {pid} executable has no file name"))
}

#[cfg(all(unix, not(target_os = "linux")))]
fn exe_file_name(pid: u32) -> Result<String, String> {
    let output = system_ps()?
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output()
        .map_err(|err| format!("ps failed: {err}"))?;
    if !output.status.success() {
        return Err(format!("ps could not read pid {pid}"));
    }
    let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if name.is_empty() {
        return Err(format!("pid {pid} has no command name"));
    }
    Ok(name)
}

#[cfg(target_os = "windows")]
fn exe_file_name(pid: u32) -> Result<String, String> {
    crate::process_liveness::process_image_name(pid)
}

#[cfg(all(not(unix), not(target_os = "windows")))]
fn exe_file_name(pid: u32) -> Result<String, String> {
    let _ = pid;
    Err("process executable checks are unsupported on this platform".to_string())
}

#[cfg(target_os = "linux")]
fn command_has_server_flag(pid: u32) -> Result<bool, String> {
    let raw = fs::read(format!("/proc/{pid}/cmdline"))
        .map_err(|err| format!("read cmdline for pid {pid}: {err}"))?;
    if raw.is_empty() {
        return Err(format!("pid {pid} has an empty command line"));
    }
    Ok(split_cmdline(&raw).iter().any(|arg| arg == SERVER_FLAG))
}

#[cfg(target_os = "linux")]
fn split_cmdline(raw: &[u8]) -> Vec<String> {
    raw.split(|byte| *byte == 0)
        .filter(|part| !part.is_empty())
        .map(|part| String::from_utf8_lossy(part).into_owned())
        .collect()
}

#[cfg(all(unix, not(target_os = "linux")))]
fn command_has_server_flag(pid: u32) -> Result<bool, String> {
    let output = system_ps()?
        .args(["-p", &pid.to_string(), "-ww", "-o", "args="])
        .output()
        .map_err(|err| format!("ps failed: {err}"))?;
    if !output.status.success() {
        return Err(format!("ps could not read pid {pid}"));
    }
    let line = String::from_utf8_lossy(&output.stdout);
    Ok(line.split_whitespace().any(|arg| arg == SERVER_FLAG))
}

#[cfg(target_os = "windows")]
fn command_has_server_flag(pid: u32) -> Result<bool, String> {
    crate::process_liveness::process_has_argument(pid, SERVER_FLAG)
}

#[cfg(all(not(unix), not(target_os = "windows")))]
fn command_has_server_flag(pid: u32) -> Result<bool, String> {
    let _ = pid;
    Err("process command checks are unsupported on this platform".to_string())
}

#[cfg(target_os = "linux")]
fn pid_listens_on(pid: u32, port: u16) -> Result<bool, String> {
    let mut inodes = HashSet::new();
    let mut readable = false;
    for name in ["tcp", "tcp6"] {
        let path = format!("/proc/{pid}/net/{name}");
        match fs::read_to_string(&path) {
            Ok(table) => {
                readable = true;
                inodes.extend(parse_listen_inodes(&table, port));
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(format!("read {path}: {err}")),
        }
    }
    if !readable {
        return Err(format!("could not read listen sockets for pid {pid}"));
    }
    if inodes.is_empty() {
        return Ok(false);
    }
    fd_dir_has_inode(Path::new(&format!("/proc/{pid}/fd")), &inodes)
}

#[cfg(target_os = "linux")]
pub(super) fn parse_listen_inodes(table: &str, port: u16) -> Vec<String> {
    let mut inodes = Vec::new();
    for line in table.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let Some(local) = fields.get(1) else {
            continue;
        };
        let Some(state) = fields.get(3) else {
            continue;
        };
        if !state.eq_ignore_ascii_case("0A") {
            continue;
        }
        let Some(port_hex) = local.rsplit(':').next() else {
            continue;
        };
        let Ok(parsed) = u16::from_str_radix(port_hex, 16) else {
            continue;
        };
        if parsed != port {
            continue;
        }
        if let Some(inode) = fields.get(9) {
            inodes.push((*inode).to_string());
        }
    }
    inodes
}

#[cfg(target_os = "linux")]
pub(super) fn fd_dir_has_inode(fd_dir: &Path, inodes: &HashSet<String>) -> Result<bool, String> {
    let entries =
        fs::read_dir(fd_dir).map_err(|err| format!("read {}: {err}", fd_dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|err| format!("read {}: {err}", fd_dir.display()))?;
        let Ok(target) = fs::read_link(entry.path()) else {
            continue;
        };
        let text = target.to_string_lossy();
        let Some(inode) = text
            .strip_prefix("socket:[")
            .and_then(|rest| rest.strip_suffix(']'))
        else {
            continue;
        };
        if inodes.contains(inode) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(all(unix, not(target_os = "linux")))]
fn pid_listens_on(pid: u32, port: u16) -> Result<bool, String> {
    const NO_LSOF: &str = "could not verify port ownership because lsof is not installed";
    let Some(lsof) = SystemBinary::Lsof.path() else {
        return Err(NO_LSOF.to_string());
    };
    let output = match std::process::Command::new(lsof)
        .args([
            "-nP",
            "-a",
            "-p",
            &pid.to_string(),
            &format!("-iTCP:{port}"),
            "-sTCP:LISTEN",
            "-Fn",
        ])
        .output()
    {
        Ok(output) => output,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            return Err(NO_LSOF.to_string());
        }
        Err(err) => return Err(format!("lsof failed: {err}")),
    };
    if output.status.success() {
        let marker = format!("p{pid}");
        let text = String::from_utf8_lossy(&output.stdout);
        return Ok(text.lines().any(|line| line.trim() == marker));
    }
    if output.status.code() == Some(1) {
        return Ok(false);
    }
    Err(format!("lsof could not verify pid {pid} on port {port}"))
}

#[cfg(target_os = "windows")]
fn pid_listens_on(pid: u32, port: u16) -> Result<bool, String> {
    crate::process_liveness::process_listens_on(pid, port)
}

#[cfg(all(not(unix), not(target_os = "windows")))]
fn pid_listens_on(pid: u32, port: u16) -> Result<bool, String> {
    let _ = (pid, port);
    Err("port ownership checks are unsupported on this platform".to_string())
}
