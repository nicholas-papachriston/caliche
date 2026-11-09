use super::types::GatewayRuntimeRegistry;
#[cfg(unix)]
use caliche::security::SystemBinary;
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::sleep;

pub(super) async fn cleanup_dead(registry: &mut GatewayRuntimeRegistry) {
    let mut kept = Vec::with_capacity(registry.entries.len());
    for mut entry in std::mem::take(&mut registry.entries) {
        if !is_process_alive(entry.pid).await {
            if let Some(recovered_pid) = find_listener_pid_for_port(entry.port).await {
                entry.pid = recovered_pid;
            } else if let Some(forwarding) = &entry.forwarding
                && is_process_alive(forwarding.tunnel_pid).await
            {
                // Keep the entry so forwarding can still be managed/stopped.
            } else {
                continue;
            }
        }
        if let Some(forwarding) = &entry.forwarding
            && !is_process_alive(forwarding.tunnel_pid).await
        {
            entry.forwarding = None;
        }
        kept.push(entry);
    }
    registry.entries = kept;
}

#[cfg(unix)]
pub(super) async fn find_listener_pid_for_port(port: u16) -> Option<u32> {
    let output = Command::new(SystemBinary::Lsof.path()?)
        .arg("-nP")
        .arg("-tiTCP")
        .arg(port.to_string())
        .arg("-sTCP:LISTEN")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .find_map(|line| line.trim().parse::<u32>().ok())
}

#[cfg(not(unix))]
pub(super) async fn find_listener_pid_for_port(_port: u16) -> Option<u32> {
    None
}

#[cfg(unix)]
pub(super) async fn find_listener_pids_in_gateway_port_range(
    start_port: u16,
    end_port: u16,
) -> Vec<u32> {
    let mut pids = Vec::new();
    for port in start_port..=end_port {
        if let Some(pid) = find_listener_pid_for_port(port).await
            && !pids.contains(&pid)
        {
            pids.push(pid);
        }
    }
    pids
}

#[cfg(not(unix))]
pub(super) async fn find_listener_pids_in_gateway_port_range(
    _start_port: u16,
    _end_port: u16,
) -> Vec<u32> {
    Vec::new()
}

#[cfg(unix)]
pub(super) async fn find_caliche_server_pids() -> Vec<u32> {
    list_process_ids_by_predicate(|command| command.contains("caliche --server")).await
}

#[cfg(not(unix))]
pub(super) async fn find_caliche_server_pids() -> Vec<u32> {
    Vec::new()
}

#[cfg(unix)]
pub(super) async fn find_cloudflared_tunnel_pids() -> Vec<u32> {
    list_process_ids_by_predicate(|command| command.contains("cloudflared tunnel")).await
}

#[cfg(not(unix))]
pub(super) async fn find_cloudflared_tunnel_pids() -> Vec<u32> {
    Vec::new()
}

#[cfg(unix)]
async fn list_process_ids_by_predicate<F>(predicate: F) -> Vec<u32>
where
    F: Fn(&str) -> bool,
{
    let Some(ps) = SystemBinary::Ps.path() else {
        return Vec::new();
    };
    let output = Command::new(ps)
        .arg("-axo")
        .arg("pid=,command=")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await;
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut pids = Vec::new();
    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut parts = trimmed.splitn(2, char::is_whitespace);
        let Some(pid_part) = parts.next() else {
            continue;
        };
        let Some(command_part) = parts.next() else {
            continue;
        };
        let command = command_part.trim();
        if predicate(command)
            && let Ok(pid) = pid_part.trim().parse::<u32>()
            && !pids.contains(&pid)
        {
            pids.push(pid);
        }
    }
    pids
}

pub(super) async fn is_process_alive(pid: u32) -> bool {
    is_process_alive_impl(pid).await
}

#[cfg(unix)]
async fn is_process_alive_impl(pid: u32) -> bool {
    // Signal 0 does not send a signal; it checks process existence/permissions.
    if crate::process_liveness::signal_pid(pid, 0).is_err() {
        return false;
    }
    // On macOS, zombie processes still satisfy kill(pid, 0). Filter them out.
    let Some(ps) = SystemBinary::Ps.path() else {
        return true;
    };
    let status = Command::new(ps)
        .arg("-p")
        .arg(pid.to_string())
        .arg("-o")
        .arg("stat=")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await;
    if let Ok(out) = status
        && out.status.success()
    {
        let stat = String::from_utf8_lossy(&out.stdout);
        if stat.trim().contains('Z') {
            return false;
        }
    }
    true
}

#[cfg(windows)]
async fn is_process_alive_impl(pid: u32) -> bool {
    match Command::new("tasklist")
        .args(["/FI", &format!("PID eq {}", pid), "/NH"])
        .output()
        .await
    {
        Ok(out) => {
            let s = String::from_utf8_lossy(&out.stdout);
            s.contains(&pid.to_string())
        }
        Err(_) => false,
    }
}

#[cfg(not(any(unix, windows)))]
async fn is_process_alive_impl(_pid: u32) -> bool {
    false
}

pub(super) async fn stop_pid(pid: u32) -> bool {
    stop_pid_impl(pid).await
}

#[cfg(unix)]
async fn stop_pid_impl(pid: u32) -> bool {
    // SIGTERM asks the process for a cooperative shutdown.
    let term_sent = crate::process_liveness::signal_pid(pid, libc::SIGTERM).is_ok();
    if !term_sent {
        return false;
    }
    // Some orphaned gateway processes survive SIGTERM in this environment.
    // Escalate to SIGKILL if they are still alive after a short grace period.
    for _ in 0..10 {
        if !is_process_alive(pid).await {
            return true;
        }
        sleep(Duration::from_millis(100)).await;
    }
    // Hard kill of a known lingering process.
    let _ = crate::process_liveness::signal_pid(pid, libc::SIGKILL);
    !is_process_alive(pid).await
}

#[cfg(not(unix))]
async fn stop_pid_impl(_pid: u32) -> bool {
    false
}
