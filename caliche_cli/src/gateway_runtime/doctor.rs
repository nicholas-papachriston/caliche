use super::process::{cleanup_dead, find_listener_pid_for_port, is_process_alive};
use super::registry::{load_registry, normalize_cwd, save_registry};
use super::types::{GatewayDoctorRecord, GatewayRuntimeEntry};
use anyhow::{Result, anyhow};
use std::path::Path;

pub async fn doctor_gateway(
    cwd: Option<&Path>,
    include_all: bool,
) -> Result<Vec<GatewayDoctorRecord>> {
    let cwd = normalize_cwd(cwd)?;
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    save_registry(&registry).await?;

    let candidates: Vec<GatewayRuntimeEntry> = if include_all {
        registry.entries.clone()
    } else {
        registry
            .entries
            .iter()
            .find(|entry| entry.cwd == cwd)
            .cloned()
            .map(|entry| vec![entry])
            .or_else(|| {
                if registry.entries.len() == 1 {
                    registry.entries.first().cloned().map(|entry| vec![entry])
                } else {
                    None
                }
            })
            .ok_or_else(|| anyhow!("no gateway runtime registered for directory: {cwd}"))?
    };

    let mut out = Vec::new();
    for entry in candidates {
        let listener_pid = find_listener_pid_for_port(entry.port).await;
        let forwarding_pid = entry.forwarding.as_ref().map(|f| f.tunnel_pid);
        let forwarding_running = match forwarding_pid {
            Some(pid) => Some(is_process_alive(pid).await),
            None => None,
        };
        let recorded_running = is_process_alive(entry.pid).await;
        out.push(GatewayDoctorRecord {
            cwd: entry.cwd.clone(),
            role: entry.role.clone(),
            port: entry.port,
            recorded_pid: entry.pid,
            recorded_running,
            listener_pid,
            listener_matches_recorded: listener_pid.map(|pid| pid == entry.pid),
            forwarding_pid,
            forwarding_running,
            forwarding_url: entry.forwarding.as_ref().and_then(|f| f.public_url.clone()),
            runtime_log_path: entry.log_path.clone(),
            forwarding_log_path: entry.forwarding.as_ref().map(|f| f.log_path.clone()),
        });
    }
    Ok(out)
}
