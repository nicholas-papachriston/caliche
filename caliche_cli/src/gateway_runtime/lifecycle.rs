use super::api::{
    call_gateway_api, gateway_supports_cluster_routes, register_worker_with_main_gateway,
    resolve_main_gateway_url,
};
use super::logs::{latest_log_file_path, read_log_tail};
use super::process::{
    cleanup_dead, find_caliche_server_pids, find_cloudflared_tunnel_pids,
    find_listener_pid_for_port, find_listener_pids_in_gateway_port_range, is_process_alive,
    stop_pid,
};
use super::registry::{
    load_registry, next_log_path, normalize_cwd, runtime_db_pool, save_registry,
};
use super::types::{
    GatewayLlmConfigStatus, GatewayLogsSnapshot, GatewayRouteInfo, GatewayRuntimeEntry,
    GatewayRuntimeRegistry, GatewayRuntimeRole, GatewayRuntimeStatus,
};
use crate::storage::credentials::SecureCredentialManager;
use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use serde_json::json;
use sqlx::Row;
use std::fs::OpenOptions;
use std::path::Path;
use std::process::Stdio;
use std::time::Duration;
use tokio::time::sleep;
use uuid::Uuid;

pub async fn start_gateway_runtime(
    cwd: Option<&Path>,
    requested_port: Option<u16>,
) -> Result<GatewayRuntimeStatus> {
    let cwd = normalize_cwd(cwd)?;
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    save_registry(&registry).await?;

    if let Some(existing) = existing_alive_entry(&registry, &cwd).await
        && let Some(status) = recycle_stale_runtime(&mut registry, existing, &cwd).await?
    {
        return Ok(status);
    }

    let running_main = find_running_main(&registry).await;
    let role = if running_main.is_some() {
        GatewayRuntimeRole::Worker
    } else {
        GatewayRuntimeRole::Main
    };

    let port = choose_port(&registry, requested_port)?;
    let log_path = next_log_path()?;
    let exe = std::env::current_exe().context("failed to resolve current executable path")?;
    let llm_config = resolve_gateway_llm_config(&cwd).await?;
    let main_gateway_url = running_main
        .as_ref()
        .map(|main| format!("http://127.0.0.1:{}", main.port))
        .or_else(|| Some(format!("http://127.0.0.1:{port}")));
    let agent_id = Uuid::now_v7().to_string();

    let pid = spawn_gateway_process(
        &exe,
        port,
        role,
        &agent_id,
        &cwd,
        &log_path,
        &llm_config,
        main_gateway_url.as_deref(),
    )?;
    wait_until_gateway_alive(pid).await?;

    let entry = GatewayRuntimeEntry {
        agent_id,
        cwd: cwd.clone(),
        pid,
        port,
        role: role.as_str().to_string(),
        main_gateway_url,
        started_at: Utc::now().to_rfc3339(),
        log_path: log_path.display().to_string(),
        forwarding: None,
    };

    registry.entries.retain(|e| e.cwd != cwd);
    registry.entries.push(entry.clone());
    save_registry(&registry).await?;
    register_worker_or_rollback(&entry, role, pid, &cwd).await?;

    let running = is_process_alive(pid).await;
    Ok(GatewayRuntimeStatus { entry, running })
}

async fn existing_alive_entry(
    registry: &GatewayRuntimeRegistry,
    cwd: &str,
) -> Option<GatewayRuntimeEntry> {
    for entry in &registry.entries {
        if entry.cwd == cwd && is_process_alive(entry.pid).await {
            return Some(entry.clone());
        }
    }
    None
}

async fn recycle_stale_runtime(
    registry: &mut GatewayRuntimeRegistry,
    existing: GatewayRuntimeEntry,
    cwd: &str,
) -> Result<Option<GatewayRuntimeStatus>> {
    if gateway_supports_cluster_routes(existing.port).await {
        return Ok(Some(GatewayRuntimeStatus {
            entry: existing,
            running: true,
        }));
    }
    let _ = stop_pid(existing.pid).await;
    registry.entries.retain(|e| e.cwd != cwd);
    save_registry(registry).await?;
    Ok(None)
}

async fn find_running_main(registry: &GatewayRuntimeRegistry) -> Option<GatewayRuntimeEntry> {
    for entry in &registry.entries {
        if entry.role == GatewayRuntimeRole::Main.as_str() && is_process_alive(entry.pid).await {
            return Some(entry.clone());
        }
    }
    None
}

fn spawn_gateway_process(
    exe: &Path,
    port: u16,
    role: GatewayRuntimeRole,
    agent_id: &str,
    cwd: &str,
    log_path: &Path,
    llm_config: &GatewayLlmConfigStatus,
    main_gateway_url: Option<&str>,
) -> Result<u32> {
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open log file {}", log_path.display()))?;
    let stderr = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .with_context(|| format!("failed to open log file {}", log_path.display()))?;

    let mut cmd = gateway_command(exe, port, role, agent_id, cwd, llm_config, main_gateway_url);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let child = cmd
        .spawn()
        .context("failed to start gateway runtime process")?;
    Ok(child.id())
}

/// The gateway runtime command. The child inherits this process's environment,
/// and also gets the credentials held in the in-process store, which are not in
/// that environment. The store never holds a name the shell exported, so a shell
/// value still reaches the child unchanged.
fn gateway_command(
    exe: &Path,
    port: u16,
    role: GatewayRuntimeRole,
    agent_id: &str,
    cwd: &str,
    llm_config: &GatewayLlmConfigStatus,
    main_gateway_url: Option<&str>,
) -> std::process::Command {
    let mut cmd = std::process::Command::new(exe);
    cmd.arg("--server")
        .arg("--port")
        .arg(port.to_string())
        .envs(caliche_contracts::credential_env::snapshot())
        .env("CALICHE_GATEWAY_ROLE", role.as_str())
        .env("CALICHE_GATEWAY_AGENT_ID", agent_id)
        .current_dir(cwd);
    if let Some(provider) = &llm_config.effective_provider {
        cmd.env("AGENT_PROVIDER", provider);
    }
    if let Some(model) = &llm_config.effective_model {
        cmd.env("AGENT_MODEL", model);
    }
    if let Some(main_url) = main_gateway_url {
        cmd.env("CALICHE_MAIN_GATEWAY_URL", main_url);
    }
    cmd
}

async fn wait_until_gateway_alive(pid: u32) -> Result<()> {
    let mut alive_after_window = false;
    for _ in 0..30 {
        sleep(Duration::from_millis(100)).await;
        if is_process_alive(pid).await {
            alive_after_window = true;
            continue;
        }
        return Err(anyhow!(
            "gateway runtime exited during startup (pid {}). Check logs under ~/.caliche/logs/gateway",
            pid
        ));
    }
    if !alive_after_window {
        return Err(anyhow!(
            "gateway runtime failed to reach a healthy startup state (pid {}). Check logs under ~/.caliche/logs/gateway",
            pid
        ));
    }
    Ok(())
}

async fn register_worker_or_rollback(
    entry: &GatewayRuntimeEntry,
    role: GatewayRuntimeRole,
    pid: u32,
    cwd: &str,
) -> Result<()> {
    if role != GatewayRuntimeRole::Worker {
        return Ok(());
    }
    if let Err(err) = register_worker_with_main_gateway(entry).await {
        let _ = stop_pid(pid).await;
        let mut cleanup = load_registry().await?;
        cleanup.entries.retain(|e| e.cwd != cwd);
        save_registry(&cleanup).await?;
        return Err(anyhow!(
            "worker failed to register with main gateway: {err}"
        ));
    }
    Ok(())
}

pub async fn stop_gateway_runtime(cwd: Option<&Path>) -> Result<bool> {
    let cwd = normalize_cwd(cwd)?;
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    let before = registry.entries.len();

    let mut stopped_any = false;
    let matches: Vec<GatewayRuntimeEntry> = registry
        .entries
        .iter()
        .filter(|e| e.cwd == cwd)
        .cloned()
        .collect();
    for entry in &matches {
        if let Some(forwarding) = &entry.forwarding {
            let _ = stop_pid(forwarding.tunnel_pid).await;
        }
        if stop_pid(entry.pid).await {
            stopped_any = true;
        }
        // Best-effort cleanup of any orphan listener still bound to this runtime port.
        if let Some(listener_pid) = find_listener_pid_for_port(entry.port).await
            && listener_pid != entry.pid
            && stop_pid(listener_pid).await
        {
            stopped_any = true;
        }
    }
    registry.entries.retain(|e| e.cwd != cwd);
    if registry.entries.len() != before {
        save_registry(&registry).await?;
    }
    Ok(stopped_any || before != registry.entries.len())
}

pub async fn stop_all_gateway_runtimes() -> Result<usize> {
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    let total = registry.entries.len();
    let orphan_pids = find_listener_pids_in_gateway_port_range(3000, 3100).await;
    let server_pids = find_caliche_server_pids().await;
    let tunnel_pids = find_cloudflared_tunnel_pids().await;

    let entries_snapshot: Vec<GatewayRuntimeEntry> = registry.entries.clone();
    for entry in &entries_snapshot {
        if let Some(forwarding) = &entry.forwarding {
            let _ = stop_pid(forwarding.tunnel_pid).await;
        }
        let _ = stop_pid(entry.pid).await;
        if let Some(listener_pid) = find_listener_pid_for_port(entry.port).await
            && listener_pid != entry.pid
        {
            let _ = stop_pid(listener_pid).await;
        }
    }
    for orphan_pid in &orphan_pids {
        let _ = stop_pid(*orphan_pid).await;
    }
    for server_pid in &server_pids {
        let _ = stop_pid(*server_pid).await;
    }
    for tunnel_pid in &tunnel_pids {
        let _ = stop_pid(*tunnel_pid).await;
    }

    registry.entries.clear();
    save_registry(&registry).await?;
    Ok(total + orphan_pids.len() + server_pids.len() + tunnel_pids.len())
}

pub async fn restart_gateway_runtime(
    cwd: Option<&Path>,
    requested_port: Option<u16>,
) -> Result<GatewayRuntimeStatus> {
    let cwd = normalize_cwd(cwd)?;
    let _ = stop_gateway_runtime(Some(Path::new(&cwd))).await;
    start_gateway_runtime(Some(Path::new(&cwd)), requested_port).await
}

pub async fn status_gateway_runtime(cwd: Option<&Path>) -> Result<Option<GatewayRuntimeStatus>> {
    let cwd = normalize_cwd(cwd)?;
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    save_registry(&registry).await?;
    let Some(entry) = registry.entries.iter().find(|e| e.cwd == cwd).cloned() else {
        return Ok(None);
    };
    let running = is_process_alive(entry.pid).await;
    Ok(Some(GatewayRuntimeStatus { entry, running }))
}

pub async fn list_gateway_runtimes() -> Result<Vec<GatewayRuntimeStatus>> {
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    save_registry(&registry).await?;
    let mut out = Vec::with_capacity(registry.entries.len());
    for entry in &registry.entries {
        let running = is_process_alive(entry.pid).await;
        out.push(GatewayRuntimeStatus {
            entry: entry.clone(),
            running,
        });
    }
    Ok(out)
}

pub async fn set_gateway_llm_config(
    cwd: Option<&Path>,
    provider: Option<&str>,
    model: Option<&str>,
) -> Result<GatewayLlmConfigStatus> {
    let cwd = normalize_cwd(cwd)?;
    let provider_clean = provider
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(|v| v.to_lowercase());
    let model_clean = model
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(std::string::ToString::to_string);

    let pool = runtime_db_pool().await?;
    sqlx::query(
        r#"
        INSERT INTO gateway_runtime_settings (cwd, provider, model, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(cwd) DO UPDATE SET
            provider = excluded.provider,
            model = excluded.model,
            updated_at = excluded.updated_at
        "#,
    )
    .bind(&cwd)
    .bind(provider_clean)
    .bind(model_clean)
    .bind(Utc::now().to_rfc3339())
    .execute(&pool)
    .await
    .context("failed to upsert gateway llm settings")?;

    resolve_gateway_llm_config(&cwd).await
}

pub async fn clear_gateway_llm_config(cwd: Option<&Path>) -> Result<GatewayLlmConfigStatus> {
    let cwd = normalize_cwd(cwd)?;
    let pool = runtime_db_pool().await?;
    sqlx::query("DELETE FROM gateway_runtime_settings WHERE cwd = ?")
        .bind(&cwd)
        .execute(&pool)
        .await
        .context("failed to clear gateway llm settings")?;
    resolve_gateway_llm_config(&cwd).await
}

pub async fn gateway_llm_config_status(cwd: Option<&Path>) -> Result<GatewayLlmConfigStatus> {
    let cwd = normalize_cwd(cwd)?;
    resolve_gateway_llm_config(&cwd).await
}

pub async fn read_gateway_logs(
    cwd: Option<&Path>,
    lines: usize,
    include_forwarding: bool,
) -> Result<GatewayLogsSnapshot> {
    let cwd = normalize_cwd(cwd)?;
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    save_registry(&registry).await?;

    let row = registry
        .entries
        .iter()
        .find(|e| e.cwd == cwd)
        .or_else(|| {
            if registry.entries.len() == 1 {
                registry.entries.first()
            } else {
                None
            }
        })
        .ok_or_else(|| anyhow!("no gateway runtime registered for directory: {cwd}"))?;

    let runtime_log_path = row.log_path.clone();
    let runtime_log_tail = read_log_tail(Path::new(&runtime_log_path), lines)
        .with_context(|| format!("failed reading gateway log {}", runtime_log_path))?;

    let (forwarding_log_path, forwarding_log_tail) = if include_forwarding {
        if let Some(forward) = &row.forwarding {
            let path = forward.log_path.clone();
            let tail = read_log_tail(Path::new(&path), lines)
                .with_context(|| format!("failed reading forwarding log {}", path))?;
            (Some(path), Some(tail))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Ok(GatewayLogsSnapshot {
        cwd: row.cwd.clone(),
        runtime_log_path,
        runtime_log_tail,
        forwarding_log_path,
        forwarding_log_tail,
    })
}

pub async fn read_gateway_logs_fallback(
    lines: usize,
    include_forwarding: bool,
) -> Result<GatewayLogsSnapshot> {
    let runtime_log_path = latest_log_file_path("gateway-")
        .ok_or_else(|| anyhow!("no gateway log files found under ~/.caliche/logs/gateway"))?;
    let runtime_log_tail = read_log_tail(Path::new(&runtime_log_path), lines)
        .with_context(|| format!("failed reading gateway log {}", runtime_log_path))?;

    let (forwarding_log_path, forwarding_log_tail) = if include_forwarding {
        if let Some(path) = latest_log_file_path("forward-") {
            let tail = read_log_tail(Path::new(&path), lines)
                .with_context(|| format!("failed reading forwarding log {}", path))?;
            (Some(path), Some(tail))
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    Ok(GatewayLogsSnapshot {
        cwd: "<latest logs>".to_string(),
        runtime_log_path,
        runtime_log_tail,
        forwarding_log_path,
        forwarding_log_tail,
    })
}

pub async fn pin_gateway_route(
    cwd: Option<&Path>,
    route_key: &str,
    target_agent_id: &str,
) -> Result<()> {
    let main_url = resolve_main_gateway_url(cwd).await?;
    let endpoint = format!(
        "{}/api/gateway/cluster/routes/pin",
        main_url.trim_end_matches('/')
    );
    let payload = json!({
        "route_key": route_key,
        "target_agent_id": target_agent_id
    });
    let _ = call_gateway_api("POST", &endpoint, Some(payload.to_string())).await?;
    Ok(())
}

pub async fn unpin_gateway_route(cwd: Option<&Path>, route_key: &str) -> Result<()> {
    let main_url = resolve_main_gateway_url(cwd).await?;
    let endpoint = format!(
        "{}/api/gateway/cluster/routes/unpin",
        main_url.trim_end_matches('/')
    );
    let payload = json!({
        "route_key": route_key
    });
    let _ = call_gateway_api("POST", &endpoint, Some(payload.to_string())).await?;
    Ok(())
}

pub async fn list_gateway_routes(cwd: Option<&Path>) -> Result<Vec<GatewayRouteInfo>> {
    let main_url = resolve_main_gateway_url(cwd).await?;
    let endpoint = format!(
        "{}/api/gateway/cluster/routes",
        main_url.trim_end_matches('/')
    );
    let body = call_gateway_api("GET", &endpoint, None).await?;
    if body.is_empty() {
        return Ok(Vec::new());
    }
    let routes: Vec<GatewayRouteInfo> =
        serde_json::from_slice(&body).context("failed to parse gateway route list")?;
    Ok(routes)
}

async fn resolve_gateway_llm_config(cwd: &str) -> Result<GatewayLlmConfigStatus> {
    let pool = runtime_db_pool().await?;
    let row = sqlx::query("SELECT provider, model FROM gateway_runtime_settings WHERE cwd = ?")
        .bind(cwd)
        .fetch_optional(&pool)
        .await
        .context("failed to query gateway llm settings")?;
    let configured_provider = row
        .as_ref()
        .and_then(|r| r.get::<Option<String>, _>("provider"))
        .filter(|v| !v.trim().is_empty());
    let configured_model = row
        .as_ref()
        .and_then(|r| r.get::<Option<String>, _>("model"))
        .filter(|v| !v.trim().is_empty());

    let mut provider_from_scope = None;
    let mut model_from_scope = None;
    if let Ok(manager) = SecureCredentialManager::new_with_scope(Some(Path::new(cwd))).await {
        provider_from_scope = manager.default_provider().await;
        let provider_for_model = configured_provider
            .clone()
            .or_else(|| provider_from_scope.clone());
        if let Some(provider) = provider_for_model {
            model_from_scope = manager.get_default_model(&provider).await;
        }
    }

    let effective_provider = configured_provider
        .clone()
        .or(provider_from_scope)
        .or_else(|| std::env::var("AGENT_PROVIDER").ok());
    let effective_model = configured_model
        .clone()
        .or(model_from_scope)
        .or_else(|| std::env::var("AGENT_MODEL").ok());

    Ok(GatewayLlmConfigStatus {
        cwd: cwd.to_string(),
        configured_provider,
        configured_model,
        effective_provider,
        effective_model,
    })
}

fn choose_port(registry: &GatewayRuntimeRegistry, requested_port: Option<u16>) -> Result<u16> {
    if let Some(port) = requested_port {
        if !is_port_available(port) {
            return Err(anyhow!("requested port {} is unavailable", port));
        }
        return Ok(port);
    }
    for port in 3000..=3100 {
        if registry.entries.iter().any(|e| e.port == port) {
            continue;
        }
        if is_port_available(port) {
            return Ok(port);
        }
    }
    Err(anyhow!("no available port found in range 3000-3100"))
}

fn is_port_available(port: u16) -> bool {
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliche_test_env::EnvGuard;
    use std::collections::HashSet;
    use std::env::VarError;
    use std::ffi::OsStr;

    fn child_env(name: &str) -> Option<Option<String>> {
        let cmd = gateway_command(
            Path::new("caliche"),
            43_210,
            GatewayRuntimeRole::Worker,
            "agent-1",
            ".",
            &GatewayLlmConfigStatus {
                cwd: ".".to_string(),
                configured_provider: None,
                configured_model: None,
                effective_provider: None,
                effective_model: None,
            },
            None,
        );
        cmd.get_envs()
            .find(|(key, _)| *key == OsStr::new(name))
            .map(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
    }

    async fn keychain_with(provider: &str, key: &str) -> SecureCredentialManager {
        let manager = SecureCredentialManager::new_with_scope(None)
            .await
            .expect("credential manager");
        manager.set_api_key(provider, key).await.expect("store key");
        manager
    }

    #[tokio::test]
    #[serial_test::serial(env)]
    async fn gateway_child_gets_a_stored_key_and_the_parent_env_stays_unset() {
        let home = tempfile::tempdir().expect("tempdir");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { EnvGuard::acquire() };
        env.set("HOME", home.path());
        env.remove("CALICHE_NO_KEYRING");
        env.set("CALICHE_USE_KEYRING", "1");
        env.remove("GOOGLE_API_KEY");
        let manager = keychain_with("google", "keychain-google").await;

        crate::bootstrap::load_startup_credentials(&manager, &HashSet::new()).await;

        assert_eq!(
            child_env("GOOGLE_API_KEY"),
            Some(Some("keychain-google".to_string()))
        );
        assert_eq!(std::env::var("GOOGLE_API_KEY"), Err(VarError::NotPresent));
    }

    // The shell key winning over the keychain is covered by
    // tests/gateway_shell_key.rs: the shell-key set is recorded once per
    // process, so that case needs a process of its own.
}
