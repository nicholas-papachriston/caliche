use super::lifecycle::start_gateway_runtime;
use super::process::{cleanup_dead, is_process_alive, stop_pid};
use super::registry::{load_registry, logs_dir, normalize_cwd, save_registry};
use super::types::{ForwardProviderKind, ForwardSessionRecord, ForwardStatus};
use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;
use tokio::time::sleep;

const FORWARD_URL_WAIT_ATTEMPTS: usize = 20;
const FORWARD_URL_WAIT_INTERVAL: Duration = Duration::from_secs(1);

pub async fn start_gateway_forwarding(
    cwd: Option<&Path>,
    requested_port: Option<u16>,
    provider: Option<ForwardProviderKind>,
) -> Result<ForwardStatus> {
    let provider = provider.unwrap_or(ForwardProviderKind::CloudflareQuick);
    let runtime = start_gateway_runtime(cwd, requested_port).await?;

    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    let row_idx = registry
        .entries
        .iter()
        .position(|e| e.cwd == runtime.entry.cwd)
        .ok_or_else(|| anyhow!("gateway runtime entry not found after startup"))?;

    let existing_forwarding = registry.entries[row_idx].forwarding.clone();
    if let Some(existing) = &existing_forwarding
        && is_process_alive(existing.tunnel_pid).await
        && existing.provider == provider.as_str()
    {
        return Ok(forward_status_from_record(
            registry.entries[row_idx].cwd.clone(),
            provider,
            existing.clone(),
        )
        .await);
    }
    if let Some(existing) = &existing_forwarding {
        let _ = stop_pid(existing.tunnel_pid).await;
    }

    let row_cwd = registry.entries[row_idx].cwd.clone();
    let row_port = registry.entries[row_idx].port;
    let session = match provider {
        ForwardProviderKind::CloudflareQuick => {
            start_cloudflare_quick_tunnel(&row_cwd, row_port).await?
        }
    };
    registry.entries[row_idx].forwarding = Some(session.clone());
    let cwd_for_status = row_cwd;
    save_registry(&registry).await?;

    Ok(forward_status_from_record(cwd_for_status, provider, session).await)
}

pub async fn stop_gateway_forwarding(cwd: Option<&Path>) -> Result<bool> {
    let cwd = normalize_cwd(cwd)?;
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    let mut changed = false;

    if let Some(row) = registry.entries.iter_mut().find(|e| e.cwd == cwd) {
        if let Some(forwarding) = row.forwarding.take() {
            let _ = stop_pid(forwarding.tunnel_pid).await;
            changed = true;
        }
    }

    if changed {
        save_registry(&registry).await?;
    }
    Ok(changed)
}

pub async fn restart_gateway_forwarding(
    cwd: Option<&Path>,
    requested_port: Option<u16>,
    provider: Option<ForwardProviderKind>,
) -> Result<ForwardStatus> {
    let cwd = normalize_cwd(cwd)?;
    let _ = stop_gateway_forwarding(Some(Path::new(&cwd))).await;
    start_gateway_forwarding(Some(Path::new(&cwd)), requested_port, provider).await
}

pub async fn status_gateway_forwarding(cwd: Option<&Path>) -> Result<Option<ForwardStatus>> {
    let cwd = normalize_cwd(cwd)?;
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    save_registry(&registry).await?;

    let Some(row) = registry.entries.iter().find(|e| e.cwd == cwd) else {
        return Ok(None);
    };
    let Some(forwarding) = &row.forwarding else {
        return Ok(None);
    };
    let provider = ForwardProviderKind::parse(&forwarding.provider)
        .ok_or_else(|| anyhow!("unknown forwarding provider '{}'", forwarding.provider))?;
    Ok(Some(
        forward_status_from_record(row.cwd.clone(), provider, forwarding.clone()).await,
    ))
}

pub async fn list_gateway_forwardings() -> Result<Vec<ForwardStatus>> {
    let mut registry = load_registry().await?;
    cleanup_dead(&mut registry).await;
    save_registry(&registry).await?;

    let mut out = Vec::new();
    for row in &registry.entries {
        let Some(forwarding) = &row.forwarding else {
            continue;
        };
        let Some(provider) = ForwardProviderKind::parse(&forwarding.provider) else {
            continue;
        };
        out.push(forward_status_from_record(row.cwd.clone(), provider, forwarding.clone()).await);
    }
    Ok(out)
}

pub fn cloudflare_tunnel_install_instructions() -> String {
    if cfg!(target_os = "macos") {
        "Install cloudflared with Homebrew:\n  brew install cloudflare/cloudflare/cloudflared"
            .to_string()
    } else if cfg!(target_os = "linux") {
        "Install cloudflared (example for Debian/Ubuntu):\n  sudo apt-get update && sudo apt-get install -y cloudflared".to_string()
    } else if cfg!(target_os = "windows") {
        "Install cloudflared with winget:\n  winget install Cloudflare.cloudflared".to_string()
    } else {
        "Install cloudflared from:\n  https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/".to_string()
    }
}

pub(super) async fn wait_for_forwarding_public_url(cwd: Option<&Path>) -> Result<String> {
    let mut initial = status_gateway_forwarding(cwd).await?;
    if initial.is_none() {
        // Fallback for edge cases where CWD resolution differs from the registry entry.
        // If there is exactly one forwarding session, prefer it.
        let all = list_gateway_forwardings().await?;
        if all.len() == 1 {
            initial = all.into_iter().next();
        }
    }
    let Some(initial) = initial else {
        return Err(anyhow!(
            "no active forwarding session for this directory; run '/gateway forward start' (or pass '--dir <path>')"
        ));
    };
    if let Some(url) = initial.public_url {
        return Ok(url);
    }

    for _ in 0..FORWARD_URL_WAIT_ATTEMPTS {
        sleep(FORWARD_URL_WAIT_INTERVAL).await;
        if let Some(status) = status_gateway_forwarding(cwd).await?
            && let Some(url) = status.public_url
        {
            return Ok(url);
        }
    }

    Err(anyhow!(
        "forwarding URL is not ready yet after waiting {} seconds; run '/gateway forward status' or '/gateway forward restart --provider cloudflare' and retry",
        FORWARD_URL_WAIT_ATTEMPTS
    ))
}

async fn forward_status_from_record(
    cwd: String,
    provider: ForwardProviderKind,
    mut session: ForwardSessionRecord,
) -> ForwardStatus {
    if session.public_url.is_none() {
        let log_path = Path::new(&session.log_path);
        session.public_url = extract_cloudflare_url_from_log(log_path);
    }
    let running = is_process_alive(session.tunnel_pid).await;
    ForwardStatus {
        cwd,
        provider,
        tunnel_pid: session.tunnel_pid,
        local_port: session.local_port,
        public_url: session.public_url,
        log_path: session.log_path,
        running,
    }
}

async fn start_cloudflare_quick_tunnel(cwd: &str, local_port: u16) -> Result<ForwardSessionRecord> {
    ensure_cloudflared_available().await?;

    let log_path = next_forwarding_log_path()?;
    let stdout = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open forwarding log {}", log_path.display()))?;
    let stderr = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("failed to open forwarding log {}", log_path.display()))?;

    // Spawn the detached tunnel process via std::process so we can drop the
    // handle without triggering tokio's signal reaper for a long-lived child.
    let mut cmd = std::process::Command::new("cloudflared");
    cmd.arg("tunnel")
        .arg("--no-autoupdate")
        .arg("--url")
        .arg(format!("http://127.0.0.1:{local_port}"))
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }

    let child = cmd
        .spawn()
        .context("failed to start cloudflared quick tunnel")?;
    let pid = child.id();

    sleep(Duration::from_millis(1200)).await;
    if !is_process_alive(pid).await {
        return Err(anyhow!(
            "cloudflared exited immediately after startup (pid {}). Check forwarding log under ~/.caliche/logs/gateway",
            pid
        ));
    }
    let public_url = extract_cloudflare_url_from_log(&log_path);

    Ok(ForwardSessionRecord {
        provider: ForwardProviderKind::CloudflareQuick.as_str().to_string(),
        tunnel_pid: pid,
        started_at: Utc::now().to_rfc3339(),
        local_port,
        log_path: log_path.display().to_string(),
        public_url,
    })
}

fn next_forwarding_log_path() -> Result<PathBuf> {
    let dir = logs_dir()?;
    std::fs::create_dir_all(&dir).context("failed to create forwarding log directory")?;
    Ok(dir.join(format!(
        "forward-{}.log",
        Utc::now().format("%Y%m%d-%H%M%S")
    )))
}

fn extract_cloudflare_url_from_log(log_path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(log_path).ok()?;
    for token in content.split_whitespace() {
        if token.contains("trycloudflare.com")
            && (token.starts_with("https://") || token.starts_with("http://"))
        {
            return Some(
                token
                    .trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == ';')
                    .to_string(),
            );
        }
    }
    extract_url_by_marker(&content, "trycloudflare.com")
}

fn extract_url_by_marker(content: &str, marker: &str) -> Option<String> {
    let marker_idx = content.find(marker)?;
    let bytes = content.as_bytes();
    let mut start = marker_idx;
    while start > 0 && !is_url_boundary(bytes[start - 1] as char) {
        start -= 1;
    }
    let mut end = marker_idx + marker.len();
    while end < bytes.len() && !is_url_boundary(bytes[end] as char) {
        end += 1;
    }
    let candidate = content.get(start..end)?.trim_matches(|c: char| {
        c == '"'
            || c == '\''
            || c == ','
            || c == ';'
            || c == '{'
            || c == '}'
            || c == '('
            || c == ')'
    });
    if candidate.starts_with("https://") || candidate.starts_with("http://") {
        return Some(candidate.to_string());
    }
    None
}

fn is_url_boundary(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '"' | '\'' | ',' | ';' | '{' | '}' | '(' | ')' | '[' | ']' | '<' | '>'
        )
}

async fn ensure_cloudflared_available() -> Result<()> {
    if is_cloudflared_available().await {
        return Ok(());
    }
    let _ = attempt_install_cloudflared().await;
    if is_cloudflared_available().await {
        return Ok(());
    }
    Err(anyhow!(
        "cloudflared not found.\n{}",
        cloudflare_tunnel_install_instructions()
    ))
}

async fn is_cloudflared_available() -> bool {
    Command::new("cloudflared")
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .await
        .map(|s| s.success())
        .unwrap_or(false)
}

async fn attempt_install_cloudflared() -> bool {
    if cfg!(target_os = "macos") {
        let primary = Command::new("brew")
            .args(["install", "cloudflare/cloudflare/cloudflared"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);
        if primary {
            return true;
        }
        return Command::new("brew")
            .args(["install", "cloudflared"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);
    }
    if cfg!(target_os = "linux") {
        return Command::new("sudo")
            .args(["-n", "apt-get", "install", "-y", "cloudflared"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);
    }
    if cfg!(target_os = "windows") {
        return Command::new("winget")
            .args(["install", "Cloudflare.cloudflared"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);
    }
    false
}
