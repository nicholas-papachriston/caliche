//! Managed endpoint (sub-server) lifecycle handler.
//!
//! The main caliche_server server can spawn and supervise child
//! `caliche_server` processes, each running in its own workspace directory
//! on a dedicated port.  Configurations survive restarts and endpoints marked
//! `auto_start = true` are re-launched automatically.

use anyhow::Result;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::process::Child;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::AppState;
#[cfg(unix)]
use caliche::security::SystemBinary;

type ApiResult<T> = Result<Json<T>, (StatusCode, String)>;

fn expand_workspace_path(workspace: &str) -> String {
    if workspace == "~" {
        return dirs::home_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| workspace.to_string());
    }

    if let Some(rest) = workspace.strip_prefix("~/") {
        return dirs::home_dir()
            .map(|p| p.join(rest).to_string_lossy().to_string())
            .unwrap_or_else(|| workspace.to_string());
    }

    workspace.to_string()
}

// ── Public API types (mirrored by the UI) ────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateEndpointRequest {
    /// Human-readable name for this endpoint
    pub name: String,
    /// Absolute path to the workspace directory
    pub workspace: String,
    /// Port the child server should listen on (1024–65535)
    pub port: u16,
    /// LLM provider override (e.g. "anthropic")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Model override (e.g. "claude-sonnet-4-6")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Start the endpoint immediately after creation
    #[serde(default)]
    pub auto_start: bool,
}

/// Persisted configuration for a managed endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    pub id: String,
    pub name: String,
    pub workspace: String,
    pub port: u16,
    pub provider: Option<String>,
    pub model: Option<String>,
    /// Re-launch automatically when the main server starts
    pub auto_start: bool,
    pub created_at: DateTime<Utc>,
}

/// Full endpoint status returned by the API (config + runtime).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointInfo {
    pub id: String,
    pub name: String,
    pub workspace: String,
    pub port: u16,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub auto_start: bool,
    /// "stopped" | "starting" | "running" | "failed"
    pub status: String,
    pub pid: Option<u32>,
    /// Base URL of the child server (http://localhost:{port})
    pub url: String,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub stopped_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointListResponse {
    pub endpoints: Vec<EndpointInfo>,
}

// ── Internal manager types ────────────────────────────────────────────────────

struct RuntimeState {
    status: String,
    child: Option<Child>,
    pid: Option<u32>,
    started_at: Option<DateTime<Utc>>,
    stopped_at: Option<DateTime<Utc>>,
    error: Option<String>,
}

struct ManagerInner {
    configs: HashMap<String, EndpointConfig>,
    runtimes: HashMap<String, RuntimeState>,
}

// ── EndpointManager ───────────────────────────────────────────────────────────

/// Manages the full lifecycle of child `caliche_server` processes.
pub struct EndpointManager {
    inner: RwLock<ManagerInner>,
    persistence_path: PathBuf,
    /// Resolved path (or name) of the server binary to spawn.
    server_binary: String,
}

/// The endpoint server command. The child inherits this process's environment,
/// and also gets the credentials held in the in-process store, which are not in
/// that environment. The store never holds a name the shell exported, so a shell
/// value still reaches the child unchanged.
fn endpoint_command(server_binary: &str, config: &EndpointConfig) -> tokio::process::Command {
    let mut cmd = tokio::process::Command::new(server_binary);
    cmd.current_dir(&config.workspace)
        .arg("--port")
        .arg(config.port.to_string())
        .arg("--cors-permissive")
        .envs(caliche_contracts::credential_env::snapshot())
        // Ensure child inherits RUST_LOG if set, defaulting to info.
        .env(
            "RUST_LOG",
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        )
        // Children outlive the parent's wait handle; they are cleaned up
        // explicitly via stop()/stop_all() or OS zombie reaping on exit.
        .kill_on_drop(false);

    if let Some(ref provider) = config.provider {
        cmd.arg("--provider").arg(provider);
    }
    if let Some(ref model) = config.model {
        cmd.arg("--model").arg(model);
    }
    cmd
}

impl EndpointManager {
    /// Create a new manager.  Call [`load_persisted`] right after to restore
    /// any previously-created endpoints.
    pub fn new() -> Arc<Self> {
        let path = dirs::data_dir()
            .map(|d| d.join("caliche").join("endpoints.json"))
            .unwrap_or_else(|| PathBuf::from(".caliche_endpoints.json"));

        // Prefer a sibling binary in the same directory as the running process
        let binary = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.join("caliche_server")))
            .filter(|p| p.exists())
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|| "caliche_server".to_string());

        Arc::new(Self {
            inner: RwLock::new(ManagerInner {
                configs: HashMap::new(),
                runtimes: HashMap::new(),
            }),
            persistence_path: path,
            server_binary: binary,
        })
    }

    /// Load persisted endpoint configs from disk and auto-start any that have
    /// `auto_start = true`.  Also spawns the background health-check monitor.
    /// Returns the number of configs loaded.
    pub async fn load_persisted(self: &Arc<Self>) -> usize {
        if !self.persistence_path.exists() {
            // Still start the health monitor even with no persisted state.
            Self::spawn_health_monitor(Arc::clone(self));
            return 0;
        }
        let data = match tokio::fs::read_to_string(&self.persistence_path).await {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Could not read endpoint store: {}", e);
                Self::spawn_health_monitor(Arc::clone(self));
                return 0;
            }
        };
        let configs: Vec<EndpointConfig> = match serde_json::from_str(&data) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Could not parse endpoint store: {}", e);
                Self::spawn_health_monitor(Arc::clone(self));
                return 0;
            }
        };
        let count = configs.len();
        let auto_start_ids: Vec<String> = configs
            .iter()
            .filter(|c| c.auto_start)
            .map(|c| c.id.clone())
            .collect();

        {
            let mut inner = self.inner.write().await;
            for config in configs {
                inner.runtimes.insert(
                    config.id.clone(),
                    RuntimeState {
                        status: "stopped".to_string(),
                        child: None,
                        pid: None,
                        started_at: None,
                        stopped_at: None,
                        error: None,
                    },
                );
                inner.configs.insert(config.id.clone(), config);
            }
        }

        // Auto-start with a small stagger so we don't hammer the system at once.
        for (i, id) in auto_start_ids.into_iter().enumerate() {
            let mgr = Arc::clone(self);
            tokio::spawn(async move {
                // Stagger: 500 ms base + 250 ms per endpoint
                let delay_ms = 500u64 + (i as u64) * 250;
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                if let Err(e) = mgr.start(&id).await {
                    tracing::warn!("Auto-start failed for endpoint {}: {}", id, e);
                }
            });
        }

        tracing::info!("Loaded {} persisted endpoint configs", count);
        Self::spawn_health_monitor(Arc::clone(self));
        count
    }

    /// Background loop: poll all running children every 10 s for unexpected exits.
    fn spawn_health_monitor(mgr: Arc<Self>) {
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                mgr.check_health().await;
            }
        });
    }

    /// Persist current configs to disk (atomically via temp-file rename).
    async fn persist(&self) {
        let inner = self.inner.read().await;
        let configs: Vec<&EndpointConfig> = inner.configs.values().collect();
        let data = match serde_json::to_string_pretty(&configs) {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!("Failed to serialize endpoint configs: {}", e);
                return;
            }
        };

        if let Some(parent) = self.persistence_path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                tracing::warn!("Failed to create endpoint store dir: {}", e);
                return;
            }
        }

        // Write to a temp file then rename for atomicity.
        let tmp = self.persistence_path.with_extension("json.tmp");
        if let Err(e) = tokio::fs::write(&tmp, &data).await {
            tracing::warn!("Failed to write endpoint store temp file: {}", e);
            return;
        }
        if let Err(e) = tokio::fs::rename(&tmp, &self.persistence_path).await {
            tracing::warn!("Failed to rename endpoint store: {}", e);
        }
    }

    /// Register a new endpoint config.  Does NOT start the process unless
    /// `auto_start` is set.  Returns an error if the port is already claimed
    /// by another endpoint or outside the valid range.
    pub async fn create(&self, req: CreateEndpointRequest) -> Result<EndpointInfo> {
        let workspace = expand_workspace_path(&req.workspace);

        // Validate port range
        if req.port < 1024 {
            anyhow::bail!("Port {} is reserved; use a port >= 1024", req.port);
        }

        // Check for port conflicts with existing endpoints
        {
            let inner = self.inner.read().await;
            if let Some(conflict) = inner.configs.values().find(|c| c.port == req.port) {
                anyhow::bail!(
                    "Port {} is already used by endpoint '{}'",
                    req.port,
                    conflict.name
                );
            }
        }

        let id = Uuid::new_v4().to_string();
        let config = EndpointConfig {
            id: id.clone(),
            name: req.name,
            workspace,
            port: req.port,
            provider: req.provider,
            model: req.model,
            auto_start: req.auto_start,
            created_at: Utc::now(),
        };
        {
            let mut inner = self.inner.write().await;
            inner.runtimes.insert(
                id.clone(),
                RuntimeState {
                    status: "stopped".to_string(),
                    child: None,
                    pid: None,
                    started_at: None,
                    stopped_at: None,
                    error: None,
                },
            );
            inner.configs.insert(id.clone(), config);
        }
        self.persist().await;
        self.get_info(&id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to load endpoint after creation: {}", id))
    }

    /// Spawn the child process for an endpoint.
    ///
    /// The guard check and the "starting" status update happen inside the same
    /// write-lock acquisition, preventing concurrent start races.
    pub async fn start(&self, id: &str) -> Result<EndpointInfo> {
        // Atomically: verify not already running and mark as "starting".
        let config = {
            let mut inner = self.inner.write().await;
            let rt = inner
                .runtimes
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("Endpoint not found: {}", id))?;

            if rt.status == "running" || rt.status == "starting" {
                anyhow::bail!("Endpoint {} is already {}", id, rt.status);
            }

            // Mark as starting while still holding the write lock so no
            // concurrent call can slip through the guard.
            if let Some(rt) = inner.runtimes.get_mut(id) {
                rt.status = "starting".to_string();
                rt.error = None;
            }

            inner
                .configs
                .get(id)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Config not found: {}", id))?
        }; // write lock released here

        // Validate workspace exists before attempting spawn.
        if !std::path::Path::new(&config.workspace).is_dir() {
            let mut inner = self.inner.write().await;
            if let Some(rt) = inner.runtimes.get_mut(id) {
                rt.status = "failed".to_string();
                rt.error = Some(format!(
                    "Workspace directory does not exist: {}",
                    config.workspace
                ));
            }
            anyhow::bail!("Workspace does not exist: {}", config.workspace);
        }

        let mut cmd = endpoint_command(&self.server_binary, &config);

        match cmd.spawn() {
            Ok(child) => {
                let pid = child.id();
                tracing::info!(
                    "Spawned endpoint {} '{}' on port {} (pid {:?})",
                    id,
                    config.name,
                    config.port,
                    pid
                );
                let mut inner = self.inner.write().await;
                if let Some(rt) = inner.runtimes.get_mut(id) {
                    rt.child = Some(child);
                    rt.pid = pid;
                    // Stay "starting" — the health monitor will upgrade to "running"
                    // once the port responds, or downgrade to "failed" if the
                    // process exits before becoming ready.
                    rt.started_at = Some(Utc::now());
                    rt.stopped_at = None;
                }
            }
            Err(e) => {
                tracing::error!("Failed to spawn endpoint {}: {}", id, e);
                let mut inner = self.inner.write().await;
                if let Some(rt) = inner.runtimes.get_mut(id) {
                    rt.status = "failed".to_string();
                    rt.error = Some(e.to_string());
                }
                anyhow::bail!("Failed to spawn process: {}", e);
            }
        }

        self.get_info(id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to load endpoint after start: {}", id))
    }

    /// Gracefully stop an endpoint (SIGTERM → poll for exit → SIGKILL).
    pub async fn stop(&self, id: &str) -> Result<EndpointInfo> {
        let pid = {
            let inner = self.inner.read().await;
            inner
                .runtimes
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("Endpoint not found: {}", id))?
                .pid
        };

        if let Some(pid) = pid {
            // Windows has no SIGTERM; the wait and kill below stop the child.
            #[cfg(not(unix))]
            let _ = pid;
            // SIGTERM on Unix
            #[cfg(unix)]
            {
                if let Some(kill) = SystemBinary::Kill.path() {
                    let _ = tokio::process::Command::new(kill)
                        .args(["-TERM", &pid.to_string()])
                        .status()
                        .await;
                }
            }

            // Poll every 100 ms for up to 5 s for graceful exit.
            for _ in 0..50 {
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                let mut inner = self.inner.write().await;
                if let Some(rt) = inner.runtimes.get_mut(id) {
                    if let Some(ref mut child) = rt.child {
                        if let Ok(Some(_)) = child.try_wait() {
                            rt.status = "stopped".to_string();
                            rt.stopped_at = Some(Utc::now());
                            rt.pid = None;
                            tracing::info!("Endpoint {} stopped gracefully", id);
                            rt.child = None;
                            break;
                        }
                    }
                }
            }

            let stopped_gracefully = {
                let inner = self.inner.read().await;
                inner
                    .runtimes
                    .get(id)
                    .map(|rt| rt.status == "stopped")
                    .unwrap_or(false)
            };

            if stopped_gracefully {
                if let Some(info) = self.get_info(id).await {
                    return Ok(info);
                }
                anyhow::bail!("Failed to load endpoint after graceful stop: {}", id);
            }

            // Force kill if still alive after 5 s.
            let mut inner = self.inner.write().await;
            if let Some(rt) = inner.runtimes.get_mut(id) {
                if let Some(ref mut child) = rt.child {
                    let _ = child.kill().await;
                }
                rt.child = None;
                rt.status = "stopped".to_string();
                rt.stopped_at = Some(Utc::now());
                rt.pid = None;
                tracing::info!("Endpoint {} force-stopped", id);
            }
        } else {
            // No PID — just mark stopped.
            let mut inner = self.inner.write().await;
            if let Some(rt) = inner.runtimes.get_mut(id) {
                rt.status = "stopped".to_string();
                rt.stopped_at = Some(Utc::now());
                rt.child = None;
            }
        }

        self.get_info(id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Failed to load endpoint after stop: {}", id))
    }

    /// Stop all running endpoints.  Called on server shutdown.
    ///
    /// Strategy: batch SIGTERM, then poll every 100 ms for up to 5 s for
    /// each child to exit gracefully, then force-kill any survivors.
    pub async fn stop_all(&self) {
        let ids: Vec<String> = {
            let inner = self.inner.read().await;
            inner
                .runtimes
                .iter()
                .filter(|(_, rt)| rt.status == "running" || rt.status == "starting")
                .map(|(id, _)| id.clone())
                .collect()
        };

        if ids.is_empty() {
            return;
        }

        tracing::info!("Gracefully stopping {} managed endpoints…", ids.len());

        // Batch SIGTERM on Unix.
        #[cfg(unix)]
        {
            let pids: Vec<String> = {
                let inner = self.inner.read().await;
                ids.iter()
                    .filter_map(|id| inner.runtimes.get(id)?.pid)
                    .map(|p| p.to_string())
                    .collect()
            };
            if !pids.is_empty()
                && let Some(kill) = SystemBinary::Kill.path()
            {
                let mut args = vec!["-TERM".to_string()];
                args.extend(pids);
                let _ = tokio::process::Command::new(kill)
                    .args(&args)
                    .status()
                    .await;
            }
        }

        // Poll each child every 100 ms for up to 5 s.
        'poll: for _ in 0..50 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            let mut inner = self.inner.write().await;
            let mut all_done = true;
            for id in &ids {
                if let Some(rt) = inner.runtimes.get_mut(id) {
                    if rt.status == "stopped" {
                        continue;
                    }
                    if let Some(ref mut child) = rt.child {
                        if let Ok(Some(_)) = child.try_wait() {
                            rt.status = "stopped".to_string();
                            rt.stopped_at = Some(Utc::now());
                            rt.pid = None;
                            tracing::info!("Endpoint {} stopped gracefully", id);
                        } else {
                            all_done = false;
                        }
                    }
                }
            }
            if all_done {
                tracing::info!("All managed endpoints stopped gracefully");
                break 'poll;
            }
        }

        // Force-kill any survivors.
        let mut inner = self.inner.write().await;
        for id in &ids {
            if let Some(rt) = inner.runtimes.get_mut(id) {
                if rt.status != "stopped" {
                    if let Some(ref mut child) = rt.child {
                        let _ = child.kill().await;
                    }
                    rt.status = "stopped".to_string();
                    rt.stopped_at = Some(Utc::now());
                    rt.pid = None;
                    tracing::warn!("Endpoint {} force-killed", id);
                }
            }
        }
        tracing::info!("All managed endpoints stopped");
    }

    /// Delete an endpoint config (stops it first if running).
    pub async fn delete(self: &Arc<Self>, id: &str) -> Result<()> {
        let is_active = {
            let inner = self.inner.read().await;
            inner
                .runtimes
                .get(id)
                .map(|rt| rt.status == "running" || rt.status == "starting")
                .unwrap_or(false)
        };
        if is_active {
            self.stop(id).await?;
        }
        {
            let mut inner = self.inner.write().await;
            if inner.configs.remove(id).is_none() {
                anyhow::bail!("Endpoint not found: {}", id);
            }
            inner.runtimes.remove(id);
        }
        self.persist().await;
        Ok(())
    }

    /// Return all endpoint infos sorted by creation time.
    pub async fn list(&self) -> Vec<EndpointInfo> {
        let inner = self.inner.read().await;
        let mut result: Vec<EndpointInfo> = inner
            .configs
            .keys()
            .filter_map(|id| Self::build_info_ref(id, &inner))
            .collect();
        result.sort_by_key(|e| e.created_at);
        result
    }

    /// Return a single endpoint info.
    pub async fn get(&self, id: &str) -> Option<EndpointInfo> {
        self.get_info(id).await
    }

    /// Scan running/starting child processes for unexpected exits and update status.
    /// Also promotes "starting" → "running" once the child's health endpoint responds.
    pub async fn check_health(&self) {
        // Collect IDs and ports of processes we need to check.
        let candidates: Vec<(String, String, u16)> = {
            let inner = self.inner.read().await;
            inner
                .runtimes
                .iter()
                .filter(|(_, rt)| rt.status == "running" || rt.status == "starting")
                .filter_map(|(id, rt)| {
                    let port = inner.configs.get(id)?.port;
                    Some((id.clone(), rt.status.clone(), port))
                })
                .collect()
        };

        let mut inner = self.inner.write().await;
        for (id, status, port) in candidates {
            if let Some(rt) = inner.runtimes.get_mut(&id) {
                if let Some(ref mut child) = rt.child {
                    match child.try_wait() {
                        Ok(Some(exit_status)) => {
                            // Process has exited.
                            rt.status = if exit_status.success() {
                                "stopped"
                            } else {
                                "failed"
                            }
                            .to_string();
                            rt.stopped_at = Some(Utc::now());
                            rt.pid = None;
                            if !exit_status.success() {
                                rt.error =
                                    Some(format!("Process exited unexpectedly: {}", exit_status));
                            }
                            tracing::info!("Endpoint {} exited: {}", id, exit_status);
                        }
                        Ok(None) => {
                            // Process is still running.  If it was "starting",
                            // try a lightweight TCP probe to upgrade status.
                            if status == "starting" {
                                // Non-blocking TCP connect probe on the configured port.
                                let addr = format!("127.0.0.1:{}", port);
                                if tokio::net::TcpStream::connect(&addr).await.is_ok() {
                                    rt.status = "running".to_string();
                                    tracing::info!(
                                        "Endpoint {} is now running on port {}",
                                        id,
                                        port
                                    );
                                }
                                // Otherwise leave as "starting" and retry next cycle.
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Failed to check child status for {}: {}", id, e);
                        }
                    }
                }
            }
        }
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    async fn get_info(&self, id: &str) -> Option<EndpointInfo> {
        let inner = self.inner.read().await;
        Self::build_info_ref(id, &inner)
    }

    fn build_info_ref(id: &str, inner: &ManagerInner) -> Option<EndpointInfo> {
        let config = inner.configs.get(id)?;
        let rt = inner.runtimes.get(id)?;
        Some(EndpointInfo {
            id: id.to_string(),
            name: config.name.clone(),
            workspace: config.workspace.clone(),
            port: config.port,
            provider: config.provider.clone(),
            model: config.model.clone(),
            auto_start: config.auto_start,
            status: rt.status.clone(),
            pid: rt.pid,
            url: format!("http://localhost:{}", config.port),
            created_at: config.created_at,
            started_at: rt.started_at,
            stopped_at: rt.stopped_at,
            error: rt.error.clone(),
        })
    }
}

// ── HTTP handlers ─────────────────────────────────────────────────────────────

pub async fn list_endpoints(State(state): State<AppState>) -> ApiResult<EndpointListResponse> {
    // Refresh health before listing so stale statuses are corrected inline.
    state.endpoint_manager.check_health().await;
    let endpoints = state.endpoint_manager.list().await;
    Ok(Json(EndpointListResponse { endpoints }))
}

pub async fn create_endpoint(
    State(state): State<AppState>,
    Json(mut req): Json<CreateEndpointRequest>,
) -> ApiResult<EndpointInfo> {
    req.workspace = expand_workspace_path(&req.workspace);

    // Validate workspace directory exists
    if !std::path::Path::new(&req.workspace).is_dir() {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Workspace directory does not exist: {}", req.workspace),
        ));
    }

    let auto_start = req.auto_start;
    let info = state
        .endpoint_manager
        .create(req)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    if auto_start {
        let mgr = Arc::clone(&state.endpoint_manager);
        let id = info.id.clone();
        tokio::spawn(async move {
            if let Err(e) = mgr.start(&id).await {
                tracing::warn!("Failed to auto-start endpoint {}: {}", id, e);
            }
        });
    }

    Ok(Json(info))
}

pub async fn get_endpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<EndpointInfo> {
    state.endpoint_manager.check_health().await;
    state
        .endpoint_manager
        .get(&id)
        .await
        .map(Json)
        .ok_or_else(|| (StatusCode::NOT_FOUND, format!("Endpoint not found: {}", id)))
}

pub async fn start_endpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<EndpointInfo> {
    state
        .endpoint_manager
        .start(&id)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

pub async fn stop_endpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> ApiResult<EndpointInfo> {
    state
        .endpoint_manager
        .stop(&id)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

pub async fn delete_endpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    Arc::clone(&state.endpoint_manager)
        .delete(&id)
        .await
        .map(|_| Json(serde_json::json!({ "deleted": true, "id": id })))
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::VarError;
    use std::ffi::OsStr;

    #[test]
    fn endpoint_child_gets_a_stored_key_and_the_parent_env_stays_unset() {
        let name = "CALICHE_ENDPOINT_CHILD_TEST_API_KEY";
        caliche_contracts::credential_env::set(name, "stored-endpoint-key");
        let config = EndpointConfig {
            id: "ep-1".to_string(),
            name: "test".to_string(),
            workspace: ".".to_string(),
            port: 43_211,
            provider: None,
            model: None,
            auto_start: false,
            created_at: Utc::now(),
        };

        let cmd = endpoint_command("caliche_server", &config);

        let value = cmd
            .as_std()
            .get_envs()
            .find(|(key, _)| *key == OsStr::new(name))
            .and_then(|(_, value)| value)
            .map(|value| value.to_string_lossy().into_owned());
        assert_eq!(value.as_deref(), Some("stored-endpoint-key"));
        assert_eq!(std::env::var(name), Err(VarError::NotPresent));
    }
}
