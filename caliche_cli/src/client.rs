//! Transport-agnostic agent client and local server lock discovery.

use anyhow::{Context, Result, anyhow};
use caliche::messages::stream_contracts::sse_name;
use caliche_contracts::sessions::{
    DiffResponseDto, RedoResponseDto, ShareCreateRequestDto, ShareCreateResponseDto,
    UndoResponseDto,
};
use caliche_server::security::join_listen_host_port;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Idle limit when a thin-client turn receives no session events.
///
/// Keep-alives do not count. Override with [`TURN_IDLE_TIMEOUT_ENV`].
pub const DEFAULT_TURN_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

/// Env override for [`DEFAULT_TURN_IDLE_TIMEOUT`], in seconds.
pub const TURN_IDLE_TIMEOUT_ENV: &str = "CALICHE_TURN_IDLE_SECS";

const TURN_EVENT_DRAIN: Duration = Duration::from_millis(500);
const TURN_FAILED_PREFIX: &str = "Turn failed:";
const CANCELLED_BY_USER: &str = "Operation cancelled by user";

/// Contents written to `~/.config/caliche/server.lock`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerLock {
    /// Listening host (usually `127.0.0.1`).
    pub host: String,
    /// Listening port.
    pub port: u16,
    /// Base HTTP URL (`http://host:port`). IPv6 hosts are bracketed.
    pub url: String,
    /// Server process id.
    pub pid: u32,
    /// Unix timestamp (seconds) when the lock was written.
    pub started_at: u64,
}

impl ServerLock {
    /// Build a lock record for a freshly bound listener.
    ///
    /// The URL reuses `join_listen_host_port` so a bare IPv6 host such as
    /// `::` is stored as `http://[::]:port`.
    pub fn new(host: impl Into<String>, port: u16) -> Self {
        let host = host.into();
        let started_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let joined = join_listen_host_port(&host, port);
        Self {
            url: format!("http://{joined}"),
            host,
            port,
            pid: process::id(),
            started_at,
        }
    }

    /// Canonical lock file path (`$XDG_CONFIG_HOME/caliche/server.lock` or `~/.config/...`).
    ///
    /// `CALICHE_CONFIG_DIR`, when set, is that `caliche` directory and
    /// the lock is `server.lock` inside it.
    pub fn default_path() -> PathBuf {
        caliche_server::paths::server_lock_path()
    }

    /// Persist this lock atomically (best-effort).
    pub fn write_to(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create lock directory {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self).context("serialize server lock")?;
        let tmp = path.with_extension("lock.tmp");
        fs::write(&tmp, json).with_context(|| format!("write {}", tmp.display()))?;
        fs::rename(&tmp, path).with_context(|| format!("rename {}", path.display()))?;
        Ok(())
    }

    /// Read a lock file if present.
    pub fn read_from(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("read server lock {}", path.display()))?;
        let lock: Self = serde_json::from_str(&raw)
            .with_context(|| format!("parse server lock {}", path.display()))?;
        Ok(Some(lock))
    }

    /// Remove the lock file when it belongs to this process (or force).
    pub fn remove_from(path: &Path, force: bool) -> Result<()> {
        match Self::read_from(path)? {
            None => Ok(()),
            Some(lock) if force || lock.pid == process::id() => {
                let _ = fs::remove_file(path);
                Ok(())
            }
            Some(lock) => Err(anyhow!(
                "refusing to remove server lock owned by pid {}",
                lock.pid
            )),
        }
    }

    /// True when the recorded pid still appears to be alive.
    pub fn process_appears_alive(&self) -> bool {
        process_alive(self.pid)
    }
}

/// Canonical server lock path.
pub fn lock_path() -> PathBuf {
    ServerLock::default_path()
}

/// Build a lock record for a bound listener.
pub fn lock_for_bind(host: impl Into<String>, port: u16) -> ServerLock {
    ServerLock::new(host, port)
}

/// Persist a server lock.
pub fn write_lock(path: &Path, lock: &ServerLock) -> Result<()> {
    lock.write_to(path)
}

/// Read a server lock, if present.
pub fn read_lock(path: &Path) -> Result<Option<ServerLock>> {
    ServerLock::read_from(path)
}

/// Remove a server lock owned by this process.
pub fn remove_lock(path: &Path) -> Result<()> {
    ServerLock::remove_from(path, false)
}

/// True when the process exists and can be signalled with `kill(pid, 0)`.
pub fn is_alive(pid: u32) -> bool {
    process_alive(pid)
}

#[cfg(unix)]
fn process_alive(pid: u32) -> bool {
    // signal 0 checks existence / permission without sending a real signal
    crate::process_liveness::signal_pid(pid, 0).is_ok()
}

#[cfg(target_os = "windows")]
fn process_alive(pid: u32) -> bool {
    // Dead clears a stale lock. Unreadable keeps it.
    crate::process_liveness::windows_pid_running(pid)
}

#[cfg(all(not(unix), not(target_os = "windows")))]
fn process_alive(pid: u32) -> bool {
    let _ = pid;
    true
}

/// How the CLI talks to an agent engine.
#[derive(Debug, Clone)]
pub enum AgentClientMode {
    /// Talk to an already-running HTTP server discovered via the lock file.
    Remote { base_url: String },
    /// No live server discovered; caller should start one in-process.
    LocalBootstrap,
}

/// Thin transport-agnostic client used by REPL / IDE integrations.
#[derive(Debug, Clone)]
pub struct AgentClient {
    mode: AgentClientMode,
    lock_path: PathBuf,
    http: reqwest::Client,
    turn_idle_timeout: Duration,
}

/// JSON body for `POST /api/agent/sessions/{id}/turn`.
///
/// `selection` is the REPL label when it differs from the identity the remote
/// session already has. Omitted fields leave that session on its current model,
/// which is the startup model until a switch is applied.
fn thin_client_turn_body(
    input: &str,
    selection: Option<(&str, &str)>,
    context_window: Option<usize>,
) -> serde_json::Value {
    let mut body = serde_json::json!({ "input": input });
    if let Some((provider, model)) = selection {
        body["provider"] = provider.into();
        body["model"] = model.into();
    }
    if let Some(window) = context_window {
        body["context_window"] = window.into();
    }
    body
}

impl AgentClient {
    /// Create a remote client for a known base URL.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::assemble(
            AgentClientMode::Remote {
                base_url: base_url.into(),
            },
            ServerLock::default_path(),
        )
    }

    /// How long a turn may go without a session event before it fails.
    #[must_use]
    pub fn with_turn_idle_timeout(mut self, timeout: Duration) -> Self {
        self.turn_idle_timeout = timeout;
        self
    }

    /// Create a remote client from a live server lock.
    pub fn from_lock(lock: ServerLock) -> Result<Self> {
        if !lock.process_appears_alive() {
            return Err(anyhow!("server lock pid {} is not alive", lock.pid));
        }
        Ok(Self::new(lock.url))
    }

    /// Discover a running server via the default lock path.
    pub fn discover() -> Result<Self> {
        Self::discover_at(ServerLock::default_path())
    }

    /// Discover a running server via an explicit lock path.
    pub fn discover_at(lock_path: PathBuf) -> Result<Self> {
        if let Some(lock) = ServerLock::read_from(&lock_path)? {
            if lock.process_appears_alive() {
                return Ok(Self::assemble(
                    AgentClientMode::Remote { base_url: lock.url },
                    lock_path,
                ));
            }
            if let Err(err) = clear_stale_server_lock(&lock_path) {
                tracing::warn!(
                    error = %err,
                    path = %lock_path.display(),
                    "could not remove stale server.lock"
                );
            }
            // Registry records for the dead lock pid go with the stale lock.
            let registry_dir = match lock_path.parent() {
                Some(parent) => parent.join("servers"),
                None => crate::server_registry::default_registry_dir(),
            };
            if let Err(err) =
                crate::server_registry::remove_records_for_pid_if_dead(&registry_dir, lock.pid)
            {
                tracing::warn!(
                    error = %err,
                    pid = lock.pid,
                    "could not remove stale server records for dead lock pid"
                );
            }
        }
        Ok(Self::assemble(AgentClientMode::LocalBootstrap, lock_path))
    }

    fn assemble(mode: AgentClientMode, lock_path: PathBuf) -> Self {
        Self {
            mode,
            lock_path,
            http: reqwest::Client::new(),
            turn_idle_timeout: turn_idle_timeout_from_env(),
        }
    }

    /// Current connection mode.
    pub fn mode(&self) -> &AgentClientMode {
        &self.mode
    }

    /// Lock path used for discovery / registration.
    pub fn lock_path(&self) -> &Path {
        &self.lock_path
    }

    /// Base URL when connected remotely.
    pub fn base_url(&self) -> Option<&str> {
        match &self.mode {
            AgentClientMode::Remote { base_url } => Some(base_url.as_str()),
            AgentClientMode::LocalBootstrap => None,
        }
    }

    /// Record that a local server is listening (updates mode to Remote).
    pub fn register_local_server(&mut self, host: &str, port: u16) -> Result<()> {
        let lock = ServerLock::new(host, port);
        lock.write_to(&self.lock_path)?;
        self.mode = AgentClientMode::Remote { base_url: lock.url };
        Ok(())
    }

    /// Clear the lock file if we own it.
    pub fn clear_lock(&self) -> Result<()> {
        ServerLock::remove_from(&self.lock_path, false)
    }

    /// Check the remote server health endpoint.
    pub async fn health(&self) -> Result<bool> {
        let Some(base_url) = self.base_url() else {
            return Ok(false);
        };
        let response = self
            .http
            .get(format!("{}/health", base_url.trim_end_matches('/')))
            .send()
            .await
            .context("send health request")?;
        Ok(response.status().is_success())
    }

    /// Create a remote API session and return its id.
    pub async fn create_session(&self) -> Result<String> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        let response: serde_json::Value = self
            .http
            .post(format!(
                "{}/api/agent/sessions",
                base_url.trim_end_matches('/')
            ))
            .json(&serde_json::json!({}))
            .send()
            .await
            .context("send create session request")?
            .error_for_status()
            .context("create session failed")?
            .json()
            .await
            .context("decode create session response")?;
        response
            .get("session_id")
            .and_then(|value| value.as_str())
            .map(str::to_string)
            .ok_or_else(|| anyhow!("create session response did not include session_id"))
    }

    /// Reload a saved transcript into the remote session the next turn reads.
    pub async fn restore_session_history(
        &self,
        session_id: &str,
        messages: &[caliche::messages::Message],
    ) -> Result<()> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        self.http
            .post(format!(
                "{}/api/agent/sessions/{session_id}/history",
                base_url.trim_end_matches('/')
            ))
            .json(&serde_json::json!({ "messages": messages }))
            .send()
            .await
            .context("send restore history request")?
            .error_for_status()
            .context("restore session history failed")?;
        Ok(())
    }

    /// Execute one remote turn and return the raw response body.
    ///
    /// `selection` is posted when the REPL label differs from the remote
    /// session. The server applies that pair before the turn. `None` omits
    /// both fields so the session keeps its current model.
    pub async fn execute_turn(
        &self,
        session_id: &str,
        input: &str,
        selection: Option<(&str, &str)>,
        context_window: Option<usize>,
    ) -> Result<serde_json::Value> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        let response = self
            .http
            .post(format!(
                "{}/api/agent/sessions/{}/turn",
                base_url.trim_end_matches('/'),
                session_id
            ))
            .json(&thin_client_turn_body(input, selection, context_window))
            .send()
            .await
            .context("send execute turn request")?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow!("{}", format_turn_failure(status, &body)));
        }
        response
            .json()
            .await
            .context("decode execute turn response")
    }

    /// Ask the server to cancel the in-flight turn for `session_id`.
    ///
    /// `POST /api/agent/sessions/{id}/cancel` stops the turn and removes the
    /// session. A missing session is success so a second cancel is quiet.
    pub async fn cancel_session(&self, session_id: &str) -> Result<()> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        let response = self
            .http
            .post(format!(
                "{}/api/agent/sessions/{session_id}/cancel",
                base_url.trim_end_matches('/')
            ))
            .send()
            .await
            .context("send cancel session request")?;
        if response.status().is_success() || response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(());
        }
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        Err(anyhow!("cancel session failed: {status} {}", body.trim()))
    }

    /// Fetch a typed session diff overview from the remote API.
    pub async fn session_diff(&self, session_id: &str) -> Result<DiffResponseDto> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        self.http
            .get(format!(
                "{}/api/agent/sessions/{}/diff",
                base_url.trim_end_matches('/'),
                session_id
            ))
            .send()
            .await
            .context("send session diff request")?
            .error_for_status()
            .context("session diff failed")?
            .json()
            .await
            .context("decode session diff response")
    }

    /// Undo the last remote session change set.
    pub async fn undo_last(&self, session_id: &str) -> Result<UndoResponseDto> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        self.http
            .post(format!(
                "{}/api/agent/sessions/{}/undo",
                base_url.trim_end_matches('/'),
                session_id
            ))
            .send()
            .await
            .context("send undo request")?
            .error_for_status()
            .context("undo failed")?
            .json()
            .await
            .context("decode undo response")
    }

    /// Redo the last reverted remote session change set.
    pub async fn redo_last(&self, session_id: &str) -> Result<RedoResponseDto> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        self.http
            .post(format!(
                "{}/api/agent/sessions/{}/unrevert",
                base_url.trim_end_matches('/'),
                session_id
            ))
            .send()
            .await
            .context("send redo request")?
            .error_for_status()
            .context("redo failed")?
            .json()
            .await
            .context("decode redo response")
    }

    /// Create a redacted share link for a remote session.
    pub async fn share_session(&self, session_id: &str) -> Result<ShareCreateResponseDto> {
        self.share_session_with_options(session_id, ShareCreateRequestDto::default())
            .await
    }

    /// Create a redacted share link with optional visibility.
    pub async fn share_session_with_options(
        &self,
        session_id: &str,
        request: ShareCreateRequestDto,
    ) -> Result<ShareCreateResponseDto> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        self.http
            .post(format!(
                "{}/api/agent/sessions/{}/share",
                base_url.trim_end_matches('/'),
                session_id
            ))
            .json(&request)
            .send()
            .await
            .context("send share request")?
            .error_for_status()
            .context("share failed")?
            .json()
            .await
            .context("decode share response")
    }

    /// Remove all share links for a remote session.
    pub async fn unshare_session(&self, session_id: &str) -> Result<()> {
        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?;
        self.http
            .delete(format!(
                "{}/api/agent/sessions/{}/share",
                base_url.trim_end_matches('/'),
                session_id
            ))
            .send()
            .await
            .context("send unshare request")?
            .error_for_status()
            .context("unshare failed")?;
        Ok(())
    }

    /// Run one turn and invoke `on_event` as each session SSE frame arrives.
    ///
    /// Subscribes first, then POSTs `/turn` so events are not missed. The
    /// callback runs on the caller task, so a REPL can render immediately.
    /// `selection` is included on the turn body when the label changed.
    ///
    /// The POST and the event stream are polled together. A failed POST or a
    /// `turn.error` frame returns immediately. Silence aside from keep-alives
    /// fails after [`Self::with_turn_idle_timeout`].
    pub async fn run_turn_stream<F>(
        &self,
        session_id: &str,
        input: &str,
        selection: Option<(&str, &str)>,
        on_event: F,
    ) -> Result<serde_json::Value>
    where
        F: FnMut(&str, &serde_json::Value),
    {
        self.run_turn_stream_with_cancel(session_id, input, selection, None, None, on_event)
            .await
    }

    /// [`Self::run_turn_stream`] that cancels the remote turn when `cancel` fires.
    pub async fn run_turn_stream_with_cancel<F>(
        &self,
        session_id: &str,
        input: &str,
        selection: Option<(&str, &str)>,
        context_window: Option<usize>,
        cancel: Option<&crate::cancellation::CancellationToken>,
        mut on_event: F,
    ) -> Result<serde_json::Value>
    where
        F: FnMut(&str, &serde_json::Value),
    {
        use futures_util::StreamExt;

        let mut events = self.subscribe_session_events(session_id).await?;
        let client = self.clone();
        let owned_session = session_id.to_string();
        let input = input.to_string();
        let selection =
            selection.map(|(provider, model)| (provider.to_string(), model.to_string()));
        let mut turn = tokio::spawn(async move {
            client
                .execute_turn(
                    &owned_session,
                    &input,
                    selection
                        .as_ref()
                        .map(|(provider, model)| (provider.as_str(), model.as_str())),
                    context_window,
                )
                .await
        });

        let idle = self.turn_idle_timeout;
        let sleep = tokio::time::sleep(idle);
        tokio::pin!(sleep);
        let mut saw_complete = false;
        let mut turn_body: Option<serde_json::Value> = None;
        let mut drain_deadline: Option<tokio::time::Instant> = None;

        loop {
            if saw_complete && turn_body.is_some() {
                break;
            }
            tokio::select! {
                biased;
                _ = cancel_notified(cancel) => {
                    turn.abort();
                    self.cancel_session(session_id).await?;
                    return Err(anyhow!(CANCELLED_BY_USER));
                }
                joined = &mut turn, if turn_body.is_none() => {
                    match joined {
                        Ok(Ok(body)) => {
                            turn_body = Some(body);
                            if saw_complete {
                                break;
                            }
                            drain_deadline = Some(tokio::time::Instant::now() + TURN_EVENT_DRAIN);
                            sleep.as_mut().reset(tokio::time::Instant::now() + TURN_EVENT_DRAIN);
                        }
                        Ok(Err(error)) => return Err(error),
                        Err(error) => return Err(anyhow!("join remote turn task: {error}")),
                    }
                }
                item = events.next() => {
                    match item {
                        Some(Ok((name, data))) => {
                            if let Some(message) = terminal_turn_error(&name, &data) {
                                turn.abort();
                                return Err(anyhow!("{TURN_FAILED_PREFIX} {message}"));
                            }
                            let is_complete = name == sse_name::TURN_COMPLETE;
                            on_event(&name, &data);
                            if is_complete {
                                saw_complete = true;
                            }
                            if drain_deadline.is_none() {
                                sleep.as_mut().reset(tokio::time::Instant::now() + idle);
                            }
                        }
                        Some(Err(error)) => return Err(error),
                        None => break,
                    }
                }
                _ = &mut sleep => {
                    if turn_body.is_some() && !saw_complete {
                        break;
                    }
                    let seconds = idle.as_secs();
                    turn.abort();
                    return Err(anyhow!(
                        "{TURN_FAILED_PREFIX} no session events for {seconds}s"
                    ));
                }
            }
        }

        finish_remote_turn(turn, turn_body, saw_complete, on_event).await
    }

    /// Run one turn while collecting session SSE frames until `turn.complete`.
    ///
    /// Prefer [`Self::run_turn_stream`] when the caller can render incrementally.
    pub async fn run_turn_with_events(
        &self,
        session_id: &str,
        input: &str,
        selection: Option<(&str, &str)>,
    ) -> Result<Vec<(String, serde_json::Value)>> {
        let mut collected = Vec::new();
        self.run_turn_stream(session_id, input, selection, |name, data| {
            collected.push((name.to_string(), data.clone()));
        })
        .await?;
        Ok(collected)
    }

    /// Subscribe to session SSE events (`GET /api/agent/sessions/{id}/events`).
    ///
    /// Yields `(event_name, data_json)` pairs as frames arrive. The stream ends
    /// when the server closes the connection or an I/O error occurs.
    pub async fn subscribe_session_events(
        &self,
        session_id: &str,
    ) -> Result<impl futures_util::Stream<Item = Result<(String, serde_json::Value)>> + use<>> {
        use futures_util::StreamExt;

        let base_url = self
            .base_url()
            .ok_or_else(|| anyhow!("remote server is not configured"))?
            .to_string();
        let url = format!(
            "{}/api/agent/sessions/{}/events",
            base_url.trim_end_matches('/'),
            session_id
        );
        let response = self
            .http
            .get(&url)
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .send()
            .await
            .context("send session events request")?
            .error_for_status()
            .context("session events request failed")?;

        let mut byte_stream = response.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<(String, serde_json::Value)>>(32);
        tokio::spawn(async move {
            let mut buffer = String::new();
            while let Some(chunk) = byte_stream.next().await {
                let chunk = match chunk {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        let _ = tx
                            .send(Err(anyhow!("session SSE read failed: {err}")))
                            .await;
                        break;
                    }
                };
                append_utf8_lossy(&mut buffer, &chunk);
                for frame in drain_complete_sse_frames(&mut buffer) {
                    if let Some(parsed) = parse_sse_frame(&frame)
                        && tx.send(Ok(parsed)).await.is_err()
                    {
                        return;
                    }
                }
            }
        });
        Ok(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}

fn turn_idle_timeout_from_env() -> Duration {
    let Ok(raw) = std::env::var(TURN_IDLE_TIMEOUT_ENV) else {
        return DEFAULT_TURN_IDLE_TIMEOUT;
    };
    raw.trim()
        .parse::<u64>()
        .ok()
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs)
        .unwrap_or(DEFAULT_TURN_IDLE_TIMEOUT)
}

async fn cancel_notified(cancel: Option<&crate::cancellation::CancellationToken>) {
    match cancel {
        Some(token) => token.cancelled().await,
        None => std::future::pending().await,
    }
}

fn terminal_turn_error(name: &str, data: &serde_json::Value) -> Option<String> {
    if name != sse_name::TURN_ERROR {
        return None;
    }
    let body = sse_event_payload(data);
    let message = body
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("turn failed");
    let trimmed = message.trim().trim_start_matches(TURN_FAILED_PREFIX).trim();
    if trimmed.is_empty() {
        Some("turn failed".to_string())
    } else {
        Some(trimmed.to_string())
    }
}

fn format_turn_failure(status: reqwest::StatusCode, body: &str) -> String {
    let trimmed = body.trim();
    if let Some(line) = provider_failure_line(trimmed) {
        return format!("{TURN_FAILED_PREFIX} {line}");
    }
    if trimmed.is_empty() {
        return format!("{TURN_FAILED_PREFIX} HTTP {}", status.as_u16());
    }
    format!("{TURN_FAILED_PREFIX} {trimmed}")
}

fn provider_failure_line(body: &str) -> Option<String> {
    const MARKER: &str = " API error (";
    let start = body.rfind(MARKER)?;
    let provider = provider_before_api_error(&body[..start])?;
    let after = &body[start + MARKER.len()..];
    let code_len = after
        .bytes()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    if code_len == 0 {
        return None;
    }
    let code = &after[..code_len];
    let close = after.find("):")?;
    let message = after[close + 2..].trim();
    if message.is_empty() {
        return None;
    }
    Some(format!(
        "{} {code}: {message}",
        provider.to_ascii_lowercase()
    ))
}

fn provider_before_api_error(prefix: &str) -> Option<&str> {
    let provider = prefix
        .rsplit(|ch: char| ch.is_whitespace() || ch == ':')
        .next()
        .unwrap_or("")
        .trim();
    if provider.is_empty() {
        None
    } else {
        Some(provider)
    }
}

fn fallback_turn_complete(body: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "turn_complete",
        "response": body.get("response").cloned().unwrap_or(serde_json::Value::Null),
        "usage": body.get("usage").cloned().unwrap_or(serde_json::Value::Null),
        "duration_ms": body
            .get("duration_ms")
            .cloned()
            .unwrap_or_else(|| serde_json::json!(0)),
    })
}

/// Await the turn POST if the stream ended first, and synthesize
/// `turn.complete` when the server never sent one.
async fn finish_remote_turn<F>(
    turn: tokio::task::JoinHandle<Result<serde_json::Value>>,
    turn_body: Option<serde_json::Value>,
    saw_complete: bool,
    mut on_event: F,
) -> Result<serde_json::Value>
where
    F: FnMut(&str, &serde_json::Value),
{
    let turn_body = match turn_body {
        Some(body) => body,
        None => turn
            .await
            .context("join remote turn task")?
            .context("remote turn failed")?,
    };
    if !saw_complete {
        on_event(sse_name::TURN_COMPLETE, &fallback_turn_complete(&turn_body));
    }
    Ok(turn_body)
}

/// Drop a server lock whose recorded pid is not running.
///
/// A live pid is left in place. The info log is the operator signal that a
/// previous process exited without deleting the lock.
pub fn clear_stale_server_lock(path: &Path) -> Result<bool> {
    let Some(lock) = ServerLock::read_from(path)? else {
        return Ok(false);
    };
    if lock.process_appears_alive() {
        return Ok(false);
    }
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
        Err(err) => {
            return Err(err)
                .with_context(|| format!("remove stale server lock {}", path.display()));
        }
    }
    tracing::info!(
        path = %path.display(),
        pid = lock.pid,
        "{STALE_SERVER_LOCK_MESSAGE}"
    );
    Ok(true)
}

/// Log line when startup deletes a lock whose pid is gone.
pub const STALE_SERVER_LOCK_MESSAGE: &str =
    "removed stale server.lock; recorded pid is not running";

fn append_utf8_lossy(buffer: &mut String, chunk: &[u8]) {
    match std::str::from_utf8(chunk) {
        Ok(text) => buffer.push_str(text),
        Err(_) => buffer.push_str(&String::from_utf8_lossy(chunk)),
    }
}

/// Split complete SSE frames out of `buffer`, leaving any trailing partial frame.
fn drain_complete_sse_frames(buffer: &mut String) -> Vec<String> {
    let mut frames = Vec::new();
    while let Some(idx) = buffer.find("\n\n") {
        let frame = buffer[..idx].to_string();
        buffer.drain(..idx + 2);
        frames.push(frame);
    }
    frames
}

/// Return the inner object for versioned SSE envelopes (`{version, type, payload}`).
///
/// Session event streams wrap harness events as `{version, type, payload}`.
/// Flat bodies from tests and the POST `/turn` fallback are returned unchanged.
#[must_use]
pub fn sse_event_payload(data: &serde_json::Value) -> &serde_json::Value {
    match data.get("payload") {
        Some(payload) if payload.is_object() => payload,
        _ => data,
    }
}

/// Parse a single SSE frame (`event:` / `data:` lines) into `(event_name, json)`.
///
/// Incomplete or non-JSON frames return `None`. Default event name is `message`.
#[must_use]
pub fn parse_sse_frame(frame: &str) -> Option<(String, serde_json::Value)> {
    let mut event_name = String::from("message");
    let mut data_lines: Vec<&str> = Vec::new();
    for line in frame.lines() {
        if line.starts_with(':') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("event:") {
            event_name = rest.trim().to_string();
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.trim_start());
        }
    }
    if data_lines.is_empty() {
        return None;
    }
    let data = data_lines.join("\n");
    let value = serde_json::from_str(&data).ok()?;
    Some((event_name, value))
}

#[cfg(test)]
#[path = "client_tests.rs"]
mod tests;
