use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use agent_client_protocol_schema::ProtocolVersion;
use agent_client_protocol_schema::v1::{
    CancelNotification, ClientCapabilities, ContentBlock, ContentChunk, FileSystemCapabilities,
    Implementation, InitializeRequest, InitializeResponse, NewSessionRequest, NewSessionResponse,
    PromptRequest, PromptResponse, SessionId, SessionUpdate, StopReason, TextContent,
};
use caliche_contracts::acp::methods;
use serde_json::Value;
use tokio::process::{Child, Command};
use tokio::runtime::Handle;

use crate::auth::{map_auth_required, reject_peer_auth};
use crate::error::AcpClientError;
use crate::notes::NotesHub;
use crate::notes::notes_channel;
use crate::rpc::WaiterFail;
use crate::stdio::{
    Io, PHASE_CLOSED, PHASE_READY, PHASE_SPAWNED, StderrTail, drain_stderr, read_stdout,
    stderr_text,
};
use crate::transcript::Transcript;

const DEFAULT_HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// Stable ACP version this client speaks.
pub const ACP_PROTOCOL_VERSION: ProtocolVersion = ProtocolVersion::V1;

/// How to start a peer ACP agent.
///
/// By default the child gets only the host variables that
/// [`caliche_contracts::child_env::is_child_env_key_allowed`] passes (search
/// path, home, user, locale, terminal, temp dirs, and the Windows startup
/// variables), never one that
/// [`caliche_contracts::child_env::is_credential_name`] matches, plus the
/// pairs passed to [`AcpSpawn::env`]. `inherit_env(true)` opts in to
/// the full parent environment. Caliche does not prompt for those values.
#[derive(Debug, Clone)]
#[must_use]
pub struct AcpSpawn {
    /// `argv[0]` is the program. Later entries are arguments.
    pub argv: Vec<std::ffi::OsString>,
    /// Working directory for the child. Relative paths are made absolute.
    pub cwd: PathBuf,
    /// Environment entries applied after the host variables.
    pub env: Vec<(std::ffi::OsString, std::ffi::OsString)>,
    /// When true, the child starts with the full parent environment instead
    /// of the scrubbed host variables.
    pub inherit_env: bool,
    /// JSONL file for client and peer ACP frames. `None` keeps the pipes quiet.
    pub transcript: Option<PathBuf>,
}

impl AcpSpawn {
    /// Build a spawn spec. The child gets the scrubbed host environment.
    pub fn new(
        argv: impl IntoIterator<Item = impl Into<std::ffi::OsString>>,
        cwd: impl Into<PathBuf>,
    ) -> Self {
        Self {
            argv: argv.into_iter().map(Into::into).collect(),
            cwd: cwd.into(),
            env: Vec::new(),
            inherit_env: false,
            transcript: None,
        }
    }

    /// Set one environment variable. Later calls for the same key win.
    pub fn env(
        mut self,
        key: impl Into<std::ffi::OsString>,
        value: impl Into<std::ffi::OsString>,
    ) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Opt in to (`true`) or out of the full parent environment, provider
    /// keys included.
    pub const fn inherit_env(mut self, inherit: bool) -> Self {
        self.inherit_env = inherit;
        self
    }

    /// Write every ACP frame to `path` as one JSON object per line.
    ///
    /// Credential-shaped environment values are replaced with `[redacted]`.
    pub fn transcript(mut self, path: impl Into<PathBuf>) -> Self {
        self.transcript = Some(path.into());
        self
    }
}

/// Updates observed before a `session/prompt` result, plus that result's stop reason.
#[derive(Debug, Clone)]
#[must_use]
pub struct PromptOutcome {
    /// `stopReason` from the prompt response.
    pub stop_reason: StopReason,
    /// `session/update` values this crate decoded for the prompted session.
    pub updates: Vec<SessionUpdate>,
    /// Update params whose `sessionUpdate` variant is not in the shared contract yet.
    pub unrecognized_updates: Vec<Value>,
}

impl PromptOutcome {
    /// Concatenate `agent_message_chunk` text blocks in arrival order.
    #[must_use]
    pub fn agent_text(&self) -> String {
        let mut text = String::new();
        for update in &self.updates {
            append_agent_text(&mut text, update);
        }
        text
    }
}

/// A spawned ACP v1 peer.
///
/// The child stdout pipe is protocol-only. This type writes JSON-RPC frames to
/// the child stdin and reads JSON-RPC frames from the child stdout. It does not
/// write log lines to either pipe. Peer stderr is kept off the protocol pipe
/// and is available from [`AcpClient::stderr_tail`].
///
/// Drop kills the child. [`AcpClient`] is cheap to clone; clones share one process.
#[derive(Clone)]
#[must_use]
pub struct AcpClient {
    inner: Arc<ClientCore>,
}

struct CallTimeouts {
    handshake: Option<Duration>,
    prompt: Option<Duration>,
}

impl Default for CallTimeouts {
    fn default() -> Self {
        Self {
            handshake: Some(DEFAULT_HANDSHAKE_TIMEOUT),
            prompt: None,
        }
    }
}

struct ClientCore {
    io: Arc<Io>,
    notes: Arc<NotesHub>,
    timeouts: StdMutex<CallTimeouts>,
    child: tokio::sync::Mutex<Option<Child>>,
    reader_abort: tokio::task::AbortHandle,
    stderr_abort: tokio::task::AbortHandle,
    stderr_tail: Arc<StdMutex<StderrTail>>,
    init_gate: tokio::sync::Mutex<()>,
    cached_init: StdMutex<Option<InitializeResponse>>,
    _reader: tokio::task::JoinHandle<()>,
    _stderr_task: tokio::task::JoinHandle<()>,
}

impl Drop for ClientCore {
    fn drop(&mut self) {
        self.notes.shutdown();
        self.reader_abort.abort();
        self.stderr_abort.abort();
        self.io.fail_all(&WaiterFail::Closed);
        self.io.set_phase(PHASE_CLOSED);
        if let Some(child) = self.child.get_mut().as_mut() {
            let _ = child.start_kill();
        }
    }
}

impl AcpClient {
    /// Spawn `spec` and start reading its stdout.
    ///
    /// Call this on a Tokio runtime. The function returns after the process
    /// exists; call [`AcpClient::initialize`] before any session method.
    ///
    /// # Errors
    ///
    /// Returns [`AcpClientError::OutsideRuntime`] when no Tokio runtime is
    /// current, [`AcpClientError::EmptyArgv`] when the program is missing, and
    /// [`AcpClientError::Spawn`] when the OS cannot start the process.
    pub fn spawn(spec: AcpSpawn) -> Result<Self, AcpClientError> {
        let AcpSpawn {
            argv,
            cwd,
            env,
            inherit_env,
            transcript,
        } = spec;
        let runtime = Handle::try_current().map_err(|_| AcpClientError::OutsideRuntime)?;
        let transcript = open_transcript(transcript.as_deref())?;
        let mut child = child_from_spec(&argv, &cwd, &env, inherit_env)?;
        let pipes = take_pipes(&mut child)?;
        let (notes_tx, notes_rx) = notes_channel();
        let io = Arc::new(Io::new(pipes.stdin, notes_tx, transcript));
        let notes = NotesHub::new(notes_rx);
        let stderr_tail = Arc::new(StdMutex::new(StderrTail::new()));
        let reader = runtime.spawn(read_stdout(pipes.stdout, Arc::clone(&io)));
        let stderr_task = runtime.spawn(drain_stderr(pipes.stderr, Arc::clone(&stderr_tail)));
        let reader_abort = reader.abort_handle();
        let stderr_abort = stderr_task.abort_handle();
        Ok(Self {
            inner: Arc::new(ClientCore {
                io,
                notes,
                timeouts: StdMutex::new(CallTimeouts::default()),
                child: tokio::sync::Mutex::new(Some(child)),
                reader_abort,
                stderr_abort,
                stderr_tail,
                init_gate: tokio::sync::Mutex::new(()),
                cached_init: StdMutex::new(None),
                _reader: reader,
                _stderr_task: stderr_task,
            }),
        })
    }

    /// Negotiate ACP `protocolVersion` 1.
    ///
    /// The child is shut down when the peer answers with another version or
    /// with any auth method id other than `opencode-login` or `hermes-setup`.
    /// An empty list, or a list of only those two ids, continues. See
    /// [`crate::reject_peer_auth`]. This request does not set `auth.terminal`
    /// or `_meta["terminal-auth"]`.
    ///
    /// # Errors
    ///
    /// Returns [`AcpClientError::UnsupportedProtocol`], [`AcpClientError::AuthRejected`],
    /// or a transport error. A second successful call returns the cached result.
    pub async fn initialize(&self) -> Result<InitializeResponse, AcpClientError> {
        let _gate = self.inner.init_gate.lock().await;
        match self.inner.io.phase() {
            PHASE_CLOSED => return Err(AcpClientError::ShutDown),
            PHASE_READY => return cached_init(&self.inner.cached_init),
            _ => {}
        }
        let raw = self
            .inner
            .io
            .request(
                methods::INITIALIZE,
                initialize_value()?,
                self.handshake_timeout(),
            )
            .await?;
        let result = serde_json::from_value::<InitializeResponse>(raw).map_err(|source| {
            AcpClientError::Decode {
                what: "initialize result",
                source,
            }
        })?;
        if result.protocol_version != ACP_PROTOCOL_VERSION {
            let offered = u32::from(result.protocol_version.as_u16());
            let _ = self.shutdown().await;
            return Err(AcpClientError::UnsupportedProtocol {
                offered,
                supported: u32::from(ACP_PROTOCOL_VERSION.as_u16()),
            });
        }
        if let Err(err) = reject_peer_auth(&result.auth_methods) {
            let _ = self.shutdown().await;
            return Err(err);
        }
        self.inner.io.set_phase(PHASE_READY);
        *lock_cache(&self.inner.cached_init) = Some(result.clone());
        Ok(result)
    }

    /// `session/new` for an absolute `cwd`. Relative paths are made absolute first.
    ///
    /// # Errors
    ///
    /// Returns [`AcpClientError::NotInitialized`] before a successful initialize,
    /// [`AcpClientError::AuthRequired`] when the peer returns ACP `auth_required`,
    /// and the peer's JSON-RPC error when `session/new` fails for another reason.
    pub async fn new_session(
        &self,
        cwd: impl AsRef<Path>,
    ) -> Result<NewSessionResponse, AcpClientError> {
        self.ensure_ready()?;
        let cwd = path_to_string(&absolute_path(cwd.as_ref())?)?;
        let params = NewSessionRequest::new(cwd);
        let value = serde_json::to_value(params).map_err(|source| AcpClientError::Encode {
            what: "session/new",
            source,
        })?;
        let raw = self
            .peer_result(methods::SESSION_NEW, value, self.handshake_timeout())
            .await?;
        serde_json::from_value(raw).map_err(|source| AcpClientError::Decode {
            what: "session/new result",
            source,
        })
    }

    /// `session/prompt`, collecting `session/update` notifications until the result.
    ///
    /// Only one prompt may run at a time. [`AcpClient::cancel`] can run beside it.
    ///
    /// # Errors
    ///
    /// Returns [`AcpClientError::PromptInFlight`] when a prompt is already running,
    /// [`AcpClientError::EmptySessionId`] when `session_id` is empty, and
    /// [`AcpClientError::AuthRequired`] when the peer returns ACP `auth_required`.
    pub async fn prompt(
        &self,
        session_id: &str,
        prompt: Vec<ContentBlock>,
    ) -> Result<PromptOutcome, AcpClientError> {
        self.ensure_ready()?;
        reject_empty_session(session_id)?;
        let _gate = self.inner.notes.try_prompt_gate()?;
        let value = serde_json::to_value(PromptRequest::new(SessionId::new(session_id), prompt))
            .map_err(|source| AcpClientError::Encode {
                what: "session/prompt",
                source,
            })?;
        let mut lease = self.inner.notes.begin(session_id).await?;
        let raw = self
            .peer_result(methods::SESSION_PROMPT, value, self.prompt_timeout())
            .await?;
        let collected = lease.finish().await?;
        let parsed = serde_json::from_value::<PromptResponse>(raw).map_err(|source| {
            AcpClientError::Decode {
                what: "session/prompt result",
                source,
            }
        })?;
        Ok(PromptOutcome {
            stop_reason: parsed.stop_reason,
            updates: collected.updates,
            unrecognized_updates: collected.unrecognized,
        })
    }

    /// `session/prompt` with one text block.
    ///
    /// # Errors
    ///
    /// See [`AcpClient::prompt`].
    pub async fn prompt_text(
        &self,
        session_id: &str,
        text: &str,
    ) -> Result<PromptOutcome, AcpClientError> {
        self.prompt(session_id, vec![ContentBlock::Text(TextContent::new(text))])
            .await
    }

    /// `session/cancel` for `session_id`. This is a notification; it does not wait
    /// for the prompt result. The in-flight [`AcpClient::prompt`] returns when the
    /// peer answers that prompt.
    ///
    /// # Errors
    ///
    /// Returns [`AcpClientError::NotInitialized`] or [`AcpClientError::EmptySessionId`].
    pub async fn cancel(&self, session_id: &str) -> Result<(), AcpClientError> {
        self.ensure_ready()?;
        reject_empty_session(session_id)?;
        let value = serde_json::to_value(CancelNotification::new(SessionId::new(session_id)))
            .map_err(|source| AcpClientError::Encode {
                what: "session/cancel",
                source,
            })?;
        self.inner.io.notify(methods::SESSION_CANCEL, value).await
    }

    /// Bound every JSON-RPC request this client sends.
    ///
    /// `initialize` and `session/new` already wait at most 30 seconds. `session/prompt`
    /// waits until the peer answers or the pipe closes, because a local model can
    /// run longer than a handshake. This method replaces that split with one limit
    /// for all three. `None` waits until the peer answers or stdout closes.
    ///
    /// `session/cancel` is a notification and is not bounded.
    pub fn set_request_timeout(&self, timeout: Option<Duration>) {
        let mut timeouts = lock_timeouts(&self.inner.timeouts);
        timeouts.handshake = timeout;
        timeouts.prompt = timeout;
    }

    fn handshake_timeout(&self) -> Option<Duration> {
        lock_timeouts(&self.inner.timeouts).handshake
    }

    fn prompt_timeout(&self) -> Option<Duration> {
        lock_timeouts(&self.inner.timeouts).prompt
    }

    /// `session/new` and `session/prompt` may return ACP `auth_required`.
    ///
    /// That outcome shuts the child down. The error tells the operator how to
    /// configure a provider key outside Caliche. It does not call
    /// `authenticate` and it does not spawn a login process.
    async fn peer_result(
        &self,
        method: &str,
        value: Value,
        timeout: Option<Duration>,
    ) -> Result<Value, AcpClientError> {
        match self.inner.io.request(method, value, timeout).await {
            Ok(raw) => Ok(raw),
            Err(err) => Err(self.fail_auth_required(err).await),
        }
    }

    async fn fail_auth_required(&self, err: AcpClientError) -> AcpClientError {
        let mapped = map_auth_required(err);
        if matches!(mapped, AcpClientError::AuthRequired { .. }) {
            let _ = self.shutdown().await;
        }
        mapped
    }

    /// Stop the reader and kill the child.
    ///
    /// # Errors
    ///
    /// Returns [`AcpClientError::Io`] when waiting for the child fails.
    pub async fn shutdown(&self) -> Result<(), AcpClientError> {
        self.inner.io.set_phase(PHASE_CLOSED);
        self.inner.io.fail_all(&WaiterFail::Closed);
        self.inner.reader_abort.abort();
        self.inner.stderr_abort.abort();
        let child = {
            let mut slot = self.inner.child.lock().await;
            slot.take()
        };
        let Some(mut child) = child else {
            return Ok(());
        };
        let _ = child.start_kill();
        child
            .wait()
            .await
            .map(|_| ())
            .map_err(|source| AcpClientError::Io { source })
    }

    /// Recent stderr lines from the child, oldest first.
    ///
    /// Stderr is not part of the ACP pipe. Use it to see why a peer exited.
    #[must_use]
    pub fn stderr_tail(&self) -> String {
        stderr_text(&self.inner.stderr_tail)
    }

    /// Session notes dropped because a parked queue was full.
    ///
    /// Counts idle notes evicted past the idle cap, foreign replay notes
    /// evicted past the replay cap, and foreign idle notes that found the replay
    /// queue full when a prompt finished. An idle note for the session that is
    /// prompted next is discarded by design and is not counted. A drop warns
    /// once, then again with the episode count at the next begin or finish.
    #[must_use]
    pub fn parked_notes_dropped(&self) -> u64 {
        self.inner.notes.dropped()
    }

    fn ensure_ready(&self) -> Result<(), AcpClientError> {
        match self.inner.io.phase() {
            PHASE_READY => Ok(()),
            PHASE_SPAWNED => Err(AcpClientError::NotInitialized),
            _ => Err(AcpClientError::ShutDown),
        }
    }
}

struct Pipes {
    stdin: tokio::process::ChildStdin,
    stdout: tokio::process::ChildStdout,
    stderr: tokio::process::ChildStderr,
}

fn open_transcript(path: Option<&Path>) -> Result<Option<Arc<Transcript>>, AcpClientError> {
    let Some(path) = path else {
        return Ok(None);
    };
    Transcript::create(path)
        .map(|transcript| Some(Arc::new(transcript)))
        .map_err(|source| AcpClientError::Transcript { source })
}

fn child_from_spec(
    argv: &[std::ffi::OsString],
    cwd: &Path,
    env: &[(std::ffi::OsString, std::ffi::OsString)],
    inherit_env: bool,
) -> Result<Child, AcpClientError> {
    let program = argv.first().ok_or(AcpClientError::EmptyArgv)?;
    if program.is_empty() {
        return Err(AcpClientError::EmptyArgv);
    }
    let cwd = absolute_path(cwd)?;
    let mut command = Command::new(program);
    command
        .args(&argv[1..])
        .current_dir(&cwd)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true);
    apply_env(&mut command, env, inherit_env);
    command.spawn().map_err(|source| AcpClientError::Spawn {
        program: program.to_string_lossy().into_owned(),
        source,
    })
}

fn apply_env(
    command: &mut Command,
    env: &[(std::ffi::OsString, std::ffi::OsString)],
    inherit_env: bool,
) {
    use caliche_contracts::child_env::{absolute_search_path, is_search_path_key};
    if !inherit_env {
        command
            .env_clear()
            .envs(caliche_contracts::child_env::host_child_env());
    }
    for (key, value) in env {
        // A configured search path is filtered like the host's.
        match (key.to_str(), value.to_str()) {
            (Some(name), Some(path)) if !inherit_env && is_search_path_key(name) => {
                command.env(key, absolute_search_path(path));
            }
            _ => {
                command.env(key, value);
            }
        }
    }
}

fn take_pipes(child: &mut Child) -> Result<Pipes, AcpClientError> {
    Ok(Pipes {
        stdin: child
            .stdin
            .take()
            .ok_or(AcpClientError::StdioUnavailable { stream: "stdin" })?,
        stdout: child
            .stdout
            .take()
            .ok_or(AcpClientError::StdioUnavailable { stream: "stdout" })?,
        stderr: child
            .stderr
            .take()
            .ok_or(AcpClientError::StdioUnavailable { stream: "stderr" })?,
    })
}

fn initialize_value() -> Result<serde_json::Value, AcpClientError> {
    let params = InitializeRequest::new(ACP_PROTOCOL_VERSION)
        .client_capabilities(
            ClientCapabilities::new()
                .fs(FileSystemCapabilities::new())
                .terminal(false),
        )
        .client_info(Implementation::new("caliche", env!("CARGO_PKG_VERSION")).title("Caliche"));
    serde_json::to_value(params).map_err(|source| AcpClientError::Encode {
        what: "initialize",
        source,
    })
}

fn cached_init(
    cache: &StdMutex<Option<InitializeResponse>>,
) -> Result<InitializeResponse, AcpClientError> {
    lock_cache(cache)
        .clone()
        .ok_or(AcpClientError::NotInitialized)
}

const fn reject_empty_session(session_id: &str) -> Result<(), AcpClientError> {
    if session_id.is_empty() {
        Err(AcpClientError::EmptySessionId)
    } else {
        Ok(())
    }
}

fn absolute_path(path: &Path) -> Result<PathBuf, AcpClientError> {
    if path.as_os_str().is_empty() {
        return Err(AcpClientError::EmptyCwd);
    }
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    let base =
        std::env::current_dir().map_err(|source| AcpClientError::WorkingDirectory { source })?;
    Ok(base.join(path))
}

fn path_to_string(path: &Path) -> Result<String, AcpClientError> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or(AcpClientError::PathNotUtf8)
}

fn append_agent_text(out: &mut String, update: &SessionUpdate) {
    let SessionUpdate::AgentMessageChunk(ContentChunk { content, .. }) = update else {
        return;
    };
    let ContentBlock::Text(text) = content else {
        return;
    };
    out.push_str(&text.text);
}

fn lock_timeouts(timeouts: &StdMutex<CallTimeouts>) -> std::sync::MutexGuard<'_, CallTimeouts> {
    timeouts
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn lock_cache(
    cache: &StdMutex<Option<InitializeResponse>>,
) -> std::sync::MutexGuard<'_, Option<InitializeResponse>> {
    cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use super::{absolute_path, initialize_value};
    use std::path::Path;

    #[test]
    fn initialize_does_not_offer_terminal_auth() {
        let value = initialize_value().expect("encode initialize");
        assert_ne!(
            value
                .pointer("/clientCapabilities/auth/terminal")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert!(
            value
                .pointer("/clientCapabilities/_meta/terminal-auth")
                .is_none()
        );
        assert!(value.pointer("/_meta/terminal-auth").is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn peer_gets_host_tokens_only_when_it_opts_in() {
        use super::{AcpSpawn, child_from_spec};

        async fn probe(spec: AcpSpawn) -> String {
            let child =
                child_from_spec(&spec.argv, &spec.cwd, &spec.env, spec.inherit_env).expect("spawn");
            let output = child.wait_with_output().await.expect("wait");
            String::from_utf8(output.stdout).expect("utf8")
        }
        let spec = || {
            AcpSpawn::new(
                [
                    "sh",
                    "-c",
                    "printf '%s|%s' \"${HOME:+home}\" \"${CALICHE_ACP_SCRUB_FOO_TOKEN-unset}\"",
                ],
                ".",
            )
        };

        // The token must be in this process's environment, which tests may
        // not mutate: re-run this one test in a child that has it.
        if std::env::var_os("CALICHE_ACP_SCRUB_FOO_TOKEN").is_none() {
            let name = "client::tests::peer_gets_host_tokens_only_when_it_opts_in";
            let output =
                tokio::process::Command::new(std::env::current_exe().expect("test binary"))
                    .args(["--exact", name, "--nocapture"])
                    .env("CALICHE_ACP_SCRUB_FOO_TOKEN", "planted")
                    .output()
                    .await
                    .expect("re-run test");
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(
                output.status.success() && stdout.contains("1 passed"),
                "{stdout}{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }
        assert_eq!(probe(spec()).await, "home|unset");
        assert_eq!(probe(spec().inherit_env(true)).await, "home|planted");
    }

    /// A configured `PATH` reaches a scrubbed peer without its relative
    /// entries.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_configured_peer_path_loses_its_relative_entries() {
        use super::{AcpSpawn, child_from_spec};
        use caliche_contracts::child_env::PATH_VAR;
        let spec = AcpSpawn::new(["sh", "-c", "printf %s \"$PATH\""], ".")
            .env(PATH_VAR, ".:/usr/bin::/bin");

        let child =
            child_from_spec(&spec.argv, &spec.cwd, &spec.env, spec.inherit_env).expect("spawn");
        let output = child.wait_with_output().await.expect("wait");

        assert_eq!(
            String::from_utf8(output.stdout).expect("utf8"),
            "/usr/bin:/bin"
        );
    }

    #[test]
    fn relative_cwd_is_absolute() {
        let path = absolute_path(Path::new(".")).unwrap();
        assert!(path.is_absolute());
    }

    #[test]
    fn empty_cwd_is_rejected() {
        assert!(absolute_path(Path::new("")).is_err());
    }
}
