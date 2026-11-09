//! Peer smoke for the ACP v1 client.
//!
//! `acp-peer-smoke` drives [`crate::AcpClient`] through `initialize`,
//! `session/new`, and a short `session/prompt`, and writes a JSONL transcript
//! of the frames. The client accepts an empty `authMethods` list, or a list
//! whose every id is exactly `opencode-login` or `hermes-setup`. Any other id
//! is refused before `session/new`. The client does not send `authenticate`
//! up front. `auth_required` fails closed with out-of-band BYO guidance.
//!
//! # Command matrix
//!
//! | Peer | argv | Auth |
//! | --- | --- | --- |
//! | `OpenCode` | `opencode acp` | Advertises `opencode-login`. That hint is informational. This client continues to `session/new` and does not run `opencode auth login`. On `auth_required`, put `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (or a local base URL in `opencode.json`) on the child outside Caliche. Optional diagnostics: `opencode --print-logs acp` (stderr only). Install: <https://opencode.ai/docs/acp/>. |
//! | Hermes | `hermes acp` | Always advertises terminal method `hermes-setup` with args `--setup`. A list that is only that id continues. A provider id beside it, such as `openrouter`, is refused before `session/new`. This client does not run `hermes acp --setup` or `hermes setup`. On `auth_required`, run `hermes config set` for the provider key outside Caliche. |
//! | Hermes launcher | `hermes-acp` | Same server and the same informational `hermes-setup` hint as `hermes acp`. |
//! | Hermes adapter | `python -m acp_adapter` | Same adapter after `python -c "import pm; pm.sync_venv(['acp'], explicit=True)"` in the Hermes checkout. Same informational hint. |
//! | Fixture | `python3 tests/fixture_agent.py <mode>` | `happy` returns empty `authMethods` and finishes a turn. `auth-terminal` and `auth-opencode` are hint-only and still finish. `auth-required` returns `auth_required` after `session/new`. `auth-vendor`, `auth-codex`, `auth-custom-terminal`, `auth-openrouter`, and `auth-generic` are refused before `session/new`. `permission` asks `session/request_permission`; this client answers `cancelled`. |
//!
//! `--peer auto` uses `OpenCode` when `opencode` is on `PATH` or at
//! `~/.opencode/bin/opencode`, otherwise `hermes` or `hermes-acp`. It does not
//! fall through to the fixture. Provider keys belong in the child environment.
//! The transcript redacts credential-shaped environment values.

use std::ffi::OsString;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::time::Duration;

use caliche_contracts::acp::methods;
use serde::Serialize;
use serde_json::Value;

use crate::client::{AcpClient, AcpSpawn};
use crate::error::AcpClientError;
use crate::transcript::Redactor;

/// One documented peer launch.
#[derive(Debug, Clone, Copy)]
pub struct MatrixEntry {
    /// Stable peer id (`opencode`, `hermes`, `hermes-acp`, `acp_adapter`, `fixture`).
    pub id: &'static str,
    /// argv an operator runs, not including shell quoting.
    pub argv: &'static [&'static str],
    /// What `initialize` advertises, and what this client does about it.
    pub auth: &'static str,
}

/// Which peer the smoke should spawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerKind {
    /// `OpenCode` if it is installed, otherwise Hermes.
    Auto,
    /// The Python fixture in this crate.
    Fixture,
    /// `opencode acp`.
    OpenCode,
    /// `hermes acp` or `hermes-acp`.
    Hermes,
}

impl PeerKind {
    /// Parse `auto`, `fixture`, `opencode`, or `hermes`.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "auto" => Some(Self::Auto),
            "fixture" => Some(Self::Fixture),
            "opencode" => Some(Self::OpenCode),
            "hermes" => Some(Self::Hermes),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Fixture => "fixture",
            Self::OpenCode => "opencode",
            Self::Hermes => "hermes",
        }
    }
}

/// How to find the peer binary and the fixture script.
#[derive(Debug, Clone)]
pub struct PeerQuery {
    /// Selected peer.
    pub peer: PeerKind,
    /// Explicit `OpenCode` program. `None` searches `PATH` and `~/.opencode/bin/opencode`.
    pub opencode_bin: Option<PathBuf>,
    /// Explicit Hermes program. `None` searches `PATH` for `hermes` and `hermes-acp`.
    pub hermes_bin: Option<PathBuf>,
    /// Path to `fixture_agent.py`.
    pub fixture_script: PathBuf,
    /// Fixture mode argument (`happy`, `auth-terminal`, `auth-vendor`, `permission`, ...).
    pub fixture_mode: String,
    /// Insert `--print-logs` before `acp` for `OpenCode`.
    pub opencode_print_logs: bool,
}

/// A peer program that exists on disk.
#[derive(Debug, Clone)]
pub struct ResolvedPeer {
    /// `opencode`, `hermes`, or `fixture:<mode>`.
    pub id: String,
    /// Program and arguments.
    pub argv: Vec<OsString>,
}

/// The selected peer program was not found.
#[derive(Debug, Clone)]
pub struct PeerMissing {
    /// Requested peer id.
    pub peer: String,
    /// Paths that were checked, plus the command matrix.
    pub detail: String,
}

impl std::fmt::Display for PeerMissing {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "ACP peer `{}` is not available: {}",
            self.peer, self.detail
        )
    }
}

impl std::error::Error for PeerMissing {}

/// Inputs for one smoke run.
#[derive(Debug, Clone)]
pub struct SmokeConfig {
    /// Peer selection.
    pub peer: PeerQuery,
    /// Absolute or relative working directory passed to the child and to `session/new`.
    pub cwd: PathBuf,
    /// User text for `session/prompt`.
    pub prompt: String,
    /// JSONL path for the frame transcript.
    pub transcript: PathBuf,
    /// Per-call timeout for `initialize`, `session/new`, and `session/prompt`.
    pub timeout: Duration,
    /// `Some(true)` passes the full parent environment, provider keys
    /// included. Otherwise the peer gets the scrubbed host environment.
    pub inherit_env: Option<bool>,
    /// Extra environment entries applied after inheritance.
    pub env: Vec<(OsString, OsString)>,
}

/// Result of one smoke run. Safe to write to disk: credential-shaped
/// environment values are replaced with `[redacted]`.
#[derive(Debug, Clone, Serialize)]
pub struct SmokeReport {
    /// Peer id that ran, or the requested id when the program was missing.
    pub peer_id: String,
    /// argv as lossy text.
    pub argv: Vec<String>,
    /// Outcome class.
    pub status: SmokeStatus,
    /// Operator-facing explanation.
    pub detail: String,
    /// `sessionId` when `session/new` completed.
    pub session_id: Option<String>,
    /// Wire `stopReason` when `session/prompt` completed.
    pub stop_reason: Option<String>,
    /// Concatenated `agent_message_chunk` text.
    pub agent_text: String,
    /// Scrubbed child stderr tail.
    pub stderr: String,
    /// `authMethods[].id` from the peer `initialize` result, when present.
    pub auth_method_ids: Vec<String>,
    /// `protocolVersion` from the peer `initialize` result, when present.
    pub protocol_version: Option<u64>,
    /// Protocol observations from the transcript.
    pub events: Vec<SmokeEvent>,
    /// `result.outcome.outcome` from the client's permission reply.
    pub permission_outcome: Option<String>,
    /// Transcript path.
    pub transcript: String,
}

/// A protocol observation recorded from the transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SmokeEvent {
    /// The client sent `authenticate`.
    Authenticate,
    /// The client sent `session/new`.
    SessionNew,
    /// The client sent `session/prompt`.
    SessionPrompt,
    /// The peer sent `session/request_permission`.
    PermissionRequest,
}

/// Smoke outcome. Process exit codes match [`SmokeStatus::code`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SmokeStatus {
    /// `session/prompt` returned a stop reason.
    Completed,
    /// Spawn, protocol, or timeout failure.
    Failed,
    /// Peer advertised an auth method outside the hint allowlist. The client disconnected before `session/new`.
    AuthRejected,
    /// Peer returned ACP `auth_required` after `initialize`. The client did not log in.
    AuthRequired,
    /// The selected peer program was not on disk.
    PeerMissing,
}

impl SmokeStatus {
    /// Process exit code for this outcome.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::Completed => 0,
            Self::Failed => 1,
            Self::AuthRejected | Self::AuthRequired => 2,
            Self::PeerMissing => 3,
        }
    }

    /// Stable status name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::AuthRejected => "auth_rejected",
            Self::AuthRequired => "auth_required",
            Self::PeerMissing => "peer_missing",
        }
    }
}

/// Documented `OpenCode`, Hermes, and fixture launches.
#[must_use]
pub const fn peer_matrix() -> &'static [MatrixEntry] {
    const MATRIX: &[MatrixEntry] = &[
        MatrixEntry {
            id: "opencode",
            argv: &["opencode", "acp"],
            auth: "OpenCode initialize advertises id `opencode-login`, name `Login with opencode`, description `Run opencode auth login in the terminal` (no `type` field). This client treats that id as informational, continues to session/new, and does not call `authenticate` or run `opencode auth login`. On auth_required, set OPENAI_API_KEY or ANTHROPIC_API_KEY on the child, or point opencode.json at a local base URL. Optional logs: `opencode --print-logs acp` writes to stderr. Install from https://opencode.ai/install or `npm install -g opencode-ai`.",
        },
        MatrixEntry {
            id: "hermes",
            argv: &["hermes", "acp"],
            auth: "Hermes always advertises terminal auth id `hermes-setup`, args `--setup`. This client continues only when every advertised id is `hermes-setup` or `opencode-login`. A provider id beside `hermes-setup` is refused before session/new. This client does not run `hermes acp --setup` or `hermes setup`. On auth_required, run `hermes config set` for the provider key outside Caliche. Install: `curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash` with `--non-interactive`, then enable ACP with `python -c \"import pm; pm.sync_venv(['acp'], explicit=True)\"` from the Hermes checkout.",
        },
        MatrixEntry {
            id: "hermes-acp",
            argv: &["hermes-acp"],
            auth: "Same ACP server and the same informational `hermes-setup` hint as `hermes acp`. This client does not append `--setup`.",
        },
        MatrixEntry {
            id: "acp_adapter",
            argv: &["python", "-m", "acp_adapter"],
            auth: "Same adapter as `hermes acp` after the ACP extra is installed. Same informational `hermes-setup` hint. This smoke does not auto-detect `python -m acp_adapter`; pass `--hermes` only for `hermes` or `hermes-acp`, or run the adapter by putting it on PATH as one of those names.",
        },
        MatrixEntry {
            id: "fixture",
            argv: &[
                "python3",
                "crates/caliche_acp_client/tests/fixture_agent.py",
                "happy",
            ],
            auth: "`happy` returns an empty authMethods list and completes a turn. `auth-terminal` (`hermes-setup`) and `auth-opencode` (`opencode-login`) still reach session/new. `auth-required` fails closed after session/new. Any other id, including `custom-terminal`, `openrouter`, and `custom-login`, is refused before session/new. `permission` sends session/request_permission; this client answers cancelled.",
        },
    ];
    MATRIX
}

/// Locate the peer program.
///
/// # Errors
///
/// Returns [`PeerMissing`] when the program or fixture script is not a file.
pub fn resolve_peer(query: &PeerQuery) -> Result<ResolvedPeer, PeerMissing> {
    match query.peer {
        PeerKind::Fixture => resolve_fixture(query),
        PeerKind::OpenCode => Ok(make_peer("opencode", resolve_opencode(query)?)),
        PeerKind::Hermes => Ok(make_peer("hermes", resolve_hermes(query)?)),
        PeerKind::Auto => resolve_auto(query),
    }
}

/// Spawn the peer and run initialize, session/new, and session/prompt.
///
/// Auth refusal and a missing binary are reports, not panics. The transcript
/// file is created when the process is spawned.
pub async fn run_smoke(config: SmokeConfig) -> SmokeReport {
    let redactor = Redactor::from_env();
    if config.prompt.is_empty() {
        return scrub_report(
            &redactor,
            bare_report(
                "unknown",
                Vec::new(),
                &config.transcript,
                SmokeStatus::Failed,
                "prompt must not be empty",
            ),
        );
    }
    let resolved = match resolve_peer(&config.peer) {
        Ok(peer) => peer,
        Err(missing) => {
            return scrub_report(&redactor, missing_report(&missing, &config.transcript));
        }
    };
    let argv = argv_text(&resolved.argv);
    let peer_id = resolved.id.clone();
    let inherit = config.inherit_env.unwrap_or(false);
    let client = match spawn_peer(&config, &resolved, inherit) {
        Ok(client) => client,
        Err(err) => {
            return scrub_report(
                &redactor,
                bare_report(
                    &peer_id,
                    argv,
                    &config.transcript,
                    SmokeStatus::Failed,
                    &err.to_string(),
                ),
            );
        }
    };
    let driven = drive(&client, &config.cwd, &config.prompt, config.timeout).await;
    let stderr = client.stderr_tail();
    let _ = client.shutdown().await;
    drop(client);
    let mut report = report_from_drive(&peer_id, argv, &config.transcript, driven, stderr);
    apply_notes(&mut report, &config.transcript);
    scrub_report(&redactor, report)
}

fn spawn_peer(
    config: &SmokeConfig,
    resolved: &ResolvedPeer,
    inherit: bool,
) -> Result<AcpClient, AcpClientError> {
    let mut spawn = AcpSpawn::new(resolved.argv.clone(), &config.cwd)
        .inherit_env(inherit)
        .transcript(&config.transcript);
    for (key, value) in &config.env {
        spawn = spawn.env(key, value);
    }
    AcpClient::spawn(spawn)
}

fn make_peer(id: &str, argv: Vec<OsString>) -> ResolvedPeer {
    ResolvedPeer {
        id: id.to_owned(),
        argv,
    }
}

fn resolve_auto(query: &PeerQuery) -> Result<ResolvedPeer, PeerMissing> {
    if let Ok(argv) = resolve_opencode(query) {
        return Ok(make_peer("opencode", argv));
    }
    if let Ok(argv) = resolve_hermes(query) {
        return Ok(make_peer("hermes", argv));
    }
    Err(PeerMissing {
        peer: "auto".to_owned(),
        detail: format!(
            "neither OpenCode nor Hermes was found.\n{}\n{}\n{}",
            opencode_search_detail(query),
            hermes_search_detail(query),
            matrix_text()
        ),
    })
}

fn resolve_fixture(query: &PeerQuery) -> Result<ResolvedPeer, PeerMissing> {
    if !is_file(&query.fixture_script) {
        return Err(PeerMissing {
            peer: "fixture".to_owned(),
            detail: format!(
                "fixture script is missing at {}",
                query.fixture_script.display()
            ),
        });
    }
    let Some(python) = command_on_path("python3") else {
        return Err(PeerMissing {
            peer: "fixture".to_owned(),
            detail: "python3 was not found on PATH".to_owned(),
        });
    };
    Ok(ResolvedPeer {
        id: format!("fixture:{}", query.fixture_mode),
        argv: vec![
            python.into_os_string(),
            query.fixture_script.as_os_str().to_os_string(),
            OsString::from(&query.fixture_mode),
        ],
    })
}

fn resolve_opencode(query: &PeerQuery) -> Result<Vec<OsString>, PeerMissing> {
    let Some(program) = opencode_program(query) else {
        return Err(PeerMissing {
            peer: "opencode".to_owned(),
            detail: format!("{}\n{}", opencode_search_detail(query), matrix_text()),
        });
    };
    let mut argv = vec![program];
    if query.opencode_print_logs {
        argv.push(OsString::from("--print-logs"));
    }
    argv.push(OsString::from("acp"));
    Ok(argv)
}

fn resolve_hermes(query: &PeerQuery) -> Result<Vec<OsString>, PeerMissing> {
    if let Some(path) = &query.hermes_bin {
        if !is_file(path) {
            return Err(PeerMissing {
                peer: "hermes".to_owned(),
                detail: format!("Hermes program is missing at {}", path.display()),
            });
        }
        return Ok(hermes_argv(path));
    }
    if let Some(path) = command_on_path("hermes") {
        return Ok(hermes_argv(&path));
    }
    if let Some(path) = command_on_path("hermes-acp") {
        return Ok(vec![path.into_os_string()]);
    }
    Err(PeerMissing {
        peer: "hermes".to_owned(),
        detail: format!("{}\n{}", hermes_search_detail(query), matrix_text()),
    })
}

fn hermes_argv(path: &Path) -> Vec<OsString> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("");
    if name == "hermes-acp" {
        vec![path.as_os_str().to_os_string()]
    } else {
        vec![path.as_os_str().to_os_string(), OsString::from("acp")]
    }
}

fn opencode_program(query: &PeerQuery) -> Option<OsString> {
    if let Some(path) = &query.opencode_bin {
        return is_file(path).then(|| path.as_os_str().to_os_string());
    }
    if let Some(found) = command_on_path("opencode") {
        return Some(found.into_os_string());
    }
    home_opencode()
}

fn home_opencode() -> Option<OsString> {
    let home = std::env::var_os("HOME")?;
    let path = PathBuf::from(home).join(".opencode/bin/opencode");
    is_file(&path).then(|| path.into_os_string())
}

fn opencode_search_detail(query: &PeerQuery) -> String {
    query.opencode_bin.as_ref().map_or_else(
        || "OpenCode: looked on PATH for `opencode` and at ~/.opencode/bin/opencode".to_owned(),
        |path| format!("OpenCode: looked for {}", path.display()),
    )
}

fn hermes_search_detail(query: &PeerQuery) -> String {
    query.hermes_bin.as_ref().map_or_else(
        || "Hermes: looked on PATH for `hermes` and `hermes-acp`".to_owned(),
        |path| format!("Hermes: looked for {}", path.display()),
    )
}

fn matrix_text() -> String {
    let mut text = String::from("command matrix:");
    for entry in peer_matrix() {
        text.push('\n');
        text.push_str(entry.id);
        text.push_str(": ");
        text.push_str(&entry.argv.join(" "));
    }
    text
}

fn command_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if is_file(&candidate) {
            return Some(candidate);
        }
    }
    None
}

fn is_file(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|meta| meta.is_file())
}

fn argv_text(argv: &[OsString]) -> Vec<String> {
    argv.iter()
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect()
}

enum DriveResult {
    AuthRejected(String),
    AuthRequired {
        detail: String,
        session_id: Option<String>,
    },
    Failed(String),
    Completed {
        session_id: String,
        stop_reason: String,
        agent_text: String,
    },
}

async fn drive(client: &AcpClient, cwd: &Path, prompt: &str, timeout: Duration) -> DriveResult {
    match timeout_op(timeout, client.initialize()).await {
        Ok(Ok(_)) => {}
        Ok(Err(err)) => return classify_init(err),
        Err(()) => return DriveResult::Failed("initialize timed out".to_owned()),
    }
    let session = match timeout_op(timeout, client.new_session(cwd)).await {
        Ok(Ok(session)) => session,
        Ok(Err(err)) => return classify_peer(err, None),
        Err(()) => return DriveResult::Failed("session/new timed out".to_owned()),
    };
    let session_id = session.session_id.0.to_string();
    match timeout_op(timeout, client.prompt_text(&session_id, prompt)).await {
        Ok(Ok(outcome)) => DriveResult::Completed {
            session_id,
            stop_reason: stop_reason_name(outcome.stop_reason),
            agent_text: outcome.agent_text(),
        },
        Ok(Err(err)) => classify_peer(err, Some(session_id)),
        Err(()) => DriveResult::Failed("session/prompt timed out".to_owned()),
    }
}

fn classify_init(err: AcpClientError) -> DriveResult {
    classify_peer(err, None)
}

fn classify_peer(err: AcpClientError, session_id: Option<String>) -> DriveResult {
    match err {
        AcpClientError::AuthRejected { message } => DriveResult::AuthRejected(message),
        AcpClientError::AuthRequired { message } => DriveResult::AuthRequired {
            detail: message,
            session_id,
        },
        other => DriveResult::Failed(other.to_string()),
    }
}

async fn timeout_op<T>(duration: Duration, future: impl Future<Output = T>) -> Result<T, ()> {
    tokio::time::timeout(duration, future).await.map_err(|_| ())
}

fn stop_reason_name(reason: impl Serialize) -> String {
    serde_json::to_value(reason)
        .ok()
        .and_then(|value| value.as_str().map(ToOwned::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn report_from_drive(
    peer_id: &str,
    argv: Vec<String>,
    transcript: &Path,
    driven: DriveResult,
    stderr: String,
) -> SmokeReport {
    let (status, detail, session_id, stop_reason, agent_text) = match driven {
        DriveResult::Completed {
            session_id,
            stop_reason,
            agent_text,
        } => (
            SmokeStatus::Completed,
            "session/prompt returned".to_owned(),
            Some(session_id),
            Some(stop_reason),
            agent_text,
        ),
        DriveResult::AuthRejected(detail) => {
            (SmokeStatus::AuthRejected, detail, None, None, String::new())
        }
        DriveResult::AuthRequired { detail, session_id } => (
            SmokeStatus::AuthRequired,
            detail,
            session_id,
            None,
            String::new(),
        ),
        DriveResult::Failed(detail) => (SmokeStatus::Failed, detail, None, None, String::new()),
    };
    SmokeReport {
        peer_id: peer_id.to_owned(),
        argv,
        status,
        detail,
        session_id,
        stop_reason,
        agent_text,
        stderr,
        auth_method_ids: Vec::new(),
        protocol_version: None,
        events: Vec::new(),
        permission_outcome: None,
        transcript: transcript.display().to_string(),
    }
}

fn missing_report(missing: &PeerMissing, transcript: &Path) -> SmokeReport {
    bare_report(
        &missing.peer,
        Vec::new(),
        transcript,
        SmokeStatus::PeerMissing,
        &missing.detail,
    )
}

fn bare_report(
    peer_id: &str,
    argv: Vec<String>,
    transcript: &Path,
    status: SmokeStatus,
    detail: &str,
) -> SmokeReport {
    SmokeReport {
        peer_id: peer_id.to_owned(),
        argv,
        status,
        detail: detail.to_owned(),
        session_id: None,
        stop_reason: None,
        agent_text: String::new(),
        stderr: String::new(),
        auth_method_ids: Vec::new(),
        protocol_version: None,
        events: Vec::new(),
        permission_outcome: None,
        transcript: transcript.display().to_string(),
    }
}

#[derive(Default)]
struct TranscriptNotes {
    auth_method_ids: Vec<String>,
    protocol_version: Option<u64>,
    events: Vec<SmokeEvent>,
    permission_outcome: Option<String>,
}

fn apply_notes(report: &mut SmokeReport, path: &Path) {
    let notes = read_transcript_notes(path);
    report.auth_method_ids = notes.auth_method_ids;
    report.protocol_version = notes.protocol_version;
    report.events = notes.events;
    report.permission_outcome = notes.permission_outcome;
}

fn read_transcript_notes(path: &Path) -> TranscriptNotes {
    let Ok(text) = std::fs::read_to_string(path) else {
        return TranscriptNotes::default();
    };
    let mut notes = TranscriptNotes::default();
    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        absorb_note(&mut notes, &value);
    }
    notes
}

fn absorb_note(notes: &mut TranscriptNotes, row: &Value) {
    let Some(frame) = row.get("frame") else {
        return;
    };
    match row.get("dir").and_then(Value::as_str) {
        Some("client") => absorb_client(notes, frame),
        Some("peer") => absorb_peer(notes, frame),
        _ => {}
    }
}

fn absorb_client(notes: &mut TranscriptNotes, frame: &Value) {
    match frame.get("method").and_then(Value::as_str) {
        Some(methods::AUTHENTICATE) => push_event(notes, SmokeEvent::Authenticate),
        Some(methods::SESSION_NEW) => push_event(notes, SmokeEvent::SessionNew),
        Some(methods::SESSION_PROMPT) => push_event(notes, SmokeEvent::SessionPrompt),
        _ => {}
    }
    if let Some(outcome) = frame
        .pointer("/result/outcome/outcome")
        .and_then(Value::as_str)
    {
        notes.permission_outcome = Some(outcome.to_owned());
    }
}

fn absorb_peer(notes: &mut TranscriptNotes, frame: &Value) {
    if frame.get("method").and_then(Value::as_str) == Some(methods::SESSION_REQUEST_PERMISSION) {
        push_event(notes, SmokeEvent::PermissionRequest);
    }
    let Some(result) = frame.get("result") else {
        return;
    };
    if let Some(version) = result.get("protocolVersion").and_then(Value::as_u64) {
        notes.protocol_version = Some(version);
    }
    if let Some(methods) = result.get("authMethods").and_then(Value::as_array) {
        notes.auth_method_ids = methods.iter().filter_map(auth_method_id).collect();
    }
}

fn push_event(notes: &mut TranscriptNotes, event: SmokeEvent) {
    if !notes.events.contains(&event) {
        notes.events.push(event);
    }
}

fn auth_method_id(value: &Value) -> Option<String> {
    value
        .get("id")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
}

fn scrub_report(redactor: &Redactor, mut report: SmokeReport) -> SmokeReport {
    report.detail = redactor.scrub(&report.detail);
    report.agent_text = redactor.scrub(&report.agent_text);
    report.stderr = redactor.scrub(&report.stderr);
    report
}

#[cfg(test)]
mod tests {
    use super::{PeerKind, PeerQuery, hermes_argv, peer_matrix, resolve_peer};
    use std::path::PathBuf;

    fn query(peer: PeerKind) -> PeerQuery {
        PeerQuery {
            peer,
            opencode_bin: None,
            hermes_bin: None,
            fixture_script: PathBuf::from("missing-fixture.py"),
            fixture_mode: "happy".to_owned(),
            opencode_print_logs: false,
        }
    }

    #[test]
    fn matrix_lists_opencode_hermes_and_fixture() {
        let ids: Vec<_> = peer_matrix().iter().map(|entry| entry.id).collect();
        assert!(ids.contains(&"opencode"));
        assert!(ids.contains(&"hermes"));
        assert!(ids.contains(&"hermes-acp"));
        assert!(ids.contains(&"acp_adapter"));
        assert!(ids.contains(&"fixture"));
        assert!(
            peer_matrix()
                .iter()
                .any(|entry| entry.id == "opencode" && entry.argv == ["opencode", "acp"])
        );
        assert!(
            peer_matrix()
                .iter()
                .any(|entry| entry.id == "hermes" && entry.argv == ["hermes", "acp"])
        );
        for entry in peer_matrix() {
            assert!(
                !entry.argv.contains(&"--setup"),
                "{} must not launch setup",
                entry.id
            );
            assert!(
                !entry.argv.contains(&"auth"),
                "{} must not launch auth login",
                entry.id
            );
        }
    }

    #[test]
    fn hermes_acp_launcher_has_no_extra_arg() {
        let argv = hermes_argv(std::path::Path::new("/tmp/hermes-acp"));
        assert_eq!(argv.len(), 1);
        let argv = hermes_argv(std::path::Path::new("/tmp/hermes"));
        assert_eq!(argv.len(), 2);
        assert_eq!(argv[1], "acp");
    }

    #[test]
    fn explicit_missing_opencode_is_an_error() {
        let mut query = query(PeerKind::OpenCode);
        query.opencode_bin = Some(PathBuf::from("/no/such/opencode"));
        let error = resolve_peer(&query).expect_err("missing");
        assert!(error.detail.contains("opencode"), "{error}");
    }

    #[test]
    fn explicit_opencode_file_resolves_to_acp() {
        let path =
            std::env::temp_dir().join(format!("caliche-acp-opencode-{}", std::process::id()));
        std::fs::write(&path, b"#!/bin/sh\n").expect("write stub");
        let mut query = query(PeerKind::OpenCode);
        query.opencode_bin = Some(path.clone());
        query.opencode_print_logs = true;
        let resolved = resolve_peer(&query).expect("resolve");
        let _ = std::fs::remove_file(&path);
        assert_eq!(resolved.id, "opencode");
        assert_eq!(resolved.argv.len(), 3);
        assert_eq!(resolved.argv[1], "--print-logs");
        assert_eq!(resolved.argv[2], "acp");
    }
}
