//! Registry of Caliche HTTP servers started by this user.
//!
//! Each `--server` process writes one JSON file under
//! `$XDG_CONFIG_HOME/caliche/servers/`. `server.lock` remains the single
//! pointer the REPL uses to attach. `caliche server list|status|stop` is
//! the supported way to see and stop those processes.
//!
//! Stop signals a pid only when the record's start token, `caliche`
//! executable (or `caliche.exe`), `--server` argument, and listen port
//! still match. A dead pid is removed as stale and is not signaled. A reused
//! pid, whose start token does not match, is removed and is not signaled.
//! Every other mismatch is left in place so a broken check cannot orphan a
//! live server.

mod inspect;

use crate::build_info::BUILD_ID;
use crate::client::ServerLock;
use anyhow::{Context, Result, bail};
use inspect::Inspection;
use serde::{Deserialize, Serialize};
use std::fs;
use std::future::Future;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

/// Environment variable set by the user daemon unit.
///
/// The value `daemon` marks a server started by launchd or systemd. Any other
/// value, including an unset variable, is a foreground `caliche --server`.
pub const SERVER_ORIGIN_ENV: &str = "CALICHE_SERVER_ORIGIN";

/// File name of the supported server binary. Stop refuses other executables.
pub const SERVER_EXE_NAME: &str = "caliche";

/// Bump on any change a reader of this version could not parse: a new field,
/// a removed or renamed field, or a new [`ServerOrigin`] value. Readers check
/// the version before the rest of the record, so an older build skips a newer
/// record instead of quarantining it.
const RECORD_VERSION: u32 = 1;
const TERM_POLLS: u32 = 40;
const KILL_POLLS: u32 = 20;

/// How a Caliche HTTP server was started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ServerOrigin {
    /// `caliche --server` in a terminal.
    Foreground,
    /// launchd or systemd unit that execs `caliche --server`.
    Daemon,
    /// Loopback server embedded in a REPL process.
    Embedded,
}

impl ServerOrigin {
    /// Stable origin token stored in the record id.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Foreground => "foreground",
            Self::Daemon => "daemon",
            Self::Embedded => "embedded",
        }
    }

    /// Map the daemon unit environment value to an origin.
    ///
    /// Only the exact string `daemon` selects [`ServerOrigin::Daemon`].
    #[must_use]
    pub fn from_origin_env(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            Some("daemon") => Self::Daemon,
            _ => Self::Foreground,
        }
    }

    /// Read [`SERVER_ORIGIN_ENV`] from the process environment.
    #[must_use]
    pub fn from_process_env() -> Self {
        let value = std::env::var(SERVER_ORIGIN_ENV).ok();
        Self::from_origin_env(value.as_deref())
    }
}

/// One recorded Caliche HTTP server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerRecord {
    pub version: u32,
    pub id: String,
    pub pid: u32,
    pub port: u16,
    pub bind: String,
    pub build_id: String,
    pub started_at: u64,
    pub cwd: String,
    pub exe: String,
    pub origin: ServerOrigin,
    /// Process-creation token captured at registration.
    ///
    /// Linux stores `starttime` from `/proc`. Windows stores the creation
    /// `FILETIME`. Other Unix systems store `ps -o lstart=` read in the C
    /// locale and UTC; a mismatch there keeps the record and does not signal. Stop requires the live process to still carry this token
    /// so a recycled pid is not signaled.
    pub start_token: String,
}

/// Computed state for `server list`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerState {
    Running,
    Stale,
    Mismatch,
}

impl ServerState {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Stale => "stale",
            Self::Mismatch => "mismatch",
        }
    }
}

/// A registry record plus the live check used by list and status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedServer {
    pub record: ServerRecord,
    pub state: ServerState,
    pub detail: String,
}

/// Which records `server stop` should consider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServerSelector {
    Id(String),
    Port(u16),
    Both { id: String, port: u16 },
}

impl std::fmt::Display for ServerSelector {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Id(id) => write!(formatter, "id {id}"),
            Self::Port(port) => write!(formatter, "port {port}"),
            Self::Both { id, port } => write!(formatter, "id {id} port {port}"),
        }
    }
}

impl ServerSelector {
    pub(crate) fn matches(&self, record: &ServerRecord) -> bool {
        match self {
            Self::Id(id) => record.id == *id,
            Self::Port(port) => record.port == *port,
            Self::Both { id, port } => record.id == *id && record.port == *port,
        }
    }
}

/// Lines printed by `server stop`, and whether the command must fail.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StopReport {
    pub lines: Vec<String>,
    pub failed: bool,
}

struct StoredRecord {
    path: PathBuf,
    record: ServerRecord,
}

/// Directory of per-server JSON records next to `server.lock`.
#[must_use]
pub fn default_registry_dir() -> PathBuf {
    match ServerLock::default_path().parent() {
        Some(parent) => parent.join("servers"),
        None => PathBuf::from("servers"),
    }
}

/// Build a record for the current process.
pub fn capture_record(origin: ServerOrigin, bind: &str, port: u16) -> Result<ServerRecord> {
    let pid = process::id();
    let started_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0);
    let cwd = std::env::current_dir().context("read server working directory")?;
    let exe = std::env::current_exe().context("read server executable")?;
    let start_token = inspect::current_start_token().map_err(|err| anyhow::anyhow!(err))?;
    let id = format!("{}-{port}-{pid}-{started_at}", origin.as_str());
    Ok(ServerRecord {
        version: RECORD_VERSION,
        id,
        pid,
        port,
        bind: bind.to_string(),
        build_id: BUILD_ID.to_string(),
        started_at,
        cwd: cwd.display().to_string(),
        exe: exe.display().to_string(),
        origin,
        start_token,
    })
}

/// Write a record for this process. Used by `--server` and the embedded REPL.
pub fn register_current(
    dir: &Path,
    origin: ServerOrigin,
    bind: &str,
    port: u16,
) -> Result<ServerRecord> {
    let record = capture_record(origin, bind, port)?;
    write_record(dir, &record)?;
    Ok(record)
}

/// Remove registry files whose pid is this process.
pub fn unregister_current(dir: &Path) -> Result<usize> {
    remove_records_matching(dir, process::id(), false)
}

/// Delete records whose pid is `pid` when that pid is not running.
pub fn remove_records_for_pid_if_dead(dir: &Path, pid: u32) -> Result<usize> {
    remove_records_matching(dir, pid, true)
}

/// Delete every record whose pid is not running. Live records are kept.
pub fn prune_dead_records(dir: &Path) -> Result<usize> {
    prune_with(dir, inspect::inspect_live)
}

/// Delete records `inspector` calls [`Inspection::Stale`].
///
/// [`Inspection::Unproven`], [`Inspection::Foreign`], and
/// [`Inspection::Recycled`] stay on disk. A recycled pid is still running.
/// `server stop` removes that record. Prune does not.
fn prune_with(dir: &Path, inspector: impl Fn(&ServerRecord) -> Inspection) -> Result<usize> {
    let mut removed = 0;
    for stored in load_stored(dir)? {
        if matches!(inspector(&stored.record), Inspection::Stale) {
            remove_path(&stored.path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Persist one record atomically.
///
/// The temp file is fsynced before rename so a crash cannot leave an empty
/// record under the final name.
pub fn write_record(dir: &Path, record: &ServerRecord) -> Result<PathBuf> {
    if record.version != RECORD_VERSION {
        bail!("unsupported server record version {}", record.version);
    }
    fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    restrict_dir(dir)?;
    let file_name = record_file_name(&record.id)?;
    let path = dir.join(&file_name);
    let json = serde_json::to_string_pretty(record).context("serialize server record")?;
    let tmp = dir.join(format!("{file_name}.tmp"));
    write_durable(&tmp, json.as_bytes())?;
    restrict_file(&tmp)?;
    fs::rename(&tmp, &path).with_context(|| format!("rename {}", path.display()))?;
    sync_dir(dir)?;
    Ok(path)
}

fn write_durable(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::File::create(path).with_context(|| format!("write {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("fsync {}", path.display()))?;
    Ok(())
}

/// Load and classify every record.
pub fn list_servers(dir: &Path) -> Result<Vec<ListedServer>> {
    list_with(dir, inspect::inspect_live)
}

fn list_with(
    dir: &Path,
    inspector: impl Fn(&ServerRecord) -> Inspection,
) -> Result<Vec<ListedServer>> {
    let stored = load_stored(dir)?;
    let mut listed = Vec::with_capacity(stored.len());
    for row in stored {
        let (state, detail) = classify_state(&inspector(&row.record));
        listed.push(ListedServer {
            record: row.record,
            state,
            detail,
        });
    }
    Ok(listed)
}

/// Stop matching servers. Stale pids and token mismatches are removed without a signal.
pub async fn stop_servers(
    dir: &Path,
    lock_path: &Path,
    selector: ServerSelector,
) -> Result<StopReport> {
    stop_with(
        dir,
        lock_path,
        selector,
        inspect::inspect_live,
        signal_pid,
        || tokio::time::sleep(std::time::Duration::from_millis(50)),
    )
    .await
}

/// Parse `server stop` id / `--port` into a selector.
pub fn selector_from_cli(id: Option<&str>, port: Option<u16>) -> Result<ServerSelector> {
    let id = id.map(str::trim).filter(|value| !value.is_empty());
    match (id, port) {
        (Some(id), Some(port)) => Ok(ServerSelector::Both {
            id: id.to_string(),
            port,
        }),
        (Some(id), None) => Ok(ServerSelector::Id(id.to_string())),
        (None, Some(port)) => Ok(ServerSelector::Port(port)),
        (None, None) => bail!("pass a server record id or --port"),
    }
}

/// Render `server list`.
#[must_use]
pub fn format_list(dir: &Path, rows: &[ListedServer], lock: Option<&ServerLock>) -> String {
    let mut out = format!("records: {}\n", dir.display());
    if rows.is_empty() {
        out.push_str("No Caliche server records.\n");
    } else {
        for row in rows {
            out.push_str(&format_row(row));
            out.push('\n');
        }
    }
    out.push_str(&format_lock(lock, rows));
    out.push('\n');
    out
}

/// Render `server status` blocks.
#[must_use]
pub fn format_status(dir: &Path, rows: &[ListedServer], lock: Option<&ServerLock>) -> String {
    let mut out = format!("records: {}\n", dir.display());
    if rows.is_empty() {
        out.push_str("No Caliche server records.\n");
    }
    for row in rows {
        out.push_str(&format_status_block(row));
        out.push('\n');
    }
    out.push_str(&format_lock(lock, rows));
    out.push('\n');
    out
}

struct PreparedStop {
    stored: StoredRecord,
    inspection: Inspection,
}

async fn stop_with<I, W, Fut>(
    dir: &Path,
    lock_path: &Path,
    selector: ServerSelector,
    inspector: I,
    mut signal: impl FnMut(u32, i32, &str) -> Result<()>,
    mut wait: W,
) -> Result<StopReport>
where
    I: Fn(&ServerRecord) -> Inspection,
    W: FnMut() -> Fut,
    Fut: Future<Output = ()>,
{
    let prepared = prepare_stop(load_stored(dir)?, &selector, &inspector)?;
    let mut report = StopReport::default();
    for row in &prepared {
        match &row.inspection {
            Inspection::Owned { token } => {
                apply_owned(
                    lock_path,
                    &row.stored,
                    token,
                    &inspector,
                    &mut signal,
                    &mut wait,
                    &mut report,
                )
                .await?;
            }
            inspection => {
                apply_unsignaled(lock_path, &row.stored, inspection, &mut report)?;
            }
        }
    }
    Ok(report)
}

fn prepare_stop(
    records: Vec<StoredRecord>,
    selector: &ServerSelector,
    inspector: &impl Fn(&ServerRecord) -> Inspection,
) -> Result<Vec<PreparedStop>> {
    let selected = select_stored(records, selector)?;
    let prepared: Vec<PreparedStop> = selected
        .into_iter()
        .map(|stored| {
            let inspection = inspector(&stored.record);
            PreparedStop { stored, inspection }
        })
        .collect();
    let owned = prepared
        .iter()
        .filter(|row| matches!(row.inspection, Inspection::Owned { .. }))
        .count();
    if owned > 1 {
        bail!(
            "multiple running Caliche servers match {selector}; pass a record id from `caliche server list`"
        );
    }
    Ok(prepared)
}

async fn apply_owned<I, W, Fut>(
    lock_path: &Path,
    stored: &StoredRecord,
    token: &str,
    inspector: &I,
    signal: &mut impl FnMut(u32, i32, &str) -> Result<()>,
    wait: &mut W,
    report: &mut StopReport,
) -> Result<()>
where
    I: Fn(&ServerRecord) -> Inspection,
    W: FnMut() -> Fut,
    Fut: Future<Output = ()>,
{
    if let Some(inspection) = recheck_other(inspector, &stored.record, token) {
        return apply_unsignaled(lock_path, stored, &inspection, report);
    }
    let pid = stored.record.pid;
    signal(pid, SIGTERM, token).with_context(|| format!("signal SIGTERM to pid {pid}"))?;
    let after = poll_until_released(stored, token, inspector, wait, TERM_POLLS).await;
    if matches!(after, Inspection::Stale) {
        return finish_stopped(lock_path, stored, report);
    }
    if !same_owner(&after, token) {
        report.failed = true;
        report.lines.push(format!(
            "refusing to send SIGKILL to {}: process no longer matches after SIGTERM",
            stored.record.id
        ));
        return Ok(());
    }
    if recheck_other(inspector, &stored.record, token).is_some() {
        report.failed = true;
        report.lines.push(format!(
            "refusing to send SIGKILL to {}: process identity changed",
            stored.record.id
        ));
        return Ok(());
    }
    signal(pid, SIGKILL, token).with_context(|| format!("signal SIGKILL to pid {pid}"))?;
    let after_kill = poll_until_released(stored, token, inspector, wait, KILL_POLLS).await;
    if matches!(after_kill, Inspection::Stale) {
        return finish_stopped(lock_path, stored, report);
    }
    report.failed = true;
    report.lines.push(format!(
        "pid {pid} did not exit after SIGKILL; record {} was left in place",
        stored.record.id
    ));
    Ok(())
}

fn apply_unsignaled(
    lock_path: &Path,
    stored: &StoredRecord,
    inspection: &Inspection,
    report: &mut StopReport,
) -> Result<()> {
    match inspection {
        Inspection::Stale => {
            remove_record_and_matching_lock(lock_path, stored)?;
            report.lines.push(format!(
                "removed stale server record {} (pid {} is not running)",
                stored.record.id, stored.record.pid
            ));
        }
        Inspection::Recycled { reason } => {
            remove_record_and_matching_lock(lock_path, stored)?;
            report.failed = true;
            report.lines.push(format!(
                "refusing to signal {}: {reason}; removed the record; the process was not signaled",
                stored.record.id
            ));
        }
        Inspection::Foreign { reason } => {
            report.failed = true;
            report
                .lines
                .push(format!("refusing to signal {}: {reason}", stored.record.id));
        }
        Inspection::Unproven { reason } => {
            report.failed = true;
            report
                .lines
                .push(format!("refusing to signal {}: {reason}", stored.record.id));
        }
        Inspection::Owned { .. } => {
            report.failed = true;
            report.lines.push(format!(
                "refusing to signal {}: ownership changed before stop",
                stored.record.id
            ));
        }
    }
    Ok(())
}

fn finish_stopped(lock_path: &Path, stored: &StoredRecord, report: &mut StopReport) -> Result<()> {
    remove_record_and_matching_lock(lock_path, stored)?;
    report.lines.push(format!(
        "stopped {} (pid {}, port {})",
        stored.record.id, stored.record.pid, stored.record.port
    ));
    Ok(())
}

fn recheck_other(
    inspector: &impl Fn(&ServerRecord) -> Inspection,
    record: &ServerRecord,
    token: &str,
) -> Option<Inspection> {
    let inspection = inspector(record);
    if same_owner(&inspection, token) {
        None
    } else {
        Some(inspection)
    }
}

fn same_owner(inspection: &Inspection, token: &str) -> bool {
    matches!(inspection, Inspection::Owned { token: live } if live == token)
}

async fn poll_until_released<I, W, Fut>(
    stored: &StoredRecord,
    token: &str,
    inspector: &I,
    wait: &mut W,
    attempts: u32,
) -> Inspection
where
    I: Fn(&ServerRecord) -> Inspection,
    W: FnMut() -> Fut,
    Fut: Future<Output = ()>,
{
    let mut seen = inspector(&stored.record);
    for _ in 0..attempts {
        if !same_owner(&seen, token) {
            return seen;
        }
        wait().await;
        seen = inspector(&stored.record);
    }
    seen
}

fn classify_state(inspection: &Inspection) -> (ServerState, String) {
    match inspection {
        Inspection::Owned { .. } => (
            ServerState::Running,
            "listening on the recorded port".to_string(),
        ),
        Inspection::Stale => (ServerState::Stale, "pid is not running".to_string()),
        Inspection::Foreign { reason }
        | Inspection::Unproven { reason }
        | Inspection::Recycled { reason } => (ServerState::Mismatch, reason.clone()),
    }
}

fn select_stored(
    records: Vec<StoredRecord>,
    selector: &ServerSelector,
) -> Result<Vec<StoredRecord>> {
    if let ServerSelector::Both { id, port } = selector
        && let Some(found) = records.iter().find(|row| row.record.id == *id)
        && found.record.port != *port
    {
        bail!(
            "server record {id} is port {}, not {port}",
            found.record.port
        );
    }
    let selected: Vec<StoredRecord> = records
        .into_iter()
        .filter(|row| selector.matches(&row.record))
        .collect();
    if selected.is_empty() {
        bail!("no Caliche server record matches {selector}");
    }
    Ok(selected)
}

fn load_stored(dir: &Path) -> Result<Vec<StoredRecord>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut stored = Vec::new();
    let entries = fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))?;
    for entry in entries {
        let entry = entry.with_context(|| format!("read {}", dir.display()))?;
        let path = entry.path();
        if !is_record_file(&path) {
            continue;
        }
        match read_stored(&path) {
            RecordRead::Loaded(row) => stored.push(row),
            RecordRead::Missing => {}
            RecordRead::Skipped(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "skipping unreadable server record"
                );
            }
            RecordRead::Corrupt(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "quarantining corrupt server record"
                );
                quarantine_record(&path);
            }
        }
    }
    stored.sort_by(|left, right| {
        left.record
            .port
            .cmp(&right.record.port)
            .then(left.record.started_at.cmp(&right.record.started_at))
            .then(left.record.id.cmp(&right.record.id))
    });
    Ok(stored)
}

/// What reading one record file found.
enum RecordRead {
    Loaded(StoredRecord),
    /// Removed between `read_dir` and the read.
    Missing,
    /// An I/O error or a version this build does not know. The file may
    /// belong to a live server (a sharing violation mid-rename, a newer
    /// build), so it stays in place.
    Skipped(anyhow::Error),
    /// Not a server record at all; safe to move aside.
    Corrupt(anyhow::Error),
}

/// Only the version, read before the rest of a record so a newer build's
/// fields or origin values never make the file look corrupt.
#[derive(Deserialize)]
struct VersionProbe {
    version: u32,
}

fn read_stored(path: &Path) -> RecordRead {
    let raw = match fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return RecordRead::Missing,
        Err(err) => {
            return RecordRead::Skipped(
                anyhow::Error::new(err).context(format!("read {}", path.display())),
            );
        }
    };
    let raw = strip_utf8_bom(&raw);
    let probe: VersionProbe = match serde_json::from_str(raw) {
        Ok(probe) => probe,
        Err(err) => {
            return RecordRead::Corrupt(
                anyhow::Error::new(err).context(format!("parse {}", path.display())),
            );
        }
    };
    if probe.version != RECORD_VERSION {
        return RecordRead::Skipped(anyhow::anyhow!(
            "server record {} has unsupported version {}",
            path.display(),
            probe.version
        ));
    }
    let record: ServerRecord = match serde_json::from_str(raw) {
        Ok(record) => record,
        Err(err) => {
            return RecordRead::Corrupt(
                anyhow::Error::new(err).context(format!("parse {}", path.display())),
            );
        }
    };
    RecordRead::Loaded(StoredRecord {
        path: path.to_path_buf(),
        record,
    })
}

fn quarantine_record(path: &Path) {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return;
    };
    let dest = path.with_file_name(format!("{name}.bad"));
    if dest.exists() {
        return;
    }
    if let Err(err) = fs::rename(path, &dest) {
        tracing::warn!(
            path = %path.display(),
            error = %err,
            "could not quarantine server record"
        );
    }
}

fn is_record_file(path: &Path) -> bool {
    path.extension().and_then(|ext| ext.to_str()) == Some("json")
}

fn strip_utf8_bom(raw: &str) -> &str {
    raw.strip_prefix('\u{feff}').unwrap_or(raw)
}

fn remove_records_matching(dir: &Path, pid: u32, only_if_dead: bool) -> Result<usize> {
    if only_if_dead && !pid_is_dead(pid) {
        return Ok(0);
    }
    let mut removed = 0;
    for stored in load_stored(dir)? {
        if stored.record.pid == pid {
            remove_path(&stored.path)?;
            removed += 1;
        }
    }
    Ok(removed)
}

fn pid_is_dead(pid: u32) -> bool {
    // A dead pid is Stale before executable or port checks run.
    matches!(
        inspect::inspect_live(&dead_probe_record(pid)),
        Inspection::Stale
    )
}

fn dead_probe_record(pid: u32) -> ServerRecord {
    ServerRecord {
        version: RECORD_VERSION,
        id: format!("probe-{pid}"),
        pid,
        port: 1,
        bind: "127.0.0.1".to_string(),
        build_id: String::new(),
        started_at: 0,
        cwd: String::new(),
        exe: format!("/usr/local/bin/{SERVER_EXE_NAME}"),
        origin: ServerOrigin::Foreground,
        start_token: "unset".to_string(),
    }
}

fn remove_record_and_matching_lock(lock_path: &Path, stored: &StoredRecord) -> Result<()> {
    remove_path(&stored.path)?;
    remove_lock_if_match(lock_path, &stored.record)
}

fn remove_lock_if_match(lock_path: &Path, record: &ServerRecord) -> Result<()> {
    let Some(lock) = ServerLock::read_from(lock_path)? else {
        return Ok(());
    };
    if lock.pid != record.pid || lock.port != record.port {
        return Ok(());
    }
    match fs::remove_file(lock_path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("remove {}", lock_path.display())),
    }
}

fn remove_path(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err).with_context(|| format!("remove {}", path.display())),
    }
}

fn record_file_name(id: &str) -> Result<String> {
    let safe = !id.is_empty()
        && id
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' || ch == '.');
    if !safe || id.contains("..") {
        bail!("invalid server record id");
    }
    Ok(format!("{id}.json"))
}

fn format_row(row: &ListedServer) -> String {
    format!(
        "{id}  {state}  {origin}  pid={pid}  bind={bind}  port={port}  build_id={build_id}  started={started}  cwd={cwd}  {detail}",
        id = row.record.id,
        state = row.state.as_str(),
        origin = row.record.origin.as_str(),
        pid = row.record.pid,
        bind = row.record.bind,
        port = row.record.port,
        build_id = row.record.build_id,
        started = format_started(row.record.started_at),
        cwd = row.record.cwd,
        detail = row.detail,
    )
}

fn format_status_block(row: &ListedServer) -> String {
    format!(
        "id: {id}\nstate: {state}\norigin: {origin}\npid: {pid}\nbind: {bind}\nport: {port}\nbuild_id: {build_id}\nstarted_at: {started}\ncwd: {cwd}\nexe: {exe}\ndetail: {detail}\n",
        id = row.record.id,
        state = row.state.as_str(),
        origin = row.record.origin.as_str(),
        pid = row.record.pid,
        bind = row.record.bind,
        port = row.record.port,
        build_id = row.record.build_id,
        started = format_started(row.record.started_at),
        cwd = row.record.cwd,
        exe = row.record.exe,
        detail = row.detail,
    )
}

fn format_lock(lock: Option<&ServerLock>, rows: &[ListedServer]) -> String {
    let Some(lock) = lock else {
        return "server.lock: absent".to_string();
    };
    let matched = rows
        .iter()
        .find(|row| row.record.pid == lock.pid && row.record.port == lock.port);
    match matched {
        Some(row) => format!(
            "server.lock: {} pid {} port {} record {}",
            lock.url, lock.pid, lock.port, row.record.id
        ),
        None => format!(
            "server.lock: {} pid {} port {} (no matching record)",
            lock.url, lock.pid, lock.port
        ),
    }
}

fn format_started(started_at: u64) -> String {
    let Ok(seconds) = i64::try_from(started_at) else {
        return started_at.to_string();
    };
    chrono::DateTime::from_timestamp(seconds, 0)
        .map(|stamp| stamp.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_else(|| started_at.to_string())
}

#[cfg(unix)]
pub(super) const SIGTERM: i32 = libc::SIGTERM;
#[cfg(unix)]
pub(super) const SIGKILL: i32 = libc::SIGKILL;
#[cfg(not(unix))]
pub(super) const SIGTERM: i32 = 15;
#[cfg(not(unix))]
pub(super) const SIGKILL: i32 = 9;

fn signal_pid(pid: u32, sig: i32, token: &str) -> Result<()> {
    if sig != SIGTERM && sig != SIGKILL {
        bail!("refusing to send signal {sig}");
    }
    if pid < 2 {
        bail!("refusing to signal pid {pid}");
    }
    signal_pid_os(pid, sig, token)
}

#[cfg(unix)]
fn signal_pid_os(pid: u32, sig: i32, token: &str) -> Result<()> {
    let _ = token;
    // The signal is only SIGTERM or SIGKILL, and only after the registry
    // ownership checks have passed.
    crate::process_liveness::signal_pid(pid, sig)
        .map_err(|err| anyhow::anyhow!("signal {sig} to pid {pid} failed: {err}"))
}

#[cfg(target_os = "windows")]
fn signal_pid_os(pid: u32, sig: i32, token: &str) -> Result<()> {
    let _ = sig;
    // Windows has no SIGTERM: both stops are a hard TerminateProcess, so the
    // server does not drain or unregister itself; stop removes its record.
    // Each re-opens the pid and compares its creation time first. The poll
    // after the first call skips the second when the process has exited.
    crate::process_liveness::terminate_if_same_birth(pid, token).map_err(anyhow::Error::msg)
}

#[cfg(all(not(unix), not(target_os = "windows")))]
fn signal_pid_os(pid: u32, sig: i32, token: &str) -> Result<()> {
    let _ = (pid, sig, token);
    bail!("signaling servers is unsupported on this platform");
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> Result<()> {
    let file = fs::File::open(dir).with_context(|| format!("fsync {}", dir.display()))?;
    file.sync_all()
        .with_context(|| format!("fsync {}", dir.display()))?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_dir(dir: &Path) -> Result<()> {
    let _ = dir;
    Ok(())
}

#[cfg(unix)]
fn restrict_dir(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .with_context(|| format!("restrict {}", path.display()))
}

#[cfg(not(unix))]
fn restrict_dir(path: &Path) -> Result<()> {
    let _ = path;
    Ok(())
}

#[cfg(unix)]
fn restrict_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .with_context(|| format!("restrict {}", path.display()))
}

#[cfg(not(unix))]
fn restrict_file(path: &Path) -> Result<()> {
    let _ = path;
    Ok(())
}

#[cfg(test)]
mod tests;
