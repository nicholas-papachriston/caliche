//! Durable ledger of shell and subagent tasks.
//!
//! Every tracked task gets a row in `tasks.db` and an append-only event log.
//! Callers use a synchronous API: [`begin`] returns a [`TaskGuard`], and
//! [`TaskGuard::finish`] records the terminal state. A guard dropped without
//! `finish` (for example, when a cancelled turn drops its future) records
//! `cancelled`. One writer task batches the writes.
//!
//! The queue to the writer is unbounded so a start or end is never dropped
//! for lack of room; producers are bounded by live tasks, and a start waits
//! for its commit before the work runs. A write the writer cannot commit
//! stays queued in order and is retried. [`shutdown`] commits what is queued;
//! writes the database still refuses are saved next to it and committed when
//! the ledger next opens.
//!
//! With no ledger installed (library users, tests, builds without
//! `sqlite-store`), every call does nothing.

use std::future::Future;
use std::sync::OnceLock;

use chrono::{DateTime, SecondsFormat, Utc};
use tokio::sync::{mpsc, oneshot};

use crate::agent::session::TranscriptRedactor;

/// File name of the ledger inside the data directory.
pub const TASKS_DB_FILE: &str = "tasks.db";

static LEDGER_TX: OnceLock<mpsc::UnboundedSender<Op>> = OnceLock::new();

/// Commit result sent back to a waiting caller.
type Ack = oneshot::Sender<Result<(), String>>;

/// Secret-like values removed from task names and end reasons before they
/// are stored. Extends the transcript defaults (`sk-` keys, Bearer tokens).
/// Order matters: a whole `Authorization:` header goes before key=value.
const SECRET_PATTERNS: &[&str] = &[
    r"sk-[A-Za-z0-9_-]{16,}",
    r"(?i)\bauthorization\s*:\s*\S+(?:\s+[A-Za-z0-9._~+/=-]+)?",
    r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+",
    r"(?i)[A-Za-z0-9_.-]*(?:key|token|secret|passw(?:or)?d)[A-Za-z0-9_.-]*\s*[=:]\s*\S+",
    r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    r"\bxox[abposr]-[A-Za-z0-9-]{10,}",
    r"\bAKIA[0-9A-Z]{16}\b",
    // URL userinfo: `user:password@` in `scheme://user:password@host`.
    r"[^\s/:@]+:[^\s/@]+@",
];

/// Redact secret-like values. If the patterns fail to compile, the text is
/// withheld rather than stored as is.
fn redact(text: &str) -> String {
    static REDACTOR: OnceLock<Option<TranscriptRedactor>> = OnceLock::new();
    let redactor = REDACTOR
        .get_or_init(|| TranscriptRedactor::from_patterns(SECRET_PATTERNS.iter().copied()).ok());
    match redactor {
        Some(redactor) => redactor.redact_text(text).text,
        None => "[withheld: redactor unavailable]".to_owned(),
    }
}

tokio::task_local! {
    static CURRENT_TASK: String;
}

/// What produced a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Shell,
    Subagent,
}

impl TaskKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shell => "shell",
            Self::Subagent => "subagent",
        }
    }
}

/// Lifecycle state of a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Running,
    Succeeded,
    Failed,
    Cancelled,
    /// The owning process went away without recording an end state.
    HostLost,
    /// The owning process went away, but the task's child process group was
    /// still alive when the ledger noticed. The child is left running.
    Orphaned,
}

impl TaskState {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::HostLost => "host_lost",
            Self::Orphaned => "orphaned",
        }
    }

    /// Whether the owner may end a task in this state with
    /// [`TaskGuard::finish`]. `host_lost` and `orphaned` are set only by
    /// reconcile, for owners that are gone.
    #[must_use]
    pub const fn owner_can_finish_as(self) -> bool {
        match self {
            Self::Succeeded | Self::Failed | Self::Cancelled => true,
            Self::Running | Self::HostLost | Self::Orphaned => false,
        }
    }
}

/// Whether a task may move from `from` to `to`. Terminal states are final.
#[must_use]
pub const fn can_transition(from: TaskState, to: TaskState) -> bool {
    match (from, to) {
        (
            TaskState::Running,
            TaskState::Succeeded
            | TaskState::Failed
            | TaskState::Cancelled
            | TaskState::HostLost
            | TaskState::Orphaned,
        ) => true,
        (
            TaskState::Running
            | TaskState::Succeeded
            | TaskState::Failed
            | TaskState::Cancelled
            | TaskState::HostLost
            | TaskState::Orphaned,
            _,
        ) => false,
    }
}

/// A write for the writer. Data ops serialize, so the writer can save them
/// at shutdown when the database refuses them.
#[derive(Debug)]
#[cfg_attr(
    not(feature = "sqlite-store"),
    expect(dead_code, reason = "fields are read by the sqlite-store writer")
)]
#[cfg_attr(
    feature = "sqlite-store",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "snake_case")
)]
enum Op {
    Begin {
        id: String,
        parent_id: Option<String>,
        kind: TaskKind,
        name: String,
        ts: String,
        #[cfg_attr(feature = "sqlite-store", serde(skip))]
        ack: Option<Ack>,
    },
    /// The task's child process started.
    Spawned {
        id: String,
        pid: Option<u32>,
        pgid: Option<u32>,
        ts: String,
    },
    Finish {
        id: String,
        state: TaskState,
        reason: Option<String>,
        ts: String,
    },
    #[cfg_attr(feature = "sqlite-store", serde(skip))]
    Flush(Option<Ack>),
    /// Commit or save every queued write, then stop the writer.
    #[cfg_attr(feature = "sqlite-store", serde(skip))]
    Shutdown(Ack),
}

/// Open task handle. Drop without [`finish`](Self::finish) records `cancelled`.
#[derive(Debug)]
#[must_use = "dropping the guard records the task as cancelled"]
pub struct TaskGuard {
    inner: Option<GuardInner>,
}

#[derive(Debug)]
struct GuardInner {
    id: String,
    tx: mpsc::UnboundedSender<Op>,
    ack: Option<oneshot::Receiver<Result<(), String>>>,
}

impl GuardInner {
    fn send_finish(self, state: TaskState, reason: Option<&str>) {
        send(
            &self.tx,
            Op::Finish {
                id: self.id,
                state,
                reason: reason.map(redact),
                ts: now(),
            },
        );
    }
}

impl TaskGuard {
    /// Ledger id, or `None` when no ledger is installed.
    #[must_use]
    pub fn id(&self) -> Option<&str> {
        self.inner.as_ref().map(|inner| inner.id.as_str())
    }

    /// Resolves once the start event is committed. An error means the start
    /// is not durable yet (it stays queued for retry); callers log it and
    /// proceed. Resolves `Ok` with no ledger installed.
    ///
    /// The returned future owns its state, so a caller can take it out from
    /// under a lock before awaiting.
    pub fn recorded(&mut self) -> impl Future<Output = anyhow::Result<()>> + Send + use<> {
        let ack = self.inner.as_mut().and_then(|inner| inner.ack.take());
        async move {
            let Some(ack) = ack else {
                return Ok(());
            };
            match ack.await {
                Ok(result) => result.map_err(anyhow::Error::msg),
                Err(_) => Err(anyhow::anyhow!("task ledger writer stopped")),
            }
        }
    }

    /// Record the child process (and its process group) running this task,
    /// so reconcile can tell an orphaned child from a lost task.
    pub fn set_process(&self, pid: Option<u32>, pgid: Option<u32>) {
        if let Some(inner) = &self.inner {
            send(
                &inner.tx,
                Op::Spawned {
                    id: inner.id.clone(),
                    pid,
                    pgid,
                    ts: now(),
                },
            );
        }
    }

    /// Record the end state. Only states the owner may set are accepted
    /// (see [`TaskState::owner_can_finish_as`]); anything else is rejected
    /// and the guard falls back to `cancelled` on drop.
    pub fn finish(mut self, state: TaskState, reason: Option<&str>) {
        if !state.owner_can_finish_as() {
            tracing::warn!(
                state = state.as_str(),
                "task ledger: owner cannot finish as this state"
            );
            return;
        }
        if let Some(inner) = self.inner.take() {
            inner.send_finish(state, reason);
        }
    }
}

impl Drop for TaskGuard {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            inner.send_finish(TaskState::Cancelled, Some("dropped"));
        }
    }
}

#[cfg(test)]
tokio::task_local! {
    /// Test-only ledger for the current task, so unit tests drive the real
    /// `begin` wiring into a temp ledger without the process-global one.
    static TEST_LEDGER: mpsc::UnboundedSender<Op>;
}

/// The ledger [`begin`] and [`flush`] write to: in tests a task-scoped
/// ledger first, then the process ledger.
fn installed() -> Option<mpsc::UnboundedSender<Op>> {
    #[cfg(test)]
    if let Ok(tx) = TEST_LEDGER.try_with(Clone::clone) {
        return Some(tx);
    }
    LEDGER_TX.get().cloned()
}

/// Start a task in the installed ledger. The parent is the task of the
/// enclosing [`scope`], when there is one.
pub fn begin(kind: TaskKind, name: impl Into<String>) -> TaskGuard {
    match installed() {
        Some(tx) => begin_on(&tx, kind, name.into()),
        None => TaskGuard { inner: None },
    }
}

fn begin_on(tx: &mpsc::UnboundedSender<Op>, kind: TaskKind, name: String) -> TaskGuard {
    let id = uuid::Uuid::now_v7().to_string();
    let parent_id = CURRENT_TASK.try_with(Clone::clone).ok();
    let (ack_tx, ack_rx) = oneshot::channel();
    send(
        tx,
        Op::Begin {
            id: id.clone(),
            parent_id,
            kind,
            name: redact(&name),
            ts: now(),
            ack: Some(ack_tx),
        },
    );
    TaskGuard {
        inner: Some(GuardInner {
            id,
            tx: tx.clone(),
            ack: Some(ack_rx),
        }),
    }
}

/// Run `fut` with `task_id` as the parent of tasks begun inside it.
///
/// Best effort: the link does not cross `tokio::spawn`.
pub async fn scope<F: Future>(task_id: Option<String>, fut: F) -> F::Output {
    match task_id {
        Some(id) => CURRENT_TASK.scope(id, fut).await,
        None => fut.await,
    }
}

/// Stop the installed ledger. Call once on graceful exit.
///
/// Waits until every write queued so far is committed. Writes the database
/// refuses are saved next to it and committed when the ledger next opens.
/// Fails only when they could not be saved. Writes sent after this are lost.
pub async fn shutdown() -> anyhow::Result<()> {
    match installed() {
        Some(tx) => request(&tx, Op::Shutdown).await,
        None => Ok(()),
    }
}

/// Send the op built around a fresh ack and wait for the writer's answer.
async fn request(tx: &mpsc::UnboundedSender<Op>, op: impl FnOnce(Ack) -> Op) -> anyhow::Result<()> {
    let (ack_tx, ack_rx) = oneshot::channel();
    tx.send(op(ack_tx))
        .map_err(|_| anyhow::anyhow!("task ledger writer stopped"))?;
    match ack_rx.await {
        Ok(result) => result.map_err(anyhow::Error::msg),
        Err(_) => Err(anyhow::anyhow!("task ledger writer stopped")),
    }
}

/// Queue `op`. The queue is unbounded, so this only fails once the writer
/// has stopped, which is logged as an error.
fn send(tx: &mpsc::UnboundedSender<Op>, op: Op) {
    if tx.send(op).is_err() {
        tracing::error!("task ledger: writer stopped; write lost");
    }
}

/// Fixed-width UTC timestamp, so stored values compare correctly as text.
fn timestamp(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Micros, true)
}

fn now() -> String {
    timestamp(Utc::now())
}

#[cfg(all(test, feature = "sqlite-store"))]
pub(crate) use store::test_support;
#[cfg(feature = "sqlite-store")]
pub use store::{Retention, TaskLedger, install};

#[cfg(feature = "sqlite-store")]
mod store {
    use super::{
        LEDGER_TX, Op, TASKS_DB_FILE, TaskGuard, TaskKind, TaskState, begin_on, now, timestamp,
    };
    #[cfg(unix)]
    use crate::security::SystemBinary;
    use anyhow::{Context, Result};
    use chrono::Utc;
    use serde_json::json;
    use sqlx::SqliteConnection;
    use sqlx::sqlite::{SqliteConnectOptions, SqliteJournalMode, SqlitePool, SqliteSynchronous};
    use std::collections::VecDeque;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::process::Stdio;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, OnceLock};
    use std::time::{Duration, Instant};
    use tokio::sync::mpsc;

    const WRITE_BATCH_MAX: usize = 64;
    /// Tries per op once a batch has failed, with doubling backoff.
    const RETRY_ATTEMPTS: u32 = 4;
    const RETRY_BACKOFF: Duration = Duration::from_millis(25);
    /// How often queued writes are retried while the queue is idle.
    const RETRY_INTERVAL: Duration = Duration::from_secs(1);

    static INSTANCE_ID: OnceLock<String> = OnceLock::new();

    /// Who may change a task: the host, pid, and per-process instance id
    /// that started it. The instance id tells a reused pid from its owner.
    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    struct Owner {
        host: String,
        pid: i64,
        instance: String,
    }

    impl Owner {
        async fn current() -> Self {
            Self {
                host: host_name().await,
                pid: i64::from(std::process::id()),
                instance: INSTANCE_ID
                    .get_or_init(|| uuid::Uuid::now_v7().to_string())
                    .clone(),
            }
        }
    }

    /// SQLite-backed task ledger with one batching writer task.
    #[derive(Debug)]
    pub struct TaskLedger {
        tx: mpsc::UnboundedSender<Op>,
        #[cfg(test)]
        faults: Arc<AtomicU32>,
    }

    /// Open `<data_dir>/tasks.db` and install it as the process ledger.
    pub async fn install(data_dir: &Path) -> Result<()> {
        TaskLedger::open(&data_dir.join(TASKS_DB_FILE))
            .await?
            .install()
    }

    /// How long ended tasks and their events stay in the ledger. Pruning runs
    /// when the ledger opens and then from the writer, after
    /// `prune_every_writes` committed writes or once `prune_interval` has
    /// passed. Running tasks are never pruned.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Retention {
        /// Ended tasks last updated before this age are deleted.
        pub max_age: chrono::Duration,
        /// At most this many ended tasks are kept; the oldest go first.
        pub max_ended_tasks: u32,
        /// Prune after this many committed writes.
        pub prune_every_writes: u32,
        /// Prune when this long has passed since the last prune.
        pub prune_interval: Duration,
    }

    impl Default for Retention {
        fn default() -> Self {
            Self {
                max_age: chrono::Duration::days(7),
                max_ended_tasks: 10_000,
                prune_every_writes: 1_000,
                prune_interval: Duration::from_secs(60 * 60),
            }
        }
    }

    impl TaskLedger {
        /// [`Self::open_with`] using the default [`Retention`].
        pub async fn open(path: &Path) -> Result<Self> {
            Self::open_with(path, Retention::default()).await
        }

        /// Open the ledger, create the schema, mark lost tasks, prune ended
        /// tasks past `retention`, and start the writer.
        pub async fn open_with(path: &Path, retention: Retention) -> Result<Self> {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .context("Failed to create task ledger directory")?;
            }
            restrict_to_owner(path);
            // tasks.db alone uses synchronous=FULL: a power loss must not drop
            // a committed start or end event, and ledger writes are small.
            let options = SqliteConnectOptions::new()
                .filename(path)
                .create_if_missing(true)
                .journal_mode(SqliteJournalMode::Wal)
                .busy_timeout(Duration::from_secs(5))
                .synchronous(SqliteSynchronous::Full);
            let pool = crate::data::persistence::sqlite_pool::local_sqlite_pool()
                .connect_with(options)
                .await
                .context("Failed to open task ledger")?;
            migrate(&pool).await?;
            // Before reconcile, so a saved end is not taken for a lost task.
            replay_saved(&pool, path).await?;
            let owner = Owner::current().await;
            reconcile(&pool, &owner).await?;
            prune(&pool, retention).await?;
            let (tx, rx) = mpsc::unbounded_channel();
            let writer = Writer {
                pool,
                db_path: path.to_path_buf(),
                owner,
                pending: VecDeque::new(),
                faults: Arc::new(AtomicU32::new(0)),
                retention,
                writes_since_prune: 0,
                last_prune: Instant::now(),
            };
            #[cfg(test)]
            let faults = Arc::clone(&writer.faults);
            tokio::spawn(writer.run(rx));
            Ok(Self {
                tx,
                #[cfg(test)]
                faults,
            })
        }

        /// Make this the ledger used by [`super::begin`]. Fails if one is installed.
        pub fn install(self) -> Result<()> {
            LEDGER_TX
                .set(self.tx)
                .map_err(|_| anyhow::anyhow!("task ledger already installed"))
        }

        /// Start a task in this ledger.
        pub fn begin(&self, kind: TaskKind, name: impl Into<String>) -> TaskGuard {
            begin_on(&self.tx, kind, name.into())
        }

        /// Wait until every write queued so far is committed. Fails when some
        /// writes are still queued for retry.
        pub async fn flush(&self) -> Result<()> {
            super::request(&self.tx, |ack| Op::Flush(Some(ack))).await
        }

        /// Stop this ledger, as [`super::shutdown`] stops the installed one.
        pub async fn shutdown(self) -> Result<()> {
            super::request(&self.tx, Op::Shutdown).await
        }

        /// Make the next `count` write transactions fail.
        #[cfg(test)]
        fn inject_write_faults(&self, count: u32) {
            self.faults.store(count, Ordering::SeqCst);
        }
    }

    /// Make the ledger owner-only (0600), as `caliche.db` is: create the
    /// file with that mode before SQLite opens it (SQLite gives `-wal` and
    /// `-shm` the database file's mode) and tighten files left by older runs.
    /// Best effort; a failure is logged.
    #[cfg(unix)]
    fn restrict_to_owner(path: &Path) {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let created = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .mode(0o600)
            .open(path);
        if let Err(error) = created {
            tracing::warn!(%error, "task ledger: could not create tasks.db as 0600");
        }
        for suffix in ["", "-wal", "-shm"] {
            let mut file = path.as_os_str().to_owned();
            file.push(suffix);
            let file = std::path::PathBuf::from(file);
            if file.exists()
                && let Err(error) =
                    std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600))
            {
                tracing::warn!(%error, file = %file.display(), "task ledger: could not set 0600");
            }
        }
    }

    #[cfg(not(unix))]
    fn restrict_to_owner(_path: &Path) {}

    /// Schema version this build writes, kept in `PRAGMA user_version`.
    const SCHEMA_VERSION: i64 = 1;

    /// Bring the schema to [`SCHEMA_VERSION`] in one write transaction
    /// (`BEGIN IMMEDIATE`, so two processes opening at once take turns).
    /// Version 0 is a new file or one written before versioning and is
    /// upgraded in place. Any other version is refused, so an older build
    /// never writes a newer schema.
    async fn migrate(pool: &SqlitePool) -> Result<()> {
        let mut tx = pool.begin_with("BEGIN IMMEDIATE").await?;
        let version: i64 = sqlx::query_scalar("PRAGMA user_version")
            .fetch_one(&mut *tx)
            .await?;
        match version {
            0 => {
                create_v1(&mut tx).await?;
                sqlx::query(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))
                    .execute(&mut *tx)
                    .await?;
            }
            SCHEMA_VERSION => {}
            other => anyhow::bail!(
                "tasks.db schema version {other} is not supported by this build (expects {SCHEMA_VERSION})"
            ),
        }
        tx.commit()
            .await
            .context("Failed to migrate task ledger schema")
    }

    /// Columns added after the first unversioned files were written.
    const LATE_COLUMNS: [(&str, &str); 3] = [
        ("owner_instance", "TEXT NOT NULL DEFAULT ''"),
        ("child_pid", "INTEGER"),
        ("child_pgid", "INTEGER"),
    ];

    /// Version 1: create the tables, and add columns an unversioned file
    /// may lack.
    async fn create_v1(conn: &mut SqliteConnection) -> Result<()> {
        create_tables(conn).await?;
        let columns: Vec<String> =
            sqlx::query_scalar("SELECT name FROM pragma_table_info('tasks')")
                .fetch_all(&mut *conn)
                .await?;
        for (column, definition) in LATE_COLUMNS {
            if !columns.iter().any(|name| name == column) {
                sqlx::query(&format!(
                    "ALTER TABLE tasks ADD COLUMN {column} {definition}"
                ))
                .execute(&mut *conn)
                .await
                .with_context(|| format!("Failed to add tasks.{column}"))?;
            }
        }
        Ok(())
    }

    async fn create_tables(conn: &mut SqliteConnection) -> Result<()> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                kind TEXT NOT NULL,
                name TEXT NOT NULL,
                state TEXT NOT NULL,
                owner_pid INTEGER NOT NULL,
                owner_instance TEXT NOT NULL,
                host TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                terminal_reason TEXT,
                child_pid INTEGER,
                child_pgid INTEGER
            )
            "#,
        )
        .execute(&mut *conn)
        .await
        .context("Failed to create tasks table")?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_state ON tasks(state, host)")
            .execute(&mut *conn)
            .await
            .context("Failed to create tasks state index")?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_tasks_updated ON tasks(updated_at)")
            .execute(&mut *conn)
            .await
            .context("Failed to create tasks updated_at index")?;

        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS task_events (
                task_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                ts TEXT NOT NULL,
                type TEXT NOT NULL,
                payload TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (task_id, seq)
            )
            "#,
        )
        .execute(&mut *conn)
        .await
        .context("Failed to create task_events table")?;
        Ok(())
    }

    #[derive(sqlx::FromRow)]
    struct RunningRow {
        id: String,
        owner_pid: i64,
        owner_instance: String,
        child_pid: Option<i64>,
        child_pgid: Option<i64>,
    }

    /// End `running` rows on this host whose owner is gone.
    ///
    /// An owner is gone when its pid is dead, or when its pid is ours but its
    /// instance is not (the pid was reused). Other hosts are never touched.
    /// A task whose recorded child process group is still alive becomes
    /// `orphaned` (the child keeps running; killing it is a later slice);
    /// otherwise it becomes `host_lost`.
    async fn reconcile(pool: &SqlitePool, me: &Owner) -> Result<()> {
        let rows: Vec<RunningRow> = sqlx::query_as(
            "SELECT id, owner_pid, owner_instance, child_pid, child_pgid FROM tasks \
             WHERE state = 'running' AND host = ?",
        )
        .bind(&me.host)
        .fetch_all(pool)
        .await
        .context("Failed to read running tasks")?;
        let mut ended = Vec::new();
        for row in rows {
            if let Some(state) = gone_owner_state(&row, me).await {
                ended.push((row, state));
            }
        }
        if ended.is_empty() {
            return Ok(());
        }
        let ts = now();
        let mut tx = pool.begin().await?;
        for (row, state) in &ended {
            end_gone_owner_task(&mut tx, me, row, *state, &ts).await?;
        }
        tx.commit()
            .await
            .context("Failed to end tasks of gone owners")?;
        tracing::info!(
            count = ended.len(),
            "task ledger: ended tasks of gone owners"
        );
        Ok(())
    }

    /// `None` while the owner lives; otherwise `orphaned` when the child
    /// still runs and `host_lost` when it does not.
    async fn gone_owner_state(row: &RunningRow, me: &Owner) -> Option<TaskState> {
        let gone = if row.owner_pid == me.pid {
            row.owner_instance != me.instance
        } else {
            !pid_alive(row.owner_pid).await
        };
        if !gone {
            return None;
        }
        let child_alive = match (row.child_pgid, row.child_pid) {
            (Some(pgid), _) => group_alive(pgid).await,
            (None, Some(pid)) => pid_alive(pid).await,
            (None, None) => false,
        };
        Some(if child_alive {
            TaskState::Orphaned
        } else {
            TaskState::HostLost
        })
    }

    async fn end_gone_owner_task(
        conn: &mut SqliteConnection,
        me: &Owner,
        row: &RunningRow,
        state: TaskState,
        ts: &str,
    ) -> Result<()> {
        let owner = Owner {
            host: me.host.clone(),
            pid: row.owner_pid,
            instance: row.owner_instance.clone(),
        };
        let payload = json!({
            "owner_pid": row.owner_pid,
            "owner_instance": row.owner_instance,
            "child_pid": row.child_pid,
            "child_pgid": row.child_pgid,
        });
        let reason = match state {
            TaskState::Orphaned => "owner process exited; child process still running",
            TaskState::Running
            | TaskState::Succeeded
            | TaskState::Failed
            | TaskState::Cancelled
            | TaskState::HostLost => "owner process exited without an end state",
        };
        let end = End {
            state,
            reason: Some(reason),
            ts,
        };
        end_task(conn, &row.id, &owner, &end, &payload.to_string()).await
    }

    /// Delete ended tasks older than `retention.max_age` or beyond the newest
    /// `retention.max_ended_tasks`, with their events, in one transaction.
    async fn prune(pool: &SqlitePool, retention: Retention) -> Result<()> {
        const DOOMED: &str = "SELECT id FROM tasks WHERE state != 'running' AND (updated_at < ?1 \
             OR id IN (SELECT id FROM tasks WHERE state != 'running' \
             ORDER BY updated_at DESC, id DESC LIMIT -1 OFFSET ?2))";
        let cutoff = timestamp(Utc::now() - retention.max_age);
        let mut tx = pool.begin().await?;
        // Events go first, while the doomed tasks still identify them.
        sqlx::query(&format!(
            "DELETE FROM task_events WHERE task_id IN ({DOOMED})"
        ))
        .bind(&cutoff)
        .bind(retention.max_ended_tasks)
        .execute(&mut *tx)
        .await?;
        let pruned = sqlx::query(&format!("DELETE FROM tasks WHERE id IN ({DOOMED})"))
            .bind(&cutoff)
            .bind(retention.max_ended_tasks)
            .execute(&mut *tx)
            .await?
            .rows_affected();
        tx.commit().await.context("Failed to prune task ledger")?;
        if pruned > 0 {
            tracing::info!(count = pruned, "task ledger: pruned ended tasks");
        }
        Ok(())
    }

    /// Writes the database refused at shutdown, as saved next to it.
    #[derive(serde::Serialize)]
    struct SavedRef<'a> {
        owner: &'a Owner,
        ops: Vec<&'a Op>,
    }

    #[derive(serde::Deserialize)]
    struct Saved {
        owner: Owner,
        ops: Vec<Op>,
    }

    /// Saved files are `<db file>.pending-<uuid>.json` beside the database.
    const SAVED_INFIX: &str = ".pending-";
    const SAVED_SUFFIX: &str = ".json";
    /// [`save`] writes `<saved file>.tmp` first.
    const TMP_SUFFIX: &str = ".json.tmp";
    /// A temporary file this old is left by a crash, not by another process
    /// sharing the ledger that is saving right now.
    const STALE_TMP_AGE: Duration = Duration::from_secs(60);

    fn db_file_name(db_path: &Path) -> String {
        db_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| TASKS_DB_FILE.to_owned())
    }

    fn db_dir(db_path: &Path) -> &Path {
        match db_path.parent() {
            Some(dir) if !dir.as_os_str().is_empty() => dir,
            _ => Path::new("."),
        }
    }

    /// Save `ops` owner-only in a new file beside the database. The file
    /// appears complete or not at all: it is written and synced under a
    /// temporary name, then renamed.
    fn save(db_path: &Path, owner: &Owner, ops: &VecDeque<Op>) -> Result<PathBuf> {
        let ops: Vec<&Op> = ops
            .iter()
            .filter(|op| !matches!(op, Op::Flush(_) | Op::Shutdown(_)))
            .collect();
        let body = serde_json::to_vec(&SavedRef { owner, ops })?;
        let dir = db_dir(db_path);
        let name = format!(
            "{}{SAVED_INFIX}{}{SAVED_SUFFIX}",
            db_file_name(db_path),
            uuid::Uuid::now_v7()
        );
        let path = dir.join(&name);
        let tmp = dir.join(format!("{name}.tmp"));
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
        let mut file = options.open(&tmp)?;
        file.write_all(&body)?;
        file.sync_all()?;
        std::fs::rename(&tmp, &path)?;
        // Syncing the directory makes the rename durable on Unix. Windows
        // skips it: `sync_all` there is FlushFileBuffers, which needs a
        // writable handle that std cannot open on a directory, and NTFS
        // journals the rename itself.
        #[cfg(unix)]
        std::fs::File::open(dir)?.sync_all()?;
        Ok(path)
    }

    /// Saved files beside the database ending in `suffix`, oldest first.
    fn saved_files(db_path: &Path, suffix: &str) -> Result<Vec<PathBuf>> {
        let prefix = format!("{}{SAVED_INFIX}", db_file_name(db_path));
        let mut files = Vec::new();
        for entry in std::fs::read_dir(db_dir(db_path))? {
            let path = entry?.path();
            let name = path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default();
            if name.starts_with(&prefix) && name.ends_with(suffix) {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }

    /// Commit the writes earlier runs saved at shutdown, each file in one
    /// transaction, then delete the file. Every op is a no-op once applied,
    /// so a file replayed again (a crash before the delete) adds nothing.
    ///
    /// A file that cannot be read is set aside rather than failing the
    /// open, so one bad file does not stop the ledger recording later tasks.
    async fn replay_saved(pool: &SqlitePool, db_path: &Path) -> Result<()> {
        remove_stale_tmp(db_path)?;
        for file in
            saved_files(db_path, SAVED_SUFFIX).context("Failed to list saved task writes")?
        {
            let saved: Saved = match std::fs::read(&file) {
                // Another process opening the same ledger replayed it first.
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    set_aside(&file, &error);
                    continue;
                }
                Ok(bytes) => match serde_json::from_slice(&bytes) {
                    Ok(saved) => saved,
                    Err(error) => {
                        set_aside(&file, &error);
                        continue;
                    }
                },
            };
            let mut tx = pool.begin().await?;
            for op in &saved.ops {
                write_op(&mut tx, &saved.owner, op).await?;
            }
            tx.commit()
                .await
                .context("Failed to commit saved task writes")?;
            match std::fs::remove_file(&file) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                    return Err(error).context("Failed to remove replayed task writes");
                }
                _ => {}
            }
            tracing::info!(
                count = saved.ops.len(),
                file = %file.display(),
                "task ledger: committed writes saved at shutdown"
            );
        }
        Ok(())
    }

    /// Rename an unreadable saved file to `<file>.corrupt-<utc time>`, which
    /// replay skips, and log one error. Its writes are lost; the file is
    /// kept for inspection.
    fn set_aside(file: &Path, error: &dyn std::fmt::Display) {
        let mut aside = file.as_os_str().to_owned();
        aside.push(format!(
            ".corrupt-{}",
            Utc::now().format("%Y%m%dT%H%M%S%.6fZ")
        ));
        let aside = PathBuf::from(aside);
        match std::fs::rename(file, &aside) {
            Ok(()) => tracing::error!(
                %error,
                file = %aside.display(),
                "task ledger: saved writes unreadable; set aside, not committed"
            ),
            Err(rename_error) => tracing::error!(
                %error,
                %rename_error,
                file = %file.display(),
                "task ledger: saved writes unreadable and not set aside; skipped"
            ),
        }
    }

    /// Delete temporary files a crash left behind mid-[`save`].
    fn remove_stale_tmp(db_path: &Path) -> Result<()> {
        for file in saved_files(db_path, TMP_SUFFIX).context("Failed to list saved task writes")? {
            let stale = std::fs::metadata(&file)
                .and_then(|meta| meta.modified())
                .is_ok_and(|at| at.elapsed().is_ok_and(|age| age >= STALE_TMP_AGE));
            if stale && let Err(error) = std::fs::remove_file(&file) {
                tracing::warn!(%error, file = %file.display(), "task ledger: could not remove a stale temporary file");
            }
        }
        Ok(())
    }

    impl Op {
        /// Answer a waiting caller once. Later calls do nothing.
        fn ack(&mut self, result: Result<(), String>) {
            match self {
                Op::Begin { ack, .. } | Op::Flush(ack) => {
                    if let Some(ack) = ack.take() {
                        let _ = ack.send(result);
                    }
                }
                Op::Spawned { .. } | Op::Finish { .. } | Op::Shutdown(_) => {}
            }
        }
    }

    /// The single writer. Ops it cannot commit stay at the front of
    /// `pending`, in order, and are retried; nothing is dropped on a
    /// transient fault (busy, disk full, IO).
    struct Writer {
        pool: SqlitePool,
        db_path: PathBuf,
        owner: Owner,
        pending: VecDeque<Op>,
        /// Test hook: the next N write transactions fail.
        faults: Arc<AtomicU32>,
        retention: Retention,
        writes_since_prune: u64,
        last_prune: Instant,
    }

    impl Writer {
        async fn run(mut self, mut rx: mpsc::UnboundedReceiver<Op>) {
            let mut shutdown = Vec::new();
            loop {
                let next = if self.pending.is_empty() {
                    rx.recv().await
                } else {
                    match tokio::time::timeout(RETRY_INTERVAL, rx.recv()).await {
                        Ok(next) => next,
                        Err(_) => {
                            self.drain().await;
                            continue;
                        }
                    }
                };
                let Some(op) = next else { break };
                self.accept(op, &mut rx, &mut shutdown);
                while let Ok(op) = rx.try_recv() {
                    self.accept(op, &mut rx, &mut shutdown);
                }
                if !shutdown.is_empty() {
                    break;
                }
                self.drain().await;
            }
            let result = self.close().await;
            for ack in shutdown {
                let _ = ack.send(result.clone());
            }
        }

        /// Queue `op`. A shutdown closes the channel: ops already sent are
        /// still received, later sends fail.
        fn accept(
            &mut self,
            op: Op,
            rx: &mut mpsc::UnboundedReceiver<Op>,
            shutdown: &mut Vec<super::Ack>,
        ) {
            match op {
                Op::Shutdown(ack) => {
                    rx.close();
                    shutdown.push(ack);
                }
                op => self.pending.push_back(op),
            }
        }

        /// Commit every queued op. Save what the database refuses, for
        /// [`replay_saved`] on the next open.
        async fn close(&mut self) -> Result<(), String> {
            self.drain().await;
            if self.pending.is_empty() {
                return Ok(());
            }
            let count = self.pending.len();
            match save(&self.db_path, &self.owner, &self.pending) {
                Ok(file) => {
                    tracing::warn!(count, file = %file.display(), "task ledger: writes not committed at shutdown; saved for the next open");
                    Ok(())
                }
                Err(error) => {
                    let message = format!(
                        "task ledger: {count} writes not committed at shutdown and not saved: {error:#}"
                    );
                    tracing::error!("{message}");
                    Err(message)
                }
            }
        }

        /// Commit queued ops in batches. When a batch fails, retry its ops one
        /// transaction each. On a lasting fault, tell the waiting callers and
        /// keep the ops queued.
        async fn drain(&mut self) {
            while !self.pending.is_empty() {
                let len = self.pending.len().min(WRITE_BATCH_MAX);
                let committed = match self.write(0..len).await {
                    Ok(()) => len,
                    Err(error) => {
                        tracing::warn!(error = %format!("{error:#}"), "task ledger: batch failed; retrying ops one by one");
                        self.write_each(len).await
                    }
                };
                for mut op in self.pending.drain(..committed) {
                    op.ack(Ok(()));
                }
                self.writes_since_prune += committed as u64;
                if committed < len {
                    self.fail_waiters();
                    return;
                }
            }
            self.maybe_prune().await;
        }

        /// Prune once enough writes have committed or enough time has passed.
        /// A failed prune is logged and tried again on the next drain.
        async fn maybe_prune(&mut self) {
            let due = self.writes_since_prune >= u64::from(self.retention.prune_every_writes)
                || self.last_prune.elapsed() >= self.retention.prune_interval;
            if !due {
                return;
            }
            match prune(&self.pool, self.retention).await {
                Ok(()) => {
                    self.writes_since_prune = 0;
                    self.last_prune = Instant::now();
                }
                Err(error) => {
                    tracing::warn!(error = %format!("{error:#}"), "task ledger: prune failed");
                }
            }
        }

        /// Commit the first `len` ops one transaction each. Returns how many
        /// committed before one failed every retry.
        async fn write_each(&self, len: usize) -> usize {
            for index in 0..len {
                if let Err(error) = self.write_with_retry(index).await {
                    tracing::error!(
                        error = %format!("{error:#}"),
                        queued = self.pending.len() - index,
                        "task ledger: write failed; kept queued for retry"
                    );
                    return index;
                }
            }
            len
        }

        async fn write_with_retry(&self, index: usize) -> Result<()> {
            let mut backoff = RETRY_BACKOFF;
            let mut attempt = 1;
            loop {
                match self.write(index..index + 1).await {
                    Ok(()) => return Ok(()),
                    Err(error) if attempt >= RETRY_ATTEMPTS => return Err(error),
                    Err(_) => {
                        tokio::time::sleep(backoff).await;
                        backoff *= 2;
                        attempt += 1;
                    }
                }
            }
        }

        /// Write `pending[range]` in one transaction.
        async fn write(&self, range: std::ops::Range<usize>) -> Result<()> {
            if self
                .faults
                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
                .is_ok()
            {
                anyhow::bail!("injected write fault");
            }
            let mut tx = self.pool.begin().await?;
            for op in self.pending.range(range) {
                write_op(&mut tx, &self.owner, op).await?;
            }
            tx.commit().await?;
            Ok(())
        }

        /// Tell every caller still waiting that its write is not durable yet.
        fn fail_waiters(&mut self) {
            let message = format!(
                "task ledger: {} writes queued for retry",
                self.pending.len()
            );
            for op in &mut self.pending {
                op.ack(Err(message.clone()));
            }
        }
    }

    async fn write_op(conn: &mut SqliteConnection, owner: &Owner, op: &Op) -> Result<()> {
        match op {
            Op::Begin {
                id,
                parent_id,
                kind,
                name,
                ts,
                ..
            } => {
                let start = Start {
                    id,
                    parent_id: parent_id.as_deref(),
                    kind: *kind,
                    name,
                    ts,
                };
                start_task(conn, owner, &start).await
            }
            Op::Finish {
                id,
                state,
                reason,
                ts,
            } => {
                let payload = json!({ "state": state.as_str(), "reason": reason }).to_string();
                let end = End {
                    state: *state,
                    reason: reason.as_deref(),
                    ts,
                };
                end_task(conn, id, owner, &end, &payload).await
            }
            Op::Spawned { id, pid, pgid, ts } => {
                record_child(conn, owner, id, (*pid, *pgid), ts).await
            }
            Op::Flush(_) | Op::Shutdown(_) => Ok(()),
        }
    }

    /// Store the child pid and process group of a running task this owner
    /// owns, with a `spawned` event. Recording the same child again does
    /// nothing.
    async fn record_child(
        conn: &mut SqliteConnection,
        owner: &Owner,
        id: &str,
        (pid, pgid): (Option<u32>, Option<u32>),
        ts: &str,
    ) -> Result<()> {
        let updated = sqlx::query(
            "UPDATE tasks SET child_pid = ?, child_pgid = ?, updated_at = ? \
             WHERE id = ? AND state = 'running' AND host = ? AND owner_pid = ? AND owner_instance = ? \
             AND NOT (child_pid IS ? AND child_pgid IS ?)",
        )
        .bind(pid)
        .bind(pgid)
        .bind(ts)
        .bind(id)
        .bind(&owner.host)
        .bind(owner.pid)
        .bind(&owner.instance)
        .bind(pid)
        .bind(pgid)
        .execute(&mut *conn)
        .await?
        .rows_affected();
        if updated == 1 {
            let payload = json!({ "child_pid": pid, "child_pgid": pgid });
            append_event(conn, id, ts, "spawned", &payload.to_string()).await?;
        }
        Ok(())
    }

    struct Start<'a> {
        id: &'a str,
        parent_id: Option<&'a str>,
        kind: TaskKind,
        name: &'a str,
        ts: &'a str,
    }

    struct End<'a> {
        state: TaskState,
        reason: Option<&'a str>,
        ts: &'a str,
    }

    /// Insert a `running` task owned by `owner`. An id that already exists
    /// keeps its row and owner, and no event is written.
    async fn start_task(
        conn: &mut SqliteConnection,
        owner: &Owner,
        start: &Start<'_>,
    ) -> Result<()> {
        let inserted = sqlx::query(
            "INSERT INTO tasks (id, parent_id, kind, name, state, owner_pid, owner_instance, host, created_at, updated_at) \
             VALUES (?, ?, ?, ?, 'running', ?, ?, ?, ?, ?) ON CONFLICT(id) DO NOTHING",
        )
        .bind(start.id)
        .bind(start.parent_id)
        .bind(start.kind.as_str())
        .bind(start.name)
        .bind(owner.pid)
        .bind(&owner.instance)
        .bind(&owner.host)
        .bind(start.ts)
        .bind(start.ts)
        .execute(&mut *conn)
        .await?
        .rows_affected();
        if inserted == 0 {
            tracing::warn!(task_id = start.id, "task ledger: start refused, id exists");
            return Ok(());
        }
        let payload = json!({
            "kind": start.kind.as_str(),
            "name": start.name,
            "parent_id": start.parent_id,
        });
        append_event(conn, start.id, start.ts, "started", &payload.to_string()).await
    }

    /// Move a `running` task owned by `owner` to `end.state`. A task that
    /// already ended keeps its state, so a late finish cannot overwrite
    /// `host_lost`, and a task owned by another instance is left alone.
    async fn end_task(
        conn: &mut SqliteConnection,
        id: &str,
        owner: &Owner,
        end: &End<'_>,
        payload: &str,
    ) -> Result<()> {
        let updated = sqlx::query(
            "UPDATE tasks SET state = ?, terminal_reason = ?, updated_at = ? \
             WHERE id = ? AND state = 'running' AND host = ? AND owner_pid = ? AND owner_instance = ?",
        )
        .bind(end.state.as_str())
        .bind(end.reason)
        .bind(end.ts)
        .bind(id)
        .bind(&owner.host)
        .bind(owner.pid)
        .bind(&owner.instance)
        .execute(&mut *conn)
        .await?
        .rows_affected();
        if updated == 1 {
            let state = end.state;
            let ts = end.ts;
            let event = match state {
                TaskState::HostLost => "host_lost",
                TaskState::Orphaned => "orphaned",
                TaskState::Running
                | TaskState::Succeeded
                | TaskState::Failed
                | TaskState::Cancelled => "finished",
            };
            append_event(conn, id, ts, event, payload).await?;
        }
        Ok(())
    }

    async fn append_event(
        conn: &mut SqliteConnection,
        task_id: &str,
        ts: &str,
        event: &str,
        payload: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO task_events (task_id, seq, ts, type, payload) \
             SELECT ?1, COALESCE(MAX(seq), -1) + 1, ?2, ?3, ?4 FROM task_events WHERE task_id = ?1",
        )
        .bind(task_id)
        .bind(ts)
        .bind(event)
        .bind(payload)
        .execute(conn)
        .await?;
        Ok(())
    }

    /// `kill -0` through the system `kill`, as `process_tracker` does, so the
    /// crate needs no libc. A reused pid looks alive; heartbeat leases close
    /// that.
    #[cfg(unix)]
    async fn pid_alive(pid: i64) -> bool {
        if pid <= 0 {
            return false;
        }
        // Without a system `kill`, assume alive and leave the row alone.
        let Some(kill) = SystemBinary::Kill.path() else {
            return true;
        };
        tokio::process::Command::new(kill)
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            // If `kill` cannot run, assume alive and leave the row alone.
            .map_or(true, |status| status.success())
    }

    #[cfg(not(unix))]
    #[expect(
        clippy::unused_async,
        reason = "same signature as the async unix probe"
    )]
    async fn pid_alive(_pid: i64) -> bool {
        true
    }

    /// Whether any process in group `pgid` is alive (`kill -0 -- -pgid`).
    #[cfg(unix)]
    async fn group_alive(pgid: i64) -> bool {
        if pgid <= 1 {
            return false;
        }
        // Without a system `kill`, assume alive so the child is not hidden.
        let Some(kill) = SystemBinary::Kill.path() else {
            return true;
        };
        tokio::process::Command::new(kill)
            .args(["-0", "--", &format!("-{pgid}")])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            // If `kill` cannot run, assume alive so the child is not hidden.
            .map_or(true, |status| status.success())
    }

    #[cfg(not(unix))]
    #[expect(
        clippy::unused_async,
        reason = "same signature as the async unix probe"
    )]
    async fn group_alive(_pgid: i64) -> bool {
        false
    }

    async fn host_name() -> String {
        #[cfg(unix)]
        let Some(hostname) = SystemBinary::Hostname.path() else {
            return "unknown".to_owned();
        };
        #[cfg(not(unix))]
        let hostname = "hostname";
        let output = tokio::process::Command::new(hostname)
            .stderr(Stdio::null())
            .output()
            .await;
        output
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| "unknown".to_owned())
    }

    /// A temp-dir ledger for unit tests in other modules. Code run through
    /// [`TestLedger::run`] writes to it via the normal [`super::begin`].
    #[cfg(test)]
    pub(crate) mod test_support {
        use super::super::TEST_LEDGER;
        use super::{TASKS_DB_FILE, TaskLedger};
        use sqlx::SqlitePool;
        use sqlx::sqlite::SqliteConnectOptions;
        use std::future::Future;

        /// One stored task, as tests read it back.
        #[derive(Debug, Clone, PartialEq, Eq, sqlx::FromRow)]
        pub(crate) struct TaskRow {
            pub id: String,
            pub kind: String,
            pub name: String,
            pub state: String,
            pub terminal_reason: Option<String>,
            pub parent_id: Option<String>,
            pub child_pid: Option<i64>,
        }

        pub(crate) struct TestLedger {
            _dir: tempfile::TempDir,
            ledger: TaskLedger,
            pool: SqlitePool,
        }

        impl TestLedger {
            pub(crate) async fn open() -> Self {
                let dir = tempfile::tempdir().expect("temp dir");
                let path = dir.path().join(TASKS_DB_FILE);
                let ledger = TaskLedger::open(&path).await.expect("open ledger");
                let pool = SqlitePool::connect_with(SqliteConnectOptions::new().filename(&path))
                    .await
                    .expect("reader pool");
                Self {
                    _dir: dir,
                    ledger,
                    pool,
                }
            }

            /// Run `fut` with this ledger installed for the current task.
            pub(crate) async fn run<F: Future>(&self, fut: F) -> F::Output {
                TEST_LEDGER.scope(self.ledger.tx.clone(), fut).await
            }

            /// Every task, oldest first, once all queued writes committed.
            pub(crate) async fn tasks(&self) -> Vec<TaskRow> {
                self.ledger.flush().await.expect("flush");
                sqlx::query_as(
                    "SELECT id, kind, name, state, terminal_reason, parent_id, child_pid \
                     FROM tasks ORDER BY created_at, id",
                )
                .fetch_all(&self.pool)
                .await
                .expect("read tasks")
            }

            pub(crate) async fn task_named(&self, name: &str) -> TaskRow {
                self.tasks()
                    .await
                    .into_iter()
                    .find(|task| task.name == name)
                    .unwrap_or_else(|| panic!("no task named {name:?}"))
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::data::persistence::task_ledger::scope;

        async fn row(pool: &SqlitePool, id: &str) -> (String, Option<String>, Option<String>) {
            sqlx::query_as("SELECT state, terminal_reason, parent_id FROM tasks WHERE id = ?")
                .bind(id)
                .fetch_one(pool)
                .await
                .unwrap()
        }

        async fn insert_running(pool: &SqlitePool, id: &str, owner: &Owner) {
            let ts = now();
            sqlx::query(
                "INSERT INTO tasks (id, kind, name, state, owner_pid, owner_instance, host, created_at, updated_at) \
                 VALUES (?, 'shell', 'sleep 10', 'running', ?, ?, ?, ?, ?)",
            )
            .bind(id)
            .bind(owner.pid)
            .bind(&owner.instance)
            .bind(&owner.host)
            .bind(&ts)
            .bind(&ts)
            .execute(pool)
            .await
            .unwrap();
        }

        async fn events(pool: &SqlitePool, id: &str) -> Vec<String> {
            sqlx::query_scalar("SELECT type FROM task_events WHERE task_id = ? ORDER BY seq")
                .bind(id)
                .fetch_all(pool)
                .await
                .unwrap()
        }

        fn send_finish(ledger: &TaskLedger, id: &str) {
            ledger
                .tx
                .send(Op::Finish {
                    id: id.to_owned(),
                    state: TaskState::Succeeded,
                    reason: None,
                    ts: now(),
                })
                .unwrap();
        }

        /// Open a ledger plus a separate pool for reading what it wrote.
        async fn open(dir: &tempfile::TempDir) -> (TaskLedger, SqlitePool) {
            let path = dir.path().join(TASKS_DB_FILE);
            let ledger = TaskLedger::open(&path).await.unwrap();
            let pool = SqlitePool::connect_with(SqliteConnectOptions::new().filename(&path))
                .await
                .unwrap();
            (ledger, pool)
        }

        #[tokio::test]
        async fn finish_records_state_and_reason() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let mut ok = ledger.begin(TaskKind::Shell, "true");
            ok.recorded().await.unwrap();
            let ok_id = ok.id().unwrap().to_owned();
            ok.finish(TaskState::Succeeded, None);
            let bad = ledger.begin(TaskKind::Shell, "false");
            let bad_id = bad.id().unwrap().to_owned();
            bad.finish(TaskState::Failed, Some("exit code 2"));
            ledger.flush().await.unwrap();

            assert_eq!(row(&pool, &ok_id).await.0, "succeeded");
            let (state, reason, _) = row(&pool, &bad_id).await;
            assert_eq!(state, "failed");
            assert_eq!(reason.as_deref(), Some("exit code 2"));
        }

        #[tokio::test]
        async fn dropped_guard_ends_cancelled() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let guard = ledger.begin(TaskKind::Subagent, "child");
            let id = guard.id().unwrap().to_owned();
            drop(guard);
            ledger.flush().await.unwrap();

            let (state, reason, _) = row(&pool, &id).await;
            assert_eq!(state, "cancelled");
            assert_eq!(reason.as_deref(), Some("dropped"));
        }

        #[tokio::test]
        async fn scope_links_child_to_parent() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let parent = ledger.begin(TaskKind::Subagent, "child-session");
            let parent_id = parent.id().map(str::to_owned);
            let child = scope(parent_id.clone(), async {
                ledger.begin(TaskKind::Shell, "echo hi")
            })
            .await;
            let child_id = child.id().unwrap().to_owned();
            child.finish(TaskState::Succeeded, None);
            parent.finish(TaskState::Succeeded, None);
            ledger.flush().await.unwrap();

            assert_eq!(row(&pool, &child_id).await.2, parent_id);
        }

        async fn user_version(pool: &SqlitePool) -> i64 {
            sqlx::query_scalar("PRAGMA user_version")
                .fetch_one(pool)
                .await
                .unwrap()
        }

        #[tokio::test]
        async fn new_ledger_is_schema_version_one() {
            let dir = tempfile::tempdir().unwrap();
            let (_ledger, pool) = open(&dir).await;
            assert_eq!(user_version(&pool).await, SCHEMA_VERSION);
            assert_eq!(SCHEMA_VERSION, 1);
        }

        #[tokio::test]
        async fn unversioned_ledger_is_upgraded_in_place() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(TASKS_DB_FILE);
            let old = SqlitePool::connect_with(
                SqliteConnectOptions::new()
                    .filename(&path)
                    .create_if_missing(true),
            )
            .await
            .unwrap();
            // The first, unversioned layout: no owner_instance or child columns.
            for statement in [
                "CREATE TABLE tasks (id TEXT PRIMARY KEY, parent_id TEXT, kind TEXT NOT NULL, \
                 name TEXT NOT NULL, state TEXT NOT NULL, owner_pid INTEGER NOT NULL, \
                 host TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, \
                 terminal_reason TEXT)",
                "CREATE TABLE task_events (task_id TEXT NOT NULL, seq INTEGER NOT NULL, \
                 ts TEXT NOT NULL, type TEXT NOT NULL, payload TEXT NOT NULL DEFAULT '{}', \
                 PRIMARY KEY (task_id, seq))",
                "INSERT INTO tasks (id, kind, name, state, owner_pid, host, created_at, updated_at) \
                 VALUES ('old', 'shell', 'true', 'succeeded', 1, 'h', '2999-01-01T00:00:00.000000Z', \
                 '2999-01-01T00:00:00.000000Z')",
            ] {
                sqlx::query(statement).execute(&old).await.unwrap();
            }
            old.close().await;

            let (ledger, pool) = open(&dir).await;
            assert_eq!(user_version(&pool).await, SCHEMA_VERSION);
            assert_eq!(row(&pool, "old").await.0, "succeeded");
            let guard = ledger.begin(TaskKind::Shell, "echo new");
            let id = guard.id().unwrap().to_owned();
            guard.set_process(Some(7), Some(7));
            guard.finish(TaskState::Succeeded, None);
            ledger.flush().await.unwrap();
            assert_eq!(row(&pool, &id).await.0, "succeeded");
        }

        #[tokio::test]
        async fn newer_schema_version_is_refused() {
            let dir = tempfile::tempdir().unwrap();
            let (_ledger, pool) = open(&dir).await;
            sqlx::query("PRAGMA user_version = 2")
                .execute(&pool)
                .await
                .unwrap();

            let error = TaskLedger::open(&dir.path().join(TASKS_DB_FILE))
                .await
                .unwrap_err();
            assert!(error.to_string().contains("version 2"), "{error:#}");
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn ledger_files_are_owner_only() {
            use std::os::unix::fs::PermissionsExt;

            let dir = tempfile::tempdir().unwrap();
            let (ledger, _pool) = open(&dir).await;
            let guard = ledger.begin(TaskKind::Shell, "true");
            guard.finish(TaskState::Succeeded, None);
            ledger.flush().await.unwrap();

            for file in ["tasks.db", "tasks.db-wal", "tasks.db-shm"] {
                let mode = std::fs::metadata(dir.path().join(file))
                    .unwrap_or_else(|error| panic!("{file}: {error}"))
                    .permissions()
                    .mode();
                assert_eq!(mode & 0o777, 0o600, "{file} mode {mode:o}");
            }
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn reopen_tightens_an_existing_world_readable_ledger() {
            use std::os::unix::fs::PermissionsExt;

            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(TASKS_DB_FILE);
            std::fs::write(&path, b"").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
            let _ledger = TaskLedger::open(&path).await.unwrap();

            let mode = std::fs::metadata(&path).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
        }

        #[tokio::test]
        async fn stored_name_reason_and_events_are_redacted() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let guard = ledger.begin(TaskKind::Shell, "API_TOKEN=tok-123 ./deploy");
            let id = guard.id().unwrap().to_owned();
            guard.finish(TaskState::Failed, Some("auth failed for password=hunter2"));
            ledger.flush().await.unwrap();

            let stored: (String, Option<String>) =
                sqlx::query_as("SELECT name, terminal_reason FROM tasks WHERE id = ?")
                    .bind(&id)
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            let payloads: Vec<String> =
                sqlx::query_scalar("SELECT payload FROM task_events WHERE task_id = ?")
                    .bind(&id)
                    .fetch_all(&pool)
                    .await
                    .unwrap();
            let all = format!("{stored:?} {payloads:?}");
            assert!(
                !all.contains("tok-123") && !all.contains("hunter2"),
                "{all}"
            );
            assert_eq!(stored.0, "[REDACTED] ./deploy");
        }

        #[tokio::test]
        async fn set_process_stores_the_child_pid_and_group() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let guard = ledger.begin(TaskKind::Shell, "sleep 1");
            let id = guard.id().unwrap().to_owned();
            guard.set_process(Some(4242), Some(4242));
            guard.finish(TaskState::Succeeded, None);
            ledger.flush().await.unwrap();

            let child: (Option<i64>, Option<i64>) =
                sqlx::query_as("SELECT child_pid, child_pgid FROM tasks WHERE id = ?")
                    .bind(&id)
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            assert_eq!(child, (Some(4242), Some(4242)));
            assert_eq!(
                events(&pool, &id).await,
                vec!["started", "spawned", "finished"]
            );
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn reopen_marks_task_orphaned_while_its_child_still_runs() {
            use std::os::unix::process::CommandExt;

            let dir = tempfile::tempdir().unwrap();
            let (_ledger, pool) = open(&dir).await;
            let mut exited = std::process::Command::new("true").spawn().unwrap();
            let dead_pid = i64::from(exited.id());
            exited.wait().unwrap();
            let mut child = std::process::Command::new("sleep")
                .arg("30")
                .process_group(0)
                .spawn()
                .unwrap();
            let child_pid = i64::from(child.id());
            let dead_owner = Owner {
                pid: dead_pid,
                instance: "dead-instance".to_owned(),
                ..Owner::current().await
            };
            for (id, child) in [("orphan", child_pid), ("lost", dead_pid)] {
                insert_running(&pool, id, &dead_owner).await;
                sqlx::query("UPDATE tasks SET child_pid = ?, child_pgid = ? WHERE id = ?")
                    .bind(child)
                    .bind(child)
                    .bind(id)
                    .execute(&pool)
                    .await
                    .unwrap();
            }

            let _reopened = open(&dir).await;
            let _ = child.kill();
            let _ = child.wait();

            assert_eq!(row(&pool, "orphan").await.0, "orphaned");
            assert_eq!(row(&pool, "lost").await.0, "host_lost");
            let payload: String = sqlx::query_scalar(
                "SELECT payload FROM task_events WHERE task_id = 'orphan' AND type = 'orphaned'",
            )
            .fetch_one(&pool)
            .await
            .unwrap();
            let payload: serde_json::Value = serde_json::from_str(&payload).unwrap();
            assert_eq!(payload["child_pid"], child_pid);
            assert_eq!(payload["owner_pid"], dead_pid);
        }

        #[tokio::test]
        async fn reopen_marks_dead_owner_host_lost_and_keeps_live_owner() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let live = ledger.begin(TaskKind::Shell, "sleep 10");
            let live_id = live.id().unwrap().to_owned();
            ledger.flush().await.unwrap();

            let mut exited = std::process::Command::new("true").spawn().unwrap();
            let dead_pid = i64::from(exited.id());
            exited.wait().unwrap();
            let me = Owner::current().await;
            let dead = Owner {
                pid: dead_pid,
                instance: "dead-instance".to_owned(),
                ..me.clone()
            };
            insert_running(&pool, "dead-owner", &dead).await;
            let reused = Owner {
                instance: "earlier-instance".to_owned(),
                ..me.clone()
            };
            insert_running(&pool, "reused-pid", &reused).await;
            let elsewhere = Owner {
                host: "other-host".to_owned(),
                ..dead.clone()
            };
            insert_running(&pool, "other-host-task", &elsewhere).await;

            let _reopened = open(&dir).await;
            assert_eq!(row(&pool, "dead-owner").await.0, "host_lost");
            assert_eq!(row(&pool, "reused-pid").await.0, "host_lost");
            assert_eq!(row(&pool, "other-host-task").await.0, "running");
            assert_eq!(row(&pool, &live_id).await.0, "running");
            assert_eq!(events(&pool, "dead-owner").await, vec!["host_lost"]);
            drop(live);
        }

        async fn insert_ended(pool: &SqlitePool, id: &str, age: chrono::Duration) {
            let me = Owner::current().await;
            insert_running(pool, id, &me).await;
            sqlx::query("UPDATE tasks SET state = 'succeeded', updated_at = ? WHERE id = ?")
                .bind(timestamp(Utc::now() - age))
                .bind(id)
                .execute(pool)
                .await
                .unwrap();
            sqlx::query(
                "INSERT INTO task_events (task_id, seq, ts, type) VALUES (?, 0, '', 'finished')",
            )
            .bind(id)
            .execute(pool)
            .await
            .unwrap();
        }

        async fn task_count(pool: &SqlitePool, id: &str) -> i64 {
            sqlx::query_scalar("SELECT COUNT(*) FROM tasks WHERE id = ?")
                .bind(id)
                .fetch_one(pool)
                .await
                .unwrap()
        }

        #[tokio::test]
        async fn reopen_prunes_ended_tasks_past_max_age() {
            let dir = tempfile::tempdir().unwrap();
            let (_ledger, pool) = open(&dir).await;
            insert_ended(&pool, "old-ended", chrono::Duration::days(8)).await;
            insert_ended(&pool, "recent-ended", chrono::Duration::days(1)).await;
            let me = Owner::current().await;
            insert_running(&pool, "old-running", &me).await;
            sqlx::query("UPDATE tasks SET updated_at = ? WHERE id = 'old-running'")
                .bind(timestamp(Utc::now() - chrono::Duration::days(30)))
                .execute(&pool)
                .await
                .unwrap();

            let _reopened = open(&dir).await;
            assert_eq!(task_count(&pool, "old-ended").await, 0);
            assert!(events(&pool, "old-ended").await.is_empty());
            assert_eq!(task_count(&pool, "recent-ended").await, 1);
            assert_eq!(events(&pool, "recent-ended").await, vec!["finished"]);
            assert_eq!(row(&pool, "old-running").await.0, "running");
        }

        #[tokio::test]
        async fn reopen_caps_ended_tasks_keeping_the_newest() {
            let dir = tempfile::tempdir().unwrap();
            let (_ledger, pool) = open(&dir).await;
            insert_ended(&pool, "older", chrono::Duration::hours(2)).await;
            insert_ended(&pool, "newer", chrono::Duration::hours(1)).await;

            let retention = Retention {
                max_ended_tasks: 1,
                ..Retention::default()
            };
            let _reopened = TaskLedger::open_with(&dir.path().join(TASKS_DB_FILE), retention)
                .await
                .unwrap();
            assert_eq!(task_count(&pool, "older").await, 0);
            assert!(events(&pool, "older").await.is_empty());
            assert_eq!(task_count(&pool, "newer").await, 1);
        }

        /// Open a ledger with `retention`, end three tasks, and return how
        /// many ended tasks remain once the writer has caught up.
        async fn ended_after_three_tasks(retention: Retention) -> i64 {
            let dir = tempfile::tempdir().unwrap();
            let (_, pool) = open(&dir).await;
            let ledger = TaskLedger::open_with(&dir.path().join(TASKS_DB_FILE), retention)
                .await
                .unwrap();
            for n in 0..3 {
                let mut guard = ledger.begin(TaskKind::Shell, format!("echo {n}"));
                guard.recorded().await.unwrap();
                guard.finish(TaskState::Succeeded, None);
                ledger.flush().await.unwrap();
            }
            // The writer prunes after acking; a second flush waits for that.
            ledger.flush().await.unwrap();
            sqlx::query_scalar("SELECT COUNT(*) FROM tasks WHERE state != 'running'")
                .fetch_one(&pool)
                .await
                .unwrap()
        }

        #[tokio::test]
        async fn writer_prunes_after_enough_writes() {
            let retention = Retention {
                max_ended_tasks: 1,
                prune_every_writes: 2,
                prune_interval: Duration::MAX,
                ..Retention::default()
            };
            assert_eq!(ended_after_three_tasks(retention).await, 1);
        }

        #[tokio::test]
        async fn writer_prunes_once_the_interval_passes() {
            let retention = Retention {
                max_ended_tasks: 1,
                prune_every_writes: u32::MAX,
                prune_interval: Duration::ZERO,
                ..Retention::default()
            };
            assert_eq!(ended_after_three_tasks(retention).await, 1);
        }

        #[tokio::test]
        async fn writer_keeps_ended_tasks_until_a_prune_is_due() {
            let retention = Retention {
                max_ended_tasks: 1,
                prune_every_writes: u32::MAX,
                prune_interval: Duration::MAX,
                ..Retention::default()
            };
            assert_eq!(ended_after_three_tasks(retention).await, 3);
        }

        #[tokio::test]
        async fn transient_write_faults_lose_no_start_or_finish() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            // The batch and the first one-op try fail; the retry commits.
            ledger.inject_write_faults(2);
            let mut guard = ledger.begin(TaskKind::Shell, "true");
            guard.recorded().await.unwrap();
            let id = guard.id().unwrap().to_owned();
            ledger.inject_write_faults(2);
            guard.finish(TaskState::Succeeded, None);
            ledger.flush().await.unwrap();

            assert_eq!(row(&pool, &id).await.0, "succeeded");
            assert_eq!(events(&pool, &id).await, vec!["started", "finished"]);
        }

        #[tokio::test]
        async fn lasting_write_fault_reports_and_keeps_writes_queued() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            // The batch and every retry of the start fail.
            ledger.inject_write_faults(1 + RETRY_ATTEMPTS);
            let mut guard = ledger.begin(TaskKind::Shell, "true");
            let error = guard.recorded().await.unwrap_err();
            assert!(error.to_string().contains("queued for retry"), "{error}");
            let id = guard.id().unwrap().to_owned();
            guard.finish(TaskState::Succeeded, None);
            ledger.flush().await.unwrap();

            assert_eq!(row(&pool, &id).await.0, "succeeded");
            assert_eq!(events(&pool, &id).await, vec!["started", "finished"]);
            // Reopening must not take the finished task for a lost one.
            let _reopened = open(&dir).await;
            assert_eq!(row(&pool, &id).await.0, "succeeded");
        }

        #[tokio::test]
        async fn shutdown_commits_writes_queued_before_it() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            // The writer has not run yet: shutdown is what commits these.
            let guard = ledger.begin(TaskKind::Shell, "echo bye");
            let id = guard.id().unwrap().to_owned();
            guard.set_process(Some(4242), Some(4242));
            guard.finish(TaskState::Succeeded, Some("exit code 0"));
            ledger.shutdown().await.unwrap();

            assert_eq!(
                row(&pool, &id).await,
                ("succeeded".to_owned(), Some("exit code 0".to_owned()), None)
            );
            assert_eq!(
                events(&pool, &id).await,
                vec!["started", "spawned", "finished"]
            );
        }

        #[tokio::test]
        async fn writes_refused_at_shutdown_are_committed_once_on_next_open() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(TASKS_DB_FILE);
            let ledger = TaskLedger::open(&path).await.unwrap();
            ledger.inject_write_faults(u32::MAX);
            let done = ledger.begin(TaskKind::Shell, "echo done");
            let done_id = done.id().unwrap().to_owned();
            done.finish(TaskState::Failed, Some("exit code 1"));
            let running = ledger.begin(TaskKind::Shell, "sleep 60");
            let running_id = running.id().unwrap().to_owned();
            running.set_process(Some(4343), Some(4343));
            ledger.shutdown().await.unwrap();

            let pool = SqlitePool::connect_with(SqliteConnectOptions::new().filename(&path))
                .await
                .unwrap();
            assert_eq!(task_count(&pool, &done_id).await, 0);
            let saved = saved_files(&path, SAVED_SUFFIX).unwrap();
            assert_eq!(saved.len(), 1, "{saved:?}");
            let saved_copy = std::fs::read(&saved[0]).unwrap();

            let expect_replayed = || async {
                assert_eq!(
                    row(&pool, &done_id).await,
                    ("failed".to_owned(), Some("exit code 1".to_owned()), None)
                );
                assert_eq!(events(&pool, &done_id).await, vec!["started", "finished"]);
                assert_eq!(row(&pool, &running_id).await.0, "running");
                assert_eq!(events(&pool, &running_id).await, vec!["started", "spawned"]);
                let child: (Option<i64>, Option<i64>) =
                    sqlx::query_as("SELECT child_pid, child_pgid FROM tasks WHERE id = ?")
                        .bind(&running_id)
                        .fetch_one(&pool)
                        .await
                        .unwrap();
                assert_eq!(child, (Some(4343), Some(4343)));
                assert_eq!(
                    saved_files(&path, SAVED_SUFFIX).unwrap(),
                    Vec::<PathBuf>::new()
                );
            };
            TaskLedger::open(&path).await.unwrap();
            expect_replayed().await;

            // A crash between the commit and the delete replays the file again.
            std::fs::write(&saved[0], &saved_copy).unwrap();
            TaskLedger::open(&path).await.unwrap();
            expect_replayed().await;
            drop(running);
        }

        /// Counts ERROR events while installed as the thread's subscriber.
        struct ErrorCount(Arc<AtomicU32>);

        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for ErrorCount {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if *event.metadata().level() == tracing::Level::ERROR {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }
        }

        /// Open the ledger at `path` and count the errors it logs.
        async fn open_counting_errors(path: &Path) -> (Result<TaskLedger>, u32) {
            use tracing_subscriber::layer::SubscriberExt;
            let count = Arc::new(AtomicU32::new(0));
            let subscriber = tracing_subscriber::registry().with(ErrorCount(Arc::clone(&count)));
            let _default = tracing::subscriber::set_default(subscriber);
            let opened = TaskLedger::open(path).await;
            (opened, count.load(Ordering::SeqCst))
        }

        fn names_in(dir: &Path) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(dir)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }

        #[tokio::test]
        async fn corrupt_and_truncated_saved_files_are_set_aside_and_the_rest_replay() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(TASKS_DB_FILE);
            let ledger = TaskLedger::open(&path).await.unwrap();
            ledger.inject_write_faults(u32::MAX);
            let good = ledger.begin(TaskKind::Shell, "echo good");
            let good_id = good.id().unwrap().to_owned();
            good.finish(TaskState::Succeeded, Some("exit code 0"));
            ledger.shutdown().await.unwrap();
            let saved = saved_files(&path, SAVED_SUFFIX).unwrap();
            assert_eq!(saved.len(), 1, "{saved:?}");
            let body = std::fs::read(&saved[0]).unwrap();
            let corrupt = "tasks.db.pending-00000000-0000-7000-8000-000000000001.json";
            let truncated = "tasks.db.pending-00000000-0000-7000-8000-000000000002.json";
            std::fs::write(dir.path().join(corrupt), b"{\"owner\": not json").unwrap();
            std::fs::write(dir.path().join(truncated), &body[..body.len() / 2]).unwrap();

            let (opened, errors) = open_counting_errors(&path).await;
            opened.unwrap();

            assert_eq!(errors, 2);
            let pool = SqlitePool::connect_with(SqliteConnectOptions::new().filename(&path))
                .await
                .unwrap();
            assert_eq!(
                row(&pool, &good_id).await,
                ("succeeded".to_owned(), Some("exit code 0".to_owned()), None)
            );
            assert_eq!(
                saved_files(&path, SAVED_SUFFIX).unwrap(),
                Vec::<PathBuf>::new()
            );
            let set_aside: Vec<(String, usize)> = names_in(dir.path())
                .into_iter()
                .filter_map(|name| {
                    let (original, at) = name.split_once(".corrupt-")?;
                    Some((original.to_owned(), at.len()))
                })
                .collect();
            assert_eq!(
                set_aside,
                vec![(corrupt.to_owned(), 23), (truncated.to_owned(), 23)]
            );
        }

        #[tokio::test]
        async fn open_removes_stale_temporary_saved_files_only() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(TASKS_DB_FILE);
            let stale = "tasks.db.pending-00000000-0000-7000-8000-000000000001.json.tmp";
            let fresh = "tasks.db.pending-00000000-0000-7000-8000-000000000002.json.tmp";
            std::fs::write(dir.path().join(stale), b"{").unwrap();
            std::fs::write(dir.path().join(fresh), b"{").unwrap();
            std::fs::File::options()
                .write(true)
                .open(dir.path().join(stale))
                .unwrap()
                .set_modified(std::time::SystemTime::now() - 2 * STALE_TMP_AGE)
                .unwrap();

            TaskLedger::open(&path).await.unwrap();

            assert_eq!(
                saved_files(&path, TMP_SUFFIX).unwrap(),
                vec![dir.path().join(fresh)]
            );
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn shutdown_fails_when_refused_writes_cannot_be_saved() {
            let dir = tempfile::tempdir().unwrap();
            let ledger = TaskLedger::open(&dir.path().join(TASKS_DB_FILE))
                .await
                .unwrap();
            ledger.inject_write_faults(u32::MAX);
            ledger
                .begin(TaskKind::Shell, "echo lost")
                .finish(TaskState::Succeeded, None);
            std::fs::remove_dir_all(dir.path()).unwrap();

            let error = ledger.shutdown().await.unwrap_err().to_string();

            let expected = "task ledger: 2 writes not committed at shutdown and not saved: ";
            assert!(error.starts_with(expected), "{error}");
        }

        #[tokio::test]
        async fn burst_past_the_old_queue_depth_loses_nothing() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let guards: Vec<_> = (0..3_000)
                .map(|n| ledger.begin(TaskKind::Shell, format!("echo {n}")))
                .collect();
            for guard in guards {
                guard.finish(TaskState::Succeeded, None);
            }
            ledger.flush().await.unwrap();

            let succeeded: i64 =
                sqlx::query_scalar("SELECT COUNT(*) FROM tasks WHERE state = 'succeeded'")
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            assert_eq!(succeeded, 3_000);
        }

        #[tokio::test]
        async fn foreign_owner_cannot_start_or_finish_a_task() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let me = Owner::current().await;
            let other_instance = Owner {
                instance: "other-instance".to_owned(),
                ..me.clone()
            };
            let other_host = Owner {
                host: "other-host".to_owned(),
                ..me.clone()
            };
            insert_running(&pool, "other-instance-task", &other_instance).await;
            insert_running(&pool, "other-host-task", &other_host).await;

            for id in ["other-instance-task", "other-host-task"] {
                send_finish(&ledger, id);
                ledger
                    .tx
                    .send(Op::Begin {
                        id: id.to_owned(),
                        parent_id: None,
                        kind: TaskKind::Shell,
                        name: "hijack".to_owned(),
                        ts: now(),
                        ack: None,
                    })
                    .unwrap();
            }
            ledger.flush().await.unwrap();

            for id in ["other-instance-task", "other-host-task"] {
                assert_eq!(row(&pool, id).await.0, "running");
                assert!(events(&pool, id).await.is_empty());
            }
            let instance: (String,) =
                sqlx::query_as("SELECT owner_instance FROM tasks WHERE id = 'other-instance-task'")
                    .fetch_one(&pool)
                    .await
                    .unwrap();
            assert_eq!(instance.0, "other-instance");
        }

        /// Env var naming the ledger the kill-test child writes to.
        const KILL_CHILD_DB: &str = "CALICHE_TASK_LEDGER_KILL_CHILD_DB";

        /// Child half of `sigkill_mid_write_leaves_a_clean_ledger`: writes
        /// to the ledger without stopping until it is killed. Ignored, and a
        /// no-op unless the parent sets [`KILL_CHILD_DB`].
        #[tokio::test]
        #[ignore = "run only as the child of sigkill_mid_write_leaves_a_clean_ledger"]
        async fn kill_child_writes_until_killed() {
            let Some(path) = std::env::var_os(KILL_CHILD_DB) else {
                return;
            };
            let ledger = TaskLedger::open(Path::new(&path)).await.unwrap();
            for n in 0u64.. {
                let mut guard = ledger.begin(TaskKind::Shell, format!("kill-child {n}"));
                guard.recorded().await.unwrap();
                if n % 2 == 0 {
                    guard.finish(TaskState::Succeeded, None);
                } else {
                    // Leave odd tasks running, as a dying process would.
                    #[expect(
                        clippy::mem_forget,
                        reason = "skip the guard's drop to simulate a crash"
                    )]
                    std::mem::forget(guard);
                }
            }
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn sigkill_mid_write_leaves_a_clean_ledger() {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(TASKS_DB_FILE);
            let mut child = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "data::persistence::task_ledger::store::tests::kill_child_writes_until_killed",
                    "--ignored",
                    "--test-threads=1",
                ])
                .env(KILL_CHILD_DB, &path)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap();

            // Wait until the child is well into its writes, then SIGKILL it.
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            let reader = loop {
                assert!(std::time::Instant::now() < deadline, "child never wrote");
                tokio::time::sleep(Duration::from_millis(50)).await;
                let Ok(pool) =
                    SqlitePool::connect_with(SqliteConnectOptions::new().filename(&path)).await
                else {
                    continue;
                };
                let count: Result<i64, _> = sqlx::query_scalar("SELECT COUNT(*) FROM tasks")
                    .fetch_one(&pool)
                    .await;
                if count.is_ok_and(|count| count >= 50) {
                    break pool;
                }
            };
            child.kill().unwrap();
            child.wait().unwrap();
            reader.close().await;

            let (_reopened, pool) = open(&dir).await;
            let integrity: String = sqlx::query_scalar("PRAGMA integrity_check")
                .fetch_one(&pool)
                .await
                .unwrap();
            assert_eq!(integrity, "ok");
            let by_state: Vec<(String, i64)> =
                sqlx::query_as("SELECT state, COUNT(*) FROM tasks GROUP BY state ORDER BY state")
                    .fetch_all(&pool)
                    .await
                    .unwrap();
            assert!(
                by_state.iter().all(|(state, _)| state != "running"),
                "no task may stay running: {by_state:?}"
            );
            assert!(
                by_state.iter().any(|(state, _)| state == "host_lost"),
                "the killed owner's open tasks become host_lost: {by_state:?}"
            );
        }

        #[tokio::test]
        async fn writer_ignores_a_second_end_for_the_same_task() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let guard = ledger.begin(TaskKind::Shell, "true");
            let id = guard.id().unwrap().to_owned();
            guard.finish(TaskState::Succeeded, None);
            ledger
                .tx
                .send(Op::Finish {
                    id: id.clone(),
                    state: TaskState::Failed,
                    reason: Some("late".to_owned()),
                    ts: now(),
                })
                .unwrap();
            ledger.flush().await.unwrap();

            let (state, reason, _) = row(&pool, &id).await;
            assert_eq!((state.as_str(), reason), ("succeeded", None));
            assert_eq!(events(&pool, &id).await, vec!["started", "finished"]);
        }

        #[tokio::test]
        async fn guard_refuses_states_only_reconcile_may_set() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            for state in [TaskState::Running, TaskState::HostLost, TaskState::Orphaned] {
                let guard = ledger.begin(TaskKind::Shell, format!("refuse {}", state.as_str()));
                let id = guard.id().unwrap().to_owned();
                guard.finish(state, Some("forged"));
                ledger.flush().await.unwrap();

                let (stored, reason, _) = row(&pool, &id).await;
                assert_eq!(
                    (stored.as_str(), reason.as_deref()),
                    ("cancelled", Some("dropped")),
                    "finish({state:?}) must be refused"
                );
            }
        }

        #[tokio::test]
        async fn writer_leaves_ended_rows_alone() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let me = Owner::current().await;
            for (id, state) in [
                ("was-host-lost", "host_lost"),
                ("was-orphaned", "orphaned"),
                ("was-cancelled", "cancelled"),
            ] {
                insert_running(&pool, id, &me).await;
                sqlx::query("UPDATE tasks SET state = ? WHERE id = ?")
                    .bind(state)
                    .bind(id)
                    .execute(&pool)
                    .await
                    .unwrap();
                send_finish(&ledger, id);
                ledger
                    .tx
                    .send(Op::Spawned {
                        id: id.to_owned(),
                        pid: Some(99),
                        pgid: Some(99),
                        ts: now(),
                    })
                    .unwrap();
                ledger.flush().await.unwrap();

                assert_eq!(row(&pool, id).await.0, state);
                assert!(events(&pool, id).await.is_empty(), "{id}");
                let child: Option<i64> =
                    sqlx::query_scalar("SELECT child_pid FROM tasks WHERE id = ?")
                        .bind(id)
                        .fetch_one(&pool)
                        .await
                        .unwrap();
                assert_eq!(child, None, "{id}");
            }
        }

        #[tokio::test]
        async fn late_finish_does_not_overwrite_host_lost() {
            let dir = tempfile::tempdir().unwrap();
            let (ledger, pool) = open(&dir).await;
            let guard = ledger.begin(TaskKind::Shell, "sleep 10");
            let id = guard.id().unwrap().to_owned();
            ledger.flush().await.unwrap();
            sqlx::query("UPDATE tasks SET state = 'host_lost' WHERE id = ?")
                .bind(&id)
                .execute(&pool)
                .await
                .unwrap();
            guard.finish(TaskState::Succeeded, None);
            ledger.flush().await.unwrap();

            assert_eq!(row(&pool, &id).await.0, "host_lost");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TaskKind, TaskState, begin, can_transition};

    const ALL: [TaskState; 6] = [
        TaskState::Running,
        TaskState::Succeeded,
        TaskState::Failed,
        TaskState::Cancelled,
        TaskState::HostLost,
        TaskState::Orphaned,
    ];

    #[test]
    fn redact_removes_secret_like_values() {
        let cases = [
            ("export OPENAI_API_KEY=sk-live123", "OPENAI_API_KEY"),
            ("curl --password=hunter2 x", "hunter2"),
            ("GITHUB_TOKEN=abc123 make", "abc123"),
            ("DB_SECRET=shh ./run", "shh"),
            (
                "curl -H 'Authorization: Basic dXNlcjpwYXNz' x",
                "dXNlcjpwYXNz",
            ),
            ("curl -H 'X: Bearer abc.def.ghi' x", "abc.def.ghi"),
            ("git clone https://user:pa55@example.com/r", "pa55"),
        ];
        for (input, secret) in cases {
            let out = super::redact(input);
            assert!(!out.contains(secret), "{input:?} -> {out:?}");
            assert!(out.contains("[REDACTED]"), "{input:?} -> {out:?}");
        }
        let dynamic = [
            (
                format!("echo s{}-{}", "k", "abcdefghijklmnopqrstuv"),
                format!("s{}-{}", "k", "abcdefghijklmnop"),
            ),
            (
                format!("git push gh{}_abcdefghijklmnopqrstuvwxyz0123", "p"),
                format!("gh{}_abcdef", "p"),
            ),
            (
                format!("slack xox{}-1234567890-abcdef", "b"),
                format!("xox{}-1234567890", "b"),
            ),
            (
                format!("aws AK{}ABCDEFGHIJKLMNOP", "IA"),
                format!("AK{}ABCDEFGHIJKLMNOP", "IA"),
            ),
        ];
        for (input, secret) in dynamic {
            let out = super::redact(&input);
            assert!(!out.contains(&secret), "{input:?} -> {out:?}");
            assert!(out.contains("[REDACTED]"), "{input:?} -> {out:?}");
        }
        assert_eq!(
            super::redact("cargo test -p caliche"),
            "cargo test -p caliche"
        );
        assert_eq!(
            super::redact("git clone https://example.com/r"),
            "git clone https://example.com/r"
        );
    }

    #[test]
    fn owner_finishes_only_with_its_own_end_states() {
        let allowed: Vec<TaskState> = ALL
            .into_iter()
            .filter(|state| state.owner_can_finish_as())
            .collect();
        assert_eq!(
            allowed,
            vec![
                TaskState::Succeeded,
                TaskState::Failed,
                TaskState::Cancelled
            ]
        );
    }

    #[test]
    fn only_running_tasks_can_end() {
        for from in ALL {
            for to in ALL {
                let expected = from == TaskState::Running && to != TaskState::Running;
                assert_eq!(can_transition(from, to), expected, "{from:?} -> {to:?}");
            }
        }
    }

    #[test]
    fn begin_without_ledger_is_inert() {
        let guard = begin(TaskKind::Shell, "true");
        assert_eq!(guard.id(), None);
        guard.finish(TaskState::Succeeded, None);
    }
}
