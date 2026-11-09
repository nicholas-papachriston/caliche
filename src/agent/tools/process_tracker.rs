//! Lightweight tracking for processes started by agent tools.
//!
//! The registry is intentionally process-local. It is used to surface long-running
//! or leaked tool processes to CLI users at turn boundaries. Each record also
//! holds a durable task in the task ledger when one is installed.

use crate::data::persistence::task_ledger::{self, TaskGuard, TaskKind, TaskState};
#[cfg(unix)]
use crate::security::SystemBinary;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

static NEXT_PROCESS_ID: AtomicU64 = AtomicU64::new(1);
static PROCESS_REGISTRY: OnceLock<Mutex<ProcessRegistry>> = OnceLock::new();

#[derive(Debug, Default)]
struct ProcessRegistry {
    records: HashMap<u64, ProcessRecord>,
}

#[derive(Debug)]
struct ProcessRecord {
    id: u64,
    tool_name: String,
    command: String,
    cwd: Option<PathBuf>,
    pid: Option<u32>,
    process_group_id: Option<u32>,
    started_at: Instant,
    finished_at: Option<Instant>,
    exit_code: Option<i64>,
    task: Option<TaskGuard>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackedProcessState {
    Active,
    Lingering,
}

#[derive(Debug, Clone)]
pub struct TrackedProcessSnapshot {
    pub id: u64,
    pub tool_name: String,
    pub command: String,
    pub cwd: Option<PathBuf>,
    pub pid: Option<u32>,
    pub process_group_id: Option<u32>,
    pub elapsed_secs: u64,
    pub state: TrackedProcessState,
    pub live_pids: Vec<u32>,
}

fn registry() -> &'static Mutex<ProcessRegistry> {
    PROCESS_REGISTRY.get_or_init(|| Mutex::new(ProcessRegistry::default()))
}

pub fn register_process(
    tool_name: impl Into<String>,
    command: impl Into<String>,
    cwd: Option<&Path>,
    pid: Option<u32>,
    process_group_id: Option<u32>,
) -> u64 {
    let id = NEXT_PROCESS_ID.fetch_add(1, Ordering::Relaxed);
    let command = command.into();
    let task = task_ledger::begin(TaskKind::Shell, command.clone());
    let record = ProcessRecord {
        id,
        tool_name: tool_name.into(),
        command,
        cwd: cwd.map(Path::to_path_buf),
        pid,
        process_group_id,
        started_at: Instant::now(),
        finished_at: None,
        exit_code: None,
        task: Some(task),
    };

    if let Ok(mut guard) = registry().lock() {
        guard.records.insert(id, record);
    }
    id
}

/// Wait until the ledger has committed the start of process `id`.
pub async fn process_recorded(id: u64) {
    let pending = registry().lock().ok().and_then(|mut guard| {
        guard
            .records
            .get_mut(&id)
            .and_then(|record| record.task.as_mut())
            .map(TaskGuard::recorded)
    });
    if let Some(pending) = pending
        && let Err(error) = pending.await
    {
        tracing::warn!(error = %format!("{error:#}"), "task ledger: shell start not recorded yet");
    }
}

/// Set the pid and process group once the process has spawned.
pub fn set_process_ids(id: u64, pid: Option<u32>, process_group_id: Option<u32>) {
    if let Ok(mut guard) = registry().lock()
        && let Some(record) = guard.records.get_mut(&id)
    {
        record.pid = pid;
        record.process_group_id = process_group_id;
        if let Some(task) = &record.task {
            task.set_process(pid, process_group_id);
        }
    }
}

pub fn mark_process_finished(id: u64, exit_code: Option<i64>) {
    let (state, reason) = exit_outcome(exit_code);
    close_record(id, exit_code, state, reason.as_deref());
}

pub fn mark_process_timed_out(id: u64) {
    close_record(id, None, TaskState::Failed, Some("timeout"));
}

/// Ledger state for a process exit code.
fn exit_outcome(exit_code: Option<i64>) -> (TaskState, Option<String>) {
    match exit_code {
        Some(0) => (TaskState::Succeeded, None),
        Some(code) => (TaskState::Failed, Some(format!("exit code {code}"))),
        None => (TaskState::Failed, Some("no exit code".to_owned())),
    }
}

/// The first close wins; later calls do nothing.
fn close_record(id: u64, exit_code: Option<i64>, state: TaskState, reason: Option<&str>) {
    let task = registry().lock().ok().and_then(|mut guard| {
        let record = guard.records.get_mut(&id)?;
        if record.finished_at.is_some() {
            return None;
        }
        record.finished_at = Some(Instant::now());
        record.exit_code = exit_code;
        record.task.take()
    });
    if let Some(task) = task {
        task.finish(state, reason);
    }
}

/// Marks the record and its task cancelled when dropped before the process
/// is marked finished, for example when a cancelled turn drops its future.
#[derive(Debug)]
#[must_use = "dropping the guard marks the process cancelled"]
pub struct ProcessCancelGuard(u64);

pub fn cancel_on_drop(id: u64) -> ProcessCancelGuard {
    ProcessCancelGuard(id)
}

impl Drop for ProcessCancelGuard {
    fn drop(&mut self) {
        close_record(self.0, None, TaskState::Cancelled, Some("dropped"));
    }
}

pub fn tracked_processes_for_turn_end() -> Vec<TrackedProcessSnapshot> {
    let Ok(mut guard) = registry().lock() else {
        return Vec::new();
    };

    let mut snapshots = Vec::new();
    let mut removable = Vec::new();

    for (id, record) in &guard.records {
        let live_pids = record
            .process_group_id
            .map(live_pids_for_process_group)
            .unwrap_or_default();

        let state = if record.finished_at.is_none() {
            Some(TrackedProcessState::Active)
        } else if !live_pids.is_empty() {
            Some(TrackedProcessState::Lingering)
        } else {
            None
        };

        let Some(state) = state else {
            removable.push(*id);
            continue;
        };

        snapshots.push(TrackedProcessSnapshot {
            id: record.id,
            tool_name: record.tool_name.clone(),
            command: record.command.clone(),
            cwd: record.cwd.clone(),
            pid: record.pid,
            process_group_id: record.process_group_id,
            elapsed_secs: record.started_at.elapsed().as_secs(),
            state,
            live_pids,
        });
    }

    for id in removable {
        guard.records.remove(&id);
    }

    snapshots.sort_by_key(|snapshot| snapshot.id);
    snapshots
}

#[cfg(unix)]
pub fn terminate_process_group(process_group_id: u32) {
    let Some(kill) = SystemBinary::Kill.path() else {
        return;
    };
    let target = format!("-{process_group_id}");
    let _ = std::process::Command::new(kill)
        .args(["-TERM", &target])
        .status();
}

#[cfg(not(unix))]
pub fn terminate_process_group(_process_group_id: u32) {}

#[cfg(unix)]
fn live_pids_for_process_group(process_group_id: u32) -> Vec<u32> {
    let Some(ps) = SystemBinary::Ps.path() else {
        return Vec::new();
    };
    let output = std::process::Command::new(ps)
        .args(["-o", "pid=", "-g", &process_group_id.to_string()])
        .output();
    let Ok(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect()
}

#[cfg(not(unix))]
fn live_pids_for_process_group(_process_group_id: u32) -> Vec<u32> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_process_is_reported_until_finished() {
        // The registry is global and other tests share it, so look only at `id`.
        let id = register_process("bash", "sleep 10", None, Some(12_345), None);
        let state_of = |id| {
            tracked_processes_for_turn_end()
                .into_iter()
                .find(|snapshot| snapshot.id == id)
                .map(|snapshot| snapshot.state)
        };

        assert_eq!(state_of(id), Some(TrackedProcessState::Active));

        mark_process_finished(id, Some(0));
        assert_eq!(state_of(id), None);
    }

    #[test]
    fn dropped_cancel_guard_ends_active_process() {
        let id = register_process("bash", "sleep 10", None, Some(12_346), None);
        drop(cancel_on_drop(id));

        let snapshots = tracked_processes_for_turn_end();
        assert!(snapshots.iter().all(|snapshot| snapshot.id != id));
    }

    #[test]
    fn exit_code_maps_to_task_state() {
        assert_eq!(exit_outcome(Some(0)), (TaskState::Succeeded, None));
        assert_eq!(
            exit_outcome(Some(1)),
            (TaskState::Failed, Some("exit code 1".to_owned()))
        );
        assert_eq!(exit_outcome(None).0, TaskState::Failed);
    }

    /// The tracker's ledger wiring, against a real temp-dir ledger.
    #[cfg(feature = "sqlite-store")]
    mod ledger {
        use super::*;
        use crate::data::persistence::task_ledger::test_support::TestLedger;

        #[tokio::test]
        async fn tracked_processes_land_in_the_ledger() {
            let ledger = TestLedger::open().await;
            ledger
                .run(async {
                    let ok = register_process("bash", "tracker-ok", None, None, None);
                    process_recorded(ok).await;
                    set_process_ids(ok, Some(4001), Some(4001));
                    mark_process_finished(ok, Some(0));

                    let bad = register_process("bash", "tracker-bad", None, None, None);
                    mark_process_finished(bad, Some(2));

                    let slow = register_process("bash", "tracker-timeout", None, None, None);
                    mark_process_timed_out(slow);

                    let cut = register_process("bash", "tracker-cancel", None, None, None);
                    drop(cancel_on_drop(cut));
                })
                .await;

            let ok = ledger.task_named("tracker-ok").await;
            assert_eq!(
                (ok.kind.as_str(), ok.state.as_str()),
                ("shell", "succeeded")
            );
            assert_eq!(ok.child_pid, Some(4001));
            let bad = ledger.task_named("tracker-bad").await;
            assert_eq!(
                (bad.state.as_str(), bad.terminal_reason.as_deref()),
                ("failed", Some("exit code 2"))
            );
            let slow = ledger.task_named("tracker-timeout").await;
            assert_eq!(
                (slow.state.as_str(), slow.terminal_reason.as_deref()),
                ("failed", Some("timeout"))
            );
            let cut = ledger.task_named("tracker-cancel").await;
            assert_eq!(
                (cut.state.as_str(), cut.terminal_reason.as_deref()),
                ("cancelled", Some("dropped"))
            );
        }

        #[tokio::test]
        async fn a_second_close_does_not_change_the_ledger() {
            let ledger = TestLedger::open().await;
            ledger
                .run(async {
                    let id = register_process("bash", "tracker-twice", None, None, None);
                    let guard = cancel_on_drop(id);
                    mark_process_finished(id, Some(0));
                    mark_process_timed_out(id);
                    drop(guard);
                })
                .await;

            let task = ledger.task_named("tracker-twice").await;
            assert_eq!(
                (task.state.as_str(), task.terminal_reason),
                ("succeeded", None)
            );
        }
    }
}
