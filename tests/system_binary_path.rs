//! A `kill`, `ps`, or `hostname` planted early on `PATH` must never run.
//!
//! The planted tools only append their name to a marker file. Every scenario
//! runs inside one test because `PATH` is process-global.

#![cfg(all(unix, feature = "sqlite-store"))]

use caliche::agent::tools::process_tracker::terminate_process_group;
use caliche::persistence::task_ledger::{TASKS_DB_FILE, TaskLedger};
use caliche::security::SystemBinary;
use sqlx::SqlitePool;
use sqlx::sqlite::SqliteConnectOptions;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;

const PLANTED: [&str; 3] = ["kill", "ps", "hostname"];
const SIGTERM: i32 = 15;

fn plant(dir: &Path, marker: &Path) -> std::io::Result<()> {
    for name in PLANTED {
        let script = dir.join(name);
        std::fs::write(
            &script,
            format!("#!/bin/sh\necho {name} >> '{}'\n", marker.display()),
        )?;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))?;
    }
    Ok(())
}

fn planted_runs(marker: &Path) -> String {
    std::fs::read_to_string(marker).unwrap_or_default()
}

/// A pid that has exited and been reaped.
fn dead_pid() -> std::io::Result<i64> {
    let mut child = std::process::Command::new("/bin/sh")
        .args(["-c", "exit 0"])
        .spawn()?;
    let pid = i64::from(child.id());
    child.wait()?;
    Ok(pid)
}

async fn insert_running(
    pool: &SqlitePool,
    id: &str,
    owner_pid: i64,
    host: &str,
) -> sqlx::Result<()> {
    sqlx::query(
        "INSERT INTO tasks (id, kind, name, state, owner_pid, owner_instance, host, created_at, updated_at) \
         VALUES (?, 'shell', 'sleep 10', 'running', ?, 'gone', ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
    )
    .bind(id)
    .bind(owner_pid)
    .bind(host)
    .execute(pool)
    .await?;
    Ok(())
}

#[tokio::test]
async fn planted_tools_on_path_never_run() -> anyhow::Result<()> {
    let planted = tempfile::tempdir().unwrap();
    let marker = planted.path().join("ran");
    plant(planted.path(), &marker)?;
    let path = format!(
        "{}:{}",
        planted.path().display(),
        std::env::var("PATH").unwrap_or_default()
    );
    // SAFETY: the only test in this binary, set before any thread reads PATH.
    #[expect(
        unsafe_code,
        clippy::disallowed_methods,
        reason = "the test needs a planted tool first on PATH"
    )]
    unsafe {
        std::env::set_var("PATH", path);
    }

    // The plant works: a bare lookup runs it.
    std::process::Command::new("kill")
        .args(["-0", "1"])
        .status()
        .unwrap();
    assert_eq!(planted_runs(&marker), "kill\n");
    std::fs::remove_file(&marker).unwrap();

    // process_tracker: the group dies by SIGTERM from the system kill.
    let mut child = std::process::Command::new("/bin/sleep")
        .arg("30")
        .process_group(0)
        .spawn()
        .unwrap();
    terminate_process_group(child.id());
    assert_eq!(child.wait().unwrap().signal(), Some(SIGTERM));

    // task_ledger: reopening checks the dead owner with the system kill and
    // reads the host with the system hostname. The planted kill exits 0,
    // which would keep the row running.
    let host = std::process::Command::new(SystemBinary::Hostname.path().unwrap())
        .output()
        .unwrap();
    let host = String::from_utf8_lossy(&host.stdout).trim().to_owned();
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join(TASKS_DB_FILE);
    drop(TaskLedger::open(&db).await.unwrap());
    let pool = SqlitePool::connect_with(SqliteConnectOptions::new().filename(&db))
        .await
        .unwrap();
    insert_running(&pool, "gone-owner", dead_pid()?, &host).await?;
    drop(TaskLedger::open(&db).await.unwrap());
    let state: String = sqlx::query_scalar("SELECT state FROM tasks WHERE id = 'gone-owner'")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(state, "host_lost");

    assert_eq!(planted_runs(&marker), "");
    Ok(())
}
