use caliche_sandbox::backends::noop::NoopBackend;
use caliche_sandbox::config::CPU_CAP_REQUIRES_WINDOWS_JOB;
use caliche_sandbox::traits::SandboxBackend;
use caliche_sandbox::{NamedPolicy, SandboxBackendType, policy_to_config};
use std::fs;
use std::time::Duration;
use tempfile::TempDir;

#[cfg(unix)]
const ECHO_OK: &[&str] = &["sh", "-c", "echo ok"];
#[cfg(windows)]
const ECHO_OK: &[&str] = &["cmd", "/C", "echo ok"];

#[cfg(any(unix, windows))]
#[tokio::test]
async fn noop_create_rejects_a_cpu_cap() {
    let workspace = tempfile::tempdir().expect("workspace");
    let mut config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace.path(),
        None,
        SandboxBackendType::Noop,
    );
    config.resource_limits.cpu_rate_hard_cap_percent = Some(20);
    let err = NoopBackend
        .create(&config, workspace.path())
        .await
        .expect_err("noop create must reject a cpu cap");
    assert_eq!(err.to_string(), CPU_CAP_REQUIRES_WINDOWS_JOB);
}

#[tokio::test]
async fn exec_echo_ok() {
    let workspace = tempfile::tempdir().expect("workspace");
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace.path(),
        None,
        SandboxBackendType::Noop,
    );
    let backend = NoopBackend;
    let instance = backend
        .create(&config, workspace.path())
        .await
        .expect("noop instance");
    let result = instance
        .exec(ECHO_OK, None, None, Duration::from_secs(5))
        .await
        .expect("exec");
    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.trim().contains("ok"));
}

#[tokio::test]
async fn noop_workspace_write_rejects_absolute_outside_read() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let backend = NoopBackend;
    let instance = backend
        .create(&config, &workspace)
        .await
        .expect("noop instance");
    let err = instance
        .read_file("/etc/hosts")
        .await
        .expect_err("absolute outside read");
    assert!(
        err.to_string().contains("outside sandbox-allowed"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn noop_workspace_write_allows_workspace_file() {
    let dir = TempDir::new().expect("tempdir");
    let workspace = dir.path().canonicalize().expect("canonical workspace");
    fs::write(workspace.join("ok.txt"), b"hello").expect("write");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );
    let backend = NoopBackend;
    let instance = backend
        .create(&config, &workspace)
        .await
        .expect("noop instance");
    let bytes = instance.read_file("ok.txt").await.expect("read");
    assert_eq!(bytes, b"hello");
}

/// Run `command` on its own thread and fail if exec does not return within
/// `timeout` plus a guard, so a stalled exec fails instead of hanging.
#[cfg(unix)]
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper panics with context"
)]
fn exec_guarded(
    command: &[&str],
    timeout: Duration,
) -> (Duration, anyhow::Result<caliche_sandbox::ExecResult>) {
    let owned: Vec<String> = command.iter().map(|arg| (*arg).to_string()).collect();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let command: Vec<&str> = owned.iter().map(String::as_str).collect();
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        let outcome = runtime.block_on(async {
            let workspace = tempfile::tempdir().expect("workspace");
            let config = policy_to_config(
                NamedPolicy::ReadOnly,
                workspace.path(),
                None,
                SandboxBackendType::Noop,
            );
            let instance = NoopBackend
                .create(&config, workspace.path())
                .await
                .expect("instance");
            let started = std::time::Instant::now();
            let result = instance.exec(&command, None, None, timeout).await;
            (started.elapsed(), result)
        });
        let _ = tx.send(outcome);
    });
    let guard = timeout + Duration::from_secs(30);
    rx.recv_timeout(guard).unwrap_or_else(|_| {
        panic!("exec did not return within {guard:?}; output drain is unbounded")
    })
}

#[cfg(unix)]
#[test]
fn output_flood_is_capped_and_exec_returns_at_its_timeout() {
    let timeout = Duration::from_secs(4);
    let (elapsed, result) = exec_guarded(
        &["/bin/sh", "-c", "echo err >&2; exec /usr/bin/yes"],
        timeout,
    );

    assert!(
        elapsed < timeout + Duration::from_secs(2),
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );
    let err = result.expect_err("yes must time out");
    let timed_out = err
        .downcast_ref::<caliche_sandbox::ExecTimeout>()
        .unwrap_or_else(|| panic!("expected ExecTimeout, got {err:#}"));
    assert_eq!(timed_out.stdout.len(), caliche_sandbox::EXEC_OUTPUT_LIMIT);
    assert!(
        timed_out.stdout_truncated,
        "stdout past the cap must be flagged"
    );
    assert_eq!(
        (timed_out.stderr.as_str(), timed_out.stderr_truncated),
        ("err\n", false)
    );
}

#[cfg(unix)]
#[test]
fn output_under_the_cap_is_returned_in_full() {
    let (_, result) = exec_guarded(
        &["/bin/sh", "-c", "printf %01000d 0; echo done >&2"],
        Duration::from_secs(10),
    );

    let result = result.expect("exec");
    assert_eq!(result.stdout, "0".repeat(1000));
    assert_eq!(
        (
            result.stdout_truncated,
            result.stderr.as_str(),
            result.stderr_truncated
        ),
        (false, "done\n", false)
    );
}
