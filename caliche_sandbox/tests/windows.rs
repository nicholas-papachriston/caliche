#![cfg(all(windows, feature = "sandbox-windows"))]

use caliche_sandbox::traits::SandboxInstance;
use caliche_sandbox::{
    EXEC_OUTPUT_LIMIT, ExecResult, ExecTimeout, NamedPolicy, SandboxBackendType, create_backend,
    policy_to_config,
};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// `ERROR_PRIVILEGE_NOT_HELD`: symlink creation needs Developer Mode or admin.
const ERROR_PRIVILEGE_NOT_HELD: i32 = 1314;

/// Slack for process startup and the post-kill drain on top of the exec timeout.
const MARGIN: Duration = Duration::from_secs(5);

#[expect(clippy::expect_used, reason = "test fixture setup panics with context")]
async fn job_instance(workspace: &Path, memory_mb: Option<u64>) -> Box<dyn SandboxInstance> {
    let mut config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        workspace,
        None,
        SandboxBackendType::WindowsJob {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );
    if let Some(memory_mb) = memory_mb {
        config.resource_limits.memory_mb = memory_mb;
    }
    let backend = create_backend(&config).await.expect("windows job backend");
    backend
        .create(&config, workspace)
        .await
        .expect("windows job instance")
}

#[expect(clippy::expect_used, reason = "test fixture setup panics with context")]
fn canonical_tempdir() -> (TempDir, std::path::PathBuf) {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().canonicalize().expect("canonical tempdir");
    (dir, path)
}

/// PowerShell needs the host environment; the backend clears it otherwise.
/// The parent environment, with `cmd.exe`'s current-directory program lookup
/// off. Fixtures are run as `.\name.cmd`, so the tests pass on hosts that
/// set this for hardening.
fn host_env() -> HashMap<String, String> {
    let mut env: HashMap<String, String> = std::env::vars().collect();
    env.insert(
        "NoDefaultCurrentDirectoryInExePath".to_string(),
        "1".to_string(),
    );
    env
}

#[expect(clippy::expect_used, reason = "test fixture setup panics with context")]
fn junction(link: &Path, target: &Path) {
    let status = Command::new("cmd")
        .arg("/C")
        .arg("mklink")
        .arg("/J")
        .arg(link)
        .arg(target)
        .status()
        .expect("run mklink /J");
    assert!(status.success(), "mklink /J failed: {status}");
}

/// Create a file or directory symlink, or return false when the host does not
/// grant the symlink privilege.
#[expect(
    clippy::print_stderr,
    clippy::panic,
    reason = "test helper reports a skipped case and fails setup with context"
)]
fn try_symlink(link: &Path, target: &Path, dir: bool) -> bool {
    let made = if dir {
        std::os::windows::fs::symlink_dir(target, link)
    } else {
        std::os::windows::fs::symlink_file(target, link)
    };
    match made {
        Ok(()) => true,
        Err(err) if err.raw_os_error() == Some(ERROR_PRIVILEGE_NOT_HELD) => {
            eprintln!(
                "skipping symlink case: this host does not permit symlink creation \
                 (enable Developer Mode or run elevated)"
            );
            false
        }
        Err(err) => panic!("create symlink '{}': {err}", link.display()),
    }
}

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn process_alive(pid: u32) -> bool {
    let output = Command::new("tasklist")
        .args(["/FI", &format!("PID eq {pid}"), "/NH", "/FO", "CSV"])
        .output()
        .expect("run tasklist");
    String::from_utf8_lossy(&output.stdout).contains(&format!(",\"{pid}\","))
}

const ALLOCATE: &str = "$ErrorActionPreference = 'Stop'; \
     $buffer = [byte[]]::new(BYTES); $buffer[$buffer.Length - 1] = 1; exit 0";

#[tokio::test]
async fn job_memory_limit_stops_allocation_over_the_limit() {
    let (_dir, workspace) = canonical_tempdir();
    let instance = job_instance(&workspace, Some(256)).await;
    let env = host_env();
    let run = |bytes: &'static str| {
        let script = ALLOCATE.replace("BYTES", bytes);
        let instance = &instance;
        let env = &env;
        async move {
            instance
                .exec(
                    &[
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        script.as_str(),
                    ],
                    None,
                    Some(env),
                    Duration::from_secs(120),
                )
                .await
                .expect("exec powershell")
        }
    };

    let under = run("16MB").await;
    assert_eq!(
        under.exit_code, 0,
        "16 MB under a 256 MB job limit must succeed: {under:?}"
    );

    let over = run("1GB").await;
    assert_ne!(
        over.exit_code, 0,
        "1 GB over a 256 MB job limit must fail: {over:?}"
    );
}

/// Run `command` on its own thread and fail if exec does not return within
/// `timeout` plus a generous guard, so a stalled exec fails the test instead
/// of hanging the suite. Returns the instance so its job stays open.
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper panics with context"
)]
fn exec_guarded(
    workspace: &Path,
    command: &'static [&'static str],
    timeout: Duration,
) -> (
    Duration,
    anyhow::Result<ExecResult>,
    Box<dyn SandboxInstance>,
) {
    let workspace = workspace.to_path_buf();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        let outcome = runtime.block_on(async {
            let instance = job_instance(&workspace, None).await;
            let started = Instant::now();
            let result = instance
                .exec(command, None, Some(&host_env()), timeout)
                .await;
            (started.elapsed(), result, instance)
        });
        let _ = tx.send(outcome);
    });
    let guard = timeout + Duration::from_secs(60);
    rx.recv_timeout(guard).unwrap_or_else(|_| {
        panic!("exec of {command:?} did not return within {guard:?}; output drain is unbounded")
    })
}

#[expect(clippy::panic, reason = "test helper panics with context")]
fn grandchild_pid(pid_file: &Path) -> u32 {
    fs::read_to_string(pid_file)
        .ok()
        .and_then(|text| text.trim().parse::<u32>().ok())
        .unwrap_or_else(|| panic!("grandchild never wrote its pid to {}", pid_file.display()))
}

async fn assert_exits_within(pid: u32, wait: Duration, message: &str) {
    let deadline = Instant::now() + wait;
    while process_alive(pid) {
        assert!(Instant::now() < deadline, "grandchild {pid} {message}");
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::test]
async fn exec_timeout_kills_processes_left_running_in_the_job() {
    let (_dir, workspace) = canonical_tempdir();
    // `start /B` leaves a grandchild in the job while `cmd` runs until the timeout.
    fs::write(
        workspace.join("spawn.cmd"),
        "@echo off\r\n\
         start \"\" /B powershell -NoProfile -NonInteractive -Command \
         \"Set-Content -Path grandchild.pid -Value $PID; Start-Sleep -Seconds 600\"\r\n\
         ping -n 600 127.0.0.1 >NUL\r\n",
    )
    .expect("write spawn script");

    let timeout = Duration::from_secs(15);
    let (elapsed, timed_out, _instance) =
        exec_guarded(&workspace, &["cmd", "/C", r".\spawn.cmd"], timeout);
    let err = timed_out.expect_err("spawn script should run until timeout");
    assert!(
        err.downcast_ref::<ExecTimeout>().is_some(),
        "expected ExecTimeout, got {err:#}"
    );
    assert!(
        elapsed < timeout + MARGIN,
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );

    let pid = grandchild_pid(&workspace.join("grandchild.pid"));
    assert_exits_within(pid, Duration::from_secs(20), "survived the exec timeout").await;
}

#[tokio::test]
async fn grandchild_holding_the_output_pipes_cannot_stall_exec_past_its_timeout() {
    let (_dir, workspace) = canonical_tempdir();
    // `cmd` exits right after `start /B`. The grandchild inherits the exec
    // output pipes and keeps them open for 600 s.
    fs::write(
        workspace.join("stall.cmd"),
        "@echo off\r\n\
         echo hi\r\n\
         start \"\" /B powershell -NoProfile -NonInteractive -Command \
         \"Set-Content -Path grandchild.pid -Value $PID; Start-Sleep -Seconds 600\"\r\n",
    )
    .expect("write stall script");

    let timeout = Duration::from_secs(15);
    let (elapsed, result, _instance) =
        exec_guarded(&workspace, &["cmd", "/C", r".\stall.cmd"], timeout);
    assert!(
        elapsed < timeout + MARGIN,
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );
    let err = result.expect_err("held output pipes must end in a timeout");
    let timed_out = err
        .downcast_ref::<ExecTimeout>()
        .unwrap_or_else(|| panic!("expected ExecTimeout, got {err:#}"));
    assert_eq!(err.to_string(), "command timed out after 15s");
    assert_eq!(timed_out.stdout, "hi\r\n");

    let pid = grandchild_pid(&workspace.join("grandchild.pid"));
    assert_exits_within(pid, Duration::from_secs(20), "survived the exec timeout").await;
}

#[tokio::test]
async fn exec_timeout_leaves_a_concurrent_exec_on_the_same_instance_running() {
    let (_dir, workspace) = canonical_tempdir();
    let instance = job_instance(&workspace, None).await;
    let env = host_env();
    let slow = instance.exec(
        &["cmd", "/C", "ping -n 11 127.0.0.1 >NUL & echo done"],
        None,
        Some(&env),
        Duration::from_secs(120),
    );
    let stuck = async {
        // Start after the slow exec is in its job.
        tokio::time::sleep(Duration::from_secs(2)).await;
        instance
            .exec(
                &["cmd", "/C", "ping -n 600 127.0.0.1 >NUL"],
                None,
                Some(&env),
                Duration::from_secs(3),
            )
            .await
    };
    let (slow, stuck) = tokio::join!(slow, stuck);

    let err = stuck.expect_err("ping -n 600 must time out");
    assert_eq!(err.to_string(), "command timed out after 3s");
    let slow = slow.expect("the slow exec must not be killed by the other exec's timeout");
    assert_eq!((slow.stdout.as_str(), slow.exit_code), ("done\r\n", 0));
}

#[tokio::test]
async fn command_that_exits_returns_its_output_and_exit_code() {
    let (_dir, workspace) = canonical_tempdir();
    fs::write(
        workspace.join("exit.cmd"),
        "@echo off\r\necho hi\r\n>&2 echo err\r\nexit /B 3\r\n",
    )
    .expect("write exit script");

    let (elapsed, result, _instance) = exec_guarded(
        &workspace,
        &["cmd", "/C", r".\exit.cmd"],
        Duration::from_secs(30),
    );
    let result = result.expect("exec");
    assert_eq!(
        (
            result.stdout.as_str(),
            result.stderr.as_str(),
            result.exit_code
        ),
        ("hi\r\n", "err\r\n", 3)
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "a finished command took {elapsed:?}"
    );
}

#[tokio::test]
async fn grandchild_of_a_finished_exec_dies_with_the_job() {
    let (_dir, workspace) = canonical_tempdir();
    let pid_file = workspace.join("grandchild.pid");
    let instance = job_instance(&workspace, None).await;

    // `Start-Process` without `-NoNewWindow` launches through ShellExecute, which
    // does not inherit handles. A grandchild started by `cmd start` inherits the
    // exec output pipes, even with its own stdio sent to NUL, so the output never
    // reaches EOF and exec cannot return while the grandchild runs.
    let started = Instant::now();
    let spawned = instance
        .exec(
            &[
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Start-Process powershell -WindowStyle Hidden -ArgumentList \
                 '-NoProfile','-NonInteractive','-Command',\
                 'Set-Content -Path grandchild.pid -Value $PID; Start-Sleep -Seconds 120'",
            ],
            None,
            Some(&host_env()),
            Duration::from_secs(30),
        )
        .await
        .expect("exec powershell Start-Process");
    assert_eq!(spawned.exit_code, 0, "Start-Process failed: {spawned:?}");
    assert!(
        started.elapsed() < Duration::from_secs(30),
        "exec waited {:?} for the grandchild to release the output pipes",
        started.elapsed()
    );

    let deadline = Instant::now() + Duration::from_secs(60);
    let pid = loop {
        let parsed = fs::read_to_string(&pid_file)
            .ok()
            .and_then(|text| text.trim().parse::<u32>().ok());
        if let Some(pid) = parsed {
            break pid;
        }
        assert!(Instant::now() < deadline, "grandchild never wrote its pid");
        tokio::time::sleep(Duration::from_millis(200)).await;
    };
    assert!(
        process_alive(pid),
        "grandchild {pid} should outlive the finished exec while the job is open"
    );

    drop(instance);

    let deadline = Instant::now() + Duration::from_secs(20);
    while process_alive(pid) {
        assert!(
            Instant::now() < deadline,
            "grandchild {pid} escaped the job and survived closing it"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::test]
async fn plain_files_read_and_write_below_the_workspace() {
    let (_dir, workspace) = canonical_tempdir();
    fs::create_dir_all(workspace.join("src")).expect("src dir");
    fs::write(workspace.join("src").join("lib.rs"), b"existing").expect("fixture");
    let instance = job_instance(&workspace, None).await;

    let bytes = instance.read_file("src/lib.rs").await.expect("read");
    assert_eq!(bytes, b"existing");

    instance
        .write_file("new/nested/out.txt", b"hello")
        .await
        .expect("write creates parents");
    instance
        .write_file("src/lib.rs", b"short")
        .await
        .expect("overwrite");
    assert_eq!(
        fs::read(workspace.join("new").join("nested").join("out.txt")).expect("on disk"),
        b"hello"
    );
    assert_eq!(
        instance.read_file("src/lib.rs").await.expect("reread"),
        b"short",
        "overwrite must truncate"
    );
}

#[tokio::test]
async fn junction_to_outside_is_refused_for_read_and_write() {
    let (_dir, workspace) = canonical_tempdir();
    let (_outside_dir, outside) = canonical_tempdir();
    fs::write(outside.join("secret.txt"), b"outside").expect("outside secret");
    junction(&workspace.join("escape"), &outside);
    let instance = job_instance(&workspace, None).await;

    instance
        .read_file("escape/secret.txt")
        .await
        .expect_err("read through junction");
    instance
        .write_file("escape/pwned.txt", b"pwned")
        .await
        .expect_err("write through junction");
    assert!(!outside.join("pwned.txt").exists());
}

#[tokio::test]
async fn dangling_junction_parent_is_refused_as_a_reparse_point() {
    let (_dir, workspace) = canonical_tempdir();
    let (_outside_dir, outside) = canonical_tempdir();
    let gone = outside.join("gone");
    fs::create_dir(&gone).expect("junction target");
    junction(&workspace.join("j"), &gone);
    fs::remove_dir(&gone).expect("leave the junction dangling");
    let instance = job_instance(&workspace, None).await;

    // Validation cannot resolve the dangling junction, so the open must refuse it.
    let read = instance.read_file("j/x.txt").await.expect_err("read");
    assert!(
        format!("{read:#}").contains("reparse point"),
        "unexpected error: {read:#}"
    );
    let write = instance
        .write_file("j/x.txt", b"pwned")
        .await
        .expect_err("write");
    assert!(
        format!("{write:#}").contains("reparse point"),
        "unexpected error: {write:#}"
    );
    assert!(
        !gone.exists(),
        "write must not recreate the junction target"
    );
}

#[tokio::test]
async fn symlinks_are_refused_for_read_and_write() {
    let (_dir, workspace) = canonical_tempdir();
    let (_outside_dir, outside) = canonical_tempdir();
    let secret = outside.join("secret.txt");
    fs::write(&secret, b"outside").expect("outside secret");
    if !try_symlink(&workspace.join("link.txt"), &secret, false) {
        return;
    }
    let missing = outside.join("missing.txt");
    assert!(try_symlink(
        &workspace.join("dangling.txt"),
        &missing,
        false
    ));
    let missing_dir = outside.join("missing-dir");
    assert!(try_symlink(&workspace.join("dirlink"), &missing_dir, true));
    let instance = job_instance(&workspace, None).await;

    instance
        .read_file("link.txt")
        .await
        .expect_err("read through file symlink");
    instance
        .write_file("link.txt", b"pwned")
        .await
        .expect_err("write through file symlink");
    assert_eq!(fs::read(&secret).expect("secret"), b"outside");

    // Dangling links pass validation, so these reach the open itself.
    for path in ["dangling.txt", "dirlink/x.txt"] {
        let read = instance.read_file(path).await.expect_err("read");
        assert!(
            format!("{read:#}").contains("reparse point"),
            "{path}: unexpected error: {read:#}"
        );
        let write = instance
            .write_file(path, b"pwned")
            .await
            .expect_err("write");
        assert!(
            format!("{write:#}").contains("reparse point"),
            "{path}: unexpected error: {write:#}"
        );
    }
    assert!(!missing.exists());
    assert!(!missing_dir.exists());
}

/// A parent directory swapped for a junction between validation and open
/// must never redirect a read or a write outside the workspace.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parent_swapped_for_junction_during_io_never_escapes() {
    let (_dir, workspace) = canonical_tempdir();
    let (_outside_dir, outside) = canonical_tempdir();
    fs::write(outside.join("secret.txt"), b"outside").expect("outside secret");
    let real = workspace.join("real");
    fs::create_dir(&real).expect("real dir");
    fs::write(real.join("secret.txt"), b"inside").expect("inside secret");
    let parked_junction = workspace.join("jn");
    junction(&parked_junction, &outside);
    let swapped = workspace.join("d");

    let stop = Arc::new(AtomicBool::new(false));
    let flipper = std::thread::spawn({
        let stop = Arc::clone(&stop);
        move || {
            // Renames fail while the sandbox holds the directory; that is fine.
            while !stop.load(Ordering::Relaxed) {
                let _ = fs::rename(&real, &swapped);
                let _ = fs::rename(&swapped, &real);
                let _ = fs::rename(&parked_junction, &swapped);
                let _ = fs::rename(&swapped, &parked_junction);
            }
        }
    });

    let instance = job_instance(&workspace, None).await;
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut reads = 0_u32;
    while Instant::now() < deadline {
        if let Ok(bytes) = instance.read_file("d/secret.txt").await {
            assert_eq!(bytes, b"inside", "read escaped through a swapped junction");
            reads += 1;
        }
        let _ = instance.write_file("d/w.txt", b"x").await;
        assert!(
            !outside.join("w.txt").exists(),
            "write escaped through a swapped junction"
        );
    }
    stop.store(true, Ordering::Relaxed);
    flipper.join().expect("flipper thread");
    eprintln!("{reads} reads hit the real directory");
}

#[tokio::test]
async fn output_flood_is_capped_and_exec_returns_at_its_timeout() {
    let (_dir, workspace) = canonical_tempdir();
    let timeout = Duration::from_secs(15);
    let (elapsed, result, _instance) = exec_guarded(
        &workspace,
        &[
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[Console]::Error.Write('err'); while ($true) { [Console]::Out.Write('y' * 1MB) }",
        ],
        timeout,
    );

    assert!(
        elapsed < timeout + MARGIN,
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );
    let err = result.expect_err("an endless write must time out");
    let timed_out = err
        .downcast_ref::<ExecTimeout>()
        .unwrap_or_else(|| panic!("expected ExecTimeout, got {err:#}"));
    assert_eq!(timed_out.stdout, "y".repeat(EXEC_OUTPUT_LIMIT));
    assert!(
        timed_out.stdout_truncated,
        "stdout past the cap must be flagged"
    );
    assert_eq!(
        (timed_out.stderr.as_str(), timed_out.stderr_truncated),
        ("err", false)
    );
}

#[tokio::test]
async fn output_past_the_cap_is_truncated_and_the_command_finishes() {
    let (_dir, workspace) = canonical_tempdir();
    let (_, result, _instance) = exec_guarded(
        &workspace,
        &[
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[Console]::Out.Write('a' * 12MB); [Console]::Error.Write('done')",
        ],
        Duration::from_secs(60),
    );

    let result = result.expect("a bounded write must finish, not block on a full pipe");
    assert_eq!(result.stdout, "a".repeat(EXEC_OUTPUT_LIMIT));
    assert!(
        result.stdout_truncated,
        "stdout past the cap must be flagged"
    );
    assert_eq!(
        (
            result.stderr.as_str(),
            result.stderr_truncated,
            result.exit_code
        ),
        ("done", false, 0)
    );
}

#[tokio::test]
async fn output_under_the_cap_is_returned_in_full() {
    let (_dir, workspace) = canonical_tempdir();
    let (_, result, _instance) = exec_guarded(
        &workspace,
        &[
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "[Console]::Out.Write('a' * 1000)",
        ],
        Duration::from_secs(60),
    );

    let result = result.expect("exec");
    assert_eq!(result.stdout, "a".repeat(1000));
    assert_eq!(
        (
            result.stdout_truncated,
            result.stderr.as_str(),
            result.stderr_truncated
        ),
        (false, "", false)
    );
}
