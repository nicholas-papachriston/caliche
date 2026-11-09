#![cfg(all(feature = "sandbox-linux", target_os = "linux"))]

use caliche_sandbox::backends::linux::LinuxLandlockBackend;
use caliche_sandbox::config::CPU_CAP_REQUIRES_WINDOWS_JOB;
use caliche_sandbox::traits::{SandboxBackend, SandboxInstance};
use caliche_sandbox::{
    EXEC_OUTPUT_LIMIT, ExecResult, ExecTimeout, NamedPolicy, SandboxBackendType, create_backend,
    policy_to_config,
};
use std::sync::mpsc;
use std::time::{Duration, Instant};

include!("common/mod.rs");

/// Slack for process startup and the post-kill drain on top of the exec timeout.
const MARGIN: Duration = Duration::from_secs(2);

/// Run `command` through the Landlock backend on its own thread and fail if
/// exec does not return within `timeout` plus a generous guard, so a stalled
/// exec fails the test instead of hanging the suite. Returns the instance so
/// processes the exec left running are not killed with it yet. Returns `None`
/// when the kernel has no Landlock, unless the tests require it.
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper panics with context"
)]
fn exec(
    command: &[&str],
    timeout: Duration,
) -> Option<(
    Duration,
    anyhow::Result<ExecResult>,
    Box<dyn SandboxInstance>,
)> {
    let owned: Vec<String> = command.iter().map(|arg| (*arg).to_string()).collect();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let command: Vec<&str> = owned.iter().map(String::as_str).collect();
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        let outcome = runtime.block_on(async {
            let workspace = std::env::temp_dir();
            let config = policy_to_config(
                NamedPolicy::ReadOnly,
                &workspace,
                None,
                SandboxBackendType::LinuxLandlock {
                    policy: NamedPolicy::ReadOnly,
                },
            );
            let backend = create_backend(&config).await.expect("landlock backend");
            let instance = match backend.create(&config, &workspace).await {
                Ok(instance) => instance,
                Err(err) if err.to_string().contains("not available") => return None,
                Err(err) => panic!("landlock instance: {err:#}"),
            };
            let started = Instant::now();
            let result = instance.exec(&command, None, None, timeout).await;
            Some((started.elapsed(), result, instance))
        });
        let _ = tx.send(outcome);
    });
    let guard = timeout + Duration::from_secs(30);
    let outcome = rx.recv_timeout(guard).unwrap_or_else(|_| {
        panic!("exec of {command:?} did not return within {guard:?}; output drain is unbounded")
    });
    if outcome.is_none() {
        require_landlock();
    }
    outcome
}

/// True while `pid` exists and is not a zombie.
fn running(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/stat"))
        .ok()
        .and_then(|stat| {
            let state = stat.rsplit_once(") ")?.1.chars().next()?;
            Some(state != 'Z')
        })
        .unwrap_or(false)
}

#[test]
fn background_process_holding_stdout_cannot_stall_exec_past_its_timeout() {
    let timeout = Duration::from_secs(2);
    let Some((elapsed, result, _instance)) = exec(
        &["bash", "-c", "echo hi; sleep 600 <&0 & echo $! >&2"],
        timeout,
    ) else {
        return;
    };

    assert!(
        elapsed < timeout + MARGIN,
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );
    let err = result.expect_err("a held stdout pipe must end in a timeout");
    let timed_out = err
        .downcast_ref::<ExecTimeout>()
        .unwrap_or_else(|| panic!("expected ExecTimeout, got {err:#}"));
    assert_eq!(err.to_string(), "command timed out after 2s");
    assert_eq!(timed_out.stdout, "hi\n");
    let pid: u32 = timed_out
        .stderr
        .trim()
        .parse()
        .unwrap_or_else(|_| panic!("stderr should hold the sleep pid: {:?}", timed_out.stderr));
    let deadline = Instant::now() + Duration::from_secs(5);
    while running(pid) {
        assert!(
            Instant::now() < deadline,
            "background sleep {pid} must be killed with the exec process group"
        );
        #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// With `CLS_REQUIRE_CGROUP_TESTS=1` the cgroup tests below must not be
/// compiled as ignored, so a runner that lost cgroup v2 fails instead of
/// skipping them.
#[test]
fn cgroup_tests_are_not_skipped_when_required() {
    let required = std::env::var("CLS_REQUIRE_CGROUP_TESTS").is_ok_and(|value| value == "1");
    assert!(
        !required || cfg!(sandbox_exec_cgroup_available),
        "CLS_REQUIRE_CGROUP_TESTS=1, but the build found no writable cgroup v2 with cgroup.kill, so the cgroup tests are skipped"
    );
}

#[test]
#[cfg_attr(
    not(sandbox_exec_cgroup_available),
    ignore = "skipped: no writable cgroup v2 with cgroup.kill"
)]
fn process_in_a_new_session_is_killed_at_the_timeout() {
    let timeout = Duration::from_secs(2);
    // The grandchild leaves the exec process group with `setsid` and closes
    // its stdio, as a daemon would. `<&0` keeps the background job on the
    // already-open stdin instead of opening a new one.
    let Some((elapsed, result, _instance)) = exec(
        &[
            "bash",
            "-c",
            "setsid sh -c 'echo $$ >&2; exec sleep 600 <&- >&- 2>&-' <&0 & sleep 600",
        ],
        timeout,
    ) else {
        return;
    };

    assert!(
        elapsed < timeout + MARGIN,
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );
    let err = result.expect_err("sleep 600 must time out");
    let timed_out = err
        .downcast_ref::<ExecTimeout>()
        .unwrap_or_else(|| panic!("expected ExecTimeout, got {err:#}"));
    let pid: u32 = timed_out.stderr.trim().parse().unwrap_or_else(|_| {
        panic!(
            "stderr should hold the grandchild pid: {:?}",
            timed_out.stderr
        )
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while running(pid) {
        assert!(
            Instant::now() < deadline,
            "grandchild {pid} in a new session survived the exec timeout"
        );
        #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
        std::thread::sleep(Duration::from_millis(50));
    }
}

#[test]
#[cfg_attr(
    not(sandbox_exec_cgroup_available),
    ignore = "skipped: no writable cgroup v2 with cgroup.kill"
)]
fn process_in_a_new_session_left_by_a_finished_exec_dies_with_the_instance() {
    let Some((_, result, instance)) = exec(
        &[
            "bash",
            "-c",
            "setsid sh -c 'echo $$ >&2; exec sleep 600 <&- >&- 2>&-' <&0 &",
        ],
        Duration::from_secs(10),
    ) else {
        return;
    };
    let result = result.expect("exec");
    assert_eq!(result.exit_code, 0);
    let pid: u32 =
        result.stderr.trim().parse().unwrap_or_else(|_| {
            panic!("stderr should hold the grandchild pid: {:?}", result.stderr)
        });
    assert!(
        running(pid),
        "grandchild {pid} should outlive the finished exec while the instance is alive"
    );
    let exec_cgroup = cgroup_dir(pid);
    let instance_cgroup = exec_cgroup.parent().expect("instance cgroup").to_path_buf();

    drop(instance);

    let deadline = Instant::now() + Duration::from_secs(5);
    while running(pid) {
        assert!(
            Instant::now() < deadline,
            "grandchild {pid} in a new session survived dropping the instance"
        );
        #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
        std::thread::sleep(Duration::from_millis(50));
    }
    while instance_cgroup.exists() {
        assert!(
            Instant::now() < deadline,
            "instance cgroup {} was not removed after the instance was dropped",
            instance_cgroup.display()
        );
        #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// The cgroup v2 directory of `pid`.
#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn cgroup_dir(pid: u32) -> std::path::PathBuf {
    let membership =
        std::fs::read_to_string(format!("/proc/{pid}/cgroup")).expect("read /proc/<pid>/cgroup");
    let relative = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .expect("cgroup v2 entry");
    std::path::Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/'))
}

#[test]
fn command_still_running_at_the_timeout_returns_its_partial_output() {
    let timeout = Duration::from_secs(2);
    let Some((elapsed, result, _instance)) =
        exec(&["sh", "-c", "echo hi; echo err >&2; sleep 600"], timeout)
    else {
        return;
    };

    assert!(
        elapsed < timeout + MARGIN,
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );
    let err = result.expect_err("sleep 600 must time out");
    let timed_out = err
        .downcast_ref::<ExecTimeout>()
        .unwrap_or_else(|| panic!("expected ExecTimeout, got {err:#}"));
    assert_eq!(
        (timed_out.stdout.as_str(), timed_out.stderr.as_str()),
        ("hi\n", "err\n")
    );
}

#[test]
fn command_that_exits_returns_its_output_and_exit_code() {
    let Some((elapsed, result, _instance)) = exec(
        &["sh", "-c", "echo hi; echo err >&2; exit 3"],
        Duration::from_secs(10),
    ) else {
        return;
    };

    let result = result.expect("exec");
    assert_eq!(
        (
            result.stdout.as_str(),
            result.stderr.as_str(),
            result.exit_code
        ),
        ("hi\n", "err\n", 3)
    );
    assert!(elapsed < MARGIN, "a finished command took {elapsed:?}");
}

/// Shell that writes `bytes` copies of `byte`, a `tr` operand, to `fd`.
/// `/dev/zero` is outside the read-only policy, so the bytes come from `yes`.
fn repeat_byte(byte: &str, bytes: usize, fd: u8) -> String {
    format!("yes | tr -d '\\n' | tr y '{byte}' | head -c {bytes} >&{fd}")
}

/// Shell that writes `bytes` copies of `a` to stdout, then `done` to stderr.
fn write_a(bytes: usize) -> String {
    format!("{}; echo done >&2", repeat_byte("a", bytes, 1))
}

#[test]
fn output_flood_is_capped_and_exec_returns_at_its_timeout() {
    let timeout = Duration::from_secs(4);
    let Some((elapsed, result, _instance)) = exec(&["sh", "-c", "echo err >&2; exec yes"], timeout)
    else {
        return;
    };

    assert!(
        elapsed < timeout + MARGIN,
        "exec took {elapsed:?} with a {timeout:?} timeout"
    );
    let err = result.expect_err("yes must time out");
    let timed_out = err
        .downcast_ref::<ExecTimeout>()
        .unwrap_or_else(|| panic!("expected ExecTimeout, got {err:#}"));
    assert_eq!(timed_out.stdout.len(), EXEC_OUTPUT_LIMIT);
    assert!(
        timed_out.stdout_truncated,
        "stdout past the cap must be flagged"
    );
    assert_eq!(
        (timed_out.stderr.as_str(), timed_out.stderr_truncated),
        ("err\n", false)
    );
}

#[test]
fn output_past_the_cap_is_truncated_per_stream_and_the_command_finishes() {
    let script = format!("{}; echo out", repeat_byte("e", 3 * EXEC_OUTPUT_LIMIT, 2));
    let Some((_, result, _instance)) = exec(&["sh", "-c", &script], Duration::from_secs(20)) else {
        return;
    };

    let result = result.expect("a bounded write must finish, not block on a full pipe");
    assert_eq!(
        (result.stdout.as_str(), result.stdout_truncated),
        ("out\n", false)
    );
    assert_eq!(result.stderr, "e".repeat(EXEC_OUTPUT_LIMIT));
    assert!(
        result.stderr_truncated,
        "stderr past the cap must be flagged"
    );
    assert_eq!(result.exit_code, 0);
}

#[test]
fn output_exactly_at_the_cap_is_kept_whole() {
    let Some((_, result, _instance)) = exec(
        &["sh", "-c", &write_a(EXEC_OUTPUT_LIMIT)],
        Duration::from_secs(20),
    ) else {
        return;
    };

    let result = result.expect("exec");
    assert_eq!(result.stdout, "a".repeat(EXEC_OUTPUT_LIMIT));
    assert_eq!(
        (
            result.stdout_truncated,
            result.stderr.as_str(),
            result.stderr_truncated
        ),
        (false, "done\n", false)
    );
}

#[test]
fn output_under_the_cap_is_returned_in_full() {
    let Some((_, result, _instance)) = exec(&["sh", "-c", &write_a(1000)], Duration::from_secs(10))
    else {
        return;
    };

    let result = result.expect("exec");
    assert_eq!(result.stdout, "a".repeat(1000));
    assert_eq!(
        (
            result.stdout_truncated,
            result.stderr.as_str(),
            result.stderr_truncated
        ),
        (false, "done\n", false)
    );
}

#[tokio::test]
async fn linux_create_rejects_a_cpu_cap() {
    let workspace = std::env::temp_dir();
    let mut config = policy_to_config(
        NamedPolicy::ReadOnly,
        &workspace,
        None,
        SandboxBackendType::LinuxLandlock {
            policy: NamedPolicy::ReadOnly,
        },
    );
    config.resource_limits.cpu_rate_hard_cap_percent = Some(20);
    let err = LinuxLandlockBackend::new()
        .create(&config, workspace.as_path())
        .await
        .expect_err("linux create must reject a cpu cap");
    assert_eq!(err.to_string(), CPU_CAP_REQUIRES_WINDOWS_JOB);
}

#[test]
fn invalid_utf8_flood_stays_within_the_cap_after_lossy_decoding() {
    let script = repeat_byte("\\377", 2 * EXEC_OUTPUT_LIMIT, 1);
    let Some((_, result, _instance)) = exec(&["sh", "-c", &script], Duration::from_secs(20)) else {
        return;
    };

    let result = result.expect("exec");
    // Each 0xFF byte decodes to the 3-byte U+FFFD, cut at a char boundary.
    assert_eq!(result.stdout, "\u{FFFD}".repeat(EXEC_OUTPUT_LIMIT / 3));
    assert!(result.stdout_truncated);
}
