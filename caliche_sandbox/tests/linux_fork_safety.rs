#![cfg(all(feature = "sandbox-linux", target_os = "linux"))]

use caliche_sandbox::traits::SandboxInstance;
use caliche_sandbox::{
    NamedPolicy, SandboxBackendType, SandboxConfig, create_backend, policy_to_config,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::Duration;

/// Aborts when the forked child allocates through the Rust allocator.
struct AbortInChild;

static PARENT_PID: AtomicI32 = AtomicI32::new(0);

#[global_allocator]
static ALLOCATOR: AbortInChild = AbortInChild;

#[expect(
    unsafe_code,
    reason = "test allocator aborts if a forked child allocates"
)]
unsafe impl GlobalAlloc for AbortInChild {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        refuse_forked_child();
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        refuse_forked_child();
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        refuse_forked_child();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        refuse_forked_child();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn refuse_forked_child() {
    #[expect(
        unsafe_code,
        reason = "getpid is async-signal-safe and does not allocate"
    )]
    let pid = unsafe { libc::getpid() };
    let parent = PARENT_PID.load(Ordering::Relaxed);
    if parent == 0 {
        let _ = PARENT_PID.compare_exchange(0, pid, Ordering::Relaxed, Ordering::Relaxed);
        return;
    }
    if pid != parent {
        #[expect(unsafe_code, reason = "abort a forked child that allocated")]
        unsafe {
            libc::abort()
        };
    }
}

/// Held by a parent thread while sandboxed children spawn.
static ALLOC_CONTENTION: Mutex<()> = Mutex::new(());

/// Sandboxed children forked while other threads allocate.
const FORK_ITERATIONS: usize = 500;
/// Bound so a stuck child fails this test instead of hanging the suite.
///
/// The assertions are the parent fd count and `NoNewPrivs`. This does not
/// catch a malloc-after-fork hang: glibc `fork` runs atfork handlers, so a
/// `pre_exec` that allocates can still pass here.
const STRESS_GUARD: Duration = Duration::from_secs(300);

/// One integration-test process shares an fd table. These tests count fds,
/// so they must not overlap.
fn exclusive() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn open_fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("read /proc/self/fd")
        .filter_map(Result::ok)
        .count()
}

/// Fd count after detached pipe readers have exited.
///
/// `exec` returns when those readers signal EOF, which is before they drop
/// the pipe. The minimum over a short yield window is the count once that
/// drop has happened. A descriptor leaked for the whole window stays in it.
fn settled_fd_count() -> usize {
    let mut low = open_fd_count();
    for _ in 0..32 {
        thread::yield_now();
        low = std::cmp::min(low, open_fd_count());
    }
    low
}

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn no_new_privs() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .expect("status")
        .lines()
        .any(|line| line == "NoNewPrivs:\t1")
}

fn landlock_config(workspace: &Path, writable: bool) -> SandboxConfig {
    let mut config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace,
        None,
        SandboxBackendType::LinuxLandlock {
            policy: NamedPolicy::ReadOnly,
        },
    );
    if writable {
        config.writable_roots = vec![workspace.to_path_buf()];
    }
    config
}

include!("common/mod.rs");

/// `None` when this kernel cannot apply Landlock and the require flag is not
/// `1`. The flag panics instead of skipping, matching the cgroup tests.
#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper panics with context"
)]
async fn landlock_instance(
    config: &SandboxConfig,
    workspace: &Path,
) -> Option<Box<dyn SandboxInstance>> {
    let backend = create_backend(config).await.expect("landlock backend");
    match backend.create(config, workspace).await {
        Ok(instance) => Some(instance),
        Err(err) if err.to_string().contains("not available") => {
            require_landlock();
            None
        }
        Err(err) => panic!("landlock instance: {err:#}"),
    }
}

/// Fd count and parent `NoNewPrivs` across 500 execs while other threads allocate.
#[test]
fn parent_fds_and_no_new_privs_stay_stable_across_execs() {
    let _guard = exclusive();
    let nnp_before = no_new_privs();
    let workspace = std::env::temp_dir();
    let config = landlock_config(&workspace, false);
    let stop = Arc::new(AtomicBool::new(false));
    let mut allocators = Vec::new();
    for _ in 0..8 {
        let stop = Arc::clone(&stop);
        allocators.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let mut held = Vec::with_capacity(8);
                for size in [64_usize, 4_096, 65_536] {
                    held.push(vec![1_u8; size]);
                }
                std::hint::black_box(held);
            }
        }));
    }

    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        let outcome = runtime.block_on(async move {
            let Some(instance) = landlock_instance(&config, &workspace).await else {
                return Ok(None);
            };
            instance
                .exec(
                    &["/bin/sh", "-c", "echo ok"],
                    None,
                    None,
                    Duration::from_secs(5),
                )
                .await
                .map_err(|err| anyhow::anyhow!("warmup exec: {err:#}"))?;
            let before = settled_fd_count();
            for iteration in 0..FORK_ITERATIONS {
                let result = instance
                    .exec(
                        &["/bin/sh", "-c", "echo ok"],
                        None,
                        None,
                        Duration::from_secs(5),
                    )
                    .await
                    .map_err(|err| anyhow::anyhow!("iteration {iteration}: {err:#}"))?;
                if result.exit_code != 0 || !result.stdout.contains("ok") {
                    anyhow::bail!("iteration {iteration}: {result:?}");
                }
            }
            let after = settled_fd_count();
            Ok(Some((before, after)))
        });
        let _ = tx.send(outcome);
    });

    let received = rx.recv_timeout(STRESS_GUARD);
    stop.store(true, Ordering::Relaxed);
    for allocator in allocators {
        allocator.join().expect("allocator thread");
    }
    let outcome = received.unwrap_or_else(|_| {
        panic!(
            "{FORK_ITERATIONS} sandboxed execs did not return within {STRESS_GUARD:?} while other threads allocated"
        )
    });
    let Some((before, after)) = outcome.expect("stress exec") else {
        return;
    };
    assert_eq!(
        before, after,
        "open fds changed across {FORK_ITERATIONS} sandboxed execs ({before} -> {after})"
    );
    assert_eq!(
        no_new_privs(),
        nnp_before,
        "prctl(PR_SET_NO_NEW_PRIVS) ran in the parent"
    );
}

#[test]
fn landlock_allows_an_inside_write_and_denies_an_outside_write() {
    let _guard = exclusive();
    let nnp_before = no_new_privs();
    let workspace_dir = tempfile::TempDir::new().expect("workspace");
    let outside_dir = tempfile::TempDir::new().expect("outside");
    let workspace = workspace_dir
        .path()
        .canonicalize()
        .expect("canonical workspace");
    std::fs::create_dir(workspace.join("src")).expect("src");
    let outside = outside_dir
        .path()
        .canonicalize()
        .expect("canonical outside");
    let config = landlock_config(&workspace, true);
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let checked = runtime.block_on(async {
        let Some(instance) = landlock_instance(&config, &workspace).await else {
            return None;
        };
        instance
            .exec(
                &["/bin/sh", "-c", "echo warmup"],
                None,
                None,
                Duration::from_secs(5),
            )
            .await
            .expect("warmup");
        let before = settled_fd_count();
        let report = write_probe(instance.as_ref(), &workspace, &outside).await;
        let after = settled_fd_count();
        drop(instance);
        Some((before, after, report))
    });
    let Some((before, after, (stdout, stderr))) = checked else {
        return;
    };
    assert!(
        stdout.contains("OUT_DENY"),
        "outside write was not denied: {stdout} {stderr}"
    );
    assert!(
        !stdout.contains("OUT_OK"),
        "outside write was allowed: {stdout}"
    );
    assert_eq!(
        std::fs::read(workspace.join("src/in.txt")).expect("inside write"),
        b"in\n"
    );
    assert!(
        !outside.join("out.txt").exists(),
        "outside write created a file"
    );
    assert_eq!(
        before, after,
        "open fds changed across a landlock exec ({before} -> {after})"
    );
    std::fs::write(outside.join("parent.txt"), b"parent").expect("parent is not landlocked");
    assert_eq!(
        no_new_privs(),
        nnp_before,
        "prctl(PR_SET_NO_NEW_PRIVS) ran in the parent"
    );
}

#[expect(clippy::expect_used, reason = "test helper panics with context")]
async fn write_probe(
    instance: &dyn SandboxInstance,
    workspace: &Path,
    outside: &Path,
) -> (String, String) {
    let mut env = HashMap::new();
    env.insert("OUTSIDE".to_string(), outside.display().to_string());
    let workspace_str = workspace.to_str().expect("utf-8 workspace");
    // A workspace without `.git` gets no grant at its root, so the probe
    // writes into a directory below it.
    let script = "echo in > src/in.txt\nif echo out > \"$OUTSIDE/out.txt\"; then echo OUT_OK; else echo OUT_DENY; fi";
    let result = instance
        .exec(
            &["/bin/sh", "-c", script],
            Some(workspace_str),
            Some(&env),
            Duration::from_secs(10),
        )
        .await
        .expect("exec");
    assert_eq!(
        result.exit_code, 0,
        "stdout {} stderr {}",
        result.stdout, result.stderr
    );
    (result.stdout, result.stderr)
}

const LOCK_ITERATIONS: usize = 32;
const LOCK_GUARD: Duration = Duration::from_secs(60);

#[test]
fn spawn_returns_while_stderr_and_alloc_locks_are_held() {
    let _guard = exclusive();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_err = Arc::clone(&stop);
    let stderr_holder = thread::spawn(move || {
        let _locked = std::io::stderr().lock();
        while !stop_err.load(Ordering::Relaxed) {
            thread::yield_now();
        }
    });
    let stop_alloc = Arc::clone(&stop);
    let alloc_holder = thread::spawn(move || {
        let _locked = ALLOC_CONTENTION
            .lock()
            .unwrap_or_else(|err| err.into_inner());
        while !stop_alloc.load(Ordering::Relaxed) {
            let block = vec![0_u8; 8_192];
            std::hint::black_box(block);
            thread::yield_now();
        }
    });

    let workspace = std::env::temp_dir();
    let config = landlock_config(&workspace, false);
    let (tx, rx) = std::sync::mpsc::channel();
    thread::spawn(move || {
        let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
        let outcome = runtime.block_on(async move {
            let Some(instance) = landlock_instance(&config, &workspace).await else {
                return Ok(0);
            };
            for iteration in 0..LOCK_ITERATIONS {
                let result = instance
                    .exec(
                        &["/bin/sh", "-c", "echo ok"],
                        None,
                        None,
                        Duration::from_secs(5),
                    )
                    .await
                    .map_err(|err| anyhow::anyhow!("iteration {iteration}: {err:#}"))?;
                if result.exit_code != 0 || !result.stdout.contains("ok") {
                    anyhow::bail!("iteration {iteration}: {result:?}");
                }
            }
            Ok(LOCK_ITERATIONS)
        });
        let _ = tx.send(outcome);
    });

    let received = rx.recv_timeout(LOCK_GUARD);
    stop.store(true, Ordering::Relaxed);
    stderr_holder.join().expect("stderr holder");
    alloc_holder.join().expect("alloc holder");
    let outcome = received.unwrap_or_else(|_| {
        panic!(
            "sandboxed execs did not return within {LOCK_GUARD:?} while stderr and the alloc lock were held"
        )
    });
    let finished = outcome.expect("lock stress");
    if finished == 0 {
        return;
    }
    assert_eq!(finished, LOCK_ITERATIONS);
}
