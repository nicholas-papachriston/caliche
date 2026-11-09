#![cfg(all(feature = "sandbox-linux", target_os = "linux"))]

use caliche_sandbox::traits::SandboxInstance;
use caliche_sandbox::{
    NamedPolicy, SandboxBackendType, SandboxConfig, create_backend, policy_to_config,
};
use landlock::{Compatible, RulesetAttr};
use std::collections::HashMap;
use std::fs;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

include!("common/mod.rs");

fn lock_process_state() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(|err| err.into_inner())
}

fn read_only(workspace: &Path) -> SandboxConfig {
    policy_to_config(
        NamedPolicy::ReadOnly,
        workspace,
        None,
        SandboxBackendType::LinuxLandlock {
            policy: NamedPolicy::ReadOnly,
        },
    )
}

fn workspace_write(workspace: &Path) -> SandboxConfig {
    policy_to_config(
        NamedPolicy::WorkspaceWrite,
        workspace,
        None,
        SandboxBackendType::LinuxLandlock {
            policy: NamedPolicy::WorkspaceWrite,
        },
    )
}

#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "test helper panics with context"
)]
async fn open_instance(
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

#[expect(clippy::expect_used, reason = "test helper panics with context")]
async fn exec_ok(
    instance: &dyn SandboxInstance,
    script: &str,
    cwd: Option<&str>,
    env: Option<&HashMap<String, String>>,
) -> caliche_sandbox::ExecResult {
    let result = instance
        .exec(
            &["/bin/sh", "-c", script],
            cwd,
            env,
            Duration::from_secs(10),
        )
        .await
        .expect("exec");
    assert_eq!(
        result.exit_code, 0,
        "stdout {} stderr {}",
        result.stdout, result.stderr
    );
    result
}

struct StdinSecret {
    saved: i32,
    read: i32,
    write: i32,
}

impl StdinSecret {
    fn install() -> Self {
        let mut ends = [0; 2];
        #[expect(unsafe_code, reason = "pipe a known stdin payload for the child")]
        let opened = unsafe { libc::pipe(ends.as_mut_ptr()) };
        assert_eq!(opened, 0, "pipe: {}", std::io::Error::last_os_error());
        #[expect(unsafe_code, reason = "write the stdin payload")]
        let wrote = unsafe { libc::write(ends[1], b"SECRET".as_ptr().cast(), 6) };
        assert_eq!(wrote, 6, "write: {}", std::io::Error::last_os_error());
        #[expect(unsafe_code, reason = "save and replace stdin for one exec")]
        let saved = unsafe { libc::dup(0) };
        assert!(saved >= 0, "dup: {}", std::io::Error::last_os_error());
        #[expect(unsafe_code, reason = "point stdin at the secret pipe")]
        let swapped = unsafe { libc::dup2(ends[0], 0) };
        assert_eq!(swapped, 0, "dup2: {}", std::io::Error::last_os_error());
        Self {
            saved,
            read: ends[0],
            write: ends[1],
        }
    }
}

impl Drop for StdinSecret {
    fn drop(&mut self) {
        #[expect(unsafe_code, reason = "restore the test process stdin")]
        unsafe {
            libc::dup2(self.saved, 0);
            libc::close(self.saved);
            libc::close(self.read);
            libc::close(self.write);
        }
    }
}

struct CwdGuard(PathBuf);

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

#[test]
fn read_only_root_denies_a_write_the_parent_can_make() {
    let _process = lock_process_state();
    let dir = tempfile::tempdir().expect("workspace");
    let root = dir.path().canonicalize().expect("canonical");
    fs::write(root.join("parent.txt"), b"parent").expect("parent write");
    let config = read_only(&root);
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, &root).await else {
            return None;
        };
        let cwd = root.to_str().expect("utf-8");
        Some(exec_ok(instance.as_ref(), "echo child > child.txt; if [ -f child.txt ]; then echo WROTE; else echo DENIED; fi", Some(cwd), None).await)
    });
    let Some(result) = report else {
        return;
    };
    assert!(
        result.stdout.contains("DENIED"),
        "read-only root allowed a write: {}",
        result.stdout
    );
    assert!(
        !root.join("child.txt").exists(),
        "child write created a file"
    );
    assert_eq!(
        fs::read(root.join("parent.txt")).expect("parent file"),
        b"parent"
    );
}

#[test]
fn relative_writable_root_is_fixed_at_create() {
    let _process = lock_process_state();
    let base = tempfile::tempdir().expect("base");
    let original = base.path().join("writable");
    fs::create_dir(&original).expect("original");
    let elsewhere = tempfile::tempdir().expect("elsewhere");
    let other = elsewhere.path().join("writable");
    fs::create_dir(&other).expect("other");
    let mut config = read_only(base.path());
    config.writable_roots = vec![PathBuf::from("writable")];
    let previous = std::env::current_dir().expect("cwd");
    let _restore = CwdGuard(previous);
    std::env::set_current_dir(base.path()).expect("chdir base");
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, base.path()).await else {
            return None;
        };
        std::env::set_current_dir(elsewhere.path()).expect("chdir elsewhere");
        let script = format!(
            "echo orig > '{}/from-child'; echo other > '{}/from-child'; echo done",
            original.display(),
            other.display()
        );
        Some(exec_ok(instance.as_ref(), &script, None, None).await)
    });
    let Some(_) = report else {
        return;
    };
    assert_eq!(
        fs::read(original.join("from-child")).expect("original write"),
        b"orig\n"
    );
    assert!(
        !other.join("from-child").exists(),
        "relative root followed the cwd after create"
    );
}

#[test]
fn swapped_symlink_does_not_receive_the_write() {
    let _process = lock_process_state();
    let parent = tempfile::tempdir().expect("parent");
    let workspace = parent.path().join("ws");
    // A workspace without `.git` gets no grant at its root, so the writes
    // go to `sub`, which the swapped-in directory also has.
    fs::create_dir_all(workspace.join("sub")).expect("workspace");
    let outside = tempfile::tempdir().expect("outside");
    fs::create_dir(outside.path().join("sub")).expect("outside sub");
    let config = {
        let mut config = read_only(&workspace);
        config.writable_roots = vec![workspace.clone()];
        config
    };
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let saved = parent.path().join("ws.real");
    let report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, &workspace).await else {
            return None;
        };
        fs::rename(&workspace, &saved).expect("rename");
        std::os::unix::fs::symlink(outside.path(), &workspace).expect("symlink");
        let script = format!(
            "echo via_link > '{}/sub/marker'; echo via_real > '{}/sub/marker'; echo done",
            workspace.display(),
            saved.display()
        );
        let cwd = saved.to_str().expect("utf-8");
        Some(exec_ok(instance.as_ref(), &script, Some(cwd), None).await)
    });
    let Some(_) = report else {
        return;
    };
    assert!(
        !outside.path().join("sub/marker").exists(),
        "write followed the swapped symlink"
    );
    assert_eq!(
        fs::read(saved.join("sub/marker")).expect("original inode"),
        b"via_real\n"
    );
}

#[test]
fn inherited_fd_and_stdin_do_not_survive_exec() {
    let _process = lock_process_state();
    let workspace = tempfile::tempdir().expect("workspace");
    let config = read_only(workspace.path());
    let source = fs::File::open("/proc/self/stat").expect("stat");
    #[expect(
        unsafe_code,
        reason = "plant a non-CLOEXEC descriptor for the child to observe"
    )]
    let leaked = unsafe { libc::dup2(source.as_raw_fd(), 250) };
    assert_eq!(leaked, 250, "dup2: {}", std::io::Error::last_os_error());
    #[expect(
        unsafe_code,
        reason = "clear CLOEXEC so the descriptor would be inherited"
    )]
    let flags = unsafe { libc::fcntl(250, libc::F_SETFD, 0) };
    assert_eq!(flags, 0, "fcntl: {}", std::io::Error::last_os_error());
    let _stdin = StdinSecret::install();
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let script = r#"
if [ -e /proc/self/fd/250 ]; then echo LEAK; else echo CLOSED; fi
echo "STDIN:$(readlink /proc/self/fd/0)"
echo "DATA:$(head -c 6 || true)"
ruleset=0
for fd in /proc/self/fd/*; do
  target=$(readlink "$fd")
  case "$target" in
    *landlock*) ruleset=1 ;;
  esac
done
if [ "$ruleset" = 1 ]; then echo RULESET; else echo NO_RULESET; fi
echo ok
"#;
    let report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, workspace.path()).await else {
            return None;
        };
        Some(exec_ok(instance.as_ref(), script, None, None).await)
    });
    #[expect(unsafe_code, reason = "close the planted descriptor")]
    unsafe {
        libc::close(250);
    }
    let Some(result) = report else {
        return;
    };
    assert!(result.stdout.contains("CLOSED"), "{}", result.stdout);
    assert!(
        result.stdout.contains("STDIN:/dev/null"),
        "{}",
        result.stdout
    );
    assert!(
        !result.stdout.contains("SECRET"),
        "child inherited the parent stdin: {}",
        result.stdout
    );
    assert!(result.stdout.contains("NO_RULESET"), "{}", result.stdout);
    assert!(result.stdout.contains("ok"), "{}", result.stdout);
}

#[test]
fn private_temp_replaces_shared_tmp() {
    let _process = lock_process_state();
    let first_dir = tempfile::tempdir().expect("first");
    let second_dir = tempfile::tempdir().expect("second");
    let first_cfg = workspace_write(first_dir.path());
    let second_cfg = workspace_write(second_dir.path());
    let marker = Path::new("/tmp/cls-landlock-shared-marker");
    let _ = fs::remove_file(marker);
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let script = r#"printf '%s\n' "$TMPDIR"; touch "$TMPDIR/file"; if touch /tmp/cls-landlock-shared-marker; then echo TMP_OK; else echo TMP_DENY; fi"#;
    let report = runtime.block_on(async {
        let Some(first) = open_instance(&first_cfg, first_dir.path()).await else {
            return None;
        };
        let Some(second) = open_instance(&second_cfg, second_dir.path()).await else {
            return None;
        };
        let left = exec_ok(first.as_ref(), script, None, None).await;
        let right = exec_ok(second.as_ref(), script, None, None).await;
        let left_temp = left.stdout.lines().next().unwrap_or("").to_string();
        let right_temp = right.stdout.lines().next().unwrap_or("").to_string();
        let left_file = Path::new(&left_temp).join("file").is_file();
        let right_file = Path::new(&right_temp).join("file").is_file();
        Some((left.stdout, right.stdout, left_file, right_file))
    });
    let Some((left, right, left_file, right_file)) = report else {
        return;
    };
    let left_temp = left.lines().next().expect("left temp");
    let right_temp = right.lines().next().expect("right temp");
    assert_ne!(left_temp, right_temp, "instances share a temp directory");
    assert!(left.contains("TMP_DENY"), "{left}");
    assert!(right.contains("TMP_DENY"), "{right}");
    assert!(!marker.exists(), "child created {marker:?}");
    assert!(left_file, "{left_temp}");
    assert!(right_file, "{right_temp}");
}

#[test]
fn read_only_does_not_publish_tmpdir() {
    let _process = lock_process_state();
    let workspace = tempfile::tempdir().expect("workspace");
    let config = read_only(workspace.path());
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, workspace.path()).await else {
            return None;
        };
        Some(
            exec_ok(
                instance.as_ref(),
                "if [ -n \"$TMPDIR\" ]; then echo SET; else echo UNSET; fi",
                None,
                None,
            )
            .await,
        )
    });
    let Some(result) = report else {
        return;
    };
    assert!(result.stdout.contains("UNSET"), "{}", result.stdout);
}

/// Whether `/proc/sys` is its own read-only mount, as Docker and most
/// container runtimes set it up.
fn proc_sys_mounted_read_only() -> bool {
    fs::read_to_string("/proc/self/mountinfo").is_ok_and(|info| {
        info.lines().any(|line| {
            let mut fields = line.split(' ').skip(4);
            fields.next() == Some("/proc/sys")
                && fields
                    .next()
                    .is_some_and(|options| options.split(',').any(|option| option == "ro"))
        })
    })
}

#[test]
fn devices_are_usable_and_proc_stays_closed() {
    let _process = lock_process_state();
    let workspace = tempfile::tempdir().expect("workspace");
    let config = read_only(workspace.path());
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    // On a read-only `/proc/sys` mount the kernel returns EROFS before
    // Landlock is asked, so EROFS also counts as closed there. Anywhere else
    // only Landlock's EACCES does.
    let closed = if proc_sys_mounted_read_only() {
        "(errno.EACCES, errno.EROFS)"
    } else {
        "(errno.EACCES,)"
    };
    let script = r#"python3 -c '
import errno, os, sys
def check(path, flags, expect_ok):
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if expect_ok or exc.errno not in CLOSED:
            sys.stderr.write("open %s errno %s\n" % (path, exc.errno))
            sys.exit(2)
        return
    os.close(fd)
    if not expect_ok:
        sys.stderr.write("open %s succeeded\n" % path)
        sys.exit(3)
check("/dev/null", os.O_RDWR, True)
check("/dev/zero", os.O_RDWR, True)
check("/dev/urandom", os.O_RDONLY, True)
check("/dev/urandom", os.O_WRONLY, False)
check("/dev/random", os.O_RDONLY, True)
check("/dev/random", os.O_WRONLY, False)
check("/dev/tty", os.O_RDWR, False)
check("/proc/sys/kernel/hostname", os.O_WRONLY, False)
'"#
    .replace("CLOSED", closed);
    let report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, workspace.path()).await else {
            return None;
        };
        Some(exec_ok(instance.as_ref(), &script, None, None).await)
    });
    if report.is_none() {
        require_landlock();
    }
}

#[test]
fn signal_scope_denies_the_parent() {
    let _process = lock_process_state();
    let workspace = tempfile::tempdir().expect("workspace");
    let config = read_only(workspace.path());
    let mut env = HashMap::new();
    env.insert("PARENT_PID".to_string(), std::process::id().to_string());
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let script = r#"python3 -c '
import errno, os, sys
ppid = int(os.environ["PARENT_PID"])
try:
    os.kill(ppid, 0)
except OSError as exc:
    if exc.errno in (errno.EACCES, errno.EPERM):
        sys.exit(0)
    sys.stderr.write("errno %s\n" % exc.errno)
    sys.exit(2)
sys.exit(1)
'"#;
    let report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, workspace.path()).await else {
            return None;
        };
        Some(exec_ok(instance.as_ref(), script, None, Some(&env)).await)
    });
    if report.is_none() {
        require_landlock();
    }
}

#[test]
fn parent_tokio_socket_is_absent_after_exec() {
    let _process = lock_process_state();
    let workspace = tempfile::tempdir().expect("workspace");
    let config = read_only(workspace.path());
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let report = runtime.block_on(async {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind tokio listener");
        let fd = listener.as_raw_fd();
        assert!(fd >= 3, "listener fd {fd} is below the close_range floor");
        clear_cloexec(fd);
        let Some(instance) = open_instance(&config, workspace.path()).await else {
            return None;
        };
        let script = format!(
            "target=$(readlink /proc/self/fd/{fd} 2>/dev/null || echo ABSENT); \
             echo \"FD:{fd}:$target\"; \
             if [ \"$target\" = ABSENT ]; then echo CLOSED; else echo LEAK; fi"
        );
        let result = exec_ok(instance.as_ref(), &script, None, None).await;
        drop(listener);
        Some(result)
    });
    let Some(result) = report else {
        return;
    };
    assert!(
        result.stdout.contains("CLOSED"),
        "parent tokio socket survived exec: {}",
        result.stdout
    );
    assert!(
        !result.stdout.contains("LEAK"),
        "parent tokio socket survived exec: {}",
        result.stdout
    );
}

#[test]
fn abstract_unix_scope_denies_an_outside_socket() {
    let _process = lock_process_state();
    let workspace = tempfile::tempdir().expect("workspace");
    let config = read_only(workspace.path());
    let name = format!(
        "cls{}{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos()
    );
    let listener = listen_abstract(&name);
    let mut env = HashMap::new();
    env.insert("ABSTRACT_NAME".to_string(), name);
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let script = r#"python3 -c '
import errno, os, socket, sys
name = b"\0" + os.environ["ABSTRACT_NAME"].encode()
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(2)
try:
    sock.connect(name)
except OSError as exc:
    if exc.errno in (errno.EACCES, errno.EPERM):
        sys.stdout.write("DENIED\n")
        sys.exit(0)
    sys.stderr.write("errno %s\n" % exc.errno)
    sys.exit(2)
sys.exit(1)
'"#;
    let v6 = v6_scopes_available();
    let report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, workspace.path()).await else {
            return None;
        };
        if !v6 {
            return Some(
                exec_ok(instance.as_ref(), "echo STILL_RUNS", None, None)
                    .await
                    .stdout,
            );
        }
        Some(
            exec_ok(instance.as_ref(), script, None, Some(&env))
                .await
                .stdout,
        )
    });
    drop(listener);
    let Some(stdout) = report else {
        return;
    };
    if v6 {
        assert!(
            stdout.contains("DENIED"),
            "abstract connect was not denied: {stdout}"
        );
    } else {
        assert!(
            stdout.contains("STILL_RUNS"),
            "older ABI rejected the sandbox: {stdout}"
        );
    }
}

fn clear_cloexec(fd: i32) {
    #[expect(unsafe_code, reason = "clear CLOEXEC so the socket would be inherited")]
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    assert!(flags >= 0, "F_GETFD: {}", std::io::Error::last_os_error());
    #[expect(unsafe_code, reason = "clear CLOEXEC so the socket would be inherited")]
    let set = unsafe { libc::fcntl(fd, libc::F_SETFD, flags & !libc::FD_CLOEXEC) };
    assert_eq!(set, 0, "F_SETFD: {}", std::io::Error::last_os_error());
}

fn v6_scopes_available() -> bool {
    landlock::Ruleset::default()
        .set_compatibility(landlock::CompatLevel::HardRequirement)
        .scope(landlock::Scope::AbstractUnixSocket | landlock::Scope::Signal)
        .and_then(landlock::Ruleset::create)
        .is_ok()
}

struct AbstractListener(i32);

impl Drop for AbstractListener {
    fn drop(&mut self) {
        #[expect(unsafe_code, reason = "close the abstract listener")]
        unsafe {
            libc::close(self.0);
        }
    }
}

fn listen_abstract(name: &str) -> AbstractListener {
    #[expect(
        unsafe_code,
        reason = "bind an abstract unix socket outside the sandbox"
    )]
    let fd = unsafe { libc::socket(libc::AF_UNIX, libc::SOCK_STREAM | libc::SOCK_CLOEXEC, 0) };
    assert!(fd >= 0, "socket: {}", std::io::Error::last_os_error());
    #[expect(unsafe_code, reason = "zero a sockaddr_un")]
    let mut addr: libc::sockaddr_un = unsafe { std::mem::zeroed() };
    addr.sun_family = libc::AF_UNIX as libc::sa_family_t;
    let bytes = name.as_bytes();
    assert!(
        bytes.len() + 1 < addr.sun_path.len(),
        "abstract name is too long"
    );
    addr.sun_path[0] = 0;
    for (index, byte) in bytes.iter().enumerate() {
        addr.sun_path[index + 1] = *byte as libc::c_char;
    }
    let len = std::mem::offset_of!(libc::sockaddr_un, sun_path) + 1 + bytes.len();
    #[expect(
        unsafe_code,
        reason = "bind an abstract unix socket outside the sandbox"
    )]
    let bound = unsafe { libc::bind(fd, (&raw const addr).cast(), len as libc::socklen_t) };
    assert_eq!(bound, 0, "bind: {}", std::io::Error::last_os_error());
    #[expect(unsafe_code, reason = "listen on the abstract unix socket")]
    let listening = unsafe { libc::listen(fd, 8) };
    assert_eq!(listening, 0, "listen: {}", std::io::Error::last_os_error());
    AbstractListener(fd)
}

const CLS_TTY_REGRESSION: &str = "CLS_TTY_REGRESSION";

#[test]
fn controlling_tty_cannot_receive_injected_input() {
    let _process = lock_process_state();
    if std::env::var_os(CLS_TTY_REGRESSION).is_none() {
        reexec_tty_regression();
        return;
    }
    let workspace = tempfile::tempdir().expect("workspace");
    let config = read_only(workspace.path());
    // This process must own a controlling terminal. The cargo test process
    // usually does not, and it is often already a session leader, so `setsid`
    // there returns EPERM. The re-exec is not a group leader.
    let _tty = session_with_controlling_tty();
    let parent = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty");
    assert!(
        parent.is_ok(),
        "parent has no controlling terminal: {}",
        parent.err().map(|err| err.to_string()).unwrap_or_default()
    );
    let mut env = HashMap::new();
    env.insert("PARENT_SID".to_string(), session_id().to_string());
    let script = r#"python3 -c '
import errno, fcntl, os, sys, termios
parent = int(os.environ["PARENT_SID"])
if os.getsid(0) == parent:
    sys.stderr.write("SAME_SID\n")
    sys.exit(2)
try:
    tty = os.open("/dev/tty", os.O_RDWR)
except OSError as exc:
    if exc.errno != errno.EACCES:
        sys.stderr.write("tty errno %s\n" % exc.errno)
        sys.exit(3)
else:
    try:
        fcntl.ioctl(tty, termios.TIOCSTI, b"x")
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EPERM):
            sys.stderr.write("TIOCSTI errno %s\n" % exc.errno)
            sys.exit(4)
    else:
        sys.stderr.write("TIOCSTI_OK\n")
        sys.exit(1)
    sys.stderr.write("TTY_OPEN\n")
    sys.exit(5)
fd = os.open("/dev/null", os.O_RDWR)
try:
    fcntl.ioctl(fd, termios.TIOCSTI, b"x")
except OSError as exc:
    if exc.errno not in (errno.EACCES, errno.EPERM):
        sys.stderr.write("null ioctl errno %s\n" % exc.errno)
        sys.exit(6)
else:
    sys.stderr.write("NULL_IOCTL_OK\n")
    sys.exit(7)
'"#;
    let runtime = tokio::runtime::Runtime::new().expect("runtime");
    let _report = runtime.block_on(async {
        let Some(instance) = open_instance(&config, workspace.path()).await else {
            return None;
        };
        Some(exec_ok(instance.as_ref(), script, None, Some(&env)).await)
    });
}

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn reexec_tty_regression() {
    // The child inherits CLS_REQUIRE_LANDLOCK_TESTS. Unset and `0` still skip.
    let exe = std::env::current_exe().expect("test binary");
    let output = std::process::Command::new(exe)
        .args([
            "controlling_tty_cannot_receive_injected_input",
            "--exact",
            "--test-threads=1",
        ])
        .env(CLS_TTY_REGRESSION, "1")
        .output()
        .expect("re-exec the tty regression");
    assert!(
        output.status.success(),
        "pty regression failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

struct ControllingTty {
    _master: OwnedFd,
    _slave: OwnedFd,
}

impl Drop for ControllingTty {
    fn drop(&mut self) {
        // Closing the master sends SIGHUP to this session. The checks have
        // finished; ignore the hangup so the test process can exit. The owned
        // fds close after this function returns.
        #[expect(unsafe_code, reason = "ignore SIGHUP from dropping the pty master")]
        unsafe {
            libc::signal(libc::SIGHUP, libc::SIG_IGN);
        }
    }
}

fn session_with_controlling_tty() -> ControllingTty {
    #[expect(
        unsafe_code,
        reason = "setsid so this process can take a controlling terminal"
    )]
    let sid = unsafe { libc::setsid() };
    assert!(sid > 0, "setsid: {}", std::io::Error::last_os_error());
    let mut master: i32 = -1;
    let mut slave: i32 = -1;
    #[expect(unsafe_code, reason = "openpty allocates a pty pair")]
    let opened = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null::<libc::termios>(),
            std::ptr::null::<libc::winsize>(),
        )
    };
    assert_eq!(opened, 0, "openpty: {}", std::io::Error::last_os_error());
    #[expect(unsafe_code, reason = "adopt the pty master")]
    let master = unsafe { OwnedFd::from_raw_fd(master) };
    #[expect(unsafe_code, reason = "adopt the pty slave")]
    let slave = unsafe { OwnedFd::from_raw_fd(slave) };
    #[expect(unsafe_code, reason = "install the slave as the controlling terminal")]
    let set = unsafe { libc::ioctl(slave.as_raw_fd(), libc::TIOCSCTTY, 0) };
    assert_eq!(set, 0, "TIOCSCTTY: {}", std::io::Error::last_os_error());
    ControllingTty {
        _master: master,
        _slave: slave,
    }
}

fn session_id() -> i32 {
    #[expect(unsafe_code, reason = "read this process session id")]
    let sid = unsafe { libc::getsid(0) };
    assert!(sid > 0, "getsid: {}", std::io::Error::last_os_error());
    sid
}
