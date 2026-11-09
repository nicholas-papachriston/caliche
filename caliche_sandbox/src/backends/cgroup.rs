#![cfg(target_os = "linux")]

//! cgroup v2 containment for Linux execs.
//!
//! Each instance owns a cgroup under the cgroup of this process, and each exec
//! runs in its own child cgroup inside it. A process cannot leave its cgroup by
//! calling `setsid` or by double-forking, so writing `cgroup.kill` kills every
//! process the exec started. On timeout only that exec's cgroup is killed;
//! anything still running when the instance is dropped is killed with it.
//!
//! This needs the unified hierarchy at `/sys/fs/cgroup`, kernel 5.14+ for
//! `cgroup.kill`, and write access to this process's cgroup (root, or a cgroup
//! delegated by systemd). Without them the instance has no cgroup and a timeout
//! falls back to killing the exec's process group.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::os::fd::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::Once;
use std::time::{Duration, Instant};
use tracing::warn;

const MOUNT: &str = "/sys/fs/cgroup";
/// How long to wait for killed processes to leave a cgroup.
const KILL_WAIT: Duration = Duration::from_secs(2);
const KILL_POLL: Duration = Duration::from_millis(5);

/// The cgroup holding every exec of one instance. Dropping it kills what is
/// left inside and removes the cgroups. If killed processes have not exited
/// yet, it waits for them on another thread, so a drop on an async runtime
/// thread does not block it.
#[derive(Debug)]
pub(crate) struct InstanceCgroup {
    dir: PathBuf,
}

impl InstanceCgroup {
    /// Create a cgroup for one instance under the cgroup of this process.
    pub(crate) fn create() -> Result<Self> {
        let parent = own_cgroup()?;
        // Moving a child into a cgroup below `parent` needs write access to
        // `parent/cgroup.procs`, the common ancestor of source and target.
        OpenOptions::new()
            .write(true)
            .open(parent.join("cgroup.procs"))
            .with_context(|| format!("open {}/cgroup.procs for writing", parent.display()))?;
        let dir = parent.join(format!("caliche-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir(&dir).with_context(|| format!("create cgroup {}", dir.display()))?;
        let cgroup = Self { dir };
        if !cgroup.dir.join("cgroup.kill").exists() {
            anyhow::bail!("cgroup.kill is missing; kernel 5.14 or later is required");
        }
        Ok(cgroup)
    }

    /// Create the cgroup for one exec.
    pub(crate) fn exec(&self) -> Result<ExecCgroup> {
        let dir = self.dir.join(format!("exec-{}", uuid::Uuid::now_v7()));
        std::fs::create_dir(&dir).with_context(|| format!("create cgroup {}", dir.display()))?;
        let procs = OpenOptions::new()
            .write(true)
            .open(dir.join("cgroup.procs"))
            .with_context(|| format!("open {}/cgroup.procs", dir.display()));
        match procs {
            Ok(procs) => Ok(ExecCgroup { dir, procs }),
            Err(err) => {
                let _ = std::fs::remove_dir(&dir);
                Err(err)
            }
        }
    }
}

impl Drop for InstanceCgroup {
    fn drop(&mut self) {
        if let Err(err) = write_kill(&self.dir) {
            warn_kill_failed_once(&self.dir, &err);
        }
        if populated(&self.dir) {
            let dir = self.dir.clone();
            let spawned = std::thread::Builder::new()
                .name("cgroup-teardown".into())
                .spawn(move || remove_when_empty(&dir));
            if spawned.is_ok() {
                return;
            }
        }
        remove_when_empty(&self.dir);
    }
}

/// The cgroup of one exec. The child joins it before exec, so every process
/// the command starts is inside it.
#[derive(Debug)]
pub(crate) struct ExecCgroup {
    dir: PathBuf,
    procs: File,
}

impl ExecCgroup {
    /// Descriptor of this cgroup's `cgroup.procs`. Writing `0` to it from the
    /// forked child moves the child in. It is close-on-exec.
    pub(crate) fn procs_fd(&self) -> RawFd {
        self.procs.as_raw_fd()
    }

    /// Kill every process in this cgroup and wait until they are gone. If
    /// `cgroup.kill` cannot be written, warn and kill the process group of
    /// `child`, the exec's group leader, instead.
    pub(crate) fn kill(&self, child: &Child) {
        match write_kill(&self.dir) {
            Ok(()) => wait_until_empty(&self.dir),
            Err(err) => {
                warn_kill_failed_once(&self.dir, &err);
                super::child_output::kill_process_group(child);
            }
        }
    }
}

impl Drop for ExecCgroup {
    /// Remove the cgroup if it is empty. Processes that outlive a successful
    /// exec keep it busy until the instance is dropped.
    fn drop(&mut self) {
        let _ = std::fs::remove_dir(&self.dir);
    }
}

// BEGIN SIGNAL_SAFE_JOIN
/// Errno for the pre-exec hook. Zero becomes `EIO` so a failed call is not
/// reported as success.
pub(super) fn raw_errno() -> i32 {
    // SAFETY: errno lives in this thread and the read is async-signal-safe.
    #[expect(unsafe_code, reason = "async-signal-safe errno read")]
    let err = unsafe { *libc::__errno_location() };
    if err == 0 { libc::EIO } else { err }
}

/// Move the calling process into the cgroup whose `cgroup.procs` is `fd`.
/// Async-signal-safe, for use between fork and exec.
pub(crate) fn join(fd: RawFd) -> std::io::Result<()> {
    // SAFETY: write reads one byte from a static buffer; `fd` stays open in
    // the parent until spawn returns.
    #[expect(unsafe_code, reason = "raw write between fork and exec")]
    let written = unsafe { libc::write(fd, b"0".as_ptr().cast(), 1) };
    if written == 1 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(raw_errno()))
    }
}
// END SIGNAL_SAFE_JOIN

/// The directory of this process's cgroup in the unified hierarchy.
fn own_cgroup() -> Result<PathBuf> {
    let mount = Path::new(MOUNT);
    if !mount.join("cgroup.controllers").exists() {
        anyhow::bail!("cgroup v2 is not mounted at {MOUNT}");
    }
    let membership =
        std::fs::read_to_string("/proc/self/cgroup").context("read /proc/self/cgroup")?;
    let relative = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .context("no cgroup v2 entry in /proc/self/cgroup")?;
    Ok(mount.join(relative.trim_start_matches('/')))
}

/// Write `cgroup.kill` in `dir`, which kills it and every cgroup below it.
fn write_kill(dir: &Path) -> std::io::Result<()> {
    std::fs::write(dir.join("cgroup.kill"), "1")
}

fn warn_kill_failed_once(dir: &Path, err: &std::io::Error) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        warn!(
            cgroup = %dir.display(),
            error = %err,
            "cannot write cgroup.kill; a process that left the exec's process group may survive"
        );
    });
}

/// Wait up to [`KILL_WAIT`] for `cgroup.events` in `dir` to report it empty.
fn wait_until_empty(dir: &Path) {
    let deadline = Instant::now() + KILL_WAIT;
    while populated(dir) && Instant::now() < deadline {
        #[expect(
            clippy::disallowed_methods,
            reason = "runs on a blocking exec thread or the teardown thread"
        )]
        std::thread::sleep(KILL_POLL);
    }
}

/// Wait for the killed instance cgroup `dir` to empty, then remove its exec
/// cgroups and itself.
fn remove_when_empty(dir: &Path) {
    wait_until_empty(dir);
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_type().is_ok_and(|kind| kind.is_dir()) {
                let _ = std::fs::remove_dir(entry.path());
            }
        }
    }
    let _ = std::fs::remove_dir(dir);
}

fn populated(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("cgroup.events"))
        .is_ok_and(|events| events.lines().any(|line| line == "populated 1"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::io::{BufRead, BufReader};
    use std::os::unix::process::{CommandExt, ExitStatusExt};
    use std::process::{Command, Stdio};
    use std::sync::{Arc, Mutex};
    use tracing::field::{Field, Visit};
    use tracing::span::{Attributes, Id, Record};
    use tracing::{Event, Metadata, Subscriber};

    /// Records each event as `LEVEL message key=value ...`.
    #[derive(Clone, Default)]
    struct Events(Arc<Mutex<Vec<String>>>);

    struct Line(String);

    impl Visit for Line {
        fn record_debug(&mut self, field: &Field, value: &dyn Debug) {
            let part = match field.name() {
                "message" => format!(" {value:?}"),
                name => format!(" {name}={value:?}"),
            };
            self.0.push_str(&part);
        }
    }

    impl Subscriber for Events {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &Attributes<'_>) -> Id {
            Id::from_u64(1)
        }
        fn record(&self, _: &Id, _: &Record<'_>) {}
        fn record_follows_from(&self, _: &Id, _: &Id) {}
        fn event(&self, event: &Event<'_>) {
            let mut line = Line(event.metadata().level().to_string());
            event.record(&mut line);
            self.0.lock().unwrap().push(line.0);
        }
        fn enter(&self, _: &Id) {}
        fn exit(&self, _: &Id) {}
    }

    /// True while `pid` exists and is not a zombie.
    fn running(pid: u32) -> bool {
        std::fs::read_to_string(format!("/proc/{pid}/stat"))
            .ok()
            .and_then(|stat| Some(stat.rsplit_once(") ")?.1.chars().next()? != 'Z'))
            .unwrap_or(false)
    }

    #[test]
    fn failed_kill_write_warns_once_and_kills_the_process_group() {
        let root = tempfile::tempdir().unwrap();
        // Writing a directory fails with EISDIR, even as root.
        std::fs::create_dir(root.path().join("cgroup.kill")).unwrap();
        let cgroup = ExecCgroup {
            dir: root.path().to_path_buf(),
            procs: File::create(root.path().join("cgroup.procs")).unwrap(),
        };
        let mut child = Command::new("sh")
            .args(["-c", "sleep 600 & echo $!; wait"])
            .stdout(Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap();
        let mut line = String::new();
        BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let grandchild: u32 = line.trim().parse().unwrap();

        let events = Events::default();
        tracing::subscriber::with_default(events.clone(), || {
            cgroup.kill(&child);
            cgroup.kill(&child);
        });

        let deadline = Instant::now() + Duration::from_secs(5);
        while running(grandchild) {
            assert!(
                Instant::now() < deadline,
                "background sleep {grandchild} survived the fallback kill"
            );
            std::thread::yield_now();
        }
        assert_eq!(child.wait().unwrap().signal(), Some(libc::SIGKILL));
        assert_eq!(
            *events.0.lock().unwrap(),
            [format!(
                "WARN cannot write cgroup.kill; a process that left the exec's process group may survive cgroup={} error=Is a directory (os error 21)",
                root.path().display()
            )]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn drop_on_an_async_thread_does_not_wait_for_the_cgroup_to_empty() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("caliche-instance");
        std::fs::create_dir_all(dir.join("exec-1")).unwrap();
        // Killed processes that have not exited yet.
        std::fs::write(dir.join("cgroup.events"), "populated 1\nfrozen 0\n").unwrap();

        let started = Instant::now();
        drop(InstanceCgroup { dir: dir.clone() });
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_secs(2),
            "drop blocked the runtime thread for {elapsed:?}"
        );
        assert_eq!(
            std::fs::read_to_string(dir.join("cgroup.kill")).unwrap(),
            "1"
        );
        // The processes exit. cgroupfs lets rmdir ignore interface files; a
        // tmpfs directory does not, so they go with the processes.
        std::fs::remove_file(dir.join("cgroup.kill")).unwrap();
        std::fs::remove_file(dir.join("cgroup.events")).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        while dir.exists() {
            assert!(
                Instant::now() < deadline,
                "instance cgroup {} was not removed after it emptied",
                dir.display()
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}
