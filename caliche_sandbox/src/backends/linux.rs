#![cfg(target_os = "linux")]

//! Linux Landlock sandbox backend.
//!
//! Runs commands with Landlock filesystem restrictions. Roots are opened once
//! at `create` and the ruleset is built from those fds in the parent. The
//! pre-exec hook only calls `setsid`, `prctl(PR_SET_NO_NEW_PRIVS)`,
//! `landlock_restrict_self` on that ruleset fd, the cgroup `write`, and
//! `close_range` to mark inherited descriptors close-on-exec. `setsid` drops
//! the agent's controlling terminal, so `/dev/tty` is not the user's session.
//! Those calls are async-signal-safe, so a malloc lock
//! held by another thread at `fork` cannot hang the child. The parent closes
//! the ruleset fd after `spawn`. Stdin is `/dev/null`.
//! Fails closed when Landlock cannot be applied on the running kernel. The
//! workspace git metadata stays read-only; `git_guard` describes the two modes.
//! `NetworkPolicy::HostOnly` is refused even when the `landlock-net` feature is
//! off, because that allowlist cannot be expressed and must not fall open.
//!
//! Each exec runs in its own cgroup v2 cgroup, so a timeout kills every
//! process it started, including ones that called `setsid`. Where cgroup v2 is
//! not writable the backend logs once and kills only the exec's process group.
//! The child's own `setsid` makes that group id its pid, so the child is
//! included. A grandchild that calls `setsid` again leaves the group.

use super::cgroup::{ExecCgroup, InstanceCgroup};
use super::git_guard::{self, MountExec, MountPlan};
use super::landlock_restrict::pre_exec_restrict;
use super::landlock_roots::{self, Grant};
use crate::config::{NetworkPolicy, SandboxConfig};
use crate::paths::{read_validated_file, write_validated_file};
use crate::traits::{ExecResult, SandboxBackend, SandboxCapabilities, SandboxInstance};
use anyhow::{Context, Result};
use async_trait::async_trait;
#[cfg(feature = "landlock-net")]
use landlock::AccessNet;
use landlock::{
    ABI, AccessFs, BitFlags, CompatLevel, Compatible, PathBeneath, Ruleset, RulesetAttr,
    RulesetCreated, RulesetCreatedAttr, Scope,
};
use std::collections::HashMap;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::{Duration, Instant};
use tracing::{info, warn};

#[derive(Debug)]
pub struct LinuxLandlockBackend {
    landlock_available: bool,
    network_available: bool,
}

impl LinuxLandlockBackend {
    pub fn new() -> Self {
        Self {
            landlock_available: probe_landlock(),
            network_available: landlock_network_probe(),
        }
    }
}

impl Default for LinuxLandlockBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn warn_no_network_unenforceable_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        warn!("cannot enforce TCP isolation with Landlock; TCP bind and connect remain allowed");
    });
}

fn warn_no_cgroup_once(err: &anyhow::Error) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        warn!(
            "cannot create a cgroup for exec ({err:#}); a timeout kills only the exec's process group, which a process in a new session escapes"
        );
    });
}

fn probe_landlock() -> bool {
    Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(AccessFs::from_read(ABI::V5))
        .and_then(|ruleset| ruleset.create())
        .is_ok()
}

/// True when this kernel accepts a TCP bind and connect ruleset.
///
/// The `landlock-net` feature must be on. A failed probe leaves network open
/// and the capability flag false.
#[must_use]
pub(crate) fn landlock_network_probe() -> bool {
    #[cfg(feature = "landlock-net")]
    {
        Ruleset::default()
            .set_compatibility(CompatLevel::HardRequirement)
            .handle_access(AccessNet::BindTcp | AccessNet::ConnectTcp)
            .and_then(|ruleset| ruleset.create())
            .is_ok()
    }
    #[cfg(not(feature = "landlock-net"))]
    {
        false
    }
}

#[async_trait]
impl SandboxBackend for LinuxLandlockBackend {
    fn capabilities(&self) -> SandboxCapabilities {
        SandboxCapabilities {
            exec: true,
            filesystem_isolation: self.landlock_available,
            tcp_isolation: self.landlock_available && self.network_available,
            resource_limits: false,
            host_mounts: false,
            pooling: false,
            backend_name: "linux_landlock",
        }
    }

    async fn create(
        &self,
        config: &SandboxConfig,
        workspace: &Path,
    ) -> Result<Box<dyn SandboxInstance>> {
        crate::config::validate_sandbox_config(config)?;
        if !self.landlock_available {
            anyhow::bail!(
                "Landlock is not available on this kernel; refusing to run without filesystem isolation"
            );
        }
        let plan = crate::landlock_net::tcp_plan(
            &config.network,
            cfg!(feature = "landlock-net"),
            self.network_available,
        );
        if matches!(plan, crate::landlock_net::TcpPlan::Unsupported) {
            anyhow::bail!(
                "Landlock cannot enforce this network policy; refusing to run with a weaker network rule"
            );
        }
        if matches!(plan, crate::landlock_net::TcpPlan::Unrestricted)
            && matches!(config.network, NetworkPolicy::None)
        {
            warn_no_network_unenforceable_once();
        }
        let roots = landlock_roots::roots(config, workspace)?;
        let git_guard = git_guard::choose(workspace, roots.writable())?;
        info!(
            git_guard = git_guard.mode().label(),
            workspace = %workspace.display(),
            "workspace .git write guard"
        );
        let prepared = landlock_roots::prepare(roots, workspace, git_guard.excluded())?;
        // Declared before `private_temp` so the cgroup drops first and kills
        // children before the private temp directory is removed.
        let cgroup = InstanceCgroup::create()
            .inspect_err(warn_no_cgroup_once)
            .ok()
            .map(Arc::new);
        Ok(Box::new(LinuxLandlockInstance {
            id: uuid::Uuid::now_v7().to_string(),
            config: config.clone(),
            workspace: workspace.to_path_buf(),
            grants: prepared.grants,
            git_mount: git_guard.mount(),
            deny_tcp: crate::landlock_net::tcp_isolation_claimed(plan),
            cgroup,
            private_temp: prepared.private_temp,
        }))
    }
}

#[cfg(test)]
#[path = "linux_fault.rs"]
mod fault;

#[derive(Debug)]
struct LinuxLandlockInstance {
    id: String,
    config: SandboxConfig,
    workspace: PathBuf,
    grants: Vec<Grant>,
    /// Set in `mount_namespace` mode; see `git_guard`.
    git_mount: Option<Arc<MountPlan>>,
    deny_tcp: bool,
    /// `None` when cgroup v2 is unavailable; see the module docs.
    /// Declared before `private_temp` so children are killed first.
    cgroup: Option<Arc<InstanceCgroup>>,
    private_temp: Option<tempfile::TempDir>,
}

#[async_trait]
impl SandboxInstance for LinuxLandlockInstance {
    fn id(&self) -> &str {
        &self.id
    }

    async fn exec(
        &self,
        command: &[&str],
        working_dir: Option<&str>,
        env: Option<&HashMap<String, String>>,
        timeout: Duration,
    ) -> Result<ExecResult> {
        let grants = landlock_roots::duplicate(&self.grants)?;
        let temp_env = self
            .private_temp
            .as_ref()
            .map(|dir| dir.path().to_path_buf());
        let deny_tcp = self.deny_tcp;
        let git_mount = self.git_mount.clone();
        let cgroup = self.cgroup.clone();
        let workspace = self.workspace.clone();
        let cmd: Vec<String> = command.iter().map(|s| (*s).to_string()).collect();
        let cwd = working_dir.map(PathBuf::from).unwrap_or(workspace);
        let env_map = env.cloned().unwrap_or_default();

        tokio::task::spawn_blocking(move || {
            let exec_cgroup = cgroup
                .as_deref()
                .map(InstanceCgroup::exec)
                .transpose()
                .context("create exec cgroup")?;
            run_exec(
                grants,
                temp_env.as_deref(),
                deny_tcp,
                git_mount,
                exec_cgroup.as_ref(),
                &cmd,
                &cwd,
                &env_map,
                timeout,
            )
        })
        .await
        .context("spawn_blocking join")?
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        tokio::task::spawn_blocking({
            let workspace = self.workspace.clone();
            let config = self.config.clone();
            let path = path.to_string();
            move || read_validated_file(&workspace, &config, &path)
        })
        .await
        .context("spawn_blocking join")?
    }

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        let content = content.to_vec();
        tokio::task::spawn_blocking({
            let workspace = self.workspace.clone();
            let config = self.config.clone();
            let path = path.to_string();
            move || write_validated_file(&workspace, &config, &path, &content)
        })
        .await
        .context("spawn_blocking join")?
    }

    async fn is_alive(&self) -> bool {
        true
    }

    async fn destroy(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(all(test, feature = "sandbox-linux", target_os = "linux"))]
mod tests {
    include!("../../tests/common/mod.rs");

    use super::*;
    use crate::config::{NamedPolicy, SandboxBackendType, policy_to_config};
    use std::sync::{Mutex, MutexGuard};

    #[test]
    fn zero_and_unset_do_not_require_landlock() {
        assert!(!value_is_required(None), "unset must skip");
        assert!(!value_is_required(Some("0")), "0 must skip");
        assert!(value_is_required(Some("1")), "1 requires Landlock");
    }

    fn serialize_exec() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|err| err.into_inner())
    }

    #[tokio::test]
    #[expect(
        clippy::await_holding_lock,
        reason = "serializes execs that share the process-wide fault flag"
    )]
    async fn exec_echo_ok() {
        let _guard = serialize_exec();
        if !probe_landlock() {
            return;
        }
        let workspace = std::env::temp_dir();
        let config = policy_to_config(
            NamedPolicy::ReadOnly,
            &workspace,
            None,
            SandboxBackendType::LinuxLandlock {
                policy: NamedPolicy::ReadOnly,
            },
        );
        let backend = LinuxLandlockBackend::new();
        let instance = backend.create(&config, workspace.as_path()).await.unwrap();
        let result = instance
            .exec(&["sh", "-c", "echo ok"], None, None, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.trim().contains("ok"));
    }

    #[tokio::test]
    #[expect(
        clippy::await_holding_lock,
        reason = "serializes execs that share the process-wide fault flag"
    )]
    async fn restrict_self_fault_does_not_run_the_command() {
        let _guard = serialize_exec();
        if !probe_landlock() {
            require_landlock();
            return;
        }
        let outside = tempfile::tempdir().expect("outside");
        let marker = outside.path().join("marker");
        let workspace = tempfile::tempdir().expect("workspace");
        let config = policy_to_config(
            NamedPolicy::ReadOnly,
            workspace.path(),
            None,
            SandboxBackendType::LinuxLandlock {
                policy: NamedPolicy::ReadOnly,
            },
        );
        let backend = LinuxLandlockBackend::new();
        let instance = backend
            .create(&config, workspace.path())
            .await
            .expect("create");
        super::fault::FAULT_RULESET_FD.store(true, std::sync::atomic::Ordering::Relaxed);
        let _reset = FaultReset;
        let script = format!("echo pwned > '{}'", marker.display());
        let result = instance
            .exec(
                &["/bin/sh", "-c", &script],
                None,
                None,
                Duration::from_secs(5),
            )
            .await;
        let created = marker.exists();
        assert!(
            result.is_err() && !created,
            "restrict_self failure must fail closed and leave no marker, result={result:?} marker={created}"
        );
    }

    struct FaultReset;

    impl Drop for FaultReset {
        fn drop(&mut self) {
            super::fault::FAULT_RULESET_FD.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[tokio::test]
    #[expect(
        clippy::await_holding_lock,
        reason = "serializes execs that share the process-wide fault flag"
    )]
    async fn ruleset_fd_is_absent_after_exec_even_if_cloexec_was_cleared() {
        let _guard = serialize_exec();
        if !probe_landlock() {
            require_landlock();
            return;
        }
        let workspace = tempfile::tempdir().expect("workspace");
        let config = policy_to_config(
            NamedPolicy::ReadOnly,
            workspace.path(),
            None,
            SandboxBackendType::LinuxLandlock {
                policy: NamedPolicy::ReadOnly,
            },
        );
        let backend = LinuxLandlockBackend::new();
        let instance = backend
            .create(&config, workspace.path())
            .await
            .expect("create");
        super::fault::CLEAR_RULESET_CLOEXEC.store(true, std::sync::atomic::Ordering::Relaxed);
        let _reset = ClearReset;
        let result = instance
            .exec(
                &[
                    "/bin/sh",
                    "-c",
                    "if [ -e \"/proc/self/fd/$CLS_RULESET_FD\" ]; then echo PRESENT; else echo ABSENT; fi",
                ],
                None,
                None,
                Duration::from_secs(5),
            )
            .await
            .expect("exec");
        assert_eq!(result.exit_code, 0, "{}", result.stderr);
        assert!(
            result.stdout.contains("ABSENT"),
            "ruleset fd survived exec: {}",
            result.stdout
        );
    }

    struct ClearReset;

    impl Drop for ClearReset {
        fn drop(&mut self) {
            super::fault::CLEAR_RULESET_CLOEXEC.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    }

    #[cfg(feature = "landlock-net")]
    #[tokio::test]
    #[expect(
        clippy::await_holding_lock,
        reason = "serializes execs that share the process-wide fault flag"
    )]
    #[cfg_attr(
        not(all(sandbox_landlock_tcp_available, sandbox_python3_available)),
        ignore = "skipped: Landlock or python3 is missing"
    )]
    async fn tcp_connect_is_denied() {
        let _guard = serialize_exec();
        assert!(
            probe_landlock() && landlock_network_probe(),
            "Landlock TCP rules are not available"
        );
        assert!(python3_available(), "python3 is not available");
        let listener =
            std::net::TcpListener::bind("127.0.0.1:0").expect("bind live 127.0.0.1 listener");
        let port = listener.local_addr().expect("listener address").port();
        let workspace = std::env::temp_dir();
        let config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            &workspace,
            None,
            SandboxBackendType::LinuxLandlock {
                policy: NamedPolicy::WorkspaceWrite,
            },
        );
        let backend = LinuxLandlockBackend::new();
        assert!(backend.capabilities().tcp_isolation);
        let instance = backend
            .create(&config, workspace.as_path())
            .await
            .expect("create");
        // A raw string keeps the Python indentation. A `\` continuation would
        // strip it and raise IndentationError before connect runs.
        let script = format!(
            r#"import errno
import socket
import sys
sock = socket.socket()
sock.settimeout(2)
try:
    sock.connect(("127.0.0.1", {port}))
except OSError as exc:
    if exc.errno == errno.EACCES:
        raise SystemExit(0)
    sys.stderr.write("connect errno=%s: %s\n" % (exc.errno, exc))
    raise SystemExit(2)
raise SystemExit(1)
"#
        );
        let result = instance
            .exec(
                &["python3", "-c", &script],
                None,
                None,
                Duration::from_secs(5),
            )
            .await
            .expect("exec");
        assert_eq!(
            result.exit_code, 0,
            "expected EACCES connecting to 127.0.0.1:{port}; stdout: {} stderr: {}",
            result.stdout, result.stderr
        );
        // Keep the listener alive until connect has been attempted.
        drop(listener);
    }

    #[cfg(feature = "landlock-net")]
    fn python3_available() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }
}

/// Build a Landlock ruleset from fds opened at `create`.
///
/// The ruleset fd is `CLOEXEC`. The caller keeps it open across `fork` and
/// closes it in the parent after `spawn`.
fn build_ruleset(grants: &[Grant], deny_tcp: bool) -> Result<OwnedFd> {
    let abi = ABI::V5;
    let read_access = AccessFs::from_read(abi);
    let write_access = AccessFs::from_write(abi);
    let mut builder = Ruleset::default()
        .set_compatibility(CompatLevel::HardRequirement)
        .handle_access(read_access | write_access)?;
    if deny_tcp {
        builder = handle_tcp_deny(builder)?;
    }
    builder = scope_best_effort(builder)?;
    let ruleset = builder.create()?;
    let ruleset = add_grant_rules(ruleset, grants, read_access, write_access)?;
    let owned: Option<OwnedFd> = ruleset.into();
    let owned = owned.context(
        "Landlock did not return a ruleset fd; refusing to run without filesystem isolation",
    )?;
    ensure_cloexec(owned.as_raw_fd())?;
    Ok(owned)
}

/// Best-effort IPC and signal scopes. Filesystem and TCP stay hard requirements.
fn scope_best_effort(builder: Ruleset) -> Result<Ruleset> {
    let scoped = builder
        .set_compatibility(CompatLevel::BestEffort)
        .scope(Scope::AbstractUnixSocket | Scope::Signal)
        .context("landlock scope")?;
    Ok(scoped.set_compatibility(CompatLevel::HardRequirement))
}

fn add_grant_rules(
    mut ruleset: RulesetCreated,
    grants: &[Grant],
    read_access: BitFlags<AccessFs>,
    write_access: BitFlags<AccessFs>,
) -> Result<RulesetCreated> {
    for grant in grants {
        let access = grant_access(&grant.fd, grant.writable, read_access, write_access);
        ruleset = ruleset
            .add_rule(PathBeneath::new(&grant.fd, access))
            .context("add landlock path rule")?;
    }
    Ok(ruleset)
}

fn grant_access(
    fd: &OwnedFd,
    writable: bool,
    read_access: BitFlags<AccessFs>,
    write_access: BitFlags<AccessFs>,
) -> BitFlags<AccessFs> {
    let mut access = read_access;
    if writable {
        access |= write_access;
    }
    if !landlock_roots::is_directory(fd) {
        access &= AccessFs::ReadFile
            | AccessFs::WriteFile
            | AccessFs::Execute
            | AccessFs::Truncate
            | AccessFs::IoctlDev;
    }
    // `IoctlDev` on a character device allows `TIOCSTI`. Device roots stay
    // readable or writable without that right.
    if landlock_roots::is_char_device(fd) {
        access.remove(AccessFs::IoctlDev);
    }
    access
}

fn child_ruleset_fd(fd: RawFd) -> RawFd {
    #[cfg(test)]
    {
        fault::child_ruleset_fd(fd)
    }
    #[cfg(not(test))]
    {
        fd
    }
}

/// Set `FD_CLOEXEC` so the executed program does not inherit the ruleset fd.
fn ensure_cloexec(fd: std::os::fd::RawFd) -> Result<()> {
    #[expect(unsafe_code, reason = "fcntl on a ruleset fd owned by this process")]
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        anyhow::bail!(
            "fcntl F_GETFD on the Landlock ruleset: {}",
            std::io::Error::last_os_error()
        );
    }
    #[expect(unsafe_code, reason = "fcntl on a ruleset fd owned by this process")]
    let set = unsafe { libc::fcntl(fd, libc::F_SETFD, flags | libc::FD_CLOEXEC) };
    if set < 0 {
        anyhow::bail!(
            "fcntl F_SETFD on the Landlock ruleset: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

#[cfg(feature = "landlock-net")]
fn handle_tcp_deny(builder: Ruleset) -> Result<Ruleset> {
    builder
        .handle_access(AccessNet::BindTcp | AccessNet::ConnectTcp)
        .map_err(Into::into)
}

#[cfg(not(feature = "landlock-net"))]
fn handle_tcp_deny(builder: Ruleset) -> Result<Ruleset> {
    drop(builder);
    anyhow::bail!("landlock-net feature is off; refusing to claim a TCP deny");
}

fn apply_temp_env(cmd: &mut std::process::Command, temp_env: Option<&Path>) -> Result<()> {
    let Some(path) = temp_env else {
        return Ok(());
    };
    let text = path.to_str().context("private temp path is not UTF-8")?;
    cmd.env("TMPDIR", text);
    cmd.env("TMP", text);
    cmd.env("TEMP", text);
    Ok(())
}

fn run_exec(
    grants: Vec<Grant>,
    temp_env: Option<&Path>,
    deny_tcp: bool,
    git_mount: Option<Arc<MountPlan>>,
    cgroup: Option<&ExecCgroup>,
    command: &[String],
    cwd: &Path,
    env: &HashMap<String, String>,
    timeout: Duration,
) -> Result<ExecResult> {
    let shell = command.first().map(String::as_str).unwrap_or("sh");
    // Finish the ruleset before fork. `ruleset` must stay alive until `spawn`
    // returns; the closure only copies the fd number. Root fds are duplicates
    // and close before fork.
    let ruleset = build_ruleset(&grants, deny_tcp)?;
    drop(grants);
    let ruleset_fd = child_ruleset_fd(ruleset.as_raw_fd());
    let cgroup_procs = cgroup.map(ExecCgroup::procs_fd);
    let git_mount = git_mount
        .map(|plan| MountExec::new(plan, cwd))
        .transpose()?;

    let mut cmd = std::process::Command::new(shell);
    cmd.args(command.get(1..).unwrap_or(&[]));
    cmd.current_dir(cwd);
    cmd.env_clear();
    cmd.envs(env);
    apply_temp_env(&mut cmd, temp_env)?;
    #[cfg(test)]
    fault::note_ruleset_fd(&mut cmd, ruleset_fd);
    cmd.stdin(std::process::Stdio::null());
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    // No `process_group(0)`. std would call `setpgid` before `pre_exec`, the
    // child would already be a group leader, and `setsid` would fail with
    // EPERM. `setsid` in the hook makes the process group id the child pid,
    // so a timeout without a cgroup can still `kill(-pid)`.

    #[expect(
        unsafe_code,
        reason = "pre_exec runs in the forked child and must stay async-signal-safe"
    )]
    // SAFETY: `ruleset_fd` and `cgroup_procs` stay open in this process until
    // `spawn` returns, and the closure owns `git_mount`. The closure calls
    // `pre_exec_restrict`, which only issues `setsid`, `write`, the
    // namespace and mount syscalls in `git_guard`, `prctl`,
    // `landlock_restrict_self`, and `close_range`. A non-zero errno fails
    // `spawn` closed.
    unsafe {
        cmd.pre_exec(move || pre_exec_restrict(cgroup_procs, git_mount.as_ref(), ruleset_fd));
    }

    let started = Instant::now();
    let spawned = cmd.spawn();
    drop(ruleset);
    let child = spawned.context("spawn")?;
    super::child_output::collect(child, started, timeout, |child| match cgroup {
        Some(cgroup) => cgroup.kill(child),
        None => super::child_output::kill_process_group(child),
    })
}
