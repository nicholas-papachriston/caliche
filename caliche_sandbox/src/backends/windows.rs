//! Windows sandbox backend (Job Objects).
//!
//! Applies Windows Job Object limits (process-tree cleanup, memory cap, active-process
//! cap, and an optional CPU-rate hard cap) and path validation for file operations.
//! Each instance owns a job with the limits; each exec runs in its own job nested
//! inside it, so a timeout terminates only that exec's processes. The CPU-rate
//! cap is off unless `resource_limits.cpu_rate_hard_cap_percent` is set. When it
//! is set, the instance job gets a hard cap of that percent of the whole machine
//! and nested exec jobs run under it. A 1% cap is a ceiling the scheduler can
//! undershoot. If Windows rejects that cap, the instance is not created.
//! The job does not allow breakaway. `CreateProcess` children stay in it.
//! WMI `Win32_Process.Create` and Task Scheduler (`schtasks`) start the new
//! process from a service outside the job, so those children are not capped.
//! Does **not** enforce
//! network egress or kernel-level filesystem isolation; use Docker on Windows
//! for stronger isolation.
//! AppContainer and restricted-token isolation are future work.

use crate::config::{NetworkPolicy, SandboxConfig};
use crate::paths::{read_validated_file, write_validated_file};
use crate::traits::{ExecResult, SandboxBackend, SandboxCapabilities, SandboxInstance};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::ptr::null;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::warn;
use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_CPU_RATE_CONTROL_ENABLE,
    JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP, JOB_OBJECT_LIMIT_ACTIVE_PROCESS,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY,
    JOBOBJECT_BASIC_LIMIT_INFORMATION, JOBOBJECT_CPU_RATE_CONTROL_INFORMATION,
    JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JobObjectCpuRateControlInformation, JobObjectExtendedLimitInformation, SetInformationJobObject,
    TerminateJobObject,
};
use windows_sys::Win32::System::Threading::{
    CREATE_SUSPENDED, IO_COUNTERS, OpenThread, ResumeThread, THREAD_SUSPEND_RESUME,
};

#[derive(Debug)]
pub struct WindowsSandboxBackend {
    job_objects_available: bool,
}

impl WindowsSandboxBackend {
    pub fn new() -> Self {
        Self {
            job_objects_available: probe_job_objects(),
        }
    }
}

impl Default for WindowsSandboxBackend {
    fn default() -> Self {
        Self::new()
    }
}

fn probe_job_objects() -> bool {
    JobObject::empty().is_ok()
}

#[async_trait]
impl SandboxBackend for WindowsSandboxBackend {
    fn capabilities(&self) -> SandboxCapabilities {
        SandboxCapabilities {
            exec: true,
            filesystem_isolation: false,
            tcp_isolation: false,
            resource_limits: self.job_objects_available,
            host_mounts: false,
            pooling: false,
            backend_name: "windows",
        }
    }

    async fn create(
        &self,
        config: &SandboxConfig,
        workspace: &Path,
    ) -> Result<Box<dyn SandboxInstance>> {
        crate::config::validate_sandbox_config(config)?;
        if !self.job_objects_available {
            anyhow::bail!(
                "Windows Job Objects are not available; refusing to run without process isolation"
            );
        }
        if matches!(config.network, NetworkPolicy::Restricted { .. }) {
            anyhow::bail!(
                "network-restricted egress allowlists are not enforced on Windows Job backend"
            );
        }
        let job = JobObject::new(config).context("create Windows Job Object")?;
        Ok(Box::new(WindowsSandboxInstance {
            id: uuid::Uuid::now_v7().to_string(),
            config: config.clone(),
            workspace: workspace.to_path_buf(),
            job: Arc::new(job),
        }))
    }
}

#[derive(Debug)]
struct JobObject {
    handle: OwnedHandle,
}

impl JobObject {
    /// A job with no limits, for one exec. Processes in it outlive the handle;
    /// the instance job they are nested in kills them when it closes.
    fn empty() -> Result<Self> {
        // SAFETY: CreateJobObjectW reads no memory when both pointers are null;
        // it then creates an unnamed job with default security.
        #[expect(unsafe_code, reason = "Win32 FFI call to create the job")]
        let raw = unsafe { CreateJobObjectW(null(), null()) };
        if raw.is_null() {
            anyhow::bail!("CreateJobObjectW failed");
        }
        // SAFETY: `raw` is a non-null job handle from CreateJobObjectW.
        // `OwnedHandle` takes exclusive ownership and closes it on drop.
        #[expect(unsafe_code, reason = "adopt the job handle returned by Win32")]
        let handle = unsafe { OwnedHandle::from_raw_handle(raw) };
        Ok(Self { handle })
    }

    /// The instance job: kills its processes when closed and applies the
    /// memory, process, and optional CPU-rate limits.
    fn new(config: &SandboxConfig) -> Result<Self> {
        let cpu_rate =
            crate::cpu_rate::cpu_rate_hard_cap(config.resource_limits.cpu_rate_hard_cap_percent)?;
        let job = Self::empty()?;

        let memory_bytes = config.resource_limits.memory_mb.saturating_mul(1024 * 1024);
        // `ProcessMemoryLimit` is `SIZE_T` (`usize` in windows-sys 0.59).
        let process_memory_limit = usize::try_from(memory_bytes).unwrap_or(usize::MAX);
        let active_process_limit =
            u32::try_from(config.resource_limits.pids_limit.max(1)).unwrap_or(u32::MAX);

        // windows-sys 0.59 does not implement `Default` for these C structs.
        // Unset numeric fields stay zero, which is the Win32 "no extra limit" value.
        let info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION {
            BasicLimitInformation: JOBOBJECT_BASIC_LIMIT_INFORMATION {
                PerProcessUserTimeLimit: 0,
                PerJobUserTimeLimit: 0,
                LimitFlags: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    | JOB_OBJECT_LIMIT_ACTIVE_PROCESS
                    | JOB_OBJECT_LIMIT_PROCESS_MEMORY,
                MinimumWorkingSetSize: 0,
                MaximumWorkingSetSize: 0,
                ActiveProcessLimit: active_process_limit,
                Affinity: 0,
                PriorityClass: 0,
                SchedulingClass: 0,
            },
            IoInfo: IO_COUNTERS {
                ReadOperationCount: 0,
                WriteOperationCount: 0,
                OtherOperationCount: 0,
                ReadTransferCount: 0,
                WriteTransferCount: 0,
                OtherTransferCount: 0,
            },
            ProcessMemoryLimit: process_memory_limit,
            JobMemoryLimit: 0,
            PeakProcessMemoryUsed: 0,
            PeakJobMemoryUsed: 0,
        };

        let info_len = u32::try_from(std::mem::size_of_val(&info)).unwrap_or(u32::MAX);
        // SAFETY: `job.handle` is a live job we own. `info` is a fully initialized
        // JOBOBJECT_EXTENDED_LIMIT_INFORMATION, the struct this info class
        // expects, `info_len` is its size, and it outlives the call.
        #[expect(unsafe_code, reason = "Win32 FFI call to set job limits")]
        let ok = unsafe {
            SetInformationJobObject(
                job.handle.as_raw_handle(),
                JobObjectExtendedLimitInformation,
                (&raw const info).cast(),
                info_len,
            )
        };
        if ok == 0 {
            anyhow::bail!("SetInformationJobObject failed");
        }
        if let Some(cpu_rate) = cpu_rate {
            set_cpu_rate_hard_cap(&job, cpu_rate)?;
        }

        Ok(job)
    }

    /// Add `child` to the job. If the child is already in a job, this job must
    /// be empty and becomes nested inside that one.
    fn assign(&self, child: &Child) -> Result<()> {
        // SAFETY: `self.handle` is a live job owned by `self`, and `child`
        // borrows an open process handle for the duration of the call.
        #[expect(unsafe_code, reason = "Win32 FFI call to add the child to the job")]
        let ok =
            unsafe { AssignProcessToJobObject(self.handle.as_raw_handle(), child.as_raw_handle()) };
        if ok == 0 {
            anyhow::bail!("assign child process to job object: AssignProcessToJobObject failed");
        }
        Ok(())
    }

    /// Add a child spawned with `CREATE_SUSPENDED` to the job, then let it run.
    fn confine(&self, child: &Child) -> Result<()> {
        self.assign(child)?;
        resume_process(child.id()).context("resume child process")
    }

    /// Kill every process in the job and in jobs nested inside it.
    fn terminate(&self) {
        // SAFETY: `self.handle` is a live job owned by `self`.
        #[expect(unsafe_code, reason = "Win32 FFI call to kill the job's processes")]
        let ok = unsafe { TerminateJobObject(self.handle.as_raw_handle(), 1) };
        if ok == 0 {
            warn!("TerminateJobObject failed; processes may outlive the exec timeout");
        }
    }
}

const _: () = {
    assert!(crate::cpu_rate::CPU_RATE_CONTROL_ENABLE == JOB_OBJECT_CPU_RATE_CONTROL_ENABLE);
    assert!(crate::cpu_rate::CPU_RATE_CONTROL_HARD_CAP == JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP);
};

/// Fill `JOBOBJECT_CPU_RATE_CONTROL_INFORMATION` for a hard cap.
fn cpu_rate_control_information(
    cap: crate::cpu_rate::CpuRateHardCap,
) -> JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
    JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
        ControlFlags: cap.control_flags,
        Anonymous: JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0 {
            CpuRate: cap.cpu_rate,
        },
    }
}

/// Apply a hard CPU cap to `job`. Failure refuses the instance instead of
/// running without the requested limit. A parent cap covers nested exec jobs.
fn set_cpu_rate_hard_cap(job: &JobObject, cap: crate::cpu_rate::CpuRateHardCap) -> Result<()> {
    let info = cpu_rate_control_information(cap);
    let info_len = u32::try_from(std::mem::size_of_val(&info)).unwrap_or(u32::MAX);
    // SAFETY: `job.handle` is a live job we own. `info` is a fully initialized
    // JOBOBJECT_CPU_RATE_CONTROL_INFORMATION, the struct this info class
    // expects, `info_len` is its size, and it outlives the call.
    #[expect(unsafe_code, reason = "Win32 FFI call to set the job CPU hard cap")]
    let ok = unsafe {
        SetInformationJobObject(
            job.handle.as_raw_handle(),
            JobObjectCpuRateControlInformation,
            (&raw const info).cast(),
            info_len,
        )
    };
    if ok == 0 {
        anyhow::bail!(
            "SetInformationJobObject CpuRateControl failed: {}",
            std::io::Error::last_os_error()
        );
    }
    Ok(())
}

/// Spawn `cmd` suspended and run `confine` before the child executes anything.
/// If `confine` fails, kill and reap the child so it never runs unconfined.
fn spawn_confined(cmd: &mut Command, confine: impl FnOnce(&Child) -> Result<()>) -> Result<Child> {
    cmd.creation_flags(CREATE_SUSPENDED);
    let mut child = cmd.spawn().context("spawn")?;
    if let Err(err) = confine(&child) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(err);
    }
    Ok(child)
}

/// `THREADENTRY32::dwSize`. The struct is 28 bytes, so the cast cannot truncate.
const THREADENTRY32_SIZE: u32 = std::mem::size_of::<THREADENTRY32>() as u32;

/// Resume every thread of a child spawned with `CREATE_SUSPENDED`, which is
/// normally just its primary thread. Stable std does not expose the primary
/// thread handle, so find the child's threads in a snapshot.
fn resume_process(pid: u32) -> Result<()> {
    // SAFETY: CreateToolhelp32Snapshot takes no pointers; the process id is
    // ignored for a thread snapshot.
    #[expect(unsafe_code, reason = "Win32 FFI call to snapshot threads")]
    let raw = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
    if raw == INVALID_HANDLE_VALUE {
        anyhow::bail!("CreateToolhelp32Snapshot failed");
    }
    // SAFETY: `raw` is a valid snapshot handle that nothing else owns.
    #[expect(unsafe_code, reason = "adopt the snapshot handle so it is closed")]
    let snapshot = unsafe { OwnedHandle::from_raw_handle(raw) };

    let mut entry = THREADENTRY32 {
        dwSize: THREADENTRY32_SIZE,
        cntUsage: 0,
        th32ThreadID: 0,
        th32OwnerProcessID: 0,
        tpBasePri: 0,
        tpDeltaPri: 0,
        dwFlags: 0,
    };
    let mut resumed = 0_usize;
    // SAFETY: `snapshot` is live and `entry` is a writable THREADENTRY32 whose
    // `dwSize` is set, as Thread32First requires.
    #[expect(unsafe_code, reason = "Win32 FFI call to read the first thread")]
    let mut more = unsafe { Thread32First(snapshot.as_raw_handle(), &raw mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == pid {
            resume_thread(entry.th32ThreadID)?;
            resumed += 1;
        }
        // SAFETY: same handle and entry as the Thread32First call above.
        #[expect(unsafe_code, reason = "Win32 FFI call to read the next thread")]
        let next = unsafe { Thread32Next(snapshot.as_raw_handle(), &raw mut entry) };
        more = next != 0;
    }
    if resumed == 0 {
        anyhow::bail!("no thread found for child process {pid}");
    }
    Ok(())
}

fn resume_thread(thread_id: u32) -> Result<()> {
    // SAFETY: OpenThread takes no pointers and returns null on failure.
    #[expect(unsafe_code, reason = "Win32 FFI call to open the child thread")]
    let raw = unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, thread_id) };
    if raw.is_null() {
        anyhow::bail!("OpenThread failed for thread {thread_id}");
    }
    // SAFETY: `raw` is a non-null thread handle that nothing else owns.
    #[expect(unsafe_code, reason = "adopt the thread handle so it is closed")]
    let thread = unsafe { OwnedHandle::from_raw_handle(raw) };
    // SAFETY: `thread` is live and was opened with THREAD_SUSPEND_RESUME.
    #[expect(unsafe_code, reason = "Win32 FFI call to resume the child thread")]
    let previous = unsafe { ResumeThread(thread.as_raw_handle()) };
    if previous == u32::MAX {
        anyhow::bail!("ResumeThread failed for thread {thread_id}");
    }
    Ok(())
}

#[derive(Debug)]
struct WindowsSandboxInstance {
    id: String,
    config: SandboxConfig,
    workspace: PathBuf,
    job: Arc<JobObject>,
}

#[async_trait]
impl SandboxInstance for WindowsSandboxInstance {
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
        let workspace = self.workspace.clone();
        let config = self.config.clone();
        let job = Arc::clone(&self.job);
        let cmd: Vec<String> = command.iter().map(|s| (*s).to_string()).collect();
        let cwd = working_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| workspace.clone());
        let env_map = env.cloned().unwrap_or_default();

        tokio::task::spawn_blocking(move || run_exec(&config, &job, &cmd, &cwd, &env_map, timeout))
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

fn run_exec(
    _config: &SandboxConfig,
    job: &JobObject,
    command: &[String],
    cwd: &Path,
    env: &HashMap<String, String>,
    timeout: Duration,
) -> Result<ExecResult> {
    let shell = command.first().map(String::as_str).unwrap_or("cmd");
    let mut cmd = Command::new(shell);
    cmd.args(command.get(1..).unwrap_or(&[]));
    cmd.current_dir(cwd);
    cmd.env_clear();
    cmd.envs(env);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());
    let exec_job = JobObject::empty().context("create exec job")?;
    let started = Instant::now();
    let child = spawn_confined(&mut cmd, |child| {
        job.assign(child)?;
        exec_job.confine(child)
    })?;
    super::child_output::collect(child, started, timeout, |_| exec_job.terminate())
}

#[cfg(all(test, feature = "sandbox-windows", target_os = "windows"))]
mod tests {
    use super::*;
    use crate::config::{NamedPolicy, SandboxBackendType, policy_to_config};
    use std::process::Stdio;
    use std::sync::mpsc;
    use windows_sys::Win32::Foundation::WAIT_OBJECT_0;
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION, PROCESS_SYNCHRONIZE,
        PROCESS_TERMINATE, TerminateProcess, WaitForSingleObject,
    };

    /// Exit code `Child::kill` passes to `TerminateProcess`.
    const KILLED: u32 = 1;

    #[test]
    fn cpu_rate_struct_is_an_enabled_hard_cap() {
        use windows_sys::Win32::System::JobObjects::QueryInformationJobObject;

        let cap = crate::cpu_rate::cpu_rate_hard_cap(Some(20))
            .expect("20")
            .expect("some");
        assert_eq!(
            (
                crate::cpu_rate::CPU_RATE_CONTROL_ENABLE,
                crate::cpu_rate::CPU_RATE_CONTROL_HARD_CAP
            ),
            (
                JOB_OBJECT_CPU_RATE_CONTROL_ENABLE,
                JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
            )
        );
        let info = super::cpu_rate_control_information(cap);
        assert_eq!(
            info.ControlFlags,
            JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
        );
        assert_eq!(info.ControlFlags, cap.control_flags);
        // SAFETY: `CpuRate` is the active union field. The hard-cap flags do
        // not select `Weight` or `MinRate`/`MaxRate`.
        #[expect(unsafe_code, reason = "read the CpuRate field of the test struct")]
        let cpu_rate = unsafe { info.Anonymous.CpuRate };
        assert_eq!(cpu_rate, 2_000);
        assert_eq!(std::mem::size_of_val(&info), 8);

        let dir = std::env::temp_dir();
        let mut config = policy_to_config(
            NamedPolicy::ReadOnly,
            &dir,
            None,
            SandboxBackendType::WindowsJob {
                policy: NamedPolicy::ReadOnly,
            },
        );
        config.resource_limits.cpu_rate_hard_cap_percent = Some(20);
        let job = JobObject::new(&config).expect("job with a hard cap");
        let mut queried = JOBOBJECT_CPU_RATE_CONTROL_INFORMATION {
            ControlFlags: 0,
            Anonymous: JOBOBJECT_CPU_RATE_CONTROL_INFORMATION_0 { CpuRate: 0 },
        };
        let queried_len = u32::try_from(std::mem::size_of_val(&queried)).unwrap_or(u32::MAX);
        // SAFETY: `job.handle` is a live job. `queried` is writable storage of
        // the struct this info class returns, and `queried_len` is its size.
        #[expect(unsafe_code, reason = "Win32 FFI call to read the job CPU cap")]
        let ok = unsafe {
            QueryInformationJobObject(
                job.handle.as_raw_handle(),
                JobObjectCpuRateControlInformation,
                (&raw mut queried).cast(),
                queried_len,
                std::ptr::null_mut(),
            )
        };
        assert_ne!(ok, 0, "QueryInformationJobObject CpuRateControl failed");
        assert_eq!(queried.ControlFlags, info.ControlFlags);
        #[expect(unsafe_code, reason = "read the CpuRate field returned by Win32")]
        let queried_rate = unsafe { queried.Anonymous.CpuRate };
        assert_eq!(queried_rate, 2_000);
    }

    #[test]
    fn missing_cpu_rate_creates_a_job_without_rate_control() {
        let dir = std::env::temp_dir();
        let config = policy_to_config(
            NamedPolicy::ReadOnly,
            &dir,
            None,
            SandboxBackendType::WindowsJob {
                policy: NamedPolicy::ReadOnly,
            },
        );
        assert_eq!(config.resource_limits.cpu_rate_hard_cap_percent, None);
        let _job = JobObject::new(&config).expect("job with no CPU cap");
    }

    #[test]
    fn zero_cpu_rate_refuses_the_job() {
        let dir = std::env::temp_dir();
        let mut config = policy_to_config(
            NamedPolicy::ReadOnly,
            &dir,
            None,
            SandboxBackendType::WindowsJob {
                policy: NamedPolicy::ReadOnly,
            },
        );
        config.resource_limits.cpu_rate_hard_cap_percent = Some(0);
        let err = JobObject::new(&config).expect_err("0% is not a hard cap");
        assert!(err.to_string().contains("1 through 100"));
    }

    fn open_process(pid: u32) -> Option<OwnedHandle> {
        let access = PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SYNCHRONIZE | PROCESS_TERMINATE;
        // SAFETY: OpenProcess takes no pointers and returns null on failure.
        #[expect(unsafe_code, reason = "Win32 FFI call to watch a test process")]
        let raw = unsafe { OpenProcess(access, 0, pid) };
        // SAFETY: a non-null `raw` is a process handle that nothing else owns.
        #[expect(unsafe_code, reason = "adopt the process handle so it is closed")]
        (!raw.is_null()).then(|| unsafe { OwnedHandle::from_raw_handle(raw) })
    }

    /// The exit code if the process has already exited, without waiting.
    fn exit_code_now(process: &OwnedHandle) -> Option<u32> {
        // SAFETY: `process` is live and was opened with PROCESS_SYNCHRONIZE.
        #[expect(unsafe_code, reason = "Win32 FFI call to poll a test process")]
        let exited = unsafe { WaitForSingleObject(process.as_raw_handle(), 0) } == WAIT_OBJECT_0;
        let mut code = 0_u32;
        // SAFETY: `process` has PROCESS_QUERY_LIMITED_INFORMATION and `code`
        // is a writable u32.
        #[expect(unsafe_code, reason = "Win32 FFI call to read an exit code")]
        let ok = unsafe { GetExitCodeProcess(process.as_raw_handle(), &raw mut code) } != 0;
        (exited && ok).then_some(code)
    }

    fn terminate(process: &OwnedHandle) {
        // SAFETY: `process` is live and was opened with PROCESS_TERMINATE.
        #[expect(unsafe_code, reason = "Win32 FFI call to clean up after a failed test")]
        let _ = unsafe { TerminateProcess(process.as_raw_handle(), KILLED) };
    }

    #[test]
    fn failed_confine_kills_and_reaps_the_suspended_child() {
        let (watch_tx, watch_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut cmd = Command::new("cmd");
            cmd.args(["/C", "exit 0"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            let spawned = spawn_confined(&mut cmd, |child| {
                let _ = watch_tx.send(open_process(child.id()));
                anyhow::bail!("injected confine failure")
            });
            let _ = done_tx.send(spawned.map(|_| ()).map_err(|err| format!("{err:#}")));
        });

        let child = watch_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("confine ran")
            .expect("open the suspended child");
        let Ok(spawned) = done_rx.recv_timeout(Duration::from_secs(10)) else {
            terminate(&child);
            panic!("spawn_confined never returned: the suspended child was not killed");
        };
        assert_eq!(spawned, Err("injected confine failure".to_owned()));
        assert_eq!(
            exit_code_now(&child),
            Some(KILLED),
            "the child must be killed and reaped before spawn_confined returns"
        );
    }

    #[test]
    fn child_cannot_spawn_outside_the_job_before_it_is_confined() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let pid_file = dir.path().join("grandchild.pid");
        // A file avoids `cmd /C` quote rules. `ping` keeps the child alive for the assign.
        std::fs::write(
            dir.path().join("spawn.cmd"),
            "@echo off\r\n\
             start \"\" /B powershell -NoProfile -NonInteractive -Command \
             \"Set-Content -Path grandchild.pid -Value $PID; Start-Sleep -Seconds 120\"\r\n\
             ping -n 30 127.0.0.1 >NUL\r\n",
        )
        .expect("write spawn script");
        let config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            dir.path(),
            None,
            SandboxBackendType::WindowsJob {
                policy: NamedPolicy::WorkspaceWrite,
            },
        );
        let job = JobObject::new(&config).expect("job");

        let mut cmd = Command::new("cmd");
        cmd.args(["/C", r".\spawn.cmd"])
            .env("NoDefaultCurrentDirectoryInExePath", "1")
            .current_dir(dir.path())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        // A child that already runs would spawn its grandchild during this delay.
        let _child = spawn_confined(&mut cmd, |child| {
            #[expect(clippy::disallowed_methods, reason = "hold the child before confine")]
            std::thread::sleep(Duration::from_secs(2));
            job.confine(child)
        })
        .expect("spawn_confined");

        let deadline = Instant::now() + Duration::from_secs(60);
        let pid = loop {
            let parsed = std::fs::read_to_string(&pid_file)
                .ok()
                .and_then(|text| text.trim().parse::<u32>().ok());
            if let Some(pid) = parsed {
                break pid;
            }
            assert!(Instant::now() < deadline, "grandchild never wrote its pid");
            #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
            std::thread::sleep(Duration::from_millis(200));
        };
        let grandchild = open_process(pid).expect("open the grandchild");
        assert_eq!(
            exit_code_now(&grandchild),
            None,
            "grandchild runs while the job is open"
        );

        drop(job);

        let deadline = Instant::now() + Duration::from_secs(20);
        while exit_code_now(&grandchild).is_none() {
            if Instant::now() >= deadline {
                terminate(&grandchild);
                panic!("grandchild {pid} escaped the job and survived closing it");
            }
            #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    #[test]
    fn failed_wait_terminates_the_exec_job_before_returning() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let pid_file = dir.path().join("grandchild.pid");
        // The grandchild holds the output pipes; `ping` keeps `cmd` running.
        std::fs::write(
            dir.path().join("spawn.cmd"),
            "@echo off\r\n\
             start \"\" /B powershell -NoProfile -NonInteractive -Command \
             \"Set-Content -Path grandchild.pid -Value $PID; Start-Sleep -Seconds 600\"\r\n\
             ping -n 600 127.0.0.1 >NUL\r\n",
        )
        .expect("write spawn script");
        let config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            dir.path(),
            None,
            SandboxBackendType::WindowsJob {
                policy: NamedPolicy::WorkspaceWrite,
            },
        );
        let job = JobObject::empty().expect("exec job");
        let mut cmd = Command::new("cmd");
        cmd.args(["/C", r".\spawn.cmd"])
            .env("NoDefaultCurrentDirectoryInExePath", "1")
            .current_dir(dir.path())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let instance_job = JobObject::new(&config).expect("instance job");
        let child = spawn_confined(&mut cmd, |child| {
            instance_job.assign(child)?;
            job.confine(child)
        })
        .expect("spawn_confined");
        let child_process = open_process(child.id()).expect("open the child");

        let mut grandchild = None;
        let wait_fails_once_the_grandchild_runs = |_: &mut Child| {
            let deadline = Instant::now() + Duration::from_secs(60);
            while grandchild.is_none() && Instant::now() < deadline {
                grandchild = std::fs::read_to_string(&pid_file)
                    .ok()
                    .and_then(|text| text.trim().parse::<u32>().ok())
                    .and_then(open_process);
                #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
                std::thread::sleep(Duration::from_millis(200));
            }
            Err(std::io::Error::other("injected wait failure"))
        };
        let outcome = super::super::child_output::collect_with(
            child,
            Instant::now(),
            Duration::from_secs(600),
            |_| job.terminate(),
            wait_fails_once_the_grandchild_runs,
        );
        assert_eq!(
            outcome.map(|_| ()).map_err(|err| format!("{err:#}")),
            Err("wait on child: injected wait failure".to_owned())
        );

        let grandchild = grandchild.expect("grandchild never wrote its pid");
        let deadline = Instant::now() + Duration::from_secs(20);
        while exit_code_now(&grandchild).is_none() {
            if Instant::now() >= deadline {
                terminate(&grandchild);
                panic!("grandchild survived a failed wait");
            }
            #[expect(clippy::disallowed_methods, reason = "poll in a sync test")]
            std::thread::sleep(Duration::from_millis(200));
        }
        assert_eq!(
            (exit_code_now(&child_process), exit_code_now(&grandchild)),
            (Some(KILLED), Some(KILLED)),
            "a failed wait must terminate the exec job"
        );
    }

    #[tokio::test]
    async fn exec_echo_ok() {
        if !probe_job_objects() {
            return;
        }
        let workspace = std::env::temp_dir();
        let config = policy_to_config(
            NamedPolicy::ReadOnly,
            &workspace,
            None,
            SandboxBackendType::WindowsJob {
                policy: NamedPolicy::ReadOnly,
            },
        );
        let backend = WindowsSandboxBackend::new();
        let instance = backend.create(&config, workspace.as_path()).await.unwrap();
        let result = instance
            .exec(
                &["cmd", "/c", "echo ok"],
                None,
                None,
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert_eq!(result.exit_code, 0);
        assert!(result.stdout.trim().contains("ok"));
    }
}
