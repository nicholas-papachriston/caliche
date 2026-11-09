//! Sandbox configuration types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Named sandbox policy (selectable via CLI or config).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NamedPolicy {
    /// No file writes. Linux network isolation is TCP bind and connect only.
    ReadOnly,
    /// Writes in the project dir and a temp dir, .git read-only.
    /// Linux network isolation is TCP bind and connect only, and Linux Landlock
    /// uses a private temp directory instead of shared `/tmp`.
    WorkspaceWrite,
    /// Writes allowed; network only to explicit allowlist.
    NetworkRestricted,
    /// No sandbox (explicit opt-in; log warning).
    DangerFullAccess,
}

impl std::str::FromStr for NamedPolicy {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace(['-', '_'], "").as_str() {
            "readonly" => Ok(Self::ReadOnly),
            "workspacewrite" => Ok(Self::WorkspaceWrite),
            "networkrestricted" => Ok(Self::NetworkRestricted),
            "dangerfullaccess" => Ok(Self::DangerFullAccess),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub mode: SandboxMode,
    pub backend: SandboxBackendType,
    pub resource_limits: ResourceLimits,
    pub tool_policy: ToolPolicy,
    pub network: NetworkPolicy,
    pub mounts: Vec<MountConfig>,
    pub env_vars: HashMap<String, String>,
    pub timeout_seconds: u64,
    /// Writable directory roots (used by OS-native backends). When set, writes outside these are denied.
    #[serde(default)]
    pub writable_roots: Vec<PathBuf>,
    /// When true, allow writes to a temp directory.
    /// macOS grants `/tmp` and `/private/tmp`. Linux Landlock creates a private
    /// directory and sets `TMPDIR` there instead of sharing `/tmp`.
    /// This does not include the host `$TMPDIR`, which on macOS is the user-wide temp tree.
    #[serde(default = "default_tmp_write")]
    pub tmp_write: bool,
    /// Policy name for logging (e.g. "workspace-write").
    #[serde(default)]
    pub policy_name: Option<String>,
}

const fn default_tmp_write() -> bool {
    true
}

/// Directories implied by `tmp_write`: `/tmp`, plus `/private/tmp` on macOS.
///
/// `$TMPDIR` is intentionally omitted. On macOS it is the user-wide temp tree
/// and would make workspace-write equivalent to "write any tempfile".
/// Linux Landlock strips these paths and grants a private directory instead.
#[must_use]
pub fn tmp_writable_roots() -> Vec<PathBuf> {
    let mut roots = vec![PathBuf::from("/tmp")];
    if cfg!(target_os = "macos") {
        roots.push(PathBuf::from("/private/tmp"));
    }
    roots
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SandboxMode {
    Off,
    NonMain,
    All,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SandboxBackendType {
    Docker {
        image: String,
        pool_size: usize,
    },
    Noop,
    #[serde(rename = "macos_seatbelt")]
    MacOSSeatbelt {
        policy: NamedPolicy,
    },
    #[serde(rename = "linux_landlock")]
    LinuxLandlock {
        policy: NamedPolicy,
    },
    #[serde(rename = "windows_job")]
    WindowsJob {
        policy: NamedPolicy,
    },
    #[serde(rename = "wasm_wasmtime")]
    WasmWasmtime {
        policy: NamedPolicy,
        #[serde(default)]
        modules: HashMap<String, PathBuf>,
    },
}

impl Default for SandboxBackendType {
    fn default() -> Self {
        Self::Docker {
            image: "caliche-sandbox:latest".into(),
            pool_size: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub memory_mb: u64,
    pub cpu_shares: u64,
    pub disk_mb: u64,
    pub pids_limit: i64,
    /// Windows Job Object hard CPU cap, as a percentage of the whole machine
    /// (every logical processor), from 1 through 100. `None`, the default,
    /// does not set `JOBOBJECT_CPU_RATE_CONTROL_INFORMATION`, so the job has
    /// no CPU-rate limit. When set, the instance job uses
    /// `JOB_OBJECT_CPU_RATE_CONTROL_ENABLE | JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP`
    /// and `CpuRate = percent * 100`. A 1% cap is that ceiling; measured use
    /// can sit under it. `validate_sandbox_config` rejects a set value on
    /// every backend other than the Windows job. `create_backend` and each
    /// backend `create` call that validation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_rate_hard_cap_percent: Option<u8>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            memory_mb: 512,
            cpu_shares: 1024,
            disk_mb: 1024,
            pids_limit: 100,
            cpu_rate_hard_cap_percent: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolPolicy {
    pub allow: Vec<String>,
    pub deny: Vec<String>,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            allow: vec!["bash".into(), "fs".into(), "calculator".into()],
            deny: vec!["http".into()],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NetworkPolicy {
    None,
    /// Host-only network. The Linux Landlock backend refuses this policy even
    /// when the `landlock-net` feature is off.
    HostOnly,
    Restricted {
        allowed_hosts: Vec<String>,
    },
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MountConfig {
    pub host_path: PathBuf,
    pub container_path: String,
    pub read_only: bool,
}

/// Map the current platform to a native sandbox backend, or error when unsupported.
///
/// # Errors
///
/// Returns an error when the current operating system has no native sandbox backend.
pub fn native_backend_for_platform(policy: NamedPolicy) -> anyhow::Result<SandboxBackendType> {
    if cfg!(target_os = "macos") {
        Ok(SandboxBackendType::MacOSSeatbelt { policy })
    } else if cfg!(target_os = "linux") {
        Ok(SandboxBackendType::LinuxLandlock { policy })
    } else if cfg!(target_os = "windows") {
        Ok(SandboxBackendType::WindowsJob { policy })
    } else {
        Err(anyhow::anyhow!(
            "native sandbox backend is not supported on this operating system; use --sandbox-backend noop or docker"
        ))
    }
}

/// Validate sandbox configuration before backend creation.
///
/// # Errors
///
/// Returns an error when the configuration requests a network restriction that
/// no available backend can enforce, or a CPU hard cap on a backend other
/// than the Windows job.
pub fn validate_sandbox_config(config: &SandboxConfig) -> anyhow::Result<()> {
    if matches!(config.network, NetworkPolicy::Restricted { .. }) {
        anyhow::bail!(
            "network-restricted egress allowlists are not enforced by any sandbox backend; \
             use workspace-write or read-only (no network) instead"
        );
    }
    crate::cpu_rate::cpu_rate_hard_cap(config.resource_limits.cpu_rate_hard_cap_percent)?;
    reject_cpu_cap_without_windows(config)
}

/// Returned when a CPU hard cap is set on a backend other than the Windows job.
pub const CPU_CAP_REQUIRES_WINDOWS_JOB: &str =
    "cpu_rate_hard_cap_percent applies only to the Windows Job Object backend";

pub(crate) fn reject_cpu_cap_without_windows(config: &SandboxConfig) -> anyhow::Result<()> {
    if config.resource_limits.cpu_rate_hard_cap_percent.is_none() {
        return Ok(());
    }
    if matches!(config.backend, SandboxBackendType::WindowsJob { .. }) {
        return Ok(());
    }
    anyhow::bail!(CPU_CAP_REQUIRES_WINDOWS_JOB)
}

/// Build a `SandboxConfig` from a named policy, workspace root, optional network allowlist,
/// and an explicit backend (callers must choose the backend; this helper does not default to noop).
#[must_use]
pub fn policy_to_config(
    policy: NamedPolicy,
    workspace_root: &Path,
    network_allowed_hosts: Option<&[String]>,
    backend: SandboxBackendType,
) -> SandboxConfig {
    let (writable_roots, tmp_write, network, policy_name) = match policy {
        NamedPolicy::ReadOnly => (
            vec![],
            false,
            NetworkPolicy::None,
            Some("read-only".to_string()),
        ),
        NamedPolicy::WorkspaceWrite => {
            let mut roots = vec![workspace_root.to_path_buf()];
            roots.extend(tmp_writable_roots());
            (
                roots,
                true,
                NetworkPolicy::None,
                Some("workspace-write".to_string()),
            )
        }
        NamedPolicy::NetworkRestricted => {
            let mut roots = vec![workspace_root.to_path_buf()];
            roots.extend(tmp_writable_roots());
            let network =
                network_allowed_hosts.map_or(NetworkPolicy::None, |h| NetworkPolicy::Restricted {
                    allowed_hosts: h.to_vec(),
                });
            (roots, true, network, Some("network-restricted".to_string()))
        }
        NamedPolicy::DangerFullAccess => (
            vec![],
            true,
            NetworkPolicy::Full,
            Some("danger-full-access".to_string()),
        ),
    };
    SandboxConfig {
        mode: SandboxMode::All,
        backend,
        resource_limits: ResourceLimits::default(),
        tool_policy: ToolPolicy::default(),
        network,
        mounts: vec![],
        env_vars: HashMap::new(),
        timeout_seconds: 60,
        writable_roots,
        tmp_write,
        policy_name,
    }
}

#[cfg(test)]
mod tests {
    use super::ResourceLimits;

    #[test]
    fn missing_cpu_rate_field_deserializes_as_off() {
        let limits: ResourceLimits = serde_json::from_str(
            r#"{"memory_mb":512,"cpu_shares":1024,"disk_mb":1024,"pids_limit":100}"#,
        )
        .expect("limits");
        assert_eq!(limits.cpu_rate_hard_cap_percent, None);
    }
}
