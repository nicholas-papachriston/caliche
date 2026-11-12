//! Trait-based sandbox isolation for agent tool execution.

pub mod backends;
#[cfg(feature = "code-mode")]
pub mod code_mode;
pub mod config;
mod cpu_rate;
pub mod deferred;
#[cfg(any(unix, windows))]
pub mod git_access;
#[cfg(any(
    feature = "docker",
    all(feature = "sandbox-linux", target_os = "linux"),
    all(feature = "sandbox-macos", target_os = "macos")
))]
mod git_targets;
pub mod handle;
pub mod landlock_net;
#[cfg(any(
    feature = "docker",
    all(feature = "sandbox-linux", target_os = "linux")
))]
mod nested_git;
pub mod ops;
pub mod paths;
pub mod policy;
pub mod traits;

#[cfg(feature = "code-mode")]
pub use code_mode::{
    CodeModeExecuteContext, CodeModeSandboxConfig, run_execute, run_execute_with_api, run_search,
};
pub use config::{
    MountConfig, NamedPolicy, NetworkPolicy, ResourceLimits, SandboxBackendType, SandboxConfig,
    SandboxMode, ToolPolicy, native_backend_for_platform, policy_to_config, tmp_writable_roots,
    validate_sandbox_config,
};
pub use deferred::{DeferredInstance, SANDBOX_UNAVAILABLE};
pub use handle::SandboxHandle;
pub use landlock_net::{TcpPlan, tcp_isolation_claimed, tcp_plan};
pub use ops::SandboxHandleOps;
pub use paths::{
    DirEntryBeneath, EntryKind, HARD_LINK_REFUSAL, external_tool_path, read_validated_file,
    resolve_sandbox_path, validate_read_path, validate_write_path, write_validated_file,
};
#[cfg(any(unix, windows))]
pub use paths::{
    append_file_beneath, create_file_beneath, list_dir_beneath, open_file_beneath, remove_beneath,
};
pub use policy::evaluate_tool_policy;
pub use traits::{
    EXEC_OUTPUT_LIMIT, ExecResult, ExecTimeout, SandboxBackend, SandboxCapabilities,
    SandboxInstance,
};

/// Whether this process can deny TCP bind and connect with Landlock.
///
/// False unless the `landlock-net` feature is on, the OS is Linux, and the
/// kernel accepts a TCP ruleset. This does not change a policy decision by
/// itself. Callers pass the flag into the host capability.
#[must_use]
pub fn landlock_tcp_enforceable() -> bool {
    #[cfg(all(feature = "landlock-net", target_os = "linux"))]
    {
        backends::linux::landlock_network_probe()
    }
    #[cfg(not(all(feature = "landlock-net", target_os = "linux")))]
    {
        false
    }
}

/// Create a sandbox backend from configuration.
///
/// # Errors
///
/// Returns an error when `validate_sandbox_config` rejects `config`, when the
/// selected backend feature is unavailable on the current platform, or when
/// backend initialization fails.
pub async fn create_backend(
    config: &SandboxConfig,
) -> Result<std::sync::Arc<dyn SandboxBackend>, anyhow::Error> {
    // Keep backend creation uniformly asynchronous across feature sets; Docker
    // initialization performs real asynchronous work when that feature is enabled.
    std::future::ready(()).await;
    crate::config::validate_sandbox_config(config)?;
    match &config.backend {
        SandboxBackendType::Docker { image, .. } => {
            #[cfg(feature = "docker")]
            {
                let backend = backends::docker::DockerBackend::new(image).await?;
                Ok(std::sync::Arc::new(backend))
            }
            #[cfg(not(feature = "docker"))]
            {
                let _ = image;
                Err(anyhow::anyhow!(
                    "Docker sandbox requested but 'docker' feature not enabled"
                ))
            }
        }
        SandboxBackendType::Noop => Ok(std::sync::Arc::new(backends::noop::NoopBackend)),
        SandboxBackendType::MacOSSeatbelt { .. } => {
            #[cfg(all(feature = "sandbox-macos", target_os = "macos"))]
            {
                Ok(std::sync::Arc::new(backends::macos::MacOSSeatbeltBackend))
            }
            #[cfg(not(all(feature = "sandbox-macos", target_os = "macos")))]
            {
                Err(anyhow::anyhow!(
                    "macOS Seatbelt sandbox requested but 'sandbox-macos' feature not enabled or not on macOS"
                ))
            }
        }
        SandboxBackendType::LinuxLandlock { .. } => {
            #[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
            {
                Ok(std::sync::Arc::new(
                    backends::linux::LinuxLandlockBackend::new(),
                ))
            }
            #[cfg(not(all(feature = "sandbox-linux", target_os = "linux")))]
            {
                Err(anyhow::anyhow!(
                    "Linux Landlock sandbox requested but 'sandbox-linux' feature not enabled or not on Linux"
                ))
            }
        }
        SandboxBackendType::WindowsJob { .. } => {
            #[cfg(all(feature = "sandbox-windows", target_os = "windows"))]
            {
                Ok(std::sync::Arc::new(
                    backends::windows::WindowsSandboxBackend::new(),
                ))
            }
            #[cfg(not(all(feature = "sandbox-windows", target_os = "windows")))]
            {
                Err(anyhow::anyhow!(
                    "Windows sandbox requested but 'sandbox-windows' feature not enabled or not on Windows"
                ))
            }
        }
        SandboxBackendType::WasmWasmtime { .. } => {
            #[cfg(feature = "sandbox-wasm")]
            {
                let backend = backends::wasm::WasmtimeBackend::new()?;
                Ok(std::sync::Arc::new(backend))
            }
            #[cfg(not(feature = "sandbox-wasm"))]
            {
                Err(anyhow::anyhow!(
                    "WASM Wasmtime sandbox requested but 'sandbox-wasm' feature not enabled"
                ))
            }
        }
    }
}
