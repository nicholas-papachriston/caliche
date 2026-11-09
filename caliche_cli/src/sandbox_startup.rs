//! The sandbox the CLI starts with.

use anyhow::Context;
use caliche_sandbox::{
    DeferredInstance, NamedPolicy, SandboxBackendType, SandboxConfig, SandboxHandle,
    SandboxInstance, create_backend, policy_to_config, validate_sandbox_config,
};
use std::path::Path;
use std::str::FromStr;

/// The policy when `--sandbox` is not given.
pub const DEFAULT_POLICY: &str = "workspace-write";

/// Give `service` the sandbox for `policy` (default [`DEFAULT_POLICY`]) and
/// `backend` (default `native`) in the workspace: the git root above the
/// current directory, or the current directory. The REPL, `-p`, and
/// `--pipe` all start their sandbox here.
///
/// Returns `false` when `policy` is `danger-full-access`: tools then run on
/// the host without a sandbox, as that policy asks.
///
/// # Errors
///
/// Returns an error for an unknown policy or backend, or when the backend
/// cannot start. A workspace the sandbox refuses is not an error here; each
/// sandboxed tool returns the refusal (see [`startup_handle`]).
pub async fn configure_sandbox(
    service: &mut caliche::AgentService,
    policy: Option<&str>,
    backend: Option<&str>,
) -> anyhow::Result<bool> {
    let policy_str = policy.unwrap_or(DEFAULT_POLICY);
    let policy = NamedPolicy::from_str(policy_str).map_err(|_| {
        eprintln!(
            "\x1b[33mUnknown sandbox policy '{policy_str}'. Use: read-only, workspace-write, network-restricted, danger-full-access\x1b[0m"
        );
        anyhow::anyhow!("Unknown sandbox policy: {policy_str}")
    })?;
    if policy == NamedPolicy::DangerFullAccess {
        tracing::warn!(
            "Sandbox policy 'danger-full-access': running without sandbox (explicit opt-in)"
        );
        caliche::agent::tools::common::set_tool_host_fs_unlocked(true);
        return Ok(false);
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let workspace_root = crate::utils::find_git_root(&cwd).unwrap_or_else(|| cwd.clone());
    let backend_choice = backend.unwrap_or("native");
    let backend = sandbox_backend(backend_choice, policy)?;
    let config = policy_to_config(policy, &workspace_root, None, backend);
    validate_sandbox_config(&config)?;
    let handle = startup_handle(&config, &workspace_root)
        .await
        .with_context(|| {
            format!(
                "sandbox backend '{backend_choice}' failed to initialize; \
                 use --sandbox-backend noop only with explicit danger-full-access, or choose docker"
            )
        })?;
    service.set_sandbox(std::sync::Arc::new(handle));
    Ok(true)
}

fn sandbox_backend(
    backend_choice: &str,
    policy: NamedPolicy,
) -> anyhow::Result<SandboxBackendType> {
    match backend_choice {
        "native" => caliche_sandbox::native_backend_for_platform(policy),
        "wasm" => Ok(SandboxBackendType::WasmWasmtime {
            policy,
            modules: std::collections::HashMap::new(),
        }),
        "noop" => Ok(SandboxBackendType::Noop),
        "docker" => Ok(SandboxBackendType::Docker {
            image: "caliche-sandbox:latest".to_string(),
            pool_size: 0,
        }),
        other => Err(anyhow::anyhow!(
            "Unknown sandbox backend '{other}'. Use: native, wasm, noop, docker"
        )),
    }
}

/// The sandbox handle for `workspace`, ready before the first prompt.
///
/// The backend's workspace checks (git indexes, plus the bounded nested
/// `.git` scan on Linux and Docker) run once, when a sandboxed tool first
/// needs the sandbox, not here. When those
/// checks refuse the workspace, each sandboxed tool returns the refusal and
/// the rest of the session keeps working.
///
/// # Errors
///
/// Returns an error when the backend itself cannot start (for example, the
/// Docker daemon is unreachable).
pub async fn startup_handle(
    config: &SandboxConfig,
    workspace: &Path,
) -> anyhow::Result<SandboxHandle> {
    let backend = create_backend(config).await?;
    let capabilities = backend.capabilities();
    let instance: Box<dyn SandboxInstance> = Box::new(DeferredInstance::new(
        backend,
        config.clone(),
        workspace.to_path_buf(),
    ));
    Ok(SandboxHandle::new(instance, config.tool_policy.clone())
        .with_mode(config.mode.clone())
        .with_capabilities(&capabilities))
}
