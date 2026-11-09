use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "sandbox")]
use anyhow::{Result, bail};
use async_trait::async_trait;
use serde_json::json;
#[cfg(feature = "sandbox")]
use std::ffi::OsStr;
#[cfg(feature = "sandbox")]
use std::path::{Path, PathBuf};
#[cfg(feature = "sandbox")]
use std::sync::OnceLock;

/// Maximum execution time for code (30 seconds)
#[cfg(feature = "sandbox")]
const EXECUTION_TIMEOUT_SECS: u64 = 30;

/// Maximum output size (1 MB)
#[cfg(feature = "sandbox")]
const MAX_OUTPUT_SIZE: usize = 1_048_576;

const NAME: &str = "execute_python";

/// Returned whenever no active OS sandbox confines file access. The tool never runs code
/// on the host, nor under a backend (noop, Windows Job object) that confines nothing.
pub const NO_SANDBOX_ERROR: &str = "execute_python refuses to run without an active OS sandbox \
        that confines file access. Start the agent with a sandbox policy such as read-only or \
        workspace-write on a backend with filesystem isolation (macOS Seatbelt, Linux Landlock, \
        Docker).";

const DESCRIPTION: &str = "Execute Python code inside the agent's OS sandbox. \
        IMPORTANT: Write complete, valid Python code. Only Python standard library is available. \
        The code runs under the same sandbox policy as the bash tool, with a 30-second timeout. \
        Use print() to display output. Without an active sandbox that confines file access, \
        the tool refuses to run. \
        \
        SANDBOX:\n\
        - File reads: system directories, /tmp, the workspace, and the policy's writable roots; \
        the home directory and other user files are denied\n\
        - File writes: only the policy's writable roots (none under read-only)\n\
        - Network: denied unless the sandbox policy allows it\n\
        - Environment: only an allowlist (PATH, HOME, USER, locale, TERM, TMPDIR and similar); \
        variables ending in _KEY, _SECRET or _TOKEN are never passed\n\
        - Isolated Python mode (no user site-packages or PYTHONPATH)\n\
        - 30-second execution timeout and 1 MB output limit\n\
        \
        CAPABILITIES:\n\
        - All Python standard library modules available\n\
        - Math, string, date/time operations\n\
        - JSON, CSV, XML parsing\n\
        \
        GUIDELINES FOR WRITING CODE:\n\
        - Write complete, executable Python code\n\
        - Use print() to show results\n\
        - Only use Python standard library (no pip packages)\n\
        - Handle potential errors with try/except\n\
        - Keep execution under 30 seconds\n\
        \
        EXAMPLES:\n\
        Good: print(sum([1, 2, 3, 4, 5]))\n\
        Good: import math\\nprint(math.sqrt(144))\n\
        Good: for i in range(5):\\n    print(i * 2)\n\
        Bad: import numpy (not in stdlib)\n\
        Bad: Missing print() statement";

/// Locate `python3` once: the first match in an absolute `PATH` entry.
#[cfg(feature = "sandbox")]
static PYTHON3: OnceLock<Option<PathBuf>> = OnceLock::new();

#[cfg(all(feature = "sandbox", windows))]
const PYTHON3_FILE: &str = "python3.exe";
#[cfg(all(feature = "sandbox", not(windows)))]
const PYTHON3_FILE: &str = "python3";

/// Resolve `python3` from `path_var` without consulting the working directory.
///
/// Relative and empty `PATH` entries are skipped, because they resolve against the
/// working directory. An interpreter inside `workspace` is skipped too: a repository
/// must not be able to supply the interpreter that runs agent code. The search
/// fails only when every `python3` on `PATH` is inside the workspace.
///
/// On macOS, `/usr/bin/python3` is an xcrun shim. It is replaced by the
/// interpreter it would run, because the shim itself cannot work in the sandbox.
#[cfg(feature = "sandbox")]
fn resolve_python3(path_var: &OsStr, workspace: &Path) -> Result<PathBuf> {
    let workspace = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    let mut skipped = None;
    for dir in std::env::split_paths(path_var).filter(|dir| dir.is_absolute()) {
        let candidate = dir.join(PYTHON3_FILE);
        if !candidate.is_file() {
            continue;
        }
        let resolved = candidate.canonicalize().unwrap_or(candidate);
        #[cfg(target_os = "macos")]
        let resolved = unshim_python3(resolved)?;
        if resolved.starts_with(&workspace) {
            skipped.get_or_insert(resolved);
            continue;
        }
        return Ok(resolved);
    }
    match skipped {
        Some(inside) => bail!(
            "Refusing to run python3 from inside the workspace, and no other python3 is on PATH: {}",
            inside.display()
        ),
        None => bail!("Python 3 is not available on this system. Please install Python 3."),
    }
}

/// The xcrun tool shim. It writes a cache under the per-user temp directory
/// and loads `libxcrun` from the developer directory, so it fails inside the
/// sandbox.
#[cfg(all(feature = "sandbox", target_os = "macos"))]
const PYTHON3_SHIM: &str = "/usr/bin/python3";

/// The interpreter the shim would run, found by the trusted `/usr/bin/xcrun`
/// outside the sandbox. Any other path is returned unchanged.
#[cfg(all(feature = "sandbox", target_os = "macos"))]
fn unshim_python3(resolved: PathBuf) -> Result<PathBuf> {
    if resolved != Path::new(PYTHON3_SHIM) {
        return Ok(resolved);
    }
    let output = std::process::Command::new("/usr/bin/xcrun")
        .args(["--find", "python3"])
        .env_clear()
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to run /usr/bin/xcrun: {e}"))?;
    if !output.status.success() {
        bail!("{PYTHON3_SHIM} needs developer tools, and none with python3 are selected");
    }
    let found = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    Ok(found.canonicalize().unwrap_or(found))
}

/// The interpreter path from the process `PATH`, resolved on first use.
#[cfg(feature = "sandbox")]
fn python3_path() -> Result<PathBuf> {
    let resolved = PYTHON3.get_or_init(|| {
        let path_var = std::env::var_os("PATH")?;
        let workspace = std::env::current_dir().ok()?;
        resolve_python3(&path_var, &workspace)
            .inspect_err(|e| tracing::warn!("python3 unavailable: {e}"))
            .ok()
    });
    resolved.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "Python 3 is not available outside the workspace. Please install Python 3 on PATH."
        )
    })
}

#[derive(Debug)]
pub struct CodeExecutionTool;

impl ToolMeta for CodeExecutionTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python code to execute. Must be valid Python syntax. Use print() to display output. Only standard library available."
                }
            },
            "required": ["code"]
        })
    }
}

#[async_trait]
impl Tool for CodeExecutionTool {
    /// Without a sandbox handle the tool fails closed.
    async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
        ToolResult::error(NO_SANDBOX_ERROR.to_string())
    }

    #[cfg(feature = "sandbox")]
    async fn execute_sandboxed(
        &self,
        arguments: serde_json::Value,
        sandbox: Option<&dyn caliche_sandbox::SandboxHandleOps>,
    ) -> ToolResult {
        let Some(sandbox) =
            sandbox.filter(|handle| handle.sandbox_active() && handle.confines_filesystem())
        else {
            return ToolResult::error(NO_SANDBOX_ERROR.to_string());
        };
        let code = params::extract_string_or(&arguments, "code", "");
        if code.is_empty() {
            return ToolResult::error(
                "Missing required parameter 'code'. Please provide Python code to execute."
                    .to_string(),
            );
        }

        tracing::info!(
            "Executing Python code in the sandbox ({} chars)",
            code.len()
        );
        tracing::debug!("Code to execute:\n{}", code);

        match run_in_sandbox(sandbox, &code).await {
            Ok(output) => {
                tracing::info!("Code execution successful ({} bytes output)", output.len());
                ToolResult::success(output)
            }
            Err(e) => {
                tracing::warn!("Code execution failed: {}", e);
                ToolResult::error(format!("Code execution failed: {e}"))
            }
        }
    }
}

/// Run the code with the absolute `python3 -I -E -s` through the sandbox,
/// with the bash tool's env allowlist.
#[cfg(feature = "sandbox")]
async fn run_in_sandbox(
    sandbox: &dyn caliche_sandbox::SandboxHandleOps,
    code: &str,
) -> Result<String> {
    let python3 = python3_path()?;
    let python3 = python3.to_string_lossy();
    let env = crate::agent::tools::bash::sandbox_env_from_host();
    let timeout = std::time::Duration::from_secs(EXECUTION_TIMEOUT_SECS);
    let command = [python3.as_ref(), "-I", "-E", "-s", "-c", code];
    let result = match sandbox.exec(&command, None, Some(&env), timeout).await {
        Ok(result) => result,
        Err(e) if e.downcast_ref::<caliche_sandbox::ExecTimeout>().is_some() => {
            anyhow::bail!(
                "Code execution timed out after {EXECUTION_TIMEOUT_SECS} seconds. Consider optimizing your code or reducing iterations."
            );
        }
        Err(e) => return Err(e.context("Failed to run Python in the sandbox")),
    };
    if result.stdout_truncated || result.stdout.len() > MAX_OUTPUT_SIZE {
        anyhow::bail!("Output too large (max {MAX_OUTPUT_SIZE} bytes). Consider reducing output.");
    }
    format_output(result.exit_code, &result.stdout, &result.stderr)
}

#[cfg(feature = "sandbox")]
fn format_output(exit_code: i64, stdout: &str, stderr: &str) -> anyhow::Result<String> {
    let mut output = String::from(stdout);
    if !stderr.is_empty() {
        if !output.is_empty() {
            output.push_str("\n\n");
        }
        output.push_str("Errors/Warnings:\n");
        output.push_str(stderr);
    }

    if exit_code != 0 {
        if output.is_empty() {
            anyhow::bail!("Code execution failed with exit code {exit_code}. No output captured.");
        }
        anyhow::bail!("Code execution failed with exit code {exit_code}:\n{output}");
    }

    if output.is_empty() {
        Ok(
            "Code executed successfully with no output. (Tip: Use print() to display results)"
                .to_string(),
        )
    } else {
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn refuses_without_a_sandbox() {
        let result = CodeExecutionTool
            .execute(json!({ "code": "print('host')" }))
            .await;
        assert_eq!(
            (result.success, result.error.as_deref()),
            (false, Some(NO_SANDBOX_ERROR))
        );
    }

    #[cfg(feature = "sandbox")]
    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(path, "").expect("write");
    }

    #[cfg(feature = "sandbox")]
    #[test]
    fn skips_python3_inside_the_workspace_for_one_further_on_path() {
        let workspace = tempfile::tempdir().expect("workspace");
        let system = tempfile::tempdir().expect("system");
        let inside = workspace.path().join(".venv/bin");
        touch(&inside.join(PYTHON3_FILE));
        touch(&system.path().join(PYTHON3_FILE));
        let path_var = std::env::join_paths([inside, system.path().to_path_buf()]).expect("join");
        let resolved = resolve_python3(&path_var, workspace.path()).expect("python3");
        assert_eq!(
            resolved,
            system
                .path()
                .join(PYTHON3_FILE)
                .canonicalize()
                .expect("canonical")
        );
    }

    #[cfg(feature = "sandbox")]
    #[test]
    fn refuses_python3_only_inside_the_workspace() {
        let workspace = tempfile::tempdir().expect("workspace");
        let bin = workspace.path().join("tools/bin");
        touch(&bin.join(PYTHON3_FILE));
        let error = resolve_python3(bin.as_os_str(), workspace.path())
            .expect_err("interpreter inside the workspace must be refused");
        assert!(
            error.to_string().starts_with(
                "Refusing to run python3 from inside the workspace, and no other python3 is on PATH"
            ),
            "{error}"
        );
    }

    #[cfg(all(feature = "sandbox", target_os = "macos"))]
    #[test]
    fn resolves_the_xcrun_shim_to_the_interpreter_it_runs() {
        let workspace = tempfile::tempdir().expect("workspace");
        let resolved = resolve_python3(OsStr::new("/usr/bin"), workspace.path());
        let found = std::process::Command::new("/usr/bin/xcrun")
            .args(["--find", "python3"])
            .env_clear()
            .output()
            .expect("xcrun");
        if !found.status.success() {
            assert_eq!(
                resolved.expect_err("no developer tools").to_string(),
                "/usr/bin/python3 needs developer tools, and none with python3 are selected"
            );
            return;
        }
        let expected = PathBuf::from(String::from_utf8_lossy(&found.stdout).trim())
            .canonicalize()
            .expect("canonical");
        assert_eq!(resolved.expect("python3"), expected);
        assert_ne!(expected, Path::new(PYTHON3_SHIM));
    }

    // The relative entry climbs from the current directory to the root, which
    // cannot reach another drive on Windows.
    #[cfg(all(feature = "sandbox", unix))]
    #[test]
    fn skips_relative_path_entries() {
        let planted = tempfile::tempdir().expect("planted");
        let system = tempfile::tempdir().expect("system");
        let workspace = tempfile::tempdir().expect("workspace");
        touch(&planted.path().join(PYTHON3_FILE));
        touch(&system.path().join(PYTHON3_FILE));
        // A relative entry that reaches the planted directory from the current directory.
        let cwd = std::env::current_dir().expect("cwd");
        let up = "../".repeat(cwd.components().count());
        let relative = PathBuf::from(format!(
            "{up}{}",
            planted.path().display().to_string().trim_start_matches('/')
        ));
        assert!(relative.join(PYTHON3_FILE).is_file());
        let path_var = std::env::join_paths([relative, system.path().to_path_buf()]).expect("join");
        let resolved = resolve_python3(&path_var, workspace.path()).expect("python3");
        assert_eq!(
            resolved,
            system
                .path()
                .join(PYTHON3_FILE)
                .canonicalize()
                .expect("canonical")
        );
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn refuses_a_missing_sandbox_handle() {
        let result = CodeExecutionTool
            .execute_sandboxed(json!({ "code": "print('host')" }), None)
            .await;
        assert_eq!(
            (result.success, result.error.as_deref()),
            (false, Some(NO_SANDBOX_ERROR))
        );
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    async fn refuses_a_backend_without_filesystem_isolation() {
        use caliche_sandbox::backends::noop::NoopBackend;
        use caliche_sandbox::{
            NamedPolicy, SandboxBackend, SandboxBackendType, SandboxHandle, ToolPolicy,
            policy_to_config,
        };
        let workspace = tempfile::tempdir().expect("workspace");
        let config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            workspace.path(),
            None,
            SandboxBackendType::Noop,
        );
        let instance = NoopBackend
            .create(&config, workspace.path())
            .await
            .expect("noop sandbox");
        let handle = SandboxHandle::new(instance, ToolPolicy::default())
            .with_capabilities(&NoopBackend.capabilities());
        let result = CodeExecutionTool
            .execute_sandboxed(json!({ "code": "print('host')" }), Some(&handle))
            .await;
        assert_eq!(
            (result.success, result.error.as_deref()),
            (false, Some(NO_SANDBOX_ERROR))
        );
    }

    #[cfg(all(
        feature = "sandbox",
        any(target_os = "macos", target_os = "linux", target_os = "windows")
    ))]
    mod native {
        use super::*;
        use caliche_sandbox::{
            NamedPolicy, SandboxCapabilities, SandboxHandle, SandboxMode, ToolPolicy,
            create_backend, native_backend_for_platform, policy_to_config,
        };
        use std::path::Path;

        async fn native_handle(workspace: &Path) -> (SandboxHandle, SandboxCapabilities) {
            let backend_type = native_backend_for_platform(NamedPolicy::WorkspaceWrite)
                .expect("native backend for this platform");
            let config =
                policy_to_config(NamedPolicy::WorkspaceWrite, workspace, None, backend_type);
            let backend = create_backend(&config)
                .await
                .expect("native sandbox backend");
            let capabilities = backend.capabilities();
            let instance = backend
                .create(&config, workspace)
                .await
                .expect("native sandbox instance");
            (
                SandboxHandle::new(instance, ToolPolicy::default())
                    .with_capabilities(&capabilities),
                capabilities,
            )
        }

        /// The native handle, or `None` (with a skip line) when its backend cannot
        /// confine files and the tool would refuse.
        async fn confining_handle(workspace: &Path) -> Option<SandboxHandle> {
            let (handle, capabilities) = native_handle(workspace).await;
            if capabilities.filesystem_isolation {
                Some(handle)
            } else {
                eprintln!("skipped: this sandbox backend does not claim filesystem isolation");
                None
            }
        }

        async fn run(handle: &SandboxHandle, code: &str) -> ToolResult {
            CodeExecutionTool
                .execute_sandboxed(json!({ "code": code }), Some(handle))
                .await
        }

        fn read_probe(path: &Path) -> String {
            format!(
                "try:\n    open({path:?}).read()\n    print('READ')\nexcept PermissionError:\n    print('DENIED')",
                path = path.display().to_string()
            )
        }

        #[tokio::test]
        async fn runs_code_inside_the_sandbox() {
            let workspace = tempfile::tempdir().expect("workspace");
            let Some(handle) = confining_handle(workspace.path()).await else {
                return;
            };
            let result = run(&handle, "print(2 + 2)").await;
            assert_eq!(
                (result.success, result.content.as_str()),
                (true, "4\n"),
                "error: {:?}",
                result.error
            );
        }

        #[tokio::test]
        async fn denies_reading_a_file_outside_the_workspace() {
            let workspace = tempfile::tempdir().expect("workspace");
            let outside = tempfile::tempdir().expect("outside dir");
            let secret = outside.path().join("secret.txt");
            std::fs::write(&secret, "secret").expect("write secret");
            let Some(handle) = confining_handle(workspace.path()).await else {
                return;
            };
            let result = run(&handle, &read_probe(&secret)).await;
            assert_eq!(
                (result.success, result.content.as_str()),
                (true, "DENIED\n"),
                "error: {:?}",
                result.error
            );
        }

        #[tokio::test]
        async fn denies_listing_the_home_directory() {
            let workspace = tempfile::tempdir().expect("workspace");
            let Some(handle) = confining_handle(workspace.path()).await else {
                return;
            };
            let code = "import os\ntry:\n    os.listdir(os.path.expanduser('~'))\n    print('LISTED')\nexcept PermissionError:\n    print('DENIED')";
            let result = run(&handle, code).await;
            assert_eq!(
                (result.success, result.content.as_str()),
                (true, "DENIED\n"),
                "error: {:?}",
                result.error
            );
        }

        #[tokio::test]
        async fn denies_tcp_connect_when_the_backend_isolates_tcp() {
            let workspace = tempfile::tempdir().expect("workspace");
            let (handle, capabilities) = native_handle(workspace.path()).await;
            if !(capabilities.filesystem_isolation && capabilities.tcp_isolation) {
                eprintln!("skipped: this sandbox backend does not claim file and TCP isolation");
                return;
            }
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("listener");
            let port = listener.local_addr().expect("addr").port();
            let code = format!(
                "import socket\ntry:\n    socket.create_connection(('127.0.0.1', {port}), timeout=5).close()\n    print('CONNECTED')\nexcept OSError:\n    print('DENIED')"
            );
            let result = run(&handle, &code).await;
            assert_eq!(
                (result.success, result.content.as_str()),
                (true, "DENIED\n"),
                "error: {:?}",
                result.error
            );
        }

        #[cfg(unix)]
        #[tokio::test]
        async fn ignores_a_venv_interpreter_in_the_workspace() {
            use std::os::unix::fs::PermissionsExt;
            let workspace = tempfile::tempdir().expect("workspace");
            let marker = workspace.path().join("planted-ran");
            // A committed virtualenv: tools such as uv pick up `<cwd>/.venv` with a pyvenv.cfg.
            std::fs::create_dir_all(workspace.path().join(".venv/bin")).expect("venv dir");
            std::fs::write(
                workspace.path().join(".venv/pyvenv.cfg"),
                "home = /usr/bin\n",
            )
            .expect("pyvenv.cfg");
            for script in [
                workspace.path().join(".venv/bin/python"),
                workspace.path().join(".venv/bin/python3"),
                workspace.path().join("python3"),
            ] {
                std::fs::write(
                    &script,
                    format!("#!/bin/sh\ntouch '{}'\necho planted\n", marker.display()),
                )
                .expect("write script");
                std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                    .expect("chmod");
            }
            let Some(handle) = confining_handle(workspace.path()).await else {
                return;
            };
            let result = run(&handle, "print('real')").await;
            assert_eq!(
                (result.success, result.content.as_str(), marker.exists()),
                (true, "real\n", false),
                "error: {:?}",
                result.error
            );
        }

        /// Windows Job objects, and Linux without Landlock, confine no files.
        #[tokio::test]
        async fn refuses_the_native_backend_when_it_cannot_confine_files() {
            let workspace = tempfile::tempdir().expect("workspace");
            let (handle, capabilities) = native_handle(workspace.path()).await;
            if capabilities.filesystem_isolation {
                eprintln!("skipped: this sandbox backend confines files");
                return;
            }
            let result = run(&handle, "print('host')").await;
            assert_eq!(
                (result.success, result.error.as_deref()),
                (false, Some(NO_SANDBOX_ERROR))
            );
        }

        #[tokio::test]
        async fn refuses_an_inactive_sandbox_handle() {
            let workspace = tempfile::tempdir().expect("workspace");
            let (handle, _) = native_handle(workspace.path()).await;
            let handle = handle.with_mode(SandboxMode::Off);
            let result = run(&handle, "print('host')").await;
            assert_eq!(
                (result.success, result.error.as_deref()),
                (false, Some(NO_SANDBOX_ERROR))
            );
        }
    }
}
