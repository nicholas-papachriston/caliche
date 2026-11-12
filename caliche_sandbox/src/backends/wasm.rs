//! Wasmtime/WASI sandbox backend.
//!
//! This backend executes `.wasm` binaries via the `wasmtime` runtime and applies
//! filesystem policy checks from `SandboxConfig` for read/write tool operations.

use crate::config::{SandboxBackendType, SandboxConfig};
use crate::traits::{ExecResult, SandboxBackend, SandboxCapabilities, SandboxInstance};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use std::collections::{BTreeSet, HashMap};
use std::ffi::OsStr;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

const WASMTIME_FILE: &str = if cfg!(windows) {
    "wasmtime.exe"
} else {
    "wasmtime"
};

/// The `wasmtime` CLI from the agent's own `PATH`, never from a `PATH` passed
/// to `exec`. Relative entries and binaries inside the workspace are skipped,
/// so code in the workspace cannot replace the runtime.
fn resolve_wasmtime(path_var: &OsStr, workspace: &Path) -> Result<PathBuf> {
    let workspace = workspace
        .canonicalize()
        .unwrap_or_else(|_| workspace.to_path_buf());
    std::env::split_paths(path_var)
        .filter(|dir| dir.is_absolute())
        .map(|dir| dir.join(WASMTIME_FILE))
        .filter(|candidate| candidate.is_file())
        .map(|candidate| candidate.canonicalize().unwrap_or(candidate))
        .find(|resolved| !resolved.starts_with(&workspace))
        .context("wasmtime is not installed outside the workspace")
}

pub struct WasmtimeBackend {
    engine: Arc<wasmtime::Engine>,
}

impl std::fmt::Debug for WasmtimeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmtimeBackend").finish_non_exhaustive()
    }
}

impl WasmtimeBackend {
    pub fn new() -> Result<Self> {
        let engine = wasmtime::Engine::default();
        // Ensure the WASI dependency is initialized/linked for this backend.
        let _wasi_ctx = wasmtime_wasi::WasiCtxBuilder::new().build();
        Ok(Self {
            engine: Arc::new(engine),
        })
    }
}

#[async_trait]
impl SandboxBackend for WasmtimeBackend {
    fn capabilities(&self) -> SandboxCapabilities {
        SandboxCapabilities {
            exec: true,
            filesystem_isolation: true,
            tcp_isolation: false,
            resource_limits: false,
            host_mounts: false,
            pooling: false,
            backend_name: "wasm_wasmtime",
        }
    }

    async fn create(
        &self,
        config: &SandboxConfig,
        workspace: &Path,
    ) -> Result<Box<dyn SandboxInstance>> {
        crate::config::validate_sandbox_config(config)?;
        let modules = match &config.backend {
            SandboxBackendType::WasmWasmtime { modules, .. } => modules.clone(),
            _ => HashMap::new(),
        };
        Ok(Box::new(WasmtimeSandboxInstance {
            id: uuid::Uuid::now_v7().to_string(),
            config: config.clone(),
            workspace: normalize_path(workspace),
            modules,
            engine: Arc::clone(&self.engine),
            alive: AtomicBool::new(true),
        }))
    }
}

struct WasmtimeSandboxInstance {
    id: String,
    config: SandboxConfig,
    workspace: PathBuf,
    modules: HashMap<String, PathBuf>,
    engine: Arc<wasmtime::Engine>,
    alive: AtomicBool,
}

impl std::fmt::Debug for WasmtimeSandboxInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WasmtimeSandboxInstance")
            .field("id", &self.id)
            .field("workspace", &self.workspace)
            .finish_non_exhaustive()
    }
}

impl WasmtimeSandboxInstance {
    fn is_full_access(&self) -> bool {
        self.config.policy_name.as_deref() == Some("danger-full-access")
    }

    fn writable_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        roots.extend(self.config.writable_roots.iter().map(|p| normalize_path(p)));
        if self.config.tmp_write {
            for root in crate::config::tmp_writable_roots() {
                roots.push(normalize_path(&root));
            }
        }
        roots
    }

    fn readable_roots(&self) -> Vec<PathBuf> {
        let mut roots = vec![self.workspace.clone()];
        roots.extend(self.writable_roots());
        roots
    }

    fn validate_sandbox_path(&self, path: &Path, write: bool) -> Result<()> {
        if self.is_full_access() {
            return Ok(());
        }

        let roots = if write {
            self.writable_roots()
        } else {
            self.readable_roots()
        };
        if roots.iter().any(|root| path_is_within(path, root)) {
            return Ok(());
        }

        let action = if write { "write" } else { "read" };
        bail!(
            "WASM sandbox denies {} access outside allowed roots: {}",
            action,
            path.display()
        );
    }

    fn resolve_user_path(&self, raw_path: &str) -> PathBuf {
        let path = Path::new(raw_path);
        if path.is_absolute() {
            normalize_path(path)
        } else {
            normalize_path(self.workspace.join(path))
        }
    }

    fn resolve_module_from_command(&self, command: &[&str]) -> Result<(PathBuf, Vec<String>)> {
        let Some(module_key) = command.first() else {
            bail!("WASM sandbox exec requires module key or .wasm path as command[0]");
        };

        let module_path = if Self::looks_like_module_path(module_key) {
            self.resolve_user_path(module_key)
        } else if let Some(mapped) = self.modules.get(*module_key) {
            self.resolve_user_path(mapped.to_string_lossy().as_ref())
        } else {
            bail!(
                "WASM sandbox requires a .wasm module path or a configured module alias; got '{}'",
                module_key
            );
        };

        self.validate_sandbox_path(&module_path, false)?;
        if !module_path.exists() {
            bail!("WASM module not found: {}", module_path.display());
        }

        let module_args = command
            .get(1..)
            .unwrap_or(&[])
            .iter()
            .map(|arg| (*arg).to_string())
            .collect::<Vec<_>>();
        Ok((module_path, module_args))
    }

    fn looks_like_module_path(raw: &str) -> bool {
        raw.ends_with(".wasm")
            || raw.contains('/')
            || raw.starts_with('.')
            || Path::new(raw).is_absolute()
    }

    fn preopened_dirs(&self) -> Vec<PathBuf> {
        let mut seen = BTreeSet::new();
        seen.insert(self.workspace.clone());
        for root in self.readable_roots() {
            seen.insert(root);
        }
        seen.into_iter().collect::<Vec<_>>()
    }
}

#[async_trait]
impl SandboxInstance for WasmtimeSandboxInstance {
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
        if !self.alive.load(Ordering::Relaxed) {
            bail!("WASM sandbox instance '{}' is destroyed", self.id);
        }

        let (module_path, module_args) = self.resolve_module_from_command(command)?;
        wasmtime::Module::from_file(self.engine.as_ref(), &module_path).map_err(|err| {
            anyhow::anyhow!("Invalid WASM module '{}': {}", module_path.display(), err)
        })?;

        let cwd = working_dir
            .map(|wd| self.resolve_user_path(wd))
            .unwrap_or_else(|| self.workspace.clone());
        self.validate_sandbox_path(&cwd, false)?;

        let mut merged_env = self.config.env_vars.clone();
        if let Some(runtime_env) = env {
            merged_env.extend(runtime_env.clone());
        }

        let wasmtime = resolve_wasmtime(
            &std::env::var_os("PATH").unwrap_or_default(),
            &self.workspace,
        )?;
        let mut cmd = std::process::Command::new(wasmtime);
        cmd.arg("run");
        for preopen in self.preopened_dirs() {
            cmd.arg("--dir").arg(preopen);
        }
        cmd.arg(module_path);
        cmd.args(module_args);
        cmd.current_dir(cwd);
        cmd.env_clear();
        cmd.envs(merged_env);
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        tokio::task::spawn_blocking(move || {
            let started = Instant::now();
            let child = cmd.spawn().context("failed to execute wasmtime process")?;
            // A WASI guest cannot spawn processes, so killing wasmtime is enough.
            super::child_output::collect(child, started, timeout, |_| {})
        })
        .await
        .context("spawn_blocking join")?
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        if !self.alive.load(Ordering::Relaxed) {
            bail!("WASM sandbox instance '{}' is destroyed", self.id);
        }
        let resolved = self.resolve_user_path(path);
        self.validate_sandbox_path(&resolved, false)?;
        let canonical = std::fs::canonicalize(&resolved)
            .with_context(|| format!("failed to read {}", resolved.display()))?;
        self.validate_sandbox_path(&canonical, false)?;
        read_canonical_no_link(&canonical)
            .with_context(|| format!("failed to read {}", resolved.display()))
    }

    async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        if !self.alive.load(Ordering::Relaxed) {
            bail!("WASM sandbox instance '{}' is destroyed", self.id);
        }
        let resolved = self.resolve_user_path(path);
        self.validate_sandbox_path(&resolved, true)?;
        let (Some(parent), Some(name)) = (resolved.parent(), resolved.file_name()) else {
            bail!("failed to write {}: no file name", resolved.display());
        };
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create parent {}", parent.display()))?;
        let canonical = std::fs::canonicalize(parent)
            .with_context(|| format!("failed to write {}", resolved.display()))?
            .join(name);
        self.validate_sandbox_path(&canonical, true)?;
        write_canonical_no_link(&canonical, content)
            .with_context(|| format!("failed to write {}", resolved.display()))?;
        Ok(())
    }

    async fn is_alive(&self) -> bool {
        self.alive.load(Ordering::Relaxed)
    }

    async fn destroy(&self) -> Result<()> {
        self.alive.store(false, Ordering::Relaxed);
        Ok(())
    }
}

/// Read the canonical file `path`, opening every component from the file
/// system root without following a link, so a link swapped in after the
/// path was validated is refused.
fn read_canonical_no_link(path: &Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let root = path.ancestors().last().unwrap_or(path);
    let mut bytes = Vec::new();
    crate::paths::open_file_beneath(root, path)?.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Write `content` to `path`, whose parent is canonical, opening every
/// component from the file system root without following a link, so a link
/// swapped in after the path was validated is refused.
fn write_canonical_no_link(path: &Path, content: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let root = path.ancestors().last().unwrap_or(path);
    crate::paths::create_file_beneath(root, path)?.write_all(content)
}

fn path_is_within(path: &Path, root: &Path) -> bool {
    if let Ok(real_root) = std::fs::canonicalize(root) {
        let real_path = if path.exists() {
            std::fs::canonicalize(path).ok()
        } else {
            path.parent()
                .and_then(|parent| std::fs::canonicalize(parent).ok())
                .and_then(|real_parent| path.file_name().map(|name| real_parent.join(name)))
        };
        if let Some(real_path) = real_path {
            return real_path == real_root || real_path.strip_prefix(&real_root).is_ok();
        }
    }

    let normalized_path = normalize_path(path);
    let normalized_root = normalize_path(root);
    normalized_path == normalized_root || normalized_path.strip_prefix(&normalized_root).is_ok()
}

fn normalize_path<P: AsRef<Path>>(path: P) -> PathBuf {
    let p = path.as_ref();
    let is_absolute = p.is_absolute();
    let mut normalized = if is_absolute {
        PathBuf::from("/")
    } else {
        PathBuf::new()
    };
    for component in p.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if normalized == Path::new("/") {
                    continue;
                }
                let _ = normalized.pop();
            }
            Component::RootDir => {
                normalized = PathBuf::from("/");
            }
            Component::Normal(part) => normalized.push(part),
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().expect("parent")).expect("mkdir");
        std::fs::write(path, "").expect("write");
    }

    #[test]
    fn resolve_wasmtime_skips_the_workspace_and_relative_entries() {
        let workspace = tempfile::tempdir().expect("workspace");
        let system = tempfile::tempdir().expect("system");
        let inside = workspace.path().join("bin");
        touch(&inside.join(WASMTIME_FILE));
        touch(&system.path().join(WASMTIME_FILE));
        let path_var =
            std::env::join_paths([PathBuf::from("bin"), inside, system.path().to_path_buf()])
                .expect("join");
        let resolved = resolve_wasmtime(&path_var, workspace.path()).expect("wasmtime");
        assert_eq!(
            resolved,
            system
                .path()
                .join(WASMTIME_FILE)
                .canonicalize()
                .expect("canonical")
        );
    }

    #[test]
    fn resolve_wasmtime_refuses_when_only_the_workspace_has_one() {
        let workspace = tempfile::tempdir().expect("workspace");
        let inside = workspace.path().join("bin");
        touch(&inside.join(WASMTIME_FILE));
        let error = resolve_wasmtime(inside.as_os_str(), workspace.path())
            .expect_err("workspace wasmtime must be refused");
        assert_eq!(
            error.to_string(),
            "wasmtime is not installed outside the workspace"
        );
    }

    /// A sandbox instance on `workspace` that may write there.
    async fn instance(workspace: &Path) -> Box<dyn SandboxInstance> {
        let config = crate::config::policy_to_config(
            crate::config::NamedPolicy::WorkspaceWrite,
            workspace,
            None,
            SandboxBackendType::WasmWasmtime {
                policy: crate::config::NamedPolicy::WorkspaceWrite,
                modules: HashMap::new(),
            },
        );
        WasmtimeBackend::new()
            .expect("backend")
            .create(&config, workspace)
            .await
            .expect("instance")
    }

    /// `a.txt` in the workspace flips between a plain file and a link to a
    /// file outside it while the instance reads and writes it. Every read
    /// and write that a check let through must stay on the plain file, so
    /// a link swapped in after the check is never followed.
    #[cfg(unix)]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_link_swapped_in_after_the_check_is_never_followed() {
        let workspace = tempfile::tempdir().expect("workspace");
        let root = workspace
            .path()
            .canonicalize()
            .expect("canonical workspace");
        // The policy also allows the temporary directories, so the file
        // outside the workspace lives in this crate's directory instead.
        let outside = tempfile::Builder::new()
            .prefix(".wasm-race-outside")
            .tempdir_in(env!("CARGO_MANIFEST_DIR"))
            .expect("outside");
        let secret = outside.path().join("secret.txt");
        std::fs::write(&secret, "OUTSIDE-SECRET").expect("secret");
        let target = root.join("a.txt");
        std::fs::write(&target, "inside").expect("inside");
        let sandbox = instance(&root).await;

        let stop = Arc::new(AtomicBool::new(false));
        let flipper = {
            let (stop, target, secret) = (Arc::clone(&stop), target.clone(), secret.clone());
            std::thread::spawn(move || {
                let staged = target.with_file_name("staged");
                while !stop.load(Ordering::Relaxed) {
                    let _ = std::os::unix::fs::symlink(&secret, &staged);
                    let _ = std::fs::rename(&staged, &target);
                    let _ = std::fs::write(&staged, "inside");
                    let _ = std::fs::rename(&staged, &target);
                }
            })
        };
        let path = target.to_string_lossy().into_owned();
        let deadline = Instant::now() + std::time::Duration::from_secs(3);
        let mut leaked = None;
        let (mut reads, mut writes) = (0_u32, 0_u32);
        while Instant::now() < deadline && leaked.is_none() {
            match sandbox.read_file(&path).await {
                Ok(bytes) if bytes == b"inside" => reads += 1,
                Ok(bytes) => leaked = Some(String::from_utf8_lossy(&bytes).into_owned()),
                Err(_) => {}
            }
            if sandbox.write_file(&path, b"inside").await.is_ok() {
                writes += 1;
            }
            if std::fs::read(&secret).expect("secret") != b"OUTSIDE-SECRET" {
                leaked = Some("the outside file was written".to_string());
            }
        }
        stop.store(true, Ordering::Relaxed);
        flipper.join().expect("flipper");

        assert_eq!(leaked, None);
        assert!(
            reads > 0 && writes > 0,
            "the plain file must be read and written: {reads} reads, {writes} writes"
        );
    }
}
