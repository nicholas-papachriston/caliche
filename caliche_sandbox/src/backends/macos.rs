//! macOS Seatbelt sandbox backend (sandbox-exec).
//!
//! Wraps each exec in `sandbox-exec -f <profile> -- shell -c "command"`.
//! Profile denies network (unless policy allows) and restricts writes to configured roots.

use crate::config::{NetworkPolicy, SandboxConfig};
use crate::paths::{read_validated_file, write_validated_file};
use crate::traits::{ExecResult, SandboxBackend, SandboxCapabilities, SandboxInstance};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// By absolute path on the sealed system volume. A bare name would be looked
/// up on `PATH`, including a `PATH` passed to `exec`, so a planted
/// `sandbox-exec` could run the command with no sandbox.
const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

#[derive(Debug)]
pub struct MacOSSeatbeltBackend;

#[async_trait]
impl SandboxBackend for MacOSSeatbeltBackend {
    fn capabilities(&self) -> SandboxCapabilities {
        SandboxCapabilities {
            exec: true,
            filesystem_isolation: true,
            tcp_isolation: true,
            resource_limits: false,
            host_mounts: false,
            pooling: false,
            backend_name: "macos_seatbelt",
        }
    }

    async fn create(
        &self,
        config: &SandboxConfig,
        workspace: &Path,
    ) -> Result<Box<dyn SandboxInstance>> {
        crate::config::validate_sandbox_config(config)?;
        let canonical = workspace
            .canonicalize()
            .with_context(|| format!("canonicalize {}", workspace.display()))?;
        let guarded = crate::git_targets::guarded_dirs(&canonical)?;
        refuse_exposed_git_dirs(config, &canonical)?;
        Ok(Box::new(MacOSSeatbeltInstance {
            id: uuid::Uuid::now_v7().to_string(),
            config: config.clone(),
            workspace: workspace.to_path_buf(),
            canonical,
            guarded,
        }))
    }
}

/// Refuse a git dir or common dir the profile would let a command write.
/// The workspace `.git` itself is covered by the profile's deny. Only the
/// `.git` paths host `git` in the workspace root reads are checked (the root
/// one and each submodule in the index), so this reads indexes and never
/// walks the tree. `workspace` is canonical.
fn refuse_exposed_git_dirs(config: &SandboxConfig, workspace: &Path) -> Result<()> {
    let mut writable = config.writable_roots.clone();
    if config.tmp_write {
        writable.extend(crate::config::tmp_writable_roots());
    }
    // A root that does not exist yet cannot hold git metadata now. This
    // holds only while the host, not the sandbox, creates writable roots.
    let writable: Vec<PathBuf> = writable
        .iter()
        .filter_map(|path| path.canonicalize().ok())
        .collect();
    // With nothing writable the sandbox cannot write any git dir. An
    // invalid `.git` or a stray `HEAD` is still refused: the workspace
    // itself may be writable to the host later.
    if writable.is_empty() {
        crate::git_targets::classify_workspace(workspace)?;
        return Ok(());
    }
    let targets = crate::git_targets::index_targets(workspace)?;
    crate::git_targets::refuse_exposed_git_dirs(workspace, &targets, &writable)?;
    Ok(())
}

#[derive(Debug)]
struct MacOSSeatbeltInstance {
    id: String,
    config: SandboxConfig,
    workspace: PathBuf,
    /// `workspace`, canonical.
    canonical: PathBuf,
    /// [`crate::git_targets::guarded_dirs`] at create.
    guarded: Vec<PathBuf>,
}

/// Escape a path for inclusion in a Seatbelt string literal.
fn sbpl_string(value: &str) -> Result<String> {
    if value.bytes().any(|b| b == 0 || b == b'\n' || b == b'\r') {
        anyhow::bail!("sandbox writable root contains an illegal control character");
    }
    Ok(value.replace('\\', "\\\\").replace('"', "\\\""))
}

/// A path as an anchored SBPL regex, with ERE metacharacters escaped.
///
/// Inside a `#"..."` literal a backslash reaches the regex engine as is, so a
/// path holding a quote, a backslash, or a control character is refused.
fn sbpl_regex_path(path: &Path) -> Result<String> {
    let text = path.display().to_string();
    if text
        .bytes()
        .any(|b| b == 0 || b == b'\n' || b == b'\r' || b == b'"' || b == b'\\')
    {
        anyhow::bail!("sandbox workspace path cannot be written as a Seatbelt regex");
    }
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        if ".^$*+?()[]{}|".contains(ch) {
            out.push('\\');
        }
        out.push(ch);
    }
    Ok(out)
}

/// Deny writes to any `.git` entry (any case, any depth, with or without
/// trailing dots and spaces, as `is_git_dir_name` reads it) in the workspace,
/// to a `.git` or `HEAD` entry in any ancestor of it, to `HEAD` in the
/// workspace root, and to the workspace directory, each of its ancestors
/// and each directory in `guarded` (relative to the workspace).
///
/// Git discovery takes a directory holding `HEAD`, `objects` and `refs` as
/// a bare repository, so a `HEAD` planted in the workspace root or a parent
/// would be read as git metadata. The root `HEAD` is denied even when the
/// workspace has a `.git`: a `.git` that git later rejects would otherwise
/// fall through to that `HEAD`.
///
/// Host-side `git` runs commands named in repository config and hooks, so a
/// sandboxed write there would run outside the sandbox, and git discovery
/// would find a `.git` planted in an ancestor. The deny matches paths, so a
/// writable root around the workspace would otherwise let a command rename
/// the workspace, a parent, or a directory leading to a submodule, write
/// `.git` under the new name, and rename it back. Denying the directories
/// themselves blocks the rename and leaves writes to entries inside them
/// alone. Seatbelt lets the last matching rule win, so this follows every
/// write allow. `workspace` is canonical.
fn deny_workspace_git_writes(workspace: &Path, guarded: &[PathBuf]) -> Result<String> {
    let root = sbpl_regex_path(workspace)?;
    let git = format!(
        "{}[. ]*",
        case_insensitive_regex(crate::paths::GIT_DIR_NAME)
    );
    let head = case_insensitive_regex(crate::paths::HEAD_FILE);
    let mut rules = vec![
        format!("(deny file-write* (regex #\"^{root}/(.*/)?{git}(/|$)\"))"),
        format!("(deny file-write* (regex #\"^{root}/{head}$\"))"),
    ];
    let mut dirs = Vec::new();
    let leads = guarded.iter().map(|dir| workspace.join(dir));
    for dir in workspace.ancestors().map(Path::to_path_buf).chain(leads) {
        dirs.push(format!(
            "(literal \"{}\")",
            sbpl_string(&dir.display().to_string())?
        ));
    }
    rules.push(format!("(deny file-write* {})", dirs.join(" ")));
    for ancestor in workspace.ancestors().skip(1) {
        let base = sbpl_regex_path(ancestor)?;
        let base = base.trim_end_matches('/');
        rules.push(format!(
            "(deny file-write* (regex #\"^{base}/{git}(/|$)\"))"
        ));
        rules.push(format!("(deny file-write* (regex #\"^{base}/{head}$\"))"));
    }
    Ok(rules.join("\n"))
}

/// A regex for `name` in any ASCII case: `.git` becomes `\.[Gg][Ii][Tt]`.
fn case_insensitive_regex(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphabetic() {
                format!("[{}{}]", c.to_ascii_uppercase(), c.to_ascii_lowercase())
            } else {
                format!("\\{c}")
            }
        })
        .collect()
}

/// Host paths dyld, shells, and system libraries need. Home directories,
/// `/Users`, and `/home` are intentionally omitted so `cat ~/.ssh/id_rsa`
/// cannot run inside the sandbox.
const SYSTEM_READ_ROOTS: &[&str] = &[
    "/usr",
    "/bin",
    "/sbin",
    "/opt",
    "/System",
    "/Library",
    "/dev",
    "/etc",
    "/private/etc",
    "/private/var/db",
    "/private/var/select",
    "/tmp",
    "/private/tmp",
    "/System/Volumes",
];

/// The selected Xcode developer directory, read-only. `/usr/bin/python3`,
/// `/usr/bin/git` and the other tool shims re-exec into it, and Xcode.app is
/// outside [`SYSTEM_READ_ROOTS`]. Command Line Tools live under `/Library`,
/// which is already readable, so only an `.app` bundle's `Contents/Developer`
/// is added.
fn xcode_developer_dir() -> Option<&'static Path> {
    static DEVELOPER_DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DEVELOPER_DIR
        .get_or_init(|| {
            // Absolute, with a clean env, so neither PATH nor DEVELOPER_DIR
            // can steer the answer.
            let output = std::process::Command::new("/usr/bin/xcode-select")
                .arg("--print-path")
                .env_clear()
                .stdin(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
                .ok()
                .filter(|output| output.status.success())?;
            let dir = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
            let dir = dir.canonicalize().ok()?;
            is_app_developer_dir(&dir).then_some(dir)
        })
        .as_deref()
}

fn is_app_developer_dir(dir: &Path) -> bool {
    let mut parts = dir.components().rev();
    dir.is_absolute()
        && parts
            .next()
            .is_some_and(|part| part.as_os_str() == "Developer")
        && parts
            .next()
            .is_some_and(|part| part.as_os_str() == "Contents")
        && parts
            .next()
            .is_some_and(|part| Path::new(part.as_os_str()).extension() == Some("app".as_ref()))
}

fn allow_file_subpath(operation: &str, path: &Path) -> Result<String> {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let escaped = sbpl_string(&canonical.display().to_string())?;
    Ok(format!("(allow {operation} (subpath \"{escaped}\"))"))
}

/// Build a sandbox-exec Seatbelt profile from sandbox configuration.
///
/// Uses `(deny default)` rather than `(allow default)` so unspecified
/// operations (IPC, sysctl-write, iokit, etc.) stay closed. Process
/// launch stays allowed; file reads are limited to system/runtime roots
/// plus the workspace and configured writable roots. Writes stay limited
/// to configured roots.
pub fn build_seatbelt_profile(config: &SandboxConfig, workspace: &Path) -> Result<String> {
    let canonical = workspace
        .canonicalize()
        .with_context(|| format!("canonicalize {}", workspace.display()))?;
    let guarded = crate::git_targets::guarded_dirs(&canonical)?;
    seatbelt_profile(config, &canonical, &guarded)
}

/// The profile for the canonical `workspace` with the `guarded` directories.
fn seatbelt_profile(
    config: &SandboxConfig,
    workspace: &Path,
    guarded: &[PathBuf],
) -> Result<String> {
    let mut rules = vec![
        "(version 1)".to_string(),
        "(deny default)".to_string(),
        "(allow process-exec*)".to_string(),
        "(allow process-fork)".to_string(),
        "(allow signal)".to_string(),
        "(allow sysctl-read)".to_string(),
        "(allow mach-lookup)".to_string(),
        "(allow file-read-metadata)".to_string(),
        "(allow file-ioctl)".to_string(),
        "(allow file-map-executable)".to_string(),
        "(allow file-read* (literal \"/\"))".to_string(),
        "(allow file-write-data (literal \"/dev/null\"))".to_string(),
    ];
    for root in SYSTEM_READ_ROOTS {
        rules.push(format!("(allow file-read* (subpath \"{root}\"))"));
    }
    if let Some(developer_dir) = xcode_developer_dir() {
        rules.push(allow_file_subpath("file-read*", developer_dir)?);
    }
    rules.push(allow_file_subpath("file-read*", workspace)?);
    for root in &config.writable_roots {
        rules.push(allow_file_subpath("file-read*", root)?);
    }
    let deny_network = !matches!(config.network, NetworkPolicy::Full);
    if deny_network {
        rules.push("(deny network*)".to_string());
    }
    rules.push("(deny file-write*)".to_string());
    for root in &config.writable_roots {
        rules.push(allow_file_subpath("file-write*", root)?);
    }
    if config.tmp_write {
        rules.push("(allow file-write* (subpath \"/tmp\"))".to_string());
        if cfg!(target_os = "macos") {
            rules.push("(allow file-write* (subpath \"/private/tmp\"))".to_string());
        }
    }
    rules.push(deny_workspace_git_writes(workspace, guarded)?);
    Ok(rules.join("\n"))
}

impl MacOSSeatbeltInstance {
    fn build_profile(&self) -> Result<String> {
        seatbelt_profile(&self.config, &self.canonical, &self.guarded)
    }
}

#[async_trait]
impl SandboxInstance for MacOSSeatbeltInstance {
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
        let profile_content = self.build_profile()?;
        let profile_file = tempfile::NamedTempFile::new().context("create temp profile")?;
        std::fs::write(profile_file.path(), profile_content).context("write profile")?;
        let profile_path = profile_file.path().to_path_buf();

        let cwd = working_dir
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .unwrap_or_else(|| self.workspace.clone());

        let mut cmd = std::process::Command::new(SANDBOX_EXEC);
        cmd.arg("-f").arg(&profile_path);
        cmd.arg("--");
        cmd.arg(command.first().copied().unwrap_or("sh"));
        cmd.args(command.get(1..).unwrap_or(&[]));
        cmd.current_dir(&cwd);
        cmd.env_clear();
        if let Some(e) = env {
            cmd.envs(e);
        }
        cmd.stdin(std::process::Stdio::null());
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());
        // Its own process group, so a timeout kills the background processes too.
        cmd.process_group(0);

        tokio::task::spawn_blocking(move || {
            // The profile file must outlive the child that reads it.
            let _profile_file = profile_file;
            let started = Instant::now();
            let child = cmd.spawn().context("sandbox-exec failed")?;
            super::child_output::collect(
                child,
                started,
                timeout,
                super::child_output::kill_process_group,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_an_app_bundle_developer_dir_is_added() {
        assert_eq!(
            [
                "/Applications/Xcode.app/Contents/Developer",
                "/Applications/Xcode-beta.app/Contents/Developer",
                "/Library/Developer/CommandLineTools",
                "/Users/me/Contents/Developer",
                "/",
                "Xcode.app/Contents/Developer",
            ]
            .map(|dir| is_app_developer_dir(Path::new(dir))),
            [true, true, false, false, false, false]
        );
    }
}
