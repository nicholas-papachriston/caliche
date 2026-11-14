//! Run `git` on a checked repository inside the OS sandbox.
//!
//! [`check_repository`](crate::git_access::check_repository) reads the
//! `.git` before git does, but a path can change between that check and
//! git's own reads. [`CheckedRepository::run_read_only`] therefore runs git
//! inside a native read-only sandbox rooted at the workspace: the kernel
//! refuses a read outside the workspace and the system roots, whatever a
//! link swapped in after the check points at. Without a native backend
//! that confines files, git does not run.

use crate::config::{NamedPolicy, native_backend_for_platform, policy_to_config};
use crate::git_access::{CheckedRepository, refusal};
use crate::traits::ExecResult;
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Set for every read-only git command, so `git status` does not refresh
/// the index (a write the sandbox would refuse anyway).
pub const GIT_OPTIONAL_LOCKS_VAR: &str = "GIT_OPTIONAL_LOCKS";

/// How long one sandboxed git command may run.
pub const GIT_TIMEOUT: Duration = Duration::from_secs(60);

/// Returned when no native sandbox backend can confine git's file reads.
pub const NO_GIT_SANDBOX: &str = "git needs a native OS sandbox that confines file access \
     (macOS Seatbelt, Linux Landlock or a Windows job); none is available, so git did not run";

/// Returned when a git write op runs where the kernel cannot keep the rest
/// of the git dir read-only.
pub const GIT_WRITES_UNSCOPED: &str = "git write ops (commit, branch create and delete) need \
     the macOS Seatbelt sandbox, which can allow writes to objects/, refs/, logs/ and the \
     index and lock files while keeping HEAD, config and hooks/ read-only; Linux Landlock \
     and the Windows job cannot scope writes in a .git directory by file name, so git did \
     not run";

/// The only variables a git write op may add: the commit identity.
pub const GIT_IDENTITY_VARS: [&str; 4] = [
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
];

/// Config every git write op gets on its command line: no automatic gc or
/// maintenance (which would write outside the granted paths) and no signing.
pub const GIT_WRITE_OVERRIDES: [(&str, &str); 4] = [
    ("gc.auto", "0"),
    ("maintenance.auto", "false"),
    ("commit.gpgSign", "false"),
    ("tag.gpgSign", "false"),
];

impl CheckedRepository {
    /// Run `git args` on this repository inside a native read-only sandbox
    /// rooted at the canonical `workspace`, with
    /// [`GIT_OPTIONAL_LOCKS_VAR`] set to `0`.
    ///
    /// The repository's config is listed inside the same sandbox first, and
    /// a repository whose config names a program is refused
    /// ([`refusal::program_keys`]).
    ///
    /// # Errors
    ///
    /// Returns [`NO_GIT_SANDBOX`] when no native backend can confine file
    /// reads, the program-key refusal, or the error running git.
    pub async fn run_read_only(
        &self,
        workspace: &Path,
        args: &[&str],
    ) -> anyhow::Result<ExecResult> {
        self.run_sandboxed(workspace, args, &[], None).await
    }

    /// Run the git write op `git args`, with the identity `env`
    /// ([`GIT_IDENTITY_VARS`] only) added, inside the native
    /// sandbox rooted at the canonical `workspace`. Writes are allowed only
    /// as [`GitWriteGrant`](crate::config::GitWriteGrant) lists, with
    /// [`GIT_WRITE_OVERRIDES`] on the command line. The config check is the
    /// same as [`Self::run_read_only`].
    ///
    /// # Errors
    ///
    /// Returns an error for a variable outside [`GIT_IDENTITY_VARS`],
    /// [`GIT_WRITES_UNSCOPED`] on a backend other than macOS
    /// Seatbelt, [`NO_GIT_SANDBOX`], the program-key refusal, or the error
    /// running git.
    pub async fn run_write(
        &self,
        workspace: &Path,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> anyhow::Result<ExecResult> {
        if let Some((name, _)) = env
            .iter()
            .find(|(name, _)| !GIT_IDENTITY_VARS.contains(name))
        {
            anyhow::bail!("a git write op may set only the commit identity, not {name}");
        }
        let grant = crate::config::GitWriteGrant {
            git_dir: self.git_dir.clone(),
            common_dir: self.common_dir.clone(),
        };
        let overrides: Vec<String> = GIT_WRITE_OVERRIDES
            .iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect();
        let mut full = Vec::new();
        for pair in &overrides {
            full.extend(["-c", pair.as_str()]);
        }
        full.extend_from_slice(args);
        self.run_sandboxed(workspace, &full, env, Some(grant)).await
    }

    async fn run_sandboxed(
        &self,
        workspace: &Path,
        args: &[&str],
        extra_env: &[(&str, &str)],
        git_writes: Option<crate::config::GitWriteGrant>,
    ) -> anyhow::Result<ExecResult> {
        let backend_type = native_backend_for_platform(NamedPolicy::ReadOnly)
            .map_err(|_| anyhow::anyhow!(NO_GIT_SANDBOX))?;
        let mut config =
            policy_to_config(NamedPolicy::ReadOnly, workspace, None, backend_type.clone());
        let writes = git_writes.is_some();
        config.git_writes = git_writes;
        let backend = crate::create_backend(&config)
            .await
            .map_err(|err| anyhow::anyhow!("{NO_GIT_SANDBOX}: {err}"))?;
        if !backend.capabilities().filesystem_isolation {
            anyhow::bail!(NO_GIT_SANDBOX);
        }
        if writes
            && !matches!(
                backend_type,
                crate::config::SandboxBackendType::MacOSSeatbelt { .. }
            )
        {
            anyhow::bail!(GIT_WRITES_UNSCOPED);
        }
        let instance = backend.create(&config, workspace).await?;
        let (argv, env, cwd) =
            self.sandbox_parts(&["config", "--list", "--name-only", "--show-scope", "-z"])?;
        let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
        let listed = instance
            .exec(&argv, Some(&cwd), Some(&env), GIT_TIMEOUT)
            .await?;
        if listed.exit_code != 0 {
            anyhow::bail!(
                "git could not list the repository config, so it did not run: {}",
                listed.stderr.trim()
            );
        }
        let refused = caliche_policy::refused_program_keys_in_listing(listed.stdout.as_bytes());
        if !refused.is_empty() {
            anyhow::bail!(refusal::program_keys(&self.git_dir, &refused));
        }

        let (argv, mut env, cwd) = self.sandbox_parts(args)?;
        for (name, value) in extra_env {
            env.insert((*name).to_string(), (*value).to_string());
        }
        let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
        instance
            .exec(&argv, Some(&cwd), Some(&env), GIT_TIMEOUT)
            .await
    }

    /// The argv, environment and directory of the pinned, hardened
    /// `git args`, for a sandbox `exec` that starts from an empty
    /// environment. Only variables the hardened command sets are kept, with
    /// the host `PATH` (so the sandbox finds git) and
    /// [`GIT_OPTIONAL_LOCKS_VAR`].
    fn sandbox_parts(
        &self,
        args: &[&str],
    ) -> anyhow::Result<(Vec<String>, HashMap<String, String>, String)> {
        let mut command = self.pinned();
        command.args(args);
        let text = |value: &std::ffi::OsStr| {
            value
                .to_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow::anyhow!("git argument is not UTF-8: {value:?}"))
        };
        let mut argv = vec![
            git_program()
                .map_or_else(|| text(command.get_program()), |git| text(git.as_os_str()))?,
        ];
        for arg in command.get_args() {
            argv.push(text(arg)?);
        }
        let mut env = HashMap::new();
        for (name, value) in command.get_envs() {
            if let Some(value) = value {
                env.insert(text(name)?, text(value)?);
            }
        }
        if let Some(path) = std::env::var_os("PATH") {
            env.insert("PATH".to_string(), text(&path)?);
        }
        env.insert(GIT_OPTIONAL_LOCKS_VAR.to_string(), "0".to_string());
        let cwd = command
            .get_current_dir()
            .map_or_else(|| Ok(String::new()), |dir| text(dir.as_os_str()))?;
        Ok((argv, env, cwd))
    }
}

/// The git binary to start in the sandbox, when the bare name would not do.
///
/// On macOS `/usr/bin/git` is an xcrun shim that writes a cache in the
/// user temp dir, which a read-only sandbox refuses, so the real binary is
/// found once with `/usr/bin/xcrun --find git` (absolute, with a clean
/// environment, so neither `PATH` nor `DEVELOPER_DIR` steers it). Elsewhere,
/// or when xcrun finds none, `git` is looked up on `PATH` as usual.
fn git_program() -> Option<&'static Path> {
    #[cfg(target_os = "macos")]
    {
        static GIT: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
        GIT.get_or_init(|| {
            let output = caliche_policy::program_command("/usr/bin/xcrun")
                .args(["--find", "git"])
                .env_clear()
                .stdin(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .output()
                .ok()
                .filter(|output| output.status.success())?;
            let path = std::path::PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
            path.is_absolute().then_some(path)
        })
        .as_deref()
    }
    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}
