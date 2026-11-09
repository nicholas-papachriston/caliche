//! Environment onboarding middleware.
//!
//! Runs once at session start. Reads the working directory, lists its
//! directories in process, looks runtimes and tools up on `PATH`, runs fixed
//! version probes, and prepends the results to the system prompt. No shell
//! runs, tool names are never passed to one, and every child gets the
//! scrubbed host environment.

use super::lifecycle::{
    HookContext, HookResult, LifecycleHook, SessionStartContext, SessionStartHookResult,
};
use async_trait::async_trait;
use caliche_contracts::child_env::PATH_VAR;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::process::Command;

const TIMEOUT_SECS: u64 = 3;

/// Config for the environment onboarding middleware.
#[derive(Debug, Clone)]
pub struct EnvironmentOnboardingConfig {
    /// Max depth of the directory listing, clamped to 1..=5.
    pub max_depth: usize,
    /// Runtimes to detect (python, node, rust, go, ruby).
    pub include_runtimes: Vec<String>,
    /// Tools to check availability (git, make, docker, rg, fd).
    pub include_tools: Vec<String>,
    /// Longest wait for the directory listing (a stalled network or FUSE
    /// mount can block it). Zero leaves the listing out.
    pub listing_timeout: Duration,
}

impl Default for EnvironmentOnboardingConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            include_runtimes: vec![
                "python".to_string(),
                "node".to_string(),
                "rust".to_string(),
                "go".to_string(),
                "ruby".to_string(),
            ],
            include_tools: vec![
                "git".to_string(),
                "make".to_string(),
                "docker".to_string(),
                "rg".to_string(),
                "fd".to_string(),
            ],
            listing_timeout: Duration::from_secs(TIMEOUT_SECS),
        }
    }
}

/// Environment onboarding middleware.
///
/// Runs read-only discovery at session start and prepends [Environment] section.
#[derive(Debug)]
pub struct EnvironmentOnboardingMiddleware {
    config: EnvironmentOnboardingConfig,
}

impl EnvironmentOnboardingMiddleware {
    /// Create a new middleware with the given config.
    #[must_use]
    pub const fn new(config: EnvironmentOnboardingConfig) -> Self {
        Self { config }
    }

    async fn run_command_timeout(mut cmd: Command, timeout_secs: u64) -> Option<String> {
        let output =
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), cmd.output())
                .await
                .ok()
                .and_then(std::result::Result::ok)?;
        let s = String::from_utf8_lossy(&output.stdout);
        let trimmed = s.trim();
        if trimmed.is_empty() {
            String::from_utf8_lossy(&output.stderr)
                .trim()
                .to_string()
                .into()
        } else {
            Some(trimmed.to_string())
        }
    }

    async fn discover(&self, workspace_root: Option<&Path>) -> String {
        let mut lines: Vec<String> = vec!["[Environment]".to_string()];

        if let Ok(cwd) = std::env::current_dir() {
            lines.push(format!("Working directory: {}", cwd.display()));
        }

        let maxdepth = self.config.max_depth.clamp(1, 5);
        let listing = within(self.config.listing_timeout, move || {
            directory_listing(Path::new("."), maxdepth, MAX_LISTED_DIRS)
        })
        .await;
        if !listing.is_empty() {
            lines.push(format!(
                "Directory structure (maxdepth {maxdepth}):\n{}",
                listing.join("\n")
            ));
        }

        self.append_runtimes(&mut lines, TIMEOUT_SECS).await;
        self.append_tools(&mut lines);
        append_project_markers(&mut lines, workspace_root);

        lines.join("\n")
    }

    async fn append_runtimes(&self, lines: &mut Vec<String>, timeout_secs: u64) {
        let mut runtimes = Vec::new();
        for runtime in &self.config.include_runtimes {
            let Some((bin, version_flag)) = runtime_probe(runtime) else {
                continue;
            };
            let Some(found) = find_on_path(bin) else {
                continue;
            };
            // The user's home, so a toolchain file (`rust-toolchain.toml`,
            // `.python-version`) cannot pick the binary. Version managers
            // look in every parent too, so neither the repository nor the
            // shared temp dir, where any local user can write, will do.
            let mut version_cmd = scrubbed_command(found);
            version_cmd.arg(version_flag).current_dir(probe_dir());
            if let Some(version) = Self::run_command_timeout(version_cmd, timeout_secs).await {
                runtimes.push(format!("{bin}: {version}"));
            }
        }
        if !runtimes.is_empty() {
            lines.push(format!("Runtimes: {}", runtimes.join(", ")));
        }
    }

    fn append_tools(&self, lines: &mut Vec<String>) {
        let mut tools = Vec::new();
        for tool in &self.config.include_tools {
            if find_on_path(tool).is_some() {
                tools.push(tool.clone());
            }
        }
        if !tools.is_empty() {
            lines.push(format!("Available tools: {}", tools.join(", ")));
        }
    }
}

/// Where runtime version probes run: the user's home, or the filesystem
/// root when there is none.
fn probe_dir() -> PathBuf {
    std::env::home_dir().unwrap_or_else(|| PathBuf::from("/"))
}

/// A command that starts from the scrubbed host environment.
fn scrubbed_command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    let mut command = Command::new(program);
    command
        .env_clear()
        .envs(caliche_contracts::child_env::host_child_env());
    command
}

/// Most directories the listing names.
const MAX_LISTED_DIRS: usize = 50;

/// Up to `limit` directories under `root`, `root` itself first, each as
/// `root` joined with its relative path, down to `maxdepth` levels below
/// `root`. Symlinks are not followed and unreadable directories are
/// skipped. Entries in a directory are sorted by name, depth first.
fn directory_listing(root: &Path, maxdepth: usize, limit: usize) -> Vec<String> {
    let mut listed = Vec::new();
    let mut pending = vec![(root.to_path_buf(), 0)];
    while let Some((dir, depth)) = pending.pop() {
        if listed.len() == limit {
            break;
        }
        listed.push(dir.display().to_string());
        if depth == maxdepth {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut children: Vec<PathBuf> = entries
            .filter_map(std::result::Result::ok)
            .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
            .map(|entry| dir.join(entry.file_name()))
            .collect();
        children.sort();
        pending.extend(children.into_iter().rev().map(|child| (child, depth + 1)));
    }
    listed
}

/// `list` run on a thread of its own, or no listing when `limit` is zero
/// or the walk takes longer. The thread is detached rather than a runtime
/// blocking task: dropping the runtime waits for every blocking task, so a
/// walk stuck in the kernel would keep the process from exiting.
async fn within(
    limit: Duration,
    list: impl FnOnce() -> Vec<String> + Send + 'static,
) -> Vec<String> {
    if limit.is_zero() {
        return Vec::new();
    }
    let (send, receive) = tokio::sync::oneshot::channel();
    let spawned = std::thread::Builder::new()
        .name("onboarding-listing".to_string())
        .spawn(move || {
            // The receiver is gone once the limit has passed.
            let _ = send.send(list());
        });
    if spawned.is_err() {
        return Vec::new();
    }
    tokio::time::timeout(limit, receive)
        .await
        .ok()
        .and_then(std::result::Result::ok)
        .unwrap_or_default()
}

/// The first executable file named `name` in a `PATH` directory. A name with
/// a path separator, or an empty name, is never looked up.
fn find_on_path(name: &str) -> Option<PathBuf> {
    if name.is_empty() || name.contains(['/', '\\']) || name == "." || name == ".." {
        return None;
    }
    let path = std::env::var_os(PATH_VAR)?;
    std::env::split_paths(&path)
        .filter(|dir| dir.is_absolute())
        .flat_map(|dir| executable_names(name).map(move |file| dir.join(file)))
        .find(|candidate| is_executable(candidate))
}

#[cfg(windows)]
fn executable_names(name: &str) -> impl Iterator<Item = String> + '_ {
    let extensions = std::env::var("PATHEXT").unwrap_or_else(|_| ".EXE;.CMD;.BAT".to_string());
    std::iter::once(name.to_string()).chain(
        extensions
            .split(';')
            .filter(|ext| !ext.is_empty())
            .map(|ext| format!("{name}{ext}"))
            .collect::<Vec<_>>(),
    )
}

#[cfg(not(windows))]
fn executable_names(name: &str) -> impl Iterator<Item = String> + '_ {
    std::iter::once(name.to_string())
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    path.metadata()
        .is_ok_and(|meta| meta.is_file() && meta.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

fn runtime_probe(runtime: &str) -> Option<(&'static str, &'static str)> {
    match runtime {
        "python" => Some(("python3", "--version")),
        "node" => Some(("node", "--version")),
        "rust" => Some(("rustc", "--version")),
        "go" => Some(("go", "version")),
        "ruby" => Some(("ruby", "--version")),
        _ => None,
    }
}

fn append_project_markers(lines: &mut Vec<String>, workspace_root: Option<&Path>) {
    const MARKERS: &[&str] = &[
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "Makefile",
        "go.mod",
    ];
    let root = workspace_root.unwrap_or_else(|| Path::new("."));
    let found: Vec<&str> = MARKERS
        .iter()
        .copied()
        .filter(|marker| root.join(marker).exists())
        .collect();
    if !found.is_empty() {
        lines.push(format!("Project markers: {}", found.join(", ")));
    }
}

#[async_trait]
impl LifecycleHook for EnvironmentOnboardingMiddleware {
    async fn on_pre_tool_use(&self, _ctx: &HookContext<'_>) -> HookResult {
        HookResult::allow()
    }

    async fn on_session_start(&self, ctx: &SessionStartContext<'_>) -> SessionStartHookResult {
        let root = ctx.workspace_root;
        let content = self.discover(root).await;
        if content.len() <= "[Environment]\n".len() {
            SessionStartHookResult::none()
        } else {
            SessionStartHookResult::prepend(content)
        }
    }
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use super::*;

    /// A tool name is looked up on PATH, never handed to a shell, so a name
    /// that is a shell command runs nothing.
    #[tokio::test]
    async fn a_tool_name_is_never_run_as_a_shell_command() {
        let dir = tempfile::tempdir().expect("tempdir");
        let pwned = dir.path().join("pwned");
        let middleware = EnvironmentOnboardingMiddleware::new(EnvironmentOnboardingConfig {
            max_depth: 1,
            include_runtimes: Vec::new(),
            include_tools: vec![
                "sh".to_string(),
                format!("git; echo $HOME > {} #", pwned.display()),
            ],
            ..EnvironmentOnboardingConfig::default()
        });

        let content = middleware.discover(Some(dir.path())).await;

        assert!(!pwned.exists());
        assert!(
            content.contains("Available tools: sh\n") || content.ends_with("Available tools: sh"),
            "{content}"
        );
    }

    /// A fake `python3` that reports the planted variable and its working
    /// directory as its version.
    fn fake_python(dir: &Path, body: &str) {
        use std::os::unix::fs::PermissionsExt;
        let script = dir.join("python3");
        std::fs::write(&script, format!("#!/bin/sh\n{body}\n")).expect("fake python3");
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).expect("chmod");
    }

    fn python_probe() -> EnvironmentOnboardingMiddleware {
        EnvironmentOnboardingMiddleware::new(EnvironmentOnboardingConfig {
            max_depth: 1,
            include_runtimes: vec!["python".to_string()],
            include_tools: Vec::new(),
            ..EnvironmentOnboardingConfig::default()
        })
    }

    /// A runtime version probe starts from the scrubbed environment, so it
    /// does not see a host credential, and runs in the user's home.
    #[tokio::test]
    async fn a_runtime_probe_gets_no_host_credentials_and_runs_in_home() {
        use crate::testing::planted_credential::{PRINT_PLANTED, rerun_in_child};
        let bin = tempfile::tempdir().expect("bin");
        fake_python(
            bin.path(),
            &format!("{PRINT_PLANTED}; printf ' cwd=%s' \"$(pwd -P)\""),
        );
        let repo = tempfile::tempdir().expect("repo");
        let home = tempfile::tempdir().expect("home");
        let path = std::env::join_paths([bin.path(), Path::new("/usr/bin"), Path::new("/bin")])
            .expect("PATH");
        if rerun_in_child(
            "agent::hooks::environment_onboarding::tests::a_runtime_probe_gets_no_host_credentials_and_runs_in_home",
            &[
                (PATH_VAR, path.as_os_str()),
                ("HOME", home.path().as_os_str()),
            ],
            Some(repo.path()),
        ) {
            return;
        }
        let repo_dir = std::env::current_dir().expect("cwd");
        let home_dir = std::env::home_dir()
            .expect("HOME")
            .canonicalize()
            .expect("canonical HOME");

        let content = python_probe().discover(Some(&repo_dir)).await;

        let line = content
            .lines()
            .find(|line| line.starts_with("Runtimes: "))
            .expect("runtime line");
        assert_eq!(
            line,
            format!("Runtimes: python3: unset cwd={}", home_dir.display())
        );
    }

    /// A relative `PATH` entry is skipped, so a repository's `./python3`
    /// does not run when `.` is on `PATH`.
    #[tokio::test]
    async fn a_relative_path_entry_is_never_searched() {
        use crate::testing::planted_credential::rerun_in_child;
        let repo = tempfile::tempdir().expect("repo");
        let marker = repo.path().join("ran");
        fake_python(
            repo.path(),
            &format!("/usr/bin/touch '{}'; echo 3.0", marker.display()),
        );
        if rerun_in_child(
            "agent::hooks::environment_onboarding::tests::a_relative_path_entry_is_never_searched",
            &[(PATH_VAR, std::ffi::OsStr::new("."))],
            Some(repo.path()),
        ) {
            assert!(!marker.exists());
            return;
        }

        let found = find_on_path("python3");
        let content = python_probe().discover(None).await;

        assert_eq!(found, None);
        assert!(!content.contains("python3"), "{content}");
    }

    /// The directory listing runs no program. `PATH` is `.` plus an absolute
    /// directory, and both hold an `sh` and a `find` that write a marker,
    /// so any program the listing starts leaves one.
    #[tokio::test]
    async fn the_directory_listing_runs_no_repository_program() {
        use crate::testing::planted_credential::rerun_in_child;
        let repo = tempfile::tempdir().expect("repo");
        let bin = tempfile::tempdir().expect("bin");
        let markers = tempfile::tempdir().expect("markers");
        for dir in [repo.path(), bin.path()] {
            for name in ["sh", "find"] {
                let marker = markers.path().join(format!("ran-{name}"));
                fake_python(dir, &format!("/usr/bin/touch '{}'", marker.display()));
                std::fs::rename(dir.join("python3"), dir.join(name)).expect("rename");
            }
        }
        std::fs::create_dir_all(repo.path().join("src").join("deep")).expect("src");
        let path = std::env::join_paths([Path::new("."), bin.path()]).expect("PATH");
        if rerun_in_child(
            "agent::hooks::environment_onboarding::tests::the_directory_listing_runs_no_repository_program",
            &[(PATH_VAR, path.as_os_str())],
            Some(repo.path()),
        ) {
            let ran: Vec<_> = std::fs::read_dir(markers.path())
                .expect("markers")
                .map(|entry| entry.expect("entry").file_name())
                .collect();
            assert!(ran.is_empty(), "{ran:?}");
            return;
        }
        let middleware = EnvironmentOnboardingMiddleware::new(EnvironmentOnboardingConfig {
            max_depth: 1,
            include_runtimes: Vec::new(),
            include_tools: Vec::new(),
            ..EnvironmentOnboardingConfig::default()
        });

        let content = middleware.discover(None).await;

        let listing: Vec<&str> = content
            .lines()
            .skip_while(|line| *line != "Directory structure (maxdepth 1):")
            .skip(1)
            .take_while(|line| line.starts_with('.'))
            .collect();
        assert_eq!(listing, [".", "./src"]);
    }

    /// The listing stops at the depth and entry limits, in name order.
    #[test]
    fn the_directory_listing_is_bounded() {
        let root = tempfile::tempdir().expect("root");
        for dir in ["b/inner/deeper", "a", "c", ".git/objects"] {
            std::fs::create_dir_all(root.path().join(dir)).expect("dir");
        }
        std::fs::write(root.path().join("file"), "x").expect("file");
        let at = |rel: &str| root.path().join(rel).display().to_string();
        let top = root.path().display().to_string();

        let deep = directory_listing(root.path(), 2, 50);
        let short = directory_listing(root.path(), 5, 3);

        assert_eq!(
            deep,
            [
                top.clone(),
                at(".git"),
                at(".git/objects"),
                at("a"),
                at("b"),
                at("b/inner"),
                at("c")
            ]
        );
        assert_eq!(short, [top, at(".git"), at(".git/objects")]);
    }

    /// Makes `dir` unreadable until dropped, so a failed assertion still
    /// leaves a directory the temp dir cleanup can remove.
    struct Unreadable(PathBuf);

    impl Unreadable {
        fn new(dir: PathBuf) -> Self {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o000)).expect("chmod");
            Self(dir)
        }
    }

    impl Drop for Unreadable {
        fn drop(&mut self) {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&self.0, std::fs::Permissions::from_mode(0o755));
        }
    }

    /// The listing names a symlink to a directory nowhere and does not
    /// follow it, and does not descend into a directory it cannot read.
    #[test]
    fn the_directory_listing_skips_symlinks_and_unreadable_directories() {
        let root = tempfile::tempdir().expect("root");
        let outside = tempfile::tempdir().expect("outside");
        std::fs::create_dir(outside.path().join("host")).expect("host");
        std::fs::create_dir_all(root.path().join("locked").join("hidden")).expect("locked");
        std::fs::create_dir(root.path().join("real")).expect("real");
        std::os::unix::fs::symlink(outside.path(), root.path().join("link")).expect("symlink");
        let locked = Unreadable::new(root.path().join("locked"));
        let unreadable = std::fs::read_dir(&locked.0).is_err();
        let at = |rel: &str| root.path().join(rel).display().to_string();

        let listing = directory_listing(root.path(), 3, 50);

        drop(locked);
        let mut expected = vec![root.path().display().to_string(), at("locked")];
        if !unreadable {
            // Root reads a 000 directory, so the skip cannot be shown.
            expected.push(at("locked/hidden"));
        }
        expected.push(at("real"));
        assert_eq!(listing, expected);
    }

    /// A listing that never returns gives no listing once the limit has
    /// passed, and does not hold up the runtime's shutdown, so the CLI can
    /// still exit.
    #[test]
    fn a_stuck_listing_is_dropped_and_does_not_hold_up_shutdown() {
        let (release, stuck) = std::sync::mpsc::channel::<()>();
        let (done, finished) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("runtime");
            let listing = runtime.block_on(within(Duration::from_millis(20), move || {
                let _ = stuck.recv();
                vec![".".to_string()]
            }));
            drop(runtime);
            let _ = done.send(listing);
        });

        let listing = finished.recv_timeout(Duration::from_secs(5));

        drop(release);
        assert_eq!(listing, Ok(Vec::new()));
    }

    /// Session start lists the directory through the configured time
    /// limit: with no time allowed the listing is left out and the rest of
    /// the section stays.
    #[tokio::test]
    async fn session_start_applies_the_listing_time_limit() {
        let root = tempfile::tempdir().expect("root");
        std::fs::write(root.path().join("Cargo.toml"), "").expect("marker");
        let middleware = EnvironmentOnboardingMiddleware::new(EnvironmentOnboardingConfig {
            max_depth: 1,
            include_runtimes: Vec::new(),
            include_tools: Vec::new(),
            listing_timeout: Duration::ZERO,
        });

        let result = middleware
            .on_session_start(&SessionStartContext {
                session_id: "s",
                workspace_root: Some(root.path()),
            })
            .await;

        let content = result.system_prepend.expect("environment section");
        let cwd = std::env::current_dir().expect("cwd");
        assert_eq!(
            content,
            format!(
                "[Environment]\nWorking directory: {}\nProject markers: Cargo.toml",
                cwd.display()
            )
        );
    }

    /// A file on `PATH` without an execute bit is not a tool: the lookup
    /// moves on to the next `PATH` directory.
    #[test]
    fn a_file_without_an_execute_bit_is_skipped() {
        use crate::testing::planted_credential::rerun_in_child;
        use std::os::unix::fs::PermissionsExt;
        let plain = tempfile::tempdir().expect("plain");
        let bin = tempfile::tempdir().expect("bin");
        fake_python(plain.path(), "echo 3.0");
        std::fs::set_permissions(
            plain.path().join("python3"),
            std::fs::Permissions::from_mode(0o644),
        )
        .expect("chmod");
        fake_python(bin.path(), "echo 3.0");
        let path = std::env::join_paths([plain.path(), bin.path()]).expect("PATH");
        if rerun_in_child(
            "agent::hooks::environment_onboarding::tests::a_file_without_an_execute_bit_is_skipped",
            &[(PATH_VAR, path.as_os_str())],
            None,
        ) {
            return;
        }
        let path = std::env::var_os(PATH_VAR).expect("PATH");
        let executable_dir = std::env::split_paths(&path)
            .nth(1)
            .expect("second PATH entry");

        let found = find_on_path("python3");

        assert_eq!(found, Some(executable_dir.join("python3")));
    }

    #[test]
    fn find_on_path_skips_names_with_a_path() {
        assert!(find_on_path("sh").is_some());
        assert_eq!(find_on_path("/bin/sh"), None);
        assert_eq!(find_on_path("../sh"), None);
        assert_eq!(find_on_path(""), None);
    }
}
