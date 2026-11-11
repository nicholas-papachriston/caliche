//! Path resolution and sandbox root enforcement for native backends.

use crate::config::SandboxConfig;
use anyhow::{Context, Result};
use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
mod unix;
#[cfg(windows)]
mod windows;

/// Resolve a tool-relative or absolute path against the sandbox workspace.
///
/// # Errors
///
/// Returns an error when `path` contains an interior NUL.
pub fn resolve_sandbox_path(workspace: &Path, path: &str) -> Result<PathBuf> {
    reject_nul(path)?;
    let raw = PathBuf::from(path);
    let resolved = if raw.is_absolute() {
        raw
    } else {
        workspace.join(raw)
    };
    Ok(resolved)
}

fn reject_nul(path: &str) -> Result<()> {
    if path.contains('\0') {
        anyhow::bail!("path contains interior NUL");
    }
    Ok(())
}

/// Collapse `.` / `..` without touching the filesystem so traversal checks work
/// for paths that do not exist yet or when intermediate parents are missing.
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if matches!(out.components().next(), Some(std::path::Component::RootDir)) {
                    // Absolute path: pop the last normal segment when possible.
                    if out.components().count() > 1 {
                        out.pop();
                    }
                } else if !out.as_os_str().is_empty() {
                    out.pop();
                }
            }
            std::path::Component::Prefix(_)
            | std::path::Component::RootDir
            | std::path::Component::Normal(_) => out.push(component.as_os_str()),
        }
    }
    out
}

/// Canonicalize the longest existing prefix, then append remaining components.
///
/// `Path::canonicalize` only succeeds when every component exists. A write to
/// `link/newfile` where `link` is a symlink would otherwise fall back to the
/// lexical path and follow the symlink at `open(2)` time.
fn resolve_for_confinement(path: &Path) -> PathBuf {
    let lexical = lexical_normalize(path);
    if lexical.as_os_str().is_empty() {
        return lexical;
    }
    if let Ok(canonical) = lexical.canonicalize() {
        return canonical;
    }

    let mut rest: Vec<OsString> = Vec::new();
    let mut cursor = lexical.clone();
    while let Some(name) = cursor.file_name() {
        rest.push(name.to_os_string());
        let Some(parent) = cursor.parent() else {
            break;
        };
        if parent.as_os_str().is_empty() {
            break;
        }
        cursor = parent.to_path_buf();
        if let Ok(canonical) = cursor.canonicalize() {
            let mut out = canonical;
            for component in rest.into_iter().rev() {
                out.push(component);
            }
            return lexical_normalize(&out);
        }
    }
    lexical
}

fn normalize_for_compare(path: &Path) -> PathBuf {
    resolve_for_confinement(path)
}

/// Returns true when `candidate` is the same as or nested under `root`.
#[must_use]
pub fn path_within_root(candidate: &Path, root: &Path) -> bool {
    let candidate = normalize_for_compare(candidate);
    let root = normalize_for_compare(root);
    candidate == root || candidate.starts_with(&root)
}

fn read_allowed_roots(workspace: &Path, config: &SandboxConfig) -> Vec<PathBuf> {
    let mut roots = vec![workspace.to_path_buf()];
    for root in &config.writable_roots {
        if !roots.iter().any(|existing| existing == root) {
            roots.push(root.clone());
        }
    }
    roots
}

fn write_allowed_roots(config: &SandboxConfig) -> Vec<PathBuf> {
    let mut roots = config.writable_roots.clone();
    if config.tmp_write {
        for root in crate::config::tmp_writable_roots() {
            if !roots.iter().any(|existing| existing == &root) {
                roots.push(root);
            }
        }
    }
    roots
}

fn path_within_any(candidate: &Path, roots: &[PathBuf]) -> bool {
    roots.iter().any(|root| path_within_root(candidate, root))
}

/// Validate a read path against workspace and configured roots.
///
/// # Errors
///
/// Returns an error when `path` is outside the workspace and configured roots.
pub fn validate_read_path(workspace: &Path, config: &SandboxConfig, path: &str) -> Result<PathBuf> {
    let resolved = resolve_for_confinement(&resolve_sandbox_path(workspace, path)?);
    let allowed = read_allowed_roots(workspace, config);
    if path_within_any(&resolved, &allowed) {
        Ok(resolved)
    } else {
        anyhow::bail!("read path '{path}' is outside sandbox-allowed directories")
    }
}

/// Validate a write path against configured writable roots.
///
/// # Errors
///
/// Returns an error when writes are disabled or `path` is outside the
/// configured writable roots.
pub fn validate_write_path(
    workspace: &Path,
    config: &SandboxConfig,
    path: &str,
) -> Result<PathBuf> {
    let resolved = resolve_for_confinement(&resolve_sandbox_path(workspace, path)?);
    let allowed = write_allowed_roots(config);
    if allowed.is_empty() {
        anyhow::bail!("sandbox policy does not allow file writes to '{path}'");
    }
    if !path_within_any(&resolved, &allowed) {
        anyhow::bail!("write path '{path}' is outside sandbox writable directories");
    }
    if in_workspace_git_dir(workspace, &resolved) {
        anyhow::bail!("write path '{path}' is in the repository's .git directory");
    }
    Ok(resolved)
}

pub use caliche_policy::{
    COMMONDIR_FILE, CONFIG_FILE, GIT_DIR_NAME, GITFILE_PREFIX, GITLINK_MODE, HEAD_FILE, INDEX_FILE,
    MODULES_DIR, OBJECTS_DIR, REFS_DIR, SHA256_OBJECT_FORMAT, SPLIT_INDEX_EXTENSION, is_git_dir,
    is_git_dir_name,
};

/// How a Linux instance keeps workspace git metadata read-only. The label
/// is recorded in the trace and named by tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GitGuardMode {
    /// No git metadata to guard, or no writable root covers it.
    Off,
    /// The root `.git` is the only metadata; a read-only clone of it is
    /// mounted over it in a new mount namespace.
    MountNamespace,
    /// The workspace root and each directory leading to git metadata stay
    /// read-only; every other entry is granted.
    TopLevelGrants,
}

impl GitGuardMode {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::MountNamespace => "mount_namespace",
            Self::TopLevelGrants => "top_level_grants",
        }
    }
}

/// Errors a backend returns when it cannot keep workspace git metadata
/// read-only and so refuses to run.
pub mod git_refusal {
    use std::path::Path;

    /// A writable root contains the workspace from outside it, so no grant
    /// can leave the workspace's git metadata out.
    pub const WRITABLE_ROOT_CONTAINS_WORKSPACE: &str = "a writable root contains the workspace and git metadata cannot be kept read-only in it; refusing to run";

    /// The index is split, and the shared part is not read.
    pub const SPLIT_INDEX: &str =
        "split index is not supported; run `git update-index --no-split-index` in the workspace";

    /// The Docker backend binds the root `.git` read-only, which needs one.
    pub const DOCKER_NO_ROOT_GIT: &str = "the Docker backend cannot stop a .git from being created in a workspace without one; refusing to run";

    #[must_use]
    pub fn overlapping_root(root: &Path, target: &Path) -> String {
        format!(
            "writable root {} overlaps git metadata at {}; refusing to run",
            root.display(),
            target.display()
        )
    }

    #[must_use]
    pub fn docker_nested(nested: &str) -> String {
        format!(
            "the Docker backend cannot keep nested git metadata read-only ({nested}); refusing to run"
        )
    }

    /// A git dir or common dir inside the workspace but outside its git
    /// metadata, where the sandbox can write it.
    #[must_use]
    pub fn git_dir_in_workspace(dir: &Path) -> String {
        format!(
            "git dir {} is inside the workspace outside its .git, so the sandbox could write it; refusing to run",
            dir.display()
        )
    }

    #[must_use]
    pub fn git_dir_under_writable_root(dir: &Path, root: &Path) -> String {
        format!(
            "git dir {} is under writable root {}; refusing to run",
            dir.display(),
            root.display()
        )
    }

    /// A Docker mount the container can write holds the workspace, so the
    /// container could move the workspace `.git` aside through it.
    #[must_use]
    pub fn docker_mount_contains_workspace(mount: &Path) -> String {
        format!(
            "writable mount {} contains the workspace, so its git metadata cannot be kept read-only; refusing to run",
            mount.display()
        )
    }

    /// A directory on the path to git metadata is a symlink, which the
    /// sandbox could retarget.
    #[must_use]
    pub fn symlink_leads_to_git(path: &Path) -> String {
        format!(
            "{} is a symlink on the path to git metadata; refusing to run",
            path.display()
        )
    }

    /// An index records a submodule path with a component that is not a
    /// plain name (`..`, `.`, or a root).
    #[must_use]
    pub fn path_not_plain(path: &Path) -> String {
        format!(
            "git metadata path {} is not plain; refusing to run",
            path.display()
        )
    }

    /// A `.git` on the way up from the workspace that is neither a gitfile
    /// naming a git dir nor a git dir. Git would skip it or fail on it.
    #[must_use]
    pub fn invalid_git_entry(path: &Path) -> String {
        format!(
            "{} is not a valid gitfile or git dir; refusing to run",
            path.display()
        )
    }

    /// A `HEAD` symlink to a missing `refs/…` path. Git accepts that for an
    /// unborn branch. The check cannot resolve the link.
    #[must_use]
    pub fn dangling_head(path: &Path) -> String {
        format!(
            "{} is a dangling HEAD symlink to an unborn branch; refusing to run",
            path.display()
        )
    }

    /// A `HEAD` in the workspace or one of its parents, in a directory that
    /// is not a git dir. A command could add the rest of a repository there.
    #[must_use]
    pub fn stray_head(path: &Path) -> String {
        format!(
            "{} is outside any git dir and could be completed into a repository; refusing to run",
            path.display()
        )
    }

    /// A Linux workspace below the root of the repository that holds it.
    /// The guard reads only the workspace's own index, so it would miss
    /// the enclosing repository's submodules inside the workspace.
    #[must_use]
    pub fn enclosing_repository(workspace: &Path, repo: &Path) -> String {
        format!(
            "{} is inside the git repository at {}; on Linux a writable workspace must be the repository root, so open {} instead",
            workspace.display(),
            repo.display(),
            repo.display()
        )
    }

    /// A directory under a writable root that the nested git scan cannot
    /// list, so a `.git` inside it could not be guarded.
    #[must_use]
    pub fn unlistable_dir(path: &Path) -> String {
        format!(
            "{} cannot be listed, so git metadata inside it cannot be kept read-only; refusing to run",
            path.display()
        )
    }

    #[must_use]
    pub fn scan_too_deep(path: &Path, limit: usize) -> String {
        format!(
            "{} is more than {limit} directories below its writable root, past the nested git scan limit; refusing to run",
            path.display()
        )
    }

    #[must_use]
    pub fn scan_too_many_entries(root: &Path, limit: usize) -> String {
        format!(
            "writable root {} has more than {limit} entries in all, past the nested git scan limit; refusing to run",
            root.display()
        )
    }

    #[must_use]
    pub fn scan_too_slow(root: &Path, limit: std::time::Duration) -> String {
        format!(
            "the nested git scan of writable root {} took more than {}s, past its time limit; refusing to run",
            root.display(),
            limit.as_secs()
        )
    }

    #[must_use]
    pub fn submodules_too_deep(path: &Path) -> String {
        format!(
            "submodules nest too deep below {}; refusing to run",
            path.display()
        )
    }
}

/// Whether `resolved` is inside `workspace` and has a `.git` component (any
/// case, any depth) below it. Host-side `git` runs commands named in
/// repository config and hooks, so the sandbox never writes there.
#[must_use]
pub fn in_workspace_git_dir(workspace: &Path, resolved: &Path) -> bool {
    let workspace = normalize_for_compare(workspace);
    let resolved = normalize_for_compare(resolved);
    resolved.strip_prefix(&workspace).is_ok_and(|below| {
        below
            .components()
            .any(|part| part.as_os_str().to_str().is_some_and(is_git_dir_name))
    })
}

/// Read file bytes after path validation.
///
/// Every component below the allowed root is opened relative to its parent
/// without following links: with `openat` and `O_NOFOLLOW` on Unix, and
/// without following reparse points on Windows, where each handle stays open
/// until the file is open. A symlink, junction, or other reparse point
/// anywhere below the root is refused, including one swapped in after
/// validation.
///
/// # Errors
///
/// Returns an error when path validation fails, the path is a symlink, or the
/// file cannot be read.
pub fn read_validated_file(
    workspace: &Path,
    config: &SandboxConfig,
    path: &str,
) -> Result<Vec<u8>> {
    let resolved = validate_read_path(workspace, config, path)?;
    let mut file = open_validated_read(&resolved, &read_allowed_roots(workspace, config))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)
        .with_context(|| format!("read '{}'", resolved.display()))?;
    Ok(bytes)
}

/// Open an existing file below `root` for reading, following no link below it.
///
/// `root` must be canonical and `target` a canonical path strictly below it.
/// The path is walked one component at a time, each opened relative to its
/// parent, as in [`read_validated_file`]: on Unix from `/`, the root's own
/// components included, and on Windows from the root down. A symlink,
/// junction, or other reparse point swapped in after the caller validated
/// `target` is refused.
///
/// # Errors
///
/// Returns an error when `target` is not below `root`, a component is a link,
/// or the file cannot be opened.
#[cfg(any(unix, windows))]
pub fn open_file_beneath(root: &Path, target: &Path) -> std::io::Result<File> {
    #[cfg(unix)]
    {
        unix::open_beneath(root, target, unix::Access::Read)
    }
    #[cfg(windows)]
    {
        windows::open_beneath(root, target, windows::Access::Read)
    }
}

/// Create or open a file below `root` for writing, following no link below it.
///
/// The walk is the one [`open_file_beneath`] uses: a symlink, junction, or
/// other reparse point anywhere below the root is refused, including one
/// swapped in after validation. Missing directories are created. The file
/// must be a regular file with a single link; it is truncated only after
/// those checks.
///
/// # Errors
///
/// Returns an error when `target` is not below `root`, a component is a link,
/// the file has more than one hard link, or it cannot be opened.
#[cfg(any(unix, windows))]
pub fn create_file_beneath(root: &Path, target: &Path) -> std::io::Result<File> {
    #[cfg(unix)]
    {
        unix::open_beneath(root, target, unix::Access::Write)
    }
    #[cfg(windows)]
    {
        windows::open_beneath(root, target, windows::Access::Write)
    }
}

/// The reason a write through a hard link is refused. The full message is
/// `refusing to write '<path>': ` followed by this text.
pub const HARD_LINK_REFUSAL: &str = "it has more than one hard link";

/// What a listed directory entry is, read without following it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    /// A directory.
    Dir,
    /// A regular file.
    File,
    /// Anything else: a symlink, junction or other reparse point, a FIFO, a
    /// socket, or a device.
    Other,
}

/// One entry of a directory listed by [`list_dir_beneath`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirEntryBeneath {
    /// The entry's name in its directory.
    pub name: OsString,
    /// What the entry is. A link is [`EntryKind::Other`], never its target.
    pub kind: EntryKind,
    /// The length in bytes of a regular file, and 0 for anything else.
    pub len: u64,
}

/// List the directory `target`, at or below `root`, following no link below it.
///
/// The directory is reached as in [`open_file_beneath`], one component at a
/// time relative to its parent, so a directory swapped for a symlink or
/// junction after the caller validated `target` is refused instead of
/// listed. The entries are read from that handle and described without
/// following any of them. `.` and `..` are left out.
///
/// # Errors
///
/// Returns an error when `target` is not at or below `root`, a component is a
/// link, `target` is not a directory, or it cannot be read.
#[cfg(any(unix, windows))]
pub fn list_dir_beneath(root: &Path, target: &Path) -> std::io::Result<Vec<DirEntryBeneath>> {
    #[cfg(unix)]
    {
        unix::list_dir_beneath(root, target)
    }
    #[cfg(windows)]
    {
        windows::list_dir_beneath(root, target)
    }
}

/// Write file bytes after path validation.
///
/// Missing parent directories are created, and every component below the
/// writable root is opened or created without following links, as for
/// [`read_validated_file`].
///
/// # Errors
///
/// Returns an error when path validation, parent directory creation, or the
/// file write fails.
pub fn write_validated_file(
    workspace: &Path,
    config: &SandboxConfig,
    path: &str,
    content: &[u8],
) -> Result<()> {
    let resolved = validate_write_path(workspace, config, path)?;
    let mut file = open_validated_write(&resolved, &write_allowed_roots(config))?;
    file.write_all(content)
        .with_context(|| format!("write '{}'", resolved.display()))?;
    Ok(())
}

#[cfg(all(test, unix))]
thread_local! {
    /// Test hook that runs after the containing root is chosen and before the
    /// open, to swap a component.
    static AFTER_VALIDATE: std::cell::RefCell<Option<Box<dyn FnOnce(&Path)>>> =
        std::cell::RefCell::new(None);
}

#[cfg(all(test, unix))]
fn after_validate(resolved: &Path) {
    if let Some(hook) = AFTER_VALIDATE.with(|cell| cell.borrow_mut().take()) {
        hook(resolved);
    }
}

#[cfg(all(unix, not(test)))]
const fn after_validate(_resolved: &Path) {}

#[cfg(unix)]
fn open_validated_read(resolved: &Path, roots: &[PathBuf]) -> Result<File> {
    let root = containing_root(resolved, roots)?;
    after_validate(resolved);
    unix::open_beneath(&root, resolved, unix::Access::Read)
        .map_err(|err| anyhow::anyhow!("read '{}': {err}", resolved.display()))
}

#[cfg(unix)]
fn open_validated_write(resolved: &Path, roots: &[PathBuf]) -> Result<File> {
    let root = containing_root(resolved, roots)?;
    after_validate(resolved);
    unix::open_beneath(&root, resolved, unix::Access::Write)
        .map_err(|err| anyhow::anyhow!("write '{}': {err}", resolved.display()))
}

#[cfg(not(any(unix, windows)))]
fn open_validated_read(resolved: &Path, _roots: &[PathBuf]) -> Result<File> {
    reject_existing_symlink(resolved)?;
    open_nofollow_read(resolved).with_context(|| format!("read '{}'", resolved.display()))
}

#[cfg(not(any(unix, windows)))]
fn open_validated_write(resolved: &Path, _roots: &[PathBuf]) -> Result<File> {
    if let Some(parent) = resolved.parent() {
        create_dir_all_nofollow(parent)
            .with_context(|| format!("create parent dirs for '{}'", resolved.display()))?;
    }
    reject_existing_symlink(resolved)?;
    open_nofollow_write(resolved).with_context(|| format!("write '{}'", resolved.display()))
}

#[cfg(windows)]
fn open_validated_read(resolved: &Path, roots: &[PathBuf]) -> Result<File> {
    let root = containing_root(resolved, roots)?;
    windows::open_beneath(&root, resolved, windows::Access::Read)
        .with_context(|| format!("read '{}'", resolved.display()))
}

#[cfg(windows)]
fn open_validated_write(resolved: &Path, roots: &[PathBuf]) -> Result<File> {
    let root = containing_root(resolved, roots)?;
    windows::open_beneath(&root, resolved, windows::Access::Write)
        .with_context(|| format!("write '{}'", resolved.display()))
}

/// The canonical form of the first allowed root that contains `resolved`.
#[cfg(any(unix, windows))]
fn containing_root(resolved: &Path, roots: &[PathBuf]) -> Result<PathBuf> {
    roots
        .iter()
        .map(|root| normalize_for_compare(root))
        .find(|root| resolved.starts_with(root))
        .with_context(|| {
            format!(
                "'{}' is outside sandbox-allowed directories",
                resolved.display()
            )
        })
}

#[cfg(not(any(unix, windows)))]
fn reject_existing_symlink(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(meta) if meta.file_type().is_symlink() => {
            anyhow::bail!("refusing to follow symlink at '{}'", path.display());
        }
        Ok(_) | Err(_) => Ok(()),
    }
}

#[cfg(not(any(unix, windows)))]
fn create_dir_all_nofollow(dir: &Path) -> Result<()> {
    if dir.as_os_str().is_empty() || existing_dir_is_usable(dir)? {
        return Ok(());
    }
    if let Some(parent) = dir.parent()
        && parent != dir
        && !parent.as_os_str().is_empty()
    {
        create_dir_all_nofollow(parent)?;
    }
    create_one_dir(dir)
}

#[cfg(not(any(unix, windows)))]
fn existing_dir_is_usable(dir: &Path) -> Result<bool> {
    match std::fs::symlink_metadata(dir) {
        Ok(meta) if meta.file_type().is_symlink() => {
            anyhow::bail!("refusing to follow symlink directory '{}'", dir.display());
        }
        Ok(meta) if meta.is_dir() => Ok(true),
        Ok(_) => anyhow::bail!("parent path '{}' is not a directory", dir.display()),
        Err(err) if err.kind() != std::io::ErrorKind::NotFound => {
            Err(err).with_context(|| format!("stat '{}'", dir.display()))
        }
        Err(_) => Ok(false),
    }
}

#[cfg(not(any(unix, windows)))]
fn create_one_dir(dir: &Path) -> Result<()> {
    match std::fs::create_dir(dir) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => accept_created_dir(dir),
        Err(err) => Err(err).with_context(|| format!("mkdir '{}'", dir.display())),
    }
}

#[cfg(not(any(unix, windows)))]
fn accept_created_dir(dir: &Path) -> Result<()> {
    let meta =
        std::fs::symlink_metadata(dir).with_context(|| format!("stat '{}'", dir.display()))?;
    if meta.file_type().is_symlink() {
        anyhow::bail!("refusing to follow symlink directory '{}'", dir.display());
    }
    if meta.is_dir() {
        Ok(())
    } else {
        anyhow::bail!("parent path '{}' is not a directory", dir.display())
    }
}

#[cfg(not(any(unix, windows)))]
fn open_nofollow_read(path: &Path) -> std::io::Result<File> {
    File::open(path)
}

#[cfg(not(any(unix, windows)))]
fn open_nofollow_write(path: &Path) -> std::io::Result<File> {
    File::options()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
}

#[cfg(all(test, unix))]
mod swap_tests {
    use super::{AFTER_VALIDATE, read_validated_file, write_validated_file};
    use crate::config::{NamedPolicy, SandboxBackendType, policy_to_config};
    use std::path::Path;

    /// Before the open, replace `workspace/sub` with a symlink to `outside`.
    #[expect(clippy::expect_used, reason = "test fixture setup panics with context")]
    fn swap_sub_for_symlink(workspace: &Path, outside: &Path) {
        let workspace = workspace.to_path_buf();
        let outside = outside.to_path_buf();
        AFTER_VALIDATE.with(|cell| {
            *cell.borrow_mut() = Some(Box::new(move |_resolved: &Path| {
                std::fs::rename(workspace.join("sub"), workspace.join("sub.moved"))
                    .expect("move sub away");
                std::os::unix::fs::symlink(&outside, workspace.join("sub")).expect("symlink");
            }));
        });
    }

    #[expect(clippy::expect_used, reason = "test fixture setup panics with context")]
    fn fixture() -> (tempfile::TempDir, tempfile::TempDir, std::path::PathBuf) {
        let ws = tempfile::tempdir().expect("workspace");
        let out = tempfile::tempdir().expect("outside");
        let workspace = ws.path().canonicalize().expect("canonical workspace");
        std::fs::create_dir(workspace.join("sub")).expect("sub");
        std::fs::write(workspace.join("sub/file.txt"), b"inside").expect("inside file");
        std::fs::write(out.path().join("file.txt"), b"orig").expect("outside file");
        (ws, out, workspace)
    }

    #[test]
    fn read_refuses_a_parent_swapped_for_a_symlink() {
        let (_ws, out, workspace) = fixture();
        let config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            &workspace,
            None,
            SandboxBackendType::Noop,
        );
        swap_sub_for_symlink(&workspace, out.path());

        let err = read_validated_file(&workspace, &config, "sub/file.txt")
            .expect_err("swapped parent must be refused");
        assert!(
            format!("{err:#}").contains("symlink"),
            "unexpected error: {err:#}"
        );
        assert_eq!(
            std::fs::read(out.path().join("file.txt")).expect("read"),
            b"orig"
        );
    }

    #[test]
    fn write_refuses_a_parent_swapped_for_a_symlink() {
        let (_ws, out, workspace) = fixture();
        let config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            &workspace,
            None,
            SandboxBackendType::Noop,
        );
        swap_sub_for_symlink(&workspace, out.path());

        let err = write_validated_file(&workspace, &config, "sub/file.txt", b"evil")
            .expect_err("swapped parent must be refused");
        assert!(
            format!("{err:#}").contains("symlink"),
            "unexpected error: {err:#}"
        );
        assert_eq!(
            std::fs::read(out.path().join("file.txt")).expect("read"),
            b"orig"
        );
    }

    #[test]
    fn write_creates_missing_parents_and_reads_back() {
        let (_ws, _out, workspace) = fixture();
        let config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            &workspace,
            None,
            SandboxBackendType::Noop,
        );
        write_validated_file(&workspace, &config, "a/b/new.txt", b"made").expect("write");
        assert_eq!(
            read_validated_file(&workspace, &config, "a/b/new.txt").expect("read"),
            b"made"
        );
        write_validated_file(&workspace, &config, "sub/file.txt", b"x").expect("overwrite");
        assert_eq!(
            std::fs::read(workspace.join("sub/file.txt")).expect("read"),
            b"x"
        );
    }

    fn set_hook(hook: impl FnOnce(&Path) + 'static) {
        AFTER_VALIDATE.with(|cell| *cell.borrow_mut() = Some(Box::new(hook)));
    }

    fn config_for(workspace: &Path) -> crate::config::SandboxConfig {
        policy_to_config(
            NamedPolicy::WorkspaceWrite,
            workspace,
            None,
            SandboxBackendType::Noop,
        )
    }

    #[test]
    fn write_refuses_a_root_swapped_for_a_symlink() {
        let outer = tempfile::tempdir().expect("outer");
        let out = tempfile::tempdir().expect("outside");
        let outer_path = outer.path().canonicalize().expect("canonical outer");
        let workspace = outer_path.join("ws");
        std::fs::create_dir(&workspace).expect("ws");
        let config = config_for(&workspace);

        let ws = workspace.clone();
        let outside = out.path().to_path_buf();
        set_hook(move |_resolved: &Path| {
            std::fs::rename(&ws, ws.with_file_name("ws.moved")).expect("move ws");
            std::os::unix::fs::symlink(&outside, &ws).expect("symlink ws");
        });

        let err = write_validated_file(&workspace, &config, "f.txt", b"evil")
            .expect_err("swapped root must be refused");
        assert!(
            err.to_string().contains("symlink"),
            "unexpected error: {err:#}"
        );
        assert!(!out.path().join("f.txt").exists());
    }

    #[test]
    fn write_into_a_new_dir_under_a_swapped_parent_creates_nothing_outside() {
        let (_ws, out, workspace) = fixture();
        let config = config_for(&workspace);
        swap_sub_for_symlink(&workspace, out.path());

        let err = write_validated_file(&workspace, &config, "sub/newdir/f.txt", b"evil")
            .expect_err("swapped parent must be refused");
        assert!(
            err.to_string().contains("symlink"),
            "unexpected error: {err:#}"
        );
        assert!(!out.path().join("newdir").exists());
    }

    #[test]
    fn read_and_write_refuse_a_last_component_swapped_for_a_symlink() {
        let (_ws, out, workspace) = fixture();
        let config = config_for(&workspace);
        let swap = |workspace: &Path, outside: &Path| {
            let file = workspace.join("sub/file.txt");
            let target = outside.join("file.txt");
            set_hook(move |_resolved: &Path| {
                std::fs::remove_file(&file).expect("remove file");
                std::os::unix::fs::symlink(&target, &file).expect("symlink file");
            });
        };

        swap(&workspace, out.path());
        let err = read_validated_file(&workspace, &config, "sub/file.txt")
            .expect_err("swapped file must not be read");
        assert!(
            err.to_string().contains("symlink"),
            "unexpected error: {err:#}"
        );

        std::fs::remove_file(workspace.join("sub/file.txt")).expect("remove link");
        std::fs::write(workspace.join("sub/file.txt"), b"inside").expect("restore file");
        swap(&workspace, out.path());
        let err = write_validated_file(&workspace, &config, "sub/file.txt", b"evil")
            .expect_err("swapped file must not be written");
        assert!(
            err.to_string().contains("symlink"),
            "unexpected error: {err:#}"
        );
        assert_eq!(
            std::fs::read(out.path().join("file.txt")).expect("read"),
            b"orig"
        );
    }

    #[test]
    fn read_and_write_refuse_a_fifo_without_blocking() {
        let (_ws, _out, workspace) = fixture();
        let config = config_for(&workspace);
        let status = std::process::Command::new("mkfifo")
            .arg(workspace.join("pipe"))
            .status()
            .expect("run mkfifo");
        assert!(status.success());

        // Hold a reader open so the write open succeeds at the syscall
        // level; without one it fails with ENXIO before any check runs.
        let reader = {
            use std::os::unix::fs::OpenOptionsExt;
            std::fs::File::options()
                .read(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(workspace.join("pipe"))
                .expect("open the FIFO's read end")
        };

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let read = read_validated_file(&workspace, &config, "pipe").map(|_| ());
            let write = write_validated_file(&workspace, &config, "pipe", b"x");
            let _ = tx.send((read, write));
        });
        let (read, write) = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("opening a FIFO must not block");
        drop(reader);
        let read = read.expect_err("a FIFO must not be read");
        assert!(
            read.to_string().contains("not a regular file"),
            "unexpected: {read:#}"
        );
        let write = write.expect_err("a FIFO must not be written");
        assert!(
            write.to_string().contains("not a regular file"),
            "unexpected: {write:#}"
        );
    }

    #[test]
    fn write_creates_a_writable_root_that_does_not_exist_yet() {
        let outer = tempfile::tempdir().expect("outer");
        let workspace = outer
            .path()
            .canonicalize()
            .expect("canonical outer")
            .join("not-yet/ws");
        let config = config_for(&workspace);
        write_validated_file(&workspace, &config, "a.txt", b"made").expect("write");
        assert_eq!(
            std::fs::read(workspace.join("a.txt")).expect("read"),
            b"made"
        );
    }

    #[test]
    fn write_refuses_a_dir_swapped_above_the_root() {
        let outer = tempfile::tempdir().expect("outer");
        let out = tempfile::tempdir().expect("outside");
        let outer_path = outer.path().canonicalize().expect("canonical outer");
        let workspace = outer_path.join("x/ws");
        std::fs::create_dir_all(&workspace).expect("x/ws");
        std::fs::create_dir(out.path().join("ws")).expect("outside ws");
        let config = config_for(&workspace);

        let x = outer_path.join("x");
        let outside = out.path().to_path_buf();
        set_hook(move |_resolved: &Path| {
            std::fs::rename(&x, x.with_file_name("x.moved")).expect("move x");
            std::os::unix::fs::symlink(&outside, &x).expect("symlink x");
        });

        let err = write_validated_file(&workspace, &config, "f.txt", b"evil")
            .expect_err("a directory swapped above the root must be refused");
        assert!(
            err.to_string().contains("symlink"),
            "unexpected error: {err:#}"
        );
        assert!(!out.path().join("ws/f.txt").exists());
    }

    #[test]
    fn write_refuses_a_hard_link_and_leaves_the_linked_file() {
        let (_ws, out, workspace) = fixture();
        let config = config_for(&workspace);
        let outside_file = out.path().join("file.txt");
        std::fs::hard_link(&outside_file, workspace.join("linked.txt")).expect("hard link");

        let err = write_validated_file(&workspace, &config, "linked.txt", b"evil")
            .expect_err("a hard-linked file must not be written");
        assert!(
            err.to_string().contains("hard link"),
            "unexpected error: {err:#}"
        );
        assert_eq!(std::fs::read(&outside_file).expect("read"), b"orig");
        assert_eq!(
            read_validated_file(&workspace, &config, "linked.txt").expect("reads are allowed"),
            b"orig"
        );
    }
}
