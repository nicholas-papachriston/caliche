#![expect(
    clippy::expect_used,
    reason = "static Regex/LazyLock literals are reviewed constants; failure is a programmer bug"
)]

use anyhow::{Context, Result, bail};
use std::cell::RefCell;
use std::env;
use std::path::{Path, PathBuf};

thread_local! {
    /// When set by the tool execution engine, relative path resolution uses this instead of env::current_dir().
    static TOOL_WORKSPACE_ROOT: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    /// When true, tools may touch host paths outside the workspace (`danger-full-access`).
    static TOOL_HOST_FS_UNLOCKED: RefCell<bool> = const { RefCell::new(false) };
}

/// Set the workspace root used for relative path resolution in tools.
/// Call this before invoking a tool so that paths like "." resolve to the agent's workspace.
pub fn set_tool_workspace_root(root: Option<PathBuf>) {
    TOOL_WORKSPACE_ROOT.with(|cell| *cell.borrow_mut() = root);
}

/// Allow or deny host filesystem paths outside the workspace for unsandboxed tools.
pub fn set_tool_host_fs_unlocked(unlocked: bool) {
    TOOL_HOST_FS_UNLOCKED.with(|cell| *cell.borrow_mut() = unlocked);
}

fn host_fs_unlocked() -> bool {
    TOOL_HOST_FS_UNLOCKED.with(|cell| *cell.borrow())
}

/// [`tokio::task::spawn_blocking`] that carries this thread's tool
/// workspace root and host file system setting into `f`, so a path helper
/// that `f` calls resolves against the same workspace. The blocking thread's
/// own settings are restored when `f` returns.
pub fn spawn_tool_blocking<F, R>(f: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let root = TOOL_WORKSPACE_ROOT.with(|cell| cell.borrow().clone());
    let unlocked = host_fs_unlocked();
    tokio::task::spawn_blocking(move || {
        let _restore = ToolSettingsRestore {
            root: TOOL_WORKSPACE_ROOT.with(|cell| cell.replace(root)),
            unlocked: TOOL_HOST_FS_UNLOCKED.with(|cell| cell.replace(unlocked)),
        };
        f()
    })
}

/// Puts a blocking thread's own tool settings back, also when `f` panics.
struct ToolSettingsRestore {
    root: Option<PathBuf>,
    unlocked: bool,
}

impl Drop for ToolSettingsRestore {
    fn drop(&mut self) {
        TOOL_WORKSPACE_ROOT.with(|cell| cell.replace(self.root.take()));
        TOOL_HOST_FS_UNLOCKED.with(|cell| cell.replace(self.unlocked));
    }
}

/// Path validation and resolution utilities
pub mod path {
    use super::{Context, Path, PathBuf, Result, bail, env};

    /// Validates and resolves a path, checking for security issues
    /// Supports both local paths and S3 URIs (<s3://bucket/key>)
    ///
    /// # Arguments
    /// * `path_str` - The path string to validate and resolve (local or S3 URI)
    /// * `must_be_file` - If Some(true), ensures it's a file. If Some(false), ensures it's a directory. If None, doesn't check.
    ///
    /// # Returns
    /// The canonicalized `PathBuf` if valid
    /// For S3 URIs, downloads the file to a temporary location and returns that path
    pub async fn validate_and_resolve_path(
        path_str: &str,
        must_be_file: Option<bool>,
    ) -> Result<PathBuf> {
        // Check if this is an S3 URI
        #[cfg(feature = "s3")]
        if crate::data::storage::object::s3_client::is_s3_uri(path_str) {
            return resolve_s3_path(path_str, must_be_file).await;
        }
        #[cfg(not(feature = "s3"))]
        if path_str.starts_with("s3://") {
            return resolve_s3_path(path_str, must_be_file).await;
        }

        // Check for path traversal attempts
        check_path_traversal(path_str)?;

        // Resolve relative paths against current working directory
        let path = if Path::new(path_str).is_absolute() {
            PathBuf::from(path_str)
        } else {
            get_current_dir()?.join(path_str)
        };

        // Canonicalize already fails when the path is missing, so skip a second
        // metadata hop unless the caller asked for a file/directory check.
        let normalized_path = path
            .canonicalize()
            .with_context(|| format!("Failed to resolve path '{path_str}'"))?;
        refuse_link_out_of_workspace(&path, &normalized_path)?;

        if let Some(should_be_file) = must_be_file {
            let metadata = std::fs::metadata(&normalized_path)
                .with_context(|| format!("Failed to access path '{path_str}'"))?;
            if should_be_file && !metadata.is_file() {
                bail!("Path '{path_str}' is not a file (it may be a directory)");
            } else if !should_be_file && !metadata.is_dir() {
                bail!("Path '{path_str}' is not a directory (it may be a file)");
            }
        }

        Ok(normalized_path)
    }

    /// Resolve S3 URI to a local temporary file
    #[cfg(feature = "s3")]
    async fn resolve_s3_path(s3_uri: &str, must_be_file: Option<bool>) -> Result<PathBuf> {
        // S3 objects are always files, not directories
        if must_be_file == Some(false) {
            bail!(
                "S3 URI '{s3_uri}' cannot be a directory. S3 URIs always refer to objects (files)."
            );
        }

        // Check if the S3 object exists
        let exists = crate::data::storage::object::s3_client::s3_object_exists(s3_uri)
            .await
            .context("Failed to check S3 object")?;

        if !exists {
            bail!("S3 object not found: {s3_uri}");
        }

        // Download to temporary file
        let temp_path = crate::data::storage::object::s3_client::download_from_s3(s3_uri)
            .await
            .context("Failed to download from S3")?;

        Ok(temp_path)
    }

    #[cfg(not(feature = "s3"))]
    #[expect(clippy::unused_async, reason = "same signature as the async s3 build")]
    async fn resolve_s3_path(s3_uri: &str, _must_be_file: Option<bool>) -> Result<PathBuf> {
        bail!("S3 support is not enabled. Enable the 's3' feature to use S3 URIs: {s3_uri}");
    }

    /// Open the file at `path`, a path from [`validate_and_resolve_path`],
    /// following no link.
    ///
    /// The policy gate decides on that canonical path. A read by path would
    /// follow a link swapped in after the check, so this opens every
    /// component from the file system root, refuses a link, and refuses a
    /// file that is not a regular file.
    ///
    /// # Errors
    ///
    /// Returns an error when a component is a link, the path is not a
    /// regular file, or it cannot be opened.
    pub fn open_resolved_file(path: &Path) -> std::io::Result<std::fs::File> {
        open_walked_file(fs_root(path), path)
    }

    /// The file system root above `path`: `/`, or a drive or share root.
    fn fs_root(path: &Path) -> &Path {
        path.ancestors().last().unwrap_or(path)
    }

    /// Open `file`, which a walk of the canonical directory `root` found,
    /// following no link below `root`.
    ///
    /// # Errors
    ///
    /// Returns an error when `file` is not below `root`, a component is a
    /// link, `file` is not a regular file, or it cannot be opened.
    pub fn open_walked_file(root: &Path, file: &Path) -> std::io::Result<std::fs::File> {
        #[cfg(any(unix, windows))]
        {
            caliche_sandbox::open_file_beneath(root, file)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (root, file);
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no-link reads are not supported on this platform",
            ))
        }
    }

    /// Read `file` as UTF-8 text through [`open_walked_file`].
    ///
    /// # Errors
    ///
    /// Returns an error when the open fails, the read fails, or the bytes are
    /// not UTF-8.
    pub fn read_walked_text(root: &Path, file: &Path) -> std::io::Result<String> {
        use std::io::Read;
        let mut text = String::new();
        open_walked_file(root, file)?.read_to_string(&mut text)?;
        Ok(text)
    }

    /// Read `path`, a path from [`validate_and_resolve_path`], as UTF-8
    /// text following no link.
    ///
    /// # Errors
    ///
    /// As [`read_walked_text`].
    pub fn read_resolved_text(path: &Path) -> std::io::Result<String> {
        read_walked_text(fs_root(path), path)
    }

    /// List the directory `path`, a path from [`validate_and_resolve_path`]
    /// or a directory a listing of one found, following no link.
    ///
    /// Each entry is described without following it: a link is
    /// [`caliche_sandbox::EntryKind::Other`].
    ///
    /// # Errors
    ///
    /// Returns an error when a component is a link, `path` is not a
    /// directory, or it cannot be read.
    pub fn list_resolved_dir(
        path: &Path,
    ) -> std::io::Result<Vec<caliche_sandbox::DirEntryBeneath>> {
        #[cfg(any(unix, windows))]
        {
            caliche_sandbox::list_dir_beneath(fs_root(path), path)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = path;
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no-link listings are not supported on this platform",
            ))
        }
    }

    /// List `dir`, a path from [`resolve_path_for_tool`] inside the
    /// workspace, following no link below the workspace.
    ///
    /// # Errors
    ///
    /// Returns an error when `dir` is not inside the workspace, the
    /// workspace cannot be resolved, or [`list_resolved_dir`] fails.
    pub fn list_workspace_dir(
        dir: &Path,
    ) -> std::io::Result<Vec<caliche_sandbox::DirEntryBeneath>> {
        let workspace = get_current_dir().map_err(std::io::Error::other)?;
        let below = dir.strip_prefix(&workspace).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("'{}' is outside the workspace", dir.display()),
            )
        })?;
        list_resolved_dir(&workspace.canonicalize()?.join(below))
    }

    /// Read `path`, a path from [`validate_and_resolve_path`], as bytes
    /// following no link.
    ///
    /// # Errors
    ///
    /// As [`open_resolved_file`], or when the read fails.
    pub fn read_resolved_bytes(path: &Path) -> std::io::Result<Vec<u8>> {
        use std::io::Read;
        let mut bytes = Vec::new();
        open_resolved_file(path)?.read_to_end(&mut bytes)?;
        Ok(bytes)
    }

    /// [`read_resolved_bytes`] on a blocking thread.
    ///
    /// # Errors
    ///
    /// As [`read_resolved_bytes`].
    pub async fn read_resolved_bytes_async(path: &Path) -> std::io::Result<Vec<u8>> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || read_resolved_bytes(&path))
            .await
            .map_err(std::io::Error::other)?
    }

    /// [`read_resolved_text`] on a blocking thread.
    ///
    /// # Errors
    ///
    /// As [`read_resolved_text`].
    pub async fn read_resolved_text_async(path: &Path) -> std::io::Result<String> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || read_resolved_text(&path))
            .await
            .map_err(std::io::Error::other)?
    }

    /// The metadata of `path`, a canonical path, following no link.
    ///
    /// A regular file's metadata is read from the file the no-link open
    /// returns. A directory is listed through [`list_resolved_dir`] first, so
    /// a path with a link in it is refused rather than described.
    ///
    /// # Errors
    ///
    /// Returns the error of the no-link open when `path` is neither a
    /// regular file nor a directory reached without a link.
    pub fn resolved_metadata(path: &Path) -> std::io::Result<std::fs::Metadata> {
        metadata_beneath(fs_root(path), path)
    }

    /// The metadata of `path`, a path from [`resolve_path_for_tool`],
    /// following no link below the workspace root, as [`open_tool_file`]
    /// reads it. Only with host file system access unlocked is a path
    /// outside the workspace described, and then by path.
    ///
    /// # Errors
    ///
    /// Returns an error when `path` is outside the workspace and host access
    /// is locked, or as [`resolved_metadata`] below the workspace.
    pub fn tool_metadata(path: &Path) -> std::io::Result<std::fs::Metadata> {
        match below_workspace(path) {
            Ok((root, target)) => metadata_beneath(&root, &target),
            Err(_) if super::host_fs_unlocked() => std::fs::metadata(path),
            Err(err) => Err(err),
        }
    }

    /// The metadata of the regular file or directory `target`, reached from
    /// the canonical `root` without following a link.
    fn metadata_beneath(root: &Path, target: &Path) -> std::io::Result<std::fs::Metadata> {
        match open_walked_file(root, target) {
            Ok(file) => file.metadata(),
            Err(open_error) => {
                #[cfg(any(unix, windows))]
                if caliche_sandbox::list_dir_beneath(root, target).is_ok() {
                    return std::fs::symlink_metadata(target);
                }
                Err(open_error)
            }
        }
    }

    /// [`resolved_metadata`] on a blocking thread.
    ///
    /// # Errors
    ///
    /// As [`resolved_metadata`].
    pub async fn resolved_metadata_async(path: &Path) -> std::io::Result<std::fs::Metadata> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || resolved_metadata(&path))
            .await
            .map_err(std::io::Error::other)?
    }

    /// The canonical form of `path`, resolved on a blocking thread. It reads
    /// no file; a caller opens the result with a no-link helper here.
    ///
    /// # Errors
    ///
    /// Returns the error canonicalizing `path`.
    pub async fn canonical_path_async(path: &Path) -> std::io::Result<PathBuf> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || path.canonicalize())
            .await
            .map_err(std::io::Error::other)?
    }

    /// [`list_resolved_dir`] on a blocking thread.
    ///
    /// # Errors
    ///
    /// As [`list_resolved_dir`].
    pub async fn list_resolved_dir_async(
        path: &Path,
    ) -> std::io::Result<Vec<caliche_sandbox::DirEntryBeneath>> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || list_resolved_dir(&path))
            .await
            .map_err(std::io::Error::other)?
    }

    /// Refuse `canonical`, the resolution of `lexical`, when `lexical` names
    /// a place inside the workspace and `canonical` is outside it.
    ///
    /// The policy gate resolves the path the model gave and decides on that.
    /// The tool resolves it again, and a link swapped in between those two
    /// resolutions would take a path the gate allowed as inside the
    /// workspace to a file outside it. A link inside the workspace that
    /// leads out of it is refused for the same reason. With host file
    /// system access unlocked, nothing is refused.
    ///
    /// # Errors
    ///
    /// Returns an error when the workspace cannot be resolved, or when
    /// `lexical` is inside it and `canonical` is not.
    pub fn refuse_link_out_of_workspace(lexical: &Path, canonical: &Path) -> Result<()> {
        if super::host_fs_unlocked() {
            return Ok(());
        }
        let workspace = get_current_dir()?;
        let workspace_canonical = workspace
            .canonicalize()
            .with_context(|| format!("Failed to resolve workspace '{}'", workspace.display()))?;
        let names_inside =
            names_place_in(lexical, &workspace) || names_place_in(lexical, &workspace_canonical);
        if names_inside && !canonical.starts_with(&workspace_canonical) {
            bail!(
                "Path '{}' is inside the workspace but leads through a link to '{}', outside it",
                lexical.display(),
                canonical.display()
            );
        }
        Ok(())
    }

    /// Whether `path` names a place in `dir`, comparing names without case
    /// where the file system usually ignores it (macOS and Windows), so a
    /// differently cased path to the workspace still counts as inside.
    fn names_place_in(path: &Path, dir: &Path) -> bool {
        if path.starts_with(dir) {
            return true;
        }
        if !cfg!(any(target_os = "macos", windows)) {
            return false;
        }
        let mut names = path.components();
        dir.components().all(|want| {
            names.next().is_some_and(|got| {
                got.as_os_str()
                    .to_string_lossy()
                    .to_lowercase()
                    .eq(&want.as_os_str().to_string_lossy().to_lowercase())
            })
        })
    }

    /// `path`, a path from [`resolve_path_for_tool`], as the canonical
    /// workspace root and the same place below that root.
    fn below_workspace(path: &Path) -> std::io::Result<(PathBuf, PathBuf)> {
        let workspace = get_current_dir().map_err(std::io::Error::other)?;
        let canonical = workspace.canonicalize()?;
        let below = path
            .strip_prefix(&workspace)
            .or_else(|_| path.strip_prefix(&canonical))
            .map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("'{}' is outside the workspace", path.display()),
                )
            })?;
        let target = canonical.join(below);
        Ok((canonical, target))
    }

    /// How a tool opens a file it was given by path.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum ToolOpen {
        /// Read an existing file.
        Read,
        /// Create or truncate the file, creating missing directories.
        Write,
        /// Create the file or add to its end, creating missing directories.
        Append,
    }

    /// Open `path`, a path from [`resolve_path_for_tool`], following no link
    /// below the workspace root, so a link swapped in after the policy gate
    /// decided cannot take the read or write outside the workspace.
    ///
    /// A write or append creates missing directories. Only with host file
    /// system access unlocked is a path outside the workspace opened, and
    /// then by path.
    ///
    /// # Errors
    ///
    /// Returns an error when `path` is outside the workspace and host access
    /// is locked, a component below the workspace is a link, the file is not
    /// a regular file (or, for a write, has more than one hard link), or it
    /// cannot be opened.
    pub fn open_tool_file(path: &Path, how: ToolOpen) -> std::io::Result<std::fs::File> {
        match below_workspace(path) {
            Ok((root, target)) => open_beneath(&root, &target, how),
            Err(_) if super::host_fs_unlocked() => {
                if how != ToolOpen::Read
                    && let Some(parent) = path.parent()
                {
                    std::fs::create_dir_all(parent)?;
                }
                match how {
                    ToolOpen::Read => std::fs::File::open(path),
                    ToolOpen::Write => std::fs::File::create(path),
                    ToolOpen::Append => std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(path),
                }
            }
            Err(err) => Err(err),
        }
    }

    fn open_beneath(root: &Path, target: &Path, how: ToolOpen) -> std::io::Result<std::fs::File> {
        #[cfg(any(unix, windows))]
        {
            match how {
                ToolOpen::Read => caliche_sandbox::open_file_beneath(root, target),
                ToolOpen::Write => caliche_sandbox::create_file_beneath(root, target),
                ToolOpen::Append => caliche_sandbox::append_file_beneath(root, target),
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (root, target, how);
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no-link opens are not supported on this platform",
            ))
        }
    }

    /// Remove `path`, a path from [`resolve_path_for_tool`], following no
    /// link below the workspace root: a link is removed itself, and a link
    /// swapped in for a directory on the way is refused. A directory is
    /// removed, with its content, only when `recursive` is set.
    ///
    /// Only with host file system access unlocked is a path outside the
    /// workspace removed, and then by path.
    ///
    /// # Errors
    ///
    /// Returns an error when `path` is outside the workspace and host access
    /// is locked, a directory on the way is a link, `path` is a directory and
    /// `recursive` is not set, or the removal fails.
    pub fn remove_tool_path(path: &Path, recursive: bool) -> std::io::Result<()> {
        match below_workspace(path) {
            Ok((root, target)) => remove_beneath(&root, &target, recursive),
            Err(_) if super::host_fs_unlocked() => {
                if recursive && std::fs::symlink_metadata(path)?.is_dir() {
                    std::fs::remove_dir_all(path)
                } else {
                    std::fs::remove_file(path)
                }
            }
            Err(err) => Err(err),
        }
    }

    fn remove_beneath(root: &Path, target: &Path, recursive: bool) -> std::io::Result<()> {
        #[cfg(any(unix, windows))]
        {
            caliche_sandbox::remove_beneath(root, target, recursive)
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = (root, target, recursive);
            Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "no-link removal is not supported on this platform",
            ))
        }
    }

    /// Read `path`, a path from [`resolve_path_for_tool`], as UTF-8 text
    /// through [`open_tool_file`].
    ///
    /// # Errors
    ///
    /// As [`open_tool_file`], or when the read fails or the bytes are not
    /// UTF-8.
    pub fn read_tool_text(path: &Path) -> std::io::Result<String> {
        use std::io::Read as _;
        let mut text = String::new();
        open_tool_file(path, ToolOpen::Read)?.read_to_string(&mut text)?;
        Ok(text)
    }

    /// Replace the content of `path`, a path from [`resolve_path_for_tool`],
    /// with `bytes` through [`open_tool_file`].
    ///
    /// # Errors
    ///
    /// As [`open_tool_file`], or when the write fails.
    pub fn write_tool_file(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
        use std::io::Write as _;
        let mut file = open_tool_file(path, ToolOpen::Write)?;
        file.write_all(bytes)?;
        file.flush()
    }

    /// Checks for path traversal attempts.
    ///
    /// Rejects interior NULs and `..` path components. A filename such as
    /// `foo..bar.txt` is allowed; `../secret` is not.
    pub fn check_path_traversal(path_str: &str) -> Result<()> {
        if path_str.contains('\0') {
            bail!("Path contains interior NUL");
        }
        if Path::new(path_str)
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            bail!("Path traversal ('..') is not allowed for security reasons");
        }
        Ok(())
    }

    /// Gets the current working directory for path resolution.
    ///
    /// When the tool execution engine has set a workspace root (via `set_tool_workspace_root`),
    /// that is used so that "." and other relative paths resolve to the agent's workspace.
    pub fn get_current_dir() -> Result<PathBuf> {
        super::TOOL_WORKSPACE_ROOT.with(|cell| {
            if let Some(ref root) = *cell.borrow() {
                return Ok(root.clone());
            }
            env::current_dir().context("Failed to get current directory")
        })
    }

    /// Resolve a path for tool use: traversal check then workspace confinement.
    /// Absolute paths outside the workspace are rejected unless host FS is
    /// unlocked (`danger-full-access`).
    pub fn resolve_path_for_tool(path_str: &str) -> Result<PathBuf> {
        check_path_traversal(path_str)?;
        let workspace = get_current_dir()?;
        let resolved = if Path::new(path_str).is_absolute() {
            PathBuf::from(path_str)
        } else {
            workspace.join(path_str)
        };
        if super::host_fs_unlocked() {
            return Ok(resolved);
        }
        if resolved == workspace || resolved.starts_with(&workspace) {
            Ok(resolved)
        } else {
            bail!(
                "Path '{}' is outside workspace '{}'",
                resolved.display(),
                workspace.display()
            )
        }
    }

    /// [`resolve_path_for_tool`] for a path the tool writes, creates,
    /// moves, or deletes, checked by [`refuse_git_metadata_write`].
    pub fn resolve_write_path_for_tool(path_str: &str) -> Result<PathBuf> {
        let resolved = resolve_path_for_tool(path_str)?;
        refuse_git_metadata_write(&resolved)?;
        Ok(resolved)
    }

    /// The one host-side guard for tool writes. Host `git` runs commands
    /// that repository metadata names, so no tool writes there, whatever the
    /// approval mode.
    ///
    /// Refuses `path` when, as given or with symlinks in its existing part
    /// resolved, it has a `.git` component (any case, any depth: the
    /// workspace repository, nested repositories, and submodule git dirs
    /// under `.git/modules`), or when an existing ancestor is itself a git
    /// directory (a bare repository or a git dir a `.git` file points to).
    pub fn refuse_git_metadata_write(path: &Path) -> Result<()> {
        let resolved = resolve_existing_prefix(path);
        let named_git = [path, resolved.as_path()].into_iter().any(|candidate| {
            candidate.components().any(|part| {
                part.as_os_str()
                    .to_str()
                    .is_some_and(caliche_policy::is_git_dir_name)
            })
        });
        if named_git || resolved.ancestors().any(caliche_policy::is_git_dir) {
            bail!(
                "refusing to write '{}': it is inside a git repository's metadata",
                path.display()
            );
        }
        Ok(())
    }

    /// `path` with its longest existing prefix canonicalized, so symlinks
    /// there are followed; the missing tail is appended unchanged.
    fn resolve_existing_prefix(path: &Path) -> PathBuf {
        let mut tail = Vec::new();
        let mut current = path;
        loop {
            if let Ok(real) = current.canonicalize() {
                return tail.iter().rev().fold(real, |acc, part| acc.join(part));
            }
            match (current.parent(), current.file_name()) {
                (Some(parent), Some(name)) => {
                    tail.push(name.to_os_string());
                    current = parent;
                }
                _ => return path.to_path_buf(),
            }
        }
    }

    /// Validates file size against a maximum
    pub fn validate_file_size(path: &Path, max_size: u64) -> Result<u64> {
        let metadata = std::fs::metadata(path).context("Failed to get file metadata")?;

        let file_size = metadata.len();
        if file_size > max_size {
            bail!(
                "File is too large ({file_size} bytes). Maximum allowed size is {max_size} bytes."
            );
        }

        Ok(file_size)
    }
}

/// Parameter extraction helpers
pub mod params {
    use anyhow::{Context, Result, bail};
    use serde_json::Value;

    use beef::lean::Cow;

    use crate::utils::string_pool;

    /// Extract a required operation string used by operation-dispatch tools.
    pub fn extract_operation(args: &Value) -> Result<String> {
        extract_string(args, "operation", None).context("Missing 'operation' parameter")
    }

    /// Extracts a string parameter from JSON arguments
    pub fn extract_string(args: &Value, name: &str, default: Option<&str>) -> Result<String> {
        match args[name].as_str() {
            Some(s) => Ok(s.to_string()),
            None => {
                if let Some(def) = default {
                    Ok(def.to_string())
                } else {
                    bail!("Parameter '{name}' is required and must be a string");
                }
            }
        }
    }

    /// Extracts a string parameter with a default value
    #[must_use]
    pub fn extract_string_or(args: &Value, name: &str, default: &str) -> String {
        args[name].as_str().unwrap_or(default).to_string()
    }

    /// Extracts a string as Cow for zero-copy usage
    pub fn extract_string_cow(
        args: &Value,
        name: &str,
        default: Option<&str>,
    ) -> Result<Cow<'static, str>> {
        match args[name].as_str() {
            Some(s) => Ok(Cow::owned(s.to_string())),
            None => {
                if let Some(def) = default {
                    Ok(Cow::owned(def.to_string()))
                } else {
                    bail!("Parameter '{name}' is required and must be a string");
                }
            }
        }
    }

    /// Extracts an operation string as interned Arc<str> for efficient reuse
    pub fn extract_operation_interned(args: &Value) -> Result<std::sync::Arc<str>> {
        match args["operation"].as_str() {
            Some(op) => Ok(string_pool::intern(op)),
            None => bail!("Missing 'operation' parameter"),
        }
    }

    /// Parse a JSON number or numeric string as `u64`.
    ///
    /// Models often emit line numbers as strings (`"120"`) or as integral
    /// floats (`120.0`). Non-integral floats and other types return `None`.
    #[must_use]
    pub fn coerce_u64(value: &Value) -> Option<u64> {
        match value {
            Value::Number(number) => number
                .as_u64()
                .or_else(|| number.as_f64().and_then(integral_f64_to_u64)),
            Value::String(text) => {
                let text = text.trim();
                if let Ok(parsed) = text.parse::<u64>() {
                    return Some(parsed);
                }
                text.parse::<f64>().ok().and_then(integral_f64_to_u64)
            }
            _ => None,
        }
    }

    /// Extracts a number parameter from JSON arguments
    pub fn extract_number(args: &Value, name: &str, default: Option<u64>) -> Result<u64> {
        match args.get(name).filter(|value| !value.is_null()) {
            Some(value) => coerce_u64(value).ok_or_else(|| {
                anyhow::anyhow!("Parameter '{name}' is required and must be a number, got {value}")
            }),
            None => default.context(format!(
                "Parameter '{name}' is required and must be a number"
            )),
        }
    }

    /// Extracts a number parameter with a default value
    #[must_use]
    pub fn extract_number_or(args: &Value, name: &str, default: u64) -> u64 {
        args.get(name).and_then(coerce_u64).unwrap_or(default)
    }

    /// Extracts a float parameter from JSON arguments
    pub fn extract_float(args: &Value, name: &str, default: Option<f64>) -> Result<f64> {
        match args[name].as_f64() {
            Some(n) => Ok(n),
            None => {
                if let Some(def) = default {
                    Ok(def)
                } else {
                    bail!("Parameter '{name}' is required and must be a number");
                }
            }
        }
    }

    /// Extracts a float parameter with a default value
    #[must_use]
    pub fn extract_float_or(args: &Value, name: &str, default: f64) -> f64 {
        args[name].as_f64().unwrap_or(default)
    }

    /// Extracts a boolean parameter from JSON arguments
    #[must_use]
    pub fn extract_bool(args: &Value, name: &str, default: bool) -> bool {
        args[name].as_bool().unwrap_or(default)
    }

    fn integral_f64_to_u64(value: f64) -> Option<u64> {
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
            return None;
        }
        // Integers above 2^53 are not exact in f64.
        if value > (1u64 << 53) as f64 {
            return None;
        }
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "value is a finite non-negative integral f64 at or below 2^53"
        )]
        Some(value as u64)
    }

    /// Extracts an array parameter from JSON arguments
    pub fn extract_array(args: &Value, name: &str) -> Result<Vec<Value>> {
        match args[name].as_array() {
            Some(arr) => Ok(arr.clone()),
            None => bail!("Parameter '{name}' must be an array"),
        }
    }
}

/// HTTP and URL utilities
pub mod http {
    use anyhow::{Result, bail};

    /// Validates that a URL uses HTTP or HTTPS and does not target private,
    /// loopback, link-local, CGNAT, or cloud-metadata addresses.
    pub fn validate_http_url(url: &str) -> Result<()> {
        crate::gateway::egress::current_policy()?
            .validate(crate::gateway::egress::EgressClass::Tool, url)
            .map(|_| ())
    }

    /// Checks if a URL is a valid HTTP/HTTPS URL
    #[must_use]
    pub fn is_valid_http_url(url: &str) -> bool {
        url.starts_with("http://") || url.starts_with("https://")
    }

    /// Resolves a relative URL against a base URL
    pub fn resolve_url(url: &str, base_url: &str) -> Result<String> {
        // If it's already absolute, return as-is
        if url.starts_with("http://") || url.starts_with("https://") {
            return Ok(url.to_string());
        }

        // Try to resolve relative URLs
        if let Ok(base) = url::Url::parse(base_url)
            && let Ok(resolved) = base.join(url)
        {
            return Ok(resolved.to_string());
        }

        bail!("Failed to resolve URL: {url}");
    }
}

/// Text processing utilities
pub mod text {
    use regex::Regex;
    use std::sync::OnceLock;

    // Compile regex patterns once at runtime using OnceLock for thread-safe lazy initialization
    static NUMERIC_HTML_ENTITY_PATTERN: OnceLock<Regex> = OnceLock::new();
    static HEX_HTML_ENTITY_PATTERN: OnceLock<Regex> = OnceLock::new();
    static SCRIPT_TAG_PATTERN: OnceLock<Regex> = OnceLock::new();
    static STYLE_TAG_PATTERN: OnceLock<Regex> = OnceLock::new();
    static HTML_COMMENT_PATTERN: OnceLock<Regex> = OnceLock::new();
    static HTML_TAG_PATTERN: OnceLock<Regex> = OnceLock::new();
    static WHITESPACE_PATTERN: OnceLock<Regex> = OnceLock::new();

    fn get_numeric_entity_pattern() -> &'static Regex {
        NUMERIC_HTML_ENTITY_PATTERN.get_or_init(|| {
            Regex::new(r"&#(\d+);")
                .expect("Failed to compile numeric HTML entity regex - this should never happen")
        })
    }

    fn get_hex_entity_pattern() -> &'static Regex {
        HEX_HTML_ENTITY_PATTERN.get_or_init(|| {
            Regex::new(r"&#x([0-9a-fA-F]+);")
                .expect("Failed to compile hex HTML entity regex - this should never happen")
        })
    }

    fn get_script_tag_pattern() -> &'static Regex {
        SCRIPT_TAG_PATTERN.get_or_init(|| {
            Regex::new(r"(?is)<script[^>]*>.*?</script>")
                .expect("Failed to compile script tag regex - this should never happen")
        })
    }

    fn get_style_tag_pattern() -> &'static Regex {
        STYLE_TAG_PATTERN.get_or_init(|| {
            Regex::new(r"(?is)<style[^>]*>.*?</style>")
                .expect("Failed to compile style tag regex - this should never happen")
        })
    }

    fn get_html_comment_pattern() -> &'static Regex {
        HTML_COMMENT_PATTERN.get_or_init(|| {
            Regex::new(r"(?s)<!--.*?-->")
                .expect("Failed to compile HTML comment regex - this should never happen")
        })
    }

    fn get_html_tag_pattern() -> &'static Regex {
        HTML_TAG_PATTERN.get_or_init(|| {
            Regex::new(r"<[^>]+>")
                .expect("Failed to compile HTML tag regex - this should never happen")
        })
    }

    fn get_whitespace_pattern() -> &'static Regex {
        WHITESPACE_PATTERN.get_or_init(|| {
            Regex::new(r"\s+")
                .expect("Failed to compile whitespace regex - this should never happen")
        })
    }

    /// Decodes common HTML entities to their text equivalents
    #[must_use]
    pub fn decode_html_entities(text: &str) -> String {
        let mut result = text.to_string();

        // Common HTML entities
        let entities = [
            ("&amp;", "&"),
            ("&lt;", "<"),
            ("&gt;", ">"),
            ("&quot;", "\""),
            ("&#39;", "'"),
            ("&apos;", "'"),
            ("&nbsp;", " "),
            ("&copy;", "©"),
            ("&reg;", "®"),
            ("&trade;", "™"),
        ];

        for (entity, replacement) in &entities {
            result = result.replace(entity, replacement);
        }

        // Handle numeric entities like &#123; or &#x7B;
        let numeric_pattern = get_numeric_entity_pattern();
        result = numeric_pattern
            .replace_all(&result, |caps: &regex::Captures| {
                if let Ok(num) = caps[1].parse::<u32>()
                    && let Some(ch) = std::char::from_u32(num)
                {
                    return ch.to_string();
                }
                caps[0].to_string()
            })
            .to_string();

        let hex_pattern = get_hex_entity_pattern();
        result = hex_pattern
            .replace_all(&result, |caps: &regex::Captures| {
                if let Ok(num) = u32::from_str_radix(&caps[1], 16)
                    && let Some(ch) = std::char::from_u32(num)
                {
                    return ch.to_string();
                }
                caps[0].to_string()
            })
            .to_string();

        result
    }

    /// Cleans HTML by removing tags and scripts, returning plain text
    #[must_use]
    pub fn clean_html_text(html: &str) -> String {
        let mut cleaned = html.to_string();

        // Remove script tags and their content
        let script_pattern = get_script_tag_pattern();
        cleaned = script_pattern.replace_all(&cleaned, "").to_string();

        // Remove style tags and their content
        let style_pattern = get_style_tag_pattern();
        cleaned = style_pattern.replace_all(&cleaned, "").to_string();

        // Remove HTML comments
        let comment_pattern = get_html_comment_pattern();
        cleaned = comment_pattern.replace_all(&cleaned, "").to_string();

        // Remove HTML tags
        let tag_pattern = get_html_tag_pattern();
        cleaned = tag_pattern.replace_all(&cleaned, " ").to_string();

        // Decode HTML entities
        cleaned = decode_html_entities(&cleaned);

        // Normalize whitespace
        let whitespace_pattern = get_whitespace_pattern();
        cleaned = whitespace_pattern.replace_all(&cleaned, " ").to_string();

        // Trim and clean up
        cleaned.trim().to_string()
    }
}

/// Parameter schema helpers for consistent tool parameter definitions
pub mod schema {
    use serde_json::json;

    /// Creates a timeout parameter schema
    #[must_use]
    pub fn timeout_param(default: u64) -> serde_json::Value {
        json!({
            "type": "number",
            "description": "Request timeout in seconds",
            "default": default
        })
    }

    /// Creates a path parameter schema
    #[must_use]
    pub fn path_param(description: &str) -> serde_json::Value {
        json!({
            "type": "string",
            "description": description
        })
    }

    /// Creates a `max_length` parameter schema
    #[must_use]
    pub fn max_length_param(default: u64, description: Option<&str>) -> serde_json::Value {
        json!({
            "type": "number",
            "description": description.unwrap_or("Maximum length of content to return"),
            "default": default
        })
    }

    /// Creates a `max_size` parameter schema
    #[must_use]
    pub fn max_size_param(default: u64) -> serde_json::Value {
        json!({
            "type": "number",
            "description": "Maximum file size in bytes",
            "default": default
        })
    }

    /// Creates a boolean parameter schema with default
    #[must_use]
    pub fn bool_param(description: &str, default: bool) -> serde_json::Value {
        json!({
            "type": "boolean",
            "description": description,
            "default": default
        })
    }

    /// Creates a number parameter schema with default
    #[must_use]
    pub fn number_param(description: &str, default: u64) -> serde_json::Value {
        json!({
            "type": "number",
            "description": description,
            "default": default
        })
    }

    /// Creates a string parameter schema
    #[must_use]
    pub fn string_param(description: &str) -> serde_json::Value {
        json!({
            "type": "string",
            "description": description
        })
    }

    /// Creates an enum parameter schema
    #[must_use]
    pub fn enum_param(description: &str, variants: Vec<&str>) -> serde_json::Value {
        json!({
            "type": "string",
            "enum": variants,
            "description": description
        })
    }
}

/// Structured error types for tools
#[derive(Debug, Clone)]
pub enum ToolError {
    InvalidParameter {
        name: String,
        reason: String,
    },
    MissingParameter(String),
    ExecutionFailed {
        operation: String,
        cause: String,
    },
    Unauthorized(String),
    Timeout {
        operation: String,
    },
    ValidationFailed {
        field: String,
        reason: String,
    },
    ResourceNotFound {
        resource_type: String,
        identifier: String,
    },
    RateLimited {
        retry_after: Option<u64>,
    },
    InternalError(String),
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter { name, reason } => {
                write!(f, "Invalid parameter '{name}': {reason}")
            }
            Self::MissingParameter(name) => {
                write!(f, "Missing required parameter: {name}")
            }
            Self::ExecutionFailed { operation, cause } => {
                write!(f, "Execution failed for '{operation}': {cause}")
            }
            Self::Unauthorized(msg) => write!(f, "Unauthorized: {msg}"),
            Self::Timeout { operation } => {
                write!(f, "Operation '{operation}' timed out")
            }
            Self::ValidationFailed { field, reason } => {
                write!(f, "Validation failed for '{field}': {reason}")
            }
            Self::ResourceNotFound {
                resource_type,
                identifier,
            } => {
                write!(f, "{resource_type} '{identifier}' not found")
            }
            Self::RateLimited { retry_after } => {
                if let Some(seconds) = retry_after {
                    write!(f, "Rate limited. Retry after {seconds} seconds")
                } else {
                    write!(f, "Rate limited")
                }
            }
            Self::InternalError(msg) => write!(f, "Internal error: {msg}"),
        }
    }
}

impl std::error::Error for ToolError {}

/// Error message formatting utilities
pub mod errors {
    use super::ToolError;

    /// Formats a file operation error message
    #[must_use]
    pub fn format_file_error(operation: &str, path: &str, error: &str) -> String {
        format!("Failed to {operation} file '{path}': {error}")
    }

    /// Formats a parameter error message
    #[must_use]
    pub fn format_param_error(param: &str, expected: &str) -> String {
        format!("Parameter '{param}' is invalid: expected {expected}")
    }

    /// Formats a validation error message
    #[must_use]
    pub fn format_validation_error(field: &str, reason: &str) -> String {
        format!("Validation failed for '{field}': {reason}")
    }

    /// Converts `ToolError` to `ToolResult` error string
    #[must_use]
    pub fn tool_error_to_string(error: ToolError) -> String {
        error.to_string()
    }
}

/// Sandbox exec output as the model sees it.
#[cfg(feature = "sandbox")]
pub mod sandbox_exec {
    use caliche_sandbox::{EXEC_OUTPUT_LIMIT, ExecResult};

    /// One line per stream the sandbox cut at [`EXEC_OUTPUT_LIMIT`], each
    /// starting with a newline; empty when neither stream was cut.
    ///
    /// Append it after all stream text: model-facing bounds keep the tail, so
    /// a note between stdout and stderr can fall out of view.
    #[must_use]
    pub fn truncation_note(result: &ExecResult) -> String {
        let limit_mib = EXEC_OUTPUT_LIMIT / (1024 * 1024);
        [
            ("stdout", result.stdout_truncated),
            ("stderr", result.stderr_truncated),
        ]
        .into_iter()
        .filter(|(_, truncated)| *truncated)
        .map(|(stream, _)| format!("\n[{stream} truncated at {limit_mib} MiB]"))
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::path::check_path_traversal;
    use std::path::PathBuf;

    #[test]
    fn traversal_rejects_parent_components() {
        check_path_traversal("../secret").expect_err("leading parent");
        check_path_traversal("nested/../../etc/passwd").expect_err("nested parent");
        check_path_traversal("a\0b").expect_err("NUL");
    }

    #[test]
    fn traversal_allows_dots_in_filename() {
        check_path_traversal("foo..bar.txt").expect("dots in name");
        check_path_traversal("./src/main.rs").expect("curdir");
        check_path_traversal("/absolute/ok.txt").expect("absolute without parent");
    }

    /// On a file system that ignores case, a path that names the workspace
    /// in other case is still inside it, so a link there leading out is
    /// refused.
    #[cfg(target_os = "macos")]
    #[test]
    fn a_link_out_named_in_other_case_is_refused() {
        let ws = tempfile::tempdir().expect("ws");
        let out = tempfile::tempdir().expect("outside");
        let root = ws.path().canonicalize().expect("root");
        let outside = out.path().canonicalize().expect("outside");
        std::os::unix::fs::symlink(&outside, root.join("escape")).expect("link");
        let name = root
            .file_name()
            .expect("name")
            .to_string_lossy()
            .to_uppercase();
        let lexical = root.with_file_name(name).join("escape");
        super::set_tool_workspace_root(Some(root));
        super::set_tool_host_fs_unlocked(false);

        let refused = super::path::refuse_link_out_of_workspace(&lexical, &outside);

        super::set_tool_workspace_root(None);
        assert_eq!(
            refused.map_err(|err| err.to_string()),
            Err(format!(
                "Path '{}' is inside the workspace but leads through a link to '{}', outside it",
                lexical.display(),
                outside.display()
            ))
        );
    }

    /// A path resolved inside the workspace and then swapped for a link to
    /// an outside file is neither read nor written by the tool file helpers.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_resolved_path_swapped_for_a_link_is_not_read_or_written() {
        use super::path;
        let ws = tempfile::tempdir().expect("ws");
        let out = tempfile::tempdir().expect("outside");
        let root = ws.path().canonicalize().expect("root");
        let secret = out
            .path()
            .canonicalize()
            .expect("outside")
            .join("secret.txt");
        std::fs::write(&secret, "OUTSIDE-SECRET").expect("secret");
        std::fs::write(root.join("a.txt"), "inside").expect("a.txt");
        super::set_tool_workspace_root(Some(root.clone()));
        super::set_tool_host_fs_unlocked(false);
        let resolved = path::validate_and_resolve_path("a.txt", Some(true))
            .await
            .expect("resolved inside");
        std::fs::remove_file(root.join("a.txt")).expect("remove");
        std::os::unix::fs::symlink(&secret, root.join("a.txt")).expect("swap");

        let reached = [
            (
                "read_resolved_bytes_async",
                path::read_resolved_bytes_async(&resolved).await.is_ok(),
            ),
            (
                "read_resolved_text_async",
                path::read_resolved_text_async(&resolved).await.is_ok(),
            ),
            ("read_tool_text", path::read_tool_text(&resolved).is_ok()),
            (
                "open_tool_file read",
                path::open_tool_file(&resolved, path::ToolOpen::Read).is_ok(),
            ),
            (
                "write_tool_file",
                path::write_tool_file(&resolved, b"PWNED").is_ok(),
            ),
        ];
        super::set_tool_workspace_root(None);

        assert_eq!(
            reached,
            [
                ("read_resolved_bytes_async", false),
                ("read_resolved_text_async", false),
                ("read_tool_text", false),
                ("open_tool_file read", false),
                ("write_tool_file", false),
            ]
        );
        assert_eq!(
            std::fs::read_to_string(&secret).expect("secret"),
            "OUTSIDE-SECRET"
        );
    }

    /// The metadata helpers describe a plain file and directory, and give
    /// nothing for a file link or a path through a directory link that
    /// leads out of the workspace.
    #[cfg(unix)]
    #[test]
    fn metadata_through_a_link_is_refused() {
        use super::path;
        let ws = tempfile::tempdir().expect("ws");
        let out = tempfile::tempdir().expect("outside");
        let root = ws.path().canonicalize().expect("root");
        let outside = out.path().canonicalize().expect("outside");
        std::fs::write(outside.join("secret.txt"), "OUTSIDE-SECRET").expect("secret");
        std::fs::write(root.join("a.txt"), "inside").expect("a.txt");
        std::fs::create_dir(root.join("dir")).expect("dir");
        std::os::unix::fs::symlink(outside.join("secret.txt"), root.join("leak.txt"))
            .expect("file link");
        std::os::unix::fs::symlink(&outside, root.join("linked")).expect("dir link");
        super::set_tool_workspace_root(Some(root.clone()));
        super::set_tool_host_fs_unlocked(false);

        let described = |name: &str| {
            let place = root.join(name);
            (
                path::resolved_metadata(&place).map(|meta| meta.len()).ok(),
                path::tool_metadata(&place).map(|meta| meta.len()).ok(),
            )
        };
        let kinds = [
            described("a.txt"),
            described("leak.txt"),
            described("linked/secret.txt"),
        ];
        let dir_is_dir = path::tool_metadata(&root.join("dir")).map(|meta| meta.is_dir());
        super::set_tool_workspace_root(None);

        assert_eq!(kinds, [(Some(6), Some(6)), (None, None), (None, None)]);
        assert!(dir_is_dir.is_ok_and(|is_dir| is_dir));
    }

    #[test]
    fn resolve_path_for_tool_rejects_foreign_absolute() {
        super::set_tool_host_fs_unlocked(false);
        let err = super::path::resolve_path_for_tool("/etc/passwd").expect_err("foreign absolute");
        assert!(err.to_string().contains("outside workspace"));
    }

    #[test]
    fn resolve_path_for_tool_allows_host_fs_when_unlocked() {
        // An absolute path outside the workspace on each platform.
        #[cfg(not(windows))]
        let (outside, expected) = ("/etc/passwd", "/etc/passwd");
        #[cfg(windows)]
        let (outside, expected) = (r"C:\Windows\win.ini", "C:/Windows/win.ini");
        super::set_tool_host_fs_unlocked(true);
        let resolved = super::path::resolve_path_for_tool(outside).expect("unlocked");
        super::set_tool_host_fs_unlocked(false);
        assert_eq!(resolved, PathBuf::from(expected));
    }

    #[test]
    fn validate_http_url_rejects_loopback_and_file() {
        super::http::validate_http_url("http://127.0.0.1/secret").expect_err("loopback");
        super::http::validate_http_url("http://169.254.169.254/latest/meta-data/")
            .expect_err("link-local metadata");
        super::http::validate_http_url("file:///etc/passwd").expect_err("file scheme");
        super::http::validate_http_url("http://[::ffff:127.0.0.1]/")
            .expect_err("mapped ipv6 loopback");
    }
}
