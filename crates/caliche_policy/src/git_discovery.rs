//! Which repository host `git` uses for a directory.
//!
//! At each directory, from the start upward, git takes `<dir>/.git` when it
//! is a gitfile naming a git dir or is a git dir itself, then `<dir>` when it
//! is a git dir (a bare repository), and otherwise moves up. Its git dir test
//! is loose: a `HEAD` that looks like a ref or an object id, and `objects`
//! and `refs` it can search, in the common dir when a `commondir` names one.
//!
//! [`discover`] does not walk past a `.git` it rejects. The caller refuses
//! that layout. It also reports a `HEAD` beside a `.git` it accepts, so a
//! later rejection of that `.git` cannot fall through to the directory as a
//! bare repository. Rejecting a `.git` that git would use is safe. Accepting
//! one that git rejects is not, because git would walk past it.

use crate::fs::{COMMONDIR_FILE, GIT_DIR_NAME, GITFILE_PREFIX, HEAD_FILE, OBJECTS_DIR, REFS_DIR};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Start of a `HEAD` file that names a ref.
const SYMREF_PREFIX: &str = "ref:";

/// Start of a ref name a `HEAD` may point to.
const REFS_PREFIX: &str = "refs/";

/// Hex digits in the shortest object id (SHA-1).
const OID_HEX_LEN: usize = 40;

/// `HEAD` contents tests use for a loose git dir.
pub const LOOSE_HEAD_CONTENTS: &str = "ref: refs/heads/main\n";

/// Environment variable that names git's global config file.
pub const GIT_CONFIG_GLOBAL_VAR: &str = "GIT_CONFIG_GLOBAL";

/// Environment variable that turns off git's system config.
pub const GIT_CONFIG_NOSYSTEM_VAR: &str = "GIT_CONFIG_NOSYSTEM";

/// Path used as an empty git config file.
pub const ISOLATED_GIT_CONFIG: &str = "/dev/null";

/// Names inside a git dir or common dir that git must be able to use.
/// A symlink at one of them can be replaced by deleting its target.
pub const GIT_DIR_LINKS: &[&str] = &[HEAD_FILE, OBJECTS_DIR, REFS_DIR, COMMONDIR_FILE];

/// What [`discover`] found walking up from a directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Discovery {
    /// Git uses `git_dir` for the work tree (or bare repository) at `dir`.
    Repository {
        /// The directory where discovery stopped.
        dir: PathBuf,
        /// The git dir it uses: `<dir>/.git`, the dir a gitfile there
        /// names, or `dir` itself for a bare repository.
        git_dir: PathBuf,
    },
    /// A `.git` entry that is neither a gitfile naming a git dir nor a git
    /// dir. Git skips it or fails on it.
    InvalidGitEntry(PathBuf),
    /// A `HEAD` entry, of any type, in a directory that is not a git dir,
    /// or beside a `.git` that was accepted.
    StrayHead(PathBuf),
    /// No directory from the start upward holds a repository.
    NoRepository,
}

/// The repository [`discover`] assigns to `workspace`, with an invalid
/// `.git` and a stray `HEAD` already separated out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkspaceGit {
    /// `workspace` has its own git dir, and that dir is not `workspace`.
    Own {
        /// The git dir `<workspace>/.git` or the dir its gitfile names.
        git_dir: PathBuf,
    },
    /// `workspace` itself is a bare repository.
    Bare {
        /// The same path as the workspace.
        git_dir: PathBuf,
    },
    /// A repository above `workspace`.
    Enclosing {
        /// The directory where discovery stopped.
        dir: PathBuf,
        /// The git dir of that repository.
        git_dir: PathBuf,
    },
    /// No repository at `workspace` or above it.
    None,
}

/// Why [`classify`] refused the walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClassifyFault {
    /// See [`Discovery::InvalidGitEntry`].
    InvalidGitEntry(PathBuf),
    /// See [`Discovery::StrayHead`].
    StrayHead(PathBuf),
}

/// Walk from `start` (itself first) to the filesystem root as git
/// discovery does. Ceiling directories and filesystem boundaries are not
/// honoured, so the walk never stops below where git would.
#[must_use]
pub fn discover(start: &Path) -> Discovery {
    for dir in start.ancestors() {
        if let Some(found) = discover_here(dir) {
            return found;
        }
    }
    Discovery::NoRepository
}

/// [`discover`] for `workspace`, as one result. Invalid entries and a
/// stray `HEAD` are [`Err`], so callers do not each decide what those mean.
#[must_use]
pub fn classify(workspace: &Path) -> Result<WorkspaceGit, ClassifyFault> {
    match discover(workspace) {
        Discovery::Repository { dir, git_dir } if dir == workspace && git_dir != workspace => {
            Ok(WorkspaceGit::Own { git_dir })
        }
        Discovery::Repository { dir, git_dir } if dir == workspace => {
            Ok(WorkspaceGit::Bare { git_dir })
        }
        Discovery::Repository { dir, git_dir } => Ok(WorkspaceGit::Enclosing { dir, git_dir }),
        Discovery::InvalidGitEntry(path) => Err(ClassifyFault::InvalidGitEntry(path)),
        Discovery::StrayHead(path) => Err(ClassifyFault::StrayHead(path)),
        Discovery::NoRepository => Ok(WorkspaceGit::None),
    }
}

/// Whether git takes `dir` as a git dir: a valid `HEAD`, and `objects` and
/// `refs` it can search (`access(X_OK)`), in the common dir that a
/// `commondir` file names when there is one.
#[must_use]
pub fn is_git_dir(dir: &Path) -> bool {
    let Ok(common) = common_dir(dir) else {
        return false;
    };
    searchable(&common.join(OBJECTS_DIR))
        && searchable(&common.join(REFS_DIR))
        && valid_head(&dir.join(HEAD_FILE))
}

/// What a host read of a git dir file or gitfile says when the path is not a
/// regular file (a FIFO, a socket, a device or a directory).
pub const NOT_A_REGULAR_GIT_FILE: &str = "is not a regular file";

/// Largest index [`read_git_file`] reads for its callers that read one.
pub const MAX_GIT_INDEX_BYTES: u64 = 1 << 30;

/// Largest git dir text file or gitfile [`read_git_text`] reads: `config`,
/// `HEAD`, `commondir`, a gitfile or a worktree record.
pub const MAX_GIT_TEXT_BYTES: u64 = 1 << 20;

/// Read a git dir file (an index, `config`, `HEAD`, `commondir` or a
/// worktree record) or a gitfile on the host, up to `max_bytes`. The open
/// never waits: on Unix it passes `O_NONBLOCK`, so a FIFO planted at the
/// path cannot stall the caller. Anything but a regular file is then
/// refused with [`NOT_A_REGULAR_GIT_FILE`]. At most `max_bytes` plus one
/// byte is read, and a file that holds more is refused, however its size
/// changes while it is read. A link is followed, as git follows it. A
/// sandboxed command can still place some of these files (a `.git` moved
/// into the workspace), so every read is limited in size and kind; a caller
/// that must not follow links, such as one reading `.gitmodules`, uses a
/// no-follow reader instead.
///
/// # Errors
///
/// The open or read error, or [`std::io::ErrorKind::InvalidData`] for a
/// path that is not a regular file or is over `max_bytes`.
pub fn read_git_file(path: &Path, max_bytes: u64) -> std::io::Result<Vec<u8>> {
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let refuse = |why: String| {
        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} {why}", path.display()),
        ))
    };
    if !file.metadata()?.is_file() {
        return refuse(NOT_A_REGULAR_GIT_FILE.to_owned());
    }
    read_bounded(file, path, max_bytes)
}

/// Read `reader`, the file at `path`, to its end, refusing it once it
/// holds more than `max_bytes`. At most `max_bytes` plus one byte is read,
/// however the file's size changes while it is read. Every host read of a
/// git dir file or gitfile goes through this limit.
///
/// # Errors
///
/// The read error, or [`std::io::ErrorKind::InvalidData`] naming `path`
/// when it holds more than `max_bytes`.
pub fn read_bounded(reader: impl Read, path: &Path, max_bytes: u64) -> std::io::Result<Vec<u8>> {
    let mut data = Vec::new();
    reader
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut data)?;
    if u64::try_from(data.len()).unwrap_or(u64::MAX) > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{} is larger than {max_bytes} bytes", path.display()),
        ));
    }
    Ok(data)
}

/// [`read_git_file`] with [`MAX_GIT_TEXT_BYTES`], as UTF-8 text.
///
/// # Errors
///
/// As [`read_git_file`], or [`std::io::ErrorKind::InvalidData`] when the
/// file is not UTF-8.
pub fn read_git_text(path: &Path) -> std::io::Result<String> {
    String::from_utf8(read_git_file(path, MAX_GIT_TEXT_BYTES)?)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))
}

/// The directory a gitfile names, relative to the gitfile's parent.
/// [`None`] when `dot_git` is missing, not a regular file, or not a gitfile.
/// A symlink is [`None`]: git follows it, and the link can be retargeted.
pub fn read_gitfile(dot_git: &Path) -> std::io::Result<Option<PathBuf>> {
    let meta = match std::fs::symlink_metadata(dot_git) {
        Ok(meta) => meta,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err),
    };
    if !meta.is_file() {
        return Ok(None);
    }
    let text = read_git_text(dot_git)?;
    let Some(named) = gitfile_path(&text) else {
        return Ok(None);
    };
    let parent = dot_git.parent().unwrap_or_else(|| Path::new(""));
    Ok(Some(parent.join(named)))
}

/// The path text of a gitfile, after the `gitdir: ` prefix. Git strips
/// every trailing `\n` and `\r` and keeps every other byte, including
/// spaces.
#[must_use]
pub fn gitfile_path(text: &str) -> Option<&str> {
    text.trim_end_matches(['\n', '\r'])
        .strip_prefix(GITFILE_PREFIX)
        .and_then(|rest| rest.strip_prefix(' '))
}

/// The dir `git_dir`'s `commondir` file names, or `git_dir` when the file
/// is absent. The file's trailing `\n` and `\r` are stripped. Spaces stay.
/// A zero-length file is an error: git dies when it reads one.
pub fn common_dir(git_dir: &Path) -> std::io::Result<PathBuf> {
    let file = git_dir.join(COMMONDIR_FILE);
    match read_git_text(&file) {
        Ok(text) if text.is_empty() => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "empty commondir",
        )),
        Ok(text) => Ok(git_dir.join(common_dir_path(&text))),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(git_dir.to_path_buf()),
        Err(err) => Err(err),
    }
}

/// `commondir` contents with trailing newlines removed and spaces kept.
#[must_use]
pub fn common_dir_path(text: &str) -> &str {
    text.trim_end_matches(['\n', '\r'])
}

/// Point `command` at an empty git config so a user's global config cannot
/// change the result.
pub fn clear_git_config(command: &mut std::process::Command) -> &mut std::process::Command {
    command
        .env(GIT_CONFIG_GLOBAL_VAR, ISOLATED_GIT_CONFIG)
        .env(GIT_CONFIG_NOSYSTEM_VAR, "1")
}

/// Write the loose git dir git's discovery accepts: `HEAD`, `objects/`,
/// and `refs/`.
pub fn write_loose_git_dir(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir.join(OBJECTS_DIR))?;
    std::fs::create_dir_all(dir.join(REFS_DIR))?;
    std::fs::write(dir.join(HEAD_FILE), LOOSE_HEAD_CONTENTS)?;
    Ok(())
}

/// A loose git dir at `common` and a linked worktree git dir
/// `common/worktrees/<name>` whose `commondir` is `../..`.
pub fn plant_linked_worktree(common: &Path, name: &str) -> std::io::Result<PathBuf> {
    let git_dir = common.join("worktrees").join(name);
    write_loose_git_dir(&git_dir)?;
    write_loose_git_dir(common)?;
    std::fs::write(git_dir.join(COMMONDIR_FILE), "../..\n")?;
    Ok(git_dir)
}

fn discover_here(dir: &Path) -> Option<Discovery> {
    let dot_git = dir.join(GIT_DIR_NAME);
    if exists(&dot_git) {
        return Some(accepted_git_or_stray(dir, &dot_git));
    }
    if is_git_dir(dir) {
        return Some(Discovery::Repository {
            dir: dir.to_path_buf(),
            git_dir: dir.to_path_buf(),
        });
    }
    let head = dir.join(HEAD_FILE);
    exists(&head).then(|| Discovery::StrayHead(head))
}

/// A `.git` git accepts, unless the directory also has a `HEAD`. That
/// `HEAD` would become a bare repository if git later rejected the `.git`.
fn accepted_git_or_stray(dir: &Path, dot_git: &Path) -> Discovery {
    let Some(git_dir) = named_git_dir(dot_git) else {
        return Discovery::InvalidGitEntry(dot_git.to_path_buf());
    };
    let head = dir.join(HEAD_FILE);
    if exists(&head) {
        Discovery::StrayHead(head)
    } else {
        Discovery::Repository {
            dir: dir.to_path_buf(),
            git_dir,
        }
    }
}

/// The git dir `dot_git` stands for, when git would use it.
fn named_git_dir(dot_git: &Path) -> Option<PathBuf> {
    if let Some(git_dir) = read_gitfile(dot_git).ok().flatten() {
        return is_git_dir(&git_dir).then_some(git_dir);
    }
    let meta = std::fs::symlink_metadata(dot_git).ok()?;
    (meta.is_dir() && is_git_dir(dot_git)).then(|| dot_git.to_path_buf())
}

/// Git's `HEAD` test. A symlink is accepted only when its own text starts
/// with `refs/`; git does not read through any other symlink. A file is
/// accepted when it holds `ref:` and a name under `refs/`, or starts with
/// an object id. Only ASCII space, tab, newline, and carriage return are
/// skipped after `ref:`.
fn valid_head(head: &Path) -> bool {
    let Ok(meta) = std::fs::symlink_metadata(head) else {
        return false;
    };
    if meta.file_type().is_symlink() {
        return std::fs::read_link(head)
            .is_ok_and(|target| target.to_string_lossy().starts_with(REFS_PREFIX));
    }
    let Ok(bytes) = read_git_file(head, MAX_GIT_TEXT_BYTES) else {
        return false;
    };
    head_text_is_valid(&String::from_utf8_lossy(&bytes))
}

fn head_text_is_valid(text: &str) -> bool {
    if let Some(name) = text.strip_prefix(SYMREF_PREFIX) {
        return name
            .trim_start_matches([' ', '\t', '\n', '\r'])
            .starts_with(REFS_PREFIX);
    }
    text.len() >= OID_HEX_LEN
        && text.as_bytes()[..OID_HEX_LEN]
            .iter()
            .all(u8::is_ascii_hexdigit)
}

/// Git's `access(path, X_OK)`. The owner bits are what the owner is
/// checked against, so a mode such as `0o010` fails for the owner.
#[cfg(unix)]
fn searchable(path: &Path) -> bool {
    use std::os::unix::ffi::OsStrExt;
    let Ok(c_path) = std::ffi::CString::new(path.as_os_str().as_bytes()) else {
        return false;
    };
    access_ok(c_path.as_ptr())
}

#[cfg(unix)]
fn access_ok(path: *const libc::c_char) -> bool {
    #[expect(
        unsafe_code,
        reason = "access(2) is how git tests that objects and refs are searchable"
    )]
    let rc = unsafe { libc::access(path, libc::X_OK) };
    rc == 0
}

#[cfg(not(unix))]
fn searchable(path: &Path) -> bool {
    std::fs::metadata(path).is_ok()
}

/// Whether `path` names an entry of any type, a dangling symlink included.
/// An entry that cannot be inspected counts as present.
fn exists(path: &Path) -> bool {
    !matches!(std::fs::symlink_metadata(path), Err(err) if err.kind() == std::io::ErrorKind::NotFound)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_gitfile_keeps_spaces_and_requires_the_prefix() {
        assert_eq!(gitfile_path("gitdir:  /usr\n"), Some(" /usr"));
        assert_eq!(gitfile_path("gitdir: x \n"), Some("x "));
        assert_eq!(gitfile_path("gitdir: meta\n\n"), Some("meta"));
        assert_eq!(gitfile_path("\ngitdir: meta\n"), None);
        assert_eq!(gitfile_path("gitdir:meta\n"), None);
    }

    #[test]
    fn a_commondir_keeps_spaces() {
        assert_eq!(common_dir_path("common \n"), "common ");
        assert_eq!(common_dir_path("../..\n\r"), "../..");
    }

    #[test]
    fn head_text_skips_only_space_tab_newline_and_cr() {
        assert!(head_text_is_valid("ref: refs/heads/main\n"));
        assert!(head_text_is_valid("ref:\trefs/heads/main\n"));
        assert!(!head_text_is_valid("ref:\u{000B}refs/heads/main\n"));
        assert!(!head_text_is_valid("ref:\u{000C}refs/heads/main\n"));
        assert!(!head_text_is_valid("ref:\u{00A0}refs/heads/main\n"));
    }
}
