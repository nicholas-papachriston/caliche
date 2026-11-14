//! The repository a tool may use, checked before any git command or
//! library call reads it.
//!
//! [`check_repository`] reads the `.git` of a directory in the workspace
//! with the no-link walk: a `.git` that is a link is refused, and so is a
//! gitfile or `commondir` that names a git dir outside the workspace. Only
//! the workspace root's own `.git` may name a git dir elsewhere, and only
//! the linked-worktree git dir whose `gitdir` file names that `.git` back.
//! [`CheckedRepository::git_env`] gives the environment that pins `git` to
//! the checked paths.

use crate::paths::{
    DirEntryBeneath, EntryKind, external_tool_path, list_dir_beneath, open_file_beneath,
};
use caliche_policy::{
    COMMONDIR_FILE, GIT_DIR_NAME, HEAD_FILE, OBJECTS_DIR, REFS_DIR, common_dir_path, gitfile_path,
};
use std::ffi::OsString;
use std::io::{self, Read as _};
use std::path::{Path, PathBuf};

/// The file in a linked worktree's git dir that names the worktree's `.git`.
pub const WORKTREE_GITDIR_FILE: &str = "gitdir";

/// Environment variables that name the repository `git` uses.
pub const GIT_DIR_VAR: &str = "GIT_DIR";
/// See [`GIT_DIR_VAR`].
pub const GIT_WORK_TREE_VAR: &str = "GIT_WORK_TREE";
/// See [`GIT_DIR_VAR`].
pub const GIT_CEILING_DIRECTORIES_VAR: &str = "GIT_CEILING_DIRECTORIES";

/// Inherited variables that would point `git` at other repository files.
pub const GIT_REDIRECT_VARS: [&str; 5] = [
    "GIT_COMMON_DIR",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
];

/// A work tree and the git dir and common dir it uses, all checked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckedRepository {
    /// The canonical directory that holds `.git`.
    pub work_tree: PathBuf,
    /// The canonical git dir.
    pub git_dir: PathBuf,
    /// The canonical common dir (the git dir unless it names one).
    pub common_dir: PathBuf,
}

impl CheckedRepository {
    /// Variables to set, then variables to remove, so `git` uses exactly
    /// these paths and discovers nothing above the work tree. The paths are
    /// given in the form an external program reads (no `\\?\` prefix on
    /// Windows, see [`external_tool_path`]).
    #[must_use]
    pub fn git_env(&self) -> ([(&'static str, OsString); 3], [&'static str; 5]) {
        let plain = |path: &Path| external_tool_path(path).into_os_string();
        let ceiling = plain(self.work_tree.parent().unwrap_or(&self.work_tree));
        (
            [
                (GIT_DIR_VAR, plain(&self.git_dir)),
                (GIT_WORK_TREE_VAR, plain(&self.work_tree)),
                (GIT_CEILING_DIRECTORIES_VAR, ceiling),
            ],
            GIT_REDIRECT_VARS,
        )
    }

    /// Apply [`Self::git_env`] to `command`.
    pub fn pin(&self, command: &mut std::process::Command) {
        let (set, remove) = self.git_env();
        for name in remove {
            command.env_remove(name);
        }
        command.envs(set);
    }

    /// A `git` command built by [`caliche_policy::git_command`], pinned
    /// with [`Self::pin`], that runs in the work tree. It is only run inside
    /// the sandbox, by [`Self::run_read_only`].
    pub(crate) fn pinned(&self) -> std::process::Command {
        let mut command = caliche_policy::git_command();
        self.pin(&mut command);
        command.current_dir(external_tool_path(&self.work_tree));
        command
    }
}

/// Messages for a refused repository.
pub mod refusal {
    use std::path::Path;

    /// `dir` is not the canonical workspace or below it.
    #[must_use]
    pub fn outside_workspace(dir: &Path) -> String {
        format!("{} is outside the workspace", dir.display())
    }

    /// `dir` has no `.git`.
    #[must_use]
    pub fn no_repository(dir: &Path) -> String {
        format!("{} has no .git", dir.display())
    }

    /// The `.git` is a link or special file.
    #[must_use]
    pub fn not_a_git_entry(dot_git: &Path) -> String {
        format!(
            "{} is a link or special file, not a git dir or gitfile",
            dot_git.display()
        )
    }

    /// A gitfile or `commondir` names a git dir outside the workspace.
    #[must_use]
    pub fn git_dir_outside(named_by: &Path, git_dir: &Path) -> String {
        format!(
            "{} names {}, outside the workspace",
            named_by.display(),
            git_dir.display()
        )
    }

    /// The repository's config names programs git could run.
    #[must_use]
    pub fn program_keys(git_dir: &Path, keys: &[String]) -> String {
        format!(
            "{} config names programs git would run: {}",
            git_dir.display(),
            keys.join(", ")
        )
    }

    /// A link, FIFO, socket or device inside a git dir.
    #[must_use]
    pub fn special_in_git_dir(path: &Path) -> String {
        format!(
            "{} is a link or special file inside a git dir",
            path.display()
        )
    }

    /// An alternates file pointing git at another object store.
    #[must_use]
    pub fn alternates(path: &Path) -> String {
        format!("{} points git at another object store", path.display())
    }

    /// A `commondir` inside a git dir that is not the checked git dir's own
    /// or a linked worktree's, such as one in a submodule's git dir.
    #[must_use]
    pub fn nested_commondir(path: &Path) -> String {
        format!("{} points git at another git dir", path.display())
    }

    /// The named directory is not a git dir.
    #[must_use]
    pub fn not_a_git_dir(git_dir: &Path) -> String {
        format!("{} is not a git dir", git_dir.display())
    }
}

/// Check the repository whose `.git` is in `dir`.
///
/// `workspace` is the canonical workspace root and `dir` a canonical
/// directory at or below it. `dir/.git` and the files it names are read
/// without following a link below the workspace. A git dir or common dir
/// must be inside the workspace, except that the workspace root's gitfile
/// may name the linked-worktree git dir that names the root back.
///
/// # Errors
///
/// Returns [`io::ErrorKind::PermissionDenied`] with a [`refusal`] message
/// for a refused layout, [`io::ErrorKind::NotFound`] when `dir` has no
/// `.git`, or the error of a read that failed.
pub fn check_repository(workspace: &Path, dir: &Path) -> io::Result<CheckedRepository> {
    if !dir.starts_with(workspace) {
        return Err(denied(refusal::outside_workspace(dir)));
    }
    let (git_dir, own_worktree) = locate_git_dir(workspace, dir)?;
    let git_root = if own_worktree {
        git_dir.as_path()
    } else {
        workspace
    };
    let git_entries = check_git_dir_contents(git_root, &git_dir)?;
    let common_dir = common_dir_of(workspace, &git_dir, own_worktree)?;
    let common_entries = if common_dir == git_dir {
        git_entries.clone()
    } else {
        let common_root = if own_worktree {
            common_dir.as_path()
        } else {
            workspace
        };
        check_git_dir_contents(common_root, &common_dir)?
    };
    if own_worktree && !is_worktree_of(&git_dir, &common_dir) {
        return Err(denied(refusal::not_a_git_dir(&git_dir)));
    }
    require_git_layout(git_root, &git_dir, &git_entries, &common_entries)?;
    Ok(CheckedRepository {
        work_tree: dir.to_path_buf(),
        git_dir,
        common_dir,
    })
}

/// The git dir `dir/.git` names, and whether it is the linked-worktree git
/// dir the workspace root's gitfile may name outside the workspace.
fn locate_git_dir(workspace: &Path, dir: &Path) -> io::Result<(PathBuf, bool)> {
    let dot_git = dir.join(GIT_DIR_NAME);
    let kind = list_dir_beneath(workspace, dir)?
        .into_iter()
        .find(|entry| entry.name == GIT_DIR_NAME)
        .map(|entry| entry.kind)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, refusal::no_repository(dir)))?;
    match kind {
        EntryKind::Dir => Ok((dot_git, false)),
        EntryKind::File => {
            let text = read_beneath(workspace, &dot_git)?;
            let named =
                gitfile_path(&text).ok_or_else(|| denied(refusal::not_a_git_entry(&dot_git)))?;
            let git_dir = canonical(&dir.join(named))?;
            if git_dir.starts_with(workspace) {
                Ok((git_dir, false))
            } else if dir == workspace && names_back(&git_dir, &dot_git) {
                Ok((git_dir, true))
            } else {
                Err(denied(refusal::git_dir_outside(&dot_git, &git_dir)))
            }
        }
        EntryKind::Other => Err(denied(refusal::not_a_git_entry(&dot_git))),
    }
}

/// Walk every entry of `git_dir`, at or below `root`, without following a
/// link, and refuse a link or special file anywhere in it, an alternates
/// file anywhere (also in a submodule's git dir under `modules/`), and a
/// `commondir` other than the git dir's own and its linked worktrees'. Git
/// reads objects, refs and the index by path, so a link inside the git
/// dir, an alternates file or a `commondir` would let it read a repository
/// outside. Returns the git dir's own entries.
fn check_git_dir_contents(root: &Path, git_dir: &Path) -> io::Result<Vec<DirEntryBeneath>> {
    let top = list_dir_beneath(root, git_dir)?;
    let mut pending: Vec<(PathBuf, Vec<DirEntryBeneath>)> =
        vec![(git_dir.to_path_buf(), top.clone())];
    while let Some((dir, entries)) = pending.pop() {
        for entry in entries {
            let path = dir.join(&entry.name);
            match entry.kind {
                EntryKind::Other => return Err(denied(refusal::special_in_git_dir(&path))),
                EntryKind::File => {
                    if ALTERNATES_FILES.iter().any(|name| path.ends_with(name)) {
                        return Err(denied(refusal::alternates(&path)));
                    }
                    if entry.name == COMMONDIR_FILE && !is_own_commondir(git_dir, &path) {
                        return Err(denied(refusal::nested_commondir(&path)));
                    }
                }
                // Hooks never run (the pinned git sets core.hooksPath to an
                // empty dir), so a hook manager's links there are not read.
                EntryKind::Dir if path == git_dir.join(HOOKS_DIR) => {}
                EntryKind::Dir => {
                    let listed = list_dir_beneath(root, &path)?;
                    pending.push((path, listed));
                }
            }
        }
    }
    Ok(top)
}

/// Whether `path` is `<git_dir>/commondir` (checked as the common dir) or
/// `<git_dir>/worktrees/<id>/commondir` (a linked worktree's, which git
/// follows only for that worktree, checked when it is the one in use).
fn is_own_commondir(git_dir: &Path, path: &Path) -> bool {
    let Ok(rest) = path.strip_prefix(git_dir) else {
        return false;
    };
    let parts: Vec<_> = rest.components().map(|part| part.as_os_str()).collect();
    match parts.as_slice() {
        [file] => *file == COMMONDIR_FILE,
        [worktrees, _, file] => *worktrees == "worktrees" && *file == COMMONDIR_FILE,
        _ => false,
    }
}

/// The hooks dir of a git dir, which no checked git reads.
pub const HOOKS_DIR: &str = "hooks";

/// Files that point git at other object stores.
pub const ALTERNATES_FILES: [&str; 2] = ["objects/info/alternates", "objects/info/http-alternates"];

/// Whether `git_dir` is `<common_dir>/worktrees/<id>`, where
/// `git worktree add` puts a linked worktree's git dir.
fn is_worktree_of(git_dir: &Path, common_dir: &Path) -> bool {
    let Some(worktrees) = git_dir.parent() else {
        return false;
    };
    worktrees
        .file_name()
        .is_some_and(|name| name == "worktrees")
        && worktrees.parent() == Some(common_dir)
}

/// A git dir has a regular `HEAD` that names a ref or an object, and its
/// common dir has `objects` and `refs` directories, all read from the
/// no-link walk.
fn require_git_layout(
    root: &Path,
    git_dir: &Path,
    git_entries: &[DirEntryBeneath],
    common_entries: &[DirEntryBeneath],
) -> io::Result<()> {
    let has = |entries: &[DirEntryBeneath], name: &str, kind: EntryKind| {
        entries
            .iter()
            .any(|entry| entry.name == name && entry.kind == kind)
    };
    let laid_out = has(git_entries, HEAD_FILE, EntryKind::File)
        && has(common_entries, OBJECTS_DIR, EntryKind::Dir)
        && has(common_entries, REFS_DIR, EntryKind::Dir);
    if !laid_out || !valid_head(&read_beneath(root, &git_dir.join(HEAD_FILE))?) {
        return Err(denied(refusal::not_a_git_dir(git_dir)));
    }
    Ok(())
}

/// `ref: refs/...`, or an object id in hex.
fn valid_head(text: &str) -> bool {
    let text = text.trim_end_matches(['\n', '\r']);
    text.strip_prefix("ref: ")
        .is_some_and(|name| name.starts_with("refs/"))
        || (matches!(text.len(), 40 | 64) && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

/// The common dir `git_dir` names, read without following a link when the
/// git dir is in the workspace. A common dir outside the workspace is
/// accepted only for the workspace's own linked worktree.
fn common_dir_of(workspace: &Path, git_dir: &Path, own_worktree: bool) -> io::Result<PathBuf> {
    let file = git_dir.join(COMMONDIR_FILE);
    let text = if own_worktree {
        std::fs::read_to_string(&file)
    } else {
        read_beneath(workspace, &file)
    };
    let common = match text {
        Ok(text) => canonical(&git_dir.join(common_dir_path(&text)))?,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(git_dir.to_path_buf()),
        Err(err) => return Err(err),
    };
    if own_worktree || common.starts_with(workspace) {
        Ok(common)
    } else {
        Err(denied(refusal::git_dir_outside(&file, &common)))
    }
}

/// Whether the linked-worktree git dir `git_dir` names `dot_git` in its
/// `gitdir` file, as `git worktree add` writes it.
fn names_back(git_dir: &Path, dot_git: &Path) -> bool {
    let Ok(text) = std::fs::read_to_string(git_dir.join(WORKTREE_GITDIR_FILE)) else {
        return false;
    };
    let named = PathBuf::from(common_dir_path(&text));
    canonical(&named).is_ok_and(|named| named == dot_git)
}

fn read_beneath(workspace: &Path, file: &Path) -> io::Result<String> {
    let mut text = String::new();
    open_file_beneath(workspace, file)?.read_to_string(&mut text)?;
    Ok(text)
}

fn canonical(path: &Path) -> io::Result<PathBuf> {
    path.canonicalize()
}

fn denied(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, message)
}
