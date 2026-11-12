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

use crate::paths::{EntryKind, list_dir_beneath, open_file_beneath};
use caliche_policy::{COMMONDIR_FILE, GIT_DIR_NAME, common_dir_path, gitfile_path, is_git_dir};
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
    /// these paths and discovers nothing above the work tree.
    #[must_use]
    pub fn git_env(&self) -> ([(&'static str, OsString); 3], [&'static str; 5]) {
        let ceiling = self
            .work_tree
            .parent()
            .unwrap_or(&self.work_tree)
            .as_os_str()
            .to_owned();
        (
            [
                (GIT_DIR_VAR, self.git_dir.as_os_str().to_owned()),
                (GIT_WORK_TREE_VAR, self.work_tree.as_os_str().to_owned()),
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

    /// A `git` command pinned with [`Self::pin`] that runs in the work tree.
    #[must_use]
    pub fn git_command(&self) -> std::process::Command {
        let mut command = std::process::Command::new("git");
        self.pin(&mut command);
        command.current_dir(&self.work_tree);
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
    let dot_git = dir.join(GIT_DIR_NAME);
    let kind = list_dir_beneath(workspace, dir)?
        .into_iter()
        .find(|entry| entry.name == GIT_DIR_NAME)
        .map(|entry| entry.kind)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, refusal::no_repository(dir)))?;
    let (git_dir, own_worktree) = match kind {
        EntryKind::Dir => (dot_git, false),
        EntryKind::File => {
            let text = read_beneath(workspace, &dot_git)?;
            let named =
                gitfile_path(&text).ok_or_else(|| denied(refusal::not_a_git_entry(&dot_git)))?;
            let git_dir = canonical(&dir.join(named))?;
            if git_dir.starts_with(workspace) {
                (git_dir, false)
            } else if dir == workspace && names_back(&git_dir, &dot_git) {
                (git_dir, true)
            } else {
                return Err(denied(refusal::git_dir_outside(&dot_git, &git_dir)));
            }
        }
        EntryKind::Other => return Err(denied(refusal::not_a_git_entry(&dot_git))),
    };
    let common_dir = common_dir_of(workspace, &git_dir, own_worktree)?;
    if !is_git_dir(&git_dir) {
        return Err(denied(refusal::not_a_git_dir(&git_dir)));
    }
    Ok(CheckedRepository {
        work_tree: dir.to_path_buf(),
        git_dir,
        common_dir,
    })
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
