//! Keeps workspace git metadata read-only under Landlock.
//!
//! Host-side `git` runs commands named in repository config and hooks, so a
//! sandboxed write to git metadata would run later outside the sandbox. The
//! metadata is every target `nested_git` returns: the root `.git`, the
//! `.git` path of each submodule in the indexes, recursively, and every
//! nested `.git` its bounded scan finds under the writable roots. Landlock only
//! adds access, so a writable workspace cannot leave a target out. The mode
//! is chosen when the instance is created:
//!
//! - `mount_namespace`: the only target is the root `.git`, and it exists.
//!   Each exec enters a new user and mount namespace and puts a read-only
//!   clone of `.git` over it before Landlock applies. Landlock then refuses
//!   mount changes, so the command cannot undo it, and the mount point
//!   cannot be renamed. A forked probe runs the same steps first and picks
//!   this mode only when they succeed.
//! - `top_level_grants`: there are nested targets, the root `.git` does not
//!   exist yet, or unprivileged user namespaces are blocked (for example by
//!   a seccomp filter or an AppArmor rule). No directory on the way from
//!   the workspace root to a target is
//!   granted; every other entry gets its own writable grant. A command
//!   cannot write, create, rename, or delete a target, and cannot rename a
//!   directory that leads to one. It also cannot create, rename, or delete
//!   an entry directly in such a directory, so a file written through a
//!   temp file and rename there fails.
//! - `off`: no writable root covers a target.
//!
//! The root `.git` is always a target, so in a workspace without one the
//! root gets no grant and a command cannot create it. Both modes refuse a
//! writable root that contains the workspace from outside it: the command
//! could swap the workspace for a copy with its own `.git`. A writable
//! root that overlaps a target in another way is refused, as is a git dir
//! or common dir the sandbox can write. The scan runs once, when the
//! instance is created; a `.git` a command creates later is not covered
//! (see `nested_git`).

use anyhow::{Context, Result};
use std::ffi::{CStr, CString};
use std::os::fd::RawFd;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// How an instance keeps workspace git metadata read-only.
#[derive(Debug)]
pub(super) enum GitGuard {
    Off,
    Mount(Arc<MountPlan>),
    /// Targets relative to the workspace, left out of the writable grants.
    Grants(Vec<PathBuf>),
}

impl GitGuard {
    /// Mode recorded in the trace when the instance is created.
    pub(super) const fn mode(&self) -> crate::paths::GitGuardMode {
        use crate::paths::GitGuardMode;
        match self {
            Self::Off => GitGuardMode::Off,
            Self::Mount(_) => GitGuardMode::MountNamespace,
            Self::Grants(_) => GitGuardMode::TopLevelGrants,
        }
    }

    /// Targets the workspace grant must leave out, in `top_level_grants`.
    pub(super) fn excluded(&self) -> Option<&[PathBuf]> {
        match self {
            Self::Grants(targets) => Some(targets),
            Self::Off | Self::Mount(_) => None,
        }
    }

    pub(super) fn mount(&self) -> Option<Arc<MountPlan>> {
        match self {
            Self::Mount(plan) => Some(Arc::clone(plan)),
            Self::Off | Self::Grants(_) => None,
        }
    }
}

/// What the forked child needs to cover `.git`, prepared in the parent.
#[derive(Debug)]
pub(super) struct MountPlan {
    git: CString,
    dev: libc::dev_t,
    ino: libc::ino_t,
    uid_map: Vec<u8>,
    gid_map: Vec<u8>,
}

/// One exec's plan plus its absolute working directory.
#[derive(Debug)]
pub(super) struct MountExec {
    plan: Arc<MountPlan>,
    cwd: CString,
}

impl MountExec {
    pub(super) fn new(plan: Arc<MountPlan>, cwd: &Path) -> Result<Self> {
        let cwd = if cwd.is_absolute() {
            cwd.to_path_buf()
        } else {
            std::env::current_dir()
                .context("read the current directory")?
                .join(cwd)
        };
        Ok(Self {
            plan,
            cwd: c_path(&cwd)?,
        })
    }
}

/// Pick the mode for `workspace` given the instance's writable roots.
///
/// Fails closed when the root `.git` is a symlink, when the targets cannot
/// be read or the scan for them hits a bound, when a writable root inside the workspace overlaps a target,
/// when a git dir or common dir is writable, and when a writable root
/// reaches the workspace and either contains it from outside or sits below
/// the root of an enclosing repository.
pub(super) fn choose(workspace: &Path, writable: &[PathBuf]) -> Result<GitGuard> {
    let root = workspace
        .canonicalize()
        .with_context(|| format!("canonicalize {}", workspace.display()))?;
    let git = root.join(crate::paths::GIT_DIR_NAME);
    let git_is_dir = match std::fs::symlink_metadata(&git) {
        Ok(meta) if meta.is_dir() => true,
        Ok(_) => false,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => false,
        Err(err) => return Err(err).with_context(|| format!("stat {}", git.display())),
    };
    let writable: Vec<PathBuf> = writable
        .iter()
        .filter_map(|path| path.canonicalize().ok())
        .collect();
    let targets = crate::nested_git::guarded_targets(&root, &writable)?;
    let reaches = refuse_unguarded(&root, &writable, &targets)?;
    crate::nested_git::refuse_git_outside(&root, &writable)?;
    if !reaches {
        return Ok(GitGuard::Off);
    }
    let covered = targets.iter().any(|target| {
        writable
            .iter()
            .any(|path| root.join(target).starts_with(path))
    });
    if !covered {
        return Ok(GitGuard::Off);
    }
    if git_is_dir && crate::git_targets::only_root(&targets) {
        let plan = Arc::new(MountPlan::new(&git)?);
        if probe(Arc::clone(&plan), &root)? {
            return Ok(GitGuard::Mount(plan));
        }
    }
    Ok(GitGuard::Grants(targets))
}

/// Refuse what neither mode can keep read-only. Returns whether any
/// writable root reaches the workspace; without one the guard is off.
fn refuse_unguarded(root: &Path, writable: &[PathBuf], targets: &[PathBuf]) -> Result<bool> {
    use crate::paths::git_refusal;
    refuse_overlapping_roots(root, writable, targets)?;
    let class = crate::git_targets::refuse_exposed_git_dirs(root, targets, writable)?;
    let reaches = writable
        .iter()
        .any(|path| path.starts_with(root) || root.starts_with(path));
    if !reaches {
        return Ok(false);
    }
    if let caliche_policy::WorkspaceGit::Enclosing { dir, .. } = class {
        anyhow::bail!(git_refusal::enclosing_repository(root, &dir));
    }
    if writable
        .iter()
        .any(|path| root.starts_with(path) && path != root)
    {
        anyhow::bail!(git_refusal::WRITABLE_ROOT_CONTAINS_WORKSPACE);
    }
    Ok(true)
}

/// A writable root strictly inside the workspace gets its own grant, so it
/// must neither lead to a target nor sit inside one.
fn refuse_overlapping_roots(root: &Path, writable: &[PathBuf], targets: &[PathBuf]) -> Result<()> {
    for path in writable {
        if path == root || !path.starts_with(root) {
            continue;
        }
        for target in targets {
            let target = root.join(target);
            if target.starts_with(path) || path.starts_with(&target) {
                anyhow::bail!(crate::paths::git_refusal::overlapping_root(path, &target));
            }
        }
    }
    Ok(())
}

impl MountPlan {
    fn new(git: &Path) -> Result<Self> {
        let path = c_path(git)?;
        let fd = open_git(&path).with_context(|| format!("open {}", git.display()))?;
        let stat = stat_fd(fd);
        close_fd(fd);
        let stat = stat.with_context(|| format!("stat {}", git.display()))?;
        #[expect(unsafe_code, reason = "geteuid and getegid cannot fail")]
        let (uid, gid) = unsafe { (libc::geteuid(), libc::getegid()) };
        Ok(Self {
            git: path,
            dev: stat.st_dev,
            ino: stat.st_ino,
            uid_map: format!("{uid} {uid} 1").into_bytes(),
            gid_map: format!("{gid} {gid} 1").into_bytes(),
        })
    }
}

fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_bytes())
        .with_context(|| format!("{} holds a NUL byte", path.display()))
}

/// Run the child's steps in a forked process that exits right after.
fn probe(plan: Arc<MountPlan>, workspace: &Path) -> Result<bool> {
    let exec = MountExec {
        plan,
        cwd: c_path(workspace)?,
    };
    // SAFETY: the child runs only the async-signal-safe `enter` and `_exit`,
    // so a lock held by another thread at `fork` cannot hang it.
    #[expect(unsafe_code, reason = "fork a probe child that only runs enter")]
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        let code = i32::from(enter(&exec).is_err());
        // SAFETY: `_exit` skips atexit handlers the parent owns.
        #[expect(unsafe_code, reason = "leave the probe child")]
        unsafe {
            libc::_exit(code)
        };
    }
    if pid < 0 {
        return Err(std::io::Error::last_os_error()).context("fork the user namespace probe");
    }
    let mut status = 0;
    loop {
        // SAFETY: `pid` is this process's child and `status` is local.
        #[expect(unsafe_code, reason = "reap the probe child")]
        let reaped = unsafe { libc::waitpid(pid, &raw mut status, 0) };
        if reaped == pid {
            return Ok(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() != Some(libc::EINTR) {
            return Err(err).context("wait for the user namespace probe");
        }
    }
}

// BEGIN SIGNAL_SAFE_GIT_GUARD
const MOVE_MOUNT_F_EMPTY_PATH: libc::c_uint = 0x4;
const MOVE_MOUNT_T_EMPTY_PATH: libc::c_uint = 0x40;
/// `MOUNT_ATTR_SIZE_VER0`, the size of `struct mount_attr`.
const MOUNT_ATTR_SIZE: usize = 32;

/// Enter a new user and mount namespace, put a read-only clone of `.git`
/// over it, and change into the working directory again so a directory
/// inside `.git` resolves through the new mount.
pub(super) fn enter(mount: &MountExec) -> std::io::Result<()> {
    // SAFETY: `unshare` is a plain syscall; this process is single-threaded
    // after `fork`.
    #[expect(unsafe_code, reason = "async-signal-safe unshare")]
    let rc = unsafe { libc::unshare(libc::CLONE_NEWUSER | libc::CLONE_NEWNS) };
    check(rc)?;
    write_proc(c"/proc/self/setgroups", b"deny")?;
    write_proc(c"/proc/self/uid_map", &mount.plan.uid_map)?;
    write_proc(c"/proc/self/gid_map", &mount.plan.gid_map)?;
    // SAFETY: null source, type, and data are valid for a propagation change.
    #[expect(unsafe_code, reason = "async-signal-safe mount")]
    let rc = unsafe {
        libc::mount(
            std::ptr::null(),
            c"/".as_ptr(),
            std::ptr::null(),
            libc::MS_REC | libc::MS_PRIVATE,
            std::ptr::null(),
        )
    };
    check(rc)?;
    let git = open_git(&mount.plan.git)?;
    let covered = cover_read_only(git, &mount.plan);
    close_fd(git);
    covered?;
    // SAFETY: `cwd` is a NUL-terminated path the parent keeps alive.
    #[expect(unsafe_code, reason = "async-signal-safe chdir")]
    let rc = unsafe { libc::chdir(mount.cwd.as_ptr()) };
    check(rc)
}

fn cover_read_only(git: RawFd, plan: &MountPlan) -> std::io::Result<()> {
    let stat = stat_fd(git)?;
    if stat.st_dev != plan.dev || stat.st_ino != plan.ino {
        return Err(std::io::Error::from_raw_os_error(libc::ESTALE));
    }
    let tree = clone_tree(git)?;
    let covered = read_only_onto(tree, git);
    close_fd(tree);
    covered
}

fn clone_tree(git: RawFd) -> std::io::Result<RawFd> {
    let flags = libc::OPEN_TREE_CLONE
        | libc::OPEN_TREE_CLOEXEC
        | libc::AT_EMPTY_PATH as libc::c_uint
        | libc::AT_RECURSIVE as libc::c_uint;
    // SAFETY: `open_tree` on an fd with an empty path; `syscall` is
    // async-signal-safe.
    #[expect(unsafe_code, reason = "async-signal-safe open_tree")]
    let tree = unsafe { libc::syscall(libc::SYS_open_tree, git, c"".as_ptr(), flags) };
    if tree < 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    Ok(tree as RawFd)
}

fn read_only_onto(tree: RawFd, git: RawFd) -> std::io::Result<()> {
    let attr = libc::mount_attr {
        attr_set: libc::MOUNT_ATTR_RDONLY,
        attr_clr: 0,
        propagation: 0,
        userns_fd: 0,
    };
    let flags = libc::AT_EMPTY_PATH as libc::c_uint | libc::AT_RECURSIVE as libc::c_uint;
    // SAFETY: `attr` is a local `mount_attr` of the size passed.
    #[expect(unsafe_code, reason = "async-signal-safe mount_setattr")]
    let rc = unsafe {
        libc::syscall(
            libc::SYS_mount_setattr,
            tree,
            c"".as_ptr(),
            flags,
            std::ptr::from_ref(&attr),
            MOUNT_ATTR_SIZE,
        )
    };
    if rc < 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    // SAFETY: both fds are open; empty paths name the fds themselves.
    #[expect(unsafe_code, reason = "async-signal-safe move_mount")]
    let rc = unsafe {
        libc::syscall(
            libc::SYS_move_mount,
            tree,
            c"".as_ptr(),
            git,
            c"".as_ptr(),
            MOVE_MOUNT_F_EMPTY_PATH | MOVE_MOUNT_T_EMPTY_PATH,
        )
    };
    if rc < 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    Ok(())
}

fn stat_fd(fd: RawFd) -> std::io::Result<libc::stat> {
    // SAFETY: an all-zero `stat` is a valid value for `fstat` to fill.
    #[expect(unsafe_code, reason = "zeroed plain-data struct")]
    let mut stat: libc::stat = unsafe { std::mem::zeroed() };
    // SAFETY: `fstat` fills the local struct.
    #[expect(unsafe_code, reason = "async-signal-safe fstat")]
    let rc = unsafe { libc::fstat(fd, &raw mut stat) };
    check(rc)?;
    Ok(stat)
}

fn open_git(path: &CStr) -> std::io::Result<RawFd> {
    // SAFETY: `path` is NUL-terminated; `O_NOFOLLOW` refuses a symlink swap.
    #[expect(unsafe_code, reason = "async-signal-safe open")]
    let fd = unsafe {
        libc::open(
            path.as_ptr(),
            libc::O_PATH | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    Ok(fd)
}

fn write_proc(path: &CStr, text: &[u8]) -> std::io::Result<()> {
    // SAFETY: `path` is NUL-terminated.
    #[expect(unsafe_code, reason = "async-signal-safe open")]
    let fd = unsafe { libc::open(path.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC) };
    if fd < 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    // SAFETY: `text` is a live slice of the length passed.
    #[expect(unsafe_code, reason = "async-signal-safe write")]
    let written = unsafe { libc::write(fd, text.as_ptr().cast(), text.len()) };
    // A short write sets no errno, so it reports EIO.
    let err = if written < 0 {
        Some(super::cgroup::raw_errno())
    } else if written.unsigned_abs() != text.len() {
        Some(libc::EIO)
    } else {
        None
    };
    close_fd(fd);
    match err {
        Some(errno) => Err(std::io::Error::from_raw_os_error(errno)),
        None => Ok(()),
    }
}

fn close_fd(fd: RawFd) {
    // SAFETY: `fd` was opened by this module and is closed once.
    #[expect(unsafe_code, reason = "async-signal-safe close")]
    let _closed = unsafe { libc::close(fd) };
}

fn check(rc: libc::c_int) -> std::io::Result<()> {
    if rc < 0 {
        return Err(std::io::Error::from_raw_os_error(super::cgroup::raw_errno()));
    }
    Ok(())
}
// END SIGNAL_SAFE_GIT_GUARD

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_with_git() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path().canonicalize().expect("canonical");
        crate::git_targets::tests::real_git_dir(&root.join(crate::paths::GIT_DIR_NAME));
        (dir, root)
    }

    /// `.git` swapped for another directory after the plan was made is
    /// refused, not covered.
    #[test]
    fn a_git_dir_swapped_after_planning_is_refused() {
        let (_dir, root) = workspace_with_git();
        let git = root.join(crate::paths::GIT_DIR_NAME);
        let plan = MountPlan::new(&git).expect("plan");
        std::fs::rename(&git, root.join("old-git")).expect("move .git away");
        std::fs::create_dir(&git).expect("new .git");

        let fd = open_git(&c_path(&git).expect("c path")).expect("open new .git");
        let covered = cover_read_only(fd, &plan);
        close_fd(fd);

        assert_eq!(
            covered.expect_err("swapped .git").raw_os_error(),
            Some(libc::ESTALE)
        );
    }

    #[test]
    fn a_symlinked_git_dir_is_refused() {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path().canonicalize().expect("canonical");
        std::fs::create_dir(root.join("real-git")).expect("real git");
        std::os::unix::fs::symlink("real-git", root.join(crate::paths::GIT_DIR_NAME))
            .expect("symlink .git");

        let err = choose(&root, std::slice::from_ref(&root)).expect_err("refused");

        assert_eq!(
            err.to_string(),
            crate::paths::git_refusal::invalid_git_entry(&root.join(crate::paths::GIT_DIR_NAME))
        );
    }

    /// A workspace whose index records a submodule at `vendored`, checked
    /// out with its own `.git` dir.
    fn workspace_with_nested_git() -> (tempfile::TempDir, PathBuf) {
        let (dir, root) = workspace_with_git();
        std::fs::create_dir_all(root.join("vendored").join(crate::paths::GIT_DIR_NAME))
            .expect("submodule .git");
        std::fs::write(
            root.join(crate::paths::GIT_DIR_NAME)
                .join(crate::paths::INDEX_FILE),
            crate::git_targets::tests::gitlink_index("vendored"),
        )
        .expect("index");
        (dir, root)
    }

    /// Without a root `.git`, the guard still leaves the root `.git` path
    /// out, so the root gets no grant.
    #[test]
    fn a_workspace_without_git_still_excludes_the_root_git() {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path().canonicalize().expect("canonical");

        let guard = choose(&root, std::slice::from_ref(&root)).expect("guard");

        assert_eq!(guard.mode(), crate::paths::GitGuardMode::TopLevelGrants);
        assert_eq!(
            guard.excluded(),
            Some([PathBuf::from(crate::paths::GIT_DIR_NAME)].as_slice())
        );
    }

    /// Refused in both modes: with only the root `.git` (mount mode) and
    /// with a nested one (grants mode).
    #[test]
    fn a_writable_root_around_the_workspace_is_refused() {
        for (_dir, root) in [workspace_with_git(), workspace_with_nested_git()] {
            let parent = root.parent().expect("parent").to_path_buf();

            let err = choose(&root, &[parent]).expect_err("refused");

            assert_eq!(
                err.to_string(),
                crate::paths::git_refusal::WRITABLE_ROOT_CONTAINS_WORKSPACE
            );
        }
    }

    /// A writable workspace below the root of a repository is refused, even
    /// with the repository read-only: its submodule under the workspace,
    /// not checked out, is in the enclosing index only, which the guard
    /// does not read.
    #[test]
    fn a_workspace_inside_an_enclosing_repository_is_refused() {
        let dir = tempfile::tempdir().expect("base");
        let repo = dir.path().canonicalize().expect("canonical");
        let git = repo.join(crate::paths::GIT_DIR_NAME);
        crate::git_targets::tests::real_git_dir(&git);
        std::fs::write(
            git.join(crate::paths::INDEX_FILE),
            crate::git_targets::tests::gitlink_index("ws/lib"),
        )
        .expect("index");
        let ws = repo.join("ws");
        std::fs::create_dir_all(ws.join("lib")).expect("submodule dir");

        let err = choose(&ws, std::slice::from_ref(&ws)).expect_err("refused");

        assert_eq!(
            err.to_string(),
            "{ws} is inside the git repository at {repo}; on Linux a writable workspace must be the repository root, so open {repo} instead"
                .replace("{ws}", &ws.display().to_string())
                .replace("{repo}", &repo.display().to_string())
        );
    }

    /// A nested repository that is not a submodule is found by the scan and
    /// left out of the grants with the root `.git`.
    #[test]
    fn a_nested_repository_that_is_not_a_submodule_is_a_target() {
        let (_dir, root) = workspace_with_git();
        crate::git_targets::tests::real_git_dir(
            &root.join("vendor").join(crate::paths::GIT_DIR_NAME),
        );

        let guard = choose(&root, std::slice::from_ref(&root)).expect("guard");

        assert_eq!(guard.mode(), crate::paths::GitGuardMode::TopLevelGrants);
        assert_eq!(
            guard.excluded(),
            Some(
                [
                    PathBuf::from(crate::paths::GIT_DIR_NAME),
                    Path::new("vendor").join(crate::paths::GIT_DIR_NAME),
                ]
                .as_slice()
            )
        );
    }

    /// The scan cannot rule out a `.git` in a directory nobody can list.
    #[test]
    fn a_directory_nobody_can_list_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let (_dir, root) = workspace_with_git();
        let locked = root.join("locked");
        std::fs::create_dir(&locked).expect("locked");
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000))
            .expect("chmod 000");
        std::fs::read_dir(&locked).expect_err("mode 000 is unlistable");

        let guard = choose(&root, std::slice::from_ref(&root)).map_err(|err| err.to_string());

        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755))
            .expect("chmod back");
        assert_eq!(
            guard.map(|guard| guard.mode()),
            Err(crate::paths::git_refusal::unlistable_dir(&locked))
        );
    }

    /// A submodule's own submodule is a target, read from the indexes, so
    /// a writable root at its path overlaps it and is refused.
    #[test]
    fn a_writable_root_at_a_nested_submodule_is_refused() {
        let dir = crate::git_targets::tests::nested_submodules();
        let root = dir.path().join("top").canonicalize().expect("canonical");
        let inner = root.join("sub").join("inner");

        let err = choose(&root, std::slice::from_ref(&inner)).expect_err("refused");

        assert_eq!(
            err.to_string(),
            crate::paths::git_refusal::overlapping_root(
                &inner,
                &inner.join(crate::paths::GIT_DIR_NAME)
            )
        );
    }

    /// `git init --separate-git-dir <ws>/meta <ws>`.
    #[test]
    fn a_git_dir_inside_the_workspace_outside_git_is_refused() {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path().canonicalize().expect("canonical");
        std::fs::create_dir(root.join("meta")).expect("meta");
        crate::git_targets::tests::real_git_dir(&root.join("meta"));
        std::fs::write(
            root.join(crate::paths::GIT_DIR_NAME),
            format!("{} meta\n", crate::paths::GITFILE_PREFIX),
        )
        .expect("gitfile");

        let err = choose(&root, std::slice::from_ref(&root)).expect_err("refused");

        assert_eq!(
            err.to_string(),
            crate::paths::git_refusal::git_dir_in_workspace(&root.join("meta"))
        );
    }

    /// A writable root inside the workspace that leads to a target, or sits
    /// inside one, is refused.
    #[test]
    fn a_writable_root_overlapping_git_metadata_is_refused() {
        let (_dir, root) = workspace_with_nested_git();
        let inside = root.join(crate::paths::GIT_DIR_NAME).join("hooks");
        std::fs::create_dir(&inside).expect("hooks");
        let leads_to = root.join("vendored");

        let leads_err = choose(&root, &[root.clone(), leads_to.clone()]).expect_err("leads");
        let inside_err = choose(&root, &[root.clone(), inside.clone()]).expect_err("inside");

        assert_eq!(
            leads_err.to_string(),
            crate::paths::git_refusal::overlapping_root(
                &leads_to,
                &root.join("vendored").join(crate::paths::GIT_DIR_NAME)
            )
        );
        assert_eq!(
            inside_err.to_string(),
            crate::paths::git_refusal::overlapping_root(
                &inside,
                &root.join(crate::paths::GIT_DIR_NAME)
            )
        );
    }

    /// A linked worktree whose main repository is under a writable root.
    #[test]
    fn a_worktree_git_dir_under_a_writable_root_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().canonicalize().expect("canonical");
        let root = base.join("ws");
        let main = base.join("main");
        std::fs::create_dir_all(&root).expect("ws");
        let git_dir =
            caliche_policy::plant_linked_worktree(&main.join(crate::paths::GIT_DIR_NAME), "ws")
                .expect("linked worktree");
        std::fs::write(
            root.join(crate::paths::GIT_DIR_NAME),
            format!("{} {}\n", crate::paths::GITFILE_PREFIX, git_dir.display()),
        )
        .expect("gitfile");

        let err = choose(&root, &[root.clone(), main.clone()]).expect_err("refused");

        assert_eq!(
            err.to_string(),
            crate::paths::git_refusal::git_dir_under_writable_root(&git_dir, &main)
        );
    }

    /// A writable root inside a linked worktree's common dir (here the
    /// main repository's hooks) is refused.
    #[test]
    fn a_writable_root_inside_the_common_dir_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().canonicalize().expect("canonical");
        let root = base.join("ws");
        let common = base.join("main").join(crate::paths::GIT_DIR_NAME);
        let hooks = common.join("hooks");
        std::fs::create_dir_all(&root).expect("ws");
        let git_dir =
            caliche_policy::plant_linked_worktree(&common, "ws").expect("linked worktree");
        std::fs::create_dir_all(&hooks).expect("hooks");
        std::fs::write(
            root.join(crate::paths::GIT_DIR_NAME),
            format!("{} {}\n", crate::paths::GITFILE_PREFIX, git_dir.display()),
        )
        .expect("gitfile");

        let err = choose(&root, &[root.clone(), hooks.clone()]).expect_err("refused");

        assert_eq!(
            err.to_string(),
            crate::paths::git_refusal::overlapping_root(&hooks, &common)
        );
    }
}
