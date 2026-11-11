//! The user's `git`, as `/commit`, `/pr` and `/review` run it.
//!
//! A sandboxed command may have written work-tree files that this `git`
//! reads, and a backend may have a gap in how it keeps git metadata
//! read-only. Two rules narrow what such a gap lets a command reach:
//!
//! - `core.fsmonitor` is off for this `git` and every `git` it starts
//!   (`-c` settings reach the `git` run in each submodule).
//! - Discovery never uses a bare repository (`safe.bareRepository` is
//!   `explicit`), and a directory that is inside a git dir but not a work
//!   tree is refused, so a repository planted in the workspace or a parent
//!   is not read.
//! - The `.git` of each submodule in the root index must be a gitfile
//!   naming a git dir inside the superproject's `modules` (in its git dir,
//!   which for a linked worktree is the worktree's own). Anything else is
//!   refused before any `git` reads the submodule.
//! - A work tree whose `.git` is a symlink, or is not a gitfile or git dir
//!   discovery accepts, is refused. Git would follow the symlink. The error
//!   names the entry. `rev-parse` has already run with `core.fsmonitor` off
//!   to learn the work tree.
//!
//! Neither rule turns off filters, hooks, or other programs named in a
//! submodule's config; keeping that config read-only is the sandbox's job.

use anyhow::{Context, Result};
use caliche::policy::{
    Discovery, FSMONITOR_KEY, GIT_DIR_NAME, GITLINK_MODE, MODULES_DIR, SAFE_BARE_REPOSITORY_KEY,
    discover, read_gitfile,
};
use caliche_sandbox::paths::git_refusal;
use std::path::{Path, PathBuf};

/// `git` run in one repository with the rules above.
pub(crate) struct HostGit {
    dir: PathBuf,
}

impl HostGit {
    /// Check the repository at `dir` and return a runner for it. A `dir`
    /// outside any repository is accepted; its `git` commands fail as usual.
    /// A `dir` git finds a git dir for but no work tree (a bare repository,
    /// or inside a git dir) is refused, and so is a work tree whose git dir
    /// is not its own `.git` (a `core.worktree` in a repository elsewhere).
    pub(crate) async fn open(dir: &Path) -> Result<Self> {
        let git = Self {
            dir: dir.to_path_buf(),
        };
        let Some(top) = git.text(&["rev-parse", "--show-toplevel"]).await? else {
            if let Some(found) = git.bare_git_dir().await? {
                anyhow::bail!(no_work_tree(dir, &found));
            }
            return Ok(git);
        };
        let git_dir = git
            .text(&["rev-parse", "--path-format=absolute", "--git-dir"])
            .await?
            .context("read the repository git dir")?;
        let top = PathBuf::from(top);
        let own = match discover(&top) {
            Discovery::Repository { dir, git_dir } if dir == top => git_dir.canonicalize().ok(),
            Discovery::InvalidGitEntry(path) => {
                anyhow::bail!(git_refusal::invalid_git_entry(&path))
            }
            Discovery::StrayHead(path) => anyhow::bail!(git_refusal::stray_head(&path)),
            Discovery::Repository { .. } | Discovery::NoRepository => None,
        };
        if own.is_none() || own != Path::new(&git_dir).canonicalize().ok() {
            anyhow::bail!(foreign_git_dir(dir, &git_dir, &top));
        }
        let modules = PathBuf::from(git_dir).join(MODULES_DIR);
        let staged = git
            .command()
            .args(["ls-files", "--stage", "-z"])
            .current_dir(&top)
            .output()
            .await
            .context("list submodules")?;
        if !staged.status.success() {
            anyhow::bail!(
                "list submodules: {}",
                String::from_utf8_lossy(&staged.stderr).trim()
            );
        }
        for entry in staged.stdout.split(|&byte| byte == 0) {
            let entry = String::from_utf8_lossy(entry);
            let Some((meta, path)) = entry.split_once('\t') else {
                continue;
            };
            if meta.split(' ').next() == Some(format!("{GITLINK_MODE:o}").as_str()) {
                check_submodule(&top, &modules, path)?;
            }
        }
        Ok(git)
    }

    /// `git -c core.fsmonitor=false -c safe.bareRepository=explicit` in the
    /// repository directory.
    pub(crate) fn command(&self) -> tokio::process::Command {
        let mut command = tokio::process::Command::new("git");
        command
            .args(["-c", &format!("{FSMONITOR_KEY}=false")])
            .args(["-c", &format!("{SAFE_BARE_REPOSITORY_KEY}=explicit")])
            .current_dir(&self.dir);
        command
    }

    /// The git dir discovery finds from the directory when bare
    /// repositories are allowed, or `None`. `rev-parse` runs no program
    /// named in config, so allowing them for this one call is safe.
    async fn bare_git_dir(&self) -> Result<Option<String>> {
        self.text(&[
            "-c",
            &format!("{SAFE_BARE_REPOSITORY_KEY}=all"),
            "rev-parse",
            "--absolute-git-dir",
        ])
        .await
    }

    /// Trimmed stdout of a successful command, or `None` when it fails.
    async fn text(&self, args: &[&str]) -> Result<Option<String>> {
        let output = self
            .command()
            .args(args)
            .output()
            .await
            .context("run git")?;
        Ok(output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string()))
    }
}

/// Refuse a submodule whose `.git` is anything but a gitfile into
/// `modules`. A submodule that is not checked out has no `.git` and passes.
fn check_submodule(top: &Path, modules: &Path, path: &str) -> Result<()> {
    let work_tree = top.join(path);
    let dot_git = work_tree.join(GIT_DIR_NAME);
    let meta = match std::fs::symlink_metadata(&dot_git) {
        Ok(meta) => meta,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(err) => return Err(err).with_context(|| format!("stat {}", dot_git.display())),
    };
    if !meta.is_file() {
        anyhow::bail!(not_a_gitfile(path, &dot_git, modules));
    }
    let named = read_gitfile(&dot_git)
        .with_context(|| format!("read {}", dot_git.display()))?
        .unwrap_or_default();
    let target = named.canonicalize().ok();
    let inside = modules.canonicalize().ok();
    match (target, inside) {
        (Some(target), Some(inside)) if target.starts_with(&inside) && target != inside => Ok(()),
        _ => anyhow::bail!(outside_modules(path, &dot_git, &named, modules)),
    }
}

fn foreign_git_dir(dir: &Path, git_dir: &str, top: &Path) -> String {
    format!(
        "{} uses git dir {git_dir}, which is not the .git of work tree {}; refusing to run git",
        dir.display(),
        top.display()
    )
}

fn no_work_tree(dir: &Path, git_dir: &str) -> String {
    format!(
        "{} is in git dir {git_dir} with no work tree; refusing to run git",
        dir.display()
    )
}

fn not_a_gitfile(path: &str, dot_git: &Path, modules: &Path) -> String {
    format!(
        "submodule {path}: {} is not a gitfile; only a gitfile into {} is accepted (run `git submodule absorbgitdirs`); refusing to run git",
        dot_git.display(),
        modules.display()
    )
}

fn outside_modules(path: &str, dot_git: &Path, named: &Path, modules: &Path) -> String {
    format!(
        "submodule {path}: {} points to {named:?}, outside {}; refusing to run git",
        dot_git.display(),
        modules.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliche::policy::{CONFIG_FILE, GITFILE_PREFIX, clear_git_config};
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) {
        let mut command = Command::new("git");
        let status = clear_git_config(&mut command)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args([
                "-c",
                "init.defaultBranch=main",
                "-c",
                "protocol.file.allow=always",
            ])
            .args(args)
            .current_dir(dir)
            .status()
            .expect("git");
        assert!(status.success(), "git {args:?}");
    }

    /// Trimmed stdout of `git args` in `dir`: a path as git reports it
    /// (`C:/...` on Windows).
    fn git_reports(dir: &Path, args: &[&str]) -> String {
        let mut command = Command::new("git");
        let output = clear_git_config(&mut command)
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git");
        assert!(output.status.success(), "git {args:?}: {output:?}");
        String::from_utf8(output.stdout)
            .expect("utf-8")
            .trim()
            .to_string()
    }

    /// The work tree and git dir git reports for `dir`.
    fn reported_top_and_git_dir(dir: &Path) -> (PathBuf, PathBuf) {
        (
            PathBuf::from(git_reports(dir, &["rev-parse", "--show-toplevel"])),
            PathBuf::from(git_reports(
                dir,
                &["rev-parse", "--path-format=absolute", "--git-dir"],
            )),
        )
    }

    /// The canonical temp dir in the form git accepts (not `\\?\` on
    /// Windows).
    fn temp_root(dir: &tempfile::TempDir) -> PathBuf {
        caliche_sandbox::external_tool_path(&dir.path().canonicalize().expect("canonical"))
    }

    /// `path` for a `sh` command line: forward slashes, single-quoted.
    fn sh_path(path: &Path) -> String {
        format!("'{}'", path.to_string_lossy().replace('\\', "/"))
    }

    /// Copy the directory tree `from` to `to`.
    fn copy_dir(from: &Path, to: &Path) {
        std::fs::create_dir(to).expect("copy dir");
        for entry in std::fs::read_dir(from).expect("read dir") {
            let entry = entry.expect("entry");
            let target = to.join(entry.file_name());
            if entry.file_type().expect("file type").is_dir() {
                copy_dir(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), &target).expect("copy file");
            }
        }
    }

    /// A superproject `top` with a submodule at `sub`, and a marker path a
    /// planted fsmonitor would create.
    fn superproject() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = temp_root(&dir);
        std::fs::create_dir(root.join("lib")).expect("lib");
        git(&root.join("lib"), &["init", "-q"]);
        git(
            &root.join("lib"),
            &["commit", "-q", "--allow-empty", "-m", "init"],
        );
        let top = root.join("top");
        std::fs::create_dir(&top).expect("top");
        git(&top, &["init", "-q"]);
        git(&top, &["submodule", "add", "-q", "../lib", "sub"]);
        git(&top, &["commit", "-q", "-m", "add"]);
        std::fs::write(top.join("sub/new.txt"), "dirty").expect("dirty submodule");
        let marker = root.join("PWNED");
        (dir, top, marker)
    }

    /// Set `core.fsmonitor` in `git_dir/config`, run from outside any
    /// repository so `git` reads no other config.
    fn plant_fsmonitor(git_dir: &Path, marker: &Path) {
        let config = git_dir.join(CONFIG_FILE);
        git(
            marker.parent().expect("marker dir"),
            &[
                "config",
                "--file",
                config.to_str().expect("utf-8"),
                FSMONITOR_KEY,
                &format!("touch {}; false", sh_path(marker)),
            ],
        );
    }

    /// What a sandboxed command can do: copy the submodule git dir into
    /// the work tree, plant fsmonitor there, and point `sub/.git` at it.
    fn plant_gitfile_outside_modules(top: &Path, marker: &Path) {
        let evil = top.join("sub/.evil");
        copy_dir(&top.join(GIT_DIR_NAME).join(MODULES_DIR).join("sub"), &evil);
        let config = evil.join(CONFIG_FILE);
        git(
            marker.parent().expect("marker dir"),
            &[
                "config",
                "--file",
                config.to_str().expect("utf-8"),
                "--unset",
                "core.worktree",
            ],
        );
        plant_fsmonitor(&evil, marker);
        std::fs::write(
            top.join("sub").join(GIT_DIR_NAME),
            format!("{GITFILE_PREFIX} .evil\n"),
        )
        .expect("gitfile");
    }

    async fn status(top: &Path) -> Result<String> {
        let mut out = Vec::new();
        let git = HostGit::open(top).await?;
        super::super::git::write_empty_staged_status(&mut out, &git).await?;
        Ok(String::from_utf8(out).expect("utf-8"))
    }

    #[tokio::test]
    async fn commit_refuses_a_submodule_gitfile_outside_git_modules() {
        let (_dir, top, marker) = superproject();
        let (git_top, git_dir) = reported_top_and_git_dir(&top);
        plant_gitfile_outside_modules(&top, &marker);

        let err = status(&top).await.expect_err("foreign gitdir");

        assert_eq!(
            err.to_string(),
            outside_modules(
                "sub",
                &git_top.join("sub").join(GIT_DIR_NAME),
                &git_top.join("sub").join(".evil"),
                &git_dir.join(MODULES_DIR)
            )
        );
        assert!(!marker.exists(), "planted fsmonitor ran");
    }

    #[tokio::test]
    async fn commit_does_not_run_a_planted_fsmonitor() {
        let (_dir, top, marker) = superproject();
        plant_fsmonitor(
            &top.join(GIT_DIR_NAME).join(MODULES_DIR).join("sub"),
            &marker,
        );

        let out = status(&top).await.expect("status");

        assert_eq!(
            out,
            "No staged changes. Stage your changes first with 'git add'.\n\nUnstaged changes:\n ? sub\n\n"
        );
        assert!(!marker.exists(), "planted fsmonitor ran");
    }

    #[tokio::test]
    async fn a_submodule_git_dir_in_the_work_tree_is_refused() {
        let (_dir, top, _marker) = superproject();
        let (git_top, git_dir) = reported_top_and_git_dir(&top);
        std::fs::remove_file(top.join("sub").join(GIT_DIR_NAME)).expect("gitfile");
        std::fs::create_dir(top.join("sub").join(GIT_DIR_NAME)).expect("git dir");

        let err = HostGit::open(&top).await.err().expect("refused");

        assert_eq!(
            err.to_string(),
            not_a_gitfile(
                "sub",
                &git_top.join("sub").join(GIT_DIR_NAME),
                &git_dir.join(MODULES_DIR)
            ),
        );
    }

    /// A bare repository with two commits that change `f`, cloned from a
    /// work repository under `root`, with `diff.external` set to a script
    /// that creates the returned marker.
    fn planted_bare_repository(root: &Path, at: &Path) -> PathBuf {
        let work = root.join("work");
        std::fs::create_dir(&work).expect("work");
        git(&work, &["init", "-q"]);
        for text in ["one", "two"] {
            std::fs::write(work.join("f"), text).expect("f");
            git(&work, &["add", "f"]);
            git(&work, &["commit", "-q", "-m", text]);
        }
        git(
            root,
            &[
                "clone",
                "-q",
                "--bare",
                work.to_str().expect("utf-8"),
                at.to_str().expect("utf-8"),
            ],
        );
        let marker = root.join("PWNED");
        let script = root.join("ext.sh");
        std::fs::write(&script, format!("#!/bin/sh\ntouch {}\n", sh_path(&marker)))
            .expect("script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755))
                .expect("chmod");
        }
        let config = at.join(CONFIG_FILE);
        git(
            root,
            &[
                "config",
                "--file",
                config.to_str().expect("utf-8"),
                "diff.external",
                &sh_path(&script),
            ],
        );
        marker
    }

    /// Set in the child of [`rerun_without_git_config`].
    const ISOLATED: &str = "CALICHE_HOST_GIT_TEST_ISOLATED";

    /// Re-runs `test` in a child test process with git's global and system
    /// config turned off, so a developer's config cannot decide the result.
    /// True in the parent, which then returns.
    fn rerun_without_git_config(test: &str) -> bool {
        if std::env::var_os(ISOLATED).is_some() {
            return false;
        }
        let mut command = Command::new(std::env::current_exe().expect("test binary"));
        let output = clear_git_config(&mut command)
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env(ISOLATED, "1")
            .output()
            .expect("re-run test");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success() && stdout.contains("1 passed"),
            "{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );
        true
    }

    /// A directory inside a bare repository has a git dir but no work
    /// tree, so it is refused before any `git` runs there.
    #[tokio::test]
    async fn open_refuses_a_directory_with_no_work_tree() {
        if rerun_without_git_config(
            "handlers::host_git::tests::open_refuses_a_directory_with_no_work_tree",
        ) {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let root = temp_root(&dir);
        let bare = root.join("bare");
        planted_bare_repository(&root, &bare);
        let ws = bare.join("ws");
        std::fs::create_dir(&ws).expect("ws");

        let allow_bare = format!("{SAFE_BARE_REPOSITORY_KEY}=all");
        let bare_git_dir =
            git_reports(&ws, &["-c", &allow_bare, "rev-parse", "--absolute-git-dir"]);

        let err = HostGit::open(&ws).await.err().expect("refused");

        assert_eq!(err.to_string(), no_work_tree(&ws, &bare_git_dir));
    }

    /// A bare repository is refused although `explicit` hides it from
    /// discovery: the probe allows bare repositories for that one call.
    #[tokio::test]
    async fn open_refuses_a_bare_repository() {
        if rerun_without_git_config("handlers::host_git::tests::open_refuses_a_bare_repository") {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let root = temp_root(&dir);
        let bare = root.join("bare");
        planted_bare_repository(&root, &bare);

        let allow_bare = format!("{SAFE_BARE_REPOSITORY_KEY}=all");
        let bare_git_dir = git_reports(
            &bare,
            &["-c", &allow_bare, "rev-parse", "--absolute-git-dir"],
        );

        let err = HostGit::open(&bare).await.err().expect("refused");

        assert_eq!(err.to_string(), no_work_tree(&bare, &bare_git_dir));
    }

    /// A repository whose `core.worktree` names another directory has a
    /// work tree, but its git dir is not that work tree's `.git`. Git
    /// before 2.38 ignores `safe.bareRepository`, so `open` refuses it.
    #[tokio::test]
    async fn open_refuses_a_work_tree_whose_git_dir_is_elsewhere() {
        if rerun_without_git_config(
            "handlers::host_git::tests::open_refuses_a_work_tree_whose_git_dir_is_elsewhere",
        ) {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let root = temp_root(&dir);
        let repo = root.join("repo");
        let work = root.join("work");
        std::fs::create_dir(&repo).expect("repo");
        std::fs::create_dir(&work).expect("work");
        git(&repo, &["init", "-q"]);
        git(
            &repo,
            &["config", "core.worktree", work.to_str().expect("utf-8")],
        );
        let (git_top, git_dir) = reported_top_and_git_dir(&repo);

        let err = HostGit::open(&repo).await.err().expect("refused");

        assert_eq!(
            err.to_string(),
            foreign_git_dir(&repo, &git_dir.to_string_lossy(), &git_top)
        );
    }

    /// The `/review` diff from a directory inside a planted bare
    /// repository does not use it, so its `diff.external` never runs.
    #[tokio::test]
    async fn host_git_does_not_discover_a_bare_repository() {
        if rerun_without_git_config(
            "handlers::host_git::tests::host_git_does_not_discover_a_bare_repository",
        ) {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let root = temp_root(&dir);
        let bare = root.join("bare");
        let marker = planted_bare_repository(&root, &bare);
        let ws = bare.join("ws");
        std::fs::create_dir(&ws).expect("ws");
        let git = HostGit { dir: ws };

        let output = git
            .command()
            .args(["diff", "HEAD~1..HEAD"])
            .output()
            .await
            .expect("run git");

        assert!(!output.status.success(), "git used the bare repository");
        assert!(!marker.exists(), "planted diff.external ran");
    }

    /// A linked worktree keeps its submodule git dirs under its own git
    /// dir, `<common>/worktrees/<name>/modules`.
    #[tokio::test]
    async fn a_linked_worktree_with_a_submodule_opens() {
        let (_dir, top, _marker) = superproject();
        let wt = top.parent().expect("root").join("wt");
        git(
            &top,
            &["worktree", "add", "-q", wt.to_str().expect("utf-8")],
        );
        git(&wt, &["submodule", "update", "--init", "-q"]);

        let out = status(&wt).await.expect("status");

        assert_eq!(out, "No changes to commit.\n");
    }

    /// Git follows a symlinked `.git`. This command refuses it, and the
    /// error names that entry rather than a foreign git dir.
    #[cfg(unix)]
    #[tokio::test]
    async fn open_refuses_a_symlinked_git_dir() {
        if rerun_without_git_config("handlers::host_git::tests::open_refuses_a_symlinked_git_dir") {
            return;
        }
        let dir = tempfile::tempdir().expect("tempdir");
        let root = temp_root(&dir);
        git(&root, &["init", "-q"]);
        std::fs::rename(root.join(GIT_DIR_NAME), root.join("real-git")).expect("rename");
        std::os::unix::fs::symlink("real-git", root.join(GIT_DIR_NAME)).expect("symlink");

        let Err(err) = HostGit::open(&root).await else {
            panic!("a symlinked .git was accepted");
        };

        assert_eq!(
            err.to_string(),
            git_refusal::invalid_git_entry(&root.join(GIT_DIR_NAME))
        );
    }
}
