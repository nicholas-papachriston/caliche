//! `check_repository` accepts a workspace's own repository and refuses a
//! `.git` that leads to a git dir outside the workspace.
#![cfg(any(unix, windows))]

use caliche_policy::{COMMONDIR_FILE, plant_linked_worktree, write_loose_git_dir};
use caliche_sandbox::git_access::{
    CheckedRepository, WORKTREE_GITDIR_FILE, check_repository, refusal,
};
use std::path::{Path, PathBuf};

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn canonical_dir() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("dir");
    let path = dir.path().canonicalize().expect("canonical");
    (dir, path)
}

#[expect(clippy::expect_used, reason = "test helper panics with context")]
fn gitfile(dir: &Path, git_dir: &Path) {
    std::fs::create_dir_all(dir).expect("dir");
    std::fs::write(dir.join(".git"), format!("gitdir: {}\n", git_dir.display())).expect("gitfile");
}

/// The refusal message `check_repository` gave, or the checked paths.
fn outcome(workspace: &Path, dir: &Path) -> Result<CheckedRepository, String> {
    check_repository(workspace, dir).map_err(|err| err.to_string())
}

#[test]
fn the_workspace_repository_and_one_inside_it_are_accepted() {
    let (_ws_dir, ws) = canonical_dir();
    write_loose_git_dir(&ws.join(".git")).expect("own");
    let inner = plant_linked_worktree(&ws.join(".git"), "inner").expect("inner");
    gitfile(&ws.join("inner"), &inner);

    assert_eq!(
        (outcome(&ws, &ws), outcome(&ws, &ws.join("inner"))),
        (
            Ok(CheckedRepository {
                work_tree: ws.clone(),
                git_dir: ws.join(".git"),
                common_dir: ws.join(".git"),
            }),
            Ok(CheckedRepository {
                work_tree: ws.join("inner"),
                git_dir: inner,
                common_dir: ws.join(".git"),
            }),
        )
    );
}

/// The workspace root may be a linked worktree of a repository elsewhere,
/// when that worktree's git dir names the root's `.git` back.
#[test]
fn the_workspace_as_a_linked_worktree_is_accepted() {
    let (_ws_dir, ws) = canonical_dir();
    let (_main_dir, main) = canonical_dir();
    let linked = plant_linked_worktree(&main.join(".git"), "ws").expect("linked");
    std::fs::write(
        linked.join(WORKTREE_GITDIR_FILE),
        format!("{}\n", ws.join(".git").display()),
    )
    .expect("gitdir back");
    gitfile(&ws, &linked);

    assert_eq!(
        outcome(&ws, &ws),
        Ok(CheckedRepository {
            work_tree: ws.clone(),
            git_dir: linked,
            common_dir: main.join(".git"),
        })
    );
}

#[test]
fn a_git_link_to_a_repository_outside_is_refused() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    write_loose_git_dir(&ws.join(".git")).expect("own");
    write_loose_git_dir(&outside.join(".git")).expect("outside");
    let linked = plant_linked_worktree(&outside.join(".git"), "x").expect("linked");
    gitfile(&ws.join("g"), &outside.join(".git"));
    gitfile(&ws.join("w"), &linked);
    std::fs::write(
        linked.join(WORKTREE_GITDIR_FILE),
        format!("{}\n", ws.join("w/.git").display()),
    )
    .expect("gitdir back");
    let inside = ws.join("c/.git");
    write_loose_git_dir(&inside).expect("inside git dir");
    std::fs::write(
        inside.join(COMMONDIR_FILE),
        format!("{}\n", outside.join(".git").display()),
    )
    .expect("commondir out");

    assert_eq!(
        [
            outcome(&ws, &ws.join("g")),
            outcome(&ws, &ws.join("w")),
            outcome(&ws, &ws.join("c")),
        ],
        [
            Err(refusal::git_dir_outside(
                &ws.join("g/.git"),
                &outside.join(".git")
            )),
            Err(refusal::git_dir_outside(&ws.join("w/.git"), &linked)),
            Err(refusal::git_dir_outside(
                &inside.join(COMMONDIR_FILE),
                &outside.join(".git")
            )),
        ]
    );
}

/// The root's gitfile may name a git dir elsewhere only when that git dir
/// names the root back.
#[test]
fn a_root_gitfile_to_a_foreign_git_dir_is_refused() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    write_loose_git_dir(&outside.join(".git")).expect("outside");
    gitfile(&ws, &outside.join(".git"));

    assert_eq!(
        outcome(&ws, &ws),
        Err(refusal::git_dir_outside(
            &ws.join(".git"),
            &outside.join(".git")
        ))
    );
}

#[cfg(unix)]
#[test]
fn a_linked_dot_git_or_directory_is_refused() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    write_loose_git_dir(&outside.join(".git")).expect("outside");
    std::fs::create_dir(ws.join("l")).expect("l");
    std::os::unix::fs::symlink(outside.join(".git"), ws.join("l/.git")).expect("link .git");
    std::os::unix::fs::symlink(&outside, ws.join("d")).expect("link dir");

    let through_dir = check_repository(&ws, &ws.join("d")).map_err(|err| err.kind());

    assert_eq!(
        outcome(&ws, &ws.join("l")),
        Err(refusal::not_a_git_entry(&ws.join("l/.git")))
    );
    assert!(through_dir.is_err(), "{through_dir:?}");
}

#[test]
fn the_pinned_environment_names_the_checked_paths() {
    let (_ws_dir, ws) = canonical_dir();
    write_loose_git_dir(&ws.join(".git")).expect("own");
    let checked = check_repository(&ws, &ws).expect("checked");

    let (set, remove) = checked.git_env();

    assert_eq!(
        set.map(|(name, value)| (name, PathBuf::from(value))),
        [
            ("GIT_DIR", ws.join(".git")),
            ("GIT_WORK_TREE", ws.clone()),
            (
                "GIT_CEILING_DIRECTORIES",
                ws.parent().expect("parent").to_path_buf()
            ),
        ]
    );
    assert_eq!(
        remove,
        [
            "GIT_COMMON_DIR",
            "GIT_INDEX_FILE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_NAMESPACE",
        ]
    );
}

/// `git` started from the checked repository reports its git dir, not one
/// it would find from the process directory.
#[test]
fn the_pinned_git_runs_in_the_checked_repository() {
    let (_ws_dir, ws) = canonical_dir();
    let inner = ws.join("inner");
    std::fs::create_dir(&inner).expect("inner");
    let init = std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(&inner)
        .status()
        .expect("git init");
    assert!(init.success());
    let checked = check_repository(&ws, &inner).expect("checked");

    let output = checked
        .git_command()
        .args(["rev-parse", "--absolute-git-dir"])
        .output()
        .expect("git rev-parse");

    let reported = String::from_utf8(output.stdout).expect("utf-8");
    assert_eq!(
        PathBuf::from(reported.trim()).canonicalize().ok(),
        Some(inner.join(".git"))
    );
}

/// The pin replaces an inherited `GIT_DIR` and removes an inherited
/// `GIT_COMMON_DIR` that would point `git` at another repository.
#[test]
fn the_pin_overrides_inherited_repository_variables() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    for dir in [&ws, &outside] {
        let init = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir)
            .status()
            .expect("git init");
        assert!(init.success());
    }
    let checked = check_repository(&ws, &ws).expect("checked");
    let mut command = std::process::Command::new("git");
    command
        .env("GIT_DIR", outside.join(".git"))
        .env("GIT_COMMON_DIR", outside.join(".git"))
        .current_dir(&ws);
    checked.pin(&mut command);

    let output = command
        .args([
            "rev-parse",
            "--absolute-git-dir",
            "--path-format=absolute",
            "--git-common-dir",
        ])
        .output()
        .expect("git rev-parse");

    let reported: Vec<Option<PathBuf>> = String::from_utf8(output.stdout)
        .expect("utf-8")
        .lines()
        .map(|line| PathBuf::from(line).canonicalize().ok())
        .collect();
    assert_eq!(reported, vec![Some(ws.join(".git")), Some(ws.join(".git"))]);
}
