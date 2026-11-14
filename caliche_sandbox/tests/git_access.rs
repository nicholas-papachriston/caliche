//! `check_repository` accepts a workspace's own repository and refuses a
//! `.git` that leads to a git dir outside the workspace.
#![cfg(any(unix, windows))]

use caliche_policy::{COMMONDIR_FILE, plant_linked_worktree, write_loose_git_dir};
use caliche_sandbox::external_tool_path;
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

/// `git` started from the checked repository reports its git dir, not one
/// it would find from the process directory.
#[test]
fn the_pinned_git_runs_in_the_checked_repository() {
    let (_ws_dir, ws) = canonical_dir();
    let inner = ws.join("inner");
    std::fs::create_dir(&inner).expect("inner");
    let init = caliche_policy::program_command("git")
        .args(["init", "-q"])
        .current_dir(&inner)
        .status()
        .expect("git init");
    assert!(init.success());
    let checked = check_repository(&ws, &inner).expect("checked");

    let output = checked
        .git_command()
        .expect("no program keys")
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
        let init = caliche_policy::program_command("git")
            .args(["init", "-q"])
            .current_dir(dir)
            .status()
            .expect("git init");
        assert!(init.success());
    }
    let checked = check_repository(&ws, &ws).expect("checked");
    let mut command = caliche_policy::program_command("git");
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

/// Set in the child run of [`inherited_config_and_programs_do_not_reach_git`].
const CHILD_MARKER_DIR: &str = "CALICHE_GIT_RUNNER_CHILD_MARKERS";

/// An inherited `GIT_EXTERNAL_DIFF`, and a filter driver added through
/// `GIT_CONFIG_COUNT`, do not run when git diffs a changed file.
#[cfg(unix)]
#[test]
fn inherited_config_and_programs_do_not_reach_git() {
    use std::os::unix::fs::PermissionsExt;
    let Some(markers) = std::env::var_os(CHILD_MARKER_DIR) else {
        let markers = tempfile::tempdir().expect("markers");
        let script = |name: &str| {
            let path = markers.path().join(format!("{name}.sh"));
            let ran = markers.path().join(name);
            std::fs::write(
                &path,
                format!("#!/bin/sh\ntouch '{}'\ncat\n", ran.display()),
            )
            .expect("script");
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).expect("chmod");
            path
        };
        let name = "inherited_config_and_programs_do_not_reach_git";
        let output = caliche_policy::program_command(std::env::current_exe().expect("test binary"))
            .args(["--exact", name, "--nocapture"])
            .env(CHILD_MARKER_DIR, markers.path())
            .env("GIT_EXTERNAL_DIFF", script("external"))
            .env("GIT_CONFIG_COUNT", "1")
            .env("GIT_CONFIG_KEY_0", "filter.planted.clean")
            .env("GIT_CONFIG_VALUE_0", script("filter"))
            .output()
            .expect("child");
        assert!(output.status.success(), "{output:?}");
        let ran: Vec<_> = ["external", "filter"]
            .into_iter()
            .filter(|name| markers.path().join(name).exists())
            .collect();
        assert_eq!(ran, Vec::<&str>::new());
        return;
    };
    let _ = markers;
    let repo = tempfile::tempdir().expect("repo");
    let git = |args: &[&str]| {
        let status = caliche_policy::git_command()
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .current_dir(repo.path())
            .status()
            .expect("git");
        assert!(status.success(), "git {args:?}");
    };
    git(&["init", "-q"]);
    std::fs::write(repo.path().join(".gitattributes"), "a.txt filter=planted\n").expect("attrs");
    std::fs::write(repo.path().join("a.txt"), "one\n").expect("a");
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "one"]);
    std::fs::write(repo.path().join("a.txt"), "two\n").expect("edit");
    git(&["diff"]);
}

/// A real `.git` whose `objects` is a link to another repository's objects
/// is refused: git would read the outside objects through it.
#[cfg(unix)]
#[test]
fn a_git_dir_with_linked_objects_is_refused() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    write_loose_git_dir(&outside.join(".git")).expect("outside");
    write_loose_git_dir(&ws.join(".git")).expect("own");
    std::fs::remove_dir_all(ws.join(".git/objects")).expect("objects");
    std::os::unix::fs::symlink(outside.join(".git/objects"), ws.join(".git/objects"))
        .expect("link");

    assert_eq!(
        outcome(&ws, &ws),
        Err(refusal::special_in_git_dir(&ws.join(".git/objects")))
    );
}

/// An alternates file is refused, with no link anywhere: it would make git
/// read an object store outside the workspace.
#[test]
fn a_git_dir_with_alternates_is_refused() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    write_loose_git_dir(&outside.join(".git")).expect("outside");
    write_loose_git_dir(&ws.join(".git")).expect("own");
    std::fs::create_dir_all(ws.join(".git/objects/info")).expect("info");
    std::fs::write(
        ws.join(".git/objects/info/alternates"),
        format!("{}\n", outside.join(".git/objects").display()),
    )
    .expect("alternates");

    assert_eq!(
        outcome(&ws, &ws),
        Err(refusal::alternates(
            &ws.join(".git/objects/info/alternates")
        ))
    );
}

/// A FIFO at `.git/HEAD` is refused from the listing, without opening it,
/// so the check does not hang.
#[cfg(unix)]
#[test]
fn a_fifo_head_is_refused_without_reading_it() {
    let (_ws_dir, ws) = canonical_dir();
    write_loose_git_dir(&ws.join(".git")).expect("own");
    std::fs::remove_file(ws.join(".git/HEAD")).expect("head");
    let status = caliche_policy::program_command("mkfifo")
        .arg(ws.join(".git/HEAD"))
        .status()
        .expect("mkfifo");
    assert!(status.success());

    let (sender, receiver) = std::sync::mpsc::channel();
    let checked_ws = ws.clone();
    std::thread::spawn(move || {
        let _ = sender.send(outcome(&checked_ws, &checked_ws));
    });

    assert_eq!(
        receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("the check returned"),
        Err(refusal::special_in_git_dir(&ws.join(".git/HEAD")))
    );
}

/// The workspace root's gitfile may name a linked worktree git dir outside
/// only when it is `<common>/worktrees/<id>`; a git dir elsewhere that
/// names the root back is refused.
#[test]
fn an_outside_worktree_git_dir_not_under_worktrees_is_refused() {
    let (_ws_dir, ws) = canonical_dir();
    let (_main_dir, main) = canonical_dir();
    write_loose_git_dir(&main.join(".git")).expect("main");
    let planted = main.join("planted");
    write_loose_git_dir(&planted).expect("planted");
    std::fs::write(planted.join(COMMONDIR_FILE), "../.git\n").expect("commondir");
    std::fs::write(
        planted.join(WORKTREE_GITDIR_FILE),
        format!("{}\n", ws.join(".git").display()),
    )
    .expect("gitdir");
    gitfile(&ws, &planted);

    assert_eq!(outcome(&ws, &ws), Err(refusal::not_a_git_dir(&planted)));
}

/// Each inherited variable that would point the pinned `git` at another
/// repository's index, objects, refs or work tree has no effect: the
/// commands see only the workspace repository.
#[test]
fn the_pin_ignores_every_inherited_repository_redirect() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    let git_in = |dir: &Path, args: &[&str]| -> String {
        let output = caliche_policy::git_command()
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git");
        assert!(output.status.success(), "git {args:?}: {output:?}");
        String::from_utf8(output.stdout)
            .expect("utf-8")
            .trim()
            .to_string()
    };
    for (dir, file) in [(&ws, "inside.txt"), (&outside, "OUTSIDE.txt")] {
        git_in(dir, &["init", "-q", "-b", "main"]);
        std::fs::write(dir.join(file), "x").expect("file");
        git_in(dir, &["add", "-A"]);
        git_in(dir, &["commit", "-q", "-m", file]);
    }
    std::fs::write(outside.join("UNTRACKED-OUTSIDE.txt"), "x").expect("untracked");
    let outside_head = git_in(&outside, &["rev-parse", "HEAD"]);
    let checked = check_repository(&ws, &ws).expect("checked");
    // "<exit ok>|<stdout>" of `git args` pinned, with `var` inherited.
    let pinned = |var: &str, value: &Path, args: &[&str]| -> String {
        let mut command = caliche_policy::git_command();
        command.env(var, value);
        checked.pin(&mut command);
        let output = command.args(args).current_dir(&ws).output().expect("git");
        format!(
            "{}|{}",
            output.status.success(),
            String::from_utf8_lossy(&output.stdout).trim()
        )
    };
    let objects = outside.join(".git/objects");

    let seen = [
        pinned("GIT_INDEX_FILE", &outside.join(".git/index"), &["ls-files"]),
        pinned(
            "GIT_OBJECT_DIRECTORY",
            &objects,
            &["cat-file", "-e", &outside_head],
        ),
        pinned(
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            &objects,
            &["cat-file", "-e", &outside_head],
        ),
        pinned(
            "GIT_NAMESPACE",
            Path::new("planted"),
            &["for-each-ref", "--format=%(refname)"],
        ),
        pinned("GIT_WORK_TREE", &outside, &["ls-files", "--others"]),
    ];

    assert_eq!(
        seen,
        [
            "true|inside.txt",
            "false|",
            "false|",
            "true|refs/heads/main",
            "true|",
        ]
    );
}

/// A directory outside the workspace, a `.git` file that is not a gitfile,
/// and a `.git` dir that is not a git dir are each refused with their own
/// message.
#[test]
fn the_check_refuses_an_outside_dir_bad_gitfile_and_non_git_dir() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    std::fs::create_dir(ws.join("text")).expect("text");
    std::fs::write(ws.join("text/.git"), "not a gitfile\n").expect("text gitfile");
    std::fs::create_dir_all(ws.join("bare/.git/objects")).expect("bare");

    assert_eq!(
        [
            outcome(&ws, &outside),
            outcome(&ws, &ws.join("text")),
            outcome(&ws, &ws.join("bare")),
        ],
        [
            Err(refusal::outside_workspace(&outside)),
            Err(refusal::not_a_git_entry(&ws.join("text/.git"))),
            Err(refusal::not_a_git_dir(&ws.join("bare/.git"))),
        ]
    );
}
