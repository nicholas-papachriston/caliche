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

/// A worktree record left behind by a deleted worktree, matched later by a
/// new `.git` at its old path, is refused: the record is older than the
/// `.git` that names it.
#[tokio::test]
async fn a_stale_worktree_record_is_refused() {
    let (_ws_dir, ws) = canonical_dir();
    let (_main_dir, main) = canonical_dir();
    let linked = plant_linked_worktree(&main.join(".git"), "ws").expect("linked");
    std::fs::write(
        linked.join(WORKTREE_GITDIR_FILE),
        format!("{}\n", ws.join(".git").display()),
    )
    .expect("gitdir back");
    tokio::time::sleep(
        caliche_sandbox::git_access::WORKTREE_RECORD_SLACK + std::time::Duration::from_secs(1),
    )
    .await;
    gitfile(&ws, &linked);

    let stale = outcome(&ws, &ws);
    // The remedy the refusal names: touch the record.
    std::fs::File::options()
        .write(true)
        .open(linked.join(WORKTREE_GITDIR_FILE))
        .expect("record")
        .set_modified(std::time::SystemTime::now())
        .expect("touch");
    let touched = outcome(&ws, &ws);

    assert_eq!(
        (stale, touched),
        (
            Err(format!(
                "{ws}/.git names the worktree record {main}/.git/worktrees/ws, which is older \
                 than it; if this is that worktree, run `touch {main}/.git/worktrees/ws/gitdir`",
                ws = ws.display(),
                main = main.display()
            )),
            Ok(CheckedRepository {
                work_tree: ws.clone(),
                git_dir: linked.clone(),
                common_dir: main.join(".git"),
            }),
        )
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

/// Git runs on a checked repository only inside the native sandbox, and
/// the kernel refuses any read a link swapped in after the check would
/// lead to.
///
/// The Windows job does not confine files, so Windows is not here: git
/// refuses there ([`git_does_not_run_without_a_native_sandbox`]).
#[cfg(any(
    all(feature = "sandbox-macos", target_os = "macos"),
    all(feature = "sandbox-linux", target_os = "linux")
))]
mod sandboxed {
    use super::*;

    /// A repository at `dir` with one commit whose subject is `subject`.
    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn commit_repo(dir: &Path, subject: &str) -> String {
        let git = |args: &[&str]| {
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
        git(&["init", "-q", "-b", "main"]);
        std::fs::write(dir.join(format!("{subject}.txt")), subject).expect("file");
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", subject]);
        git(&["rev-parse", "HEAD"])
    }

    /// `(exit code, stdout)` of a sandboxed `git args`, or the error.
    async fn sandboxed(
        checked: &CheckedRepository,
        workspace: &Path,
        args: &[&str],
    ) -> Result<(i64, String), String> {
        checked
            .run_read_only(workspace, args)
            .await
            .map(|run| (run.exit_code, run.stdout.trim().to_string()))
            .map_err(|err| err.to_string())
    }

    #[tokio::test]
    async fn git_reads_the_checked_repository_inside_the_sandbox() {
        let (_ws_dir, ws) = canonical_dir();
        commit_repo(&ws, "INSIDE");
        let checked = check_repository(&ws, &ws).expect("checked");

        assert_eq!(
            sandboxed(&checked, &ws, &["log", "--format=%s"]).await,
            Ok((0, "INSIDE".to_string()))
        );
        let git_dir = sandboxed(&checked, &ws, &["rev-parse", "--absolute-git-dir"])
            .await
            .expect("rev-parse");
        assert_eq!(
            PathBuf::from(git_dir.1).canonicalize().ok(),
            Some(ws.join(".git"))
        );
    }

    /// `objects` swapped for a link to another repository's objects after
    /// the check: git cannot read the outside commit.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_link_swapped_into_the_git_dir_after_the_check_is_not_read() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        commit_repo(&ws, "INSIDE");
        let outside_head = commit_repo(&outside, "OUTSIDE");
        let checked = check_repository(&ws, &ws).expect("checked");

        std::fs::rename(ws.join(".git/objects"), ws.join("moved-objects")).expect("move");
        std::os::unix::fs::symlink(outside.join(".git/objects"), ws.join(".git/objects"))
            .expect("swap");
        let seen = sandboxed(&checked, &ws, &["log", "--format=%s", &outside_head]).await;

        // Seatbelt lets git run and refuses the read through the link; the
        // Linux git guard refuses the linked git dir before git starts.
        #[cfg(target_os = "macos")]
        let expected = Ok((false, String::new()));
        #[cfg(target_os = "linux")]
        let expected = Err(caliche_sandbox::paths::git_refusal::git_dir_in_workspace(
            &ws.join(".git/objects"),
        ));
        assert_eq!(seen.map(|(code, out)| (code == 0, out)), expected);
    }

    /// A checked repository below the workspace swapped for a link to an
    /// outside repository, and back, around the run: git cannot read the
    /// outside repository while the link is in place.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_repository_swapped_for_a_link_after_the_check_is_not_read() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        let sub = ws.join("sub");
        std::fs::create_dir(&sub).expect("sub");
        commit_repo(&sub, "INSIDE");
        commit_repo(&outside, "OUTSIDE");
        let checked = check_repository(&ws, &sub).expect("checked");

        std::fs::rename(&sub, ws.join("moved")).expect("move");
        std::os::unix::fs::symlink(&outside, &sub).expect("swap");
        let seen = sandboxed(&checked, &ws, &["log", "--format=%s"]).await;
        std::fs::remove_file(&sub).expect("unlink");
        std::fs::rename(ws.join("moved"), &sub).expect("back");

        // git reads its working directory first, so the kernel refuses the
        // config listing and the command never runs.
        let read_outside = matches!(&seen, Ok((_, out)) if out.contains("OUTSIDE"));
        assert!(!read_outside, "{seen:?}");
        #[cfg(target_os = "macos")]
        assert_eq!(
            seen,
            Err(
                "git could not list the repository config, so it did not run: \
                 fatal: Unable to read current working directory: Operation not permitted"
                    .to_string()
            )
        );
        assert_eq!(
            sandboxed(&checked, &ws, &["log", "--format=%s"]).await,
            Ok((0, "INSIDE".to_string()))
        );
    }

    /// A repository whose own config names a program is refused before
    /// the command runs.
    #[tokio::test]
    async fn a_repository_that_names_a_program_is_refused() {
        let (_ws_dir, ws) = canonical_dir();
        commit_repo(&ws, "INSIDE");
        let mut config = std::fs::read_to_string(ws.join(".git/config")).expect("config");
        config.push_str("[diff]\n\texternal = /bin/echo\n");
        std::fs::write(ws.join(".git/config"), config).expect("write config");
        let checked = check_repository(&ws, &ws).expect("checked");

        assert_eq!(
            sandboxed(&checked, &ws, &["log", "--format=%s"]).await,
            Err(refusal::program_keys(
                &ws.join(".git"),
                &["diff.external".to_string()]
            ))
        );
    }

    /// A workspace holding a submodule gitlink at `sub` whose `sub/.git`
    /// names `victim`'s git dir, with a change in the victim's tree.
    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn submodule_to(ws: &Path, victim: &Path) {
        let head = commit_repo(victim, "VICTIM");
        std::fs::write(victim.join("VICTIM.txt"), "VICTIM-TMP-SECRET line\n").expect("change");
        commit_repo(ws, "INSIDE");
        let git = |args: &[&str]| {
            let output = caliche_policy::git_command()
                .args(["-c", "user.name=t", "-c", "user.email=t@t"])
                .args(args)
                .current_dir(ws)
                .output()
                .expect("git");
            assert!(output.status.success(), "git {args:?}: {output:?}");
        };
        git(&[
            "update-index",
            "--add",
            "--cacheinfo",
            &format!("160000,{head},sub"),
        ]);
        git(&["commit", "-q", "-m", "sub"]);
        gitfile(&ws.join("sub"), &victim.join(".git"));
    }

    /// A submodule whose gitfile names a repository under the shared temp
    /// dir: `diff` and `status` with the safety flags do not enter it, and
    /// a repository that sets `diff.submodule` is refused.
    #[tokio::test]
    async fn git_does_not_enter_a_submodule_that_names_the_outside() {
        let (_ws_dir, ws) = canonical_dir();
        let victim_dir = tempfile::tempdir_in("/tmp").expect("victim");
        let victim = victim_dir.path().canonicalize().expect("canonical");
        submodule_to(&ws, &victim);
        let checked = check_repository(&ws, &ws).expect("checked");
        let mut diff = vec!["diff"];
        diff.extend(caliche_policy::GIT_DIFF_SAFETY_FLAGS);
        diff.extend(["--submodule=diff", "HEAD"]);
        let mut status = vec!["status"];
        status.extend(caliche_policy::GIT_STATUS_SAFETY_FLAGS);
        status.push("--porcelain=v1");

        let plain = (
            sandboxed(&checked, &ws, &diff).await,
            sandboxed(&checked, &ws, &status).await,
        );
        let mut config = std::fs::read_to_string(ws.join(".git/config")).expect("config");
        config.push_str("[diff]\n\tsubmodule = diff\n");
        std::fs::write(ws.join(".git/config"), config).expect("write config");
        let configured = sandboxed(&checked, &ws, &diff).await;

        assert_eq!(plain, (Ok((0, String::new())), Ok((0, String::new()))));
        assert_eq!(
            configured,
            Err(format!(
                "{}/.git config names programs git would run or submodules it would read: \
                 diff.submodule",
                ws.display()
            ))
        );
    }

    /// The workspace may be a linked worktree whose git dir and common dir
    /// are elsewhere: git reads them inside the sandbox.
    #[tokio::test]
    async fn git_runs_in_a_linked_worktree_workspace() {
        let (_main_dir, main) = canonical_dir();
        let (_ws_parent, parent) = canonical_dir();
        commit_repo(&main, "INSIDE");
        let ws = parent.join("wt");
        let output = caliche_policy::git_command()
            .args(["worktree", "add", "-q", "-b", "side"])
            .arg(&ws)
            .current_dir(&main)
            .output()
            .expect("worktree add");
        assert!(output.status.success(), "{output:?}");
        let checked = check_repository(&ws, &ws).expect("checked");
        let mut status = vec!["status"];
        status.extend(caliche_policy::GIT_STATUS_SAFETY_FLAGS);
        status.push("--porcelain=v1");

        assert_eq!(
            (
                sandboxed(&checked, &ws, &["log", "--format=%s"]).await,
                sandboxed(&checked, &ws, &status).await,
                sandboxed(&checked, &ws, &["branch", "--show-current"]).await,
            ),
            (
                Ok((0, "INSIDE".to_string())),
                Ok((0, String::new())),
                Ok((0, "side".to_string())),
            )
        );
    }

    /// Landlock cannot scope writes in a git dir by file name, so a git
    /// write op refuses on Linux and writes nothing.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn a_git_write_op_refuses_on_linux() {
        let (_ws_dir, ws) = canonical_dir();
        let head = commit_repo(&ws, "INSIDE");
        let checked = check_repository(&ws, &ws).expect("checked");

        let result = checked
            .run_write(&ws, &["branch", "--", "other"], &[])
            .await
            .map(|run| run.exit_code)
            .map_err(|err| err.to_string());

        assert_eq!(
            result,
            Err(caliche_sandbox::git_run::GIT_WRITES_UNSCOPED.to_string())
        );
        assert!(!ws.join(".git/refs/heads/other").exists());
        assert_eq!(
            sandboxed(&checked, &ws, &["rev-parse", "main"]).await,
            Ok((0, head))
        );
    }
}

/// Git write ops under Seatbelt: a commit and a branch change write only
/// the granted paths, and every other write in the git dir is refused by
/// the kernel during the commit.
#[cfg(all(feature = "sandbox-macos", target_os = "macos"))]
mod sandboxed_writes {
    use super::*;

    const IDENTITY: [(&str, &str); 4] = [
        ("GIT_AUTHOR_NAME", "A"),
        ("GIT_AUTHOR_EMAIL", "a@example.com"),
        ("GIT_COMMITTER_NAME", "A"),
        ("GIT_COMMITTER_EMAIL", "a@example.com"),
    ];

    /// A repository at `dir` with one commit and `staged.txt` staged.
    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn staged_repo(dir: &Path) {
        let git = |args: &[&str]| {
            let output = caliche_policy::git_command()
                .args(["-c", "user.name=t", "-c", "user.email=t@t"])
                .args(args)
                .current_dir(dir)
                .output()
                .expect("git");
            assert!(output.status.success(), "git {args:?}: {output:?}");
        };
        git(&["init", "-q", "-b", "main"]);
        std::fs::write(dir.join("first.txt"), "first").expect("file");
        git(&["add", "-A"]);
        git(&["commit", "-q", "-m", "first"]);
        std::fs::write(dir.join("staged.txt"), "staged").expect("file");
        git(&["add", "staged.txt"]);
    }

    /// `(exit code, stdout)` of a host `git args` in `dir`.
    #[expect(clippy::expect_used, reason = "test helper panics with context")]
    fn host(dir: &Path, args: &[&str]) -> String {
        let output = caliche_policy::git_command()
            .args(args)
            .current_dir(dir)
            .output()
            .expect("git");
        String::from_utf8(output.stdout)
            .expect("utf-8")
            .trim()
            .to_string()
    }

    async fn write(checked: &CheckedRepository, ws: &Path, args: &[&str]) -> Result<i64, String> {
        checked
            .run_write(ws, args, &IDENTITY)
            .await
            .map(|run| run.exit_code)
            .map_err(|err| err.to_string())
    }

    #[tokio::test]
    async fn a_commit_and_branch_changes_run_inside_the_sandbox() {
        let (_ws_dir, ws) = canonical_dir();
        staged_repo(&ws);
        let checked = check_repository(&ws, &ws).expect("checked");

        let results = [
            write(
                &checked,
                &ws,
                &["commit", "--cleanup=verbatim", "--message=second"],
            )
            .await,
            write(&checked, &ws, &["branch", "--", "side"]).await,
            write(&checked, &ws, &["branch", "-d", "--", "side"]).await,
        ];

        assert_eq!(results, [Ok(0), Ok(0), Ok(0)]);
        assert_eq!(
            host(&ws, &["log", "--format=%s|%an|%ae"]),
            "second|A|a@example.com\nfirst|t|t@t"
        );
        assert_eq!(host(&ws, &["branch", "--format=%(refname:short)"]), "main");
    }

    /// A git write op may add only the commit identity.
    #[tokio::test]
    async fn a_git_write_op_refuses_other_variables() {
        let (_ws_dir, ws) = canonical_dir();
        staged_repo(&ws);
        let checked = check_repository(&ws, &ws).expect("checked");

        let result = checked
            .run_write(&ws, &["branch", "--", "side"], &[("GIT_DIR", "/tmp")])
            .await
            .map(|run| run.exit_code)
            .map_err(|err| err.to_string());

        assert_eq!(
            result,
            Err("a git write op may set only the commit identity, not GIT_DIR".to_string())
        );
    }

    /// A commit does not enter a submodule whose gitfile names a repository
    /// outside the workspace: with nothing staged, git reads it for the
    /// status it prints, and the sandbox would refuse the read. The commit
    /// ends as git ends one with nothing to commit.
    #[tokio::test]
    async fn a_commit_does_not_enter_a_submodule_that_names_the_outside() {
        let (_ws_dir, ws) = canonical_dir();
        let victim_dir = tempfile::tempdir_in("/tmp").expect("victim");
        let victim = victim_dir.path().canonicalize().expect("canonical");
        staged_repo(&victim);
        host(
            &victim,
            &[
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "commit",
                "-q",
                "-m",
                "v",
            ],
        );
        let head = host(&victim, &["rev-parse", "HEAD"]);
        staged_repo(&ws);
        host(
            &ws,
            &[
                "update-index",
                "--add",
                "--cacheinfo",
                &format!("160000,{head},sub"),
            ],
        );
        gitfile(&ws.join("sub"), &victim.join(".git"));
        host(
            &ws,
            &[
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "commit",
                "-q",
                "-m",
                "with sub",
            ],
        );
        let checked = check_repository(&ws, &ws).expect("checked");

        let committed = write(
            &checked,
            &ws,
            &["commit", "--cleanup=verbatim", "--message=nothing"],
        )
        .await;

        assert_eq!(committed, Ok(1));
        assert_eq!(host(&ws, &["log", "-1", "--format=%s"]), "with sub");
    }

    /// A commit in a workspace that is a linked worktree writes the
    /// worktree's index and the shared objects and refs.
    #[tokio::test]
    async fn a_commit_runs_in_a_linked_worktree_workspace() {
        let (_main_dir, main) = canonical_dir();
        let (_ws_parent, parent) = canonical_dir();
        staged_repo(&main);
        let ws = parent.join("wt");
        let output = caliche_policy::git_command()
            .args(["worktree", "add", "-q", "-b", "side"])
            .arg(&ws)
            .current_dir(&main)
            .output()
            .expect("worktree add");
        assert!(output.status.success(), "{output:?}");
        std::fs::write(ws.join("linked.txt"), "linked").expect("file");
        host(&ws, &["add", "linked.txt"]);
        let checked = check_repository(&ws, &ws).expect("checked");

        let committed = write(
            &checked,
            &ws,
            &["commit", "--cleanup=verbatim", "--message=linked"],
        )
        .await;

        assert_eq!(committed, Ok(0));
        assert_eq!(
            host(&main, &["log", "-1", "--format=%s|%an", "side"]),
            "linked|A"
        );
    }

    /// `refs` swapped for a link to a copy outside `.git` after the check:
    /// the commit's ref update resolves outside the grant and the kernel
    /// refuses it, so the copy is unchanged. The copy is in the workspace,
    /// which the sandbox may read, so only the write is refused.
    #[tokio::test]
    async fn a_refs_dir_swapped_for_a_link_after_the_check_is_not_written() {
        let (_ws_dir, ws) = canonical_dir();
        staged_repo(&ws);
        let outside = ws.join("copy");
        std::fs::create_dir(&outside).expect("copy dir");
        let checked = check_repository(&ws, &ws).expect("checked");
        let main_before = std::fs::read(ws.join(".git/refs/heads/main")).expect("main");
        std::fs::rename(ws.join(".git/refs"), outside.join("refs")).expect("move refs");
        std::os::unix::fs::symlink(outside.join("refs"), ws.join(".git/refs")).expect("link");

        let committed = write(
            &checked,
            &ws,
            &["commit", "--cleanup=verbatim", "--message=second"],
        )
        .await;

        assert_eq!(committed, Ok(128));
        assert_eq!(
            std::fs::read(outside.join("refs/heads/main")).ok(),
            Some(main_before)
        );
    }
}

/// Without a native backend that confines files, git does not run. That
/// includes Windows, whose job backend does not confine files.
#[cfg(not(any(
    all(feature = "sandbox-macos", target_os = "macos"),
    all(feature = "sandbox-linux", target_os = "linux")
)))]
#[tokio::test]
async fn git_does_not_run_without_a_native_sandbox() {
    let (_ws_dir, ws) = canonical_dir();
    write_loose_git_dir(&ws.join(".git")).expect("own");
    let checked = check_repository(&ws, &ws).expect("checked");

    let read = checked
        .run_read_only(&ws, &["status"])
        .await
        .expect_err("refused");
    let write = checked
        .run_write(&ws, &["branch", "--", "other"], &[])
        .await
        .expect_err("refused");

    for error in [read, write] {
        assert!(
            error
                .to_string()
                .starts_with(caliche_sandbox::git_run::NO_GIT_SANDBOX),
            "{error}"
        );
    }
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
    let inside_head = git_in(&ws, &["rev-parse", "HEAD"]);
    // Only upload-pack honors a namespace; plant one for `ls-remote` to see.
    git_in(
        &ws,
        &[
            "update-ref",
            "refs/namespaces/planted/refs/heads/other",
            "HEAD",
        ],
    );
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
            &["ls-remote", "--refs", "."],
        ),
        pinned("GIT_WORK_TREE", &outside, &["ls-files", "--others"]),
    ];

    assert_eq!(
        seen,
        [
            "true|inside.txt",
            "false|",
            "false|",
            format!(
                "true|{inside_head}\trefs/heads/main\n\
                 {inside_head}\trefs/namespaces/planted/refs/heads/other"
            )
            .as_str(),
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

/// An alternates file or a `commondir` in a submodule's git dir under
/// `modules/` is refused too: git reads it when it enters the submodule.
#[test]
fn nested_alternates_and_commondir_are_refused() {
    let (_ws_dir, ws) = canonical_dir();
    let (_out_dir, outside) = canonical_dir();
    write_loose_git_dir(&outside.join(".git")).expect("outside");
    write_loose_git_dir(&ws.join(".git")).expect("own");
    let module = ws.join(".git/modules/sub");
    write_loose_git_dir(&module).expect("module");
    std::fs::create_dir_all(module.join("objects/info")).expect("info");
    std::fs::write(
        module.join("objects/info/alternates"),
        format!("{}\n", outside.join(".git/objects").display()),
    )
    .expect("alternates");
    let alternates = outcome(&ws, &ws);
    std::fs::remove_file(module.join("objects/info/alternates")).expect("unplant");
    std::fs::write(
        module.join(COMMONDIR_FILE),
        format!("{}\n", outside.join(".git").display()),
    )
    .expect("commondir");
    let commondir = outcome(&ws, &ws);

    assert_eq!(
        (alternates, commondir),
        (
            Err(format!(
                "{}/.git/modules/sub/objects/info/alternates points git at another object store",
                ws.display()
            )),
            Err(format!(
                "{}/.git/modules/sub/commondir points git at another git dir",
                ws.display()
            )),
        )
    );
}
