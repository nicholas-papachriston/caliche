pub mod branch;
pub mod commit;
pub mod diff;
pub mod log;
pub mod status;
pub mod tool;

pub use branch::GitBranchTool;
pub use commit::GitCommitTool;
pub use diff::GitDiffTool;
pub use log::GitLogTool;
pub use status::GitStatusTool;
pub use tool::GitTool;

/// The repository a git tool's `path` names, checked, with the workspace
/// the sandbox that runs git on it is rooted at.
#[cfg(feature = "tools-git")]
#[derive(Debug)]
pub(crate) struct ToolRepository {
    checked: caliche_sandbox::git_access::CheckedRepository,
    workspace: std::path::PathBuf,
}

#[cfg(feature = "tools-git")]
impl ToolRepository {
    /// The stdout of `git args`, run read-only inside the OS sandbox
    /// ([`CheckedRepository::run_read_only`](caliche_sandbox::git_access::CheckedRepository::run_read_only)).
    ///
    /// # Errors
    ///
    /// Returns the sandbox refusal, or git's stderr when it exits non-zero.
    pub(crate) async fn read(&self, args: &[&str]) -> Result<String, String> {
        let run = self
            .checked
            .run_read_only(&self.workspace, args)
            .await
            .map_err(|err| err.to_string())?;
        if run.exit_code != 0 {
            return Err(run.stderr.trim().to_string());
        }
        Ok(run.stdout)
    }

    /// The output of the git write op `git args`, with the commit identity
    /// `env`, run inside the OS sandbox with writes allowed only to the
    /// repository's objects, refs, logs, index and lock files
    /// ([`CheckedRepository::run_write`](caliche_sandbox::git_access::CheckedRepository::run_write)).
    /// Only macOS can scope those writes; on Linux and Windows the op
    /// refuses.
    ///
    /// # Errors
    ///
    /// Returns the sandbox refusal, or git's stderr (its stdout when stderr
    /// is empty) when it exits non-zero.
    pub(crate) async fn write(
        &self,
        args: &[&str],
        env: &[(&str, &str)],
    ) -> Result<String, String> {
        let run = self
            .checked
            .run_write(&self.workspace, args, env)
            .await
            .map_err(|err| err.to_string())?;
        if run.exit_code != 0 {
            let stderr = run.stderr.trim();
            let message = if stderr.is_empty() {
                run.stdout.trim()
            } else {
                stderr
            };
            return Err(message.to_string());
        }
        Ok(run.stdout)
    }

    /// The checked, canonical work tree.
    pub(crate) fn work_tree(&self) -> &std::path::Path {
        &self.checked.work_tree
    }
}

/// Check the repository a git tool's `path` names, for git to run on
/// inside the sandbox.
///
/// The path resolves against the tool workspace, not the process directory,
/// and is refused outside the workspace unless host access is unlocked. A
/// link inside the workspace may not lead out of it. The `.git` is checked
/// by [`caliche_sandbox::git_access::check_repository`] before git reads it,
/// and git runs on the checked git dir and work tree, so a committed gitfile
/// or `commondir` cannot point a tool at another repository on the host. A
/// repository outside the workspace (host access unlocked) is its own
/// sandbox root.
///
/// # Errors
///
/// Returns a message for the tool result when the path or its `.git` is
/// refused.
#[cfg(feature = "tools-git")]
pub(crate) fn tool_repository(path_str: &str) -> Result<ToolRepository, String> {
    use crate::agent::tools::common::path::{
        canonical_tool_path, canonical_workspace, resolve_path_for_tool,
    };
    let refused =
        |e: &dyn std::fmt::Display| format!("Failed to open git repository at '{path_str}': {e}");
    let resolved = resolve_path_for_tool(path_str).map_err(|e| refused(&e))?;
    let canonical = canonical_tool_path(&resolved, path_str).map_err(|e| refused(&e))?;
    let workspace = canonical_workspace().map_err(|e| refused(&e))?;
    let workspace = if canonical.starts_with(&workspace) {
        workspace
    } else {
        canonical.clone()
    };
    let checked = caliche_sandbox::git_access::check_repository(&workspace, &canonical)
        .map_err(|e| refused(&e))?;
    Ok(ToolRepository { checked, workspace })
}

/// The repository at the root of `workspace`, checked for git to run on
/// inside the sandbox.
///
/// # Errors
///
/// Returns the error canonicalizing `workspace`, or the refusal of
/// [`check_repository`](caliche_sandbox::git_access::check_repository).
#[cfg(feature = "tools-git")]
#[cfg_attr(
    tool_file_lint,
    expect(
        clippy::disallowed_methods,
        reason = "the harness workspace root, set by the host, not the model"
    )
)]
pub(crate) fn workspace_tool_repository(
    workspace: &std::path::Path,
) -> anyhow::Result<ToolRepository> {
    let workspace = workspace.canonicalize()?;
    let checked = caliche_sandbox::git_access::check_repository(&workspace, &workspace)?;
    Ok(ToolRepository { checked, workspace })
}

#[cfg(test)]
#[cfg(feature = "tools-git")]
mod tests {
    use super::{GitBranchTool, GitDiffTool, GitLogTool, GitStatusTool, GitTool};
    use crate::agent::tools::Tool;
    use crate::agent::tools::common::set_tool_workspace_root;
    use serde_json::json;
    use std::path::Path;

    /// Host `git args` in `dir`, for setting up a test repository.
    fn git_in(dir: &Path, args: &[&str]) -> String {
        let output = crate::policy::git_command()
            .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
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

    /// A repository at `dir` with one commit whose message is `message`;
    /// returns the commit id.
    fn repo_with_commit(dir: &Path, message: &str) -> String {
        git_in(dir, &["init", "-q", "-b", "main"]);
        std::fs::write(dir.join("file.txt"), message).expect("file");
        git_in(dir, &["add", "file.txt"]);
        git_in(dir, &["commit", "-q", "-m", message]);
        git_in(dir, &["rev-parse", "HEAD"])
    }

    fn canonical_dir() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().canonicalize().expect("canonical");
        (dir, path)
    }

    /// A workspace with a submodule gitlink `sub` to `victim`'s head, whose
    /// `sub/.git` holds `gitfile`; the victim's tree has an uncommitted
    /// secret line.
    fn submodule_workspace(ws: &Path, victim: &Path, gitfile: &str) {
        let head = repo_with_commit(victim, "VICTIM");
        std::fs::write(victim.join("file.txt"), "VICTIM-TMP-SECRET line\n").expect("secret");
        repo_with_commit(ws, "INSIDE");
        git_in(
            ws,
            &[
                "update-index",
                "--add",
                "--cacheinfo",
                &format!("160000,{head},sub"),
            ],
        );
        git_in(ws, &["commit", "-q", "-m", "sub"]);
        std::fs::create_dir(ws.join("sub")).expect("sub");
        std::fs::write(ws.join("sub/.git"), gitfile).expect("gitfile");
    }

    /// git_diff does not read a repository outside the workspace through a
    /// submodule: not through a gitfile naming it with `diff.submodule`
    /// set, and not through alternates in the submodule's git dir under
    /// `modules/`.
    #[tokio::test]
    async fn git_diff_does_not_read_the_outside_through_a_submodule() {
        let (_ws_dir, ws) = canonical_dir();
        let (_ws2_dir, ws2) = canonical_dir();
        let victim_dir = tempfile::tempdir_in("/tmp").expect("victim");
        let victim = victim_dir.path().canonicalize().expect("canonical");
        submodule_workspace(
            &ws,
            &victim,
            &format!("gitdir: {}\n", victim.join(".git").display()),
        );
        git_in(&ws, &["config", "diff.submodule", "diff"]);
        let victim2_dir = tempfile::tempdir_in("/tmp").expect("victim");
        let victim2 = victim2_dir.path().canonicalize().expect("canonical");
        submodule_workspace(&ws2, &victim2, "gitdir: ../.git/modules/sub\n");
        let module = ws2.join(".git/modules/sub");
        caliche_policy::write_loose_git_dir(&module).expect("module");
        std::fs::create_dir_all(module.join("objects/info")).expect("info");
        std::fs::write(
            module.join("objects/info/alternates"),
            format!("{}\n", victim2.join(".git/objects").display()),
        )
        .expect("alternates");

        let mut seen = Vec::new();
        for workspace in [&ws, &ws2] {
            set_tool_workspace_root(Some(workspace.clone()));
            let result = GitDiffTool
                .execute(
                    json!({"path": "."}),
                    &crate::agent::tools::ToolPermit::for_tests(),
                )
                .await;
            set_tool_workspace_root(None);
            seen.push((result.success, result.content, result.error));
        }

        assert_eq!(
            seen,
            vec![
                (
                    false,
                    String::new(),
                    Some(format!(
                        "Failed to get git diff: {}/.git config names programs git would run \
                         or submodules it would read: diff.submodule",
                        ws.display()
                    ))
                ),
                (
                    false,
                    String::new(),
                    Some(format!(
                        "Failed to open git repository at '.': {}/.git/modules/sub/objects/info/\
                         alternates points git at another object store",
                        ws2.display()
                    ))
                ),
            ]
        );
    }

    /// A git tool opens only the workspace's own repository: a committed
    /// git link to another repository, an outside path and a relative path
    /// read against the process directory are all refused or resolved in
    /// the workspace.
    #[tokio::test]
    async fn a_git_tool_reads_only_the_workspace_repository() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        repo_with_commit(&ws, "INSIDE-COMMIT");
        repo_with_commit(&outside, "OUTSIDE-COMMIT");
        std::fs::create_dir(ws.join("g")).expect("g");
        std::fs::write(
            ws.join("g/.git"),
            format!("gitdir: {}\n", outside.join(".git").display()),
        )
        .expect("gitfile");
        git_in(
            &ws,
            &[
                "worktree",
                "add",
                "-q",
                "-b",
                "linked",
                &ws.join("linked").to_string_lossy(),
            ],
        );
        set_tool_workspace_root(Some(ws.clone()));

        let tools: [(&str, Box<dyn Tool>, serde_json::Value); 6] = [
            ("git_log", Box::new(GitLogTool), json!({"path": "g"})),
            ("git_status", Box::new(GitStatusTool), json!({"path": "g"})),
            ("git_diff", Box::new(GitDiffTool), json!({"path": "g"})),
            ("git_branch", Box::new(GitBranchTool), json!({"path": "g"})),
            (
                "git log",
                Box::new(GitTool),
                json!({"operation": "log", "path": "g"}),
            ),
            (
                "git_log outside",
                Box::new(GitLogTool),
                json!({"path": outside.to_string_lossy()}),
            ),
        ];
        let mut refused = Vec::new();
        for (name, tool, args) in tools {
            let result = tool
                .execute(args, &crate::agent::tools::ToolPermit::for_tests())
                .await;
            let text = format!("{}{}", result.content, result.error.unwrap_or_default());
            refused.push((name, result.success, text.contains("OUTSIDE-COMMIT")));
        }
        let own_log = GitLogTool
            .execute(
                json!({"path": "."}),
                &crate::agent::tools::ToolPermit::for_tests(),
            )
            .await;
        let linked_log = GitLogTool
            .execute(
                json!({"path": "linked"}),
                &crate::agent::tools::ToolPermit::for_tests(),
            )
            .await;
        let link_error = GitLogTool
            .execute(
                json!({"path": "g"}),
                &crate::agent::tools::ToolPermit::for_tests(),
            )
            .await
            .error
            .unwrap_or_default();
        set_tool_workspace_root(None);

        assert_eq!(
            refused,
            vec![
                ("git_log", false, false),
                ("git_status", false, false),
                ("git_diff", false, false),
                ("git_branch", false, false),
                ("git log", false, false),
                ("git_log outside", false, false),
            ]
        );
        assert_eq!(
            link_error,
            format!(
                "Failed to open git repository at 'g': {}",
                caliche_sandbox::git_access::refusal::git_dir_outside(
                    &ws.join("g/.git"),
                    &outside.join(".git")
                )
            )
        );
        assert!(own_log.content.contains("INSIDE-COMMIT"), "{own_log:?}");
        assert!(
            linked_log.content.contains("INSIDE-COMMIT"),
            "{linked_log:?}"
        );
    }

    /// The git tools print a repository's log, status, staged diff and
    /// branches, read by git inside the sandbox.
    #[tokio::test]
    async fn the_git_tools_report_the_workspace_repository() {
        let (_ws_dir, ws) = canonical_dir();
        let git = |args: &[&str]| {
            let output = crate::policy::git_command()
                .args(["-c", "user.name=t", "-c", "user.email=t@example.com"])
                .args(args)
                .current_dir(&ws)
                .output()
                .expect("git");
            assert!(output.status.success(), "git {args:?}: {output:?}");
            String::from_utf8(output.stdout).expect("utf-8")
        };
        git(&["init", "-q", "-b", "main"]);
        std::fs::write(ws.join("file.txt"), "one\n").expect("file");
        git(&["add", "file.txt"]);
        git(&["commit", "-q", "-m", "INSIDE-COMMIT\n\nbody"]);
        git(&["branch", "side"]);
        let head = git(&["rev-parse", "--short=7", "HEAD"]);
        std::fs::write(ws.join("file.txt"), "two\n").expect("modify");
        std::fs::write(ws.join("new.txt"), "new\n").expect("new");
        git(&["add", "new.txt"]);
        std::fs::write(ws.join("junk.txt"), "junk\n").expect("untracked");
        set_tool_workspace_root(Some(ws.clone()));

        let permit = crate::agent::tools::ToolPermit::for_tests();
        let log = GitLogTool.execute(json!({}), &permit).await;
        let status = GitStatusTool.execute(json!({}), &permit).await;
        let staged = GitDiffTool.execute(json!({"staged": true}), &permit).await;
        let branches = GitBranchTool.execute(json!({}), &permit).await;
        set_tool_workspace_root(None);

        assert_eq!(
            log.content,
            format!(
                "Git Log:\n========\n\n{} t <t@example.com>\n  INSIDE-COMMIT\n\n",
                head.trim()
            )
        );
        assert_eq!(
            status.content,
            "Git Status:\n==========\n\nStaged changes:\n  A new.txt\n\n\
             Modified files:\n  M file.txt\n\nUntracked files:\n  ? junk.txt"
        );
        assert!(
            staged.content.starts_with(
                "Staged changes:\n==================\n\ndiff --git a/new.txt b/new.txt\n"
            ),
            "{staged:?}"
        );
        assert!(staged.content.ends_with("+new\n"), "{staged:?}");
        assert_eq!(branches.content, "Branches:\n========\n\n* main\n  side\n");
    }

    /// A real `.git` in the workspace whose alternates file names an outside
    /// object store, with a branch at the outside commit, does not show the
    /// outside commit through git_log, git_status or git operation=log.
    #[tokio::test]
    async fn a_git_dir_with_alternates_does_not_read_an_outside_repository() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        let outside_head = repo_with_commit(&outside, "OUTSIDE-COMMIT");
        let g = ws.join("g");
        std::fs::create_dir(&g).expect("g");
        git_in(&g, &["init", "-q", "-b", "main"]);
        std::fs::create_dir_all(g.join(".git/objects/info")).expect("info");
        std::fs::write(
            g.join(".git/objects/info/alternates"),
            format!("{}\n", outside.join(".git/objects").display()),
        )
        .expect("alternates");
        std::fs::create_dir_all(g.join(".git/refs/heads")).expect("heads");
        std::fs::write(g.join(".git/refs/heads/main"), format!("{outside_head}\n")).expect("ref");
        std::fs::write(g.join(".git/HEAD"), "ref: refs/heads/main\n").expect("HEAD");
        set_tool_workspace_root(Some(ws.clone()));

        let mut seen = Vec::new();
        let tools: [(&str, Box<dyn Tool>, serde_json::Value); 3] = [
            ("git_log", Box::new(GitLogTool), json!({"path": "g"})),
            ("git_status", Box::new(GitStatusTool), json!({"path": "g"})),
            (
                "git log",
                Box::new(GitTool),
                json!({"operation": "log", "path": "g"}),
            ),
        ];
        for (name, tool, args) in tools {
            let result = tool
                .execute(args, &crate::agent::tools::ToolPermit::for_tests())
                .await;
            let text = format!("{}{}", result.content, result.error.unwrap_or_default());
            seen.push((name, result.success, text.contains("OUTSIDE-COMMIT")));
        }
        set_tool_workspace_root(None);

        assert_eq!(
            seen,
            vec![
                ("git_log", false, false),
                ("git_status", false, false),
                ("git log", false, false),
            ]
        );
    }

    /// A repository with one commit, `staged.txt` staged and a branch
    /// `unmerged` one commit ahead, as the tool workspace.
    fn write_repo(ws: &Path) {
        repo_with_commit(ws, "first");
        git_in(ws, &["switch", "-q", "-c", "unmerged"]);
        std::fs::write(ws.join("ahead.txt"), "ahead").expect("ahead");
        git_in(ws, &["add", "ahead.txt"]);
        git_in(ws, &["commit", "-q", "-m", "ahead"]);
        git_in(ws, &["switch", "-q", "main"]);
        std::fs::write(ws.join("staged.txt"), "staged").expect("staged");
        git_in(ws, &["add", "staged.txt"]);
    }

    /// `(success, content or error)` of each git write tool call, in order.
    async fn run_writes(calls: Vec<(Box<dyn Tool>, serde_json::Value)>) -> Vec<(bool, String)> {
        let permit = crate::agent::tools::ToolPermit::for_tests();
        let mut seen = Vec::new();
        for (tool, args) in calls {
            let result = tool.execute(args, &permit).await;
            seen.push((
                result.success,
                result.error.unwrap_or_else(|| result.content.to_string()),
            ));
        }
        seen
    }

    /// git_commit with nothing staged does not run a commit, which would
    /// read a submodule a committed `.gitmodules` sets to `ignore = none`
    /// (here one whose gitfile names a repository outside the workspace).
    #[tokio::test]
    async fn git_commit_with_nothing_staged_does_not_enter_a_submodule() {
        use super::GitCommitTool;
        let (_ws_dir, ws) = canonical_dir();
        let victim_dir = tempfile::tempdir_in("/tmp").expect("victim");
        let victim = victim_dir.path().canonicalize().expect("canonical");
        let head = repo_with_commit(&victim, "VICTIM");
        repo_with_commit(&ws, "INSIDE");
        git_in(
            &ws,
            &[
                "update-index",
                "--add",
                "--cacheinfo",
                &format!("160000,{head},sub"),
            ],
        );
        std::fs::write(
            ws.join(".gitmodules"),
            "[submodule \"sub\"]\n\tpath = sub\n\turl = ./x\n\tignore = none\n",
        )
        .expect("gitmodules");
        git_in(&ws, &["add", ".gitmodules"]);
        git_in(&ws, &["commit", "-q", "-m", "sub"]);
        std::fs::create_dir(ws.join("sub")).expect("sub");
        std::fs::write(
            ws.join("sub/.git"),
            format!("gitdir: {}\n", victim.join(".git").display()),
        )
        .expect("gitfile");
        set_tool_workspace_root(Some(ws.clone()));

        let result = GitCommitTool
            .execute(
                json!({"message": "nothing"}),
                &crate::agent::tools::ToolPermit::for_tests(),
            )
            .await;
        set_tool_workspace_root(None);

        assert_eq!(
            (result.success, result.error),
            (
                false,
                Some(
                    "No staged changes to commit. Use 'git add' to stage files first.".to_string()
                )
            )
        );
        assert_eq!(git_in(&ws, &["log", "-1", "--format=%s"]), "sub");
    }

    /// git_commit and git_branch create and delete write the repository
    /// through git inside the sandbox. Delete keeps an unmerged branch, and
    /// a commit on a detached HEAD is refused.
    #[cfg(target_os = "macos")]
    #[tokio::test]
    async fn the_git_write_tools_change_the_workspace_repository() {
        use super::GitCommitTool;
        let (_ws_dir, ws) = canonical_dir();
        write_repo(&ws);
        set_tool_workspace_root(Some(ws.clone()));

        let seen = run_writes(vec![
            (
                Box::new(GitCommitTool),
                json!({"message": "second", "author_name": "A", "author_email": "a@example.com"}),
            ),
            (
                Box::new(GitBranchTool),
                json!({"action": "create", "branch_name": "side"}),
            ),
            (
                Box::new(GitBranchTool),
                json!({"action": "delete", "branch_name": "side"}),
            ),
            (
                Box::new(GitBranchTool),
                json!({"action": "delete", "branch_name": "unmerged"}),
            ),
        ])
        .await;
        // git's hints after the first line differ between versions.
        let seen: Vec<(bool, String)> = seen
            .into_iter()
            .map(|(ok, text)| (ok, text.lines().next().unwrap_or_default().to_string()))
            .collect();
        let head = git_in(&ws, &["rev-parse", "--short=7", "HEAD"]);
        git_in(&ws, &["switch", "-q", "--detach"]);
        std::fs::write(ws.join("detached.txt"), "d").expect("detached");
        git_in(&ws, &["add", "detached.txt"]);
        let detached = run_writes(vec![(Box::new(GitCommitTool), json!({"message": "x"}))]).await;
        set_tool_workspace_root(None);

        assert_eq!(
            seen,
            vec![
                (true, format!("Commit created successfully: {head}")),
                (true, "Branch 'side' created successfully".to_string()),
                (true, "Branch 'side' deleted successfully".to_string()),
                (
                    false,
                    "Failed to delete branch: error: the branch 'unmerged' is not fully merged"
                        .to_string()
                ),
            ]
        );
        assert_eq!(
            git_in(&ws, &["log", "-1", "--format=%s|%an|%ae", "main"]),
            "second|A|a@example.com"
        );
        assert_eq!(
            git_in(&ws, &["branch", "--format=%(refname:short)"]),
            "(HEAD detached at {head})\nmain\nunmerged".replace("{head}", &head)
        );
        assert_eq!(
            detached,
            vec![(false, super::commit::DETACHED_HEAD.to_string())]
        );
    }

    /// On Linux the git write tools refuse: Landlock cannot keep the rest
    /// of the git dir read-only while git writes refs and the index.
    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn the_git_write_tools_refuse_on_linux() {
        use super::GitCommitTool;
        let (_ws_dir, ws) = canonical_dir();
        write_repo(&ws);
        set_tool_workspace_root(Some(ws.clone()));

        let seen = run_writes(vec![
            (Box::new(GitCommitTool), json!({"message": "second"})),
            (
                Box::new(GitBranchTool),
                json!({"action": "create", "branch_name": "side"}),
            ),
        ])
        .await;
        set_tool_workspace_root(None);

        let unscoped = caliche_sandbox::git_run::GIT_WRITES_UNSCOPED;
        assert_eq!(
            seen,
            vec![
                (false, format!("Failed to create commit: {unscoped}")),
                (false, format!("Failed to create branch: {unscoped}")),
            ]
        );
        assert_eq!(git_in(&ws, &["log", "--format=%s", "main"]), "first");
    }
}
