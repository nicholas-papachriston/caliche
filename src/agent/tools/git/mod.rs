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

    /// The checked, canonical work tree.
    pub(crate) fn work_tree(&self) -> &std::path::Path {
        &self.checked.work_tree
    }
}

/// Check the repository a git tool's `path` names, as
/// [`open_tool_repository`] does, for git to run on inside the sandbox.
/// A repository outside the workspace (host access unlocked) is its own
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

/// Open the repository a git tool's `path` names.
///
/// The path resolves against the tool workspace, not the process directory,
/// and is refused outside the workspace unless host access is unlocked. A
/// link inside the workspace may not lead out of it. Inside the workspace
/// the `.git` is checked by [`caliche_sandbox::git_access::check_repository`]
/// before git reads it, and the repository is opened at the checked git dir
/// and work tree, so a committed gitfile or `commondir` cannot point a tool
/// at another repository on the host.
///
/// # Errors
///
/// Returns a message for the tool result when the path is refused, the
/// repository cannot be opened, or its `.git` is refused.
#[cfg(feature = "tools-git")]
#[cfg_attr(
    tool_file_lint,
    expect(
        clippy::disallowed_methods,
        reason = "libgit2 opens the checked git dir only for the write ops, git_commit and git_branch create and delete"
    )
)]
pub(crate) fn open_tool_repository(path_str: &str) -> Result<git2::Repository, String> {
    use crate::agent::tools::common::path::{
        canonical_tool_path, canonical_workspace, resolve_path_for_tool,
    };
    let refused =
        |e: &dyn std::fmt::Display| format!("Failed to open git repository at '{path_str}': {e}");
    let resolved = resolve_path_for_tool(path_str).map_err(|e| refused(&e))?;
    let canonical = canonical_tool_path(&resolved, path_str).map_err(|e| refused(&e))?;
    let workspace = canonical_workspace().map_err(|e| refused(&e))?;
    if !canonical.starts_with(&workspace) {
        return git2::Repository::open_ext(
            &canonical,
            git2::RepositoryOpenFlags::NO_SEARCH,
            std::iter::empty::<&std::ffi::OsStr>(),
        )
        .map_err(|e| refused(&e));
    }
    open_checked_repository(&workspace, &canonical).map_err(|e| refused(&e))
}

/// Open the repository whose `.git` is in `dir`, a canonical directory at or
/// below the canonical `workspace`, after
/// [`caliche_sandbox::git_access::check_repository`] accepts its layout.
/// libgit2 opens the checked git dir with the checked work tree and does
/// not search a parent directory.
///
/// # Errors
///
/// Returns the refusal, or the error libgit2 gave opening the checked dir.
#[cfg(feature = "tools-git")]
#[cfg_attr(
    tool_file_lint,
    expect(
        clippy::disallowed_methods,
        reason = "libgit2 opens the checked git dir only for the write ops, git_commit and git_branch create and delete"
    )
)]
pub(crate) fn open_checked_repository(
    workspace: &std::path::Path,
    dir: &std::path::Path,
) -> anyhow::Result<git2::Repository> {
    let checked = caliche_sandbox::git_access::check_repository(workspace, dir)?;
    let repo = git2::Repository::open_bare(&checked.git_dir)?;
    repo.set_workdir(&checked.work_tree, false)?;
    Ok(repo)
}

#[cfg(test)]
#[cfg(feature = "tools-git")]
mod tests {
    use super::{GitBranchTool, GitDiffTool, GitLogTool, GitStatusTool, GitTool};
    use crate::agent::tools::Tool;
    use crate::agent::tools::common::set_tool_workspace_root;
    use serde_json::json;
    use std::path::Path;

    /// A repository at `dir` with one commit whose message is `message`.
    fn repo_with_commit(dir: &Path, message: &str) -> git2::Repository {
        let repo = git2::Repository::init(dir).expect("init");
        std::fs::write(dir.join("file.txt"), message).expect("file");
        let mut index = repo.index().expect("index");
        index.add_path(Path::new("file.txt")).expect("add");
        index.write().expect("write index");
        let tree_id = index.write_tree().expect("tree");
        {
            let tree = repo.find_tree(tree_id).expect("find tree");
            let who = git2::Signature::now("t", "t@example.com").expect("sig");
            repo.commit(Some("HEAD"), &who, &who, message, &tree, &[])
                .expect("commit");
        }
        repo
    }

    fn canonical_dir() -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().canonicalize().expect("canonical");
        (dir, path)
    }

    /// A git tool opens only the workspace's own repository: a committed
    /// git link to another repository, an outside path and a relative path
    /// read against the process directory are all refused or resolved in
    /// the workspace.
    #[tokio::test]
    async fn a_git_tool_reads_only_the_workspace_repository() {
        let (_ws_dir, ws) = canonical_dir();
        let (_out_dir, outside) = canonical_dir();
        let own = repo_with_commit(&ws, "INSIDE-COMMIT");
        repo_with_commit(&outside, "OUTSIDE-COMMIT");
        std::fs::create_dir(ws.join("g")).expect("g");
        std::fs::write(
            ws.join("g/.git"),
            format!("gitdir: {}\n", outside.join(".git").display()),
        )
        .expect("gitfile");
        let head = own.head().expect("head").peel_to_commit().expect("commit");
        own.worktree(
            "linked",
            &ws.join("linked"),
            Some(
                git2::WorktreeAddOptions::new().reference(Some(
                    &own.branch("linked", &head, false)
                        .expect("branch")
                        .into_reference(),
                )),
            ),
        )
        .expect("linked worktree");
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
        let foreign = repo_with_commit(&outside, "OUTSIDE-COMMIT");
        let outside_head = foreign.head().expect("head").target().expect("oid");
        let g = ws.join("g");
        std::fs::create_dir(&g).expect("g");
        git2::Repository::init(&g).expect("init");
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
}
