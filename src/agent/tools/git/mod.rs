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

/// Open the repository a git tool's `path` names.
///
/// The path resolves against the tool workspace, not the process directory,
/// and is refused outside the workspace unless host access is unlocked. A
/// link inside the workspace may not lead out of it. The repository opened
/// at a place inside the workspace must be the workspace's own repository
/// (a linked worktree of it included) or one whose git dir is inside the
/// workspace, so a committed git link cannot point a tool at another
/// repository on the host.
///
/// # Errors
///
/// Returns a message for the tool result when the path is refused, the
/// repository cannot be opened, or it is not the workspace's.
#[cfg(feature = "tools-git")]
pub(crate) fn open_tool_repository(path_str: &str) -> Result<git2::Repository, String> {
    use crate::agent::tools::common::path::{
        get_current_dir, refuse_link_out_of_workspace, resolve_path_for_tool,
    };
    let refused =
        |e: &dyn std::fmt::Display| format!("Failed to open git repository at '{path_str}': {e}");
    let resolved = resolve_path_for_tool(path_str).map_err(|e| refused(&e))?;
    let canonical = resolved.canonicalize().map_err(|e| refused(&e))?;
    refuse_link_out_of_workspace(&resolved, &canonical).map_err(|e| refused(&e))?;
    let workspace = get_current_dir()
        .and_then(|dir| Ok(dir.canonicalize()?))
        .map_err(|e| refused(&e))?;
    let repo = git2::Repository::open_ext(
        &canonical,
        git2::RepositoryOpenFlags::NO_SEARCH,
        std::iter::empty::<&std::ffi::OsStr>(),
    )
    .map_err(|e| refused(&e))?;
    if canonical.starts_with(&workspace) {
        check_workspace_repository(&repo, &workspace).map_err(|e| refused(&e))?;
    }
    Ok(repo)
}

/// Refuse `repo` unless it is the workspace's own repository or its git
/// dir, common dir and work tree are all inside `workspace`.
#[cfg(feature = "tools-git")]
fn check_workspace_repository(
    repo: &git2::Repository,
    workspace: &std::path::Path,
) -> std::io::Result<()> {
    let git_dir = repo.path().canonicalize()?;
    let common = repo.commondir().canonicalize()?;
    let work_tree = repo
        .workdir()
        .map(std::path::Path::canonicalize)
        .transpose()?;
    let own_common = match caliche_policy::classify(workspace) {
        Ok(caliche_policy::WorkspaceGit::Own { git_dir }) => {
            Some(caliche_policy::common_dir(&git_dir)?.canonicalize()?)
        }
        _ => None,
    };
    let work_tree_inside = work_tree
        .as_deref()
        .is_none_or(|dir| dir.starts_with(workspace));
    let own = own_common.as_deref() == Some(common.as_path());
    let inside = git_dir.starts_with(workspace) && common.starts_with(workspace);
    if work_tree_inside && (own || inside) {
        return Ok(());
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::PermissionDenied,
        format!(
            "it is a repository outside the workspace ('{}'), not the workspace's own",
            common.display()
        ),
    ))
}

#[cfg(all(test, feature = "tools-git"))]
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
            let result = tool.execute(args).await;
            let text = format!("{}{}", result.content, result.error.unwrap_or_default());
            refused.push((name, result.success, text.contains("OUTSIDE-COMMIT")));
        }
        let own_log = GitLogTool.execute(json!({"path": "."})).await;
        let linked_log = GitLogTool.execute(json!({"path": "linked"})).await;
        let link_error = GitLogTool
            .execute(json!({"path": "g"}))
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
                "Failed to open git repository at 'g': it is a repository outside the workspace ('{}'), not the workspace's own",
                outside.join(".git").display()
            )
        );
        assert!(own_log.content.contains("INSIDE-COMMIT"), "{own_log:?}");
        assert!(
            linked_log.content.contains("INSIDE-COMMIT"),
            "{linked_log:?}"
        );
    }
}
