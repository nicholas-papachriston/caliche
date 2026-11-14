#[cfg(feature = "tools-git")]
use crate::agent::tools::common::params;
#[cfg(feature = "tools-git")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-git")]
use async_trait::async_trait;
#[cfg(feature = "tools-git")]
use serde_json::json;

const NAME: &str = "git_commit";
const DESCRIPTION: &str = "Create a git commit with staged changes on the current branch. Requires files to be staged first (use git add). This tool commits staged changes with the provided message. Runs only on macOS, where the sandbox can limit git's writes to the repository's objects, refs, logs and index; on Linux and Windows it refuses.";

/// Returned when `HEAD` is detached: the commit would rewrite `HEAD`, which
/// stays read-only in the sandbox.
pub const DETACHED_HEAD: &str =
    "HEAD is detached; git_commit commits only on a branch, because HEAD stays read-only";

#[derive(Debug)]
pub struct GitCommitTool;

#[cfg(feature = "tools-git")]
impl ToolMeta for GitCommitTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the git repository (default: current directory)",
                    "default": "."
                },
                "message": {
                    "type": "string",
                    "description": "Commit message"
                },
                "author_name": {
                    "type": "string",
                    "description": "Author name (default: uses git config)"
                },
                "author_email": {
                    "type": "string",
                    "description": "Author email (default: uses git config)"
                }
            },
            "required": ["message"]
        })
    }
}

#[async_trait]
impl Tool for GitCommitTool {
    async fn execute(
        &self,
        arguments: serde_json::Value,
        _permit: &crate::agent::tools::ToolPermit<'_>,
    ) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let message = match params::extract_string(&arguments, "message", None) {
            Ok(m) => m,
            Err(e) => return ToolResult::error(e),
        };

        let repo = match super::tool_repository(&path_str) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(e),
        };
        // git commit with nothing staged reads every submodule for the
        // status it prints, and a .gitmodules `ignore` setting overrides
        // the write ops' diff.ignoreSubmodules=all. The flag on this diff
        // overrides it, so check here and commit only staged changes,
        // which git commits without entering a submodule. A change to a
        // submodule's commit alone is not counted.
        let mut staged = vec!["diff", "--cached", "--name-only", "-z"];
        staged.extend(crate::policy::GIT_DIFF_SAFETY_FLAGS);
        match repo.read(&staged).await {
            Ok(listed) if listed.is_empty() => {
                return ToolResult::error(
                    "No staged changes to commit. Use 'git add' to stage files first.".to_string(),
                );
            }
            Ok(_) => {}
            Err(e) => return ToolResult::error(format!("Failed to get index: {e}")),
        }
        // HEAD stays read-only in the sandbox, so only a branch can move.
        if repo.read(&["symbolic-ref", "-q", "HEAD"]).await.is_err() {
            return ToolResult::error(DETACHED_HEAD.to_string());
        }

        let identity = match (
            arguments["author_name"].as_str(),
            arguments["author_email"].as_str(),
        ) {
            (Some(name), Some(email)) => vec![
                ("GIT_AUTHOR_NAME", name),
                ("GIT_AUTHOR_EMAIL", email),
                ("GIT_COMMITTER_NAME", name),
                ("GIT_COMMITTER_EMAIL", email),
            ],
            _ => Vec::new(),
        };
        let message_arg = format!("--message={message}");
        if let Err(e) = repo
            .write(
                &["commit", "--cleanup=verbatim", "--no-verify", &message_arg],
                &identity,
            )
            .await
        {
            return ToolResult::error(format!("Failed to create commit: {e}"));
        }
        let commit_id = match repo.read(&["rev-parse", "--short=7", "HEAD"]).await {
            Ok(id) => id.trim().to_string(),
            Err(e) => return ToolResult::error(format!("Failed to read the new commit: {e}")),
        };

        ToolResult::success(format!("Commit created successfully: {commit_id}"))
    }
}

#[cfg(not(feature = "tools-git"))]
#[async_trait]
impl Tool for GitCommitTool {
    fn name(&self) -> &str {
        "git_commit"
    }

    fn description(&self) -> &str {
        "Create git commit. Requires 'tools-git' feature."
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("git_commit")
    }
    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("Create git commit. Requires 'tools-git' feature.")
    }

    fn parameters(&self) -> serde_json::Value {
        json!({})
    }

    async fn execute(
        &self,
        _arguments: serde_json::Value,
        _permit: &crate::agent::tools::ToolPermit<'_>,
    ) -> ToolResult {
        ToolResult::error("Git tools require 'tools-git' feature to be enabled")
    }
}
