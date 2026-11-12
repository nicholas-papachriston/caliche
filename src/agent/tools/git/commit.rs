#[cfg(feature = "tools-git")]
use crate::agent::tools::common::params;
#[cfg(feature = "tools-git")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-git")]
use async_trait::async_trait;
#[cfg(feature = "tools-git")]
use serde_json::json;

const NAME: &str = "git_commit";
const DESCRIPTION: &str = "Create a git commit with staged changes. Requires files to be staged first (use git add). This tool commits staged changes with the provided message.";

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
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let message = match params::extract_string(&arguments, "message", None) {
            Ok(m) => m,
            Err(e) => return ToolResult::error(e),
        };

        let repo = match super::open_tool_repository(&path_str) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(e),
        };

        // Get signature
        let signature = if let (Some(name), Some(email)) = (
            arguments["author_name"].as_str(),
            arguments["author_email"].as_str(),
        ) {
            match git2::Signature::now(name, email) {
                Ok(sig) => sig,
                Err(e) => {
                    return ToolResult::error(format!("Failed to create signature: {e}"));
                }
            }
        } else {
            match repo.signature() {
                Ok(sig) => sig,
                Err(e) => {
                    return ToolResult::error(format!("Failed to get git signature: {e}"));
                }
            }
        };

        // Get index and find staged changes
        let mut index = match repo.index() {
            Ok(idx) => idx,
            Err(e) => {
                return ToolResult::error(format!("Failed to get index: {e}"));
            }
        };

        if index.is_empty() {
            return ToolResult::error(
                "No staged changes to commit. Use 'git add' to stage files first.".to_string(),
            );
        }

        // Get tree from index
        let tree_id = match index.write_tree() {
            Ok(id) => id,
            Err(e) => {
                return ToolResult::error(format!("Failed to write tree: {e}"));
            }
        };
        let tree = match repo.find_tree(tree_id) {
            Ok(t) => t,
            Err(e) => {
                return ToolResult::error(format!("Failed to find tree: {e}"));
            }
        };

        // Get HEAD commit (if exists)
        let parent_commit = repo
            .head()
            .ok()
            .and_then(|head| head.target().and_then(|oid| repo.find_commit(oid).ok()));

        // Create commit
        let commit_id = match if let Some(parent) = parent_commit {
            repo.commit(
                Some("HEAD"),
                &signature,
                &signature,
                &message,
                &tree,
                &[&parent],
            )
        } else {
            // Initial commit
            repo.commit(Some("HEAD"), &signature, &signature, &message, &tree, &[])
        } {
            Ok(id) => id,
            Err(e) => {
                return ToolResult::error(format!("Failed to create commit: {e}"));
            }
        };

        ToolResult::success(format!(
            "Commit created successfully: {}",
            commit_id.to_string().chars().take(7).collect::<String>()
        ))
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

    async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
        ToolResult::error("Git tools require 'tools-git' feature to be enabled")
    }
}
