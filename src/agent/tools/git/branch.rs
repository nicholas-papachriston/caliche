#[cfg(feature = "tools-git")]
use crate::agent::tools::common::params;
#[cfg(feature = "tools-git")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-git")]
use async_trait::async_trait;
#[cfg(feature = "tools-git")]
use serde_json::json;

const NAME: &str = "git_branch";
const DESCRIPTION: &str = "List, create, or delete git branches. Delete keeps a branch whose commits are not merged. Create and delete run only on macOS, where the sandbox can limit git's writes to the repository's refs and logs; on Linux and Windows they refuse.";
#[derive(Debug)]
pub struct GitBranchTool;

#[cfg(feature = "tools-git")]
impl ToolMeta for GitBranchTool {
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
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "delete"],
                    "description": "Action to perform: list branches, create new branch, or delete branch (default: list)",
                    "default": "list"
                },
                "branch_name": {
                    "type": "string",
                    "description": "Branch name (required for create/delete actions)"
                }
            },
            "required": []
        })
    }
}

#[async_trait]
impl Tool for GitBranchTool {
    async fn execute(
        &self,
        arguments: serde_json::Value,
        _permit: &crate::agent::tools::ToolPermit<'_>,
    ) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let action = params::extract_string_or(&arguments, "action", "list");

        if action == "list" {
            return list_branches(&path_str).await;
        }
        let repo = match super::tool_repository(&path_str) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(e),
        };

        match action.as_str() {
            "create" => {
                let branch_name = match params::extract_string(&arguments, "branch_name", None) {
                    Ok(n) => n,
                    Err(e) => return ToolResult::error(e),
                };
                if let Err(e) = repo.write(&["branch", "--", &branch_name], &[]).await {
                    return ToolResult::error(format!("Failed to create branch: {e}"));
                }
                ToolResult::success(format!("Branch '{branch_name}' created successfully"))
            }
            "delete" => {
                let branch_name = match params::extract_string(&arguments, "branch_name", None) {
                    Ok(n) => n,
                    Err(e) => return ToolResult::error(e),
                };
                // -d, not -D: a branch whose commits are not merged is kept.
                if let Err(e) = repo.write(&["branch", "-d", "--", &branch_name], &[]).await {
                    return ToolResult::error(format!("Failed to delete branch: {e}"));
                }
                ToolResult::success(format!("Branch '{branch_name}' deleted successfully"))
            }
            _ => ToolResult::error(format!("Unknown action: {action}")),
        }
    }
}

/// Local and remote-tracking branches, the current one marked `*`.
#[cfg(feature = "tools-git")]
async fn list_branches(path_str: &str) -> ToolResult {
    let repo = match super::tool_repository(path_str) {
        Ok(r) => r,
        Err(e) => return ToolResult::error(e),
    };
    let listed = match repo
        .read(&[
            "for-each-ref",
            "--format=%(HEAD)%(refname:short)",
            "refs/heads/",
            "refs/remotes/",
        ])
        .await
    {
        Ok(listed) => listed,
        Err(e) => return ToolResult::error(format!("Failed to list branches: {e}")),
    };
    let mut output = String::from("Branches:\n========\n\n");
    for line in listed.lines() {
        let (head, name) = line.split_at(line.len().min(1));
        let prefix = if head == "*" { "* " } else { "  " };
        output.push_str(&format!("{prefix}{name}\n"));
    }
    ToolResult::success(output)
}

#[cfg(not(feature = "tools-git"))]
#[async_trait]
impl Tool for GitBranchTool {
    fn name(&self) -> &str {
        "git_branch"
    }

    fn description(&self) -> &str {
        "Manage git branches. Requires 'tools-git' feature."
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("git_branch")
    }
    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("Manage git branches. Requires 'tools-git' feature.")
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
