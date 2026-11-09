#[cfg(feature = "tools-git")]
use super::{GitBranchTool, GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool};
#[cfg(feature = "tools-git")]
use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use async_trait::async_trait;
use serde_json::json;

const NAME: &str = "git";
const DESCRIPTION: &str = "Unified git tool. Use operation=status|diff|log|branch|commit for repository status, diffs, history, branch management, and commits.";

#[derive(Debug)]
pub struct GitTool;

impl ToolMeta for GitTool {
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
                "operation": {
                    "type": "string",
                    "enum": ["status", "diff", "log", "branch", "commit"],
                    "description": "Git operation to run (default: status)",
                    "default": "status"
                },
                "path": {
                    "type": "string",
                    "description": "Path to the git repository (default: current directory)",
                    "default": "."
                },
                "file": {
                    "type": "string",
                    "description": "File path filter for diff/log operations"
                },
                "staged": {
                    "type": "boolean",
                    "description": "For diff operation: show staged changes",
                    "default": false
                },
                "max_count": {
                    "type": "number",
                    "description": "For log operation: maximum commits to show (default: 10)",
                    "default": 10
                },
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "delete"],
                    "description": "For branch operation: branch action",
                    "default": "list"
                },
                "branch_name": {
                    "type": "string",
                    "description": "For branch operation create/delete: target branch name"
                },
                "message": {
                    "type": "string",
                    "description": "For commit operation: commit message"
                },
                "author_name": {
                    "type": "string",
                    "description": "For commit operation: author name override"
                },
                "author_email": {
                    "type": "string",
                    "description": "For commit operation: author email override"
                }
            },
            "required": []
        })
    }
}

#[cfg(feature = "tools-git")]
#[async_trait]
impl Tool for GitTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let operation = params::extract_string_or(&arguments, "operation", "status");
        let path = params::extract_string_or(&arguments, "path", ".");

        match operation.as_str() {
            "status" => {
                let tool = GitStatusTool;
                tool.execute(json!({ "path": path })).await
            }
            "diff" => {
                let tool = GitDiffTool;
                tool.execute(json!({
                    "path": path,
                    "file": arguments.get("file").and_then(|v| v.as_str()),
                    "staged": params::extract_bool(&arguments, "staged", false)
                }))
                .await
            }
            "log" => {
                let tool = GitLogTool;
                tool.execute(json!({
                    "path": path,
                    "file": arguments.get("file").and_then(|v| v.as_str()),
                    "max_count": params::extract_number_or(&arguments, "max_count", 10)
                }))
                .await
            }
            "branch" => {
                let tool = GitBranchTool;
                tool.execute(json!({
                    "path": path,
                    "action": params::extract_string_or(&arguments, "action", "list"),
                    "branch_name": arguments.get("branch_name").and_then(|v| v.as_str())
                }))
                .await
            }
            "commit" => {
                let tool = GitCommitTool;
                tool.execute(json!({
                    "path": path,
                    "message": arguments.get("message").and_then(|v| v.as_str()),
                    "author_name": arguments.get("author_name").and_then(|v| v.as_str()),
                    "author_email": arguments.get("author_email").and_then(|v| v.as_str())
                }))
                .await
            }
            _ => ToolResult::error(format!(
                "Unknown git operation '{operation}'. Use one of: status, diff, log, branch, commit."
            )),
        }
    }
}

#[cfg(not(feature = "tools-git"))]
#[async_trait]
impl Tool for GitTool {
    async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
        ToolResult::error("Git tools require 'tools-git' feature to be enabled")
    }
}
