#[cfg(feature = "tools-git")]
use crate::agent::tools::common::params;
#[cfg(feature = "tools-git")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-git")]
use async_trait::async_trait;
#[cfg(feature = "tools-git")]
use serde_json::json;

const NAME: &str = "git_diff";
const DESCRIPTION: &str = "Show the differences in a git repository. Can show diff for specific files, staged changes, or working directory changes. Useful for reviewing changes before committing.";
#[derive(Debug)]
pub struct GitDiffTool;

#[cfg(feature = "tools-git")]
impl ToolMeta for GitDiffTool {
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
                "file": {
                    "type": "string",
                    "description": "Specific file to show diff for (optional). If not provided, shows diff for all changed files."
                },
                "staged": {
                    "type": "boolean",
                    "description": "Show staged changes instead of working directory changes (default: false)",
                    "default": false
                }
            },
            "required": []
        })
    }
}

#[async_trait]
impl Tool for GitDiffTool {
    async fn execute(
        &self,
        arguments: serde_json::Value,
        _permit: &crate::agent::tools::ToolPermit<'_>,
    ) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let file_path = arguments["file"]
            .as_str()
            .map(std::string::ToString::to_string);
        let staged = params::extract_bool(&arguments, "staged", false);

        let repo = match super::tool_repository(&path_str) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(e),
        };
        let mut args = vec!["diff"];
        if staged {
            args.push("--cached");
        }
        args.extend(crate::policy::GIT_DIFF_SAFETY_FLAGS);
        if let Some(file) = &file_path {
            args.extend(["--", file.as_str()]);
        }
        let patch = match repo.read(&args).await {
            Ok(patch) => patch,
            Err(e) => return ToolResult::error(format!("Failed to get git diff: {e}")),
        };
        if patch.is_empty() {
            return ToolResult::success("No changes found");
        }
        let title = if staged {
            "Staged changes:"
        } else {
            "Working directory changes:"
        };
        ToolResult::success(format!("{title}\n==================\n\n{patch}"))
    }
}

#[cfg(not(feature = "tools-git"))]
#[async_trait]
impl Tool for GitDiffTool {
    fn name(&self) -> &str {
        "git_diff"
    }

    fn description(&self) -> &str {
        "Show git diff. Requires 'tools-git' feature."
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("git_diff")
    }
    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("Show git diff. Requires 'tools-git' feature.")
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
