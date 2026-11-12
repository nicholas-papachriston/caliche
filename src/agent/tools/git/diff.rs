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
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let file_path = arguments["file"]
            .as_str()
            .map(std::string::ToString::to_string);
        let staged = params::extract_bool(&arguments, "staged", false);

        let repo = match super::open_tool_repository(&path_str) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(e),
        };

        let mut diff = if staged {
            // Diff between HEAD and index (staged changes)
            let head = match repo.head() {
                Ok(h) => h.target(),
                Err(_) => None,
            };
            let tree = head.and_then(|oid| repo.find_tree(oid).ok());
            let index = match repo.index() {
                Ok(idx) => idx,
                Err(e) => {
                    return ToolResult::error(format!("Failed to get index: {e}"));
                }
            };
            match repo.diff_tree_to_index(tree.as_ref(), Some(&index), None) {
                Ok(d) => d,
                Err(e) => {
                    return ToolResult::error(format!("Failed to get staged diff: {e}"));
                }
            }
        } else {
            // Diff between index and working directory (unstaged changes)
            let index = match repo.index() {
                Ok(idx) => idx,
                Err(e) => {
                    return ToolResult::error(format!("Failed to get index: {e}"));
                }
            };
            match repo.diff_index_to_workdir(Some(&index), None) {
                Ok(d) => d,
                Err(e) => {
                    return ToolResult::error(format!("Failed to get working directory diff: {e}"));
                }
            }
        };

        // Filter by file if specified
        if file_path.is_some()
            && let Err(e) = diff.find_similar(None)
        {
            return ToolResult::error(format!("Failed to find similar files in diff: {e}"));
        }
        // Note: git2 doesn't have direct file filtering, so we'll filter in output

        let mut output = String::new();
        if staged {
            output.push_str("Staged changes:\n");
        } else {
            output.push_str("Working directory changes:\n");
        }
        output.push_str("==================\n\n");

        let mut has_changes = false;
        match diff.print(git2::DiffFormat::Patch, |delta, _hunk, line| {
            has_changes = true;
            let line_str = std::str::from_utf8(line.content()).unwrap_or("");

            // Filter by file if specified
            if let Some(ref file) = file_path
                && let Some(path) = delta.new_file().path()
                && !path.to_string_lossy().contains(file)
            {
                return true; // Continue
            }

            output.push_str(line_str);
            true
        }) {
            Ok(()) => {}
            Err(e) => {
                return ToolResult::error(format!("Failed to print diff: {e}"));
            }
        }

        if !has_changes {
            return ToolResult::success("No changes found");
        }

        ToolResult::success(output)
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

    async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
        ToolResult::error("Git tools require 'tools-git' feature to be enabled")
    }
}
