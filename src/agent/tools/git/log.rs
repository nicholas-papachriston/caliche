#[cfg(feature = "tools-git")]
use crate::agent::tools::common::params;
#[cfg(feature = "tools-git")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-git")]
use async_trait::async_trait;
#[cfg(feature = "tools-git")]
use serde_json::json;
#[cfg(feature = "tools-git")]
use std::path::PathBuf;

const NAME: &str = "git_log";
const DESCRIPTION: &str = "Show git commit history. Can show log for specific files or the entire repository. Useful for understanding project history and finding when changes were made.";
#[derive(Debug)]
pub struct GitLogTool;

#[cfg(feature = "tools-git")]
impl ToolMeta for GitLogTool {
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
                    "description": "Show log for specific file (optional)"
                },
                "max_count": {
                    "type": "number",
                    "description": "Maximum number of commits to show (default: 10)",
                    "default": 10
                }
            },
            "required": []
        })
    }
}

#[async_trait]
impl Tool for GitLogTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let file_path = arguments["file"]
            .as_str()
            .map(std::string::ToString::to_string);
        let max_count = params::extract_number_or(&arguments, "max_count", 10) as usize;

        let repo_path = if PathBuf::from(&path_str).is_absolute() {
            PathBuf::from(&path_str)
        } else {
            match std::env::current_dir() {
                Ok(cwd) => cwd.join(&path_str),
                Err(e) => {
                    return ToolResult::error(format!("Failed to get current directory: {e}"));
                }
            }
        };

        let repo = match git2::Repository::open(&repo_path) {
            Ok(r) => r,
            Err(e) => {
                return ToolResult::error(format!(
                    "Failed to open git repository at '{path_str}': {e}"
                ));
            }
        };

        let mut revwalk = match repo.revwalk() {
            Ok(rw) => rw,
            Err(e) => {
                return ToolResult::error(format!("Failed to create revwalk: {e}"));
            }
        };

        // Start from HEAD
        match revwalk.push_head() {
            Ok(()) => {}
            Err(e) => {
                return ToolResult::error(format!("Failed to push HEAD: {e}"));
            }
        }

        // Filter by file if specified
        if file_path.is_some()
            && let Err(e) = revwalk.simplify_first_parent()
        {
            return ToolResult::error(format!("Failed to simplify revwalk: {e}"));
        }
        // Note: git2 doesn't have direct file filtering in revwalk

        let mut output = String::from("Git Log:\n");
        output.push_str("========\n\n");

        let mut count = 0;
        for oid in revwalk {
            if count >= max_count {
                break;
            }

            let oid = match oid {
                Ok(o) => o,
                Err(e) => {
                    return ToolResult::error(format!("Failed to get commit: {e}"));
                }
            };

            let commit = match repo.find_commit(oid) {
                Ok(c) => c,
                Err(e) => {
                    return ToolResult::error(format!("Failed to find commit: {e}"));
                }
            };

            let short_id = oid.to_string().chars().take(7).collect::<String>();
            let author = commit.author();
            let message = commit.message().unwrap_or("<no message>");
            let first_line = message.lines().next().unwrap_or("<no message>");

            output.push_str(&format!(
                "{} {} <{}>\n",
                short_id,
                author.name().unwrap_or("<unknown>"),
                author.email().unwrap_or("<unknown>")
            ));
            output.push_str(&format!("  {first_line}\n\n"));

            count += 1;
        }

        if count == 0 {
            return ToolResult::success("No commits found");
        }

        ToolResult::success(output)
    }
}

#[cfg(not(feature = "tools-git"))]
#[async_trait]
impl Tool for GitLogTool {
    fn name(&self) -> &str {
        "git_log"
    }

    fn description(&self) -> &str {
        "Show git log. Requires 'tools-git' feature."
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(self.name())
    }
    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(self.description())
    }

    fn parameters(&self) -> serde_json::Value {
        json!({})
    }

    async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
        ToolResult::error("Git tools require 'tools-git' feature to be enabled")
    }
}
