#[cfg(feature = "tools-git")]
use crate::agent::tools::common::params;
#[cfg(feature = "tools-git")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-git")]
use async_trait::async_trait;
#[cfg(feature = "tools-git")]
use serde_json::json;

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
    async fn execute(
        &self,
        arguments: serde_json::Value,
        _permit: &crate::agent::tools::ToolPermit<'_>,
    ) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let file_path = arguments["file"]
            .as_str()
            .map(std::string::ToString::to_string);
        let max_count = params::extract_number_or(&arguments, "max_count", 10) as usize;

        let repo = match super::tool_repository(&path_str) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(e),
        };
        let max = format!("--max-count={max_count}");
        let mut args = vec!["log", max.as_str(), LOG_FORMAT];
        if let Some(file) = &file_path {
            args.extend(["--", file.as_str()]);
        }
        let listed = match repo.read(&args).await {
            Ok(listed) => listed,
            Err(e) => return ToolResult::error(format!("Failed to read git log: {e}")),
        };
        let output = format_log(&listed);
        if output.is_empty() {
            return ToolResult::success("No commits found");
        }
        ToolResult::success(format!("Git Log:\n========\n\n{output}"))
    }
}

/// One record per commit: full id, author name, author email and message,
/// split by unit separators, each record ended by a record separator.
#[cfg(feature = "tools-git")]
const LOG_FORMAT: &str = "--format=%H%x1f%an%x1f%ae%x1f%B%x1e";

/// `{short id} {name} <{email}>` and the message's first line, per commit.
#[cfg(feature = "tools-git")]
fn format_log(listed: &str) -> String {
    let mut output = String::new();
    for record in listed.split('\x1e') {
        let mut fields = record.trim_start_matches('\n').split('\x1f');
        let (Some(id), Some(name), Some(email), Some(message)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let short_id: String = id.chars().take(7).collect();
        let first_line = message.lines().next().unwrap_or("<no message>");
        output.push_str(&format!("{short_id} {name} <{email}>\n  {first_line}\n\n"));
    }
    output
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

    async fn execute(
        &self,
        _arguments: serde_json::Value,
        _permit: &crate::agent::tools::ToolPermit<'_>,
    ) -> ToolResult {
        ToolResult::error("Git tools require 'tools-git' feature to be enabled")
    }
}
