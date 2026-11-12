#[cfg(feature = "tools-git")]
use crate::agent::tools::common::params;
#[cfg(feature = "tools-git")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-git")]
use async_trait::async_trait;
#[cfg(feature = "tools-git")]
use serde_json::json;

const NAME: &str = "git_status";
const DESCRIPTION: &str = "Get the status of a git repository. Shows modified, staged, untracked, and deleted files. Useful for understanding what changes have been made before committing.";
#[derive(Debug)]
pub struct GitStatusTool;

#[cfg(feature = "tools-git")]
impl ToolMeta for GitStatusTool {
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
                }
            },
            "required": []
        })
    }
}

#[cfg(feature = "tools-git")]
struct StatusLists {
    staged: Vec<String>,
    modified: Vec<String>,
    untracked: Vec<String>,
    deleted: Vec<String>,
}

#[cfg(feature = "tools-git")]
fn collect_status_lists(statuses: &git2::Statuses<'_>) -> StatusLists {
    let mut lists = StatusLists {
        staged: Vec::new(),
        modified: Vec::new(),
        untracked: Vec::new(),
        deleted: Vec::new(),
    };
    for entry in statuses.iter() {
        classify_status_entry(&mut lists, entry);
    }
    lists
}

#[cfg(feature = "tools-git")]
fn classify_status_entry(lists: &mut StatusLists, entry: git2::StatusEntry<'_>) {
    let status = entry.status();
    let path = entry.path().unwrap_or("<unknown>");
    if status.is_index_new() || status.is_index_modified() || status.is_index_deleted() {
        lists.staged.push(path.to_string());
    }
    if status.is_wt_modified() {
        lists.modified.push(path.to_string());
    }
    if status.is_wt_new() {
        lists.untracked.push(path.to_string());
    }
    if status.is_wt_deleted() {
        lists.deleted.push(path.to_string());
    }
}

#[cfg(feature = "tools-git")]
fn append_status_section(output: &mut String, title: &str, prefix: &str, files: &[String]) {
    if files.is_empty() {
        return;
    }
    output.push_str(title);
    output.push('\n');
    for file in files {
        output.push_str(&format!("  {prefix} {file}\n"));
    }
    output.push('\n');
}

#[cfg(feature = "tools-git")]
fn format_status_output(lists: &StatusLists) -> String {
    let mut output = String::from("Git Status:\n");
    output.push_str("==========\n\n");
    append_status_section(&mut output, "Staged changes:", "A", &lists.staged);
    append_status_section(&mut output, "Modified files:", "M", &lists.modified);
    append_status_section(&mut output, "Untracked files:", "?", &lists.untracked);
    append_status_section(&mut output, "Deleted files:", "D", &lists.deleted);
    output.trim().to_string()
}

#[async_trait]
impl Tool for GitStatusTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let repo = match super::open_tool_repository(&path_str) {
            Ok(r) => r,
            Err(e) => return ToolResult::error(e),
        };

        let mut status_opts = git2::StatusOptions::new();
        status_opts.include_untracked(true);
        status_opts.include_ignored(false);

        let statuses = match repo.statuses(Some(&mut status_opts)) {
            Ok(s) => s,
            Err(e) => {
                return ToolResult::error(format!("Failed to get git status: {e}"));
            }
        };

        if statuses.is_empty() {
            return ToolResult::success("Working tree clean - no changes");
        }

        ToolResult::success(format_status_output(&collect_status_lists(&statuses)))
    }
}

#[cfg(not(feature = "tools-git"))]
#[async_trait]
impl Tool for GitStatusTool {
    fn name(&self) -> &str {
        "git_status"
    }

    fn description(&self) -> &str {
        "Get git repository status. Requires 'tools-git' feature."
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("git_status")
    }
    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed("Get git repository status. Requires 'tools-git' feature.")
    }

    fn parameters(&self) -> serde_json::Value {
        json!({})
    }

    async fn execute(&self, _arguments: serde_json::Value) -> ToolResult {
        ToolResult::error("Git tools require 'tools-git' feature to be enabled")
    }
}
