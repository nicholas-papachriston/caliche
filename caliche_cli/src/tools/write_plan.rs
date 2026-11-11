//! CLI-only `write_plan` tool.
//!
//! Writes the plan document to PLAN.md. Only available in Plan mode.
//! The path is fixed to PLAN.md to prevent the tool from being abused to create
//! arbitrary documentation files (ENHANCEMENT_PLAN.md, DELIVERABLES.md, etc.).

use async_trait::async_trait;
use caliche::agent::tools::common::params;
use caliche::agent::tools::file_creation_context;
use caliche::agent::tools::tool::ToolMeta;
use caliche::agent::tools::{Tool, ToolResult};
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Tool name for Plan mode.
pub const WRITE_PLAN_TOOL_NAME: &str = "write_plan";

const PLAN_PATH: &str = "PLAN.md";

/// CLI-only tool that writes the plan document to PLAN.md.
/// Only surfaced in Plan mode. The path is always PLAN.md.
#[derive(Debug)]
pub struct WritePlanTool;

impl ToolMeta for WritePlanTool {
    fn name(&self) -> &str {
        WRITE_PLAN_TOOL_NAME
    }

    fn description(&self) -> &str {
        "Write the plan document to PLAN.md. Use ONLY for the final plan in Plan mode. \
         The file is always named PLAN.md — do NOT use this to create other documentation files. \
         For source or config files use fs with op=write instead."
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The plan content in markdown format. Written to PLAN.md."
                }
            },
            "required": ["content"]
        })
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(WRITE_PLAN_TOOL_NAME)
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(
            "Write the plan document to PLAN.md. Use ONLY for the final plan in Plan mode. \
             The file is always named PLAN.md — do NOT use this to create other documentation files. \
             For source or config files use fs with op=write instead.",
        )
    }

    fn is_readonly(&self) -> bool {
        false
    }
}

#[async_trait]
impl Tool for WritePlanTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let content = match params::extract_string(&arguments, "content", None) {
            Ok(c) => c,
            Err(e) => return ToolResult::error(e),
        };

        let file_path = PathBuf::from(PLAN_PATH);

        let is_new = !file_path.exists();

        // Respect the session doc-file creation limit. PLAN.md is a documentation file
        // and counts against the same quota as other markdown files.
        if let Some(msg) = file_creation_context::check_may_create(&file_path, is_new) {
            return ToolResult::error(msg);
        }

        let written = tokio::task::spawn_blocking(move || {
            let workspace = std::env::current_dir()?.canonicalize()?;
            write_plan_beneath(&workspace, &content)
        })
        .await
        .unwrap_or_else(|join| Err(std::io::Error::other(join.to_string())));
        match written {
            Ok(()) => {
                if is_new {
                    file_creation_context::record_created(file_path);
                }
                ToolResult::success(format!("Plan written to '{}'", PLAN_PATH))
            }
            Err(e) => ToolResult::error(format!("Failed to write plan to '{}': {}", PLAN_PATH, e)),
        }
    }
}

/// Write `content` to PLAN.md directly in the canonical `workspace` (blocking).
///
/// The file is opened below the workspace without following any link, so a
/// PLAN.md symlink (or a hard link to an outside file) planted by a command
/// in the workspace is refused instead of written through.
fn write_plan_beneath(workspace: &Path, content: &str) -> std::io::Result<()> {
    let mut file = caliche_sandbox::create_file_beneath(workspace, &workspace.join(PLAN_PATH))?;
    file.write_all(content.as_bytes())
}

#[cfg(all(test, unix))]
mod tests {
    use super::write_plan_beneath;
    use std::path::PathBuf;

    fn workspace() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path().canonicalize().expect("canonical workspace");
        (dir, root)
    }

    #[test]
    fn the_plan_is_written_to_plan_md() {
        let (_dir, root) = workspace();
        std::fs::write(root.join("PLAN.md"), "old plan that is longer").expect("old plan");

        write_plan_beneath(&root, "# Plan\n").expect("write plan");

        assert_eq!(
            std::fs::read_to_string(root.join("PLAN.md")).expect("read plan"),
            "# Plan\n"
        );
    }

    #[test]
    fn a_plan_md_symlink_is_refused_and_the_outside_file_is_unchanged() {
        let (_dir, root) = workspace();
        let outside = tempfile::tempdir().expect("outside");
        let outside_file = outside.path().join("victim.txt");
        std::fs::write(&outside_file, "outside").expect("outside file");
        std::os::unix::fs::symlink(&outside_file, root.join("PLAN.md")).expect("symlink");

        let err = write_plan_beneath(&root, "# Plan\n").expect_err("a symlinked PLAN.md");

        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            err.to_string(),
            format!(
                "refusing to follow symlink at 'PLAN.md' in '{}'",
                root.join("PLAN.md").display()
            )
        );
        assert_eq!(
            std::fs::read_to_string(&outside_file).expect("read outside"),
            "outside"
        );
    }

    #[test]
    fn a_plan_md_hard_linked_to_an_outside_file_is_refused() {
        let (_dir, root) = workspace();
        let outside = tempfile::tempdir().expect("outside");
        let outside_file = outside.path().join("victim.txt");
        std::fs::write(&outside_file, "outside").expect("outside file");
        std::fs::hard_link(&outside_file, root.join("PLAN.md")).expect("hard link");

        let err = write_plan_beneath(&root, "# Plan\n").expect_err("a hard-linked PLAN.md");

        assert_eq!(err.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(
            err.to_string(),
            format!(
                "refusing to write '{}': {}",
                root.join("PLAN.md").display(),
                caliche_sandbox::HARD_LINK_REFUSAL
            )
        );
        assert_eq!(
            std::fs::read_to_string(&outside_file).expect("read outside"),
            "outside"
        );
    }
}
