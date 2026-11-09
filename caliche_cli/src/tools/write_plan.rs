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
use std::path::PathBuf;
use tokio::fs;

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

        match fs::write(&file_path, &content).await {
            Ok(_) => {
                if is_new {
                    file_creation_context::record_created(file_path);
                }
                ToolResult::success(format!("Plan written to '{}'", PLAN_PATH))
            }
            Err(e) => ToolResult::error(format!("Failed to write plan to '{}': {}", PLAN_PATH, e)),
        }
    }
}
