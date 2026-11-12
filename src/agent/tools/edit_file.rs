use crate::agent::tools::common::{errors, params, path};
use crate::agent::tools::edit_args::{
    coerce_operations, line_or_anchor, require_hash, require_line,
};
use crate::agent::tools::hashline;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::json;
use std::path::Path;

/// Validate 1-indexed line range; returns error message if invalid.
fn validate_line_range(start: usize, end: usize, max_lines: usize, op_idx: usize) -> Result<()> {
    if start == 0 || end == 0 {
        bail!("Operation {op_idx}: Line numbers must be 1-indexed (start: {start}, end: {end})");
    }
    if start > end {
        bail!("Operation {op_idx}: start_line ({start}) must be <= end_line ({end})");
    }
    if start > max_lines {
        bail!("Operation {op_idx}: start_line ({start}) exceeds file length ({max_lines})");
    }
    Ok(())
}

/// Surface the full anyhow chain. `Display` only prints the outer
/// `Operation N` context and hides the hash mismatch the model must fix.
fn format_edit_error(error: &anyhow::Error) -> String {
    format!("{error:#}")
}

/// Apply one edit operation; returns error string on validation/apply failure.
fn apply_operation(
    content: &mut String,
    lines: &mut Vec<String>,
    changes_applied: &mut u32,
    op: &serde_json::Value,
    idx: usize,
) -> Result<()> {
    let op_type = op["type"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Operation {}: missing 'type' field", idx + 1))?;
    match op_type {
        "str_replace" => apply_str_replace(content, changes_applied, op, idx),
        "insert" => apply_insert(lines, changes_applied, op, idx),
        "delete_lines" => apply_delete_lines(lines, changes_applied, op, idx),
        "replace_lines" => apply_replace_lines(lines, changes_applied, op, idx),
        "hashline_replace" => apply_hashline_replace(lines, changes_applied, op, idx),
        "hashline_insert" => apply_hashline_insert(lines, changes_applied, op, idx),
        "hashline_delete" => apply_hashline_delete(lines, changes_applied, op, idx),
        _ => bail!(
            "Operation {}: Unknown operation type '{}'",
            idx + 1,
            op_type
        ),
    }
}

fn count_non_overlapping(haystack: &str, needle: &str) -> usize {
    if needle.is_empty() {
        return 0;
    }
    let mut count = 0usize;
    let mut start = 0usize;
    while let Some(pos) = haystack[start..].find(needle) {
        count += 1;
        start += pos + needle.len();
    }
    count
}

fn apply_str_replace(
    content: &mut String,
    changes_applied: &mut u32,
    op: &serde_json::Value,
    idx: usize,
) -> Result<()> {
    let old_str = op["old_string"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Operation {}: 'old_string' must be a string", idx + 1))?;
    let new_str = op["new_string"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Operation {}: 'new_string' must be a string", idx + 1))?;
    if old_str.is_empty() {
        bail!("Operation {}: 'old_string' must not be empty", idx + 1);
    }
    let matches = count_non_overlapping(content, old_str);
    if matches == 0 {
        bail!(
            "Operation {}: old_string not found. Provide more context or re-read the file.",
            idx + 1
        );
    }
    if matches > 1 {
        bail!(
            "Operation {}: found {} occurrences of old_string; require a unique match. \
             Provide more surrounding context.",
            idx + 1,
            matches
        );
    }
    *content = content.replacen(old_str, new_str, 1);
    *changes_applied += 1;
    Ok(())
}

/// Apply all `str_replace` ops against the original content (pi-style), then other ops sequentially.
///
/// Matching against the original avoids chained-offset drift; replacements are applied from
/// last occurrence to first so earlier offsets remain valid.
fn apply_str_replaces_against_original(
    original: &str,
    operations: &[serde_json::Value],
) -> Result<(String, u32, Vec<usize>)> {
    #[derive(Clone, Copy)]
    struct PlannedReplace<'a> {
        op_idx: usize,
        start: usize,
        old_str: &'a str,
        new_str: &'a str,
    }

    let mut planned: Vec<PlannedReplace<'_>> = Vec::new();
    let mut non_replace_indices = Vec::new();

    for (idx, op) in operations.iter().enumerate() {
        let op_type = op["type"].as_str().unwrap_or("");
        if op_type != "str_replace" {
            non_replace_indices.push(idx);
            continue;
        }
        let old_str = op["old_string"].as_str().ok_or_else(|| {
            anyhow::anyhow!("Operation {}: 'old_string' must be a string", idx + 1)
        })?;
        let new_str = op["new_string"].as_str().ok_or_else(|| {
            anyhow::anyhow!("Operation {}: 'new_string' must be a string", idx + 1)
        })?;
        if old_str.is_empty() {
            bail!("Operation {}: 'old_string' must not be empty", idx + 1);
        }
        let matches = count_non_overlapping(original, old_str);
        if matches == 0 {
            bail!(
                "Operation {}: old_string not found in original file content. \
                 All str_replace ops match the original; provide more context or re-read.",
                idx + 1
            );
        }
        if matches > 1 {
            bail!(
                "Operation {}: found {} occurrences of old_string in original; require unique match.",
                idx + 1,
                matches
            );
        }
        let Some(start) = original.find(old_str) else {
            bail!(
                "Operation {}: old_string vanished between count and find",
                idx + 1
            );
        };
        let end = start + old_str.len();
        for prev in &planned {
            let prev_end = prev.start + prev.old_str.len();
            let overlaps = start < prev_end && prev.start < end;
            if overlaps {
                bail!(
                    "Operation {}: str_replace overlaps operation {}. \
                     Edits must be disjoint when applied against the original.",
                    idx + 1,
                    prev.op_idx + 1
                );
            }
        }
        planned.push(PlannedReplace {
            op_idx: idx,
            start,
            old_str,
            new_str,
        });
    }

    planned.sort_by(|a, b| b.start.cmp(&a.start));
    let mut content = original.to_string();
    let mut changes = 0u32;
    for plan in planned {
        let end = plan.start + plan.old_str.len();
        if content.get(plan.start..end) != Some(plan.old_str) {
            bail!(
                "Operation {}: original match drifted while applying edits",
                plan.op_idx + 1
            );
        }
        content.replace_range(plan.start..end, plan.new_str);
        changes += 1;
    }
    Ok((content, changes, non_replace_indices))
}

fn apply_insert(
    lines: &mut Vec<String>,
    changes_applied: &mut u32,
    op: &serde_json::Value,
    idx: usize,
) -> Result<()> {
    let line_num = require_line(op, "line", idx)?;
    let insert_content = op["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Operation {}: 'content' must be a string", idx + 1))?;
    let insert_pos = if line_num == 0 {
        0
    } else if line_num > lines.len() {
        lines.len()
    } else {
        line_num - 1
    };
    let insert_lines: Vec<String> = insert_content
        .lines()
        .map(std::string::ToString::to_string)
        .collect();
    for (i, line) in insert_lines.iter().enumerate() {
        lines.insert(insert_pos + i, line.clone());
    }
    *changes_applied += 1;
    Ok(())
}

fn apply_delete_lines(
    lines: &mut Vec<String>,
    changes_applied: &mut u32,
    op: &serde_json::Value,
    idx: usize,
) -> Result<()> {
    let start = require_line(op, "start_line", idx)?;
    let end = require_line(op, "end_line", idx)?;
    validate_line_range(start, end, lines.len(), idx + 1)?;
    let end_pos = end.min(lines.len());
    lines.drain((start - 1)..end_pos);
    *changes_applied += 1;
    Ok(())
}

fn apply_replace_lines(
    lines: &mut Vec<String>,
    changes_applied: &mut u32,
    op: &serde_json::Value,
    idx: usize,
) -> Result<()> {
    let start = require_line(op, "start_line", idx)?;
    let end = require_line(op, "end_line", idx)?;
    let replace_content = op["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Operation {}: 'content' must be a string", idx + 1))?;
    validate_line_range(start, end, lines.len(), idx + 1)?;
    let end_pos = end.min(lines.len());
    let replace_lines: Vec<String> = replace_content
        .lines()
        .map(std::string::ToString::to_string)
        .collect();
    lines.splice((start - 1)..end_pos, replace_lines);
    *changes_applied += 1;
    Ok(())
}

fn apply_hashline_replace(
    lines: &mut Vec<String>,
    changes_applied: &mut u32,
    op: &serde_json::Value,
    idx: usize,
) -> Result<()> {
    let start = line_or_anchor(op, "start_line", "start_hash", idx)?;
    let start_hash = require_hash(op, "start_hash", idx)?;
    let end = line_or_anchor(op, "end_line", "end_hash", idx)?;
    let end_hash = require_hash(op, "end_hash", idx)?;
    let replace_content = op["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Operation {}: 'content' must be a string", idx + 1))?;
    let (start, end) = hashline::locate_line_range(lines, start, start_hash, end, end_hash)
        .with_context(|| format!("Operation {}", idx + 1))?;
    if start > end {
        bail!(
            "Operation {}: start_line ({}) must be <= end_line ({})",
            idx + 1,
            start,
            end
        );
    }
    let end_pos = end.min(lines.len());
    let replace_lines: Vec<String> = replace_content
        .lines()
        .map(std::string::ToString::to_string)
        .collect();
    lines.splice((start - 1)..end_pos, replace_lines);
    *changes_applied += 1;
    Ok(())
}

fn apply_hashline_insert(
    lines: &mut Vec<String>,
    changes_applied: &mut u32,
    op: &serde_json::Value,
    idx: usize,
) -> Result<()> {
    let anchor_line = line_or_anchor(op, "anchor_line", "anchor_hash", idx)?;
    let anchor_hash = require_hash(op, "anchor_hash", idx)?;
    let position = op["position"].as_str().unwrap_or("after");
    let insert_content = op["content"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Operation {}: 'content' must be a string", idx + 1))?;
    if anchor_line == 0 {
        bail!("Operation {}: anchor_line must be 1-indexed", idx + 1);
    }
    let anchor_line = hashline::locate_line_by_hash(lines, anchor_line, anchor_hash)
        .with_context(|| format!("Operation {}", idx + 1))?;
    let insert_pos = if position == "before" {
        anchor_line.saturating_sub(1)
    } else {
        anchor_line.min(lines.len())
    };
    let insert_lines: Vec<String> = insert_content
        .lines()
        .map(std::string::ToString::to_string)
        .collect();
    for (i, line) in insert_lines.iter().enumerate() {
        lines.insert(insert_pos + i, line.clone());
    }
    *changes_applied += 1;
    Ok(())
}

fn apply_hashline_delete(
    lines: &mut Vec<String>,
    changes_applied: &mut u32,
    op: &serde_json::Value,
    idx: usize,
) -> Result<()> {
    let start = line_or_anchor(op, "start_line", "start_hash", idx)?;
    let start_hash = require_hash(op, "start_hash", idx)?;
    let end = line_or_anchor(op, "end_line", "end_hash", idx)?;
    let end_hash = require_hash(op, "end_hash", idx)?;
    let (start, end) = hashline::locate_line_range(lines, start, start_hash, end, end_hash)
        .with_context(|| format!("Operation {}", idx + 1))?;
    if start > end {
        bail!(
            "Operation {}: start_line ({}) must be <= end_line ({})",
            idx + 1,
            start,
            end
        );
    }
    validate_line_range(start, end, lines.len(), idx + 1)?;
    let end_pos = end.min(lines.len());
    lines.drain((start - 1)..end_pos);
    *changes_applied += 1;
    Ok(())
}

const NAME: &str = "fs_edit_internal";
const DESCRIPTION: &str = "Edit a file by applying text replacements, insertions, or deletions. \
Supports multiple edit operations in a single call. \
\
OPERATIONS:\n\
- str_replace: Replace a unique occurrence of a string (fails if 0 or >1 matches; multi-edit matches original)\n\
- insert: Insert text at a specific line number\n\
- delete_lines: Delete lines by line number range\n\
- replace_lines: Replace lines by line number range\n\
- hashline_replace: Replace line range using line:hash anchors from fs op=read (validates file unchanged)\n\
- hashline_insert: Insert before/after a line using line:hash anchor from fs op=read\n\
- hashline_delete: Delete line range using line:hash anchors from fs op=read\n\
\
HASHLINE OPERATIONS: fs op=read outputs lines as '42:ab12| content'. Use start_line=42, start_hash='ab12' \
(or start_hash='42:ab12'). The hash token must be exactly 4 hex digits (or exactly 2 for a legacy hash). \
The hash ignores trailing whitespace and CR. If the strict hash misses, one retry also ignores leading \
whitespace and accepts the hinted line or a unique 4-hex match. Two whitespace-equivalent lines are refused. \
A legacy 2-hex hash is checked only at the hinted line and is never relocated. \
Duplicate hashes are not moved to a nearby collision; the error lists candidate lines.\n\
\
SECURITY:\n\
- Path traversal ('..') is blocked\n\
- File must exist (use fs with op=write to create new files)\n\
\
USAGE:\n\
- Use 'path' to specify the file to edit\n\
- Use 'operations' array to specify one or more edit operations\n\
- Operations are applied in order\n\
\
EXAMPLES:\n\
- Replace function name (str_replace or hashline_replace)\n\
- Insert import at top of file (insert or hashline_insert)\n\
- Delete obsolete code block (delete_lines or hashline_delete)\n\
- Use hashline ops when fs op=read showed line:hash tags for more reliable edits";

#[derive(Debug)]
pub struct EditFileTool;

impl ToolMeta for EditFileTool {
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
                    "description": "The path to the file to edit. File must exist."
                },
                "operations": {
                    "type": "array",
                    "description": "Array of edit operations to apply. Operations are applied in order.",
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["str_replace"]
                                    },
                                    "old_string": {
                                        "type": "string",
                                        "description": "The string to replace"
                                    },
                                    "new_string": {
                                        "type": "string",
                                        "description": "The replacement string"
                                    }
                                },
                                "required": ["type", "old_string", "new_string"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["insert"]
                                    },
                                    "line": {
                                        "type": "number",
                                        "description": "Line number to insert at (1-indexed). Use 0 to insert at beginning."
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content to insert"
                                    }
                                },
                                "required": ["type", "line", "content"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["delete_lines"]
                                    },
                                    "start_line": {
                                        "type": "number",
                                        "description": "Start line number (1-indexed, inclusive)"
                                    },
                                    "end_line": {
                                        "type": "number",
                                        "description": "End line number (1-indexed, inclusive)"
                                    }
                                },
                                "required": ["type", "start_line", "end_line"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["replace_lines"]
                                    },
                                    "start_line": {
                                        "type": "number",
                                        "description": "Start line number (1-indexed, inclusive)"
                                    },
                                    "end_line": {
                                        "type": "number",
                                        "description": "End line number (1-indexed, inclusive)"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Replacement content"
                                    }
                                },
                                "required": ["type", "start_line", "end_line", "content"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["hashline_replace"]
                                    },
                                    "start_line": {
                                        "type": "number",
                                        "description": "Start line number (1-indexed, inclusive). Must match hash from fs op=read output."
                                    },
                                    "start_hash": {
                                        "type": "string",
                                        "description": "4-hex hash (legacy 2-hex only at this hinted line; not relocated) for start line (from fs op=read line:hash format)"
                                    },
                                    "end_line": {
                                        "type": "number",
                                        "description": "End line number (1-indexed, inclusive). Must match hash from fs op=read output."
                                    },
                                    "end_hash": {
                                        "type": "string",
                                        "description": "4-hex hash (legacy 2-hex only at this hinted line; not relocated) for end line (from fs op=read line:hash format)"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Replacement content"
                                    }
                                },
                                "required": ["type", "start_line", "start_hash", "end_line", "end_hash", "content"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["hashline_insert"]
                                    },
                                    "anchor_line": {
                                        "type": "number",
                                        "description": "Line number to insert before/after (1-indexed). Must match hash from fs op=read output."
                                    },
                                    "anchor_hash": {
                                        "type": "string",
                                        "description": "4-hex hash (legacy 2-hex only at this hinted line; not relocated) for anchor line (from fs op=read line:hash format)"
                                    },
                                    "position": {
                                        "type": "string",
                                        "enum": ["before", "after"],
                                        "description": "Insert before or after the anchor line (default: after)"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "Content to insert"
                                    }
                                },
                                "required": ["type", "anchor_line", "anchor_hash", "content"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["hashline_delete"]
                                    },
                                    "start_line": {
                                        "type": "number",
                                        "description": "Start line number (1-indexed, inclusive). Must match hash from fs op=read output."
                                    },
                                    "start_hash": {
                                        "type": "string",
                                        "description": "4-hex hash (legacy 2-hex only at this hinted line; not relocated) for start line (from fs op=read line:hash format)"
                                    },
                                    "end_line": {
                                        "type": "number",
                                        "description": "End line number (1-indexed, inclusive). Must match hash from fs op=read output."
                                    },
                                    "end_hash": {
                                        "type": "string",
                                        "description": "4-hex hash (legacy 2-hex only at this hinted line; not relocated) for end line (from fs op=read line:hash format)"
                                    }
                                },
                                "required": ["type", "start_line", "start_hash", "end_line", "end_hash"]
                            }
                        ]
                    }
                }
            },
            "required": ["path", "operations"]
        })
    }
}

async fn execute_edit_file(arguments: serde_json::Value) -> ToolResult {
    let path_str = match params::extract_string(&arguments, "path", None) {
        Ok(p) => p,
        Err(e) => return ToolResult::error(e),
    };

    let operations = match coerce_operations(&arguments) {
        Ok(ops) => ops,
        Err(error) => return ToolResult::error(error),
    };

    let file_path = match path::resolve_write_path_for_tool(&path_str) {
        Ok(p) => p,
        Err(e) => return ToolResult::error(e),
    };

    if !file_path.exists() {
        return ToolResult::error(format!(
            "File '{path_str}' does not exist. Use fs with op=write to create new files."
        ));
    }

    crate::agent::tools::file_mutation_queue::with_path_lock(&file_path, || {
        let file_path = file_path.clone();
        let path_str = path_str.clone();
        let operations = operations.clone();
        async move {
            match crate::agent::tools::common::spawn_tool_blocking(move || {
                apply_edit_sync(&file_path, &path_str, &operations)
            })
            .await
            {
                Ok(result) => result,
                Err(e) => ToolResult::error(format!("edit worker failed: {e}")),
            }
        }
    })
    .await
}

fn apply_edit_sync(
    file_path: &Path,
    path_str: &str,
    operations: &[serde_json::Value],
) -> ToolResult {
    let mut perf = crate::observability::perf::span("edit.apply");
    let content = match path::read_tool_text(file_path) {
        Ok(c) => c,
        Err(e) => {
            return ToolResult::error(errors::format_file_error("read", path_str, &e.to_string()));
        }
    };

    let original_had_trailing_newline = content.ends_with('\n');

    let (mut content, mut changes_applied, remaining_indices) =
        match apply_str_replaces_against_original(&content, operations) {
            Ok(v) => v,
            Err(e) => return ToolResult::error(format_edit_error(&e)),
        };
    let mut lines: Vec<String> = content
        .lines()
        .map(std::string::ToString::to_string)
        .collect();

    for idx in remaining_indices {
        let op = &operations[idx];
        if let Err(e) = apply_operation(&mut content, &mut lines, &mut changes_applied, op, idx) {
            return ToolResult::error(format_edit_error(&e));
        }
        match op["type"].as_str() {
            Some("str_replace") => {
                lines = content
                    .lines()
                    .map(std::string::ToString::to_string)
                    .collect();
            }
            _ => {
                content = lines.join("\n");
            }
        }
    }

    if original_had_trailing_newline && !content.is_empty() && !content.ends_with('\n') {
        content.push('\n');
    }

    match path::write_tool_file(file_path, content.as_bytes()) {
        Ok(()) => {
            if let Some(ref mut span) = perf {
                span.set_file_count(1);
                span.set_match_count(u64::from(changes_applied));
                span.set_byte_count(content.len() as u64);
            }
            drop(perf);
            ToolResult::success(format!(
                "Successfully applied {changes_applied} operation(s) to file '{path_str}'"
            ))
        }
        Err(e) => ToolResult::error(errors::format_file_error("write", path_str, &e.to_string())),
    }
}

#[async_trait]
impl Tool for EditFileTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        execute_edit_file(arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tools::common::set_tool_host_fs_unlocked;
    use tempfile::TempDir;

    /// Unlock host FS for a test and return a guard that re-locks on drop.
    fn unlock_host_fs() -> impl Drop {
        set_tool_host_fs_unlocked(true);
        struct Guard;
        impl Drop for Guard {
            fn drop(&mut self) {
                set_tool_host_fs_unlocked(false);
            }
        }
        Guard
    }

    #[tokio::test]
    async fn test_str_replace() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test.txt");
        tokio::fs::write(&file_path, "Hello, World!")
            .await
            .expect("Failed to write test file");

        let tool = EditFileTool;
        let args = json!({
            "path": file_path.to_str().expect("Failed to convert path to string"),
            "operations": [
                {
                    "type": "str_replace",
                    "old_string": "World",
                    "new_string": "Universe"
                }
            ]
        });

        let result = tool.execute(args).await;
        assert!(result.success);

        let content = tokio::fs::read_to_string(&file_path)
            .await
            .expect("Failed to read test file");
        assert_eq!(content, "Hello, Universe!");
    }

    #[tokio::test]
    async fn test_insert() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test.txt");
        tokio::fs::write(&file_path, "Line 1\nLine 2")
            .await
            .expect("Failed to write test file");

        let tool = EditFileTool;
        let args = json!({
            "path": file_path.to_str().expect("Failed to convert path to string"),
            "operations": [
                {
                    "type": "insert",
                    "line": 2,
                    "content": "Inserted line\n"
                }
            ]
        });

        let result = tool.execute(args).await;
        assert!(result.success);

        let content = tokio::fs::read_to_string(&file_path)
            .await
            .expect("Failed to read test file");
        assert!(content.contains("Inserted line"));
    }

    #[tokio::test]
    async fn test_delete_lines() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        let file_path = temp_dir.path().join("test.txt");
        tokio::fs::write(&file_path, "Line 1\nLine 2\nLine 3\nLine 4")
            .await
            .expect("Failed to write test file");

        let tool = EditFileTool;
        let args = json!({
            "path": file_path.to_str().expect("Failed to convert path to string"),
            "operations": [
                {
                    "type": "delete_lines",
                    "start_line": 2,
                    "end_line": 3
                }
            ]
        });

        let result = tool.execute(args).await;
        assert!(result.success);

        let content = tokio::fs::read_to_string(&file_path)
            .await
            .expect("Failed to read test file");
        assert!(!content.contains("Line 2"));
        assert!(!content.contains("Line 3"));
    }

    #[tokio::test]
    async fn str_replace_requires_unique_match() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("tempdir");
        let file_path = temp_dir.path().join("dup.txt");
        tokio::fs::write(&file_path, "foo bar foo")
            .await
            .expect("write");

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": file_path.to_str().expect("path"),
                "operations": [{
                    "type": "str_replace",
                    "old_string": "foo",
                    "new_string": "baz"
                }]
            }))
            .await;
        assert!(!result.success);
        assert!(
            result
                .error
                .as_deref()
                .unwrap_or_default()
                .contains("occurrences")
        );
    }

    #[tokio::test]
    async fn multi_str_replace_matches_original() {
        let _guard = unlock_host_fs();
        let temp_dir = TempDir::new().expect("tempdir");
        let file_path = temp_dir.path().join("multi.txt");
        tokio::fs::write(&file_path, "alpha\nbeta\ngamma\n")
            .await
            .expect("write");

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": file_path.to_str().expect("path"),
                "operations": [
                    {
                        "type": "str_replace",
                        "old_string": "alpha",
                        "new_string": "ALPHA"
                    },
                    {
                        "type": "str_replace",
                        "old_string": "gamma",
                        "new_string": "GAMMA"
                    }
                ]
            }))
            .await;
        assert!(result.success, "{:?}", result.error);
        let content = tokio::fs::read_to_string(&file_path).await.expect("read");
        assert_eq!(content, "ALPHA\nbeta\nGAMMA\n");
    }

    #[tokio::test]
    async fn large_disjoint_str_replaces_apply_independently() {
        let manifest_target = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
        tokio::fs::create_dir_all(&manifest_target)
            .await
            .expect("create target dir");
        let temp_dir = TempDir::new_in(&manifest_target).expect("tempdir");
        let file_path = temp_dir.path().join("lib.rs");
        let original: String = (0..24)
            .map(|i| format!("    fn item_{i}() {{ {i} }}\n"))
            .collect();
        tokio::fs::write(&file_path, &original)
            .await
            .expect("write");

        let operations: Vec<serde_json::Value> = (0..24)
            .step_by(2)
            .map(|i| {
                json!({
                    "type": "str_replace",
                    "old_string": format!("    fn item_{i}() {{ {i} }}"),
                    "new_string": format!("    fn item_{i}() {{ {} }}", i + 100)
                })
            })
            .collect();

        let tool = EditFileTool;
        let result = tool
            .execute(json!({
                "path": file_path.to_str().expect("path"),
                "operations": operations
            }))
            .await;
        assert!(result.success, "{:?}", result.error);
        let content = tokio::fs::read_to_string(&file_path).await.expect("read");
        for i in 0..24 {
            if i % 2 == 0 {
                assert!(
                    content.contains(&format!("    fn item_{i}() {{ {} }}", i + 100)),
                    "item_{i} should be updated"
                );
            } else {
                assert!(
                    content.contains(&format!("    fn item_{i}() {{ {i} }}")),
                    "item_{i} should be unchanged"
                );
            }
        }
    }
}
