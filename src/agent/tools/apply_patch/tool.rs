//! `apply_patch` tool and filesystem application helpers.

use super::parser::{Hunk, UpdateFileChunk, parse_patch};
use super::placement::{
    insertion_index, lift_underindented_lines, reindent_insert, reindent_to_match,
};
use super::seek_sequence::seek_sequence;
use crate::agent::tools::common::path;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "lsp")]
use crate::lsp::LspManager;
#[cfg(feature = "lsp")]
use crate::lsp::inject_paths_diagnostics;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::json;
use std::path::{Path, PathBuf};
#[cfg(feature = "lsp")]
use std::sync::Arc;

/// How file mutations are exposed to the model.
///
/// `EditWrite` and `ApplyPatch` are mutually exclusive: when patch mode is
/// selected, `fs` omits write/edit ops and `apply_patch` is registered instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MutationStyle {
    /// Structured `fs` write/edit operations (default).
    #[default]
    EditWrite,
    /// Freeform Codex-style `apply_patch` tool.
    ApplyPatch,
}

const NAME: &str = "apply_patch";
const DESCRIPTION: &str = "Apply a freeform multi-file patch. Wrap changes in *** Begin Patch / *** End Patch. Use *** Add File: / *** Delete File: / *** Update File: hunks with +/-/space lines. Mutually exclusive with fs write/edit.";

/// Freeform multi-file patch tool.
#[derive(Debug, Default)]
pub struct ApplyPatchTool {
    #[cfg(feature = "lsp")]
    lsp_manager: Option<Arc<LspManager>>,
}

impl ApplyPatchTool {
    /// Construct without an LSP manager.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "lsp")]
            lsp_manager: None,
        }
    }

    /// Attach an LSP manager so successful patches inject diagnostics per touched file.
    #[cfg(feature = "lsp")]
    #[must_use]
    pub fn with_lsp_manager(mut self, manager: Option<Arc<LspManager>>) -> Self {
        self.lsp_manager = manager;
        self
    }
}

impl ToolMeta for ApplyPatchTool {
    fn name(&self) -> &str {
        NAME
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Full patch text including Begin/End Patch markers."
                }
            },
            "required": ["patch"]
        })
    }

    fn is_readonly(&self) -> bool {
        false
    }
}

#[async_trait]
impl Tool for ApplyPatchTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let patch = arguments
            .get("patch")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let Some(patch) = patch else {
            return ToolResult::error("apply_patch requires 'patch' string");
        };

        let mut perf = crate::observability::perf::span("apply_patch");
        match apply_hunks_to_cwd(&patch, None).await {
            Ok(summary) => {
                if let Some(ref mut span) = perf {
                    span.set_file_count(1);
                }
                drop(perf);
                let result = ToolResult::success_with_payload(
                    summary.clone(),
                    json!({
                        "op": "apply_patch",
                        "summary": summary,
                    }),
                );
                #[cfg(feature = "lsp")]
                {
                    if let Some(manager) = self.lsp_manager.as_ref() {
                        let paths = touched_paths_from_patch(&patch, None);
                        return inject_paths_diagnostics(manager, &paths, result).await;
                    }
                }
                result
            }
            Err(err) => ToolResult::error(err.to_string()),
        }
    }
}

#[cfg_attr(
    not(feature = "lsp"),
    expect(dead_code, reason = "only the lsp build reports touched paths")
)]
fn touched_paths_from_patch(patch: &str, cwd: Option<&Path>) -> Vec<PathBuf> {
    let Ok(args) = parse_patch(patch) else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    for hunk in &args.hunks {
        match hunk {
            Hunk::AddFile { path, .. } => {
                if let Ok(resolved) = resolve_hunk_path(path, cwd) {
                    paths.push(resolved);
                }
            }
            Hunk::UpdateFile {
                path, move_path, ..
            } => {
                let target = move_path.as_ref().unwrap_or(path);
                if let Ok(resolved) = resolve_hunk_path(target, cwd) {
                    paths.push(resolved);
                }
            }
            Hunk::DeleteFile { .. } => {}
        }
    }
    paths
}

/// Apply a parsed patch relative to `cwd` (defaults to process cwd).
pub async fn apply_hunks_to_cwd(patch: &str, cwd: Option<&Path>) -> Result<String> {
    let args = parse_patch(patch).context("failed to parse apply_patch")?;
    let mut summaries = Vec::new();

    for hunk in &args.hunks {
        match hunk {
            Hunk::AddFile { path, contents } => {
                let resolved = resolve_hunk_path(path, cwd)?;
                let contents = contents.clone();
                super::super::file_mutation_queue::with_path_lock(&resolved, || {
                    let resolved = resolved.clone();
                    async move {
                        tokio::task::spawn_blocking(move || apply_add_sync(&resolved, &contents))
                            .await
                            .map_err(|e| anyhow::anyhow!("apply_patch worker failed: {e}"))?
                    }
                })
                .await?;
                summaries.push(format!("Added {}", resolved.display()));
            }
            Hunk::DeleteFile { path } => {
                let resolved = resolve_hunk_path(path, cwd)?;
                super::super::file_mutation_queue::with_path_lock(&resolved, || {
                    let resolved = resolved.clone();
                    async move {
                        tokio::task::spawn_blocking(move || apply_delete_sync(&resolved))
                            .await
                            .map_err(|e| anyhow::anyhow!("apply_patch worker failed: {e}"))?
                    }
                })
                .await?;
                summaries.push(format!("Deleted {}", resolved.display()));
            }
            Hunk::UpdateFile {
                path,
                move_path,
                chunks,
            } => {
                let resolved = resolve_hunk_path(path, cwd)?;
                let dest = match move_path {
                    Some(p) => Some(resolve_hunk_path(p, cwd)?),
                    None => None,
                };
                let summary = super::super::file_mutation_queue::with_path_lock(&resolved, || {
                    let resolved = resolved.clone();
                    let dest = dest.clone();
                    let chunks = chunks.clone();
                    async move { apply_update(&resolved, dest.as_deref(), &chunks).await }
                })
                .await?;
                summaries.push(summary);
            }
        }
    }

    Ok(summaries.join("\n"))
}

/// Every hunk path is a write target (add, update, delete, or the `Move to`
/// destination), so each goes through the host-side `.git` write guard.
fn resolve_hunk_path(path: &Path, cwd: Option<&Path>) -> Result<PathBuf> {
    let path_str = path.to_string_lossy();
    if path.is_absolute() {
        path::check_path_traversal(&path_str)?;
        path::refuse_git_metadata_write(path)?;
        return Ok(path.to_path_buf());
    }
    if let Some(cwd) = cwd {
        path::check_path_traversal(&path_str)?;
        let joined = cwd.join(path);
        path::refuse_git_metadata_write(&joined)?;
        return Ok(joined);
    }
    path::resolve_write_path_for_tool(&path_str)
}

fn apply_add_sync(path: &Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create dirs for {}", path.display()))?;
    }
    std::fs::write(path, contents).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn apply_delete_sync(path: &Path) -> Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path).with_context(|| format!("delete dir {}", path.display()))?;
    } else {
        std::fs::remove_file(path).with_context(|| format!("delete {}", path.display()))?;
    }
    Ok(())
}

async fn apply_update(
    path: &Path,
    move_path: Option<&Path>,
    chunks: &[UpdateFileChunk],
) -> Result<String> {
    let path = path.to_path_buf();
    let dest = move_path.map(Path::to_path_buf);
    let chunks = chunks.to_vec();
    tokio::task::spawn_blocking(move || apply_update_sync(&path, dest.as_deref(), &chunks))
        .await
        .map_err(|e| anyhow::anyhow!("apply_patch worker failed: {e}"))?
}

fn apply_update_sync(
    path: &Path,
    move_path: Option<&Path>,
    chunks: &[UpdateFileChunk],
) -> Result<String> {
    let original =
        std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut original_lines: Vec<String> = original.split('\n').map(String::from).collect();
    if original_lines.last().is_some_and(String::is_empty) {
        original_lines.pop();
    }

    let replacements = compute_replacements(&original_lines, &path.display().to_string(), chunks)?;
    let mut new_lines = apply_replacements(original_lines, &replacements);
    if !new_lines.last().is_some_and(String::is_empty) {
        new_lines.push(String::new());
    }
    let new_contents = new_lines.join("\n");

    let write_path = move_path.unwrap_or(path);
    if let Some(parent) = write_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create dirs for {}", write_path.display()))?;
    }
    std::fs::write(write_path, &new_contents)
        .with_context(|| format!("write {}", write_path.display()))?;
    if let Some(dest) = move_path
        && dest != path
    {
        std::fs::remove_file(path)
            .with_context(|| format!("remove moved source {}", path.display()))?;
        return Ok(format!("Updated {} -> {}", path.display(), dest.display()));
    }
    Ok(format!("Updated {}", path.display()))
}

fn compute_replacements(
    original_lines: &[String],
    path: &str,
    chunks: &[UpdateFileChunk],
) -> Result<Vec<(usize, usize, Vec<String>)>> {
    let mut replacements = Vec::new();
    let mut line_index = 0usize;

    for chunk in chunks {
        if let Some(ctx_line) = &chunk.change_context {
            if let Some(idx) = seek_sequence(
                original_lines,
                std::slice::from_ref(ctx_line),
                line_index,
                false,
            ) {
                line_index = idx + 1;
            } else {
                bail!("Failed to find context '{ctx_line}' in {path}");
            }
        }

        if chunk.old_lines.is_empty() {
            let insertion_idx =
                insertion_index(original_lines.len(), line_index, chunk.is_end_of_file);
            let mut new_lines = reindent_insert(original_lines, insertion_idx, &chunk.new_lines);
            if let Some(neighbor) = original_lines.get(insertion_idx).or_else(|| {
                insertion_idx
                    .checked_sub(1)
                    .and_then(|idx| original_lines.get(idx))
            }) {
                new_lines = lift_underindented_lines(neighbor, &new_lines);
            }
            replacements.push((insertion_idx, 0, new_lines));
            continue;
        }

        let mut pattern: &[String] = &chunk.old_lines;
        let mut found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file);
        let mut new_slice: &[String] = &chunk.new_lines;

        if found.is_none() && pattern.last().is_some_and(String::is_empty) {
            pattern = &pattern[..pattern.len() - 1];
            if new_slice.last().is_some_and(String::is_empty) {
                new_slice = &new_slice[..new_slice.len() - 1];
            }
            found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file);
        }

        if let Some(start_idx) = found {
            let new_lines = match (original_lines.get(start_idx), pattern.first()) {
                (Some(file_line), Some(pattern_line)) => {
                    let aligned = reindent_to_match(file_line, pattern_line, new_slice);
                    lift_underindented_lines(file_line, &aligned)
                }
                _ => new_slice.to_vec(),
            };
            replacements.push((start_idx, pattern.len(), new_lines));
            line_index = start_idx + pattern.len();
        } else {
            bail!(
                "Failed to find expected lines in {path}:\n{}",
                chunk.old_lines.join("\n")
            );
        }
    }

    replacements.sort_by_key(|(index, _, _)| *index);
    for pair in replacements.windows(2) {
        let (prev_start, prev_len, _) = pair[0];
        let (start, _, _) = pair[1];
        if prev_start + prev_len > start {
            bail!(
                "overlapping hunks in {path} at original lines {} and {}",
                prev_start + 1,
                start + 1
            );
        }
    }
    Ok(replacements)
}

fn apply_replacements(
    mut lines: Vec<String>,
    replacements: &[(usize, usize, Vec<String>)],
) -> Vec<String> {
    for (start_idx, old_len, new_segment) in replacements.iter().rev() {
        let start_idx = *start_idx;
        let old_len = *old_len;
        for _ in 0..old_len {
            if start_idx < lines.len() {
                lines.remove(start_idx);
            }
        }
        for (offset, new_line) in new_segment.iter().enumerate() {
            lines.insert(start_idx + offset, new_line.clone());
        }
    }
    lines
}
