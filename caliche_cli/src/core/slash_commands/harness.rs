use super::helpers::*;
use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use anyhow::Result;
use std::path::PathBuf;

impl SlashCommandExecutor {
    pub(crate) fn handle_track(
        state: &mut InteractiveState,
        file: Option<&str>,
    ) -> Result<SlashCommandResult> {
        let file = match file {
            Some(f) => PathBuf::from(f),
            None => {
                return Ok(SlashCommandResult::Error(
                    "Usage: /track <file>".to_string(),
                ));
            }
        };

        if file.exists() {
            if !state.tracked_files.contains(&file) {
                state.tracked_files.push(file.clone());
            }
            Ok(SlashCommandResult::StatusMessage(format!(
                "Tracking: {}",
                file.display()
            )))
        } else {
            Ok(SlashCommandResult::Error(format!(
                "File not found: {}",
                file.display()
            )))
        }
    }

    pub(crate) async fn handle_add(
        state: &mut InteractiveState,
        pattern: Option<&str>,
    ) -> Result<SlashCommandResult> {
        let pattern = match pattern {
            Some(p) => p,
            None => {
                return Ok(SlashCommandResult::Error(
                    "Usage: /add <glob_pattern>".to_string(),
                ));
            }
        };

        use crate::handlers::interactive::context::find_files_by_glob;
        match find_files_by_glob(pattern).await {
            Ok(files) => {
                if files.is_empty() {
                    Ok(SlashCommandResult::StatusMessage(format!(
                        "No files found matching: {}",
                        pattern
                    )))
                } else {
                    for file in &files {
                        if !state.tracked_files.contains(file) {
                            state.tracked_files.push(file.clone());
                        }
                    }
                    Ok(SlashCommandResult::StatusMessage(format!(
                        "Added {} files to context",
                        files.len()
                    )))
                }
            }
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Error finding files: {}",
                e
            ))),
        }
    }

    pub(crate) async fn handle_recent() -> Result<SlashCommandResult> {
        use crate::handlers::interactive::context::find_recent_files;
        match find_recent_files(10).await {
            Ok(files) => {
                if files.is_empty() {
                    Ok(SlashCommandResult::StatusMessage(
                        "No recent files found.".to_string(),
                    ))
                } else {
                    let items: Vec<String> =
                        files.iter().map(|f| f.0.display().to_string()).collect();
                    let len = items.len();
                    Ok(SlashCommandResult::ShowList {
                        title: "Recent Files".to_string(),
                        items,
                        descriptions: vec![None; len],
                    })
                }
            }
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Error finding recent files: {}",
                e
            ))),
        }
    }

    pub(crate) async fn handle_related(file: Option<&str>) -> Result<SlashCommandResult> {
        let file = match file {
            Some(f) => f,
            None => {
                return Ok(SlashCommandResult::Error(
                    "Usage: /related <file>".to_string(),
                ));
            }
        };

        use crate::handlers::interactive::context::find_related_files;
        match find_related_files(file).await {
            Ok(related) => {
                if related.is_empty() {
                    Ok(SlashCommandResult::StatusMessage(format!(
                        "No related files found for: {}",
                        file
                    )))
                } else {
                    let items: Vec<String> =
                        related.iter().map(|f| f.0.display().to_string()).collect();
                    let len = items.len();
                    Ok(SlashCommandResult::ShowList {
                        title: format!("Files related to '{}'", file),
                        items,
                        descriptions: vec![None; len],
                    })
                }
            }
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Error finding related files: {}",
                e
            ))),
        }
    }

    pub(crate) async fn handle_session_diff_overview(
        harness: &caliche::AgentHarness,
    ) -> Result<SlashCommandResult> {
        let summary = harness.session_diff_summary().await;
        let content = format_session_diff_for_repl(&summary);
        Ok(SlashCommandResult::ShowPopup {
            title: "Session diff (per-message undo)".to_string(),
            content,
        })
    }

    pub(crate) async fn handle_undo(
        state: &mut InteractiveState,
        harness: Option<&mut caliche::AgentHarness>,
    ) -> Result<SlashCommandResult> {
        if let Some(h) = harness {
            match h.undo_last().await {
                Ok(Some(diff)) => Ok(SlashCommandResult::StatusMessage(format!(
                    "Reverted {} file change(s) from message {}",
                    diff.snapshots.len(),
                    diff.message_id
                ))),
                Ok(None) => Ok(SlashCommandResult::StatusMessage(
                    "Nothing to undo".to_string(),
                )),
                Err(e) => Ok(SlashCommandResult::Error(format!("Undo failed: {}", e))),
            }
        } else {
            if let Some(_undo) = state.pop_undo_state() {
                Ok(SlashCommandResult::StatusMessage(
                    "Undo state restored".to_string(),
                ))
            } else {
                Ok(SlashCommandResult::Error("Nothing to undo".to_string()))
            }
        }
    }

    pub(crate) async fn handle_redo(
        harness: Option<&mut caliche::AgentHarness>,
    ) -> Result<SlashCommandResult> {
        let h = match harness {
            Some(h) => h,
            None => {
                return Ok(SlashCommandResult::Error(
                    "Redo requires harness (agent mode).".to_string(),
                ));
            }
        };
        match h.redo_last().await {
            Ok(Some(diff)) => Ok(SlashCommandResult::StatusMessage(format!(
                "Restored {} file change(s) from message {}",
                diff.snapshots.len(),
                diff.message_id
            ))),
            Ok(None) => Ok(SlashCommandResult::StatusMessage(
                "Nothing to redo".to_string(),
            )),
            Err(e) => Ok(SlashCommandResult::Error(format!("Redo failed: {}", e))),
        }
    }

    pub(crate) async fn handle_checkpoint(
        name: Option<&str>,
        harness: Option<&mut caliche::AgentHarness>,
    ) -> Result<SlashCommandResult> {
        let harness = match harness {
            Some(h) => h,
            None => {
                return Ok(SlashCommandResult::Error(
                    "Checkpoint requires harness support. This feature is not available in this mode.".to_string(),
                ));
            }
        };

        let label = name.unwrap_or("manual");
        let checkpoint_id = harness.checkpoint(label).await;
        Ok(SlashCommandResult::StatusMessage(format!(
            "Checkpoint '{}' created with ID: {}",
            label, checkpoint_id
        )))
    }

    pub(crate) async fn handle_rollback(
        id: Option<&str>,
        harness: Option<&mut caliche::AgentHarness>,
    ) -> Result<SlashCommandResult> {
        let harness = match harness {
            Some(h) => h,
            None => {
                return Ok(SlashCommandResult::Error(
                    "Rollback requires harness support. This feature is not available in this mode.".to_string(),
                ));
            }
        };

        let checkpoint_id = id.ok_or_else(|| anyhow::anyhow!("Checkpoint ID required"))?;
        match harness.rollback(checkpoint_id).await {
            Ok(files) => {
                if files.is_empty() {
                    Ok(SlashCommandResult::StatusMessage(format!(
                        "Rolled back to checkpoint: {} (no files restored)",
                        checkpoint_id
                    )))
                } else {
                    Ok(SlashCommandResult::StatusMessage(format!(
                        "Rolled back to checkpoint: {}. Restored {} file(s): {:?}",
                        checkpoint_id,
                        files.len(),
                        files
                    )))
                }
            }
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Failed to rollback: {}",
                e
            ))),
        }
    }

    pub(crate) fn handle_diff(
        state: &InteractiveState,
        file: Option<&str>,
    ) -> Result<SlashCommandResult> {
        use crate::core::state::FileChangeType;
        use crate::diff::unified_diff;

        let file = match file {
            Some(f) => f.trim(),
            None => {
                if state.modified_files.is_empty() {
                    return Ok(SlashCommandResult::StatusMessage(
                        "No files modified in this session. In agent mode with a harness, \
                         /diff alone shows the per-message undo overview."
                            .to_string(),
                    ));
                }
                let list: Vec<String> = state
                    .modified_files
                    .values()
                    .map(|fc| {
                        let typ = match fc.change_type {
                            FileChangeType::Created => "Created",
                            FileChangeType::Modified => "Modified",
                            FileChangeType::Deleted => "Deleted",
                        };
                        format!("{} ({})", fc.path.display(), typ)
                    })
                    .collect();
                return Ok(SlashCommandResult::ShowPopup {
                    title: "Modified files (use /diff <path>)".to_string(),
                    content: list.join("\n"),
                });
            }
        };

        if state.modified_files.is_empty() {
            return Ok(SlashCommandResult::StatusMessage(
                "No files modified in this session.".to_string(),
            ));
        }

        if file.is_empty() {
            return Ok(SlashCommandResult::Error("Usage: /diff <path>".to_string()));
        }

        let path = PathBuf::from(file);
        let path_abs = if path.is_absolute() {
            path.clone()
        } else {
            state.working_directory.join(&path)
        };

        let change = state
            .modified_files
            .get(&path_abs)
            .or_else(|| {
                state
                    .modified_files
                    .iter()
                    .find(|(p, _)| p.as_os_str() == path.as_os_str() || p.ends_with(file))
                    .map(|(_, v)| v)
            })
            .or_else(|| {
                let file_name = path.file_name();
                state
                    .modified_files
                    .iter()
                    .find(|(p, _)| p.file_name() == file_name)
                    .map(|(_, v)| v)
            });

        let change = match change {
            Some(c) => c,
            None => {
                return Ok(SlashCommandResult::Error(format!(
                    "File not modified in this session: {}",
                    file
                )));
            }
        };

        let (old_content, new_content) = match (&change.original_content, &change.new_content) {
            (Some(orig), Some(new)) => (orig.as_str(), new.as_str()),
            (None, Some(new)) => ("", new.as_str()),
            (Some(orig), None) => (orig.as_str(), ""),
            (None, None) => {
                return Ok(SlashCommandResult::StatusMessage(
                    "No content stored for this change.".to_string(),
                ));
            }
        };

        let label = change.path.display().to_string();
        let content = unified_diff(old_content, new_content, &label, &label);
        Ok(SlashCommandResult::ShowPopup {
            title: format!("Diff: {}", change.path.display()),
            content,
        })
    }
}
