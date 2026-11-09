use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use anyhow::Result;
use std::path::PathBuf;

impl SlashCommandExecutor {
    pub(crate) async fn handle_attach(
        state: &mut InteractiveState,
        paths: &[PathBuf],
    ) -> Result<SlashCommandResult> {
        if paths.is_empty() {
            if state.pending_attachments.is_empty() {
                return Ok(SlashCommandResult::StatusMessage(
                    "No pending attachments. Usage: /attach <file> [file...]".to_string(),
                ));
            }
            let names: Vec<String> = state
                .pending_attachments
                .iter()
                .filter_map(|a| a.filename.clone())
                .collect();
            return Ok(SlashCommandResult::StatusMessage(format!(
                "Pending attachments ({}): {}",
                state.pending_attachments.len(),
                names.join(", ")
            )));
        }

        let mut loaded = Vec::new();
        let mut errors = Vec::new();

        for path in paths {
            match caliche::messages::Attachment::from_file(path).await {
                Ok(att) => {
                    let name = att
                        .filename
                        .clone()
                        .unwrap_or_else(|| path.display().to_string());
                    let media_type = att.media_type.clone();
                    loaded.push(name.clone());
                    eprintln!(
                        "\x1b[36m[attached: {} ({}), {} bytes]\x1b[0m",
                        name,
                        media_type,
                        att.data.len()
                    );
                    state.pending_attachments.push(att);
                }
                Err(e) => {
                    errors.push(format!("{}: {}", path.display(), e));
                }
            }
        }

        let msg = if errors.is_empty() {
            format!(
                "Attached {} file(s): {}. Send your next message to include them.",
                loaded.len(),
                loaded.join(", ")
            )
        } else {
            format!(
                "Attached {} file(s): {}. Errors: {}",
                loaded.len(),
                loaded.join(", "),
                errors.join("; ")
            )
        };

        Ok(SlashCommandResult::StatusMessage(msg))
    }

    pub(crate) async fn handle_commands() -> Result<SlashCommandResult> {
        use crate::custom_commands::CommandRegistry;

        let registry = CommandRegistry::load()
            .await
            .unwrap_or_else(|_| CommandRegistry::new());

        if registry.is_empty() {
            let user_dir = CommandRegistry::user_commands_dir()
                .map(|d| d.display().to_string())
                .unwrap_or_else(|_| "~/.config/caliche/commands/".to_string());
            let project_dir = CommandRegistry::project_commands_dir()
                .map(|d| d.display().to_string())
                .unwrap_or_else(|| ".caliche/commands/".to_string());

            Ok(SlashCommandResult::ShowPopup {
                title: "Custom Commands".to_string(),
                content: format!(
                    "No custom commands found.\n\nCreate commands by placing .md files in:\n  {}\n  {}\n\nEach file becomes a slash command.\n\nExample: test.md becomes /test\n\nNamed arguments: Use $PLACEHOLDER in the prompt.",
                    user_dir, project_dir
                ),
            })
        } else {
            let commands = registry.list();
            let items: Vec<String> = commands
                .iter()
                .map(|(name, _)| format!("/{}", name))
                .collect();
            let descriptions: Vec<Option<String>> = commands
                .iter()
                .map(|(_, desc)| Some(desc.to_string()))
                .collect();
            Ok(SlashCommandResult::ShowList {
                title: "Custom Commands".to_string(),
                items,
                descriptions,
            })
        }
    }
}
