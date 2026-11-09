use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use anyhow::Result;

impl SlashCommandExecutor {
    pub(crate) fn handle_lsp() -> Result<SlashCommandResult> {
        use crate::lsp::{LspConfig, LspManager, format_lsp_status};

        let config = LspConfig::default();
        let manager = LspManager::new(config);
        let content = format_lsp_status(&manager);
        Ok(SlashCommandResult::ShowPopup {
            title: "LSP Status".to_string(),
            content,
        })
    }

    pub(crate) async fn handle_editor() -> Result<SlashCommandResult> {
        use crate::handlers::editor::run_editor_command;

        match run_editor_command().await {
            Ok(Some(content)) => {
                // Return the content as a special result that the REPL can use
                Ok(SlashCommandResult::EditorContent(content))
            }
            Ok(None) => Ok(SlashCommandResult::StatusMessage(
                "Editor cancelled or empty content.".to_string(),
            )),
            Err(e) => Ok(SlashCommandResult::Error(format!("Editor error: {}", e))),
        }
    }

    /// Handle `/copy` - copy last agent response to clipboard
    pub(crate) fn handle_copy(state: &InteractiveState) -> Result<SlashCommandResult> {
        use caliche::messages::MessageRole;

        let text = state
            .messages
            .iter()
            .rev()
            .find(|m| m.role == MessageRole::Assistant)
            .and_then(|m| m.as_text())
            .map(|s| s.to_string());

        let text = match text {
            Some(t) if !t.trim().is_empty() => t,
            _ => {
                return Ok(SlashCommandResult::Error(
                    "No agent response to copy.".to_string(),
                ));
            }
        };

        match arboard::Clipboard::new() {
            Ok(mut clipboard) => match clipboard.set_text(&text) {
                Ok(()) => Ok(SlashCommandResult::StatusMessage(
                    "Last agent response copied to clipboard.".to_string(),
                )),
                Err(e) => Ok(SlashCommandResult::Error(format!(
                    "Failed to write to clipboard: {}",
                    e
                ))),
            },
            Err(e) => Ok(SlashCommandResult::Error(format!(
                "Failed to access clipboard: {}",
                e
            ))),
        }
    }
}
