//! Editor command handler.
//!
//! Handles opening external editors for multi-line input.

use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;

/// Arguments for the editor command.
#[derive(Debug, Clone, Default)]
pub struct EditorArgs;

/// Handler for the `/editor` command - open external editor for input.
pub struct EditorHandler;

#[async_trait]
impl CommandHandler for EditorHandler {
    type Args = EditorArgs;

    async fn execute(
        &self,
        _args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        match run_editor_command().await {
            Ok(Some(content)) => Ok(CommandResult::Output(serde_json::json!({
                "content": content,
                "source": "editor"
            }))),
            Ok(None) => Ok(CommandResult::Cancelled),
            Err(e) => Err(CliError::Command(e.to_string())),
        }
    }

    fn name(&self) -> &'static str {
        "editor"
    }

    fn description(&self) -> &'static str {
        "Open external editor for multi-line input"
    }

    fn help(&self) -> Option<&'static str> {
        Some(
            "Opens your configured $EDITOR or $VISUAL for composing longer messages.\n\
             Save and close the editor to submit the content.",
        )
    }

    fn examples(&self) -> &[&'static str] {
        &["/editor (in REPL)"]
    }

    fn requires_credentials(&self) -> bool {
        false
    }

    fn supports_offline(&self) -> bool {
        true
    }
}

/// Run the editor command implementation.
pub async fn run_editor_command() -> Result<Option<String>> {
    // Get editor from environment
    let editor = std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "notepad".to_string()
            } else {
                "vim".to_string()
            }
        });

    let temp = crate::utils::create_private_temp_file("caliche_input_", ".md")?;
    let temp_path = crate::utils::temp_file_path(&temp);

    // Write initial content
    let initial_content = "# Enter your message below\n# Lines starting with # are ignored\n# Save and close the editor when done\n\n";
    tokio::fs::write(&temp_path, initial_content).await?;

    println!("Opening editor: {}", editor);
    println!("File: {}", temp_path.display());

    // Open editor (blocking operation)
    let status = std::process::Command::new(&editor)
        .arg(&temp_path)
        .status()?;

    if !status.success() {
        anyhow::bail!("Editor exited with error");
    }

    // Read content (temp file removed when `temp` is dropped)
    let content = tokio::fs::read_to_string(&temp_path).await?;

    // Filter out comment lines
    let cleaned: String = content
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    if cleaned.is_empty() {
        Ok(None)
    } else {
        Ok(Some(cleaned))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_editor_handler_metadata() {
        let handler = EditorHandler;
        assert_eq!(handler.name(), "editor");
        assert!(!handler.description().is_empty());
        assert!(!handler.requires_credentials());
        assert!(handler.supports_offline());
    }
}
