//! LSP (Language Server Protocol) command handler.

use crate::commands::LspAction;
use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;

/// Handler for LSP commands.
pub struct LspHandler;

#[async_trait]
impl CommandHandler for LspHandler {
    type Args = LspAction;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        handle_lsp_command(args)
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "lsp"
    }

    fn description(&self) -> &'static str {
        "Language Server Protocol integration"
    }

    fn requires_credentials(&self) -> bool {
        true
    }
}

/// Handle LSP subcommands.
pub async fn handle_lsp_command(action: LspAction) -> Result<()> {
    let config = crate::lsp::LspConfig::default();
    let manager = crate::lsp::LspManager::new(config);

    match action {
        LspAction::Status => {
            crate::lsp::print_lsp_status(&manager);
        }
        LspAction::Install { language } => {
            crate::lsp::install_language_server(&language).await?;
        }
        LspAction::List => {
            println!("\nSupported Languages:");
            for lang in manager.supported_languages() {
                let available = if manager.is_server_available(&lang) {
                    "\x1b[32mavailable\x1b[0m"
                } else {
                    "\x1b[31mnot installed\x1b[0m"
                };
                println!("  {:<12} [{}]", lang, available);
            }
            println!();
        }
        LspAction::Enable => {
            println!("LSP is enabled for this session.");
            println!("Note: LSP state is not persisted between sessions yet.");
        }
        LspAction::Disable => {
            println!("LSP is disabled for this session.");
        }
    }

    crate::shutdown::exit(0);
}
