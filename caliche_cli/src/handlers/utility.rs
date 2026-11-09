//! Utility command handlers.
//!
//! Handles listing models, tools, and other informational commands.

use anyhow::Result;
use async_trait::async_trait;
use caliche::llm::{ModelCatalogOptions, ProviderType, discover_provider_models};

use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};

/// Handler for listing models.
pub struct ListModelsHandler;

#[async_trait]
impl CommandHandler for ListModelsHandler {
    type Args = String; // provider name

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        handle_list_models(args)
            .await
            .map_err(|e| CliError::Provider(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "list-models"
    }

    fn description(&self) -> &'static str {
        "List available models for a provider"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

/// Handler for listing tools.
pub struct ListToolsHandler;

#[async_trait]
impl CommandHandler for ListToolsHandler {
    type Args = ();

    async fn execute(
        &self,
        _args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        handle_list_tools()
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "list-tools"
    }

    fn description(&self) -> &'static str {
        "List available tools"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

/// List available models for a provider.
pub async fn handle_list_models(provider_str: String) -> Result<()> {
    let provider: ProviderType = provider_str
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid provider: {}", e))?;

    let entry = discover_provider_models(provider, &cli_model_catalog_options().await).await?;
    println!("Available {} models:", entry.display_name);

    if entry.models.is_empty() {
        println!("  No configured models discovered for {}", entry.id);
        return Ok(());
    }

    for model in entry.models {
        println!("  - {}", model.id);
    }

    Ok(())
}

async fn cli_model_catalog_options() -> ModelCatalogOptions {
    let manager = if crate::storage::keyring_prompts_allowed() {
        crate::storage::SecureCredentialManager::new().await.ok()
    } else {
        None
    };
    ModelCatalogOptions::from_environment().with_credential_provider(
        crate::storage::session_credential_provider(manager.as_ref()).await,
    )
}

/// List available tools.
pub async fn handle_list_tools() -> Result<()> {
    use caliche::AgentService;

    println!("\nAvailable Tools\n");

    let service = AgentService::new().await?;
    let tools = service.list_tools();

    if tools.is_empty() {
        println!("No tools available.");
    } else {
        for (name, description, _schema) in &tools {
            println!("  {} - {}", name, description);
        }
        println!("\n{} tools available", tools.len());
    }

    Ok(())
}
