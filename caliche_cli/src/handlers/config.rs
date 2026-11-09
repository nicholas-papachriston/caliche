//! Configuration command handlers.
//!
//! Handles provider credential setup, API key management, and default settings.

use crate::commands::credentials_service::CredentialService;
use crate::commands::{ConfigAction, GatewayConfigAction};
use crate::config::ConfigManager;
use crate::credentials::interactive_setup;
use crate::errors::{CliError, CliResult};
use crate::storage::{Database, SecureCredentialManager};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::io::Write;
use std::path::Path;

use super::vault::handle_vault_command;

fn write_secret_export_file(path: &Path, contents: &str) -> Result<()> {
    #[cfg(unix)]
    let mut file = {
        use std::os::unix::fs::OpenOptionsExt;

        std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(0o600)
            .open(path)
            .with_context(|| format!("failed to open {}", path.display()))?
    };

    #[cfg(not(unix))]
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;

    file.write_all(contents.as_bytes())
        .with_context(|| format!("failed to write {}", path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to set permissions on {}", path.display()))?;
    }

    Ok(())
}

async fn model_catalog_options(
    manager: &SecureCredentialManager,
) -> caliche::llm::ModelCatalogOptions {
    caliche::llm::ModelCatalogOptions::from_environment()
        .with_credential_provider(crate::storage::session_credential_provider(Some(manager)).await)
}

/// Handler for config commands.
pub struct ConfigHandler;

#[async_trait]
impl CommandHandler for ConfigHandler {
    type Args = ConfigAction;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        handle_config_command(args)
            .await
            .map_err(|e| CliError::Config(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "config"
    }

    fn description(&self) -> &'static str {
        "Manage provider credentials and configuration"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

fn resolve_secret_value(key: Option<String>, env: Option<String>, prompt: &str) -> Result<String> {
    if let Some(k) = key {
        return Ok(k);
    }
    if let Some(env_var) = env {
        return caliche_contracts::credential_env::var(&env_var)
            .context(format!("Environment variable '{}' not set", env_var));
    }
    use dialoguer::{Password, theme::ColorfulTheme};
    Password::with_theme(&ColorfulTheme::default())
        .with_prompt(prompt)
        .interact()
        .context("Failed to read API key")
}

fn resolve_gateway_secret_value(
    key: Option<String>,
    env: Option<String>,
    cred_name: &str,
) -> Result<String> {
    if let Some(k) = key {
        return Ok(k);
    }
    if let Some(env_var) = env {
        return caliche_contracts::credential_env::var(&env_var)
            .context(format!("Environment variable '{}' not set", env_var));
    }
    use std::io::{self, BufRead};
    print!("Enter value for {}: ", cred_name);
    std::io::stdout().flush()?;
    let mut input = String::new();
    io::stdin().lock().read_line(&mut input)?;
    Ok(input.trim().to_string())
}

fn require_known_gateway(name: &str) -> Result<String> {
    let cred_name = CredentialService::normalize_gateway_name(name);
    if !CredentialService::is_known_gateway(&cred_name) {
        anyhow::bail!(
            "Unknown gateway credential '{}'. Run 'caliche config gateway list' to see supported names.",
            name
        );
    }
    Ok(cred_name)
}

fn print_masked_or_plain(
    name: &str,
    value: &str,
    unmask: bool,
    confirm_flag: bool,
    command: &str,
) -> Result<()> {
    if unmask {
        CredentialService::require_secret_confirmation(confirm_flag, command)?;
        println!("{}", value);
    } else {
        println!("{}: {}", name, CredentialService::mask_for_display(value));
    }
    Ok(())
}

async fn handle_config_set(
    provider: String,
    key: Option<String>,
    env: Option<String>,
) -> Result<()> {
    let service = CredentialService::new().await?;
    let api_key = resolve_secret_value(key, env, &format!("API key for {provider}"))?;
    service.set_provider_key(&provider, &api_key).await?;
    println!("API key saved for '{}'", provider);
    Ok(())
}

async fn handle_config_get(
    provider: String,
    unmask: bool,
    yes_i_understand_this_prints_secrets: bool,
) -> Result<()> {
    let service = CredentialService::new().await?;
    match service.get_provider_key(&provider).await {
        Some(key) => print_masked_or_plain(
            &provider,
            &key,
            unmask,
            yes_i_understand_this_prints_secrets,
            "config get --unmask",
        )?,
        None => println!("No API key configured for '{}'", provider),
    }
    Ok(())
}

async fn handle_config_remove(provider: String) -> Result<()> {
    let service = CredentialService::new().await?;
    if service.remove_provider_key(&provider).await? {
        println!("API key removed for '{}'", provider);
    } else {
        println!("No API key was configured for '{}'", provider);
    }
    Ok(())
}

async fn handle_config_list(configured: bool) -> Result<()> {
    let service = CredentialService::new().await?;
    let providers = service.list_provider_status(configured).await;

    println!("\nProvider Configuration Status\n");

    for status in providers {
        if configured && !status.is_configured {
            continue;
        }

        let marker = if status.is_default { " *" } else { "" };
        println!("  {}{}", status.name, marker);
        println!("    Description: {}", status.description);
        println!("    Status: {}", status.status_display());
        if let Some(ref model) = status.default_model {
            println!("    Default Model: {}", model);
        }
        println!();
    }

    if service.default_provider().await.is_some() {
        println!("* = default provider\n");
    }
    println!("Database: ~/.caliche/caliche.db");
    Ok(())
}

async fn select_default_provider_name(
    service: &CredentialService,
    provider: Option<String>,
) -> Result<String> {
    if let Some(p) = provider {
        p.parse::<caliche::llm::ProviderType>()
            .map_err(|err| anyhow::anyhow!("Invalid provider: {err}"))?;
        return Ok(p);
    }

    use dialoguer::{FuzzySelect, theme::ColorfulTheme};
    let configured: Vec<_> = service.list_provider_status(true).await;
    if configured.is_empty() {
        anyhow::bail!("No providers are configured. Run 'caliche config setup' first.");
    }

    let items: Vec<String> = configured
        .iter()
        .map(|s| {
            let marker = if s.is_default {
                " [current default]"
            } else {
                ""
            };
            format!("{} - {}{}", s.name, s.description, marker)
        })
        .collect();

    let selection = FuzzySelect::with_theme(&ColorfulTheme::default())
        .with_prompt("Select default provider")
        .items(&items)
        .default(0)
        .interact()?;

    Ok(configured[selection].name.clone())
}

async fn handle_config_default(provider: Option<String>) -> Result<()> {
    let service = CredentialService::new().await?;
    let provider_name = select_default_provider_name(&service, provider).await?;
    service.set_default_provider(&provider_name).await?;
    println!("'{}' set as default provider", provider_name);
    Ok(())
}

async fn fetch_model_items(service: &CredentialService, provider: &str) -> Result<Vec<String>> {
    use crate::credentials::get_recommended_models;
    use caliche::llm::{ProviderType, discover_provider_models};

    let provider_type: ProviderType = provider
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid provider: {}", e))?;
    let fetched_models = discover_provider_models(
        provider_type,
        &model_catalog_options(service.manager())
            .await
            .include_unconfigured(false),
    )
    .await?
    .models
    .into_iter()
    .map(|model| model.id)
    .collect::<Vec<_>>();
    let recommended = get_recommended_models(provider);

    let mut model_items: Vec<String> = Vec::new();
    for m in &fetched_models {
        let suffix = recommended
            .iter()
            .find(|(recommended_model, _)| recommended_model == &m.as_str())
            .map(|(_, desc)| format!(" - {desc} [recommended]"))
            .unwrap_or_default();
        model_items.push(format!("{m}{suffix}"));
    }
    Ok(model_items)
}

fn prompt_custom_model(theme: &dialoguer::theme::ColorfulTheme) -> Result<Option<String>> {
    use dialoguer::Input;
    let custom: String = Input::with_theme(theme)
        .with_prompt("Enter model name")
        .interact_text()?;
    if custom.is_empty() {
        Ok(None)
    } else {
        Ok(Some(custom))
    }
}

fn select_model_name(provider: &str, model_items: &mut Vec<String>) -> Result<Option<String>> {
    use dialoguer::{FuzzySelect, theme::ColorfulTheme};
    let theme = ColorfulTheme::default();
    if model_items.is_empty() {
        return prompt_custom_model(&theme);
    }

    model_items.push("Custom - enter model name manually".to_string());
    let selection = FuzzySelect::with_theme(&theme)
        .with_prompt(format!("Select default model for {}", provider))
        .items(&mut *model_items)
        .default(0)
        .interact()?;

    if selection == model_items.len() - 1 {
        prompt_custom_model(&theme)
    } else {
        let selected = &model_items[selection];
        let model_name = selected.split(" - ").next().unwrap_or(selected);
        Ok(Some(model_name.to_string()))
    }
}

async fn handle_config_model(provider: String, model: Option<String>) -> Result<()> {
    let service = CredentialService::new().await?;
    if let Some(model_name) = model {
        service.set_default_model(&provider, &model_name).await?;
        println!("Default model for '{}' set to '{}'", provider, model_name);
        return Ok(());
    }

    println!(
        "\x1b[2mFetching available models for {}...\x1b[0m",
        provider
    );
    let mut model_items = fetch_model_items(&service, &provider).await?;
    if let Some(m) = select_model_name(&provider, &mut model_items)? {
        service.set_default_model(&provider, &m).await?;
        println!("Default model for '{}' set to '{}'", provider, m);
    }
    Ok(())
}

async fn handle_config_show(format: String, verbose: bool) -> Result<()> {
    let mut manager = ConfigManager::with_standard_sources();
    let (config, contributed) = manager
        .load_with_sources()
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    if verbose && !contributed.is_empty() {
        println!("# Sources: {}\n", contributed.join(", "));
    }

    match format.as_str() {
        "toml" => {
            let serialized = toml::to_string_pretty(config)
                .map_err(|e| anyhow::anyhow!("Failed to serialize config: {}", e))?;
            println!("{}", serialized);
        }
        "json" => {
            let serialized = serde_json::to_string_pretty(config)
                .map_err(|e| anyhow::anyhow!("Failed to serialize config: {}", e))?;
            println!("{}", serialized);
        }
        _ => anyhow::bail!("Unsupported format: {}", format),
    }
    Ok(())
}

fn handle_config_schema(output: Option<std::path::PathBuf>) -> Result<()> {
    let schema = schemars::schema_for!(crate::config::Config);
    let json = serde_json::to_string_pretty(&schema)
        .map_err(|e| anyhow::anyhow!("Failed to serialize schema: {}", e))?;
    match output {
        Some(path) => {
            std::fs::write(&path, &json)
                .map_err(|e| anyhow::anyhow!("Failed to write schema: {}", e))?;
            println!("Schema written to {}", path.display());
        }
        None => println!("{}", json),
    }
    Ok(())
}

async fn handle_config_export(
    shell: String,
    output: Option<std::path::PathBuf>,
    stdout: bool,
    yes_i_understand_this_prints_secrets: bool,
) -> Result<()> {
    let service = CredentialService::new().await?;
    let lines = service.build_export_lines(&shell).await?;
    let contents = format!("{}\n", lines.join("\n"));

    if stdout {
        CredentialService::require_secret_confirmation(
            yes_i_understand_this_prints_secrets,
            "config export --stdout",
        )?;
        print!("{contents}");
        return Ok(());
    }

    let path = output.unwrap_or_else(|| CredentialService::default_export_path(&shell));
    write_secret_export_file(&path, &contents)?;
    println!("Credentials exported to {}", path.display());
    match shell.as_str() {
        "powershell" | "pwsh" => println!("To use: . {}", path.display()),
        _ => println!("To use: source {}", path.display()),
    }
    Ok(())
}

async fn handle_gateway_set(name: String, key: Option<String>, env: Option<String>) -> Result<()> {
    let service = CredentialService::new().await?;
    let cred_name = require_known_gateway(&name)?;
    let value = resolve_gateway_secret_value(key, env, &cred_name)?;
    service.set_gateway_credential(&name, &value).await?;
    println!(
        "Gateway credential '{}' saved",
        CredentialService::normalize_gateway_name(&name)
    );
    Ok(())
}

async fn handle_gateway_get(
    name: String,
    unmask: bool,
    yes_i_understand_this_prints_secrets: bool,
) -> Result<()> {
    let service = CredentialService::new().await?;
    let cred_name = require_known_gateway(&name)?;
    match service.get_provider_key(&cred_name).await {
        Some(value) => print_masked_or_plain(
            &cred_name,
            &value,
            unmask,
            yes_i_understand_this_prints_secrets,
            "config gateway get --unmask",
        )?,
        None => println!("No value configured for '{}'", cred_name),
    }
    Ok(())
}

async fn handle_gateway_remove(name: String) -> Result<()> {
    let service = CredentialService::new().await?;
    let cred_name = require_known_gateway(&name)?;
    if service.remove_provider_key(&cred_name).await? {
        println!("Gateway credential '{}' removed", cred_name);
    } else {
        println!("No value configured for '{}'", cred_name);
    }
    Ok(())
}

async fn handle_gateway_list(configured: bool) -> Result<()> {
    let service = CredentialService::new().await?;
    println!("\nGateway Credential Status\n");
    for (name, env_var, description, has_value) in service.gateway_status_rows(configured).await {
        println!("  {}", name);
        println!("    Description: {}", description);
        println!("    Environment: {}", env_var);
        println!(
            "    Status: {}",
            if has_value {
                "configured"
            } else {
                "not configured"
            }
        );
        println!();
    }
    Ok(())
}

async fn handle_gateway_config(action: GatewayConfigAction) -> Result<()> {
    match action {
        GatewayConfigAction::Set { name, key, env } => handle_gateway_set(name, key, env).await,
        GatewayConfigAction::Get {
            name,
            unmask,
            yes_i_understand_this_prints_secrets,
        } => handle_gateway_get(name, unmask, yes_i_understand_this_prints_secrets).await,
        GatewayConfigAction::Remove { name } => handle_gateway_remove(name).await,
        GatewayConfigAction::List { configured } => handle_gateway_list(configured).await,
    }
}

/// Handle config subcommands.
pub async fn handle_config_command(action: ConfigAction) -> Result<()> {
    match action {
        ConfigAction::Setup => {
            if interactive_setup().await?.outcome == crate::credentials::SetupOutcome::Cancelled {
                println!("{}", crate::credentials::SETUP_CANCELLED);
            }
            Ok(())
        }
        ConfigAction::Set { provider, key, env } => handle_config_set(provider, key, env).await,
        ConfigAction::Get {
            provider,
            unmask,
            yes_i_understand_this_prints_secrets,
        } => handle_config_get(provider, unmask, yes_i_understand_this_prints_secrets).await,
        ConfigAction::Remove { provider } => handle_config_remove(provider).await,
        ConfigAction::List { configured } => handle_config_list(configured).await,
        ConfigAction::Default { provider } => handle_config_default(provider).await,
        ConfigAction::Model { provider, model } => handle_config_model(provider, model).await,
        ConfigAction::Aws {
            access_key: _,
            secret_key: _,
            region: _,
            profile: _,
        } => {
            println!("AWS credentials configuration will be available in a future update.");
            println!("For now, use environment variables or AWS CLI profiles.");
            Ok(())
        }
        ConfigAction::Show { format, verbose } => handle_config_show(format, verbose).await,
        ConfigAction::Schema { output } => handle_config_schema(output),
        ConfigAction::Path => {
            let db_path = Database::default_db_path()?;
            println!("{}", db_path.display());
            Ok(())
        }
        ConfigAction::Vault { action } => handle_vault_command(action).await,
        ConfigAction::Export {
            shell,
            output,
            stdout,
            yes_i_understand_this_prints_secrets,
        } => {
            handle_config_export(shell, output, stdout, yes_i_understand_this_prints_secrets).await
        }
        ConfigAction::Gateway { action } => handle_gateway_config(action).await,
    }
}
