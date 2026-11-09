//! Interactive provider configuration helpers.

use super::{
    get_recommended_models, model_ids_for_provider, offered_providers, parse_structured_credential,
    validate_model_for_provider,
};
use crate::storage::SecureCredentialManager;
use anyhow::{Context, Result};
use caliche_contracts::credential_env;
use dialoguer::{Confirm, FuzzySelect, Input, Password, theme::ColorfulTheme};
use std::collections::HashMap;
use std::io::{self, Write};

pub(super) fn prompt_cloudflare_account_id(
    theme: &ColorfulTheme,
    existing: Option<&HashMap<String, String>>,
) -> Result<String> {
    let existing_account_id = existing
        .and_then(|m| m.get("account_id").cloned())
        .or_else(|| credential_env::var("CLOUDFLARE_ACCOUNT_ID").ok());
    let account_id = if let Some(existing) = existing_account_id {
        Input::with_theme(theme)
            .with_prompt("Cloudflare account ID")
            .default(existing)
            .interact_text()?
    } else {
        Input::with_theme(theme)
            .with_prompt("Cloudflare account ID")
            .interact_text()?
    };
    accept_cloudflare_account_id(account_id)
}

/// Store the entered Cloudflare account ID for this process and return it.
fn accept_cloudflare_account_id(account_id: String) -> Result<String> {
    if account_id.trim().is_empty() {
        anyhow::bail!("Cloudflare account ID cannot be empty");
    }
    credential_env::set("CLOUDFLARE_ACCOUNT_ID", account_id.trim());
    Ok(account_id)
}

pub(super) fn prompt_cloudflare_api_key_role(
    theme: &ColorfulTheme,
    existing: Option<&HashMap<String, String>>,
) -> Result<&'static str> {
    let auth_mode_items = [
        "Cloudflare AI Gateway token (stored/BYOK provider keys)".to_string(),
        "Upstream provider API key passthrough".to_string(),
    ];
    let current_role = if existing
        .is_some_and(|m| m.contains_key("cf_aig_authorization") && !m.contains_key("api_key"))
    {
        "gateway"
    } else {
        "provider"
    };
    let default_role_index = usize::from(current_role == "provider");
    let auth_mode = FuzzySelect::with_theme(theme)
        .with_prompt("What does the configured API key authenticate?")
        .items(&auth_mode_items)
        .default(default_role_index)
        .interact()?;
    if auth_mode == 0 {
        Ok("gateway")
    } else {
        Ok("provider")
    }
}

pub(super) fn prompt_cloudflare_gateway_id(
    theme: &ColorfulTheme,
    existing: Option<&HashMap<String, String>>,
) -> Result<String> {
    let gateway_id = existing
        .and_then(|m| m.get("gateway_id").cloned())
        .or_else(|| credential_env::var("CLOUDFLARE_AIG_GATEWAY_ID").ok())
        .unwrap_or_else(|| "default".to_string());
    let gateway_id: String = Input::with_theme(theme)
        .with_prompt("Cloudflare AI Gateway ID")
        .default(gateway_id)
        .interact_text()?;
    Ok(accept_cloudflare_gateway_id(&gateway_id))
}

/// Store the entered gateway ID (`default` when blank) for this process and return it.
fn accept_cloudflare_gateway_id(gateway_id: &str) -> String {
    let gateway_id = if gateway_id.trim().is_empty() {
        "default".to_string()
    } else {
        gateway_id.trim().to_string()
    };
    credential_env::set("CLOUDFLARE_AIG_GATEWAY_ID", gateway_id.as_str());
    gateway_id
}

pub(super) fn prompt_optional_gateway_auth_token(
    theme: &ColorfulTheme,
    existing: Option<&HashMap<String, String>>,
    api_key_role: &str,
) -> Result<Option<String>> {
    if api_key_role != "provider" {
        return Ok(None);
    }
    let existing_auth_token = existing.is_some_and(|m| m.contains_key("cf_aig_authorization"))
        || credential_env::is_set("CLOUDFLARE_AIG_AUTH_TOKEN");
    let should_update_auth = Confirm::with_theme(theme)
        .with_prompt(if existing_auth_token {
            "Update optional Cloudflare AI Gateway auth token?"
        } else {
            "Set optional Cloudflare AI Gateway auth token?"
        })
        .default(false)
        .interact()?;

    if should_update_auth {
        return read_updated_gateway_auth_token(theme);
    }
    if let Some(existing_token) = existing.and_then(|m| m.get("cf_aig_authorization").cloned()) {
        return Ok(Some(existing_token));
    }
    if let Ok(env_token) = credential_env::var("CLOUDFLARE_AIG_AUTH_TOKEN") {
        return Ok(Some(env_token));
    }
    Ok(None)
}

fn read_updated_gateway_auth_token(theme: &ColorfulTheme) -> Result<Option<String>> {
    let token: String = Password::with_theme(theme)
        .with_prompt("Cloudflare AI Gateway auth token (leave empty to clear)")
        .allow_empty_password(true)
        .interact()?;
    Ok(accept_gateway_auth_token(&token))
}

/// Store the entered gateway auth token for this process; blank means none.
fn accept_gateway_auth_token(token: &str) -> Option<String> {
    if token.trim().is_empty() {
        return None;
    }
    credential_env::set("CLOUDFLARE_AIG_AUTH_TOKEN", token.trim());
    Some(token.trim().to_string())
}

pub(super) async fn configure_bedrock(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
) -> Result<()> {
    let has_aws_creds = manager.get_api_key("bedrock").await.is_some()
        || (credential_env::is_set("AWS_ACCESS_KEY_ID")
            && credential_env::is_set("AWS_SECRET_ACCESS_KEY"));
    let should_update = if has_aws_creds {
        Confirm::with_theme(theme)
            .with_prompt("AWS credentials already configured. Update them?")
            .default(false)
            .interact()?
    } else {
        true
    };

    if !should_update {
        println!("\x1b[2mKeeping existing AWS credentials.\x1b[0m\n");
        return Ok(());
    }

    println!("AWS Bedrock requires AWS credentials.\n");
    let access_key: String = Input::with_theme(theme)
        .with_prompt("AWS Access Key ID")
        .interact_text()?;
    let secret_key: String = Password::with_theme(theme)
        .with_prompt("AWS Secret Access Key")
        .interact()?;
    let region: String = Input::with_theme(theme)
        .with_prompt("AWS Region")
        .default("us-east-1".to_string())
        .interact_text()?;
    let mut payload = HashMap::new();
    payload.insert("access_key".to_string(), access_key);
    payload.insert("secret_key".to_string(), secret_key);
    payload.insert("region".to_string(), region);
    let credential =
        serde_json::to_string(&payload).context("failed to serialize Bedrock AWS credentials")?;
    manager.set_api_key("bedrock", &credential).await?;
    println!("\x1b[32mAWS credentials saved.\x1b[0m\n");
    Ok(())
}

fn is_local_provider(provider: &str) -> bool {
    matches!(provider, "lmstudio" | "ollama" | "local-inference")
}

fn local_provider_message(provider: &str) -> &'static str {
    match provider {
        "lmstudio" => "LM Studio doesn't require an API key (local provider).\n",
        "ollama" => "Ollama doesn't require an API key (local provider).\n",
        "local-inference" => "Local inference doesn't require an API key (local provider).\n",
        _ => "",
    }
}

async fn configure_generic_api_key(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    provider: &str,
) -> Result<bool> {
    let existing_key = manager.get_api_key(provider).await;
    let should_update = if existing_key.is_some() {
        Confirm::with_theme(theme)
            .with_prompt(format!(
                "API key for {provider} already configured. Update it?"
            ))
            .default(false)
            .interact()?
    } else {
        true
    };

    if should_update {
        save_or_keep_api_key(manager, theme, provider, existing_key.as_deref()).await
    } else {
        println!("\x1b[2mKeeping existing API key.\x1b[0m\n");
        maybe_configure_existing_cloudflare(manager, theme, provider, existing_key.as_deref())
            .await?;
        Ok(true)
    }
}

async fn save_or_keep_api_key(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    provider: &str,
    existing_key: Option<&str>,
) -> Result<bool> {
    let api_key: String = Password::with_theme(theme)
        .with_prompt(format!("API Key for {provider}"))
        .interact()?;

    if api_key.is_empty() {
        if existing_key.is_none() {
            println!("No key entered, skipping provider.\n");
            return Ok(false);
        }
        println!("\x1b[2mKeeping existing API key.\x1b[0m\n");
        return Ok(true);
    }
    if provider == "cloudflare-ai-gateway" {
        super::configure_cloudflare_ai_gateway(manager, theme, &api_key).await?;
    } else {
        manager.set_api_key(provider, &api_key).await?;
        println!("\x1b[32mAPI key saved for {provider}.\x1b[0m\n");
    }
    Ok(true)
}

async fn maybe_configure_existing_cloudflare(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    provider: &str,
    existing_key: Option<&str>,
) -> Result<()> {
    if provider != "cloudflare-ai-gateway" {
        return Ok(());
    }
    let Some(existing) = existing_key else {
        return Ok(());
    };
    let plain_key = parse_structured_credential(existing)
        .and_then(|m| {
            m.get("api_key")
                .or_else(|| m.get("cf_aig_authorization"))
                .cloned()
        })
        .unwrap_or_else(|| existing.to_string());
    super::configure_cloudflare_ai_gateway(manager, theme, &plain_key).await
}

async fn select_default_model(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    provider: &str,
) -> Result<()> {
    println!("\x1b[2mFetching available models...\x1b[0m");
    io::stdout().flush()?;

    let api_key = manager.get_api_key(provider).await;
    let catalog_key = api_key.as_deref().and_then(catalog_key_from_raw);
    let fetched_models = model_ids_for_provider(provider, catalog_key.as_deref()).await;
    let recommended = get_recommended_models(provider);

    let mut model_items: Vec<String> = Vec::new();
    for model in &fetched_models {
        let suffix = recommended
            .iter()
            .find(|(recommended_model, _)| recommended_model == &model.as_str())
            .map(|(_, desc)| format!(" - {desc} [recommended]"))
            .unwrap_or_default();
        model_items.push(format!("{model}{suffix}"));
    }
    model_items.push("Skip - don't set default model".to_string());
    model_items.push("Custom - enter model name manually".to_string());

    if model_items.len() > 2 {
        select_model_from_list(manager, theme, provider, &model_items).await
    } else {
        prompt_custom_model(manager, theme, provider).await
    }
}

fn catalog_key_from_raw(raw: &str) -> Option<String> {
    if let Some(map) = parse_structured_credential(raw) {
        map.get("api_key")
            .or_else(|| map.get("cf_aig_authorization"))
            .cloned()
    } else {
        Some(raw.to_string())
    }
}

async fn select_model_from_list(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    provider: &str,
    model_items: &[String],
) -> Result<()> {
    let model_selection = FuzzySelect::with_theme(theme)
        .with_prompt("Select default model")
        .items(model_items)
        .default(0)
        .interact()?;

    let selected_model = if model_selection == model_items.len() - 1 {
        let custom: String = Input::with_theme(theme)
            .with_prompt("Enter model name")
            .interact_text()?;
        if custom.is_empty() {
            None
        } else {
            Some(custom)
        }
    } else if model_selection == model_items.len() - 2 {
        None
    } else {
        let selected = &model_items[model_selection];
        let model_name = selected.split(" - ").next().unwrap_or(selected);
        Some(model_name.to_string())
    };

    if let Some(model) = selected_model {
        let model = validate_model_for_provider(provider, model)?;
        manager.set_default_model(provider, &model).await?;
        println!("\x1b[32mDefault model set to '{model}'\x1b[0m\n");
    }
    Ok(())
}

async fn prompt_custom_model(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    provider: &str,
) -> Result<()> {
    let custom: String = Input::with_theme(theme)
        .with_prompt("Enter model name (or press Enter to skip)")
        .allow_empty(true)
        .interact_text()?;

    if custom.is_empty() {
        return Ok(());
    }
    let custom = validate_model_for_provider(provider, custom)?;
    manager.set_default_model(provider, &custom).await?;
    println!("\x1b[32mDefault model set to '{custom}'\x1b[0m\n");
    Ok(())
}

async fn maybe_make_default_provider(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    provider: &str,
    default: &mut Option<(String, String)>,
) -> Result<()> {
    let no_default_provider = manager.default_provider().await.is_none();
    let make_default = Confirm::with_theme(theme)
        .with_prompt(format!("Make {provider} the default provider?"))
        .default(no_default_provider)
        .interact()?;

    if make_default {
        let model = make_default_provider(manager, provider).await?;
        println!("\x1b[32m{provider} set as default provider.\x1b[0m\n");
        *default = Some((provider.to_string(), model));
    }
    println!();
    Ok(())
}

/// Make `provider` the default: the stored default, and the selection the
/// next launch starts with. The model is the default model chosen for it,
/// else its built-in default. Returns that model.
pub(crate) async fn make_default_provider(
    manager: &SecureCredentialManager,
    provider: &str,
) -> Result<String> {
    let provider_type: caliche::llm::ProviderType = provider
        .parse()
        .map_err(|err| anyhow::anyhow!("Invalid provider: {err}"))?;
    manager.set_default_provider(provider).await?;
    let model = match manager.get_default_model(provider).await {
        Some(model) => model,
        None => caliche::agent::service::resolve_model_for_provider(provider_type, None),
    };
    let database = crate::storage::Database::open().await.ok();
    crate::boot_provider::save_last_selection(database.as_ref(), provider, &model, None, None)
        .await;
    Ok(model)
}

pub(super) async fn configure_one_provider(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    provider: &str,
    desc: &str,
    default: &mut Option<(String, String)>,
) -> Result<()> {
    println!("\n\x1b[1mConfiguring {provider}\x1b[0m - {desc}\n");

    if provider == "bedrock" {
        configure_bedrock(manager, theme).await?;
    } else if is_local_provider(provider) {
        println!("{}", local_provider_message(provider));
    } else if !configure_generic_api_key(manager, theme, provider).await? {
        return Ok(());
    }

    select_default_model(manager, theme, provider).await?;
    maybe_make_default_provider(manager, theme, provider, default).await
}

pub(super) fn provider_menu_items(
    configured: &[crate::credentials::ProviderStatus],
) -> Vec<String> {
    offered_providers()
        .iter()
        .map(|(name, _, desc)| {
            let status = configured
                .iter()
                .find(|s| s.name == *name)
                .map(|s| if s.is_configured { " [configured]" } else { "" })
                .unwrap_or("");
            format!("{name} - {desc}{status}")
        })
        .chain(std::iter::once("Done - finish provider setup".to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    //! The Cloudflare prompts need a terminal, so these drive the step that
    //! takes the entered text: it trims, applies defaults, and stores the value.

    use super::{
        accept_cloudflare_account_id, accept_cloudflare_gateway_id, accept_gateway_auth_token,
    };
    use caliche_contracts::credential_env;
    use std::env::VarError;

    #[test]
    #[serial_test::serial(env)]
    fn cloudflare_prompt_values_resolve_and_env_stays_unset() {
        let mut env = crate::test_env::EnvGuard::acquire();
        for key in [
            "CLOUDFLARE_ACCOUNT_ID",
            "CLOUDFLARE_AIG_GATEWAY_ID",
            "CLOUDFLARE_AIG_AUTH_TOKEN",
        ] {
            env.remove(key);
        }

        let account_id =
            accept_cloudflare_account_id("  cf-account-1  ".to_string()).expect("account id");
        let gateway_id = accept_cloudflare_gateway_id(" my-gateway ");
        let token = accept_gateway_auth_token(" cf-aig-token ");

        assert_eq!(account_id, "  cf-account-1  ");
        assert_eq!(gateway_id, "my-gateway");
        assert_eq!(token.as_deref(), Some("cf-aig-token"));
        for (key, value) in [
            ("CLOUDFLARE_ACCOUNT_ID", "cf-account-1"),
            ("CLOUDFLARE_AIG_GATEWAY_ID", "my-gateway"),
            ("CLOUDFLARE_AIG_AUTH_TOKEN", "cf-aig-token"),
        ] {
            assert_eq!(credential_env::var(key).as_deref(), Ok(value), "{key}");
            assert_eq!(std::env::var(key), Err(VarError::NotPresent), "{key}");
        }
    }

    #[test]
    fn blank_cloudflare_inputs_are_rejected_or_defaulted() {
        let error = accept_cloudflare_account_id("   ".to_string()).expect_err("blank id");
        assert_eq!(error.to_string(), "Cloudflare account ID cannot be empty");
        assert_eq!(accept_gateway_auth_token("  "), None);
    }
}
