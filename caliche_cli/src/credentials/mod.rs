//! CLI credential catalog and interactive setup.
//!
//! # Credential layers
//!
//! | Module | Role | Persistence |
//! |--------|------|-------------|
//! | `credentials` (this tree) | Interactive setup UX, provider catalog | None (orchestrates only) |
//! | `storage::credentials` | Secure runtime store, scopes, keyring, env apply | OS keyring + SQLite (`caliche.db`) |
//! | `bootstrap::credentials` | Startup: load secure store, set process env | None (orchestration only) |
//!
//! Runtime and setup both use [`crate::storage::SecureCredentialManager`]. Secrets are never
//! staged in `credentials.toml`.

mod provider;
mod provider_setup;

pub use provider::{is_known_provider, normalize_provider_name, provider_has_env_credentials};
#[cfg(test)]
pub(crate) use provider_setup::make_default_provider;

use anyhow::{Context, Result};
use dialoguer::{FuzzySelect, theme::ColorfulTheme};
use std::collections::HashMap;

use crate::storage::SecureCredentialManager;

/// Every provider caliche knows, with its key variable and description.
/// Offers use [`offered_providers`]; credentials use [`known_providers`].
const KNOWN_PROVIDERS: &[(&str, &str, &str)] = &[
    ("openai", "OPENAI_API_KEY", "OpenAI (GPT-4, GPT-4o)"),
    ("anthropic", "ANTHROPIC_API_KEY", "Anthropic (Claude)"),
    (
        "openrouter",
        "OPENROUTER_API_KEY",
        "OpenRouter (Multi-provider)",
    ),
    (
        "perplexity",
        "PERPLEXITY_API_KEY",
        "Perplexity (Search-augmented)",
    ),
    ("google", "GOOGLE_API_KEY", "Google (Gemini)"),
    ("cohere", "COHERE_API_KEY", "Cohere (Command R)"),
    (
        "bedrock",
        "AWS_ACCESS_KEY_ID",
        "AWS Bedrock (Claude, Llama)",
    ),
    ("azure-openai", "AZURE_OPENAI_API_KEY", "Azure OpenAI"),
    ("lmstudio", "", "LM Studio (Local, no key required)"),
    ("ollama", "", "Ollama (Local, no key required)"),
    (
        "local-inference",
        "",
        "Local Inference (Dripstone, no key required)",
    ),
    (
        "cloudflare-ai-gateway",
        "CLOUDFLARE_AIG_API_KEY",
        "Cloudflare AI Gateway (OpenAI-compatible)",
    ),
];

/// Every known provider, for credentials: storing, exporting and reading
/// keys. A key for a provider this build left out is still kept.
#[must_use]
pub const fn known_providers() -> &'static [(&'static str, &'static str, &'static str)] {
    KNOWN_PROVIDERS
}

/// The providers this build can run, for what the user is offered:
/// [`KNOWN_PROVIDERS`] without the ones [`caliche::llm::ProviderType`]
/// reports as left out of the build.
#[must_use]
pub fn offered_providers() -> &'static [(&'static str, &'static str, &'static str)] {
    static BUILT: std::sync::LazyLock<Vec<(&str, &str, &str)>> = std::sync::LazyLock::new(|| {
        KNOWN_PROVIDERS
            .iter()
            .copied()
            .filter(|(name, _, _)| {
                caliche::llm::ProviderType::feature_left_out_of_build(name).is_none()
            })
            .collect()
    });
    &BUILT
}

/// Every embedding provider caliche knows, with its description.
const KNOWN_EMBEDDING_PROVIDERS: &[(&str, &str)] = &[
    ("openai", "OpenAI text-embedding models"),
    ("bedrock", "AWS Bedrock Titan embeddings"),
    ("openrouter", "OpenRouter embedding models"),
    (
        "cloudflare-ai-gateway",
        "Cloudflare AI Gateway routed embeddings",
    ),
    ("lmstudio", "LM Studio local embeddings"),
    ("ollama", "Ollama local embeddings"),
];

/// The embedding providers this build can run, for `/embedding`, its
/// picker and its completion.
#[must_use]
pub fn offered_embedding_providers() -> &'static [(&'static str, &'static str)] {
    static BUILT: std::sync::LazyLock<Vec<(&str, &str)>> = std::sync::LazyLock::new(|| {
        KNOWN_EMBEDDING_PROVIDERS
            .iter()
            .copied()
            .filter(|(name, _)| {
                caliche::llm::ProviderType::parse_in_build(name)
                    .is_ok_and(|p| caliche::agent::service::default_embedding_model(p).is_some())
            })
            .collect()
    });
    &BUILT
}

/// Gateway integration credentials mapped to environment variables.
pub const GATEWAY_CREDENTIALS: &[(&str, &str, &str)] = &[
    (
        "telegram_bot_token",
        "TELEGRAM_BOT_TOKEN",
        "Telegram Bot Token",
    ),
    (
        "discord_bot_token",
        "DISCORD_BOT_TOKEN",
        "Discord Bot Token",
    ),
    (
        "twilio_account_sid",
        "TWILIO_ACCOUNT_SID",
        "Twilio Account SID",
    ),
    (
        "twilio_auth_token",
        "TWILIO_AUTH_TOKEN",
        "Twilio Auth Token",
    ),
    ("twilio_from", "TWILIO_FROM", "Twilio sender address/number"),
    ("slack_bot_token", "SLACK_BOT_TOKEN", "Slack Bot Token"),
    (
        "slack_signing_secret",
        "SLACK_SIGNING_SECRET",
        "Slack webhook Signing Secret",
    ),
    (
        "teams_webhook_url",
        "TEAMS_WEBHOOK_URL",
        "Microsoft Teams incoming webhook URL",
    ),
];

/// Status of a provider's configuration
#[derive(Debug, Clone)]
pub struct ProviderStatus {
    pub name: String,
    pub description: String,
    pub has_config_key: bool,
    pub has_env_key: bool,
    pub is_configured: bool,
    pub default_model: Option<String>,
    pub is_default: bool,
    /// Whether the local provider is currently running (for ollama/lmstudio)
    pub is_running: Option<bool>,
    /// Available models if running
    pub available_models: Option<Vec<String>>,
}

impl ProviderStatus {
    /// Get a display string for the status
    pub fn status_display(&self) -> &'static str {
        if self.is_configured {
            if self.has_config_key {
                "configured (keyring)"
            } else if self.has_env_key {
                "configured (env)"
            } else {
                "available (no key needed)"
            }
        } else {
            "not configured"
        }
    }
}

async fn model_ids_for_provider(provider: &str, api_key: Option<&str>) -> Vec<String> {
    use caliche::llm::{ModelCatalogOptions, ProviderType, discover_provider_models};
    use caliche::traits::credentials::{
        Credentials, EnvironmentCredentialProvider, StaticCredentialProvider,
    };
    use std::sync::Arc;

    let Ok(provider_type) = provider.parse::<ProviderType>() else {
        return Vec::new();
    };

    let credential_provider: Arc<dyn caliche::traits::credentials::CredentialProvider> =
        match api_key {
            Some(key) if !key.trim().is_empty() => Arc::new(StaticCredentialProvider::new(
                Credentials::ApiKey(key.to_string()),
            )),
            _ => Arc::new(EnvironmentCredentialProvider::new()),
        };

    discover_provider_models(
        provider_type,
        &ModelCatalogOptions::from_environment().with_credential_provider(credential_provider),
    )
    .await
    .map(|entry| entry.models.into_iter().map(|model| model.id).collect())
    .unwrap_or_default()
}

fn validate_model_for_provider(provider: &str, model: String) -> Result<String> {
    if provider == "cloudflare-ai-gateway" {
        caliche::llm::openai_compat::validate_cloudflare_ai_gateway_model(model)
    } else {
        Ok(model)
    }
}

/// Get default/recommended models for a provider.
/// Local runtimes use live discovery so we do not suggest not-installed models.
pub fn get_recommended_models(provider: &str) -> Vec<(&'static str, &'static str)> {
    match provider {
        "openai" => vec![
            ("gpt-5.4", "GPT-5.4 - current default"),
            ("gpt-5.4-mini", "GPT-5.4 Mini - fast"),
            ("gpt-5", "GPT-5"),
            ("gpt-4.1", "GPT-4.1"),
            ("gpt-4o", "GPT-4o"),
        ],
        "anthropic" => vec![
            ("claude-sonnet-4-6", "Claude Sonnet 4.6 - current default"),
            ("claude-sonnet-5", "Claude Sonnet 5"),
            ("claude-opus-5", "Claude Opus 5 - most capable"),
            ("claude-haiku-4-5-20251001", "Claude Haiku 4.5 - fastest"),
            ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5"),
        ],
        "openrouter" => vec![
            ("anthropic/claude-sonnet-4.6", "Claude Sonnet 4.6"),
            ("anthropic/claude-sonnet-5", "Claude Sonnet 5"),
            ("openai/gpt-5.4", "GPT-5.4"),
            ("x-ai/grok-4.6", "Grok 4.6"),
            ("google/gemini-3.5-flash", "Gemini 3.5 Flash"),
            ("anthropic/claude-haiku-4.5", "Claude Haiku 4.5"),
        ],
        "google" => vec![
            ("gemini-2.5-flash", "Gemini 2.5 Flash - current default"),
            ("gemini-3.5-flash", "Gemini 3.5 Flash"),
            ("gemini-3.7-flash", "Gemini 3.7 Flash"),
            ("gemini-2.5-pro", "Gemini 2.5 Pro"),
        ],
        "perplexity" => vec![
            ("sonar-pro", "Sonar Pro - best quality"),
            ("sonar", "Sonar - balanced"),
            ("sonar-reasoning-pro", "Sonar Reasoning Pro"),
            ("sonar-deep-research", "Sonar Deep Research"),
        ],
        "bedrock" => vec![
            ("anthropic.claude-sonnet-4-20250514-v1:0", "Claude Sonnet 4"),
            ("anthropic.claude-opus-4-20250514-v1:0", "Claude Opus 4"),
            (
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "Claude 3.5 Sonnet",
            ),
            ("amazon.nova-pro-v1:0", "Amazon Nova Pro"),
            ("meta.llama3-3-70b-instruct-v1:0", "Llama 3.3 70B"),
        ],
        "lmstudio" | "ollama" => Vec::new(),
        "local-inference" => vec![(
            "hf-internal-testing/tiny-random-LlamaForCausalLM",
            "Tiny local inference smoke-test model",
        )],
        "cohere" => vec![
            ("command-r-plus", "Command R+ - most capable"),
            ("command-r", "Command R - balanced"),
            ("command", "Command - fast"),
        ],
        "azure" => vec![
            ("gpt-4o", "GPT-4o deployment"),
            ("gpt-4", "GPT-4 deployment"),
        ],
        "cloudflare-ai-gateway" => vec![
            ("openai/gpt-5.4", "OpenAI GPT-5.4"),
            ("openai/gpt-5", "OpenAI GPT-5"),
            ("openai/gpt-4o-mini", "OpenAI GPT-4o mini"),
            ("openai/gpt-4o", "OpenAI GPT-4o"),
            ("anthropic/claude-sonnet-5", "Anthropic Claude Sonnet 5"),
            ("anthropic/claude-haiku-4-5", "Anthropic Claude Haiku 4.5"),
            ("anthropic/claude-sonnet-4-5", "Anthropic Claude Sonnet 4.5"),
            ("anthropic/claude-opus-4-5", "Anthropic Claude Opus 4.5"),
            (
                "google-ai-studio/gemini-3.5-flash",
                "Google Gemini 3.5 Flash",
            ),
            ("google-ai-studio/gemini-2.5-pro", "Google Gemini 2.5 Pro"),
            (
                "google-ai-studio/gemini-2.5-flash",
                "Google Gemini 2.5 Flash",
            ),
            (
                "workers-ai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                "Workers AI Llama 3.3 70B",
            ),
        ],
        _ => vec![],
    }
}

/// Setup sections that can be composed into interactive setup flows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredentialSetupKind {
    Providers,
    Gateway,
}

impl CredentialSetupKind {
    pub fn parse(input: &str) -> Option<Self> {
        match input.to_ascii_lowercase().as_str() {
            "provider" | "providers" => Some(Self::Providers),
            "gateway" | "gateways" => Some(Self::Gateway),
            _ => None,
        }
    }

    pub fn all() -> &'static [Self] {
        &[Self::Providers, Self::Gateway]
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Providers => "providers",
            Self::Gateway => "gateway",
        }
    }
}

/// Interactive setup for all credential sections.
pub async fn interactive_setup() -> Result<SetupReport> {
    interactive_setup_for(CredentialSetupKind::all()).await
}

/// The line printed when the user cancels a setup prompt.
pub const SETUP_CANCELLED: &str = "Setup cancelled.";

/// How an interactive setup ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupOutcome {
    /// Every prompt was answered.
    Completed,
    /// The user cancelled a prompt (Ctrl+C). Choices saved before it stay
    /// saved and in use.
    Cancelled,
}

/// What an interactive setup did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupReport {
    /// Whether every prompt was answered.
    pub outcome: SetupOutcome,
    /// The provider and model the user made the default, if any. The next
    /// launch already starts with them; a running session switches to them.
    pub default: Option<(String, String)>,
}

/// Whether `err` comes from a prompt the user cancelled with Ctrl+C.
#[must_use]
pub fn is_prompt_cancelled(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        let io = cause
            .downcast_ref::<dialoguer::Error>()
            .map(|dialoguer::Error::IO(io)| io)
            .or_else(|| cause.downcast_ref::<std::io::Error>());
        io.is_some_and(|io| io.kind() == std::io::ErrorKind::Interrupted)
    })
}

/// One line for a provider without a key: what is missing and the command
/// that fixes it.
#[must_use]
pub fn missing_key_line(provider: &str) -> String {
    let variable = known_providers()
        .iter()
        .find(|(id, _, _)| *id == provider)
        .map(|(_, variable, _)| *variable)
        .filter(|variable| !variable.is_empty());
    let set_hint = variable.map(|v| format!(" or set {v}")).unwrap_or_default();
    format!("{provider} has no API key. Run /creds setup providers{set_hint}.")
}

/// Print how a setup run before the prompt ended, in one line when it did
/// not complete. Startup goes on either way.
pub fn report_startup_setup(result: &Result<SetupReport>) {
    match result {
        Ok(report) if report.outcome == SetupOutcome::Cancelled => println!("{SETUP_CANCELLED}"),
        Ok(_) => {}
        Err(err) if is_prompt_cancelled(err) => println!("{SETUP_CANCELLED}"),
        Err(err) => println!("Setup failed: {err:#}"),
    }
}

/// Interactive setup for selected credential sections.
///
/// A cancelled prompt ends setup with [`SetupOutcome::Cancelled`], not an
/// error. Each key and choice is saved and put in use as it is entered.
pub async fn interactive_setup_for(kinds: &[CredentialSetupKind]) -> Result<SetupReport> {
    println!("\n\x1b[1;34mCaliche Credential Setup\x1b[0m");
    println!("============================\n");

    let manager = SecureCredentialManager::new().await?;
    let mut default = None;
    let outcome = match run_setup_sections(&manager, kinds, &mut default).await {
        Ok(()) => {
            print_setup_complete(kinds);
            SetupOutcome::Completed
        }
        Err(err) if is_prompt_cancelled(&err) => SetupOutcome::Cancelled,
        Err(err) => return Err(err),
    };
    Ok(SetupReport { outcome, default })
}

async fn run_setup_sections(
    manager: &SecureCredentialManager,
    kinds: &[CredentialSetupKind],
    default: &mut Option<(String, String)>,
) -> Result<()> {
    let theme = ColorfulTheme::default();
    if kinds.contains(&CredentialSetupKind::Providers) {
        print_provider_status(manager).await;
        println!();
        run_provider_setup(manager, &theme, default).await?;
    }
    if kinds.contains(&CredentialSetupKind::Gateway) {
        print_gateway_status(manager).await;
        println!();
        run_gateway_setup(manager, &theme).await?;
    }
    Ok(())
}

fn print_setup_complete(kinds: &[CredentialSetupKind]) {
    println!("\n\x1b[32mSetup complete!\x1b[0m");
    println!("Credentials saved to system keyring.");
    println!("\nYou can also set credentials via environment variables.");
    if kinds.contains(&CredentialSetupKind::Providers) {
        println!("Run '\x1b[36mcaliche config list\x1b[0m' to see provider configuration.");
    }
    if kinds.contains(&CredentialSetupKind::Gateway) {
        println!("Run '\x1b[36mcaliche config gateway list\x1b[0m' to see gateway credentials.");
    }
}

/// Build a Cloudflare AI Gateway credential JSON payload for the secure store.
pub fn cloudflare_ai_gateway_credential_payload(
    api_key: &str,
    api_key_role: &str,
    account_id: Option<&str>,
    gateway_id: Option<&str>,
    auth_token: Option<&str>,
) -> Result<String> {
    let mut payload = HashMap::new();
    if api_key_role == "gateway" {
        payload.insert("cf_aig_authorization".to_string(), api_key.to_string());
    } else {
        payload.insert("api_key".to_string(), api_key.to_string());
    }

    if let Some(account_id) = account_id.filter(|v| !v.trim().is_empty()) {
        payload.insert("account_id".to_string(), account_id.to_string());
    }
    if let Some(gateway_id) = gateway_id.filter(|v| !v.trim().is_empty()) {
        payload.insert("gateway_id".to_string(), gateway_id.to_string());
    }
    if api_key_role != "gateway"
        && let Some(auth_token) = auth_token.filter(|v| !v.trim().is_empty())
    {
        payload.insert("cf_aig_authorization".to_string(), auth_token.to_string());
    }

    serde_json::to_string(&payload).context("failed to serialize Cloudflare AI Gateway settings")
}

fn parse_structured_credential(raw: &str) -> Option<HashMap<String, String>> {
    let trimmed = raw.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

async fn print_provider_status(manager: &SecureCredentialManager) {
    println!("Current provider status:\n");
    for status in manager.list_configured().await {
        let marker = if status.is_default {
            " \x1b[33m(default)\x1b[0m"
        } else {
            ""
        };
        let status_str = if status.is_configured {
            "\x1b[32mconfigured\x1b[0m"
        } else {
            "\x1b[31mnot configured\x1b[0m"
        };
        println!("  \x1b[1m{}\x1b[0m - {}{}", status.name, status_str, marker);
        if let Some(ref model) = status.default_model {
            println!("    Model: \x1b[36m{}\x1b[0m", model);
        }
    }
}

async fn print_gateway_status(manager: &SecureCredentialManager) {
    println!("Gateway credential status:\n");
    for (name, _, description) in GATEWAY_CREDENTIALS {
        let configured = manager.get_api_key(name).await.is_some();
        let status_str = if configured {
            "\x1b[32mconfigured\x1b[0m"
        } else {
            "\x1b[31mnot configured\x1b[0m"
        };
        println!(
            "  \x1b[1m{}\x1b[0m - {} ({})",
            name, status_str, description
        );
    }
}

async fn configure_cloudflare_ai_gateway(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    api_key: &str,
) -> Result<()> {
    let existing = manager
        .get_api_key("cloudflare-ai-gateway")
        .await
        .and_then(|raw| parse_structured_credential(&raw));

    let account_id = provider_setup::prompt_cloudflare_account_id(theme, existing.as_ref())?;
    let api_key_role = provider_setup::prompt_cloudflare_api_key_role(theme, existing.as_ref())?;
    let gateway_id = provider_setup::prompt_cloudflare_gateway_id(theme, existing.as_ref())?;
    let auth_token =
        provider_setup::prompt_optional_gateway_auth_token(theme, existing.as_ref(), api_key_role)?;

    let payload = cloudflare_ai_gateway_credential_payload(
        api_key,
        api_key_role,
        Some(account_id.trim()),
        Some(&gateway_id),
        auth_token.as_deref(),
    )?;
    manager
        .set_api_key("cloudflare-ai-gateway", &payload)
        .await?;

    println!("\x1b[32mCloudflare AI Gateway settings saved.\x1b[0m\n");
    Ok(())
}

async fn run_provider_setup(
    manager: &SecureCredentialManager,
    theme: &ColorfulTheme,
    default: &mut Option<(String, String)>,
) -> Result<()> {
    loop {
        let configured = manager.list_configured().await;
        let provider_items = provider_setup::provider_menu_items(&configured);

        let selection = FuzzySelect::with_theme(theme)
            .with_prompt("Select a provider to configure")
            .items(&provider_items)
            .default(0)
            .interact()?;

        if selection >= offered_providers().len() {
            break;
        }

        let (provider, _, desc) = offered_providers()[selection];
        provider_setup::configure_one_provider(manager, theme, provider, desc, default).await?;
    }
    Ok(())
}

async fn run_gateway_setup(manager: &SecureCredentialManager, theme: &ColorfulTheme) -> Result<()> {
    use dialoguer::{Confirm, Input, Password};

    println!("\n\x1b[1mGateway integration credentials (optional)\x1b[0m");
    println!("Configure Telegram/Discord/Twilio/Slack/Teams credentials for gateway adapters.\n");

    loop {
        let gateway_items: Vec<String> = {
            let mut items = Vec::with_capacity(GATEWAY_CREDENTIALS.len() + 1);
            for (name, _env_var, description) in GATEWAY_CREDENTIALS {
                let configured = if manager.get_api_key(name).await.is_some() {
                    " [configured]"
                } else {
                    ""
                };
                items.push(format!("{name} - {description}{configured}"));
            }
            items.push("Done - finish gateway setup".to_string());
            items
        };

        let selection = FuzzySelect::with_theme(theme)
            .with_prompt("Select a gateway credential to configure")
            .items(&gateway_items)
            .default(0)
            .interact()?;

        if selection >= GATEWAY_CREDENTIALS.len() {
            break;
        }

        let (name, _env_var, description) = GATEWAY_CREDENTIALS[selection];
        println!("\n\x1b[1mConfiguring {}\x1b[0m - {}\n", name, description);

        let existing = manager.get_api_key(name).await;
        let should_update = if existing.is_some() {
            Confirm::with_theme(theme)
                .with_prompt(format!(
                    "Credential for {} already configured. Update it?",
                    name
                ))
                .default(false)
                .interact()?
        } else {
            true
        };

        if !should_update {
            println!("\x1b[2mKeeping existing credential.\x1b[0m\n");
            continue;
        }

        let value = if name.contains("token") {
            Password::with_theme(theme)
                .with_prompt(format!("Value for {}", name))
                .interact()?
        } else {
            Input::with_theme(theme)
                .with_prompt(format!("Value for {}", name))
                .interact_text()?
        };

        if value.is_empty() {
            if existing.is_none() {
                println!("No value entered, skipping.\n");
                continue;
            }
            println!("\x1b[2mKeeping existing credential.\x1b[0m\n");
            continue;
        }

        manager.set_api_key(name, &value).await?;
        println!("\x1b[32mCredential saved for {}.\x1b[0m\n", name);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliche::llm::ProviderType;

    #[test]
    fn cloudflare_gateway_payload_uses_gateway_token_role() {
        let payload = cloudflare_ai_gateway_credential_payload(
            "gw-token",
            "gateway",
            Some("acct"),
            Some("default"),
            None,
        )
        .expect("payload");
        let map: HashMap<String, String> = serde_json::from_str(&payload).expect("json");
        assert_eq!(
            map.get("cf_aig_authorization").map(String::as_str),
            Some("gw-token")
        );
        assert!(!map.contains_key("api_key"));
        assert_eq!(map.get("account_id").map(String::as_str), Some("acct"));
    }

    #[test]
    fn cloudflare_provider_payload_keeps_optional_auth_token() {
        let payload = cloudflare_ai_gateway_credential_payload(
            "provider-key",
            "provider",
            Some("acct"),
            Some("gw"),
            Some("auth"),
        )
        .expect("payload");
        let map: HashMap<String, String> = serde_json::from_str(&payload).expect("json");
        assert_eq!(map.get("api_key").map(String::as_str), Some("provider-key"));
        assert_eq!(
            map.get("cf_aig_authorization").map(String::as_str),
            Some("auth")
        );
    }

    #[test]
    fn provider_status_display_keyring() {
        let status = ProviderStatus {
            name: "openai".to_string(),
            description: "OpenAI".to_string(),
            has_config_key: true,
            has_env_key: false,
            is_configured: true,
            default_model: None,
            is_default: false,
            is_running: None,
            available_models: None,
        };
        assert_eq!(status.status_display(), "configured (keyring)");
    }

    #[test]
    fn provider_catalog_includes_openai() {
        assert!(
            offered_providers()
                .iter()
                .any(|(name, _, _)| *name == "openai")
        );
        assert!(is_known_provider("openai"));
        assert!(is_known_provider("azure"));
    }

    /// Every provider the CLI offers (completion, pickers, setup, doctor)
    /// parses in the build under test, and every known provider is either
    /// offered or reported as left out of the build.
    #[test]
    fn every_offered_provider_parses_in_this_build() {
        let unparsed: Vec<(&str, String)> = offered_providers()
            .iter()
            .filter_map(|(name, _, _)| {
                ProviderType::parse_in_build(name)
                    .err()
                    .map(|err| (*name, err))
            })
            .collect();
        let dropped: Vec<&str> = KNOWN_PROVIDERS
            .iter()
            .map(|(name, _, _)| *name)
            .filter(|name| {
                !offered_providers()
                    .iter()
                    .any(|(offered, _, _)| offered == name)
            })
            .collect();
        let left_out: Vec<&str> = KNOWN_PROVIDERS
            .iter()
            .map(|(name, _, _)| *name)
            .filter(|name| ProviderType::feature_left_out_of_build(name).is_some())
            .collect();

        assert_eq!(unparsed, Vec::new());
        assert_eq!(dropped, left_out);
    }

    /// The default build offers the providers it can run; credentials keep
    /// every known provider.
    #[test]
    #[cfg(not(any(
        feature = "bedrock",
        feature = "azure-openai",
        feature = "local-inference-base"
    )))]
    fn the_default_build_offers_what_it_runs_and_knows_every_key() {
        let names = |rows: &[(&str, &str, &str)]| -> Vec<String> {
            rows.iter()
                .map(|(name, _, _)| (*name).to_string())
                .collect()
        };
        let embedding: Vec<&str> = offered_embedding_providers()
            .iter()
            .map(|(name, _)| *name)
            .collect();

        assert_eq!(
            names(offered_providers()),
            vec![
                "openai",
                "anthropic",
                "openrouter",
                "perplexity",
                "google",
                "cohere",
                "lmstudio",
                "ollama",
                "cloudflare-ai-gateway",
            ]
        );
        assert_eq!(
            names(known_providers()),
            vec![
                "openai",
                "anthropic",
                "openrouter",
                "perplexity",
                "google",
                "cohere",
                "bedrock",
                "azure-openai",
                "lmstudio",
                "ollama",
                "local-inference",
                "cloudflare-ai-gateway",
            ]
        );
        assert_eq!(
            embedding,
            vec![
                "openai",
                "openrouter",
                "cloudflare-ai-gateway",
                "lmstudio",
                "ollama"
            ]
        );
    }

    #[test]
    fn recommended_models_use_current_ids() {
        let anthropic: Vec<_> = get_recommended_models("anthropic")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(anthropic.contains(&"claude-sonnet-4-6"));
        assert!(!anthropic.contains(&"claude-sonnet-4-20250514"));

        let openrouter: Vec<_> = get_recommended_models("openrouter")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(openrouter.contains(&"anthropic/claude-sonnet-4.6"));
        assert!(openrouter.contains(&"x-ai/grok-4.6"));

        let openai: Vec<_> = get_recommended_models("openai")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(openai.contains(&"gpt-5.4"));

        let google: Vec<_> = get_recommended_models("google")
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(google.contains(&"gemini-2.5-flash"));
        assert!(!google.contains(&"gemini-1.5-pro"));
    }
}
