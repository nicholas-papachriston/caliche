use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use crate::credentials::{SETUP_CANCELLED, SetupOutcome, SetupReport};
use crate::storage::SecureCredentialManager;
use anyhow::Result;
use caliche::AgentService;

/// The running session a `/creds setup` switches when the user makes a
/// provider the default.
pub(crate) struct CredsSession<'a> {
    pub(crate) state: &'a mut InteractiveState,
    pub(crate) service: &'a mut AgentService,
}

async fn open_credential_manager() -> Result<SecureCredentialManager, SlashCommandResult> {
    SecureCredentialManager::new()
        .await
        .map_err(|e| SlashCommandResult::Error(format!("Credential manager error: {e}")))
}

fn empty_status_lines() -> Vec<String> {
    vec![
        "No credentials available for this directory.".to_string(),
        String::new(),
        "Try:  /creds setup   -- interactive setup wizard".to_string(),
        "      /gateway setup   -- gateway-only setup wizard".to_string(),
        "      /creds import   -- load from .env file".to_string(),
        "      /creds set <provider> <key>   -- set manually".to_string(),
        "      /gateway setup   -- setup + start gateway runtime".to_string(),
        "      /creds link <dir>   -- use another project's creds".to_string(),
    ]
}

fn format_status_sources(status: &crate::storage::ScopeStatus) -> Vec<String> {
    let mut lines = Vec::new();
    for source in &status.sources {
        let marker = if source.is_linked { " [linked]" } else { "" };
        lines.push(format!("  {}{}", source.label, marker));
        for provider in &source.providers {
            lines.push(format!("    {provider}"));
        }
        lines.push(String::new());
    }
    if !status.env_providers.is_empty() {
        lines.push("  Environment variables:".to_string());
        for provider in &status.env_providers {
            lines.push(format!("    {provider}"));
        }
        lines.push(String::new());
    }
    lines
}

async fn handle_creds_status() -> Result<SlashCommandResult> {
    let cm = match open_credential_manager().await {
        Ok(manager) => manager,
        Err(error) => return Ok(error),
    };
    let status = cm.status_for_current_scope().await;

    let mut lines = vec![
        "Credential Vault Status".to_string(),
        "=======================".to_string(),
        format!("Directory: {}", status.cwd),
        String::new(),
    ];

    if status.sources.is_empty() && status.env_providers.is_empty() {
        lines.extend(empty_status_lines());
    } else {
        lines.extend(format_status_sources(&status));
    }

    Ok(SlashCommandResult::ShowPopup {
        title: "Credentials".to_string(),
        content: lines.join("\n"),
    })
}

async fn handle_creds_set(args: &[String]) -> Result<SlashCommandResult> {
    let Some(provider) = args.get(1).cloned() else {
        return Ok(SlashCommandResult::Error(
            "Usage: /creds set <provider> <api-key>\nExample: /creds set openai sk-...".to_string(),
        ));
    };
    let Some(key) = args.get(2).cloned() else {
        return Ok(SlashCommandResult::Error(format!(
            "Usage: /creds set {provider} <api-key>"
        )));
    };
    let cm = match open_credential_manager().await {
        Ok(manager) => manager,
        Err(error) => return Ok(error),
    };
    match cm.set_api_key(&provider, &key).await {
        Ok(()) => Ok(SlashCommandResult::StatusMessage(format!(
            "Stored API key for '{provider}' in global keychain.\n\
                 Run '/provider {provider}' to start using it in this session."
        ))),
        Err(e) => Ok(SlashCommandResult::Error(format!(
            "Failed to store key: {e}"
        ))),
    }
}

async fn handle_creds_import() -> Result<SlashCommandResult> {
    use crate::storage::env_scanner;
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

    let detected = env_scanner::scan_for_credentials(&cwd, false);
    if detected.is_empty() {
        return Ok(SlashCommandResult::StatusMessage(
            "No provider credentials found in .env files in the current directory.\n\nCheck for: .env, .env.local, .env.development".to_string()
        ));
    }

    let cm = match open_credential_manager().await {
        Ok(manager) => manager,
        Err(error) => return Ok(error),
    };

    let scope_hash = cm.current_scope_hash().to_string();
    let existing = cm.get_scope_credentials(&scope_hash);
    let new_creds: Vec<_> = detected
        .into_iter()
        .filter(|d| !existing.contains_key(&d.provider_name))
        .collect();

    if new_creds.is_empty() {
        return Ok(SlashCommandResult::StatusMessage(
            "All credentials from .env are already stored in the keychain for this directory."
                .to_string(),
        ));
    }

    let scope_label = cm
        .current_scope()
        .map(|s| s.path.clone())
        .unwrap_or_else(|| "global".to_string());

    match cm.ingest_env_credentials(&new_creds).await {
        Ok(()) => {
            let names: Vec<String> = new_creds.iter().map(|d| d.provider_name.clone()).collect();
            Ok(SlashCommandResult::StatusMessage(format!(
                "Stored {} credential(s) in keychain for \"{}\":\n  {}",
                names.len(),
                scope_label,
                names.join(", ")
            )))
        }
        Err(e) => Ok(SlashCommandResult::Error(format!("Import failed: {e}"))),
    }
}

async fn handle_creds_list() -> Result<SlashCommandResult> {
    let cm = match open_credential_manager().await {
        Ok(manager) => manager,
        Err(error) => return Ok(error),
    };
    let scopes = cm.list_scopes().await;

    if scopes.is_empty() {
        return Ok(SlashCommandResult::StatusMessage(
            "No credential scopes found.\nUse '/creds import' or '/creds set <provider> <key>' to add credentials.".to_string()
        ));
    }

    let mut lines = vec!["Credential Scopes".to_string(), String::new()];
    for scope in &scopes {
        let label = scope.label.as_deref().unwrap_or(scope.path.as_str());
        lines.push(format!("  {label}"));
        if !scope.providers.is_empty() {
            lines.push(format!("    Providers: {}", scope.providers.join(", ")));
        }
        lines.push(String::new());
    }

    Ok(SlashCommandResult::ShowPopup {
        title: "Credential Scopes".to_string(),
        content: lines.join("\n"),
    })
}

async fn gateway_start_note(prefix: &str) -> String {
    match crate::gateway_runtime::start_gateway_runtime(None, None).await {
        Ok(runtime) => format!(
            "{prefix}{} (pid {}).",
            runtime.entry.port, runtime.entry.pid
        ),
        Err(err) => format!(" Gateway runtime did not auto-start: {err}"),
    }
}

/// The `/creds setup` result: `done` or the cancel line, then the switch to a
/// default the user picked, when a session is running and is not already on
/// it.
async fn setup_result(
    report: SetupReport,
    done: String,
    session: Option<CredsSession<'_>>,
) -> SlashCommandResult {
    let mut lines = vec![match report.outcome {
        SetupOutcome::Completed => done,
        SetupOutcome::Cancelled => SETUP_CANCELLED.to_string(),
    }];
    if let (Some((provider, model)), Some(session)) = (report.default, session)
        && let Ok(provider_type) = provider.parse::<caliche::llm::ProviderType>()
        && (session.service.provider(), session.service.model()) != (provider_type, model.as_str())
    {
        match super::provider::switch_session_provider(
            session.state,
            session.service,
            provider_type,
            Some(model.clone()),
        )
        .await
        {
            Ok(()) => lines.push(format!("Now using {provider}/{model}.")),
            Err(line) => lines.push(line),
        }
    }
    SlashCommandResult::StatusMessage(lines.join("\n"))
}

async fn handle_creds_setup(
    args: &[String],
    session: Option<CredsSession<'_>>,
) -> Result<SlashCommandResult> {
    let setup_kinds = if let Some(scope) = args.get(1) {
        match crate::credentials::CredentialSetupKind::parse(scope) {
            Some(kind) => vec![kind],
            None => {
                return Ok(SlashCommandResult::Error(format!(
                    "Unknown setup type '{}'. Available: providers, gateway",
                    scope
                )));
            }
        }
    } else {
        crate::credentials::CredentialSetupKind::all().to_vec()
    };

    let report = match crate::credentials::interactive_setup_for(&setup_kinds).await {
        Ok(report) => report,
        Err(e) => return Ok(SlashCommandResult::Error(format!("Setup failed: {e}"))),
    };
    let mut done =
        "Credential setup complete. Run /provider <name> to use a newly configured provider."
            .to_string();
    if report.outcome == SetupOutcome::Completed
        && setup_kinds
            .iter()
            .any(|k| *k == crate::credentials::CredentialSetupKind::Gateway)
    {
        done.push_str(&gateway_start_note(" Gateway runtime running on port ").await);
    }
    Ok(setup_result(report, done, session).await)
}

async fn handle_creds_typed_setup(
    setup_type: &str,
    session: Option<CredsSession<'_>>,
) -> Result<SlashCommandResult> {
    let kind = match crate::credentials::CredentialSetupKind::parse(setup_type) {
        Some(kind) => kind,
        None => {
            return Ok(SlashCommandResult::Error(format!(
                "Unknown setup type '{}'. Available: providers, gateway",
                setup_type
            )));
        }
    };
    let report = match crate::credentials::interactive_setup_for(&[kind]).await {
        Ok(report) => report,
        Err(e) => return Ok(SlashCommandResult::Error(format!("Setup failed: {e}"))),
    };
    let mut done = format!("{} credential setup complete.", kind.display_name());
    if report.outcome == SetupOutcome::Completed
        && kind == crate::credentials::CredentialSetupKind::Gateway
    {
        done.push_str(&gateway_start_note(" Runtime on port ").await);
    }
    Ok(setup_result(report, done, session).await)
}

fn expand_link_source_dir(source_dir: &str) -> std::path::PathBuf {
    let expanded: std::path::PathBuf = if let Some(suffix) = source_dir.strip_prefix("~/") {
        match dirs::home_dir() {
            Some(home) => home.join(suffix),
            None => std::path::PathBuf::from(source_dir),
        }
    } else {
        std::path::PathBuf::from(source_dir)
    };
    if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir().unwrap_or_default().join(expanded)
    }
}

fn linked_reload_hint(
    provider: &str,
    linked_creds: std::collections::HashMap<String, String>,
) -> String {
    let available: Vec<String> = if provider == "*" {
        linked_creds.into_keys().collect()
    } else {
        linked_creds.into_keys().filter(|p| p == provider).collect()
    };
    if available.is_empty() {
        String::new()
    } else {
        format!(
            "\nRun '/provider {}' to start using the linked credentials.",
            available.first().map(String::as_str).unwrap_or("openai")
        )
    }
}

async fn handle_creds_link(args: &[String]) -> Result<SlashCommandResult> {
    let Some(source_dir) = args.get(1) else {
        return Ok(SlashCommandResult::Error(
            "Usage: /creds link <source-directory>\n\
             This makes credentials stored in <source-directory> available here.\n\
             Example: /creds link ~/my-api-project"
                .to_string(),
        ));
    };

    let source_str = expand_link_source_dir(source_dir)
        .to_string_lossy()
        .into_owned();
    let cm = match open_credential_manager().await {
        Ok(manager) => manager,
        Err(error) => return Ok(error),
    };

    let provider = args.get(2).map(String::as_str).unwrap_or("*");
    match cm.link_scope(&source_str, provider).await {
        Ok(()) => {
            cm.publish_to_credential_store().await;
            let prov_display = if provider == "*" {
                "all providers".to_string()
            } else {
                format!("'{provider}'")
            };
            use crate::storage::compute_scope_hash;
            let linked_creds = cm.get_scope_credentials(&compute_scope_hash(&source_str));
            let reload_hint = linked_reload_hint(provider, linked_creds);
            Ok(SlashCommandResult::StatusMessage(format!(
                "Linked {prov_display} from \"{source_str}\" to the current directory.{reload_hint}"
            )))
        }
        Err(e) => Ok(SlashCommandResult::Error(format!("Link failed: {e}"))),
    }
}

impl SlashCommandExecutor {
    /// Handle `/creds [subcommand] [args...]`
    ///
    /// With a running `session`, a setup that makes a provider the default
    /// also switches the session to it.
    pub(crate) async fn handle_creds(
        args: &[String],
        session: Option<CredsSession<'_>>,
    ) -> Result<SlashCommandResult> {
        let subcommand = args.first().map(String::as_str).unwrap_or("status");

        match subcommand {
            "gateway" => Self::handle_creds_gateway(args).await,
            "status" | "" => handle_creds_status().await,
            "set" => handle_creds_set(args).await,
            "import" => handle_creds_import().await,
            "list" => handle_creds_list().await,
            "setup" => handle_creds_setup(args, session).await,
            setup_type if args.get(1).is_some_and(|a| a == "setup") => {
                handle_creds_typed_setup(setup_type, session).await
            }
            "link" => handle_creds_link(args).await,
            other => Ok(SlashCommandResult::Error(format!(
                "Unknown /creds subcommand: '{other}'\n\nAvailable: status, set, import, list, link, setup, gateway, <type> setup (providers|gateway)"
            ))),
        }
    }
}
