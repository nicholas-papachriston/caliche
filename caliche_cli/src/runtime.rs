//! Runtime startup orchestration for the CLI binary.
//!
//! This module owns the policy-heavy pre-service setup path so `main.rs` can
//! stay focused on selecting the final execution lane.

use crate::bootstrap;
use crate::commands::Args;
use crate::{handlers, startup_prompts};
use anyhow::Result;
use caliche::llm::ProviderType;
use std::collections::HashSet;
use std::io::IsTerminal;
use std::sync::Arc;

/// Returns true if any advanced-feature argument is set.
pub fn has_advanced_feature_args(args: &Args) -> bool {
    args.reasoning_pattern.is_some()
        || args.chain_type.is_some()
        || args.document.is_some()
        || args.index_to.is_some()
        || args.query_collection.is_some()
        || args.hashline
        || args.multi_agent_pattern.is_some()
        || args.workflow.is_some()
        || args.template.is_some()
}

/// Startup state needed by the execution lanes after service creation.
pub struct PreparedRuntime {
    pub cancellation_token: crate::cancellation::CancellationToken,
    pub startup_ctx: bootstrap::StartupContext,
    pub session_id_for_repl: Option<String>,
    pub use_interactive: bool,
    pub preloaded_config: Option<crate::config::Config>,
    pub provider: Option<ProviderType>,
    pub model: Option<String>,
    pub tool_names: Option<Vec<String>>,
    pub tools_source: Option<String>,
    pub agent_profile: Option<caliche::agent::profile::AgentProfile>,
    pub workspace_options: bootstrap::WorkspaceEmbedOptions,
    /// Embedding discovery running in the background for the interactive REPL.
    pub deferred_workspace: Option<crate::deferred::DeferredWorkspaceRag>,
    /// Non-blocking startup notices (e.g. unused .env keys).
    pub startup_notices: Vec<String>,
    /// Temperature from `settings.json`, before CLI `--temperature`.
    pub saved_temperature: Option<f32>,
    /// Max tokens from `settings.json`, before CLI `--max-tokens`.
    pub saved_max_tokens: Option<u32>,
}

/// Outcome of [`prepare`]: either a finished utility/subcommand or a ready runtime.
pub enum PrepareOutcome {
    /// Subcommand or `--list-models` already completed; the process should exit.
    Finished,
    /// Continue into one-shot, server, or REPL execution.
    Ready(PreparedRuntime),
}

/// Prepare the shared runtime context used by interactive, one-shot, server,
/// MCP and advanced execution lanes.
pub async fn prepare(args: &Arc<Args>) -> Result<PrepareOutcome> {
    let shell_env_keys = crate::storage::shell_env_keys();
    let env_file_mode = args.env_file.is_some();

    crate::commands::validate_explicit_providers(args)?;
    if crate::commands::invocation_uses_provider(args.command.as_ref()) {
        crate::commands::validate_env_providers()?;
    }
    if args.quiet {
        let log_config = bootstrap::LogConfig::from_args(args);
        let _ = bootstrap::init_logging(&log_config)?;
    }

    if let Some(ref provider_str) = args.list_models {
        handlers::run_list_models(provider_str.clone()).await?;
        return Ok(PrepareOutcome::Finished);
    }

    let cancellation_token = crate::cancellation::setup_signal_handler();
    match handlers::process_command(args.as_ref(), Some(&cancellation_token)).await? {
        handlers::ProcessCommandOutcome::Completed => {
            return Ok(PrepareOutcome::Finished);
        }
        handlers::ProcessCommandOutcome::Continue { session_id } => {
            let session_id_for_repl = session_id.or_else(|| args.session.clone());
            prepare_ready(
                args,
                shell_env_keys,
                env_file_mode,
                cancellation_token,
                session_id_for_repl,
            )
            .await
        }
    }
}

async fn prepare_ready(
    args: &Arc<Args>,
    shell_env_keys: HashSet<String>,
    env_file_mode: bool,
    cancellation_token: crate::cancellation::CancellationToken,
    session_id_for_repl: Option<String>,
) -> Result<PrepareOutcome> {
    let tty = std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    let defer_workspace_embed = bootstrap::effective_workspace_embed(args) && args.wants_repl(tty);

    let (cred_manager, embedding_discovery) =
        load_credentials_and_embeddings(args, env_file_mode, tty).await;

    if !env_file_mode {
        if let Some(ref cm) = cred_manager {
            bootstrap::load_startup_credentials(cm, &shell_env_keys).await;
        }
    }

    let use_interactive = args.wants_repl(tty);
    let startup_notices = collect_startup_notices(use_interactive, cred_manager.as_ref());
    let startup_ctx = open_startup_context(args, cred_manager, use_interactive).await?;
    let allow_credential_prompt = use_interactive && !args.quiet;
    ensure_initial_credentials(&startup_ctx, allow_credential_prompt).await?;

    let mut provider = parse_provider(args)?;
    warn_mismatched_openai_openrouter_key(provider.as_ref());
    let mut model = args.model.clone();
    let restored = restore_last_provider_model(
        args,
        use_interactive,
        &startup_ctx,
        &mut provider,
        &mut model,
    )
    .await?;
    let used_last_model = restored.used_last_model;
    // Start the probe only when no provider is chosen yet. A saved or flagged
    // provider used to pay for localhost HTTP calls and then abort them.
    let local_providers_handle =
        spawn_local_provider_discovery(args, tty, used_last_model, provider.is_some());

    let preloaded_config = load_preloaded_config_for_runtime(args, use_interactive).await?;
    apply_local_first_token_timeout(preloaded_config.as_ref());
    apply_config_provider_model_fallback(
        &preloaded_config,
        used_last_model,
        &mut provider,
        &mut model,
    )?;

    let local_providers = resolve_local_providers(
        local_providers_handle,
        used_last_model,
        provider.is_some(),
        args,
    )
    .await;
    auto_select_local_provider(
        args,
        used_last_model,
        &startup_ctx,
        &local_providers,
        &mut provider,
    )
    .await;

    let mut tool_names = parse_tool_names(args);
    let agent_profile = load_agent_profile(args).await;
    apply_profile_overrides(&agent_profile, &mut model, &mut tool_names);

    let tools_source = agent_profile
        .as_ref()
        .and_then(|_| args.agent.as_deref().map(|n| format!("from profile {}", n)))
        .or_else(|| args.tools.as_deref().map(|_| "from --tools".to_string()))
        .or_else(|| {
            args.is_lean_profile()
                .then(|| "from --profile lean".to_string())
        });

    apply_provider_credentials_if_needed(use_interactive, &startup_ctx, provider).await;

    let mut workspace_options = bootstrap::WorkspaceEmbedOptions::from_args(args);
    let persist_embed = persisted_embedding_set(&startup_ctx).await;
    workspace_options.skip_embedding_discovery = persist_embed;
    if persist_embed {
        workspace_options.pre_discovered_embedding = None;
    } else {
        workspace_options.pre_discovered_embedding = embedding_discovery;
    }
    workspace_options.defer_until_interactive = defer_workspace_embed;
    let deferred_workspace = spawn_deferred_workspace(&workspace_options, defer_workspace_embed);

    Ok(PrepareOutcome::Ready(PreparedRuntime {
        cancellation_token,
        startup_ctx,
        session_id_for_repl,
        use_interactive,
        preloaded_config,
        provider,
        model,
        tool_names,
        tools_source,
        agent_profile,
        workspace_options,
        deferred_workspace,
        startup_notices,
        saved_temperature: restored.saved_temperature,
        saved_max_tokens: restored.saved_max_tokens,
    }))
}

fn should_probe_local_providers(
    args: &Args,
    tty: bool,
    used_last_model: bool,
    provider_set: bool,
) -> bool {
    // Plain `--server` is not a REPL, so it does not probe unless `--prefer-local` is set.
    !used_last_model
        && !provider_set
        && !args.no_auto_detect
        && args.provider.is_none()
        && (args.prefer_local || args.wants_repl(tty))
}

fn spawn_local_provider_discovery(
    args: &Args,
    tty: bool,
    used_last_model: bool,
    provider_set: bool,
) -> Option<tokio::task::JoinHandle<Vec<crate::discovery::LocalProviderStatus>>> {
    should_probe_local_providers(args, tty, used_last_model, provider_set)
        .then(|| tokio::spawn(async { crate::discovery::discover_local_providers().await }))
}

async fn load_credentials_and_embeddings(
    args: &Args,
    env_file_mode: bool,
    tty: bool,
) -> (
    Option<crate::storage::SecureCredentialManager>,
    Option<caliche::discovery::embedding::EmbeddingProviderDiscovery>,
) {
    tokio::join!(
        async {
            if env_file_mode {
                None
            } else {
                crate::storage::SecureCredentialManager::new().await.ok()
            }
        },
        async {
            if bootstrap::effective_workspace_embed(args) && !args.wants_repl(tty) {
                Box::pin(caliche::discovery::embedding::discover_best_embedding_provider()).await
            } else {
                None
            }
        },
    )
}

fn collect_startup_notices(
    use_interactive: bool,
    cred_manager: Option<&crate::storage::SecureCredentialManager>,
) -> Vec<String> {
    let mut startup_notices = Vec::new();
    if use_interactive
        && let Some(cm) = cred_manager
        && let Some(notice) = startup_prompts::env_credential_ingest_notice(cm)
    {
        startup_notices.push(notice);
    }
    startup_notices
}

async fn resolve_local_providers(
    local_providers_handle: Option<
        tokio::task::JoinHandle<Vec<crate::discovery::LocalProviderStatus>>,
    >,
    used_last_model: bool,
    provider_set: bool,
    args: &Args,
) -> Vec<crate::discovery::LocalProviderStatus> {
    if used_last_model || provider_set || args.no_auto_detect {
        if let Some(handle) = local_providers_handle {
            handle.abort();
        }
        return Vec::new();
    }
    if let Some(handle) = local_providers_handle {
        handle.await.unwrap_or_else(|_| Vec::new())
    } else {
        Vec::new()
    }
}

fn apply_profile_overrides(
    agent_profile: &Option<caliche::agent::profile::AgentProfile>,
    model: &mut Option<String>,
    tool_names: &mut Option<Vec<String>>,
) {
    let Some(profile) = agent_profile else {
        return;
    };
    if let Some(m) = profile.model.clone() {
        *model = Some(m);
    }
    if let Some(tools) = profile.enabled_tools() {
        *tool_names = Some(tools);
    }
}

async fn persisted_embedding_set(ctx: &bootstrap::StartupContext) -> bool {
    let Some(db) = ctx.database.as_ref() else {
        return false;
    };
    match db.load_setting("embedding_provider").await {
        Ok(Some(value)) => !value.trim().is_empty(),
        _ => false,
    }
}

fn spawn_deferred_workspace(
    workspace_options: &bootstrap::WorkspaceEmbedOptions,
    defer_workspace_embed: bool,
) -> Option<crate::deferred::DeferredWorkspaceRag> {
    if !defer_workspace_embed {
        return None;
    }
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let config = bootstrap::workspace_indexer_config(workspace_root, workspace_options);
    if workspace_options.skip_embedding_discovery {
        Some(crate::deferred::DeferredWorkspaceRag::spawn_without_discovery(config))
    } else {
        Some(crate::deferred::DeferredWorkspaceRag::spawn(config))
    }
}

/// Create an agent service and recover missing credentials in interactive mode.
pub async fn create_service_with_recovery(
    args: &Args,
    provider: Option<ProviderType>,
    model: Option<String>,
    tool_names: Option<Vec<String>>,
    workspace_options: bootstrap::WorkspaceEmbedOptions,
    startup_ctx: &bootstrap::StartupContext,
    use_interactive: bool,
    cli_config: Option<&crate::config::Config>,
) -> Result<caliche::AgentService> {
    let skip_discovery = workspace_options.skip_embedding_discovery;
    let defer_embed = workspace_options.defer_until_interactive;
    let options_for_attach = workspace_options.clone();
    let model_for_retry = model.clone();
    let tool_names_for_retry = tool_names.clone();
    let service_result = Box::pin(bootstrap::create_agent_service(
        provider,
        model,
        tool_names,
        args,
        workspace_options,
        startup_ctx.cred_manager.as_ref(),
        cli_config,
        args.server,
    ))
    .await;

    let mut service = if !use_interactive {
        service_result?
    } else {
        match service_result {
            Ok(s) => s,
            Err(e) => {
                let error_msg = e.to_string().to_lowercase();
                if !(error_msg.contains("no credentials") || error_msg.contains("credential")) {
                    return Err(e);
                }

                let recovered = if let Some(ref cm) = startup_ctx.cred_manager {
                    startup_prompts::try_recover_credentials(cm).await
                } else {
                    false
                };

                if !recovered {
                    println!(
                        "{}",
                        crate::credentials::missing_key_line(
                            &caliche::agent::service::resolve_provider(provider).to_string()
                        )
                    );
                }
                // A key the recovery promoted is used. Without one the session
                // still starts; its completions report the missing key until
                // `/creds setup` or `/provider` fixes it.
                Box::pin(bootstrap::create_agent_service(
                    provider,
                    model_for_retry,
                    tool_names_for_retry,
                    args,
                    options_for_attach.clone(),
                    startup_ctx.cred_manager.as_ref(),
                    cli_config,
                    true,
                ))
                .await?
            }
        }
    };
    apply_persisted_runtime_settings(&mut service, startup_ctx).await;
    attach_saved_workspace_embedding(
        &mut service,
        args,
        skip_discovery,
        defer_embed,
        &options_for_attach,
    )
    .await;
    persist_cli_subagent_overrides(args, startup_ctx).await;
    apply_live_subagent_selection(&service, args, startup_ctx, cli_config).await;
    Ok(service)
}

/// Record the shell's variable names, then load `--env-file` with
/// [`caliche_server::env_file::load_before_runtime`].
///
/// The file can hold any setting (provider, model, config dir, base URLs),
/// and those are read from the process environment throughout, so this is
/// the one remaining environment write. Call it from synchronous `main`
/// before the async runtime is built, while no other thread exists.
/// Credentials loaded later go to [`caliche_contracts::credential_env`].
///
/// # Errors
///
/// Returns an error when the file cannot be read or parsed.
pub fn load_env_file_before_runtime(args: &Args) -> Result<()> {
    crate::storage::record_shell_env_keys();
    caliche_server::env_file::load_before_runtime(args.env_file.as_deref())
}

async fn persist_cli_subagent_overrides(args: &Args, startup_ctx: &bootstrap::StartupContext) {
    let Some(ref db) = startup_ctx.database else {
        return;
    };
    if let Some(ref provider) = args.subagent_provider
        && let Err(error) = db.save_setting("subagent_provider", provider).await
    {
        tracing::warn!(error = %error, "failed to persist --subagent-provider");
    }
    if let Some(ref model) = args.subagent_model
        && let Err(error) = db.save_setting("subagent_model", model).await
    {
        tracing::warn!(error = %error, "failed to persist --subagent-model");
    }
}

async fn apply_live_subagent_selection(
    service: &caliche::AgentService,
    args: &Args,
    startup_ctx: &bootstrap::StartupContext,
    cli_config: Option<&crate::config::Config>,
) {
    let mut provider = args.subagent_provider.clone();
    let mut model = args.subagent_model.clone();
    if let Some(db) = startup_ctx.database.as_ref() {
        if provider
            .as_ref()
            .is_none_or(|value| value.trim().is_empty())
        {
            provider = nonempty_setting(db.load_setting("subagent_provider").await.ok().flatten());
        }
        if model.as_ref().is_none_or(|value| value.trim().is_empty()) {
            model = nonempty_setting(db.load_setting("subagent_model").await.ok().flatten());
        }
    }
    if provider.is_none() {
        provider =
            cli_config.and_then(|cfg| cfg.subagent.as_ref().and_then(|s| s.provider.clone()));
    }
    if model.is_none() {
        model = cli_config.and_then(|cfg| cfg.subagent.as_ref().and_then(|s| s.model.clone()));
    }
    service.set_subagent_selection(caliche::SubagentModelSelection::from_parts(
        provider.as_deref(),
        model.as_deref(),
    ));
}

async fn attach_saved_workspace_embedding(
    service: &mut caliche::AgentService,
    args: &Args,
    skip_discovery: bool,
    defer_embed: bool,
    workspace_options: &bootstrap::WorkspaceEmbedOptions,
) {
    if !skip_discovery || defer_embed || !bootstrap::effective_workspace_embed(args) {
        return;
    }
    let Some(provider) = service.embedding_provider() else {
        return;
    };
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    if let Err(error) =
        bootstrap::attach_workspace_embedding(service, workspace_root, workspace_options, provider)
            .await
    {
        tracing::warn!(error = %error, "failed to attach workspace RAG for saved embedding provider");
    }
}

fn nonempty_setting(value: Option<String>) -> Option<String> {
    value.filter(|s| !s.trim().is_empty())
}

fn should_open_database(args: &Args, use_interactive: bool) -> bool {
    use_interactive || args.has_message_input() || has_advanced_feature_args(args) || args.server
}

fn should_restore_persisted_model(args: &Args, use_interactive: bool) -> bool {
    // `--provider` must not skip this load. The saved model still applies when
    // it belongs to that provider. `--model` is applied by the caller and is
    // not overwritten. Temperature is read from the same record.
    use_interactive || args.has_message_input() || args.server
}

struct PersistedRestore {
    used_last_model: bool,
    saved_temperature: Option<f32>,
    saved_max_tokens: Option<u32>,
}

/// Apply saved temperature and max tokens.
///
/// `--temperature` / `--max-tokens` win. A value already set from project
/// config stays when the flag is absent. Otherwise `settings.json` fills it.
pub fn apply_saved_sampling(
    service: &mut caliche::AgentService,
    cli_temperature: Option<f32>,
    cli_max_tokens: Option<u32>,
    saved_temperature: Option<f32>,
    saved_max_tokens: Option<u32>,
) {
    if let Some(temperature) = caliche_server::durable_settings::sampling_value(
        cli_temperature,
        saved_temperature,
        service.agent().temperature().is_some(),
    ) {
        service.agent_mut().set_temperature(Some(temperature));
    }
    if let Some(max_tokens) = caliche_server::durable_settings::sampling_value(
        cli_max_tokens,
        saved_max_tokens,
        service.agent().max_tokens().is_some(),
    ) {
        service.agent_mut().set_max_tokens(Some(max_tokens));
    }
}

async fn apply_persisted_runtime_settings(
    service: &mut caliche::AgentService,
    startup_ctx: &bootstrap::StartupContext,
) {
    let Some(ref db) = startup_ctx.database else {
        return;
    };

    if let Ok(Some(value)) = db.load_setting("show_sources").await {
        if value.eq_ignore_ascii_case("false") {
            service.set_show_sources(false);
        } else if value.eq_ignore_ascii_case("true") {
            service.set_show_sources(true);
        }
    }

    let Ok(Some(provider_name)) = db.load_setting("embedding_provider").await else {
        return;
    };
    let provider_name = provider_name.trim();
    if provider_name.is_empty() {
        return;
    }
    let Ok(provider_type) = provider_name.parse::<ProviderType>() else {
        tracing::warn!("Ignoring persisted invalid embedding provider: {provider_name}");
        return;
    };
    let model = match db.load_setting("embedding_model").await {
        Ok(Some(model)) if !model.trim().is_empty() => model,
        _ => match caliche::agent::service::default_embedding_model(provider_type) {
            Some(model) => model,
            None => {
                tracing::warn!(
                    "Ignoring persisted unsupported embedding provider: {provider_name}"
                );
                return;
            }
        },
    };

    let mut settings = service.snapshot_settings();
    settings.embedding_provider = Some(provider_type.to_string());
    settings.embedding_model = Some(model);
    if let Err(error) = service.apply_settings(settings).await {
        tracing::warn!(
            error = %error,
            "failed to restore persisted embedding provider"
        );
    }
}

async fn open_startup_context(
    args: &Args,
    cred_manager: Option<crate::storage::SecureCredentialManager>,
    use_interactive: bool,
) -> Result<bootstrap::StartupContext> {
    let need_database = should_open_database(args, use_interactive);
    let log_config = bootstrap::LogConfig::from_args(args);
    caliche::observability::perf::configure_from_cli(args.debug, args.no_perf);

    let (database, log_path) = if need_database {
        let log_config_clone = log_config.clone();
        let (db_result, log_result) = tokio::join!(
            async { crate::storage::Database::open().await },
            tokio::task::spawn_blocking(move || bootstrap::init_logging(&log_config_clone)),
        );
        let log_path = log_result.map_err(|e| anyhow::anyhow!("Log init: {}", e))??;
        (db_result.ok(), log_path)
    } else {
        let log_path = bootstrap::init_logging(&log_config)?;
        (None, log_path)
    };

    Ok(bootstrap::StartupContext {
        cred_manager,
        database,
        log_path,
    })
}

async fn ensure_initial_credentials(
    startup_ctx: &bootstrap::StartupContext,
    allow_prompt: bool,
) -> Result<()> {
    if !allow_prompt {
        return Ok(());
    }
    if let Some(ref cm) = startup_ctx.cred_manager {
        let configured = cm.list_configured().await;
        let has_any_configured = configured.iter().any(|p| p.is_configured);

        if !has_any_configured {
            println!("\n\x1b[33mNo API credentials configured.\x1b[0m");
            println!(
                "\x1b[2mTip: local providers like Ollama and LM Studio do not need API keys.\x1b[0m"
            );
            if let Err(err) = crate::config::config_file_wizard() {
                crate::credentials::report_startup_setup(&Err(err));
                return Ok(());
            }
            println!("Let's set up your first provider.\n");
            let setup = crate::credentials::interactive_setup().await;
            crate::credentials::report_startup_setup(&setup);
            println!(
                "\x1b[2mRun `caliche doctor` to verify setup, or start chatting with `caliche`.\x1b[0m\n"
            );
        }
    }
    Ok(())
}

fn parse_provider(args: &Args) -> Result<Option<ProviderType>> {
    args.provider
        .as_deref()
        .map(|p| p.parse())
        .transpose()
        .map_err(|e| anyhow::anyhow!("Invalid provider: {}", e))
}

async fn restore_last_provider_model(
    args: &Args,
    use_interactive: bool,
    startup_ctx: &bootstrap::StartupContext,
    provider: &mut Option<ProviderType>,
    model: &mut Option<String>,
) -> Result<PersistedRestore> {
    let boot = if should_restore_persisted_model(args, use_interactive) {
        let (sqlite_provider, sqlite_model) = persisted_sqlite_provider_model(startup_ctx).await;
        crate::boot_provider::load_persisted_boot(sqlite_provider, sqlite_model).await
    } else {
        crate::boot_provider::PersistedBoot::default()
    };
    let agent_provider = std::env::var("AGENT_PROVIDER").ok();
    let default_provider_env = std::env::var("DEFAULT_PROVIDER").ok();
    let choice = crate::boot_provider::select_server_boot_provider(
        crate::boot_provider::ServerBootProviderInput {
            explicit_provider: args.provider.as_deref(),
            explicit_model: args.model.is_some(),
            agent_provider: agent_provider.as_deref(),
            default_provider_env: default_provider_env.as_deref(),
            last_provider: boot.provider.as_deref(),
            last_model: boot.model.as_deref(),
        },
    );
    let used_last_model = crate::boot_provider::apply_server_boot_provider(
        &choice,
        boot.provider.as_deref(),
        boot.model.as_deref(),
        provider,
        model,
    )?;
    Ok(PersistedRestore {
        used_last_model,
        saved_temperature: boot.temperature,
        saved_max_tokens: boot.max_tokens,
    })
}

async fn persisted_sqlite_provider_model(
    startup_ctx: &bootstrap::StartupContext,
) -> (Option<String>, Option<String>) {
    let Some(ref db) = startup_ctx.database else {
        return (None, None);
    };
    let provider = db.load_setting("last_provider").await.ok().flatten();
    let model = db.load_setting("last_model").await.ok().flatten();
    (provider, model)
}

async fn load_preloaded_config_for_runtime(
    args: &Args,
    use_interactive: bool,
) -> Result<Option<crate::config::Config>> {
    let mut config = if use_interactive || args.has_message_input() {
        Some(crate::config::load_standard_config_strict().await?)
    } else {
        None
    };
    if let Some(ref level) = args.output_verbosity {
        let mut cfg = config.unwrap_or_default();
        cfg.output_verbosity = Some(level.clone());
        config = Some(cfg);
    }
    if let Some(ref mode) = args.banner {
        let mut cfg = config.unwrap_or_default();
        cfg.banner = Some(mode.clone());
        config = Some(cfg);
    }
    Ok(config)
}

/// Set the first-token limit of local providers from
/// `local_first_token_timeout_secs`, clamped to its bounds.
pub(crate) fn apply_local_first_token_timeout(config: Option<&crate::config::Config>) {
    let Some(secs) = config.and_then(|cfg| cfg.local_first_token_timeout_secs) else {
        return;
    };
    let applied = caliche::llm::provider_utils::first_token::set_local_first_token_timeout(
        std::time::Duration::from_secs(secs),
    );
    if applied.as_secs() != secs {
        tracing::warn!(
            requested = secs,
            applied = applied.as_secs(),
            "local_first_token_timeout_secs is out of range; clamped"
        );
    }
}

/// Logged when the config model is not used because another provider was
/// selected on the command line.
const CONFIG_MODEL_NOT_USED: &str =
    "config model is for another provider; not used with the selected one";

/// Start of the error for a config provider name that does not parse.
const INVALID_CONFIG_PROVIDER: &str = "Invalid provider in config: ";

/// Fill `provider` and `model` from the loaded config where the command
/// line left them unset. The config provider is parsed even when
/// `--provider` is given, because the config model is kept only for that
/// provider, so an unknown provider name in the config fails start-up.
pub(crate) fn apply_config_provider_model_fallback(
    preloaded_config: &Option<crate::config::Config>,
    used_last_model: bool,
    provider: &mut Option<ProviderType>,
    model: &mut Option<String>,
) -> Result<()> {
    if used_last_model || (provider.is_some() && model.is_some()) {
        return Ok(());
    }
    let Some(cfg) = preloaded_config else {
        return Ok(());
    };
    let config_provider = cfg
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| {
            ProviderType::parse_in_build(name)
                .map_err(|err| anyhow::anyhow!("{INVALID_CONFIG_PROVIDER}{err}"))
        })
        .transpose()?;
    if provider.is_none() {
        *provider = config_provider;
    }
    // The config model is checked against the config provider's model list
    // (`Config::keep_untrusted_keys`), so it is not used for another provider.
    let other_provider = config_provider.is_some() && *provider != config_provider;
    if model.is_none() && !other_provider {
        model.clone_from(&cfg.model);
    } else if model.is_none()
        && let Some(config_model) = &cfg.model
    {
        tracing::info!(model = %config_model, "{CONFIG_MODEL_NOT_USED}");
    }
    Ok(())
}

async fn auto_select_local_provider(
    args: &Args,
    used_last_model: bool,
    startup_ctx: &bootstrap::StartupContext,
    local_providers: &[crate::discovery::LocalProviderStatus],
    provider: &mut Option<ProviderType>,
) {
    if used_last_model || provider.is_some() || args.no_auto_detect {
        return;
    }
    if !(args.prefer_local || local_providers.iter().any(|p| p.is_available)) {
        return;
    }

    let has_cloud_creds = if let Some(ref cm) = startup_ctx.cred_manager {
        let configured = cm.list_configured().await;
        configured.iter().any(|p| {
            p.is_configured && !matches!(p.name.as_str(), "ollama" | "lmstudio" | "local-inference")
        })
    } else {
        false
    };

    if !(args.prefer_local || !has_cloud_creds) {
        return;
    }

    if let Some(local_prov) = local_providers.iter().find(|p| p.is_available) {
        *provider = match local_prov.name {
            "ollama" => Some(ProviderType::Ollama),
            "lmstudio" => Some(ProviderType::LMStudio),
            _ => None,
        };
        if provider.is_some() && !args.acp_stdio {
            println!(
                "\x1b[32mDetected {} at {} with {} model(s)\x1b[0m",
                local_prov.name,
                local_prov.base_url,
                local_prov.models.len()
            );
        }
    }
}

fn preset_names(tools: &[&str]) -> Vec<String> {
    tools.iter().map(|name| (*name).to_string()).collect()
}

pub(crate) fn parse_tool_names(args: &Args) -> Option<Vec<String>> {
    let mut names = args.tools.as_deref().map(expand_tool_preset_or_list);
    if names.is_none() && args.is_lean_profile() {
        names = Some(preset_names(caliche::agent::tools::names::LEAN_DEMO_TOOLS));
    }
    if let Some(exclude) = args.exclude_tools.as_deref() {
        let exclude_set: std::collections::HashSet<String> = exclude
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect();
        if !exclude_set.is_empty() {
            let base = names.unwrap_or_else(|| {
                preset_names(caliche::agent::tools::names::CODING_DEFAULT_TOOLS)
            });
            names = Some(
                base.into_iter()
                    .filter(|n| !exclude_set.contains(&n.to_ascii_lowercase()))
                    .collect(),
            );
        }
    }
    names
}

fn expand_tool_preset_or_list(tools_str: &str) -> Vec<String> {
    let trimmed = tools_str.trim();
    match trimmed.to_ascii_lowercase().as_str() {
        "coding" | "default" => preset_names(caliche::agent::tools::names::CODING_DEFAULT_TOOLS),
        "readonly" | "read-only" | "ask" => vec!["fs".to_string()],
        "none" | "no-tools" => Vec::new(),
        "extended" => preset_names(caliche::agent::tools::names::EXTENDED_DEFAULT_TOOLS),
        _ => trimmed
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
    }
}

async fn load_agent_profile(args: &Args) -> Option<caliche::agent::profile::AgentProfile> {
    if let Some(ref name) = args.agent {
        handlers::load_profile_by_name(name).await.ok().flatten()
    } else {
        None
    }
}

fn warn_mismatched_openai_openrouter_key(provider: Option<&ProviderType>) {
    if !matches!(provider, Some(ProviderType::OpenAI)) {
        return;
    }
    let Ok(key) = caliche_contracts::credential_env::var("OPENAI_API_KEY") else {
        return;
    };
    if !caliche::llm::provider_utils::looks_like_openrouter_api_key(&key) {
        return;
    }
    eprintln!(
        "OPENAI_API_KEY looks like an OpenRouter key (sk-or-...). Native openai requests will return 401. Use --provider openrouter or set a platform.openai.com key."
    );
}

async fn apply_provider_credentials_if_needed(
    use_interactive: bool,
    startup_ctx: &bootstrap::StartupContext,
    provider: Option<ProviderType>,
) {
    if !use_interactive {
        return;
    }

    let resolved_provider = provider
        .or_else(|| {
            std::env::var("AGENT_PROVIDER")
                .ok()
                .and_then(|p| p.parse().ok())
        })
        .unwrap_or(ProviderType::OpenRouter);

    let needs_credentials = !matches!(
        resolved_provider,
        ProviderType::Ollama | ProviderType::LMStudio
    );
    if !needs_credentials {
        return;
    }

    let provider_name = resolved_provider.to_string();
    let has_env_creds = crate::credentials::provider_has_env_credentials(&provider_name);

    let mut provider_configured = false;
    if let Some(ref cm) = startup_ctx.cred_manager {
        let configured = cm.list_configured().await;
        provider_configured = configured
            .iter()
            .find(|p| p.name == provider_name)
            .map(|p| p.is_configured)
            .unwrap_or(false);
    }

    if !provider_configured && !has_env_creds {
        if let Some(ref cm) = startup_ctx.cred_manager {
            cm.apply_to_credential_store().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// A `tracing` writer that keeps everything written to it.
    #[derive(Clone, Default)]
    struct Captured(Arc<std::sync::Mutex<Vec<u8>>>);

    impl std::io::Write for Captured {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture lock")
                .extend_from_slice(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn openai_config(provider: &str) -> Option<crate::config::Config> {
        Some(crate::config::Config {
            provider: Some(provider.to_string()),
            model: Some("gpt-4o".to_string()),
            ..Default::default()
        })
    }

    #[test]
    #[cfg(feature = "local-inference-base")]
    fn a_config_model_for_another_provider_is_logged_and_not_used() {
        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::INFO)
            .finish();
        let (mut provider, mut model) = (Some(ProviderType::LocalInference), None);

        tracing::subscriber::with_default(subscriber, || {
            apply_config_provider_model_fallback(
                &openai_config("openai"),
                false,
                &mut provider,
                &mut model,
            )
        })
        .expect("resolve");

        let log =
            String::from_utf8(captured.0.lock().expect("capture lock").clone()).expect("utf-8 log");
        assert_eq!(model, None);
        assert!(
            log.contains(CONFIG_MODEL_NOT_USED) && log.contains("model=gpt-4o"),
            "{log}"
        );
    }

    #[test]
    fn an_unknown_config_provider_fails_start_up_even_with_a_cli_provider() {
        let (mut provider, mut model) = (Some(ProviderType::OpenAI), None);

        let err = apply_config_provider_model_fallback(
            &openai_config("no-such-provider"),
            false,
            &mut provider,
            &mut model,
        )
        .expect_err("unknown config provider");

        assert!(
            err.to_string().starts_with(INVALID_CONFIG_PROVIDER),
            "{err}"
        );
    }

    /// A config that names a provider this build left out stops start-up
    /// with one line that says how to rebuild.
    #[test]
    #[cfg(not(feature = "local-inference-base"))]
    fn a_config_provider_left_out_of_the_build_says_how_to_rebuild() {
        let (mut provider, mut model) = (None, None);

        let err = apply_config_provider_model_fallback(
            &openai_config("local-inference"),
            false,
            &mut provider,
            &mut model,
        )
        .expect_err("left-out config provider");

        assert_eq!(
            err.to_string(),
            "Invalid provider in config: local-inference is not in this build. \
             Rebuild with --features local-inference."
        );
    }

    /// A config `provider = "bedrock"` in the default build stops start-up
    /// with the rebuild line.
    #[test]
    #[cfg(not(feature = "bedrock"))]
    fn a_config_cloud_provider_left_out_of_the_build_says_how_to_rebuild() {
        let (mut provider, mut model) = (None, None);

        let err = apply_config_provider_model_fallback(
            &openai_config("bedrock"),
            false,
            &mut provider,
            &mut model,
        )
        .expect_err("left-out config provider");

        assert_eq!(
            err.to_string(),
            "Invalid provider in config: bedrock is not in this build. \
             Rebuild with --features bedrock."
        );
    }

    #[test]
    fn server_mode_opens_database_and_restores_last_model() {
        let args = Args::parse_from(["caliche", "--server"]);
        assert!(should_open_database(&args, false));
        assert!(should_restore_persisted_model(&args, false));
    }

    #[test]
    fn explicit_provider_flag_still_loads_saved_model() {
        let args = Args::parse_from(["caliche", "--server", "--provider", "openai"]);
        assert!(should_open_database(&args, false));
        assert!(should_restore_persisted_model(&args, false));
    }

    #[test]
    fn bare_noninteractive_run_skips_database() {
        let args = Args::parse_from(["caliche"]);
        assert!(!should_open_database(&args, false));
        assert!(!should_restore_persisted_model(&args, false));
    }

    #[test]
    fn interactive_repl_opens_database_and_restores_last_model() {
        let args = Args::parse_from(["caliche"]);
        assert!(should_open_database(&args, true));
        assert!(should_restore_persisted_model(&args, true));
    }

    #[test]
    fn repl_without_a_provider_probes_local_models() {
        let args = Args::parse_from(["caliche"]);
        assert!(should_probe_local_providers(&args, true, false, false));
    }

    #[test]
    fn saved_or_selected_provider_skips_local_probes() {
        let args = Args::parse_from(["caliche"]);
        assert!(!should_probe_local_providers(&args, true, true, false));
        assert!(!should_probe_local_providers(&args, true, false, true));
        let flagged = Args::parse_from(["caliche", "--provider", "anthropic"]);
        assert!(!should_probe_local_providers(&flagged, true, false, false));
        let disabled = Args::parse_from(["caliche", "--no-auto-detect"]);
        assert!(!should_probe_local_providers(&disabled, true, false, false));
    }

    #[test]
    fn lean_profile_uses_the_small_tool_roster() {
        let lean = Args::parse_from(["caliche", "--profile", "lean"]);
        let names = parse_tool_names(&lean).expect("lean roster");
        assert_eq!(names, ["bash".to_string(), "fs".to_string()]);
        let overridden = Args::parse_from(["caliche", "--profile", "lean", "--tools", "fs"]);
        let names = parse_tool_names(&overridden).expect("explicit tools");
        assert_eq!(names, ["fs".to_string()]);
    }

    #[test]
    fn server_probes_local_models_only_when_preferred() {
        let server = Args::parse_from(["caliche", "--server"]);
        assert!(!should_probe_local_providers(&server, true, false, false));
        let preferred = Args::parse_from(["caliche", "--server", "--prefer-local"]);
        assert!(should_probe_local_providers(&preferred, true, false, false));
    }

    #[test]
    fn coding_and_default_presets_include_web_fetch() {
        for preset in ["coding", "default", "extended"] {
            let names = expand_tool_preset_or_list(preset);
            assert!(
                names.iter().any(|name| name == "web_fetch"),
                "{preset} should include web_fetch: {names:?}"
            );
            assert!(
                !names.iter().any(|name| name == "web_search"),
                "{preset} should leave web_search opt-in: {names:?}"
            );
        }
    }

    #[test]
    fn readonly_and_none_presets_exclude_web_fetch() {
        let readonly = expand_tool_preset_or_list("readonly");
        assert_eq!(readonly, vec!["fs".to_string()]);
        assert!(!readonly.iter().any(|name| name == "web_fetch"));

        let none = expand_tool_preset_or_list("none");
        assert!(none.is_empty());
        let args = Args::parse_from(["caliche", "--tools", "none"]);
        let parsed = parse_tool_names(&args).expect("none is an explicit empty roster");
        assert!(parsed.is_empty(), "none must not expand into default tools");
    }

    #[test]
    fn lean_profile_without_tools_stays_fs_and_bash() {
        let args = Args::parse_from(["caliche", "--profile", "lean"]);
        let names = parse_tool_names(&args).expect("lean sets an explicit roster");
        assert!(names.iter().any(|name| name == "fs"));
        assert!(names.iter().any(|name| name == "bash"));
        assert!(!names.iter().any(|name| name == "web_fetch"));
    }

    #[test]
    fn unset_tools_use_registry_defaults() {
        let args = Args::parse_from(["caliche"]);
        assert!(parse_tool_names(&args).is_none());
    }

    #[test]
    #[serial_test::serial(env)]
    fn env_file_adds_new_values_and_keeps_shell_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.env");
        std::fs::write(
            &path,
            "CALICHE_ENV_FILE_TEST_SHELL=file-value\nCALICHE_ENV_FILE_TEST_NEW=file-new\n",
        )
        .expect("write env file");
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_ENV_FILE_TEST_SHELL", "shell-value");
        env.remove("CALICHE_ENV_FILE_TEST_NEW");
        let path = path.to_string_lossy().into_owned();
        let args = Args::parse_from(["caliche", "--env-file", path.as_str()]);

        load_env_file_before_runtime(&args).expect("load env file");

        assert_eq!(
            std::env::var("CALICHE_ENV_FILE_TEST_SHELL").as_deref(),
            Ok("shell-value")
        );
        assert_eq!(
            std::env::var("CALICHE_ENV_FILE_TEST_NEW").as_deref(),
            Ok("file-new")
        );
    }
}
