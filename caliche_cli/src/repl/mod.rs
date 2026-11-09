//! Read-eval-print loop for interactive CLI mode.
//!
//! ## Architecture
//!
//! - [`commands`]: Slash command parsing
//! - [`engine`]: REPL engine orchestration (`run_loop`, `message_turn`, `stream_sink`, `turn_quality`)
//! - [`io`]: Input/output implementations
//! - [`renderer`]: Output rendering utilities
//! - [`terminal`]: Terminal capability helpers

pub mod commands;
mod engine;
mod io;
mod profile_harness;
pub mod renderer;
mod resume;
mod startup;
pub(crate) mod stdout_sync;
pub mod terminal;

pub use commands::SlashCommand;
pub use engine::ReplEngine;
pub use io::{OutputHandler, ReedlineInput, TerminalOutput};
pub use renderer::ReplRenderer;

use crate::core::traits::OutputRenderer;
use anyhow::Result;
use caliche::AgentService;
use std::sync::Arc;
use std::time::Instant;

/// Run the REPL interactive loop.
///
/// Uses the ReplEngine with reedline for full-featured REPL with:
/// - Tab completion for slash commands (e.g., /help, /welcome, /undo, /save, /export, /compact)
/// - Tab completion for slash-command arguments: `/provider`, `/model`, etc.
/// - Tab completion for @ context references (e.g., @file.rs, @diff)
/// - Command history with up/down arrows
/// - Slash commands, context references, and session management
///
/// When `shared_database` is `Some`, reuses its pool for the session store instead of opening a new connection.
/// When `cred_manager` is `Some`, the initial provider/model completion lists are populated from the keychain.
/// When `cancellation_token` is `Some`, Ctrl+C will cancel in-flight agent operations.
/// When `preloaded_config` is `Some`, config is not loaded again (faster REPL startup).
/// When `tools_source` is `Some`, the full banner shows it on the Tools line (e.g. "from --tools" or "from profile X"); `None` means "all defaults".
/// When `debug` is true, the full banner prints the startup duration before the REPL prompt.
/// `startup_log_path` and `startup_messages` are printed via with_stdout before the banner to avoid terminal interleaving.
/// `deferred_workspace` attaches workspace RAG after the prompt is up.
pub async fn run_repl(
    mut service: AgentService,
    session_id: Option<String>,
    save_session: bool,
    autosave_enabled: bool,
    shared_database: Option<&crate::storage::Database>,
    cred_manager: Option<&crate::storage::SecureCredentialManager>,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
    no_auto_skill: bool,
    preloaded_config: Option<crate::config::Config>,
    tools_source: Option<String>,
    debug: bool,
    startup_log_path: Option<&std::path::Path>,
    startup_messages: &[String],
    thin_client: Option<(crate::client::AgentClient, String)>,
    mut deferred_workspace: Option<crate::deferred::DeferredWorkspaceRag>,
) -> Result<()> {
    use crate::core::state::InteractiveState;
    use crate::input::CompletionData;
    use uuid::Uuid;

    let startup_start = Instant::now();
    let session_id = session_id.unwrap_or_else(|| Uuid::now_v7().to_string());
    let bundle =
        startup::load_repl_session_and_config(&session_id, shared_database, preloaded_config)
            .await?;
    let session_store = bundle.session_store;
    let cli_session = bundle.cli_session;
    let config = bundle.config;

    startup::install_repl_memory(&mut service, config.as_ref()).await;

    let provider = service.provider().to_string();
    let model = service.model().to_string();
    let system_prompt = service.agent().system_prompt().to_string();

    let configured_providers = if let Some(cm) = cred_manager {
        startup::configured_providers_from_manager(cm.list_configured().await)
    } else {
        Vec::new()
    };
    let completion_data = Arc::new(std::sync::RwLock::new(CompletionData::new(
        configured_providers,
        &provider,
    )));

    let mut state = InteractiveState::new(cli_session, &provider, &model, system_prompt);
    startup::apply_config_to_repl_state(&mut state, config.as_ref());
    state.autosave_enabled = autosave_enabled;
    state.save_session = save_session;
    startup::apply_shared_database_settings(&mut state, shared_database).await;

    apply_persisted_embedding_settings(&mut service, &state, cred_manager).await;
    apply_persisted_image_settings(&mut service, &state, cred_manager).await;
    service.set_subagent_selection(profile_harness::selection_from_state(
        state.subagent_provider.as_deref(),
        state.subagent_model.as_deref(),
    ));

    if let Some(ref mut deferred) = deferred_workspace {
        let _ = deferred.poll_apply(&mut service).await;
    }

    let formatter_rules = config
        .as_ref()
        .map(crate::bootstrap::formatter_rules_from_config)
        .unwrap_or_default();
    let notify_command = config.as_ref().and_then(|c| c.notify_command.clone());
    let max_new_doc_files = config.as_ref().and_then(|c| c.max_new_files).unwrap_or(3);
    let tool_execution_driver = config
        .as_ref()
        .and_then(|c| c.tool_execution_driver.clone());
    let auto_skill = !no_auto_skill && config.as_ref().and_then(|c| c.auto_skill).unwrap_or(true);
    let use_harness_turns = config
        .as_ref()
        .and_then(|c| c.memory.as_ref().and_then(|m| m.use_harness_turns))
        .unwrap_or(false);
    let auto_decompose_repl = config
        .as_ref()
        .and_then(|c| c.planning.as_ref().and_then(|p| p.auto_decompose))
        .unwrap_or(false);

    let (input, mut renderer) = startup::new_repl_input(completion_data, cancellation_token)?;
    renderer.set_output_verbosity(state.output_verbosity);

    startup::print_startup_banner(startup::ReplBannerArgs {
        provider: &provider,
        model: &model,
        session_id: &session_id,
        tools_source: tools_source.as_deref(),
        startup_log_path,
        startup_messages,
        debug,
        startup_start,
        has_workspace_rag: service.has_workspace_rag(),
        deferred_workspace: deferred_workspace.is_some(),
        tool_list: service.list_tools(),
        mode: startup::repl_banner_mode(config.as_ref()),
    });

    startup::spawn_workspace_index_wait(&service);
    let path_restriction_extra_allowed = startup::register_path_restriction_hook(&mut service);

    let mention_policy = config
        .as_ref()
        .and_then(|c| c.orchestration.as_ref())
        .map(crate::config::OrchestrationConfig::mention_policy)
        .unwrap_or_else(caliche::orchestration::TaskInvocationPolicy::allow_all);

    let resumed = if let Some((client, remote_session_id)) = thin_client.as_ref() {
        resume::install_resumed_context(
            &mut service,
            &state.messages,
            Some((client, remote_session_id.as_str())),
        )
        .await?
    } else if !state.messages.is_empty() {
        resume::install_resumed_context(&mut service, &state.messages, None).await?
    } else {
        None
    };

    let completion_handle = input.completion_data();
    let mut engine = ReplEngine::new_with_completions(
        input,
        renderer,
        state,
        completion_handle,
        notify_command.clone(),
        max_new_doc_files,
        path_restriction_extra_allowed,
    )
    .with_mention_policy(mention_policy);
    if let Some((client, remote_session_id)) = thin_client {
        engine = engine.with_thin_client(client, remote_session_id);
        crate::repl::stdout_sync::with_stdout(write_thin_client_transport_line);
        if let Some(installed) = resumed.as_ref() {
            let line = resume::resume_status_line(installed);
            crate::repl::stdout_sync::with_stdout(|out| {
                use std::io::Write;
                let _ = writeln!(out, "\x1b[90m{line}\x1b[0m");
            });
        }
    }
    if let Some(deferred) = deferred_workspace {
        engine = engine.with_deferred_workspace(deferred);
    }

    engine.spawn_initial_model_fetch(&provider);
    engine.refresh_context_ref_completions(&service);

    Box::pin(engine.run(
        &mut service,
        session_store,
        Some((
            formatter_rules,
            notify_command,
            tool_execution_driver,
            auto_skill,
            use_harness_turns,
            auto_decompose_repl,
        )),
        cancellation_token,
        shared_database,
    ))
    .await?;

    startup::persist_last_provider_model(&service, shared_database).await;
    crate::print_goodbye(crate::generate_exit_message());
    crate::print_session_resume(&session_id);
    Ok(())
}

fn write_thin_client_transport_line(out: &mut std::io::StdoutLock<'_>) {
    use std::io::Write;
    let _ = writeln!(out, "\x1b[90mTransport:\x1b[0m thin-client (HTTP + SSE)");
}

async fn apply_persisted_embedding_settings(
    service: &mut AgentService,
    state: &crate::core::state::InteractiveState,
    cred_manager: Option<&crate::storage::SecureCredentialManager>,
) {
    use caliche::agent::service::ProviderFactory;
    use caliche::llm::ProviderType;

    let Some(provider_name) = state.embedding_provider.as_deref() else {
        return;
    };
    let Ok(provider_type) = provider_name.parse::<ProviderType>() else {
        tracing::warn!("Ignoring persisted invalid embedding provider: {provider_name}");
        return;
    };
    let Some(model) = state
        .embedding_model
        .clone()
        .or_else(|| default_embedding_model(provider_type))
    else {
        tracing::warn!("Ignoring persisted unsupported embedding provider: {provider_name}");
        return;
    };

    match ProviderFactory::create_embedding_provider(
        provider_type,
        model.clone(),
        crate::storage::session_credential_provider(cred_manager).await,
    )
    .await
    {
        Ok(Some(provider)) => service.set_embedding_selection(
            Some(provider),
            Some(provider_type.to_string()),
            Some(model),
        ),
        Ok(None) => tracing::warn!(
            "Persisted embedding provider does not support embeddings: {provider_name}"
        ),
        Err(error) => tracing::warn!(
            error = %error,
            "failed to restore persisted embedding provider"
        ),
    }
}

fn default_embedding_model(provider_type: caliche::llm::ProviderType) -> Option<String> {
    caliche::agent::service::default_embedding_model(provider_type)
}

async fn apply_persisted_image_settings(
    service: &mut AgentService,
    state: &crate::core::state::InteractiveState,
    cred_manager: Option<&crate::storage::SecureCredentialManager>,
) {
    use caliche::agent::service::ProviderFactory;
    use caliche::llm::ProviderType;

    if state.artifact_dir.is_some() || service.artifact_store().is_none() {
        let store = caliche::ArtifactStore::for_workspace(
            &state.working_directory,
            state.artifact_dir.as_deref(),
        );
        service.set_artifact_store(Some(Arc::new(store)));
    }

    let Some(provider_name) = state.image_generation_provider.as_deref() else {
        return;
    };
    let Ok(provider_type) = provider_name.parse::<ProviderType>() else {
        tracing::warn!("Ignoring persisted invalid image generation provider: {provider_name}");
        return;
    };

    match ProviderFactory::create_image_generation_provider(
        provider_type,
        state.image_generation_model.clone(),
        crate::storage::session_credential_provider(cred_manager).await,
    )
    .await
    {
        Ok(Some(provider)) => service.set_image_generation_provider(Some(provider)),
        Ok(None) => tracing::warn!(
            "Persisted image generation provider does not support image generation: {provider_name}"
        ),
        Err(error) => tracing::warn!(
            error = %error,
            "failed to restore persisted image generation provider"
        ),
    }
}
