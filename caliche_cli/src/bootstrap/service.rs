//! Agent service construction for CLI bootstrap.

use super::{
    RESEARCH_AGENT_SYSTEM_PROMPT, WorkspaceEmbedOptions, configured_artifact_dir,
    configured_image_generation, effective_learning_config, effective_workspace_embed,
    parse_provider_model, use_builder_path, workspace_indexer_config,
};
use crate::commands::Args;
use crate::storage::SecureCredentialManager;
use anyhow::Result;
use caliche::agent::core::ToolExecutionDriver;
use caliche::agent::core::compaction::{
    CompactionConfig, CompactionStrategy, SlidingWindowCompaction,
};
use caliche::agent::service::{
    ProviderFactory, resolve_provider, resolve_runtime_identity_with_discovery,
};
use caliche::agent::subagent::SubagentModelSelection;
use caliche::agent::tools::session_tools::SessionToolsConfig;
use caliche::agent::tools::{ConfigurableResearchServiceFactory, ParallelResearchRunner};
use caliche::harness::HarnessSessionSpawner;
use caliche::llm::{
    FailoverConfig, FailoverConfigBuilder, ImageGenerationProvider, LlmClient, ProviderType,
};
use caliche::traits::credentials::CredentialProvider;
use caliche::{AgentBuilder, AgentService, ArtifactStore};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

fn startup_model_for_provider(
    resolved_provider: ProviderType,
    model: Option<String>,
    args: &Args,
    cli_config: Option<&crate::config::Config>,
) -> Option<String> {
    #[cfg(feature = "local-inference-base")]
    if matches!(resolved_provider, ProviderType::LocalInference) {
        return model
            .or_else(|| args.local_inference_download_repo.clone())
            .or_else(|| cli_config.and_then(|cfg| cfg.local_inference_download_repo.clone()));
    }
    #[cfg(not(feature = "local-inference-base"))]
    {
        let _ = (resolved_provider, args, cli_config);
    }
    model
}

fn config_needs_local_inference_builder(cli_config: Option<&crate::config::Config>) -> bool {
    cli_config.is_some_and(|cfg| {
        cfg.local_inference_download_repo.is_some()
            || cfg.local_inference_backend.is_some()
            || cfg.local_inference_model_dir.is_some()
            || cfg.local_inference_hf_cache_dir.is_some()
    })
}

async fn agent_builder_with_failover(
    prov: ProviderType,
    model_name: String,
    args: &Args,
    cli_config: Option<&crate::config::Config>,
) -> Result<(AgentBuilder, String)> {
    let fallbacks: Vec<(ProviderType, String)> = args
        .failover_fallback
        .iter()
        .map(|s| parse_provider_model(s))
        .collect::<Result<Vec<_>>>()?;
    let mut failover_builder = FailoverConfig::builder().primary(prov, model_name);
    for (p, m) in fallbacks {
        failover_builder = failover_builder.fallback(p, m);
    }
    failover_builder = apply_auth_profiles_from_config(failover_builder, cli_config);
    let failover_config = failover_builder.build()?;
    let client = LlmClient::builder()
        .failover_config(failover_config)
        .build()
        .await?;
    let provider_instance = client.provider().clone();
    let model_used = client.model().to_string();
    Ok((
        AgentBuilder::new()
            .provider_instance(provider_instance)
            .model(model_used.clone()),
        model_used,
    ))
}

fn apply_auth_profiles_from_config(
    mut failover_builder: FailoverConfigBuilder,
    cli_config: Option<&crate::config::Config>,
) -> FailoverConfigBuilder {
    let Some(cfg) = cli_config else {
        return failover_builder;
    };
    for (provider_name, provider_cfg) in &cfg.providers {
        if provider_cfg.auth_profiles.is_empty() {
            continue;
        }
        let Ok(provider_type) = ProviderType::from_str(provider_name) else {
            tracing::warn!(
                provider = %provider_name,
                "Ignoring auth_profiles for unknown provider name"
            );
            continue;
        };
        failover_builder =
            failover_builder.auth_profiles(provider_type, provider_cfg.to_auth_profiles());
    }
    failover_builder
}

fn config_has_auth_profiles(cli_config: Option<&crate::config::Config>) -> bool {
    cli_config.is_some_and(|cfg| cfg.providers.values().any(|p| !p.auth_profiles.is_empty()))
}

async fn initial_cli_agent_builder(
    prov: ProviderType,
    model_name: String,
    args: &Args,
    cli_config: Option<&crate::config::Config>,
) -> Result<(AgentBuilder, String)> {
    if args.failover_fallback.is_empty() && !config_has_auth_profiles(cli_config) {
        return Ok((
            AgentBuilder::new().provider(prov).model(model_name.clone()),
            model_name,
        ));
    }
    agent_builder_with_failover(prov, model_name, args, cli_config).await
}

fn apply_image_generation_provider(
    builder: AgentBuilder,
    image_generation_provider: &Option<Arc<dyn ImageGenerationProvider>>,
) -> AgentBuilder {
    match image_generation_provider.as_ref().map(Arc::clone) {
        Some(provider) => builder.image_generation_provider_instance(provider),
        None => builder,
    }
}

fn apply_compaction_config(builder: AgentBuilder, args: &Args) -> Result<AgentBuilder> {
    if !args.compaction {
        return Ok(builder);
    }
    let strategy: Arc<dyn CompactionStrategy> =
        Arc::new(SlidingWindowCompaction::builder().build());
    let compaction_config = CompactionConfig::builder()
        .strategy(strategy)
        .trigger_threshold(args.compaction_trigger.unwrap_or(0.90))
        .target_threshold(args.compaction_target.unwrap_or(0.70))
        .max_attempts(args.compaction_max_attempts.unwrap_or(2))
        .build()?;
    Ok(builder.compaction_config(compaction_config))
}

fn apply_context_window(
    builder: AgentBuilder,
    args: &Args,
    cli_config: Option<&crate::config::Config>,
) -> AgentBuilder {
    match args
        .context_window
        .or_else(|| cli_config.and_then(|cfg| cfg.context_window))
    {
        Some(context_window) => builder.context_window(context_window),
        None => builder,
    }
}

#[cfg(feature = "local-inference-base")]
fn first_owned_or_config(arg: Option<&String>, config_value: Option<String>) -> Option<String> {
    arg.cloned().or(config_value)
}

#[cfg(feature = "local-inference-base")]
fn first_path_or_config(arg: Option<&PathBuf>, config_value: Option<PathBuf>) -> Option<PathBuf> {
    arg.cloned().or(config_value)
}

#[cfg_attr(not(feature = "local-inference-base"), allow(unused_mut))]
fn apply_local_inference_options(
    mut builder: AgentBuilder,
    args: &Args,
    cli_config: Option<&crate::config::Config>,
    prov: ProviderType,
) -> AgentBuilder {
    #[cfg(not(feature = "local-inference-base"))]
    {
        let _ = (args, cli_config, prov);
        return builder;
    }
    #[cfg(feature = "local-inference-base")]
    {
        if !matches!(prov, ProviderType::LocalInference) {
            return builder;
        }
        if let Some(repo) = first_owned_or_config(
            args.local_inference_download_repo.as_ref(),
            cli_config.and_then(|cfg| cfg.local_inference_download_repo.clone()),
        ) {
            let revision = first_owned_or_config(
                args.local_inference_download_revision.as_ref(),
                cli_config.and_then(|cfg| cfg.local_inference_download_revision.clone()),
            );
            builder = builder.local_inference_download_target(repo, revision);
        }
        if let Some(backend) = first_owned_or_config(
            args.local_inference_backend.as_ref(),
            cli_config.and_then(|cfg| cfg.local_inference_backend.clone()),
        ) {
            builder = builder.local_inference_backend(backend);
        }
        if let Some(dir) = first_path_or_config(
            args.local_inference_model_dir.as_ref(),
            cli_config.and_then(|cfg| cfg.local_inference_model_dir.clone()),
        ) {
            builder = builder.local_inference_model_dir(dir);
        }
        if let Some(dir) = first_path_or_config(
            args.local_inference_hf_cache_dir.as_ref(),
            cli_config.and_then(|cfg| cfg.local_inference_hf_cache_dir.clone()),
        ) {
            builder = builder.local_inference_hf_cache_dir(dir);
        }
        builder
    }
}

fn session_tools_enabled(args: &Args, cli_config: Option<&crate::config::Config>) -> bool {
    args.session_tools
        || args.complex
        || cli_config
            .and_then(|cfg| cfg.subagent.as_ref().and_then(|subagent| subagent.complex))
            .unwrap_or(false)
}

fn apply_session_tools(
    builder: AgentBuilder,
    args: &Args,
    provider: ProviderType,
    model: &str,
    workspace_root: PathBuf,
    cli_config: Option<&crate::config::Config>,
) -> AgentBuilder {
    if !session_tools_enabled(args, cli_config) {
        return builder;
    }
    let mut selection = SubagentModelSelection::from_parts(
        args.subagent_provider.as_deref(),
        args.subagent_model.as_deref(),
    );
    if let Some(subagent) = cli_config.and_then(|cfg| cfg.subagent.as_ref()) {
        if selection.provider.is_none() {
            selection.provider = subagent
                .provider
                .as_deref()
                .and_then(|name| ProviderType::from_str(name).ok());
        }
        if selection.model.is_none() {
            selection.model = subagent.model.clone();
        }
    }
    let mut spawner = HarnessSessionSpawner::new(provider, model.to_owned(), workspace_root);
    spawner.selection.set(selection);
    spawner.complex = args.complex
        || cli_config
            .and_then(|cfg| cfg.subagent.as_ref().and_then(|subagent| subagent.complex))
            .unwrap_or(false);
    builder.session_tools_config(
        SessionToolsConfig::builder()
            .spawner(Arc::new(spawner))
            .build(),
    )
}

fn apply_workspace_embedding(
    mut builder: AgentBuilder,
    args: &Args,
    workspace_root: PathBuf,
    workspace_options: WorkspaceEmbedOptions,
) -> AgentBuilder {
    if !effective_workspace_embed(args) || workspace_options.defer_until_interactive {
        return builder;
    }
    if workspace_options.skip_embedding_discovery {
        return builder;
    }
    let config = workspace_indexer_config(workspace_root, &workspace_options);
    if let Some(discovery) = workspace_options.pre_discovered_embedding {
        builder = builder.with_pre_discovered_embedding(discovery);
    }
    builder.with_workspace_embedding(config)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolRosterKind {
    Defaults,
    None,
    Named,
}

fn tool_roster_kind(tools: Option<&[String]>) -> ToolRosterKind {
    match tools {
        None => ToolRosterKind::Defaults,
        Some([]) => ToolRosterKind::None,
        Some(_) => ToolRosterKind::Named,
    }
}

fn apply_enabled_tools(
    builder: AgentBuilder,
    tools: Option<&Vec<String>>,
    args: &Args,
) -> AgentBuilder {
    match tool_roster_kind(tools.map(Vec::as_slice)) {
        ToolRosterKind::Defaults => builder.enable_all_default_tools(),
        // `--tools none` is an explicit empty roster. Do not fall back to defaults.
        ToolRosterKind::None => builder,
        ToolRosterKind::Named => apply_named_tools(builder, tools, args),
    }
}

fn apply_named_tools(
    builder: AgentBuilder,
    tools: Option<&Vec<String>>,
    args: &Args,
) -> AgentBuilder {
    let Some(names) = tools else {
        return builder;
    };
    let mut refs: Vec<&str> = names.iter().map(String::as_str).collect();
    if refs.is_empty() {
        return builder;
    }
    if args.session_tools || args.complex {
        append_session_tool_names(&mut refs);
    }
    builder.enable_tools(&refs)
}

fn append_session_tool_names(refs: &mut Vec<&str>) {
    const SESSION_TOOL_NAMES: &[&str] = &[
        "sessions_list",
        "sessions_history",
        "sessions_send",
        "sessions_spawn",
    ];
    for name in SESSION_TOOL_NAMES {
        if !refs.contains(name) {
            refs.push(name);
        }
    }
}

fn apply_cli_config_builder_fields(
    mut builder: AgentBuilder,
    cli_config: Option<&crate::config::Config>,
) -> AgentBuilder {
    let Some(cfg) = cli_config else {
        return builder;
    };
    if let Some(t) = cfg.temperature {
        builder = builder.temperature(t);
    }
    if let Some(mt) = cfg.max_tokens {
        builder = builder.max_tokens(mt);
    }
    if let Some(ref driver_str) = cfg.tool_execution_driver
        && let Some(driver) = ToolExecutionDriver::parse(driver_str)
    {
        builder = builder.tool_execution_driver(driver);
    }
    builder
}

fn apply_research_runner(
    builder: AgentBuilder,
    args: &Args,
    cli_config: Option<&crate::config::Config>,
    prov: ProviderType,
    model_name: &str,
) -> AgentBuilder {
    if args.is_lean_profile() || !args.failover_fallback.is_empty() {
        return builder;
    }
    let research_tools = vec![
        "fs".to_string(),
        "fs".to_string(),
        "semantic_search".to_string(),
        "workspace".to_string(),
    ];
    let research_model = cli_config
        .and_then(|cfg| cfg.lightweight_model().map(str::to_owned))
        .unwrap_or_else(|| model_name.to_string());
    let factory = Arc::new(ConfigurableResearchServiceFactory::new(
        prov,
        research_model,
        research_tools,
        RESEARCH_AGENT_SYSTEM_PROMPT.to_string(),
    ));
    let runner = Arc::new(ParallelResearchRunner::new(factory, 4));
    builder.research_runner(runner).with_research_skill()
}

fn apply_plugins_store(builder: AgentBuilder, args: &Args) -> AgentBuilder {
    if args.is_lean_profile() {
        return builder;
    }
    builder.with_plugins_store(caliche_server::paths::plugin_store_path())
}

fn apply_learning_loop(
    builder: AgentBuilder,
    learning_config: Option<caliche::agent::learning::CheckpointConfig>,
) -> AgentBuilder {
    match learning_config {
        Some(config) => builder.with_learning_loop(config),
        None => builder,
    }
}

fn apply_discovery_flags(
    mut builder: AgentBuilder,
    args: &Args,
    defer_until_interactive: bool,
) -> AgentBuilder {
    if defer_until_interactive || args.is_lean_profile() {
        return builder;
    }
    builder = builder.auto_discover_mcp(true);
    builder.auto_discover_skills(true)
}

async fn build_cli_agent_via_builder(
    resolved_provider: ProviderType,
    startup_model: String,
    tools: Option<Vec<String>>,
    args: &Args,
    workspace_options: WorkspaceEmbedOptions,
    credential_provider: Arc<dyn CredentialProvider>,
    artifact_store: Arc<ArtifactStore>,
    image_generation_provider: Option<Arc<dyn ImageGenerationProvider>>,
    learning_config: Option<caliche::agent::learning::CheckpointConfig>,
    workspace_root: PathBuf,
    cli_config: Option<&crate::config::Config>,
    defer_missing_credentials: bool,
) -> Result<AgentService> {
    let prov = resolved_provider;
    let defer_until_interactive = workspace_options.defer_until_interactive;
    let (builder, model_name) =
        initial_cli_agent_builder(prov, startup_model, args, cli_config).await?;
    let mut builder = builder
        .credential_provider(Arc::clone(&credential_provider))
        .artifact_store(Arc::clone(&artifact_store))
        .defer_missing_credentials(defer_missing_credentials);
    builder = apply_image_generation_provider(builder, &image_generation_provider);
    builder = apply_compaction_config(builder, args)?;
    builder = apply_context_window(builder, args, cli_config);
    builder = apply_local_inference_options(builder, args, cli_config, prov);
    builder = apply_session_tools(
        builder,
        args,
        prov,
        &model_name,
        workspace_root.clone(),
        cli_config,
    );
    builder = apply_workspace_embedding(builder, args, workspace_root, workspace_options);
    builder = apply_enabled_tools(builder, tools.as_ref(), args);
    builder = apply_cli_config_builder_fields(builder, cli_config);
    builder = apply_research_runner(builder, args, cli_config, prov, &model_name);
    builder = apply_plugins_store(builder, args);
    builder = apply_learning_loop(builder, learning_config);
    builder = builder.max_iterations(if args.complex { 200 } else { 100 });
    builder = apply_discovery_flags(builder, args, defer_until_interactive);
    Box::pin(builder.build()).await
}

async fn optional_image_generation_provider(
    args: &Args,
    cli_config: Option<&crate::config::Config>,
    credential_provider: Arc<dyn CredentialProvider>,
) -> Result<Option<Arc<dyn ImageGenerationProvider>>> {
    let Some((image_provider, image_model)) = configured_image_generation(args, cli_config)? else {
        return Ok(None);
    };
    match ProviderFactory::create_image_generation_provider(
        image_provider,
        image_model,
        credential_provider,
    )
    .await
    {
        Ok(provider) => Ok(provider),
        Err(err)
            if args.server && caliche::llm::unconfigured::should_defer_provider_setup(&err) =>
        {
            // The chat provider is unchanged. Image generation is omitted.
            // This is not a switch to another provider. An egress-blocked
            // image URL is not a reason to refuse listen.
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

/// Create an agent service from provider, model, tools, and optional workspace embedding.
/// Used by both bootstrap create_service and main service creation.
///
/// Credentials come from [`crate::storage::session_credential_provider`].
/// With `defer_missing_credentials`, a provider without a key still builds;
/// its completions fail with the missing-key error until a key is added.
pub async fn create_agent_service(
    provider: Option<ProviderType>,
    model: Option<String>,
    tools: Option<Vec<String>>,
    args: &Args,
    workspace_options: WorkspaceEmbedOptions,
    cred_manager: Option<&SecureCredentialManager>,
    cli_config: Option<&crate::config::Config>,
    defer_missing_credentials: bool,
) -> Result<AgentService> {
    let resolved_provider = resolve_provider(provider);
    let effective_model = startup_model_for_provider(resolved_provider, model, args, cli_config);
    let identity =
        resolve_runtime_identity_with_discovery(Some(resolved_provider), effective_model, None)
            .await?;
    let resolved_provider = identity.provider;
    let startup_model = identity.model;
    let credential_provider = crate::storage::session_credential_provider(cred_manager).await;
    let workspace_root =
        std::env::current_dir().map_err(|e| anyhow::anyhow!("Current dir unavailable: {}", e))?;
    let artifact_store = Arc::new(ArtifactStore::for_workspace(
        &workspace_root,
        configured_artifact_dir(args, cli_config).as_deref(),
    ));
    let image_generation_provider =
        optional_image_generation_provider(args, cli_config, Arc::clone(&credential_provider))
            .await?;

    let learning_config = effective_learning_config(args, cli_config, &workspace_root);
    let use_builder = use_builder_path(args, workspace_options.defer_until_interactive)
        || args.complex
        || session_tools_enabled(args, cli_config)
        || config_needs_local_inference_builder(cli_config)
        || learning_config.is_some();

    let mut service = if use_builder {
        build_cli_agent_via_builder(
            resolved_provider,
            startup_model,
            tools,
            args,
            workspace_options,
            credential_provider,
            artifact_store,
            image_generation_provider,
            learning_config,
            workspace_root,
            cli_config,
            defer_missing_credentials,
        )
        .await?
    } else {
        AgentService::from_config(
            caliche::AgentServiceConfig::new()
                .with_provider(Some(resolved_provider), Some(startup_model))
                .with_context_window(
                    args.context_window
                        .or_else(|| cli_config.and_then(|cfg| cfg.context_window)),
                )
                .with_tool_names_or_defaults(tools)
                .with_image_generation_provider(image_generation_provider)
                .with_artifact_store(Some(artifact_store))
                .with_credential_provider(credential_provider)
                .with_defer_missing_credentials(defer_missing_credentials),
        )
        .await?
    };

    super::finalize_cli_service_with(
        &mut service,
        effective_workspace_embed(args),
        args.complex,
        args.is_lean_profile(),
    );
    Ok(service)
}

#[cfg(test)]
mod tests {
    use super::{ToolRosterKind, create_agent_service, tool_roster_kind};
    use crate::bootstrap::WorkspaceEmbedOptions;
    use crate::commands::Args;
    use crate::runtime::parse_tool_names;
    use caliche::llm::ProviderType;
    use clap::Parser;

    #[test]
    fn none_roster_does_not_fall_back_to_defaults() {
        assert_eq!(tool_roster_kind(None), ToolRosterKind::Defaults);
        assert_eq!(tool_roster_kind(Some(&[])), ToolRosterKind::None);
        let named = vec!["fs".to_string()];
        assert_eq!(tool_roster_kind(Some(&named)), ToolRosterKind::Named);
    }

    #[tokio::test]
    async fn none_tools_builds_an_empty_registry() {
        let args = Args::parse_from([
            "caliche",
            "--tools",
            "none",
            "--profile",
            "lean",
            "--compaction",
            "--server",
            "--provider",
            "openai",
            "--model",
            "gpt-4o-mini",
        ]);
        let tools = parse_tool_names(&args);
        assert!(
            tools.as_ref().is_some_and(Vec::is_empty),
            "none must be an explicit empty roster, got {tools:?}"
        );
        let service = create_agent_service(
            Some(ProviderType::OpenAI),
            Some("gpt-4o-mini".to_owned()),
            tools,
            &args,
            WorkspaceEmbedOptions::default(),
            None,
            None,
            args.server,
        )
        .await
        .expect("none roster builds");
        let names: Vec<String> = service
            .list_tools()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        assert!(
            names.is_empty(),
            "--tools none must not register fs, bash, or web_fetch: {names:?}"
        );
    }
}
