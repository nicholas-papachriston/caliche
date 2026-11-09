#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "CLI binary intentionally writes to stdout/stderr"
)]

use anyhow::Context;
use caliche_cli::{bootstrap, runtime};
use caliche_cli::{commands::Args, handlers, server};
use clap::Parser;
use std::sync::Arc;

/// Stream a single message that includes binary media attachments (images, audio, video, docs).
async fn run_single_message_multimodal(
    service: &mut caliche::AgentService,
    message: &str,
    attachments: Vec<caliche::messages::Attachment>,
    output_format: &caliche_cli::commands::OutputFormat,
    quiet: bool,
    cancellation_token: Option<&caliche_cli::cancellation::CancellationToken>,
) -> anyhow::Result<()> {
    if !quiet {
        caliche_cli::output::eprint_multimodal_banner(&attachments);
    }

    let cancel_flag =
        cancellation_token.map(caliche_cli::cancellation::CancellationToken::shared_cancelled_flag);
    let stream =
        service.stream_chat_with_attachments_and_cancel_flag(message, attachments, cancel_flag);

    caliche_cli::output::run_multimodal_oneshot(stream, output_format, cancellation_token).await
}

fn register_interactive_permission(args: &Args, service: &mut caliche::AgentService) {
    if let Some(ref mode_str) = args.permission_mode
        && mode_str == "interactive"
    {
        use caliche::agent::hooks::{HookMatcher, LifecycleHook};
        use caliche_cli::permission_prompt::InteractivePermissionHook;

        let hook = Arc::new(InteractivePermissionHook::new(true)) as Arc<dyn LifecycleHook>;
        service.agent_mut().register_hook(HookMatcher::All, hook);
        tracing::info!("Interactive permission mode enabled");
    }
}

async fn configure_sandbox(
    args: &Args,
    service: &mut caliche::AgentService,
) -> anyhow::Result<Option<String>> {
    let sandboxed = caliche_cli::sandbox_startup::configure_sandbox(
        service,
        args.sandbox.as_deref(),
        args.sandbox_backend.as_deref(),
    )
    .await?;
    if sandboxed && !args.quiet {
        let policy = args
            .sandbox
            .as_deref()
            .unwrap_or(caliche_cli::sandbox_startup::DEFAULT_POLICY);
        Ok(Some(format!("\x1b[2mSandbox enabled: {policy}\x1b[0m")))
    } else {
        Ok(None)
    }
}

fn should_attach_stdio_approval(args: &Args) -> bool {
    args.permission_mode
        .as_deref()
        .is_none_or(|m| m == "default" || m == "interactive")
}

fn attach_stdio_approval(service: &mut caliche::AgentService) {
    service.agent_mut().set_approval_resolver(Some(Arc::new(
        caliche_cli::approval_resolver::StdioApprovalResolver::new(),
    )));
    service.agent_mut().set_amendment_persister(Some(Arc::new(
        caliche_cli::approval_resolver::CliAmendmentPersister::new(None),
    )));
}

fn register_exec_policy(
    service: &mut caliche::AgentService,
    args: &Args,
    rules: Vec<caliche::agent::hooks::BashRule>,
) {
    use caliche::agent::hooks::{ExecPolicy, ExecPolicyHook, HookMatcher, LifecycleHook};
    let policy = ExecPolicy::new(rules);
    let hook = Arc::new(ExecPolicyHook::new(Arc::new(policy))) as Arc<dyn LifecycleHook>;
    service
        .agent_mut()
        .register_hook(HookMatcher::tool("bash"), hook);
    if should_attach_stdio_approval(args) {
        attach_stdio_approval(service);
    }
}

fn apply_agent_profile_prompt(
    service: &mut caliche::AgentService,
    profile: &caliche::agent::profile::AgentProfile,
) {
    let Some(body) = profile.system_prompt() else {
        return;
    };
    let current = service.agent().system_prompt().to_string();
    let combined = if current.trim().is_empty() {
        body.to_string()
    } else {
        let mut combined = String::with_capacity(body.len() + 2 + current.len());
        combined.push_str(body);
        combined.push_str("\n\n");
        combined.push_str(&current);
        combined
    };
    service.agent_mut().set_system_prompt(combined);
}

fn apply_agent_profile_bash_rules(
    args: &Args,
    service: &mut caliche::AgentService,
    profile: &caliche::agent::profile::AgentProfile,
) -> anyhow::Result<()> {
    let Some(ref perm) = profile.permission else {
        return Ok(());
    };
    let Some(ref rules) = perm.bash else {
        return Ok(());
    };
    if rules.is_empty() {
        return Ok(());
    }
    for (i, rule) in rules.iter().enumerate() {
        if let Err(e) = rule.validate_examples() {
            anyhow::bail!(
                "profile '{}' permissions.bash rule {}: {}",
                profile.description.as_deref().unwrap_or("unknown"),
                i + 1,
                e
            );
        }
    }
    register_exec_policy(service, args, rules.clone());
    Ok(())
}

fn apply_agent_profile(
    args: &Args,
    agent_profile: Option<&caliche::agent::profile::AgentProfile>,
    service: &mut caliche::AgentService,
) -> anyhow::Result<()> {
    let Some(profile) = agent_profile else {
        return Ok(());
    };
    apply_agent_profile_prompt(service, profile);
    apply_agent_profile_bash_rules(args, service, profile)
}

fn spawn_discovery(debug: bool) {
    // The snapshot is only logged. Skills, commands, and MCP load their own
    // slices later. A normal start must not walk those trees on a worker.
    if !debug {
        return;
    }
    let discovery_options = caliche::discovery::auto::DiscoveryOptions::default();
    tokio::task::spawn_blocking(move || {
        let discovery_snapshot = caliche::discovery::auto::discover_all(&discovery_options);
        discovery_snapshot.log_summary("cli.bootstrap");
    });
}

async fn apply_project_context(
    args: &Args,
    config: Option<&caliche_cli::config::Config>,
    service: &mut caliche::AgentService,
) -> Option<String> {
    if args.is_lean_profile() {
        return None;
    }
    let project_config = handlers::load_project_config().await?;
    let project_path = project_config.path.clone();
    let current_prompt = service.agent().system_prompt().to_string();
    let project_section =
        service.budget_project_context(project_config.into_context(project_context_share(config)));
    let combined = if current_prompt.trim().is_empty() {
        project_section
    } else {
        caliche_cli::prompts::PromptBuilder::new()
            .base(current_prompt)
            .project_context_owned(project_section)
            .build()
    };
    service.agent_mut().set_system_prompt(combined);
    if args.quiet {
        None
    } else {
        Some(format!(
            "Loaded project context from {}",
            project_path.display()
        ))
    }
}

/// `project_context_share` from the user configuration, else the default.
fn project_context_share(config: Option<&caliche_cli::config::Config>) -> u8 {
    config
        .and_then(|cfg| cfg.project_context_share)
        .unwrap_or(caliche::DEFAULT_PROJECT_CONTEXT_SHARE)
}

fn bash_rules_from_config(
    cfg: &caliche_cli::config::Config,
) -> Vec<caliche::agent::hooks::BashRule> {
    let rules = cfg
        .permissions
        .as_ref()
        .and_then(|p| p.bash.as_ref())
        .filter(|v| !v.is_empty())
        .cloned();
    let bash_default_deny = cfg.permissions.as_ref().and_then(|p| p.bash_default)
        == Some(caliche_cli::config::BashDefaultMode::Deny);
    if let Some(r) = rules {
        r
    } else if bash_default_deny {
        use caliche::agent::hooks::{BashDecision, BashRule, PatternPos};
        vec![BashRule {
            pattern: vec![PatternPos::One("*".to_string())],
            decision: BashDecision::Forbidden,
            justification: Some(
                "Bash denied by default (permissions.bash_default = deny)".to_string(),
            ),
            match_examples: None,
            not_match_examples: None,
        }]
    } else {
        vec![]
    }
}

fn register_retry_advisor(service: &mut caliche::AgentService) {
    use caliche::agent::hooks::{HookMatcher, LifecycleHook, RetryAdvisorHook};
    let hook = Arc::new(RetryAdvisorHook::default()) as Arc<dyn LifecycleHook>;
    service.agent_mut().register_hook(HookMatcher::All, hook);
    tracing::debug!("Retry advisor hook registered");
}

fn register_narration_guard(service: &mut caliche::AgentService) {
    use caliche::agent::hooks::{
        ActionNarrationGuardConfig, ActionNarrationGuardMiddleware, HookMatcher,
    };
    let hook: Arc<dyn caliche::agent::hooks::LifecycleHook> = Arc::new(
        ActionNarrationGuardMiddleware::new(ActionNarrationGuardConfig::default()),
    );
    service
        .agent_mut()
        .register_hook(HookMatcher::All, Arc::clone(&hook));
    service.agent_mut().register_completion_hook(hook);
    tracing::debug!("Action narration guard middleware registered");
}

fn register_resolved_middleware(
    service: &mut caliche::AgentService,
    cfg: &caliche_cli::config::Config,
) {
    let mw = cfg.middleware.clone().unwrap_or_default();
    let (pcc, ld, eo) = mw.resolved_hooks();
    if let Some(config) = pcc {
        use caliche::agent::hooks::{HookMatcher, PreCompletionChecklistMiddleware};
        let hook: Arc<dyn caliche::agent::hooks::LifecycleHook> =
            Arc::new(PreCompletionChecklistMiddleware::new(config));
        service
            .agent_mut()
            .register_hook(HookMatcher::All, Arc::clone(&hook));
        service.agent_mut().register_completion_hook(hook);
        tracing::debug!("Pre-completion checklist middleware registered");
    }
    if let Some(config) = ld {
        use caliche::agent::hooks::{HookMatcher, LoopDetectionMiddleware};
        let hook = Arc::new(LoopDetectionMiddleware::new(config));
        service.agent_mut().register_hook(HookMatcher::All, hook);
        tracing::debug!("Loop detection middleware registered");
    }
    if let Some(config) = eo {
        use caliche::agent::hooks::{EnvironmentOnboardingMiddleware, HookMatcher};
        let hook = Arc::new(EnvironmentOnboardingMiddleware::new(config));
        service.agent_mut().register_hook(HookMatcher::All, hook);
        tracing::debug!("Environment onboarding middleware registered");
    }
}

fn apply_preloaded_config(
    args: &Args,
    preloaded_config: Option<&caliche_cli::config::Config>,
    service: &mut caliche::AgentService,
) -> anyhow::Result<()> {
    let Some(cfg) = preloaded_config else {
        return Ok(());
    };
    bootstrap::apply_cli_config_to_service(service, cfg)?;
    let rules_to_use = bash_rules_from_config(cfg);
    if !rules_to_use.is_empty() {
        let n = rules_to_use.len();
        register_exec_policy(service, args, rules_to_use);
        tracing::debug!("Bash exec policy hook registered ({} rules)", n);
    }
    register_retry_advisor(service);
    register_narration_guard(service);
    register_resolved_middleware(service, cfg);
    Ok(())
}

fn print_provider_model_if_needed(
    args: &Args,
    use_interactive: bool,
    service: &caliche::AgentService,
) {
    if !args.acp_stdio
        && !use_interactive
        && !args.quiet
        && matches!(
            args.output_format,
            caliche_cli::commands::OutputFormat::Text
        )
    {
        println!("Provider: {}", service.provider());
        println!("Model: {}", service.model());
    }
}

fn maybe_list_tools(args: &Args, service: &caliche::AgentService) -> bool {
    if !args.list_tools {
        return false;
    }
    println!("\nAvailable tools:");
    for (name, description, _) in service.list_tools() {
        println!("  - {name}: {description}");
    }
    true
}

async fn maybe_run_advanced_features(
    args: &Arc<Args>,
    startup_ctx: &bootstrap::StartupContext,
    cancellation_token: &caliche_cli::cancellation::CancellationToken,
) -> anyhow::Result<bool> {
    use caliche_cli::output_mode::OutputMode;
    use caliche_cli::storage::{credentials::SecureCredentialManager, database::Database};
    use caliche_cli::traits::command::{CommandContext, CommandHandler};
    use caliche_cli::traits::factory::create_service_factory;
    use handlers::advanced::{AdvancedFeaturesArgs, AdvancedFeaturesHandler};

    if !runtime::has_advanced_feature_args(args.as_ref()) {
        return Ok(false);
    }

    let cred_owned: SecureCredentialManager;
    let db_owned: Database;
    let (adv_cred, adv_db): (&SecureCredentialManager, &Database) =
        if let (Some(c), Some(d)) = (&startup_ctx.cred_manager, &startup_ctx.database) {
            (c, d)
        } else {
            cred_owned = SecureCredentialManager::new()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize credential manager: {e}"))?;
            db_owned = Database::open()
                .await
                .map_err(|e| anyhow::anyhow!("Failed to initialize database: {e}"))?;
            (&cred_owned, &db_owned)
        };
    let service_factory = create_service_factory();
    let mut ctx = CommandContext {
        credentials: adv_cred,
        database: adv_db,
        service_factory,
        output_mode: OutputMode::default(),
        verbose: args.verbose,
        working_dir: std::env::current_dir().unwrap_or_default(),
        cancellation_token: Some(cancellation_token),
        provider: args.provider.clone(),
        model: args.model.clone(),
        yolo: args.yolo,
    };
    let advanced_args = AdvancedFeaturesArgs(Arc::clone(args));
    match AdvancedFeaturesHandler
        .execute(advanced_args, &mut ctx)
        .await
    {
        Ok(_) => {}
        Err(e) => {
            if e.to_string().contains("No advanced features") {
                return Ok(true);
            }
            eprintln!("Error in advanced features: {e}");
            return Err(anyhow::anyhow!("{e}"));
        }
    }
    Ok(true)
}

async fn maybe_run_stdio_or_server(
    args: &Args,
    service: caliche::AgentService,
    startup_ctx: &bootstrap::StartupContext,
) -> anyhow::Result<Option<caliche::AgentService>> {
    if args.mcp_stdio {
        server::handle_mcp_stdio(service).await?;
        return Ok(None);
    }
    if args.acp_stdio {
        let cli_config = caliche_cli::config::load_standard_config_strict().await?;
        let shared_service = std::sync::Arc::new(tokio::sync::RwLock::new(service));
        server::handle_acp_stdio(shared_service, args, Some(&cli_config)).await?;
        return Ok(None);
    }
    if args.server {
        let cli_config = caliche_cli::config::load_standard_config_strict().await?;
        let shared_service = std::sync::Arc::new(tokio::sync::RwLock::new(service));
        let router_options = server::build_router_options(
            shared_service.clone(),
            args,
            Some(&cli_config),
            startup_ctx.database.clone(),
        )
        .await?;
        server::start_server(
            shared_service,
            args.bind.clone(),
            args.port,
            args.mcp_transport.clone(),
            Some(router_options),
        )
        .await?;
        return Ok(None);
    }
    Ok(Some(service))
}

fn is_binary_attachment_path(path: &str) -> bool {
    let p = std::path::Path::new(path);
    let mt = caliche::messages::detect_media_type(p);
    let at = caliche::messages::detect_attachment_type(&mt);
    matches!(
        at,
        caliche::messages::AttachmentType::Image
            | caliche::messages::AttachmentType::Audio
            | caliche::messages::AttachmentType::Video
            | caliche::messages::AttachmentType::Document
    )
}

fn append_text_attach_refs(mut message: String, attach: &[String]) -> String {
    for path in attach {
        if !is_binary_attachment_path(path) {
            message.push(' ');
            message.push('@');
            message.push_str(path);
        }
    }
    message
}

async fn load_binary_attachments(attach: &[String]) -> Vec<caliche::messages::Attachment> {
    let mut binary_attachments = Vec::new();
    for path in attach {
        if !is_binary_attachment_path(path) {
            continue;
        }
        match caliche::messages::Attachment::from_file(std::path::Path::new(path)).await {
            Ok(att) => binary_attachments.push(att),
            Err(e) => {
                eprintln!("\x1b[33mWarning: could not load attachment {path}: {e}\x1b[0m");
            }
        }
    }
    binary_attachments
}

async fn persist_last_provider_model(
    startup_ctx: &bootstrap::StartupContext,
    service: &caliche::AgentService,
) {
    let Some(ref db) = startup_ctx.database else {
        return;
    };
    let provider = service.provider().to_string();
    let model = service.model().to_string();
    if let Err(e) = db.save_setting("last_provider", &provider).await {
        tracing::warn!(error = %e, "failed to persist last_provider setting");
    }
    if let Err(e) = db.save_setting("last_model", &model).await {
        tracing::warn!(error = %e, "failed to persist last_model setting");
    }
}

async fn resolve_oneshot_session(
    args: &Args,
    store: &caliche_cli::storage::SessionStore,
) -> anyhow::Result<Option<caliche_cli::session::CliSession>> {
    if let Some(ref sid) = args.session {
        return Ok(Some(store.load_async(sid).await?));
    }
    if !args.continue_session {
        return Ok(None);
    }
    if let Some(s) = store.load_last_async().await? {
        Ok(Some(s))
    } else {
        eprintln!("No previous session to continue. Start a new conversation first.");
        caliche_cli::shutdown::exit(1);
    }
}

async fn run_oneshot_message(
    args: &Args,
    service: &mut caliche::AgentService,
    cancellation_token: &caliche_cli::cancellation::CancellationToken,
    startup_ctx: &bootstrap::StartupContext,
) -> anyhow::Result<bool> {
    let Some(message) = args.effective_message() else {
        return Ok(false);
    };
    let message = if args.attach.is_empty() {
        message
    } else {
        append_text_attach_refs(message, &args.attach)
    };
    if args.yolo {
        use caliche::agent::hooks::permission::PermissionMode;
        service
            .agent_mut()
            .set_permission_mode(PermissionMode::AcceptAll);
    }
    if args.complex && args.attach.is_empty() {
        let config = caliche::ComplexTaskConfig {
            max_rounds: args.complex_max_rounds.unwrap_or(8),
            verify_before_finish: true,
        };
        let outcome = caliche::run_complex_task(service, &message, &config).await?;
        println!("{}", outcome.text);
        persist_last_provider_model(startup_ctx, service).await;
        return Ok(true);
    }
    let mut binary_attachments = load_binary_attachments(&args.attach).await;
    if binary_attachments.is_empty() {
        let store = caliche_cli::storage::SessionStore::new().await?;
        let existing_session = resolve_oneshot_session(args, &store).await?;
        handlers::run_single_message_session(
            service,
            message,
            &args.output_format,
            args.quiet,
            Some(cancellation_token),
            &store,
            existing_session,
        )
        .await?;
    } else {
        let resolved = caliche_cli::context_refs::ContextResolver::from_service(service)
            .resolve_prompt(&message)
            .await;
        binary_attachments.extend(resolved.attachments);
        run_single_message_multimodal(
            service,
            &resolved.message,
            binary_attachments,
            &args.output_format,
            args.quiet,
            Some(cancellation_token),
        )
        .await?;
    }
    persist_last_provider_model(startup_ctx, service).await;
    Ok(true)
}

async fn resolve_continue_session_id(
    args: &Args,
    session_id_for_repl: Option<String>,
) -> anyhow::Result<Option<String>> {
    if !(args.continue_session && !args.has_message_input()) {
        return Ok(session_id_for_repl);
    }
    let store = caliche_cli::storage::SessionStore::new().await?;
    if let Some(id) = store.get_last_session_id()? {
        Ok(Some(id))
    } else {
        eprintln!("No previous session to continue. Start a new conversation first.");
        caliche_cli::shutdown::exit(1);
    }
}

async fn run_repl_lane(
    args: Arc<Args>,
    mut service: caliche::AgentService,
    startup_ctx: bootstrap::StartupContext,
    session_id_for_repl: Option<String>,
    preloaded_config: Option<caliche_cli::config::Config>,
    tools_source: Option<String>,
    mut deferred_workspace: Option<caliche_cli::deferred::DeferredWorkspaceRag>,
    startup_notices: Vec<String>,
    sandbox_message: Option<String>,
    project_context_loaded: Option<String>,
    cancellation_token: caliche_cli::cancellation::CancellationToken,
) -> anyhow::Result<()> {
    let autosave_enabled = !args.no_autosave;
    let file_watcher_warning = service.start_file_watching().err().map(|e| {
        format!(
            "File watching could not be started: {e}.\n  Workspace search still works; only live re-indexing on file changes is disabled.\n  If this directory should be watchable, try running from the project root, check permissions, or avoid symlinked paths."
        )
    });
    if let Some(ref e) = file_watcher_warning {
        tracing::debug!("File watcher error details: {}", e);
    }
    let startup_messages: Vec<String> = [
        sandbox_message,
        project_context_loaded,
        file_watcher_warning,
    ]
    .into_iter()
    .flatten()
    .chain(startup_notices)
    .collect();

    let (service, thin) = Box::pin(caliche_cli::thin_client::bootstrap_thin_client(
        service,
        args.as_ref(),
        preloaded_config.as_ref(),
        startup_ctx.database.clone(),
    ))
    .await
    .map_err(|e| caliche_cli::thin_client::format_bootstrap_failure(&e))?;
    let thin_pair = Some((thin.client.clone(), thin.session_id.clone()));
    if let (Some(ref mut deferred), Some(shared)) =
        (deferred_workspace.as_mut(), thin.shared_service.clone())
    {
        deferred.set_server_service(shared);
    }

    let repl_result = Box::pin(caliche_cli::repl::run_repl(
        service,
        session_id_for_repl,
        args.save_session,
        autosave_enabled,
        startup_ctx.database.as_ref(),
        startup_ctx.cred_manager.as_ref(),
        Some(&cancellation_token),
        args.no_auto_skill || args.is_lean_profile(),
        preloaded_config,
        tools_source,
        args.debug,
        startup_ctx.log_path.as_deref(),
        &startup_messages,
        thin_pair,
        deferred_workspace,
    ))
    .await;
    thin.shutdown().await;
    repl_result
}

/// Build the clap command on an 8 MiB stack.
///
/// Dev builds (`cargo run`, opt-level 0) construct this CLI's clap `Command`
/// with more than 1 MiB of live stack. On Linux, `--help` overflows a 1280
/// KiB stack and succeeds at 1536 KiB. The Windows main thread is 1 MiB, so
/// `caliche.exe --help` dies with `STATUS_STACK_OVERFLOW` (`0xc00000fd`)
/// before usage is printed. A spawned thread can set its own stack; 8 MiB
/// matches the usual Linux main-thread size. `--help` and usage errors exit
/// the process from this thread via clap.
fn parse_args() -> Args {
    const PARSER_STACK_BYTES: usize = 8 * 1024 * 1024;
    let handle = match std::thread::Builder::new()
        .name("cli-args".to_string())
        .stack_size(PARSER_STACK_BYTES)
        .spawn(Args::parse)
    {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("failed to start argument parser: {err}");
            caliche_cli::shutdown::exit(1);
        }
    };
    match handle.join() {
        Ok(args) => args,
        Err(_) => caliche_cli::shutdown::exit(1),
    }
}

/// Worker count for the CLI runtime.
///
/// The default Tokio pool starts one thread per CPU. This host has many cores,
/// and those threads are created before `--help` work or the first prompt.
/// Four workers cover the local HTTP server and tool tasks.
fn cli_worker_threads() -> usize {
    const MIN_WORKERS: usize = 2;
    const MAX_WORKERS: usize = 4;
    match std::thread::available_parallelism() {
        Ok(count) => count.get().clamp(MIN_WORKERS, MAX_WORKERS),
        Err(_) => MIN_WORKERS,
    }
}

fn main() -> anyhow::Result<()> {
    bootstrap::configure_stdio_buffering();
    let args = Arc::new(parse_args());
    runtime::load_env_file_before_runtime(&args)?;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cli_worker_threads())
        .enable_all()
        .build()
        .context("failed to start the agent runtime")?;
    let result = runtime.block_on(caliche_cli::shutdown::run_with_task_ledger(
        caliche_cli::shutdown::task_ledger_dir(),
        Box::pin(run(args)),
    ));
    // Dropping the runtime would wait for every blocking task, such as a
    // workspace walk still listing a large tree. Recorded work has already
    // stopped with the task ledger above, so bound the wait.
    runtime.shutdown_timeout(RUNTIME_SHUTDOWN_GRACE);
    result
}

/// How long exit waits for blocking tasks that are still running.
const RUNTIME_SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

async fn run(args: Arc<Args>) -> anyhow::Result<()> {
    handlers::remember_cli_listen(&args.bind, args.port);
    handlers::remember_boot_provider(args.provider.as_deref(), args.model.is_some());
    let runtime::PreparedRuntime {
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
        saved_temperature,
        saved_max_tokens,
    } = match runtime::prepare(&args).await? {
        runtime::PrepareOutcome::Finished => return Ok(()),
        runtime::PrepareOutcome::Ready(prepared) => prepared,
    };

    let mut service = runtime::create_service_with_recovery(
        args.as_ref(),
        provider,
        model,
        tool_names,
        workspace_options,
        &startup_ctx,
        use_interactive,
        preloaded_config.as_ref(),
    )
    .await?;

    register_interactive_permission(args.as_ref(), &mut service);
    let sandbox_message = configure_sandbox(args.as_ref(), &mut service).await?;
    apply_agent_profile(args.as_ref(), agent_profile.as_ref(), &mut service)?;
    spawn_discovery(args.debug || args.verbose);
    let project_context_loaded =
        apply_project_context(args.as_ref(), preloaded_config.as_ref(), &mut service).await;
    apply_preloaded_config(args.as_ref(), preloaded_config.as_ref(), &mut service)?;
    runtime::apply_saved_sampling(
        &mut service,
        args.temperature,
        args.max_tokens,
        saved_temperature,
        saved_max_tokens,
    );
    print_provider_model_if_needed(args.as_ref(), use_interactive, &service);

    if maybe_list_tools(args.as_ref(), &service) {
        return Ok(());
    }
    if maybe_run_advanced_features(&args, &startup_ctx, &cancellation_token).await? {
        return Ok(());
    }
    let Some(mut service) = maybe_run_stdio_or_server(args.as_ref(), service, &startup_ctx).await?
    else {
        return Ok(());
    };
    if run_oneshot_message(
        args.as_ref(),
        &mut service,
        &cancellation_token,
        &startup_ctx,
    )
    .await?
    {
        return Ok(());
    }

    let session_id_for_repl =
        resolve_continue_session_id(args.as_ref(), session_id_for_repl).await?;
    Box::pin(run_repl_lane(
        args,
        service,
        startup_ctx,
        session_id_for_repl,
        preloaded_config,
        tools_source,
        deferred_workspace,
        startup_notices,
        sandbox_message,
        project_context_loaded,
        cancellation_token,
    ))
    .await
}
