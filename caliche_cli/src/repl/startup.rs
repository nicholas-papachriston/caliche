//! REPL startup: session load, memory, banner, and persisted settings.

use super::{ReedlineInput, ReplRenderer, TerminalOutput};
use crate::core::state::InteractiveState;
use crate::input::CompletionData;
use crate::storage::SessionStore;
use anyhow::Result;
use caliche::AgentService;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

pub(super) struct ReplSessionBundle {
    pub session_store: SessionStore,
    pub cli_session: crate::session::CliSession,
    pub config: Option<crate::config::Config>,
}

pub(super) async fn open_or_create_session(
    shared_database: Option<&crate::storage::Database>,
    session_id: &str,
) -> Result<(SessionStore, crate::session::CliSession)> {
    let store = if let Some(db) = shared_database {
        SessionStore::from_pool(db.pool()).await
    } else {
        SessionStore::new().await
    };
    let store = store?;
    let sess = match store.try_load_async(session_id).await {
        Ok(Some(s)) => s,
        Ok(None) => {
            let wd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            store.create(session_id.to_string(), wd).await?
        }
        Err(e) => {
            anyhow::bail!(
                "Failed to load session {}: {}. Refusing to start a fresh session. \
                 Use `--session <new-id>` or resolve the storage issue.",
                session_id,
                e
            );
        }
    };
    Ok((store, sess))
}

pub(super) async fn load_repl_session_and_config(
    session_id: &str,
    shared_database: Option<&crate::storage::Database>,
    preloaded_config: Option<crate::config::Config>,
) -> Result<ReplSessionBundle> {
    if let Some(cfg) = preloaded_config {
        let (session_store, cli_session) =
            open_or_create_session(shared_database, session_id).await?;
        return Ok(ReplSessionBundle {
            session_store,
            cli_session,
            config: Some(cfg),
        });
    }
    let config_fut = crate::config::load_standard_config();
    let store_fut = open_or_create_session(shared_database, session_id);
    let (session_result, config_result) = tokio::join!(store_fut, config_fut);
    let (session_store, cli_session) = session_result?;
    Ok(ReplSessionBundle {
        session_store,
        cli_session,
        config: config_result,
    })
}

pub(super) async fn install_repl_memory(
    service: &mut AgentService,
    config: Option<&crate::config::Config>,
) {
    let strategy_str = config
        .and_then(|c| c.memory.as_ref().and_then(|m| m.strategy.as_deref()))
        .unwrap_or("composite");
    let strategy_type = strategy_str
        .parse()
        .unwrap_or(crate::core::memory_manager::MemoryStrategyType::Composite);
    let max_tokens = config.and_then(|c| c.memory_max_tokens).unwrap_or(16000);
    if let Ok(memory) = crate::core::memory_manager::create_memory_strategy(
        strategy_type,
        max_tokens,
        service.llm_provider(),
        vec![],
        service.embedding_provider(),
    )
    .await
    {
        service.set_memory_strategy(Some(memory));
        return;
    }
    let token_counter: Arc<dyn caliche::agent::memory::TokenCounter> =
        Arc::new(caliche::agent::memory::token_counter::ApproximateTokenCounter::default());
    let memory_config =
        caliche::agent::memory::BufferMemoryConfig::default().with_max_tokens(max_tokens);
    let memory = caliche::agent::memory::BufferMemory::new(memory_config, token_counter);
    service.set_memory_strategy(Some(Box::new(memory)));
}

fn nonempty_setting(value: Option<String>) -> Option<String> {
    value.filter(|s| !s.is_empty())
}

pub(super) async fn apply_shared_database_settings(
    state: &mut InteractiveState,
    shared_database: Option<&crate::storage::Database>,
) {
    let Some(db) = shared_database else {
        return;
    };
    state.subagent_provider =
        nonempty_setting(db.load_setting("subagent_provider").await.ok().flatten());
    state.subagent_model = nonempty_setting(db.load_setting("subagent_model").await.ok().flatten());
    state.embedding_provider =
        nonempty_setting(db.load_setting("embedding_provider").await.ok().flatten());
    state.embedding_model =
        nonempty_setting(db.load_setting("embedding_model").await.ok().flatten());
    state.image_generation_provider = nonempty_setting(
        db.load_setting("image_generation_provider")
            .await
            .ok()
            .flatten(),
    );
    state.image_generation_model = nonempty_setting(
        db.load_setting("image_generation_model")
            .await
            .ok()
            .flatten(),
    );
    state.artifact_dir =
        nonempty_setting(db.load_setting("artifact_dir").await.ok().flatten()).map(PathBuf::from);
}

pub(super) fn apply_config_to_repl_state(
    state: &mut InteractiveState,
    config: Option<&crate::config::Config>,
) {
    if let Some(level) = config.and_then(|cfg| cfg.output_verbosity.as_deref()) {
        match level.parse() {
            Ok(verbosity) => state.output_verbosity = verbosity,
            Err(error) => tracing::warn!(
                error = %error,
                "ignoring invalid output_verbosity in config"
            ),
        }
    }
    let Some(c) = config else {
        return;
    };
    if let Some(ref memory) = c.memory {
        if let Some(auto_compact) = memory.auto_compact {
            state.auto_compact_enabled = auto_compact;
        }
        if let Some(threshold) = memory.auto_compact_threshold {
            state.auto_compact_threshold = (threshold * 100.0).clamp(50.0, 99.0);
        }
    }
    if let Some(ref subagent) = c.subagent {
        if let Some(ref provider) = subagent.provider {
            state.subagent_provider = Some(provider.clone());
        }
        if let Some(ref model) = subagent.model {
            state.subagent_model = Some(model.clone());
        }
    }
    let Some(ref context) = c.context else {
        return;
    };
    let Some(ref paths) = context.paths else {
        return;
    };
    for path_str in paths {
        let path = PathBuf::from(path_str);
        if path.exists() && !state.tracked_files.contains(&path) {
            state.tracked_files.push(path);
        }
    }
}

/// How much of the REPL opening banner to print.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ReplBannerMode {
    /// Warnings only.
    Off,
    /// Product name, provider, model, and one hint line.
    #[default]
    Compact,
    /// Compact lines plus tools, session id, logs, and startup notes.
    Full,
}

impl ReplBannerMode {
    fn from_config_value(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "compact" => Some(Self::Compact),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
}

pub(super) fn repl_banner_mode(config: Option<&crate::config::Config>) -> ReplBannerMode {
    let Some(value) = config.and_then(|cfg| cfg.banner.as_deref()) else {
        return ReplBannerMode::Compact;
    };
    match ReplBannerMode::from_config_value(value) {
        Some(mode) => mode,
        None => {
            tracing::warn!(banner = value, "ignoring unknown banner mode");
            ReplBannerMode::Compact
        }
    }
}

const BANNER_HINT: &str = "\x1b[90m/help  ·  Tab  ·  Ctrl+C  ·  \\ multiline\x1b[0m";

pub(super) struct ReplBannerArgs<'a> {
    pub provider: &'a str,
    pub model: &'a str,
    pub session_id: &'a str,
    pub tools_source: Option<&'a str>,
    pub startup_log_path: Option<&'a Path>,
    pub startup_messages: &'a [String],
    pub debug: bool,
    pub startup_start: Instant,
    pub has_workspace_rag: bool,
    pub deferred_workspace: bool,
    pub tool_list: Vec<(String, String, serde_json::Value)>,
    pub mode: ReplBannerMode,
}

fn write_tool_line(
    out: &mut impl Write,
    tool_list: &[(String, String, serde_json::Value)],
    tools_source: Option<&str>,
) {
    let tool_count = tool_list.len();
    let source_suffix = tools_source
        .map(|s| format!(" ({s})"))
        .unwrap_or_else(|| " (all defaults)".to_string());
    if tool_count <= 6 {
        let names: Vec<&str> = tool_list.iter().map(|(n, _, _)| n.as_str()).collect();
        let _ = writeln!(
            out,
            "\x1b[90mTools:\x1b[0m {}{}",
            names.join(", "),
            source_suffix
        );
        return;
    }
    let _ = writeln!(
        out,
        "\x1b[90mTools:\x1b[0m {} enabled{}. Use /tools to list.",
        tool_count, source_suffix
    );
}

fn write_workspace_status(out: &mut impl Write, has_workspace_rag: bool, deferred_workspace: bool) {
    if has_workspace_rag {
        let _ = writeln!(out, "\x1b[90mIndexing workspace in background.\x1b[0m");
    } else if deferred_workspace {
        let _ = writeln!(
            out,
            "\x1b[90mWorkspace search starting in the background.\x1b[0m"
        );
    }
}

fn write_startup_duration(out: &mut impl Write, startup_start: Instant) {
    let elapsed = startup_start.elapsed();
    let secs = elapsed.as_secs_f64();
    let startup_msg = if secs < 1.0 {
        format!("{:.2} ms", secs * 1000.0)
    } else {
        format!("{:.3} s", secs)
    };
    let _ = writeln!(out, "\x1b[90mStartup: {startup_msg}\x1b[0m");
}

fn write_startup_warnings(out: &mut impl Write, messages: &[String]) {
    for msg in messages {
        let _ = writeln!(out, "{msg}");
    }
}

fn write_provider_line(out: &mut impl Write, provider: &str, model: &str) {
    let _ = writeln!(
        out,
        "\x1b[90mProvider:\x1b[0m {provider}  \x1b[90mModel:\x1b[0m {model}"
    );
}

fn write_hint(out: &mut impl Write) {
    let _ = writeln!(out, "{BANNER_HINT}");
}

fn write_off_banner(out: &mut impl Write, messages: &[String]) {
    if messages.is_empty() {
        return;
    }
    let _ = out.write_all(b"\n");
    write_startup_warnings(out, messages);
    let _ = out.flush();
}

fn write_compact_banner(out: &mut impl Write, args: &ReplBannerArgs<'_>) {
    let _ = out.write_all(b"\n");
    write_startup_warnings(out, args.startup_messages);
    let _ = writeln!(out, "Caliche");
    write_provider_line(out, args.provider, args.model);
    write_hint(out);
    let _ = writeln!(out);
    let _ = out.flush();
}

fn write_full_banner(out: &mut impl Write, args: &ReplBannerArgs<'_>) {
    let _ = out.write_all(b"\n");
    write_startup_warnings(out, args.startup_messages);
    if let Some(path) = args.startup_log_path {
        let _ = writeln!(out, "Logs written to: {}", path.display());
    }
    let _ = writeln!(out, "Caliche");
    write_provider_line(out, args.provider, args.model);
    write_tool_line(out, &args.tool_list, args.tools_source);
    let _ = writeln!(out, "\x1b[90mSession:\x1b[0m {}", args.session_id);
    write_workspace_status(out, args.has_workspace_rag, args.deferred_workspace);
    if args.debug {
        write_startup_duration(out, args.startup_start);
    }
    write_hint(out);
    let _ = writeln!(out);
    let _ = out.flush();
}

fn write_startup_banner_to(out: &mut impl Write, args: &ReplBannerArgs<'_>) {
    match args.mode {
        ReplBannerMode::Off => write_off_banner(out, args.startup_messages),
        ReplBannerMode::Compact => write_compact_banner(out, args),
        ReplBannerMode::Full => write_full_banner(out, args),
    }
}

pub(super) fn print_startup_banner(args: ReplBannerArgs<'_>) {
    crate::repl::stdout_sync::with_stdout(|out| write_startup_banner_to(out, &args));
}

pub(super) fn spawn_workspace_index_wait(service: &AgentService) {
    if !service.has_workspace_rag() {
        return;
    }
    let Some(handle) = service.workspace_indexing_handle() else {
        return;
    };
    tokio::spawn(async move {
        use std::time::Duration;
        handle
            .wait_once_with_timeout(Duration::from_secs(300))
            .await;
        if handle.is_ready() {
            tracing::debug!("Workspace indexed");
        }
    });
}

pub(super) fn register_path_restriction_hook(
    service: &mut AgentService,
) -> caliche::agent::hooks::AllowedPaths {
    use caliche::agent::hooks::{HookMatcher, PathRestrictionConfig, PathRestrictionHook};
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let (hook, extra_allowed) =
        PathRestrictionHook::new(workspace_root, PathRestrictionConfig::default());
    service
        .agent_mut()
        .register_hook(HookMatcher::All, Arc::new(hook));
    tracing::debug!("Path restriction hook registered");
    extra_allowed
}

pub(super) async fn persist_last_provider_model(
    service: &AgentService,
    shared_database: Option<&crate::storage::Database>,
) {
    crate::boot_provider::save_last_selection(
        shared_database,
        &service.provider().to_string(),
        service.model(),
        service.agent().temperature(),
        service.agent().max_tokens(),
    )
    .await;
}

pub(super) fn configured_providers_from_manager(
    configured: Vec<crate::credentials::ProviderStatus>,
) -> Vec<String> {
    configured
        .into_iter()
        .filter(|p| p.is_configured)
        .map(|p| p.name)
        .collect()
}

pub(super) fn new_repl_input(
    completion_data: Arc<std::sync::RwLock<CompletionData>>,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
) -> Result<(ReedlineInput, ReplRenderer)> {
    let mut input = ReedlineInput::with_completion_data(completion_data, false)?;
    input.set_interrupt_token(cancellation_token.cloned());
    input.spawn_history_load();
    let output = TerminalOutput::with_external_printer(input.external_printer());
    let renderer = ReplRenderer::new(Box::new(output));
    Ok((input, renderer))
}

#[cfg(test)]
mod banner_tests {
    use super::{ReplBannerArgs, ReplBannerMode, write_startup_banner_to};
    use std::time::Instant;

    fn sample_args<'a>(mode: ReplBannerMode, messages: &'a [String]) -> ReplBannerArgs<'a> {
        ReplBannerArgs {
            provider: "openrouter",
            model: "test-model",
            session_id: "sess-1",
            tools_source: None,
            startup_log_path: None,
            startup_messages: messages,
            debug: false,
            startup_start: Instant::now(),
            has_workspace_rag: false,
            deferred_workspace: false,
            tool_list: vec![(
                "read".to_string(),
                "Read a file".to_string(),
                serde_json::json!({}),
            )],
            mode,
        }
    }

    fn render(mode: ReplBannerMode, messages: &[String]) -> String {
        let mut buf = Vec::new();
        let args = sample_args(mode, messages);
        write_startup_banner_to(&mut buf, &args);
        String::from_utf8(buf).unwrap_or_default()
    }

    #[test]
    fn compact_banner_hides_tools_session_and_the_long_hint() {
        let text = render(ReplBannerMode::Compact, &[]);
        assert!(text.contains("Caliche"));
        assert!(text.contains("openrouter"));
        assert!(text.contains("/help"));
        assert!(!text.contains("Tools:"));
        assert!(!text.contains("Session:"));
        assert!(!text.contains("/help all"));
        assert!(!text.contains("/doctor"));
    }

    #[test]
    fn full_banner_keeps_tools_and_session() {
        let text = render(ReplBannerMode::Full, &[]);
        assert!(text.contains("Tools:"));
        assert!(text.contains("read"));
        assert!(text.contains("Session:"));
        assert!(text.contains("sess-1"));
        assert!(!text.contains("/help all"));
    }

    #[test]
    fn off_banner_prints_warnings_only() {
        let messages = vec!["File watching could not be started.".to_string()];
        let text = render(ReplBannerMode::Off, &messages);
        assert!(text.contains("File watching"));
        assert!(!text.contains("Caliche"));
    }

    #[test]
    fn off_banner_with_no_warnings_is_empty() {
        let text = render(ReplBannerMode::Off, &[]);
        assert!(text.is_empty());
    }

    #[test]
    fn banner_mode_accepts_known_values_only() {
        assert_eq!(
            ReplBannerMode::from_config_value(" FULL "),
            Some(ReplBannerMode::Full)
        );
        assert_eq!(
            ReplBannerMode::from_config_value("off"),
            Some(ReplBannerMode::Off)
        );
        assert!(ReplBannerMode::from_config_value("loud").is_none());
    }
}
