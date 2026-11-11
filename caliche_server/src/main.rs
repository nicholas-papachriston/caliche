//! Standalone HTTP server for Caliche agents.
//!
//! It chooses the provider with the same rule as `caliche --server`
//! ([`caliche_server::boot_provider`]).

use caliche::agent::core::ToolExecutionDriver;
use caliche::agent::hooks::parse_permission_mode_str;
use caliche::agent::service::resolve_runtime_identity_with_discovery;
use caliche::harness::config::{HarnessConfig, PermissionModeConfig};
use caliche::llm::ProviderType;
use caliche_server::boot_provider::{
    PersistedBoot, ServerBootProvider, ServerBootProviderInput, apply_server_boot_provider,
    load_persisted_boot, select_server_boot_provider,
};
use caliche_server::{RouterOptions, create_router_with_shutdown};
use clap::Parser;
use std::net::SocketAddr;
use std::path::PathBuf;
use tower_http::cors::{Any, CorsLayer};

/// Caliche HTTP server
#[derive(Parser, Debug)]
#[command(name = "caliche_server")]
#[command(about = "HTTP server for Caliche agents")]
#[command(version)]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "3000", env = "PORT")]
    port: u16,

    /// Address to bind to. Unset `--bind` and `BIND_ADDRESS` listen on loopback.
    #[arg(short, long, default_value = "127.0.0.1", env = "BIND_ADDRESS")]
    bind: String,

    /// Model provider (openai, anthropic, bedrock, openrouter, perplexity, cohere, google, lmstudio, ollama, azure-openai, cloudflare-ai-gateway).
    /// When it is unset, the server uses the saved settings, then AGENT_PROVIDER or DEFAULT_PROVIDER, then openrouter.
    #[arg(long)]
    provider: Option<String>,

    /// Default model name
    #[arg(long, env = "DEFAULT_MODEL")]
    model: Option<String>,

    /// Override model context window size in tokens
    #[arg(long, env = "DEFAULT_CONTEXT_WINDOW")]
    context_window: Option<usize>,

    /// Comma-separated list of tools to enable for harness/session routes
    #[arg(long, env = "CALICHE_TOOLS")]
    tools: Option<String>,

    /// Maximum agent turns for harness/session routes
    #[arg(long, default_value = "50", env = "CALICHE_MAX_TURNS")]
    max_turns: usize,

    /// Working directory for harness/session routes
    #[arg(long, env = "CALICHE_WORKSPACE")]
    workspace: Option<PathBuf>,

    /// System prompt override for harness/session routes
    #[arg(long, env = "CALICHE_SYSTEM_PROMPT")]
    system_prompt: Option<String>,

    /// Permission mode (default, accept-edits, accept-all, deny-all, interactive)
    #[arg(long, env = "CALICHE_PERMISSION_MODE")]
    permission_mode: Option<String>,

    /// Tool execution driver override (auto, off, planner-first-phased, etc.)
    #[arg(long, env = "CALICHE_TOOL_EXECUTION_DRIVER")]
    tool_execution_driver: Option<String>,

    /// Enable CORS for all origins (development mode)
    #[arg(long, default_value = "false", env = "CORS_PERMISSIVE")]
    cors_permissive: bool,

    /// Load environment variables from this file. A variable already set in
    /// the shell keeps its value. Without this flag no `.env` file is read,
    /// not even one in the working directory.
    #[arg(long, value_name = "PATH")]
    env_file: Option<String>,
}

fn parse_permission_mode(value: Option<&str>) -> anyhow::Result<PermissionModeConfig> {
    parse_permission_mode_str(value).map_err(anyhow::Error::msg)
}

fn parse_tools(value: Option<&str>) -> Option<Vec<String>> {
    value
        .map(|tools| {
            tools
                .split(',')
                .map(str::trim)
                .filter(|tool| !tool.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|tools| !tools.is_empty())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    load_env_before_runtime(&args)?;
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run(args))
}

/// Load `--env-file`. This writes the process environment, so it runs before
/// the runtime starts its threads.
fn load_env_before_runtime(args: &Args) -> anyhow::Result<()> {
    caliche_server::env_file::load_before_runtime(args.env_file.as_deref())
}

async fn run(args: Args) -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(true)
        .init();

    // Initialise OpenTelemetry tracing. When the `otlp` feature is compiled in
    // and `OTEL_EXPORTER_OTLP_ENDPOINT` is set, spans are exported to the
    // collector via batch export; otherwise a no-op provider is installed.
    if let Err(err) = caliche::observability::init_telemetry("caliche_server", None) {
        tracing::warn!(
            error = %err,
            "Failed to initialise OpenTelemetry pipeline; continuing without tracing exporter"
        );
    }

    caliche_server::shutdown::run_with_task_ledger(
        caliche_server::shutdown::task_ledger_dir(),
        Box::pin(serve(args)),
    )
    .await
}

/// Provider and model to boot with, and the rule step that chose them.
struct BootSelection {
    choice: ServerBootProvider,
    provider: Option<ProviderType>,
    model: Option<String>,
}

/// Apply the `caliche --server` provider rule to the parsed flags.
///
/// `None` provider means the built-in default; `resolve_provider` turns it
/// into `openrouter`.
fn select_boot(
    args: &Args,
    agent_provider: Option<&str>,
    default_provider_env: Option<&str>,
    saved: &PersistedBoot,
) -> anyhow::Result<BootSelection> {
    let mut provider = args
        .provider
        .as_deref()
        .map(|raw| {
            raw.parse::<ProviderType>()
                .map_err(|_| anyhow::anyhow!("Invalid provider: {raw}"))
        })
        .transpose()?;
    let mut model = args.model.clone();
    let choice = select_server_boot_provider(ServerBootProviderInput {
        explicit_provider: args.provider.as_deref(),
        explicit_model: args.model.is_some(),
        agent_provider,
        default_provider_env,
        last_provider: saved.provider.as_deref(),
        last_model: saved.model.as_deref(),
    });
    apply_server_boot_provider(
        &choice,
        saved.provider.as_deref(),
        saved.model.as_deref(),
        &mut provider,
        &mut model,
    )?;
    Ok(BootSelection {
        choice,
        provider,
        model,
    })
}

/// [`select_boot`] with `AGENT_PROVIDER` and `DEFAULT_PROVIDER` read from the
/// process environment.
fn select_boot_from_env(args: &Args, saved: &PersistedBoot) -> anyhow::Result<BootSelection> {
    let agent_provider = std::env::var("AGENT_PROVIDER").ok();
    let default_provider_env = std::env::var("DEFAULT_PROVIDER").ok();
    select_boot(
        args,
        agent_provider.as_deref(),
        default_provider_env.as_deref(),
        saved,
    )
}

async fn serve(args: Args) -> anyhow::Result<()> {
    let saved = load_persisted_boot(None, None).await;
    let selection = select_boot_from_env(&args, &saved)?;
    tracing::info!(
        provider = %selection.choice.provider,
        source = selection.choice.source,
        "boot provider selected"
    );
    let identity =
        resolve_runtime_identity_with_discovery(selection.provider, selection.model, None).await?;
    let default_provider = identity.provider;
    let default_model = identity.model;

    tracing::info!(
        "Starting caliche API server with provider: {:?}, model: {}",
        default_provider,
        default_model
    );

    let mut harness_config = HarnessConfig::default()
        .with_provider(default_provider)
        .with_model(default_model.clone())
        .with_max_turns(args.max_turns);
    harness_config.tools = parse_tools(args.tools.as_deref());
    harness_config.workspace_root = args
        .workspace
        .clone()
        .or_else(|| std::env::current_dir().ok());
    harness_config.system_prompt = args.system_prompt.clone();
    harness_config.permission_mode = parse_permission_mode(args.permission_mode.as_deref())?;
    harness_config.tool_execution_driver = args
        .tool_execution_driver
        .as_deref()
        .filter(|value| !value.trim().eq_ignore_ascii_case("auto"))
        .and_then(ToolExecutionDriver::parse);

    // Create the API router + endpoint shutdown handle
    let (router, endpoint_manager) = create_router_with_shutdown(
        default_provider,
        default_model,
        Some(RouterOptions {
            harness_config: Some(harness_config),
            context_window: args.context_window,
            ..RouterOptions::default()
        }),
    )
    .await?;

    // Apply CORS middleware
    let router = if caliche_server::security::allow_cors_permissive(args.cors_permissive)? {
        tracing::warn!("CORS permissive mode enabled - allowing all origins");
        router.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
    } else {
        router
    };

    // Parse bind address. Clap defaults this to 127.0.0.1 when BIND_ADDRESS is unset.
    caliche_server::security::warn_if_open_unspecified_bind(&args.bind);
    let addr = caliche_server::security::listen_socket_addr(&args.bind, args.port)?;

    tracing::info!("Listening on http://{}", addr);

    // Create and run the server with graceful shutdown.
    // SIGINT and SIGTERM share one drain so Ctrl-C does not wait on the
    // endpoint poll while SIGTERM kills the process.
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let signal_tx = shutdown_tx.clone();
    tokio::spawn(async move {
        let signal = caliche_server::shutdown::recv_process_signal().await;
        tracing::info!(
            signal = signal.as_str(),
            "shutdown signal received; draining"
        );
        let _ = signal_tx.send(true);
    });
    let endpoints = endpoint_manager.clone();
    let endpoint_rx = shutdown_rx.clone();
    tokio::spawn(async move {
        caliche_server::shutdown::wait_for_shutdown(endpoint_rx).await;
        let _ = tokio::time::timeout(
            caliche_server::shutdown::SERVER_SHUTDOWN_DRAIN,
            endpoints.stop_all(),
        )
        .await;
    });
    let server = axum::serve(
        listener,
        router.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(caliche_server::shutdown::wait_for_shutdown(
        shutdown_rx.clone(),
    ))
    .into_future();
    match caliche_server::shutdown::select_server_or_drain(
        server,
        shutdown_rx,
        caliche_server::shutdown::SERVER_SHUTDOWN_DRAIN,
    )
    .await
    {
        caliche_server::shutdown::ServeStop::Finished(result) => {
            result?;
        }
        caliche_server::shutdown::ServeStop::DrainElapsed => {}
    }

    Ok(())
}

#[cfg(test)]
mod provider_selection_tests {
    use super::{Args, PersistedBoot, select_boot};
    use clap::Parser;

    #[test]
    fn no_flag_settings_or_env_selects_openrouter() {
        let args = Args::parse_from(["caliche_server"]);
        let selection = select_boot(&args, None, None, &PersistedBoot::default()).expect("select");
        assert_eq!(selection.choice.provider, "openrouter");
        assert_eq!(selection.choice.source, "default");
        assert_eq!(selection.provider, None);
    }

    #[test]
    fn saved_settings_beat_env_and_bring_their_model() {
        let args = Args::parse_from(["caliche_server"]);
        let saved = PersistedBoot {
            provider: Some("google".to_owned()),
            model: Some("gemini-2.5-flash".to_owned()),
            ..PersistedBoot::default()
        };
        let selection = select_boot(&args, Some("openai"), None, &saved).expect("select");
        assert_eq!(selection.choice.provider, "google");
        assert_eq!(selection.choice.source, "last");
        assert_eq!(selection.model.as_deref(), Some("gemini-2.5-flash"));
    }

    #[test]
    fn flag_beats_saved_settings_and_env() {
        let args = Args::parse_from(["caliche_server", "--provider", "anthropic"]);
        let saved = PersistedBoot {
            provider: Some("google".to_owned()),
            model: Some("gemini-2.5-flash".to_owned()),
            ..PersistedBoot::default()
        };
        let selection = select_boot(&args, Some("openai"), None, &saved).expect("select");
        assert_eq!(selection.choice.provider, "anthropic");
        assert_eq!(selection.choice.source, "flag");
        assert_eq!(selection.model, None);
    }

    #[test]
    fn default_provider_env_applies_without_flag_or_settings() {
        let args = Args::parse_from(["caliche_server"]);
        let selection =
            select_boot(&args, None, Some("ollama"), &PersistedBoot::default()).expect("select");
        assert_eq!(selection.choice.provider, "ollama");
        assert_eq!(selection.choice.source, "env");
    }
}

#[cfg(test)]
mod bind_default_tests {
    use super::Args;
    use clap::{CommandFactory, Parser};

    #[test]
    fn omitted_bind_defaults_to_loopback() {
        let command = Args::command();
        let mut bind_default = String::new();
        let mut port_default = String::new();
        for arg in command.get_arguments() {
            if arg.get_id() == "bind" {
                if let Some(value) = arg.get_default_values().first() {
                    bind_default = value.to_string_lossy().into_owned();
                }
            }
            if arg.get_id() == "port" {
                if let Some(value) = arg.get_default_values().first() {
                    port_default = value.to_string_lossy().into_owned();
                }
            }
        }
        assert_eq!(bind_default, "127.0.0.1");
        assert_eq!(port_default, "3000");
    }

    #[test]
    fn explicit_unspecified_bind_overrides_the_default() {
        let args = Args::parse_from(["caliche_server", "--bind", "0.0.0.0", "--port", "3000"]);
        assert_eq!(args.bind, "0.0.0.0");
        assert_eq!(args.port, 3000);
    }

    #[test]
    fn bare_ipv6_unspecified_bind_parses_as_socket_addr() {
        let args = Args::parse_from(["caliche_server", "--bind", "::", "--port", "3000"]);
        assert_eq!(args.bind, "::");
        let addr = caliche_server::security::listen_socket_addr(&args.bind, args.port)
            .expect("bare :: must parse");
        assert_eq!(
            addr,
            std::net::SocketAddr::from((std::net::Ipv6Addr::UNSPECIFIED, 3000))
        );
    }
}

#[cfg(test)]
mod env_file_tests {
    use super::{Args, PersistedBoot, load_env_before_runtime, select_boot_from_env};
    use clap::Parser;

    const NAMES: [&str; 2] = ["DEFAULT_PROVIDER", "OLLAMA_BASE_URL"];
    const FILE: &str = "DEFAULT_PROVIDER=ollama\nOLLAMA_BASE_URL=http://file.invalid:11434\n";

    /// Clears [`NAMES`] and `AGENT_PROVIDER` through the guard, which puts
    /// back their values (also the ones the env file loader sets) on drop,
    /// and puts back the working directory.
    struct EnvScope {
        env: caliche_test_env::EnvGuard,
        cwd: std::path::PathBuf,
    }

    impl EnvScope {
        fn new(mut env: caliche_test_env::EnvGuard) -> Self {
            for name in NAMES.into_iter().chain(["AGENT_PROVIDER"]) {
                env.remove(name);
            }
            Self {
                env,
                cwd: std::env::current_dir().expect("cwd"),
            }
        }
    }

    impl Drop for EnvScope {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.cwd);
        }
    }

    fn resolved_provider(args: &Args) -> String {
        select_boot_from_env(args, &PersistedBoot::default())
            .expect("select")
            .choice
            .provider
    }

    #[test]
    #[serial_test::serial(env)]
    fn dotenv_in_the_working_directory_is_ignored() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial(env), like every test in this binary
        // that touches the environment, and it starts no thread.
        let _scope = EnvScope::new(unsafe { caliche_test_env::EnvGuard::acquire() });
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(".env"), FILE).expect("write .env");
        std::env::set_current_dir(dir.path()).expect("chdir");
        let args = Args::parse_from(["caliche_server"]);

        load_env_before_runtime(&args).expect("load");

        assert_eq!(resolved_provider(&args), "openrouter");
        assert_eq!(std::env::var_os("OLLAMA_BASE_URL"), None);
    }

    #[test]
    #[serial_test::serial(env)]
    fn env_file_flag_is_applied() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial(env), like every test in this binary
        // that touches the environment, and it starts no thread.
        let _scope = EnvScope::new(unsafe { caliche_test_env::EnvGuard::acquire() });
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("server.env");
        std::fs::write(&path, FILE).expect("write env file");
        let path = path.to_string_lossy().into_owned();
        let args = Args::parse_from(["caliche_server", "--env-file", path.as_str()]);

        load_env_before_runtime(&args).expect("load");

        assert_eq!(resolved_provider(&args), "ollama");
        assert_eq!(
            std::env::var("OLLAMA_BASE_URL").as_deref(),
            Ok("http://file.invalid:11434")
        );
    }

    #[test]
    #[serial_test::serial(env)]
    fn shell_value_beats_the_env_file() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial(env), like every test in this binary
        // that touches the environment, and it starts no thread.
        let mut scope = EnvScope::new(unsafe { caliche_test_env::EnvGuard::acquire() });
        scope.env.set("DEFAULT_PROVIDER", "anthropic");
        scope
            .env
            .set("OLLAMA_BASE_URL", "http://shell.invalid:11434");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("server.env");
        std::fs::write(&path, FILE).expect("write env file");
        let path = path.to_string_lossy().into_owned();
        let args = Args::parse_from(["caliche_server", "--env-file", path.as_str()]);

        load_env_before_runtime(&args).expect("load");

        assert_eq!(resolved_provider(&args), "anthropic");
        assert_eq!(
            std::env::var("OLLAMA_BASE_URL").as_deref(),
            Ok("http://shell.invalid:11434")
        );
    }

    #[test]
    fn missing_env_file_is_an_error() {
        let args = Args::parse_from(["caliche_server", "--env-file", "/nonexistent/server.env"]);
        let error = load_env_before_runtime(&args).expect_err("missing file");
        assert!(
            error
                .to_string()
                .starts_with("Failed to load --env-file '/nonexistent/server.env'"),
            "{error}"
        );
    }
}
