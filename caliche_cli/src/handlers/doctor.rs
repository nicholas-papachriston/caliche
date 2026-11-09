//! Doctor command handler for system diagnostics.

use super::doctor_readiness::{self, DEFAULT_BIND, DEFAULT_PORT};
use crate::credentials::offered_providers;
use crate::errors::{CliError, CliResult};
use crate::storage::{SecureCredentialManager, SessionStore};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;
use std::io::{IsTerminal, Write};

/// CLI `--bind` / `--port` that `caliche --server` would listen on.
///
/// Defaults match the server flags (`127.0.0.1` and `3000`). `API_HOST` is not
/// consulted. The default only sets the socket address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoctorArgs {
    pub(crate) bind: String,
    pub(crate) port: u16,
    /// `--provider`, when the process was started with that flag.
    pub(crate) explicit_provider: Option<String>,
    /// True when `--model` was set. That skips the saved provider.
    pub(crate) explicit_model: bool,
}

impl DoctorArgs {
    /// Defaults used when `--bind` and `--port` are omitted.
    #[must_use]
    pub fn server_defaults() -> Self {
        Self::from_cli(DEFAULT_BIND, DEFAULT_PORT)
    }

    /// Listen pair from parsed CLI flags, with no provider override.
    #[must_use]
    pub fn from_cli(bind: &str, port: u16) -> Self {
        Self::from_cli_boot(bind, port, None, false)
    }

    /// Listen pair plus the `--provider` / `--model` flags `--server` would see.
    #[must_use]
    pub fn from_cli_boot(
        bind: &str,
        port: u16,
        explicit_provider: Option<&str>,
        explicit_model: bool,
    ) -> Self {
        Self {
            bind: bind.to_owned(),
            port,
            explicit_provider: explicit_provider
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            explicit_model,
        }
    }

    /// Flags remembered at process start, otherwise [`Self::server_defaults`].
    #[must_use]
    pub fn from_remembered() -> Self {
        let (provider, explicit_model) = doctor_readiness::remembered_boot_provider();
        doctor_readiness::remembered_bind().map_or_else(
            || Self::from_cli_boot(DEFAULT_BIND, DEFAULT_PORT, provider, explicit_model),
            |(bind, port)| Self::from_cli_boot(bind, port, provider, explicit_model),
        )
    }
}

/// Remember parsed `--bind` / `--port` for REPL and HTTP doctor reports.
pub fn remember_cli_listen(bind: &str, port: u16) {
    doctor_readiness::remember_cli_listen(bind, port);
}

/// Remember `--provider` / `--model` so a later doctor report matches `--server` boot.
pub fn remember_boot_provider(explicit_provider: Option<&str>, explicit_model: bool) {
    doctor_readiness::remember_boot_provider(explicit_provider, explicit_model);
}

/// Handler for doctor command.
pub struct DoctorHandler;

#[async_trait]
impl CommandHandler for DoctorHandler {
    type Args = DoctorArgs;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        handle_doctor_command(args)
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "doctor"
    }

    fn description(&self) -> &'static str {
        "Check system configuration and diagnose issues"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

fn write_path_status<W: Write>(w: &mut W, label: &str, path: &std::path::Path) -> Result<()> {
    write!(w, "{label}: ")?;
    if path.exists() {
        writeln!(w, "{} [OK]", path.display())?;
    } else {
        writeln!(w, "{} [NOT CREATED]", path.display())?;
    }
    Ok(())
}

async fn write_session_db_status<W: Write>(w: &mut W) -> Result<()> {
    write!(w, "Session Database: ")?;
    let session_store = SessionStore::new().await.ok();
    if let Some(store) = session_store {
        let count = store.list_async().await.map(|s| s.len()).unwrap_or(0);
        writeln!(w, "~/.caliche/caliche.db [OK, {} sessions]", count)?;
    } else {
        writeln!(w, "~/.caliche/caliche.db [ERROR]")?;
    }
    Ok(())
}

fn provider_env_set(env_var: &str) -> bool {
    !env_var.is_empty() && caliche_contracts::credential_env::var(env_var).is_ok()
}

async fn provider_config_set(manager: Option<&SecureCredentialManager>, provider: &str) -> bool {
    match manager {
        Some(manager) => manager.get_api_key(provider).await.is_some(),
        None => false,
    }
}

fn provider_source_label(provider: &str, env_set: bool, config_set: bool) -> &'static str {
    if env_set {
        return "[ENV] ";
    }
    if config_set {
        return "[CONFIG] ";
    }
    if provider == "bedrock" {
        if caliche_contracts::credential_env::var("AWS_ACCESS_KEY_ID").is_ok() {
            return "[AWS ENV] ";
        }
        return "[NOT SET] ";
    }
    if matches!(provider, "lmstudio" | "ollama" | "local-inference") {
        return "[LOCAL] ";
    }
    "[NOT SET] "
}

async fn write_provider_status<W: Write>(
    w: &mut W,
    manager: Option<&SecureCredentialManager>,
) -> Result<()> {
    writeln!(w)?;
    writeln!(w, "=== Provider Status ===\n")?;

    for (provider, env_var, _desc) in offered_providers() {
        write!(w, "{:<12} ", provider)?;
        let env_set = provider_env_set(env_var);
        let config_set = if env_set {
            false
        } else {
            provider_config_set(manager, provider).await
        };
        write!(
            w,
            "{}",
            provider_source_label(provider, env_set, config_set)
        )?;
        writeln!(w)?;
    }
    Ok(())
}

async fn readiness_for_listen(listen: &DoctorArgs) -> Result<doctor_readiness::ReadinessLines> {
    let (last_provider, last_model) = if listen.explicit_model {
        (None, None)
    } else {
        persisted_last_provider_model().await
    };
    let mut readiness = doctor_readiness::ReadinessLines::from_process(&listen.bind, listen.port);
    let agent_provider = caliche_contracts::credential_env::var("AGENT_PROVIDER").ok();
    let default_provider_env = caliche_contracts::credential_env::var("DEFAULT_PROVIDER").ok();
    let choice = crate::boot_provider::select_server_boot_provider(
        crate::boot_provider::ServerBootProviderInput {
            explicit_provider: listen.explicit_provider.as_deref(),
            explicit_model: listen.explicit_model,
            agent_provider: agent_provider.as_deref(),
            default_provider_env: default_provider_env.as_deref(),
            last_provider: last_provider.as_deref(),
            last_model: last_model.as_deref(),
        },
    );
    if choice
        .provider
        .parse::<caliche::llm::ProviderType>()
        .is_err()
    {
        anyhow::bail!("Invalid provider in boot selection: {}", choice.provider);
    }
    readiness.default_provider = choice.provider;
    choice
        .source
        .clone_into(&mut readiness.default_provider_source);
    Ok(readiness)
}

async fn persisted_last_provider_model() -> (Option<String>, Option<String>) {
    let (sqlite_provider, sqlite_model) = match crate::storage::Database::open().await {
        Ok(db) => {
            let provider = db.load_setting("last_provider").await.ok().flatten();
            let model = db.load_setting("last_model").await.ok().flatten();
            (provider, model)
        }
        Err(_) => (None, None),
    };
    let boot = crate::boot_provider::load_persisted_boot(sqlite_provider, sqlite_model).await;
    (boot.provider, boot.model)
}

fn write_default_provider<W: Write>(w: &mut W, provider: &str, source: &str) -> Result<()> {
    writeln!(w)?;
    writeln!(w, "Default Provider: {provider} ({source})")?;
    Ok(())
}

fn write_tool_availability<W: Write>(w: &mut W) -> Result<()> {
    writeln!(w)?;
    writeln!(w, "=== Tool Availability ===\n")?;

    let tools_to_check = [
        ("git", "Git version control"),
        ("rg", "ripgrep (fast search)"),
        ("grep", "grep (fallback search)"),
        ("curl", "HTTP requests"),
        ("python3", "Python 3"),
    ];

    for (tool, desc) in tools_to_check {
        write!(w, "{:<12} {:<25} ", tool, desc)?;
        if which::which(tool).is_ok() {
            writeln!(w, "[FOUND]")?;
        } else {
            writeln!(w, "[NOT FOUND]")?;
        }
    }
    Ok(())
}

fn write_system_info<W: Write>(w: &mut W) -> Result<()> {
    writeln!(w)?;
    writeln!(w, "=== System ===\n")?;
    writeln!(w, "OS:     {}", std::env::consts::OS)?;
    writeln!(w, "Arch:   {}", std::env::consts::ARCH)?;

    write!(w, "TTY:    ")?;
    if std::io::stdout().is_terminal() {
        writeln!(w, "yes")?;
    } else {
        writeln!(w, "no (piped)")?;
    }
    Ok(())
}

#[cfg(feature = "local-inference-base")]
fn write_host_capabilities<W: Write>(
    w: &mut W,
    host: &caliche::llm::HostCapabilities,
) -> Result<()> {
    writeln!(
        w,
        "RAM:    {} available / {} total",
        crate::core::slash_commands::helpers::format_bytes(host.available_ram_bytes),
        crate::core::slash_commands::helpers::format_bytes(host.total_ram_bytes)
    )?;
    if let Some(vram) = host.available_vram_bytes {
        writeln!(
            w,
            "VRAM:   {} free",
            crate::core::slash_commands::helpers::format_bytes(vram)
        )?;
    } else {
        writeln!(w, "VRAM:   (unified / not probed)")?;
    }
    writeln!(w, "Device: {:?}", host.recommended_device)?;
    writeln!(
        w,
        "Hint:   set CALICHE_LOCAL_INFERENCE_RESERVE_RAM_BYTES to raise/lower the RAM reserve"
    )?;
    for note in &host.notes {
        writeln!(w, "Note:   {note}")?;
    }
    Ok(())
}

#[cfg(feature = "local-inference-base")]
fn write_local_inference_fit<W: Write>(w: &mut W) -> Result<()> {
    writeln!(w)?;
    writeln!(w, "=== Local Inference Fit ===\n")?;
    match caliche::llm::LocalInferenceProvider::probe_host_capabilities(
        caliche::llm::LocalInferenceProvider::fit_options_from_env(),
    ) {
        Ok(host) => write_host_capabilities(w, &host)?,
        Err(error) => writeln!(w, "Host probe failed: {error}")?,
    }

    match caliche::llm::LocalInferenceProvider::fit_catalog_models(
        caliche::llm::LocalInferenceProvider::fit_options_from_env(),
    ) {
        Ok(fits) => {
            writeln!(w)?;
            writeln!(
                w,
                "{}",
                crate::core::slash_commands::helpers::format_model_fit_report(&fits)
            )?;
        }
        Err(error) => writeln!(w, "Catalog fit failed: {error}")?,
    }
    Ok(())
}

/// Run system diagnostics, writing to the given sink.
///
/// Listen lines use the bind remembered at startup, or the `--server` defaults.
pub async fn write_doctor_report<W: Write>(w: &mut W) -> Result<()> {
    write_doctor_report_for(w, &DoctorArgs::from_remembered()).await
}

/// Run system diagnostics for an explicit CLI listen pair.
pub async fn write_doctor_report_for<W: Write>(w: &mut W, listen: &DoctorArgs) -> Result<()> {
    writeln!(w, "\n=== Caliche Health Check ===\n")?;
    writeln!(w, "Version: {}", env!("CARGO_PKG_VERSION"))?;
    writeln!(w)?;

    let manager = SecureCredentialManager::new().await.ok();
    let readiness = match readiness_for_listen(listen).await {
        Ok(lines) => lines,
        Err(err) => {
            let lines = doctor_readiness::ReadinessLines::from_process(&listen.bind, listen.port);
            lines.write(w)?;
            return Err(err);
        }
    };
    readiness.write(w)?;

    let config_dir = dirs::home_dir()
        .map(|h| h.join(".caliche"))
        .unwrap_or_default();
    write_path_status(w, "Config Directory", &config_dir)?;
    write_session_db_status(w).await?;

    write_provider_status(w, manager.as_ref()).await?;
    write_default_provider(
        w,
        &readiness.default_provider,
        &readiness.default_provider_source,
    )?;
    write_tool_availability(w)?;
    write_system_info(w)?;
    #[cfg(feature = "local-inference-base")]
    write_local_inference_fit(w)?;

    writeln!(w)?;
    writeln!(w, "Health check complete.")?;
    Ok(())
}

/// Run system diagnostics and return the report as a string (for REPL popup).
pub async fn doctor_report_to_string() -> Result<String> {
    let mut s = Vec::new();
    write_doctor_report(&mut s).await?;
    String::from_utf8(s).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Run system diagnostics to stdout and exit (CLI entry point).
pub async fn handle_doctor_command(listen: DoctorArgs) -> Result<()> {
    let mut stdout = std::io::stdout();
    write_doctor_report_for(&mut stdout, &listen).await?;
    crate::shutdown::exit(0);
}
