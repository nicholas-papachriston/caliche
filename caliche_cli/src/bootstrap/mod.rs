//! Bootstrap module for CLI initialization.
//!
//! This module handles the complex initialization sequence for the CLI,
//! including credential loading, logging setup, provider discovery, and
//! service creation.
//!
//! ## Architecture
//!
//! - [`credentials`]: Credential loading and setup
//! - [`logging`]: Log initialization for interactive vs non-interactive modes
//! - [`discovery`]: Local provider auto-detection

mod credentials;
mod logging;
mod service;

pub use credentials::{load_credentials, load_startup_credentials, setup_credentials_if_needed};
pub use logging::{LogConfig, init_logging};
pub use service::create_agent_service;

use crate::commands::{Args, Commands};
use crate::storage::{SecureCredentialManager, database::Database};
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::core::ToolExecutionDriver;
use caliche::discovery::embedding::EmbeddingProviderDiscovery;
use caliche::workspace::WorkspaceIndexerConfig;
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::Arc;

use caliche::llm::ProviderType;

/// CLI options that customize workspace embedding behavior.
#[derive(Debug, Clone, Default)]
pub struct WorkspaceEmbedOptions {
    pub line_level_bm25: bool,
    pub build_code_graph: bool,
    pub workspace_hashline: Option<bool>,
    pub workspace_hashline_format: Option<bool>,
    /// Enable MMR diversity reranking for RAG workspace queries.
    pub rag_mmr: bool,
    /// MMR lambda when rag_mmr is true.
    pub rag_mmr_lambda: Option<f64>,
    /// Enable temporal decay for RAG workspace queries.
    pub rag_temporal_decay: bool,
    /// Temporal decay half-life in days when rag_temporal_decay is true.
    pub rag_temporal_decay_half_life: Option<f64>,
    /// Pre-discovered embedding provider from concurrent startup probing.
    /// When set, the builder skips its own discovery and uses this directly.
    pub pre_discovered_embedding: Option<EmbeddingProviderDiscovery>,
    /// When true, skip embedding probes during `AgentService` build and attach
    /// workspace RAG after the REPL prompt is up.
    pub defer_until_interactive: bool,
    /// When true, do not probe for an embedding provider. A saved selection is
    /// restored onto the service after build.
    pub skip_embedding_discovery: bool,
}

impl WorkspaceEmbedOptions {
    /// Copy indexing flags from parsed CLI args. Discovery/deferral are filled in later.
    #[must_use]
    pub fn from_args(args: &Args) -> Self {
        Self {
            line_level_bm25: args.line_level_bm25,
            build_code_graph: args.build_code_graph,
            workspace_hashline: args.workspace_hashline,
            workspace_hashline_format: args.workspace_hashline_format,
            rag_mmr: args.rag_mmr,
            rag_mmr_lambda: args.rag_mmr_lambda,
            rag_temporal_decay: args.rag_temporal_decay,
            rag_temporal_decay_half_life: args.rag_temporal_decay_half_life,
            pre_discovered_embedding: None,
            defer_until_interactive: false,
            skip_embedding_discovery: false,
        }
    }
}

/// Shared resources created once at startup and reused (credentials, database).
pub struct StartupContext {
    pub cred_manager: Option<SecureCredentialManager>,
    pub database: Option<Database>,
    /// Log file path when REPL logs to file; printed via with_stdout to avoid interleaving.
    pub log_path: Option<PathBuf>,
}

/// System prompt for the research sub-agent to reduce hallucinations when describing codebase APIs.
/// Instructs the agent to ground all type/trait/module names in actual exports and verify with fs(op=grep/read).
const RESEARCH_AGENT_SYSTEM_PROMPT: &str = r#"You are a research assistant that explores the codebase and answers questions. To avoid inventing names:

- Only use type, class, and module names that you have seen in the repo. Do not infer or guess (e.g. do not assume "Memory" or "ConversationMemory"); if you have not seen a name in files you read or fs op=grep results, do not use it.
- For public APIs and modules: read the actual export or entry files (e.g. lib.rs, mod.rs, __init__.py, index.ts) and list only the names that appear there (public exports, modules, types, or classes).
- Before stating that "X is the main type" or "Y is used for Z", confirm with fs(op=grep) or fs(op=read) that X or Y exists in the codebase.
- When summarizing modules, cite the file and (if relevant) the line or export list you used. If information is missing, say so instead of filling in with guesses.
- Prefer targeted queries: e.g. "List every public export in src/agent/memory/" (or the equivalent for the project's language) rather than broad "What are the main modules?"
"#;

/// Flush stdout at startup so the first one-shot/`code` lines are not stuck
/// in a fully-buffered pipe until process exit. Event printers also flush.
pub fn configure_stdio_buffering() {
    let _ = std::io::Write::flush(&mut std::io::stdout());
}

/// True when workspace RAG should be enabled (default on; disabled by --no-workspace-embed).
pub fn effective_workspace_embed(args: &Args) -> bool {
    effective_workspace_embed_with_tty(
        args,
        std::io::stdin().is_terminal() && std::io::stdout().is_terminal(),
    )
}

/// Workspace RAG is on for interactive REPL and `code` by default, off for one-shots.
#[must_use]
pub fn effective_workspace_embed_with_tty(args: &Args, tty: bool) -> bool {
    if args.no_workspace_embed || (args.is_lean_profile() && !args.workspace_embed) {
        return false;
    }
    if args.workspace_embed {
        return true;
    }
    args.wants_repl(tty) || matches!(args.command, Some(Commands::Code { .. }))
}

/// Attach workspace RAG to an already-built service. Indexing continues in the background.
pub async fn attach_workspace_embedding(
    service: &mut AgentService,
    workspace_root: PathBuf,
    options: &WorkspaceEmbedOptions,
    embedding: Arc<dyn caliche::llm::EmbeddingProvider>,
) -> Result<PathBuf> {
    service
        .enable_workspace_embedding(workspace_indexer_config(workspace_root, options), embedding)
        .await
}

/// Build workspace indexer config from CLI embed options.
#[must_use]
pub fn workspace_indexer_config(
    workspace_root: PathBuf,
    options: &WorkspaceEmbedOptions,
) -> WorkspaceIndexerConfig {
    let mut workspace_config = WorkspaceIndexerConfig::new(workspace_root)
        .line_level_bm25(options.line_level_bm25)
        .build_code_graph(options.build_code_graph)
        .workspace_hashline_enabled(options.workspace_hashline)
        .workspace_hashline_format(options.workspace_hashline_format)
        .with_env_overrides();
    if options.rag_mmr
        || options.rag_temporal_decay
        || options.rag_mmr_lambda.is_some()
        || options.rag_temporal_decay_half_life.is_some()
    {
        let rag_opts = caliche::rag::RagSearchOptions {
            use_mmr: if options.rag_mmr { Some(true) } else { None },
            mmr_lambda: options.rag_mmr_lambda,
            use_temporal_decay: if options.rag_temporal_decay {
                Some(true)
            } else {
                None
            },
            temporal_decay_half_life_days: options.rag_temporal_decay_half_life,
        };
        workspace_config = workspace_config.rag_search_options(Some(rag_opts));
    }
    workspace_config
}

/// Action verbs that suggest a task instruction (use phased execution).
const TASK_ACTION_VERBS: &[&str] = &[
    "add",
    "fix",
    "refactor",
    "create",
    "implement",
    "write",
    "change",
    "update",
    "remove",
    "delete",
    "modify",
    "replace",
    "improve",
    "debug",
    "investigate",
    "port",
    "migrate",
];

/// Returns true if the message looks like a multi-step task (long or contains action verbs).
pub fn message_looks_like_task(message: &str) -> bool {
    let trimmed = message.trim();
    if trimmed.len() < 10 {
        return false;
    }
    let word_count = trimmed.split_whitespace().count();
    if word_count >= 30 {
        return true;
    }
    let lower = trimmed.to_lowercase();
    TASK_ACTION_VERBS.iter().any(|verb| {
        lower.contains(&format!(" {} ", verb)) || lower.starts_with(&format!("{} ", verb))
    })
}

/// Resolve tool execution driver for "auto" mode: phased for task-like messages, off for short chat.
pub fn resolve_tool_execution_driver_for_message(
    mode: Option<&str>,
    message: &str,
) -> Option<ToolExecutionDriver> {
    let mode = mode.map(|s| s.trim().to_lowercase());
    let mode = mode.as_deref();
    if mode.is_none() || mode == Some("auto") {
        return Some(if message_looks_like_task(message) {
            ToolExecutionDriver::PlannerFirstPhased
        } else {
            ToolExecutionDriver::Off
        });
    }
    mode.and_then(ToolExecutionDriver::parse)
}

fn use_builder_path(args: &Args, defer_workspace_embed: bool) -> bool {
    (effective_workspace_embed(args) && !defer_workspace_embed)
        || args.learning_loop
        || args.compaction
        || !args.failover_fallback.is_empty()
        || args.session_tools
        || args.complex
        || args.local_inference_download_repo.is_some()
        || args.local_inference_backend.is_some()
        || args.local_inference_model_dir.is_some()
        || args.local_inference_hf_cache_dir.is_some()
}

fn effective_learning_config(
    args: &Args,
    cli_config: Option<&crate::config::Config>,
    workspace_root: &std::path::Path,
) -> Option<caliche::agent::learning::CheckpointConfig> {
    let cfg = cli_config.and_then(|config| config.learning.as_ref());
    let enabled = args.learning_loop || cfg.and_then(|learning| learning.enabled).unwrap_or(false);
    if !enabled {
        return None;
    }

    let mut learning = caliche::agent::learning::CheckpointConfig::default();
    if let Some(interval) = args
        .learning_interval
        .or_else(|| cfg.and_then(|learning| learning.interval_tool_calls))
    {
        learning.interval_tool_calls = interval.max(1);
    }
    if let Some(min_tool_calls) = args
        .learning_min_tool_calls
        .or_else(|| cfg.and_then(|learning| learning.min_tool_calls_for_skill))
    {
        learning.min_tool_calls_for_skill = min_tool_calls.max(1);
    }
    if let Some(max_skills) = args
        .learning_max_skills
        .or_else(|| cfg.and_then(|learning| learning.max_skills_per_session))
    {
        learning.max_skills_per_session = max_skills;
    }
    if let Some(auto_create) = cfg.and_then(|learning| learning.auto_create_skills) {
        learning.auto_create_skills = auto_create;
    }
    if let Some(auto_patch) = cfg.and_then(|learning| learning.auto_patch_skills) {
        learning.auto_patch_skills = auto_patch;
    }
    if let Some(prefer_weak) = cfg.and_then(|learning| learning.prefer_weak_skills) {
        learning.prefer_weak_skills = prefer_weak;
    }
    if let Some(require_holdout) = cfg.and_then(|learning| learning.require_holdout_pass) {
        learning.require_holdout_pass = require_holdout;
    }
    if let Some(min_holdout) = cfg.and_then(|learning| learning.min_holdout_extent) {
        learning.min_holdout_extent = min_holdout.clamp(0.0, 1.0);
    }

    let memory_file = args
        .learning_memory_file
        .clone()
        .or_else(|| cfg.and_then(|learning| learning.memory_file_path.clone()))
        .unwrap_or_else(|| workspace_root.join(".caliche").join("MEMORY.md"));
    learning.memory_file_path = Some(if memory_file.is_absolute() {
        memory_file
    } else {
        workspace_root.join(memory_file)
    });

    Some(learning)
}

/// Parse "provider:model" into (ProviderType, String).
fn parse_provider_model(s: &str) -> Result<(ProviderType, String)> {
    let (p, m) = s.split_once(':').ok_or_else(|| {
        anyhow::anyhow!(
            "failover fallback must be provider:model (e.g. anthropic:claude-3-5-sonnet)"
        )
    })?;
    let pt: ProviderType = p
        .trim()
        .parse()
        .map_err(|e: String| anyhow::anyhow!("invalid provider in failover fallback: {}", e))?;
    let model = m.trim().to_string();
    if model.is_empty() {
        anyhow::bail!("model part must be non-empty in provider:model");
    }
    Ok((pt, model))
}

fn configured_image_generation(
    args: &Args,
    cli_config: Option<&crate::config::Config>,
) -> Result<Option<(ProviderType, Option<String>)>> {
    let provider = args.image_provider.clone().or_else(|| {
        cli_config.and_then(|cfg| {
            cfg.image_generation
                .as_ref()
                .and_then(|image| image.provider.clone())
        })
    });
    let Some(provider) = provider else {
        return Ok(None);
    };
    if provider.trim().eq_ignore_ascii_case("clear") || provider.trim().is_empty() {
        return Ok(None);
    }
    let provider_type = provider
        .parse::<ProviderType>()
        .map_err(|e| anyhow::anyhow!("invalid image generation provider: {e}"))?;
    let model = args.image_model.clone().or_else(|| {
        cli_config.and_then(|cfg| {
            cfg.image_generation
                .as_ref()
                .and_then(|image| image.model.clone())
        })
    });
    Ok(Some((provider_type, model)))
}

fn configured_artifact_dir(
    args: &Args,
    cli_config: Option<&crate::config::Config>,
) -> Option<PathBuf> {
    args.artifact_dir.clone().or_else(|| {
        cli_config.and_then(|cfg| {
            cfg.artifacts
                .as_ref()
                .and_then(|artifacts| artifacts.dir.clone())
        })
    })
}

/// Apply CLI-specific defaults/tools to an already-constructed service.
///
/// This keeps post-construction behavior consistent across main, handlers and
/// factory-driven service creation paths.
pub fn finalize_cli_service(service: &mut AgentService, workspace_embed_used: bool) {
    finalize_cli_service_with(service, workspace_embed_used, false, false);
}

/// Apply CLI defaults, optionally with the methodical complex-task prompt.
pub fn finalize_cli_service_with(
    service: &mut AgentService,
    workspace_embed_used: bool,
    complex: bool,
    lean: bool,
) {
    let mut builder = if lean {
        crate::prompts::PromptBuilder::new()
            .base(crate::prompts::LEAN_DEMO_SYSTEM_PROMPT.to_owned())
    } else {
        crate::prompts::PromptBuilder::new().with_default_base(workspace_embed_used)
    };
    if complex && !lean {
        builder = builder.add_skill(caliche::harness::COMPLEX_TASK_PROMPT.to_owned());
    }
    service.agent_mut().set_system_prompt(builder.build());

    // Note: max_context_tokens is already set during service construction based on
    // the provider's actual context window (90% of reported window). See
    // AgentService::with_provider_and_credentials_with_rag_options in service/mod.rs.

    if !lean {
        service.add_tool(std::sync::Arc::new(crate::tools::WritePlanTool));
    }
}

/// Apply CLI config fields to an already-built service.
///
/// This covers post-build settable fields: temperature, max_tokens, tool_execution_driver.
/// Provider and model are baked in at build time and cannot be changed here.
pub fn apply_cli_config_to_service(
    service: &mut AgentService,
    config: &crate::config::Config,
) -> Result<()> {
    if config.temperature.is_some() {
        service.agent_mut().set_temperature(config.temperature);
    }
    if config.max_tokens.is_some() {
        service.agent_mut().set_max_tokens(config.max_tokens);
    }
    if let Some(context_window) = config.context_window {
        service.set_context_window(context_window)?;
    }
    if let Some(ref driver_str) = config.tool_execution_driver {
        let s = driver_str.trim().to_lowercase();
        if s != "auto" {
            if let Some(driver) = ToolExecutionDriver::parse(&s) {
                service.agent_mut().set_tool_execution_driver(driver);
            }
        }
        // When "auto", driver is resolved per-message in REPL/code path
    }
    if let Some(light) = config.lightweight_model() {
        tracing::debug!(
            lightweight_model = %light,
            "CLI config exposes lightweight_model for compaction/summary helpers"
        );
    }
    Ok(())
}

/// Build formatter rules from CLI config for use with ToolHooks.
pub fn formatter_rules_from_config(
    cfg: &crate::config::Config,
) -> Vec<caliche::harness::formatter::FormatterRule> {
    cfg.formatter
        .values()
        .filter(|e| !e.disabled)
        .map(|e| caliche::harness::formatter::FormatterRule {
            extensions: e.extensions.clone(),
            command: e.command.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn workspace_embed_defaults_on_for_repl_and_code_off_for_oneshot() {
        let repl = Args::parse_from(["caliche"]);
        assert!(effective_workspace_embed_with_tty(&repl, true));
        assert!(!effective_workspace_embed_with_tty(&repl, false));

        let oneshot = Args::parse_from(["caliche", "-p", "hi"]);
        assert!(!effective_workspace_embed_with_tty(&oneshot, true));

        let code = Args::parse_from(["caliche", "code", "add tests"]);
        assert!(effective_workspace_embed_with_tty(&code, true));
        assert!(effective_workspace_embed_with_tty(&code, false));

        let code_disabled =
            Args::parse_from(["caliche", "--no-workspace-embed", "code", "add tests"]);
        assert!(!effective_workspace_embed_with_tty(&code_disabled, true));

        let forced = Args::parse_from(["caliche", "-p", "hi", "--workspace-embed"]);
        assert!(effective_workspace_embed_with_tty(&forced, false));

        let disabled = Args::parse_from(["caliche", "--no-workspace-embed"]);
        assert!(!effective_workspace_embed_with_tty(&disabled, true));
    }

    #[test]
    fn workspace_embed_options_from_args_copy_rag_flags() {
        let args = Args::parse_from([
            "caliche",
            "--rag-mmr",
            "--build-code-graph",
            "code",
            "add tests",
        ]);
        let options = WorkspaceEmbedOptions::from_args(&args);
        assert!(options.rag_mmr);
        assert!(options.build_code_graph);
        assert!(options.line_level_bm25);
        assert!(!options.defer_until_interactive);
        assert!(!options.skip_embedding_discovery);
        assert!(options.pre_discovered_embedding.is_none());
    }

    #[test]
    fn workspace_indexer_config_preserves_rag_options() {
        let options = WorkspaceEmbedOptions {
            rag_mmr: true,
            rag_mmr_lambda: Some(0.4),
            defer_until_interactive: true,
            ..WorkspaceEmbedOptions::default()
        };
        let config = workspace_indexer_config(PathBuf::from("/tmp"), &options);
        let rag = config.rag_search_options.expect("rag options");
        assert_eq!(rag.use_mmr, Some(true));
        assert_eq!(rag.mmr_lambda, Some(0.4));
    }
}
