//! Slash command execution for interactive modes.
//!
//! This module provides a unified executor for all slash commands that works
//! across REPL and CLI modes. Commands return mode-agnostic results that
//! renderers can interpret appropriately.

mod attach;
mod creds;
mod creds_gateway;
mod creds_gateway_parse;
#[cfg(test)]
mod creds_store_tests;
mod editor;
mod git_ops;
mod harness;
pub(crate) mod helpers;
mod knowledge;
mod local_model;
mod modes;
#[cfg(test)]
mod project_context_tests;
mod provider;
mod session;
#[cfg(test)]
mod setup_session_tests;
mod workspace;

use crate::core::session::SessionManager;
use crate::core::state::InteractiveState;
use crate::input::{QUICK_HELP_COMMANDS, slash_commands, suggest_slash_command};
use crate::repl::commands::{
    HarnessCommand, KnowledgeCommand, ModeCommand, ProjectCommand, ProviderCommand, SessionCommand,
    SlashCommand, SystemCommand, UsageCommand,
};
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::modes::InteractionMode;
use std::path::PathBuf;

/// Result of executing a slash command.
/// Result of executing a slash command.
///
/// These results are mode-agnostic and can be interpreted by any renderer.
#[derive(Debug, Clone)]
pub enum SlashCommandResult {
    /// Continue normally
    Continue,
    /// Exit the session
    Exit,
    /// Show help with command list
    ShowHelp(Vec<(String, String)>),
    /// Show a popup/modal with title and content
    ShowPopup { title: String, content: String },
    /// Show a list popup
    ShowList {
        title: String,
        items: Vec<String>,
        descriptions: Vec<Option<String>>,
    },
    /// Show a status message
    StatusMessage(String),
    /// Show an error message
    Error(String),
    /// Editor content ready to use as message
    EditorContent(String),
    /// Mode was changed
    ModeChanged(InteractionMode),
    /// Show token usage information
    ShowTokenUsage {
        session_tokens: u64,
        context_tokens: u64,
        cache_creation_tokens: u64,
        cache_read_tokens: u64,
        session_prompt_tokens: u64,
    },
    /// Show budget information
    ShowBudget {
        session_tokens: u64,
        context_tokens: u64,
    },
    /// Show stats
    ShowStats {
        total_tokens: u64,
        message_count: usize,
        provider: String,
        model: String,
        quality_summary: String,
    },
}

/// Unified slash command executor.
///
/// Executes slash commands and returns mode-agnostic results that can be
/// interpreted by any renderer (REPL, JSON, etc.).
pub struct SlashCommandExecutor;

#[cfg(not(feature = "local-inference-base"))]
impl SlashCommandExecutor {
    #[expect(
        clippy::unused_async,
        reason = "the local-inference build of this method is async"
    )]
    pub(crate) async fn handle_local_model(_args: &[String]) -> Result<SlashCommandResult> {
        Ok(SlashCommandResult::Error(
            "Local inference is not in this build. Rebuild with --features local-inference."
                .to_string(),
        ))
    }
}

impl SlashCommandExecutor {
    /// Reference to the lazily initialized knowledge store, or a user-facing error.
    pub(crate) fn knowledge_store_ref(
        state: &InteractiveState,
    ) -> std::result::Result<&crate::core::knowledge_store::SharedKnowledgeStore, SlashCommandResult>
    {
        state.knowledge_store.as_ref().ok_or_else(|| {
            SlashCommandResult::Error(
                "Knowledge store is unavailable after setup. Try /doctor or a provider with embeddings."
                    .to_string(),
            )
        })
    }

    /// Execute a slash command.
    /// When harness is Some, the effective service is taken from it; otherwise `service` is used.
    pub async fn execute(
        cmd: &SlashCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
        mut harness: Option<&mut caliche::AgentHarness>,
        session_manager: Option<&SessionManager>,
    ) -> Result<SlashCommandResult> {
        // Handle harness-only commands first so we don't need service from harness
        if let SlashCommand::Harness(harness_cmd) = cmd {
            return Self::execute_harness(harness_cmd, state, harness).await;
        }
        if let SlashCommand::Knowledge(KnowledgeCommand::Diff(file)) = cmd
            && file.is_none()
            && let Some(active) = harness.as_mut()
        {
            return Self::handle_session_diff_overview(active).await;
        }
        let effective_service = harness.as_mut().map(|h| h.service_mut()).unwrap_or(service);
        Self::execute_with_service(cmd, state, effective_service, session_manager).await
    }

    async fn execute_harness(
        cmd: &HarnessCommand,
        state: &mut InteractiveState,
        harness: Option<&mut caliche::AgentHarness>,
    ) -> Result<SlashCommandResult> {
        match cmd {
            HarnessCommand::Checkpoint(name) => {
                Self::handle_checkpoint(name.as_deref(), harness).await
            }
            HarnessCommand::Rollback(id) => Self::handle_rollback(id.as_deref(), harness).await,
            HarnessCommand::Undo => Self::handle_undo(state, harness).await,
            HarnessCommand::Redo => Self::handle_redo(harness).await,
            HarnessCommand::Share | HarnessCommand::Unshare => Ok(SlashCommandResult::Error(
                "Share commands require a thin-client remote session.".to_owned(),
            )),
        }
    }

    /// Dispatch slash commands that use the agent service.
    ///
    /// Harness commands return a routing error here. `execute` handles them
    /// before this borrow of the service.
    async fn execute_with_service(
        cmd: &SlashCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
        session_manager: Option<&SessionManager>,
    ) -> Result<SlashCommandResult> {
        match cmd {
            SlashCommand::Mode(cmd) => Self::execute_mode(cmd, state, service).await,
            SlashCommand::Harness(cmd) => Err(anyhow::anyhow!(
                "internal routing error: {cmd:?} must be handled by the REPL/router before execute_with_service"
            )),
            SlashCommand::Session(cmd) => {
                Self::execute_session(cmd, state, service, session_manager).await
            }
            SlashCommand::Provider(cmd) => Self::execute_provider(cmd, state, service).await,
            SlashCommand::Usage(cmd) => Self::execute_usage(cmd, state, service),
            SlashCommand::Knowledge(cmd) => Self::execute_knowledge(cmd, state, service).await,
            SlashCommand::Project(cmd) => Self::execute_project(cmd, state, service).await,
            SlashCommand::System(cmd) => Self::execute_system(cmd, state, service).await,
        }
    }

    async fn execute_mode(
        cmd: &ModeCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        match cmd {
            ModeCommand::Plan => Self::handle_plan(state, service).await,
            ModeCommand::Ask => Self::handle_ask(state, service).await,
            ModeCommand::Agent(arg) => Self::handle_agent(state, service, arg.as_deref()).await,
            ModeCommand::Mode => Self::handle_mode(state),
            ModeCommand::ShowPlan => Self::handle_showplan(state),
        }
    }

    async fn execute_session(
        cmd: &SessionCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
        session_manager: Option<&SessionManager>,
    ) -> Result<SlashCommandResult> {
        match cmd {
            SessionCommand::Clear => Self::handle_clear(state, service).await,
            SessionCommand::Save(path) => Self::handle_save(state, path.as_ref()).await,
            SessionCommand::Load(path) => Self::handle_load(state, service, path.as_ref()).await,
            SessionCommand::Sessions(search) => {
                Self::handle_sessions(session_manager, search.as_deref()).await
            }
            SessionCommand::Resume(id) => {
                Self::handle_resume(state, service, session_manager, id.as_deref()).await
            }
            SessionCommand::Suspend(id) => Self::handle_suspend(state, id.as_deref()),
            SessionCommand::Status => Self::handle_status(state, service),
            SessionCommand::Context => Self::handle_context(state),
            SessionCommand::Track(file) => Self::handle_track(state, file.as_deref()),
            SessionCommand::Add(pattern) => Self::handle_add(state, pattern.as_deref()).await,
            SessionCommand::Recent => Self::handle_recent().await,
            SessionCommand::Related(file) => Self::handle_related(file.as_deref()).await,
        }
    }

    async fn execute_provider(
        cmd: &ProviderCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        match cmd {
            ProviderCommand::Providers => Self::handle_providers().await,
            ProviderCommand::Provider(name) => {
                Self::handle_provider(state, service, name.as_deref()).await
            }
            ProviderCommand::Model(name) => {
                Self::handle_model(state, service, name.as_deref()).await
            }
            ProviderCommand::Models(provider) => {
                Self::handle_models(service, provider.as_deref()).await
            }
            ProviderCommand::ContextWindow(tokens) => {
                Self::handle_context_window(state, service, tokens.as_deref())
            }
            ProviderCommand::LocalModel(args) => Self::handle_local_model(args).await,
            ProviderCommand::SubagentProvider(name) => {
                Self::handle_subagent_provider(state, name.as_deref())
            }
            ProviderCommand::SubagentModel(name) => {
                Self::handle_subagent_model(state, name.as_deref())
            }
            ProviderCommand::Embedding(spec) => {
                Self::handle_embedding(state, service, spec.as_deref()).await
            }
            ProviderCommand::Embeddings => Self::handle_embeddings(),
            ProviderCommand::ImageGeneration(spec) => {
                Self::handle_image_generation(state, service, spec.as_deref()).await
            }
            ProviderCommand::ImageGenerators(provider) => {
                Self::handle_image_generators(provider.as_deref()).await
            }
            ProviderCommand::ArtifactDir(path) => {
                Self::handle_artifact_dir(state, service, path.as_deref())
            }
        }
    }

    fn execute_usage(
        cmd: &UsageCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        match cmd {
            UsageCommand::Tools => Self::handle_tools(service),
            UsageCommand::Cost => Self::handle_cost(state),
            UsageCommand::Budget => Self::handle_budget(state),
            UsageCommand::Stats => Self::handle_stats(state, service),
        }
    }

    async fn execute_knowledge(
        cmd: &KnowledgeCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        match cmd {
            KnowledgeCommand::Skill(args) => Self::handle_skill(service, args),
            KnowledgeCommand::Diff(file) => Self::handle_diff(state, file.as_deref()),
            KnowledgeCommand::Export {
                path,
                include_thinking,
            } => {
                let path_opt = path.as_ref().map(PathBuf::from);
                Self::handle_export(state, path_opt.as_ref(), *include_thinking).await
            }
            KnowledgeCommand::Compress => Self::handle_compress(state, service).await,
            KnowledgeCommand::AutoCompact(arg) => Self::handle_autocompact(state, arg.as_deref()),
            KnowledgeCommand::Learning(args) => Self::handle_learning(state, service, args),
            KnowledgeCommand::Memory(arg) => {
                Self::handle_memory(state, service, arg.as_deref()).await
            }
            KnowledgeCommand::Remember(text) => {
                Self::handle_remember(state, service, text.as_deref()).await
            }
            KnowledgeCommand::Recall(query) => {
                Self::handle_recall(state, service, query.as_deref()).await
            }
            KnowledgeCommand::Knowledge => Self::handle_knowledge(state, service).await,
            KnowledgeCommand::Forget(id) => {
                Self::handle_forget(state, service, id.as_deref()).await
            }
        }
    }

    async fn execute_project(
        cmd: &ProjectCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        match cmd {
            ProjectCommand::Changes => Self::handle_changes(state),
            ProjectCommand::Init => Self::handle_init(service).await,
            ProjectCommand::Doctor => Self::handle_doctor().await,
            ProjectCommand::Reindex => Self::handle_reindex(service).await,
            ProjectCommand::Check(args) => Self::handle_check(args).await,
            ProjectCommand::Pr => Self::handle_pr(service).await,
            ProjectCommand::Commit => Self::handle_commit(service).await,
            ProjectCommand::Review => Self::handle_review(service).await,
            ProjectCommand::Lsp => Self::handle_lsp(),
        }
    }

    async fn execute_system(
        cmd: &SystemCommand,
        state: &mut InteractiveState,
        service: &mut AgentService,
    ) -> Result<SlashCommandResult> {
        match cmd {
            SystemCommand::Compact => Self::handle_compact(state),
            SystemCommand::Verbose(level) => Self::handle_verbose(state, level.as_deref()),
            SystemCommand::Editor => Self::handle_editor().await,
            SystemCommand::Copy => Self::handle_copy(state),
            SystemCommand::Commands => Self::handle_commands().await,
            SystemCommand::Creds(args) => {
                let session = creds::CredsSession { state, service };
                Self::handle_creds(args, Some(session)).await
            }
            SystemCommand::Attach(paths) => Self::handle_attach(state, paths).await,
            SystemCommand::Help(scope) => Ok(show_help(scope.as_deref())),
            SystemCommand::Welcome => Self::handle_welcome(state, service),
            SystemCommand::Exit | SystemCommand::Quit => Ok(SlashCommandResult::Exit),
            SystemCommand::Unknown(cmd) => Ok(unknown_command(cmd)),
        }
    }
}

fn show_help(scope: Option<&str>) -> SlashCommandResult {
    let show_all = scope == Some("all");
    let commands: Vec<(String, String)> = slash_commands()
        .iter()
        .filter(|(command, _)| {
            let base = command.split_whitespace().next().unwrap_or(command);
            show_all || QUICK_HELP_COMMANDS.contains(&base)
        })
        .map(|(command, description)| (command.to_string(), description.to_string()))
        .collect();
    SlashCommandResult::ShowHelp(commands)
}

fn unknown_command(cmd: &str) -> SlashCommandResult {
    let msg = match suggest_slash_command(cmd) {
        Some(suggested) => {
            format!("Unknown command: {cmd}. Did you mean {suggested}? Type /help for commands.")
        }
        None => format!("Unknown command: {cmd}. Type /help for commands."),
    };
    SlashCommandResult::Error(msg)
}
