//! Command handlers for the CLI.
//!
//! This module contains all command handler implementations, organized by domain:
//!
//! - [`config`]: Configuration and credential management
//! - [`session`]: Session management (save, load, list, etc.)
//! - [`code`]: Code execution and agent loop
//! - [`git`]: Git operations (PR, commit, review)
//! - [`evaluation`]: Dataset evaluation and testing
//! - [`editor`]: External editor integration
//! - [`init`]: Project initialization
//! - [`pipe`]: Stdin pipe processing
//! - [`interactive`]: Interactive mode and REPL
//! - [`doctor`]: System diagnostics
//! - [`lsp`]: Language Server Protocol
//! - [`utility`]: Utility commands (list models, list tools, etc.)
//!
//! ## Command Registry
//!
//! All handlers implement [`CommandHandler`]. The [`CommandRegistry`] provides
//! O(1) lookup by name and supports dynamic registration of custom commands.

pub mod advanced;
mod agent;
mod code;
mod common;
mod config;
mod context;
#[cfg(feature = "scheduling")]
mod cron;
pub mod daemon;
pub mod doctor;
mod doctor_readiness;
pub mod editor;
mod evaluation;
pub mod git;
mod host_git;
pub mod init;
pub mod interactive;
mod lsp;
mod pipe;
mod policy;
pub mod server;
mod session;
mod share;
mod utility;
mod vault;

mod template;

mod shell;
mod skills;

mod command_dispatch;
mod stream_output;

use command_dispatch::RegistryDispatch;

// Re-export common helpers
pub use common::{detect_project_type, detect_project_type_for_display, format_tool_args};

// Re-export handler structs
pub use advanced::{AdvancedFeaturesArgs, AdvancedFeaturesHandler};
pub use code::{CodeArgs, CodeHandler};
pub use config::ConfigHandler;
pub use context::HandlerContext;
pub use daemon::DaemonHandler;
pub use doctor::{DoctorArgs, DoctorHandler, remember_boot_provider, remember_cli_listen};
pub use editor::{EditorArgs, EditorHandler};
pub use evaluation::{EvaluateArgs, EvaluateHandler};
pub use git::{CommitArgs, CommitHandler, PrArgs, PrHandler, ReviewArgs, ReviewHandler};
pub use init::{InitArgs, InitHandler};
pub use interactive::{
    InteractiveArgs, InteractiveHandler, ResumeArgs, ResumeHandler, SingleMessageArgs,
    SingleMessageHandler, run_single_message_session, run_single_message_with_service,
};
pub use server::ServerHandler;
pub use share::{ShareArgs, ShareHandler, UnshareArgs, UnshareHandler};

pub use lsp::LspHandler;
pub use pipe::{PipeArgs, PipeHandler};
pub use session::SessionHandler;
pub use utility::{ListModelsHandler, ListToolsHandler};

pub use evaluation::{TestArgs, TestHandler};
pub use template::TemplateHandler;

// Re-export handler functions for direct use
pub use editor::run_editor_command;
pub use git::{run_commit_command, run_pr_command, run_review_command};
pub use init::{ProjectConfig, load_project_config, run_init_ai_for_repl, run_init_command};
pub use interactive::{run_interactive_mode_harness, run_resume_last_session, run_single_message};
pub use pipe::run_pipe_input;
pub use utility::{handle_list_models as run_list_models, handle_list_tools as run_list_tools};

pub use agent::{list_primary_profile_names, load_profile_by_name};
pub use template::{handle_prompt_command, handle_template_command};

use crate::commands::{Args, Commands};
use crate::errors::CliResult;
use crate::traits::command::{CommandContext, CommandRegistry, CommandResult};
use anyhow::Result;

/// Create the default command registry with all built-in handlers.
pub fn create_registry() -> CommandRegistry {
    let mut registry = CommandRegistry::new();
    // Core handlers
    registry.register(agent::AgentHandler);
    registry.register(ConfigHandler);
    registry.register(SessionHandler);
    registry.register(DoctorHandler);
    registry.register(ShareHandler);
    registry.register(UnshareHandler);
    registry.register(DaemonHandler);
    registry.register(ServerHandler);
    registry.register(LspHandler);

    // Code and interactive handlers
    registry.register(CodeHandler);
    registry.register(ResumeHandler);

    // Git handlers
    registry.register(PrHandler);
    registry.register(CommitHandler);
    registry.register(ReviewHandler);

    // Utility handlers
    registry.register(ListModelsHandler);
    registry.register(ListToolsHandler);
    registry.register(EditorHandler);
    registry.register(InitHandler);
    registry.register(PipeHandler);
    registry.register(EvaluateHandler);
    registry.register(AdvancedFeaturesHandler);

    registry.register(TemplateHandler);
    registry.register(TestHandler);

    #[cfg(feature = "scheduling")]
    registry.register(cron::CronHandler);

    registry
}

/// Dispatch a command to the appropriate handler using the registry.
///
/// This is the new unified dispatch mechanism that replaces pattern matching.
pub async fn dispatch(args: &Args, ctx: &mut CommandContext<'_>) -> CliResult<CommandResult> {
    let registry = create_registry();

    // Handle subcommands
    if let Some(ref command) = args.command {
        return dispatch_subcommand(command, ctx, &registry, args).await;
    }

    // Handle flags
    if args.resume_last {
        return registry.execute("resume", Box::new(ResumeArgs), ctx).await;
    }

    if let Some(ref prompt) = args.pipe {
        let pipe_args = PipeArgs {
            prompt: prompt.clone(),
            output_format: args.output_format.clone(),
            sandbox: args.sandbox.clone(),
            sandbox_backend: args.sandbox_backend.clone(),
        };
        return registry.execute("pipe", Box::new(pipe_args), ctx).await;
    }

    // Single-message mode is handled by main.rs (session-aware path) after full
    // bootstrap, so we intentionally skip it here to avoid double-execution.

    // No subcommand matched - return success to let main.rs handle interactive mode
    Ok(CommandResult::Success(None))
}

/// Dispatch a subcommand to the appropriate handler.
async fn dispatch_subcommand(
    command: &Commands,
    ctx: &mut CommandContext<'_>,
    registry: &CommandRegistry,
    cli_args: &Args,
) -> CliResult<CommandResult> {
    let RegistryDispatch { name, args } = command.registry_dispatch(ctx, cli_args)?;
    registry.execute(name, args, ctx).await
}

/// Result of early CLI command dispatch during runtime prepare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessCommandOutcome {
    /// Continue CLI bootstrap (REPL, one-shot, or server).
    Continue { session_id: Option<String> },
    /// A subcommand finished; the process should exit without opening the REPL.
    Completed,
}

/// Process command and route to appropriate handler.
///
/// Returns [`ProcessCommandOutcome::Continue`] when bootstrap should proceed
/// (no subcommand, or session resume). Returns
/// [`ProcessCommandOutcome::Completed`] when a subcommand finished.
pub async fn process_command(
    args: &Args,
    cancellation_token: Option<&crate::cancellation::CancellationToken>,
) -> Result<ProcessCommandOutcome> {
    use crate::output_mode::OutputMode;
    use crate::storage::{credentials::SecureCredentialManager, database::Database};
    use crate::traits::factory::create_service_factory;

    if let Some(outcome) = dispatch_standalone(args).await? {
        return Ok(outcome);
    }

    // No subcommand and no flags that need early dispatch - return so main can handle
    // interactive mode or session-aware single-message mode after full bootstrap.
    let needs_context = args.command.is_some() || args.resume_last || args.pipe.is_some();
    if !needs_context {
        return Ok(ProcessCommandOutcome::Continue { session_id: None });
    }

    // Create context for handlers that need it
    let cred_manager = SecureCredentialManager::new()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize credential manager: {}", e))?;

    let database = Database::open()
        .await
        .map_err(|e| anyhow::anyhow!("Failed to initialize database: {}", e))?;

    let service_factory = create_service_factory();

    let mut ctx = CommandContext {
        credentials: &cred_manager,
        database: &database,
        service_factory,
        output_mode: OutputMode::default(),
        verbose: args.verbose,
        working_dir: std::env::current_dir().unwrap_or_default(),
        cancellation_token,
        provider: args.provider.clone(),
        model: args.model.clone(),
        yolo: args.yolo,
    };

    outcome_from_dispatch(args, dispatch(args, &mut ctx).await)
}

/// Commands that finish before the credential store and database open.
async fn dispatch_standalone(args: &Args) -> Result<Option<ProcessCommandOutcome>> {
    if let Some(Commands::Policy { action }) = &args.command {
        policy::run(action.clone()).map_err(|err| anyhow::anyhow!("{err}"))?;
        return Ok(Some(ProcessCommandOutcome::Completed));
    }
    if let Some(Commands::Config { action }) = &args.command {
        config::handle_config_command(action.clone())
            .await
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        return Ok(Some(ProcessCommandOutcome::Completed));
    }
    if let Some(Commands::Completions { shell, output }) = &args.command {
        shell::handle_completions_command(shell.clone(), output.clone())?;
        return Ok(Some(ProcessCommandOutcome::Completed));
    }
    if let Some(Commands::Man { action }) = &args.command {
        shell::handle_man_command(action.clone())?;
        return Ok(Some(ProcessCommandOutcome::Completed));
    }
    if let Some(Commands::Skills { action }) = &args.command {
        skills::handle_skills(action.clone()).map_err(|err| anyhow::anyhow!("{err}"))?;
        return Ok(Some(ProcessCommandOutcome::Completed));
    }
    // Inventory and stop must work when the session database cannot be opened.
    if let Some(Commands::Server { action }) = &args.command {
        server::run(action)
            .await
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        return Ok(Some(ProcessCommandOutcome::Completed));
    }
    Ok(None)
}

fn outcome_from_dispatch(
    args: &Args,
    result: CliResult<CommandResult>,
) -> Result<ProcessCommandOutcome> {
    match result {
        Ok(CommandResult::ResumeSession(session_id)) => Ok(ProcessCommandOutcome::Continue {
            session_id: Some(session_id),
        }),
        Ok(_) if args.command.as_ref().is_some_and(|c| c.continues_to_repl()) => {
            Ok(ProcessCommandOutcome::Continue { session_id: None })
        }
        Ok(_) if args.command.is_some() || args.pipe.is_some() => {
            Ok(ProcessCommandOutcome::Completed)
        }
        Ok(_) => Ok(ProcessCommandOutcome::Continue { session_id: None }),
        Err(err) => Err(anyhow::anyhow!("{err}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_create_registry() {
        let registry = create_registry();

        // Check core handlers are registered
        assert!(registry.contains("config"));
        assert!(registry.contains("session"));
        assert!(registry.contains("doctor"));
        assert!(registry.contains("code"));
        assert!(registry.contains("pr"));
        assert!(registry.contains("commit"));
        assert!(registry.contains("review"));
    }

    #[tokio::test]
    async fn config_path_completes_without_repl() {
        let args = Args::parse_from(["caliche", "config", "path"]);
        assert!(!args.wants_repl(true));
        let outcome = process_command(&args, None)
            .await
            .expect("config path should not fail");
        assert_eq!(outcome, ProcessCommandOutcome::Completed);
    }

    #[test]
    fn test_registry_handler_metadata() {
        let registry = create_registry();

        // Check handler metadata
        if let Some(handler) = registry.get("code") {
            assert_eq!(handler.name(), "code");
            assert!(!handler.description().is_empty());
        }

        if let Some(handler) = registry.get("config") {
            assert!(!handler.requires_credentials());
        }
    }
}
