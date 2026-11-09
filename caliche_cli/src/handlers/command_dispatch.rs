//! Maps [`Commands`] subcommands to registry handler names and typed arguments.

use crate::commands::{Args, Commands};
use crate::errors::CliError;
use crate::handlers::{
    CodeArgs, DoctorArgs, EvaluateArgs, InitArgs, ShareArgs, TestArgs, UnshareArgs,
};
use crate::traits::command::CommandContext;
use std::any::Any;

/// Registry handler name plus type-erased args for [`CommandRegistry::execute`].
pub(crate) struct RegistryDispatch {
    pub name: &'static str,
    pub args: Box<dyn Any + Send>,
}

impl RegistryDispatch {
    fn new<A: Any + Send + 'static>(name: &'static str, args: A) -> Self {
        Self {
            name,
            args: Box::new(args),
        }
    }
}

impl Commands {
    /// Resolve a subcommand to the registry handler name and argument payload.
    pub(crate) fn registry_dispatch(
        &self,
        ctx: &CommandContext<'_>,
        cli_args: &Args,
    ) -> Result<RegistryDispatch, CliError> {
        Ok(match self {
            Commands::Code {
                task,
                cwd,
                tools,
                max_turns,
                accept_edits,
                dry_run,
                stream,
                no_repomap,
                verify,
                no_verify,
            } => RegistryDispatch::new(
                "code",
                CodeArgs::from_command(
                    task,
                    cwd,
                    tools,
                    *max_turns,
                    *accept_edits,
                    *dry_run,
                    *stream,
                    *no_repomap,
                    verify.clone(),
                    *no_verify,
                    ctx,
                    cli_args,
                ),
            ),

            Commands::Agent { action } => RegistryDispatch::new("agent", action.clone()),
            Commands::Config { action } => RegistryDispatch::new("config", action.clone()),
            Commands::Session { action } => RegistryDispatch::new("session", action.clone()),
            Commands::Doctor => RegistryDispatch::new(
                "doctor",
                DoctorArgs::from_cli_boot(
                    &cli_args.bind,
                    cli_args.port,
                    cli_args.provider.as_deref(),
                    cli_args.model.is_some(),
                ),
            ),

            Commands::Share {
                session,
                visibility,
            } => RegistryDispatch::new(
                "share",
                ShareArgs::from_flags(session.as_deref(), visibility.as_deref()),
            ),
            Commands::Unshare { session } => {
                RegistryDispatch::new("unshare", UnshareArgs::from_flags(session.as_deref()))
            }

            Commands::Daemon { action } => RegistryDispatch::new("daemon", action.clone()),
            Commands::Server { action } => RegistryDispatch::new("server", action.clone()),

            Commands::Init {
                no_ai,
                output,
                force,
            } => RegistryDispatch::new("init", InitArgs::from_flags(*no_ai, output, *force)),

            Commands::Lsp { action } => RegistryDispatch::new("lsp", action.clone()),

            Commands::Evaluate {
                dataset,
                evaluator,
                pattern,
                criteria,
                provider,
                model,
                output_format,
            } => RegistryDispatch::new(
                "evaluate",
                EvaluateArgs::from_command(
                    dataset,
                    evaluator,
                    pattern,
                    criteria,
                    provider,
                    model,
                    output_format,
                ),
            ),

            Commands::Template { action } => RegistryDispatch::new("template", action.clone()),
            Commands::Prompt { action } => RegistryDispatch::new("template", action.clone()),

            Commands::Test { action } => {
                RegistryDispatch::new("test", TestArgs::from(action.clone()))
            }

            #[cfg(feature = "scheduling")]
            Commands::Cron { action } => RegistryDispatch::new("cron", action.clone()),

            Commands::Policy { .. }
            | Commands::Skills { .. }
            | Commands::Completions { .. }
            | Commands::Man { .. } => {
                return Err(CliError::Command("Unknown command".to_string()));
            }
        })
    }
}
