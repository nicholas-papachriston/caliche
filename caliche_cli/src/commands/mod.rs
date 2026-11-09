//! CLI command definitions and argument parsing structures.

mod args;
mod config;
pub mod credentials_service;
mod output;
mod provider_check;
mod subcommands;

pub use args::Args;
pub use config::{ConfigAction, GatewayConfigAction, VaultAction};
pub use output::{ChainType, CompletionShell, OutputFormat, RagCliFusionMode, RagCliSearchMode};
pub use provider_check::{
    invocation_uses_provider, validate_env_providers, validate_explicit_providers,
};
#[cfg(feature = "scheduling")]
pub use subcommands::CronAction;
pub use subcommands::{
    AgentAction, Commands, DaemonAction, LspAction, ManAction, PolicyAction, PolicySetAction,
    PromptAction, ServerAction, SessionAction, SkillsAction, TemplateAction, TestAction,
};
