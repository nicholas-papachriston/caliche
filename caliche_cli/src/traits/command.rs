//! Command handler trait and registry.
//!
//! Provides a unified interface for CLI command handlers, enabling:
//! - Consistent command execution patterns
//! - Automatic help generation
//! - Plugin-style command registration
//! - Testable command handlers through trait objects

use crate::errors::CliResult;
use crate::output_mode::OutputMode;
use crate::storage::credentials::SecureCredentialManager;
use crate::storage::database::Database;
use crate::traits::factory::ServiceFactory;
use async_trait::async_trait;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

/// Context provided to command handlers during execution.
///
/// Contains shared resources that commands may need, avoiding
/// the need to pass many parameters to each handler.
pub struct CommandContext<'a> {
    /// Credential manager for API key access
    pub credentials: &'a SecureCredentialManager,

    /// Database for session/history storage
    pub database: &'a Database,

    /// Service factory for creating agent services
    pub service_factory: Arc<dyn ServiceFactory>,

    /// Output mode for formatting results
    pub output_mode: OutputMode,

    /// Whether verbose output is enabled
    pub verbose: bool,

    /// Working directory for file operations
    pub working_dir: std::path::PathBuf,

    /// Cancellation token for graceful shutdown (optional)
    pub cancellation_token: Option<&'a crate::cancellation::CancellationToken>,

    /// Provider from top-level --provider (e.g. openai, anthropic)
    pub provider: Option<String>,

    /// Model from top-level --model (e.g. gpt-4o, claude-3-5-sonnet)
    pub model: Option<String>,

    /// Top-level `--yolo` auto-approves tool permissions.
    pub yolo: bool,
}

/// Result of command execution.
#[derive(Debug)]
pub enum CommandResult {
    /// Command completed successfully with optional message
    Success(Option<String>),

    /// Command produced output data
    Output(serde_json::Value),

    /// Command started an interactive session
    Interactive,

    /// Command requires user confirmation to proceed
    NeedsConfirmation(String),

    /// Command was cancelled by user
    Cancelled,

    /// Session resume requested - continue to REPL with this session ID
    ResumeSession(String),
}

/// Trait for command handlers.
///
/// Each command handler implements this trait to provide:
/// - Command execution logic
/// - Metadata for help generation
/// - Type-safe argument handling
#[async_trait]
pub trait CommandHandler: Send + Sync {
    /// The argument type for this command.
    type Args: Clone + Send + Sync + 'static;

    /// Execute the command with the given arguments and context.
    async fn execute(
        &self,
        args: Self::Args,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult>;

    /// Get the command name (used for registration and help).
    fn name(&self) -> &'static str;

    /// Get a brief description of the command.
    fn description(&self) -> &'static str;

    /// Get detailed help text for the command.
    fn help(&self) -> Option<&'static str> {
        None
    }

    /// Get example usage strings.
    fn examples(&self) -> &[&'static str] {
        &[]
    }

    /// Whether this command requires credentials to be configured.
    fn requires_credentials(&self) -> bool {
        true
    }

    /// Whether this command can run in offline mode.
    fn supports_offline(&self) -> bool {
        false
    }
}

/// Type-erased command handler for dynamic dispatch.
///
/// This allows storing heterogeneous command handlers in a registry
/// while still supporting type-safe argument passing.
#[async_trait]
pub trait ErasedHandler: Send + Sync {
    /// Execute with boxed arguments.
    async fn execute_erased(
        &self,
        args: Box<dyn Any + Send>,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult>;

    /// Get the command name.
    fn name(&self) -> &'static str;

    /// Get the command description.
    fn description(&self) -> &'static str;

    /// Get detailed help.
    fn help(&self) -> Option<&'static str>;

    /// Get examples.
    fn examples(&self) -> &[&'static str];

    /// Check if credentials are required.
    fn requires_credentials(&self) -> bool;

    /// Check if offline mode is supported.
    fn supports_offline(&self) -> bool;
}

/// Blanket implementation of ErasedHandler for any CommandHandler.
#[async_trait]
impl<T: CommandHandler> ErasedHandler for T
where
    T::Args: 'static,
{
    async fn execute_erased(
        &self,
        args: Box<dyn Any + Send>,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        let args = args
            .downcast::<T::Args>()
            .map_err(|_| crate::errors::CliError::Command("Invalid argument type".to_string()))?;
        self.execute(*args, ctx).await
    }

    fn name(&self) -> &'static str {
        CommandHandler::name(self)
    }

    fn description(&self) -> &'static str {
        CommandHandler::description(self)
    }

    fn help(&self) -> Option<&'static str> {
        CommandHandler::help(self)
    }

    fn examples(&self) -> &[&'static str] {
        CommandHandler::examples(self)
    }

    fn requires_credentials(&self) -> bool {
        CommandHandler::requires_credentials(self)
    }

    fn supports_offline(&self) -> bool {
        CommandHandler::supports_offline(self)
    }
}

/// Registry for command handlers.
///
/// Provides O(1) lookup of handlers by name and supports
/// dynamic registration of custom commands.
pub struct CommandRegistry {
    handlers: HashMap<&'static str, Arc<dyn ErasedHandler>>,
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a command handler.
    pub fn register<H: CommandHandler + 'static>(&mut self, handler: H) {
        let name = handler.name();
        self.handlers.insert(name, Arc::new(handler));
    }

    /// Register a pre-wrapped handler.
    pub fn register_erased(&mut self, name: &'static str, handler: Arc<dyn ErasedHandler>) {
        self.handlers.insert(name, handler);
    }

    /// Get a handler by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn ErasedHandler>> {
        self.handlers.get(name)
    }

    /// Check if a handler exists.
    pub fn contains(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// List all registered command names.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.handlers.keys().copied()
    }

    /// Get all handlers with their metadata.
    pub fn handlers(&self) -> impl Iterator<Item = (&'static str, &Arc<dyn ErasedHandler>)> + '_ {
        self.handlers.iter().map(|(k, v)| (*k, v))
    }

    /// Execute a command by name with type-erased arguments.
    pub async fn execute(
        &self,
        name: &str,
        args: Box<dyn Any + Send>,
        ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        let handler = self.handlers.get(name).ok_or_else(|| {
            crate::errors::CliError::Command(format!("Unknown command: {}", name))
        })?;

        handler.execute_erased(args, ctx).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestHandler;

    #[async_trait]
    impl CommandHandler for TestHandler {
        type Args = String;

        async fn execute(
            &self,
            args: Self::Args,
            _ctx: &mut CommandContext<'_>,
        ) -> CliResult<CommandResult> {
            Ok(CommandResult::Success(Some(format!(
                "Executed with: {}",
                args
            ))))
        }

        fn name(&self) -> &'static str {
            "test"
        }

        fn description(&self) -> &'static str {
            "A test command"
        }
    }

    #[test]
    fn test_registry_registration() {
        let mut registry = CommandRegistry::new();
        registry.register(TestHandler);

        assert!(registry.contains("test"));
        assert!(!registry.contains("nonexistent"));
    }

    #[test]
    fn test_registry_names() {
        let mut registry = CommandRegistry::new();
        registry.register(TestHandler);

        let names: Vec<_> = registry.names().collect();
        assert!(names.contains(&"test"));
    }
}
