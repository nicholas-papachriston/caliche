//! # Caliche CLI
//!
//! Command-line interface for caliche.
//!
//! ## Architecture
//!
//! The CLI is organized into several key modules:
//!
//! - [`errors`]: Structured error types for CLI operations
//! - [`traits`]: Core traits for extensibility (CommandHandler, ServiceFactory, etc.)
//! - [`handlers`]: Command handler implementations
//! - [`repl`]: Read-eval-print loop implementation
//! - [`storage`]: Persistence layer (SQLite, credentials)
//!
//! Host lint policy inherits `[workspace.lints]`; stdout/stderr printing is
//! allowed here because this crate is the interactive CLI surface.

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "CLI host intentionally writes to stdout/stderr"
)]

// Core infrastructure
pub mod boot_provider;
pub mod bootstrap;
pub mod build_info;
pub mod collections;
pub mod config;
pub mod context_refs;
pub mod core;
pub mod errors;
pub mod traits;

// Existing modules
pub mod approval_resolver;
pub mod cancellation;
pub mod chains;
pub mod client;
pub mod commands;
pub mod cost;
pub mod credentials;
pub mod custom_commands;
pub mod daemon;
pub mod deferred;
pub mod diff;
pub mod discovery;
pub mod documents;
pub mod export;
pub mod gateway_runtime;
pub mod handlers;
pub mod input;
pub mod lsp;
pub mod notify;
pub mod orchestration;
pub mod output;
pub mod output_mode;
pub mod permission_prompt;
pub mod process_liveness;
pub mod prompts;
pub mod reasoning;
pub mod runtime;
pub mod sandbox_startup;
pub mod repomap {
    use std::path::Path;

    pub fn build(workspace: &Path, _task: Option<&str>, max_tokens: usize) -> String {
        caliche::repomap::build(workspace, max_tokens)
    }
}
pub mod server;
pub mod server_registry;
pub mod session;
pub mod shutdown;
pub mod slash_catalog;
pub mod storage;
pub mod thin_client;

pub use context_refs::{ContextReference, parse_context_references};
pub use credentials::ProviderStatus;
pub use input::{InputHelper, print_slash_commands, slash_commands};
pub use output_mode::OutputMode;

pub mod startup_prompts;
pub mod templates;
pub mod test;

#[cfg(test)]
mod test_env;
pub mod utils;

pub mod repl;
pub mod tools;

use caliche::config::Config;

/// Status line printed before the session id and resume commands.
///
/// One plain sentence. It does not mention tools, the workspace index,
/// or the active model.
pub const fn generate_exit_message() -> &'static str {
    "Session ended."
}

/// Print the REPL exit status as a dim line.
pub fn print_goodbye(message: &str) {
    crate::repl::stdout_sync::with_stdout(|out| {
        use std::io::Write;
        let _ = writeln!(out, "\n\x1b[90m{message}\x1b[0m");
        let _ = out.flush();
    });
}

/// Print session ID and commands to resume this session (call after print_goodbye when exiting REPL).
pub fn print_session_resume(session_id: &str) {
    crate::repl::stdout_sync::with_stdout(|out| {
        use std::io::Write;
        let _ = writeln!(out, "\x1b[90mSession:\x1b[0m \x1b[1m{}\x1b[0m", session_id);
        let _ = writeln!(
            out,
            "\x1b[90mJump back in:\x1b[0m \x1b[36mcaliche --session {}\x1b[0m",
            session_id
        );
        let _ = writeln!(
            out,
            "\x1b[90mOr:\x1b[0m           \x1b[36mcaliche session resume {}\x1b[0m\n",
            session_id
        );
        let _ = out.flush();
    });
}

/// CLI execution context
pub struct CliContext {
    pub config: Config,
    pub output_format: crate::commands::OutputFormat,
}

impl CliContext {
    pub fn new(config: Config, output_format: crate::commands::OutputFormat) -> Self {
        Self {
            config,
            output_format,
        }
    }

    /// Format output based on the configured format
    pub fn format_output(&self, data: &serde_json::Value) -> anyhow::Result<String> {
        use crate::commands::OutputFormat;
        match &self.output_format {
            OutputFormat::Json => serde_json::to_string_pretty(data).map_err(Into::into),
            OutputFormat::Jsonl => serde_json::to_string(data).map_err(Into::into),
            OutputFormat::Text => Ok(format!("{}", data)),
            OutputFormat::Stream => {
                // For stream format, output as JSONL with event type
                serde_json::to_string(data).map_err(Into::into)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn repl_exit_status_is_a_plain_goodbye() {
        assert_eq!(super::generate_exit_message(), "Session ended.");
    }
}
