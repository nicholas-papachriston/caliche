//! Core traits for CLI extensibility.
//!
//! This module provides the trait-based architecture for the CLI, enabling:
//! - Pluggable command handlers
//! - Configurable service factories
//! - Extensible output formatting
//! - I/O abstraction for testability

pub mod command;
pub mod factory;
pub mod io;
pub mod output;

pub use command::{CommandContext, CommandHandler, CommandResult, ErasedHandler};
pub use factory::{ServiceConfig, ServiceFactory};
pub use io::{InputSource, OutputSink, OutputStyle as IoStyle, TerminalSink, TestInput, TestSink};
pub use output::{OutputFormatter, OutputStyle};
