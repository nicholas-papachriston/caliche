//! Caliche Desktop Library
//!
//! This crate provides a Dioxus-based desktop UI for Caliche.

#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "UI host may emit operator diagnostics on stdout/stderr"
)]

pub mod api;
pub mod app;
pub mod catalog_select;
pub mod commands;
pub mod components;
pub mod context;
pub mod harness;
pub mod hooks;
pub mod state;
pub mod styles;

pub use api::ApiClient;
pub use app::App;
pub use commands::{
    Command, CommandResult, InteractionMode, SLASH_COMMANDS, execute_command, generate_help_text,
    is_command, parse_command,
};
pub use context::{CONTEXT_REFERENCES, ContextReference, ParsedInput, parse_context_references};
pub use state::{ChatState, ConfigState, Message};
