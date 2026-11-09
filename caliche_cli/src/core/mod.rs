//! Core shared logic for interactive modes (REPL, CLI).
//!
//! This module provides unified implementations for:
//! - State management (`InteractiveState`)
//! - Slash command execution (`SlashCommandExecutor`)
//! - Stream processing (`StreamProcessor`)
//! - Session management (`SessionManager`)
//! - Renderer traits (`OutputRenderer`, `InputSource`)

pub mod autocompact;
pub mod decomposition;
pub mod ignore;
pub mod knowledge_store;
pub mod memory_manager;
pub mod session;
pub mod slash_commands;
pub mod state;
pub mod stream;
pub mod traits;

pub use session::SessionManager;
pub use slash_commands::{SlashCommandExecutor, SlashCommandResult};
pub use state::InteractiveState;
pub use stream::{EventSink, StreamProcessor, StreamResult};
pub use traits::{InputSource, OutputRenderer};
