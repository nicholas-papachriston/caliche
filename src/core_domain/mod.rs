//! Core domain ownership zone.
//!
//! Stable framework contracts that should stay free of infrastructure
//! concerns (HTTP servers, persistence backends, language bindings).
//! Prefer importing from this module in library code that must not depend
//! on host-only crates (`caliche_server`, `caliche_cli`).

pub mod contracts {
    pub use crate::agent::kernel::{Handoff, TurnResult};
    pub use crate::agent::public::tools::{Tool, ToolCall, ToolCapability, ToolMeta, ToolResult};
    pub use crate::messages::{ChatResponse, Message, MessageRole, TokenUsage};
}

pub mod memory {
    pub use crate::agent::memory::{MemoryBuilder, MemoryStrategy, MemoryType};
}
