//! Error types and error context utilities.
//!
//! This module provides error context helpers for better debugging and
//! production error tracking with request IDs, timestamps, and metadata.
//!
//! It also provides domain-specific error types for type-safe error handling
//! throughout the codebase.

pub mod agent;
pub mod context;
pub mod control;
pub mod tool;
pub mod workflow;

pub use agent::AgentError;
pub use context::{ErrorContext, ErrorContextExt};
pub use control::ControlError;
pub use tool::ToolError;
pub use workflow::WorkflowError;
