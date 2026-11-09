//! Stable public facade for agent-facing APIs.
//!
//! Keep this module focused on externally-consumable contracts and avoid
//! re-exporting deep implementation details from `core`, `chat_helpers`, etc.

pub use super::builder::AgentBuilder;
pub use super::core::Agent;
pub use super::modes::{Clarification, InteractionMode, PlanDocument, PlanTask};
pub use super::presets::AgentPresets;
pub use super::service::{AgentService, AgentSettings};
pub use super::session::AgentSession;

pub mod hooks {
    pub use crate::agent::hooks::{
        HookContext, HookMatcher, HookRegistry, HookResult, LifecycleHook,
        permission::PermissionMode,
    };
}

pub mod tools {
    pub use crate::agent::tools::{
        BuiltInTool, Tool, ToolCall, ToolCapability, ToolMeta, ToolResult, Toolkit,
    };
}
