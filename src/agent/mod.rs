//! Agent module - core agent functionality, builder, presets, service, memory, skills, and tools

pub mod builder;
pub mod core;
pub mod hooks;
pub mod kernel;
#[cfg(feature = "learning-loop")]
pub mod learning;
pub mod memory;
pub mod modes;
pub mod operations;
pub mod presets;
#[cfg(feature = "profiles")]
pub mod profile;
pub mod public;
pub mod runtime;
pub mod service;
pub mod session;
pub mod skills;
pub mod subagent;
pub mod tools;
#[cfg(feature = "tools-rag")]
pub mod workspace_indexing;

// Re-export core types
pub use core::Agent;
pub(crate) use core::ToolExecutionDriver;
pub use kernel::{Handoff, TurnResult};

// Re-export builder and presets
pub use builder::AgentBuilder;
pub(crate) use builder::AwsCredentials;
pub use presets::AgentPresets;
pub use session::AgentSession;
pub use subagent::{SharedSubagentSelection, SubagentModelSelection, parse_model_spec};

// Re-export skills
pub use skills::{Skill, SkillBuilder};

// Re-export modes
pub use modes::{Clarification, InteractionMode, PlanDocument, PlanTask};
