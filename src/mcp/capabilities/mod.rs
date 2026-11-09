//! MCP capability implementations.

pub mod prompts;
pub mod resources;
pub mod tools;

// Explicit exports from prompts module
pub use prompts::{handle_prompts_get, handle_prompts_list};

// Explicit exports from resources module
pub use resources::handle_resources_list;
#[cfg(feature = "tools-rag")]
pub use resources::handle_resources_read;
pub use resources::handle_resources_templates_list;

// Explicit exports from tools module
pub use tools::{handle_tools_call, handle_tools_list};
