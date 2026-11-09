//! State management module.

pub mod agents;
pub mod chat;
pub mod config;
pub mod history;
pub mod persist;
pub mod streaming;

#[allow(unused_imports)]
pub use agents::{AgentOutputChunk, AgentSource, AgentStatus, InjectedCommand, SubAgent};
#[allow(unused_imports)]
pub use chat::{ChatState, Message, SourceCitation, TokenTotals, ToolCall};
pub use config::ConfigState;
pub use history::SavedConversation;
#[allow(unused_imports)]
pub use persist::{ConfigStore, HistoryStore, default_config_store, default_history_store};
