//! Streaming chat loop: LLM turns, tool execution, and stream events.

use crate::agent::memory::MemoryStrategy;
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) type PersistMemory = Option<Arc<RwLock<Box<dyn MemoryStrategy>>>>;

mod history;
mod llm;
mod run;
mod tools;
mod turn;

pub(crate) use run::stream_chat_with_messages;
