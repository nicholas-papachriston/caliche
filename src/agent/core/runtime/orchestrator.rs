use async_trait::async_trait;

use crate::agent::core::chat_loop;
use crate::agent::core::{Agent, ToolCallingConfig};
use crate::agent::kernel::turn::Handoff;
use crate::errors::AgentError;
use crate::messages::{ChatResponse, Message, TokenUsage};

#[async_trait]
pub trait ChatOrchestrator: Send + Sync {
    async fn execute_chat(
        &self,
        agent: &Agent,
        messages: Vec<Message>,
        tool_cfg: ToolCallingConfig,
    ) -> Result<ChatResponse, AgentError>;

    async fn execute_chat_with_handoff(
        &self,
        agent: &Agent,
        messages: Vec<Message>,
        tool_cfg: ToolCallingConfig,
    ) -> Result<(String, TokenUsage, Option<Handoff>), AgentError>;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultChatOrchestrator;

#[async_trait]
impl ChatOrchestrator for DefaultChatOrchestrator {
    async fn execute_chat(
        &self,
        agent: &Agent,
        messages: Vec<Message>,
        tool_cfg: ToolCallingConfig,
    ) -> Result<ChatResponse, AgentError> {
        chat_loop::chat_with_messages_impl(agent, messages, tool_cfg).await
    }

    async fn execute_chat_with_handoff(
        &self,
        agent: &Agent,
        messages: Vec<Message>,
        tool_cfg: ToolCallingConfig,
    ) -> Result<(String, TokenUsage, Option<Handoff>), AgentError> {
        chat_loop::chat_with_messages_and_track_handoff_impl(agent, messages, tool_cfg).await
    }
}
