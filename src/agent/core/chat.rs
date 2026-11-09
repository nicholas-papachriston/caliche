use super::message_builder;
use super::message_history::{arc_unwrap_or_clone, sanitize_history_for_request};
use super::runtime::{ChatOrchestrator, DefaultChatOrchestrator};
#[cfg(feature = "tools-rag")]
use crate::adapters::context::{DefaultWorkspaceContextProvider, WorkspaceContextProvider};
use crate::agent::kernel::turn::{Handoff, TurnResult};
use crate::agent::tools::control::HANDOFF_TOOL_NAME;
use crate::agent::tools::tool::{ToolCall, ToolResult as ToolResultType};
use crate::errors::AgentError;
use crate::messages::{CompletionResponse, Message, MessageRole, TokenUsage};
#[cfg(feature = "tools-rag")]
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Result of a single agent step (one LLM call + immediate tool executions).
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The response text from the LLM (Arc for O(1) cloning).
    pub response_text: Arc<String>,
    /// Tool calls that were made in this step (if any).
    pub tool_calls: Vec<ToolCall>,
    /// Tool execution results (if any tools were called).
    pub tool_results: Vec<(String, ToolResultType)>,
    /// Updated message history after this step.
    pub updated_history: Arc<Vec<Message>>,
    /// Token usage for this step.
    pub usage: TokenUsage,
    /// Whether there are pending tool calls that need follow-up.
    pub has_pending_tool_calls: bool,
    /// LLM call duration for this step (ms).
    pub llm_duration_ms: u64,
    /// Tool execution duration for this step (ms).
    pub tool_duration_ms: u64,
}

/// Extract token usage from a completion response, or default with a warning if missing.
pub(super) fn extract_step_usage(response: &CompletionResponse) -> TokenUsage {
    if let Some(ref usage) = response.usage {
        (*usage).ensure_total_tokens()
    } else {
        tracing::warn!("No token usage found in LLM response");
        TokenUsage::default()
    }
}

/// Extract the last assistant response from message history.
pub(super) fn get_last_assistant_response(messages: &[Message]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|m| matches!(m.role, MessageRole::Assistant))
        .map(|m| m.content.extract_text())
}

impl super::Agent {
    /// Fetch workspace RAG context for the given query when tools-rag is enabled.
    #[cfg_attr(
        not(feature = "tools-rag"),
        expect(
            clippy::unused_async,
            reason = "same signature as the async feature build"
        )
    )]
    pub(crate) async fn get_workspace_context(
        &self,
        #[cfg_attr(not(feature = "tools-rag"), allow(unused_variables))] query: &str,
    ) -> Option<String> {
        #[cfg(feature = "tools-rag")]
        {
            let query_opts = self.workspace_context_query();
            if let Some(cache) = &self.workspace_rag_context_cache {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                query.hash(&mut hasher);
                query_opts.top_k.hash(&mut hasher);
                query_opts.min_similarity.to_bits().hash(&mut hasher);
                query_opts.line_level_bm25.hash(&mut hasher);
                let key = format!("rag_ctx:{}", hasher.finish());
                if let Ok(Some(bytes)) = cache.get(&key).await
                    && let Ok(s) = String::from_utf8(bytes)
                {
                    return Some(s);
                }
            }
            match &self.workspace_rag {
                Some(rag) => {
                    let provider = DefaultWorkspaceContextProvider::with_rag(
                        Some(Arc::clone(rag)),
                        self.workspace_indexing_ready.as_ref().map(Arc::clone),
                        query_opts.clone(),
                    );
                    let context = provider.fetch_workspace_context(query).await;
                    if let (Some(cache), Some(ctx)) = (&self.workspace_rag_context_cache, &context)
                    {
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        query.hash(&mut hasher);
                        query_opts.top_k.hash(&mut hasher);
                        query_opts.min_similarity.to_bits().hash(&mut hasher);
                        query_opts.line_level_bm25.hash(&mut hasher);
                        let key = format!("rag_ctx:{}", hasher.finish());
                        let ttl = std::time::Duration::from_secs(
                            self.workspace_rag_context_cache_ttl_secs,
                        );
                        let _ = cache.set(&key, ctx.as_bytes(), Some(ttl)).await;
                    }
                    context
                }
                None => None,
            }
        }
        #[cfg(not(feature = "tools-rag"))]
        None
    }

    pub async fn chat(
        &mut self,
        user_message: &str,
    ) -> Result<crate::messages::ChatResponse, AgentError> {
        let tool_cfg = self.tool_calling_config();
        let workspace_context = self.get_workspace_context(user_message).await;

        let cached_tools = self.get_cached_tools_description(tool_cfg.include_tool_descriptions);
        let messages = message_builder::build_messages(
            &self.system_prompt,
            &self.current_dir,
            &self.tools,
            user_message,
            tool_cfg.include_tool_descriptions,
            workspace_context.as_deref(),
            cached_tools.as_deref(),
            self.include_context_in_system,
            &self.skill_index,
        );

        let orchestrator = DefaultChatOrchestrator;
        orchestrator.execute_chat(self, messages, tool_cfg).await
    }

    /// Chat with the agent using pre-built message history for conversation continuity.
    pub async fn chat_with_history(
        &mut self,
        user_message: &str,
        history: Vec<Message>,
    ) -> Result<crate::messages::ChatResponse, AgentError> {
        let tool_cfg = self.tool_calling_config();
        let workspace_context = self.get_workspace_context(user_message).await;

        let mut messages =
            self.get_or_build_system_messages(&tool_cfg, workspace_context.as_deref());

        tracing::info!("Adding {} history messages to conversation", history.len());
        for (i, msg) in history.iter().enumerate() {
            let preview = msg.content.as_text().map_or("[no text]", |t| {
                let max_chars = 50;
                let char_count = t.chars().count();
                if char_count <= max_chars {
                    t
                } else {
                    t.char_indices()
                        .nth(max_chars)
                        .map_or(t, |(idx, _)| &t[..idx])
                }
            });
            tracing::debug!("  History[{}]: {:?} - {}", i, msg.role, preview);
        }

        messages.extend(sanitize_history_for_request(history));
        messages.push(Message {
            role: MessageRole::User,
            content: crate::messages::MessageContent::text(user_message.to_string()),
            metadata: None,
        });

        tracing::info!(
            "Sending {} total messages to LLM (system + history + new message)",
            messages.len()
        );

        let orchestrator = DefaultChatOrchestrator;
        orchestrator.execute_chat(self, messages, tool_cfg).await
    }

    /// Chat with the agent using pre-built message history as Arc.
    /// Avoids cloning when the Arc has a single owner (e.g. from `BufferMemory::get_context_arc()`).
    pub async fn chat_with_history_arc(
        &mut self,
        user_message: &str,
        history: Arc<Vec<Message>>,
    ) -> Result<crate::messages::ChatResponse, AgentError> {
        let tool_cfg = self.tool_calling_config();
        let workspace_context = self.get_workspace_context(user_message).await;

        let mut messages =
            self.get_or_build_system_messages(&tool_cfg, workspace_context.as_deref());

        let history_vec = sanitize_history_for_request(arc_unwrap_or_clone(history));
        messages.extend(history_vec);
        messages.push(Message {
            role: MessageRole::User,
            content: crate::messages::MessageContent::text(user_message.to_string()),
            metadata: None,
        });

        let orchestrator = DefaultChatOrchestrator;
        orchestrator.execute_chat(self, messages, tool_cfg).await
    }

    /// Execute a turn and return a [`TurnResult`] that may indicate a handoff.
    pub async fn execute_turn(&mut self, message: &str) -> Result<TurnResult, AgentError> {
        let tool_cfg = self.tool_calling_config();
        let workspace_context = self.get_workspace_context(message).await;

        let cached_tools = self.get_cached_tools_description(tool_cfg.include_tool_descriptions);
        let messages = message_builder::build_messages(
            &self.system_prompt,
            &self.current_dir,
            &self.tools,
            message,
            tool_cfg.include_tool_descriptions,
            workspace_context.as_deref(),
            cached_tools.as_deref(),
            self.include_context_in_system,
            &self.skill_index,
        );

        let orchestrator = DefaultChatOrchestrator;
        let (response_text, usage, handoff) = orchestrator
            .execute_chat_with_handoff(self, messages, tool_cfg)
            .await?;

        match handoff {
            Some(h) => Ok(TurnResult::Handoff(h, usage)),
            None => Ok(TurnResult::Reply(response_text, usage)),
        }
    }

    /// Check for a handoff tool call in the tool results of the last step.
    pub(super) fn check_for_handoff_in_results(
        tool_results: &[(String, ToolResultType)],
    ) -> Option<Handoff> {
        for (tool_name, tool_result) in tool_results {
            if tool_name == HANDOFF_TOOL_NAME
                && tool_result.success
                && let Ok(handoff) = serde_json::from_str::<Handoff>(&tool_result.content)
            {
                return Some(handoff);
            }
        }
        None
    }
}
