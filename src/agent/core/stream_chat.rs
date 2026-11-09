use super::stream_chat_loop::{self, PersistMemory};
use super::tool_output;
use crate::messages::{Message, MessageRole};

impl super::Agent {
    /// Stream a chat response with tools (for SSE endpoints)
    /// Returns a stream of `StreamEvent`s that can be forwarded to clients.
    /// Handles tool execution iteratively and sends final token usage at the end
    #[must_use]
    pub fn stream_chat_with_tools(
        &self,
        user_message: &str,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        self.stream_chat_with_tools_and_context(user_message, None)
    }

    /// Like `stream_chat_with_tools_and_context` but cancellation-aware.
    #[must_use]
    pub fn stream_chat_with_tools_and_context_with_cancel_flag(
        &self,
        user_message: &str,
        workspace_context: Option<&str>,
        cancellation_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        persist_memory: PersistMemory,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        let tool_cfg = self.tool_calling_config();
        let mut messages = self.get_or_build_system_messages(&tool_cfg, workspace_context);
        messages.push(Message {
            role: MessageRole::User,
            content: crate::messages::MessageContent::text(user_message.to_string()),
            metadata: None,
        });

        self.stream_chat_with_messages(messages, tool_cfg, cancellation_flag, persist_memory)
    }

    /// Like `stream_chat_with_tools` but with optional workspace context for RAG-guided responses.
    #[must_use]
    pub fn stream_chat_with_tools_and_context(
        &self,
        user_message: &str,
        workspace_context: Option<&str>,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        self.stream_chat_with_tools_and_context_with_cancel_flag(
            user_message,
            workspace_context,
            None,
            None,
        )
    }

    /// Stream a chat response with conversation history
    #[must_use]
    pub fn stream_chat_with_history(
        &self,
        user_message: &str,
        history: Vec<Message>,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        self.stream_chat_with_history_and_context(user_message, history, None)
    }

    /// Like `stream_chat_with_history_and_context` but accepts `Arc<Vec<Message>>` to avoid
    /// cloning when solely owned.
    #[must_use]
    pub fn stream_chat_with_history_arc_and_context(
        &self,
        user_message: &str,
        history: std::sync::Arc<Vec<Message>>,
        workspace_context: Option<&str>,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        self.stream_chat_with_history_arc_and_context_with_cancel_flag(
            user_message,
            history,
            workspace_context,
            None,
            None,
        )
    }

    /// Like `stream_chat_with_history_arc_and_context` but cancellation-aware.
    #[must_use]
    pub fn stream_chat_with_history_arc_and_context_with_cancel_flag(
        &self,
        user_message: &str,
        history: std::sync::Arc<Vec<Message>>,
        workspace_context: Option<&str>,
        cancellation_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        persist_memory: PersistMemory,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        let tool_cfg = self.tool_calling_config();
        let mut messages = self.get_or_build_system_messages(&tool_cfg, workspace_context);

        let history_vec = super::message_history::sanitize_history_for_request(
            super::message_history::arc_unwrap_or_clone(history),
        );
        messages.extend(history_vec);
        messages.push(Message {
            role: MessageRole::User,
            content: crate::messages::MessageContent::text(user_message.to_string()),
            metadata: None,
        });

        self.stream_chat_with_messages(messages, tool_cfg, cancellation_flag, persist_memory)
    }

    /// Like `stream_chat_with_history_arc_and_context_with_cancel_flag` but accepts a pre-built
    /// multimodal `Message` instead of a plain string. Used for image/audio/document attachments.
    #[must_use]
    pub fn stream_chat_with_multimodal_message_and_history_arc(
        &self,
        user_message: crate::messages::Message,
        history: std::sync::Arc<Vec<Message>>,
        workspace_context: Option<&str>,
        cancellation_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        persist_memory: PersistMemory,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        let tool_cfg = self.tool_calling_config();
        let mut messages = self.get_or_build_system_messages(&tool_cfg, workspace_context);
        let history_vec = super::message_history::sanitize_history_for_request(
            super::message_history::arc_unwrap_or_clone(history),
        );
        messages.extend(history_vec);
        messages.push(user_message);
        self.stream_chat_with_messages(messages, tool_cfg, cancellation_flag, persist_memory)
    }

    /// Like `stream_chat_with_history` but with optional workspace context for RAG-guided responses.
    #[must_use]
    pub fn stream_chat_with_history_and_context(
        &self,
        user_message: &str,
        history: Vec<Message>,
        workspace_context: Option<&str>,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        let tool_cfg = self.tool_calling_config();
        let mut messages = self.get_or_build_system_messages(&tool_cfg, workspace_context);

        tracing::info!(
            "[Streaming] Adding {} history messages to conversation",
            history.len()
        );
        for (i, msg) in history.iter().enumerate() {
            let preview = msg
                .content
                .as_text()
                .map_or("[no text]", |text| tool_output::truncate_utf8(text, 50));
            tracing::debug!("  History[{i}]: {:?} - {preview}", msg.role);
        }

        messages.extend(super::message_history::sanitize_history_for_request(
            history,
        ));
        messages.push(Message {
            role: MessageRole::User,
            content: crate::messages::MessageContent::text(user_message.to_string()),
            metadata: None,
        });
        tracing::info!(
            "[Streaming] Sending {} total messages to LLM",
            messages.len()
        );

        self.stream_chat_with_messages(messages, tool_cfg, None, None)
    }

    fn stream_chat_with_messages(
        &self,
        messages: Vec<Message>,
        tool_cfg: super::ToolCallingConfig,
        cancellation_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        persist_memory: PersistMemory,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        stream_chat_loop::stream_chat_with_messages(
            self,
            messages,
            tool_cfg,
            cancellation_flag,
            persist_memory,
        )
    }
}
