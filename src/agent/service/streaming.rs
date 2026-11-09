#[cfg(feature = "tools-rag")]
use crate::adapters::context::fetch_workspace_context_from_rag;

use super::{AgentService, MemoryStrategy};

impl AgentService {
    /// Stream a chat response with tools enabled.
    ///
    /// Returns a stream of `StreamEvent`s that can be processed incrementally.
    /// Tool execution happens automatically during streaming.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentService;
    /// use futures_util::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let agent = AgentService::new().await?;
    ///     let mut stream = agent.stream_chat("What is 2 + 2?");
    ///     
    ///     while let Some(event) = stream.next().await {
    ///         match event? {
    ///             caliche::messages::StreamEvent::TextChunk { content } => {
    ///                 print!("{}", content);
    ///             },
    ///             caliche::messages::StreamEvent::Done => break,
    ///             _ => {}
    ///         }
    ///     }
    ///     
    ///     Ok(())
    /// }
    /// ```
    #[must_use]
    pub fn stream_chat(
        &self,
        message: &str,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        self.stream_chat_with_cancel_flag(message, None)
    }

    /// Stream a chat response with tools enabled and optional cancellation flag.
    pub fn stream_chat_with_cancel_flag(
        &self,
        message: &str,
        cancellation_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        use futures_util::stream::{StreamExt as _, once};

        tracing::info!(
            session_id = %self.session.id,
            model = %self.model,
            "agent_service_stream_chat"
        );
        let persist_memory = self.session.memory.clone();
        let memory = persist_memory.clone();
        let agent = &self.agent;
        let message = message.to_string();
        let cancellation_flag = cancellation_flag;
        #[cfg(feature = "session-tools")]
        let session_id = self.session.id.clone();

        #[cfg(feature = "tools-rag")]
        let rag = self.rag_service();
        #[cfg(feature = "tools-rag")]
        let indexing_ready = self.workspace_indexing_ready.clone();
        #[cfg(feature = "tools-rag")]
        let workspace_query = agent.workspace_context_query();

        // Create a stream that first gets the history, then fetches workspace context (if RAG),
        // then forwards the chat stream with context.
        #[cfg(feature = "tools-rag")]
        {
            Box::pin(
                once(async move {
                    if let Some(memory) = memory {
                        memory
                            .read()
                            .await
                            .get_context_arc()
                            .await
                            .unwrap_or_else(|e| {
                                tracing::warn!(error = %e, "memory load failed; using empty history");
                                std::sync::Arc::new(Vec::new())
                            })
                    } else {
                        std::sync::Arc::new(Vec::new())
                    }
                })
                .then(move |history| {
                    let message = message.clone();
                    let cancellation_flag = cancellation_flag.clone();
                    let persist_memory = persist_memory.clone();
                    #[cfg(feature = "session-tools")]
                    let session_id = session_id.clone();
                    let rag = rag.clone();
                    let ready = indexing_ready.clone();
                    let query_opts = workspace_query.clone();
                    async move {
                        let workspace_context = match &rag {
                            Some(r) => {
                                fetch_workspace_context_from_rag(
                                    Some(r),
                                    &message,
                                    ready.as_ref(),
                                    &query_opts,
                                )
                                .await
                            }
                            None => None,
                        };
                        let inner = if history.is_empty() {
                            agent.stream_chat_with_tools_and_context_with_cancel_flag(
                                &message,
                                workspace_context.as_deref(),
                                cancellation_flag,
                                persist_memory,
                            )
                        } else {
                            agent.stream_chat_with_history_arc_and_context_with_cancel_flag(
                                &message,
                                history,
                                workspace_context.as_deref(),
                                cancellation_flag,
                                persist_memory,
                            )
                        };
                        #[cfg(feature = "session-tools")]
                        let inner =
                            crate::agent::tools::session_tools::StreamWithCallerIdGuard::new(
                                session_id, inner,
                            );
                        inner
                    }
                })
                .flatten(),
            )
        }

        #[cfg(not(feature = "tools-rag"))]
        {
            Box::pin(
                once(async move {
                    if let Some(memory) = memory {
                        memory
                            .read()
                            .await
                            .get_context_arc()
                            .await
                            .unwrap_or_else(|e| {
                                tracing::warn!(error = %e, "memory load failed; using empty history");
                                std::sync::Arc::new(Vec::new())
                            })
                    } else {
                        std::sync::Arc::new(Vec::new())
                    }
                })
                .then(move |history| {
                    let message = message.clone();
                    let cancellation_flag = cancellation_flag.clone();
                    let persist_memory = persist_memory.clone();
                    #[cfg(feature = "session-tools")]
                    let session_id = session_id.clone();
                    async move {
                        let inner = if history.is_empty() {
                            agent.stream_chat_with_tools_and_context_with_cancel_flag(
                                &message,
                                None,
                                cancellation_flag,
                                persist_memory,
                            )
                        } else {
                            agent.stream_chat_with_history_arc_and_context_with_cancel_flag(
                                &message,
                                history,
                                None,
                                cancellation_flag,
                                persist_memory,
                            )
                        };
                        #[cfg(feature = "session-tools")]
                        let inner =
                            crate::agent::tools::session_tools::StreamWithCallerIdGuard::new(
                                session_id, inner,
                            );
                        inner
                    }
                })
                .flatten(),
            )
        }
    }

    /// Stream a chat response with file/image/audio/video attachments.
    ///
    /// Equivalent to `stream_chat_with_cancel_flag` but the user message is built as a
    /// multimodal `Message` containing the text plus binary attachments (images, audio, etc.).
    /// The provider must support the relevant modality or an error event will be emitted.
    pub fn stream_chat_with_attachments_and_cancel_flag(
        &self,
        message: &str,
        attachments: Vec<crate::messages::Attachment>,
        cancellation_flag: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        use futures_util::stream::{StreamExt as _, once};

        tracing::info!(
            session_id = %self.session.id,
            model = %self.model,
            "agent_service_stream_chat_attachments"
        );
        let persist_memory = self.session.memory.clone();
        let memory = persist_memory.clone();
        let agent = &self.agent;
        let message = message.to_string();
        let cancellation_flag = cancellation_flag;
        #[cfg(feature = "session-tools")]
        let session_id = self.session.id.clone();

        Box::pin(
            once(async move {
                if let Some(memory) = memory {
                    memory
                        .read()
                        .await
                        .get_context_arc()
                        .await
                        .unwrap_or_else(|e| {
                            tracing::warn!(error = %e, "memory load failed; using empty history");
                            std::sync::Arc::new(Vec::new())
                        })
                } else {
                    std::sync::Arc::new(Vec::new())
                }
            })
            .then(move |history| {
                let message = message.clone();
                let cancellation_flag = cancellation_flag.clone();
                let attachments = attachments.clone();
                let persist_memory = persist_memory.clone();
                #[cfg(feature = "session-tools")]
                let session_id = session_id.clone();
                async move {
                    let user_message =
                        crate::messages::Message::user_with_attachments(&message, attachments);
                    let inner = agent.stream_chat_with_multimodal_message_and_history_arc(
                        user_message,
                        history,
                        None,
                        cancellation_flag,
                        persist_memory,
                    );
                    #[cfg(feature = "session-tools")]
                    let inner = crate::agent::tools::session_tools::StreamWithCallerIdGuard::new(
                        session_id, inner,
                    );
                    inner
                }
            })
            .flatten(),
        )
    }

    /// Stream a chat response with conversation history.
    ///
    /// Returns a stream of `StreamEvent`s that includes previous conversation context.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::{AgentService, Message};
    /// use futures_util::StreamExt;
    ///
    /// #[tokio::main]
    /// async fn main() -> anyhow::Result<()> {
    ///     let agent = AgentService::new().await?;
    ///     let history = vec![
    ///         Message::user("My name is Alice"),
    ///         Message::assistant("Nice to meet you, Alice!"),
    ///     ];
    ///     
    ///     let mut stream = agent.stream_chat_with_history("What's my name?", history);
    ///     
    ///     while let Some(event) = stream.next().await {
    ///         match event? {
    ///             caliche::messages::StreamEvent::TextChunk { content } => {
    ///                 print!("{}", content);
    ///             },
    ///             caliche::messages::StreamEvent::Done => break,
    ///             _ => {}
    ///         }
    ///     }
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub fn stream_chat_with_history(
        &self,
        message: &str,
        history: Vec<crate::messages::Message>,
    ) -> std::pin::Pin<
        Box<
            dyn futures_util::Stream<Item = anyhow::Result<crate::messages::StreamEvent>>
                + Send
                + '_,
        >,
    > {
        use futures_util::stream::StreamExt as _;
        #[cfg(feature = "tools-rag")]
        use futures_util::stream::once;

        tracing::info!(
            session_id = %self.session.id,
            model = %self.model,
            "agent_service_stream_chat_history"
        );
        let agent = &self.agent;
        let message = message.to_string();

        #[cfg(feature = "tools-rag")]
        {
            let rag = self.rag_service();
            let indexing_ready = self.workspace_indexing_ready.clone();
            let message_fetch = message.clone();
            let query_opts = agent.workspace_context_query();
            Box::pin(
                once(async move {
                    match &rag {
                        Some(r) => {
                            fetch_workspace_context_from_rag(
                                Some(r),
                                &message_fetch,
                                indexing_ready.as_ref(),
                                &query_opts,
                            )
                            .await
                        }
                        None => None,
                    }
                })
                .then(move |workspace_context| {
                    let msg = message.clone();
                    let hist = history.clone();
                    async move {
                        agent.stream_chat_with_history_and_context(
                            &msg,
                            hist,
                            workspace_context.as_deref(),
                        )
                    }
                })
                .flatten(),
            )
        }

        #[cfg(not(feature = "tools-rag"))]
        {
            Box::pin(
                futures_util::stream::once(async move {
                    agent.stream_chat_with_history(&message, history)
                })
                .flatten(),
            )
        }
    }
}
