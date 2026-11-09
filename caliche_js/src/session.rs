//! JavaScript bindings for `AgentSession`
#![expect(
    clippy::use_self,
    reason = "NAPI expands methods outside the impl where Self is not available"
)]
#![expect(
    clippy::unused_async,
    reason = "the NAPI merge callback must preserve its Promise-returning JavaScript ABI"
)]

use caliche::agent::session::AgentSession as RustAgentSession;
use caliche::messages::Message;
use napi_derive::napi;
use std::sync::Arc;

use crate::MessageJs;
use crate::SharedState;

/// JavaScript wrapper for `AgentSession`
#[napi]
pub struct AgentSession {
    inner: Arc<RustAgentSession>,
}

#[napi]
impl AgentSession {
    #[napi(constructor, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn new(
        id: String,
        history: Option<Vec<MessageJs>>,
        shared_state: Option<&SharedState>,
    ) -> Self {
        let rust_session = if let Some(shared_state_ref) = shared_state {
            RustAgentSession::with_shared_state(id, (*shared_state_ref.inner).clone())
        } else if let Some(hist) = history {
            // Convert JS messages to Rust messages
            let rust_messages: Vec<Message> = hist
                .into_iter()
                .map(|msg| {
                    let role = match msg.role.as_str() {
                        "assistant" => caliche::messages::MessageRole::Assistant,
                        "system" => caliche::messages::MessageRole::System,
                        "tool" => caliche::messages::MessageRole::Tool,
                        "user" | _ => caliche::messages::MessageRole::User,
                    };
                    Message::new(role, msg.content)
                })
                .collect();
            RustAgentSession::with_history(id, rust_messages)
        } else {
            RustAgentSession::new(id)
        };

        Self {
            inner: Arc::new(rust_session),
        }
    }

    #[napi(factory, catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn with_history(id: String, history: Vec<MessageJs>) -> Self {
        let rust_messages: Vec<Message> = history
            .into_iter()
            .map(|msg| {
                let role = match msg.role.as_str() {
                    "assistant" => caliche::messages::MessageRole::Assistant,
                    "system" => caliche::messages::MessageRole::System,
                    "tool" => caliche::messages::MessageRole::Tool,
                    "user" | _ => caliche::messages::MessageRole::User,
                };
                Message::new(role, msg.content)
            })
            .collect();
        Self {
            inner: Arc::new(RustAgentSession::with_history(id, rust_messages)),
        }
    }

    #[napi(getter, catch_unwind)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Fork this session into a new isolated session (async to fork memory).
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn fork(&self, new_id: String) -> napi::Result<Self> {
        let forked = self
            .inner
            .fork(new_id)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to fork session: {e}")))?;
        Ok(Self {
            inner: Arc::new(forked),
        })
    }

    /// Merge another session's conversation history and memory into this session.
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn merge(&self, other: &AgentSession) -> napi::Result<()> {
        let other_history = other.inner.get_history().map_err(|e| {
            napi::Error::from_reason(format!("Failed to read other session history: {e}"))
        })?;
        {
            let mut self_inner = self.inner.conversation_history.write().map_err(|e| {
                napi::Error::from_reason(format!("Failed to acquire write lock: {e}"))
            })?;
            self_inner.extend(other_history.iter().cloned());
        }
        Ok(())
    }

    /// Get the current conversation history
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn get_history(&self) -> napi::Result<Vec<MessageJs>> {
        let rust_messages = self
            .inner
            .get_history()
            .map_err(|e| napi::Error::from_reason(format!("Failed to get history: {e}")))?;

        // Convert Rust messages to JS messages
        Ok(rust_messages
            .into_iter()
            .map(|msg| MessageJs {
                role: match msg.role {
                    caliche::messages::MessageRole::User => "user".to_string(),
                    caliche::messages::MessageRole::Assistant => "assistant".to_string(),
                    caliche::messages::MessageRole::System => "system".to_string(),
                    caliche::messages::MessageRole::Tool => "tool".to_string(),
                },
                content: msg.extract_text(),
                content_parts: None,
            })
            .collect())
    }

    /// Add a message to the conversation history
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn add_message(&self, message: MessageJs) -> napi::Result<()> {
        let role = match message.role.as_str() {
            "assistant" => caliche::messages::MessageRole::Assistant,
            "system" => caliche::messages::MessageRole::System,
            "tool" => caliche::messages::MessageRole::Tool,
            "user" | _ => caliche::messages::MessageRole::User,
        };
        let rust_message = Message::new(role, message.content);

        self.inner
            .add_message(rust_message)
            .map_err(|e| napi::Error::from_reason(format!("Failed to add message: {e}")))
    }

    /// Clear the conversation history
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn clear_history(&self) -> napi::Result<()> {
        self.inner
            .clear_history()
            .map_err(|e| napi::Error::from_reason(format!("Failed to clear history: {e}")))
    }

    /// Get the shared state
    #[napi(getter, catch_unwind)]
    pub fn shared_state(&self) -> SharedState {
        SharedState {
            inner: Arc::new(self.inner.shared_state.clone()),
        }
    }
}
