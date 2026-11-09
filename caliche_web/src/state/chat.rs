//! Chat state management.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::api::types::{InteractionMode, MessageMediaItem, MessageRole, TokenUsage};
use caliche_contracts::chat::parse_sources_footer;

/// Appended when the operator stops an in-flight assistant turn.
pub const GENERATION_STOPPED_NOTICE: &str = "\n\n[Generation stopped by user]";

/// Global counter for generating unique message IDs
static MESSAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique message ID
fn generate_message_id() -> String {
    let counter = MESSAGE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp = Utc::now().timestamp_millis();
    format!("msg-{}-{}", timestamp, counter)
}

/// A tool call made during a message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub success: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<i64>,
}

impl ToolCall {
    /// Create a new tool call
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: None,
            result: None,
            success: None,
            start_time: Some(Utc::now().timestamp_millis()),
            end_time: None,
        }
    }

    /// Mark the tool call as complete
    pub fn complete(&mut self, result: String, success: bool) {
        self.result = Some(result);
        self.success = Some(success);
        self.end_time = Some(Utc::now().timestamp_millis());
    }

    /// Check if the tool call is still running
    pub fn is_running(&self) -> bool {
        self.end_time.is_none()
    }
}

/// A message in the conversation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub message_id: String,
    pub role: MessageRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    #[serde(default)]
    pub streaming: bool,
    #[serde(default)]
    pub error: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_usage: Option<TokenUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Attached media items (images, audio, etc.) for inline display
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub media: Vec<MessageMediaItem>,
    /// Structured source citations extracted from the assistant reply
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<SourceCitation>,
}

/// A cited source attached to an assistant message
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceCitation {
    pub source_type: String,
    pub resource: String,
}

impl Message {
    /// Create a new user message
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            message_id: generate_message_id(),
            role: MessageRole::User,
            content: content.into(),
            timestamp: Utc::now(),
            streaming: false,
            error: false,
            token_usage: None,
            tool_calls: None,
            media: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Create a user message with media attachments for display
    pub fn user_with_media(content: impl Into<String>, media: Vec<MessageMediaItem>) -> Self {
        Self {
            message_id: generate_message_id(),
            role: MessageRole::User,
            content: content.into(),
            timestamp: Utc::now(),
            streaming: false,
            error: false,
            token_usage: None,
            tool_calls: None,
            media,
            sources: Vec::new(),
        }
    }

    /// Create a streaming assistant message
    pub fn streaming_assistant() -> Self {
        Self {
            message_id: generate_message_id(),
            role: MessageRole::Assistant,
            content: String::new(),
            timestamp: Utc::now(),
            streaming: true,
            error: false,
            token_usage: None,
            tool_calls: None,
            media: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Create an error message
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn error(content: impl Into<String>) -> Self {
        Self {
            message_id: generate_message_id(),
            role: MessageRole::Assistant,
            content: content.into(),
            timestamp: Utc::now(),
            streaming: false,
            error: true,
            token_usage: None,
            tool_calls: None,
            media: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Create a system message (for command results)
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            message_id: generate_message_id(),
            role: MessageRole::Assistant,
            content: content.into(),
            timestamp: Utc::now(),
            streaming: false,
            error: false,
            token_usage: None,
            tool_calls: None,
            media: Vec::new(),
            sources: Vec::new(),
        }
    }

    /// Check if this is a system/command message
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn is_system(&self) -> bool {
        // System messages don't have token usage and aren't errors
        self.role == MessageRole::Assistant && self.token_usage.is_none() && !self.error
    }
}

/// Token usage totals
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenTotals {
    pub prompt: u32,
    pub completion: u32,
    pub embedding: u32,
    pub total: u32,
}

impl TokenTotals {
    /// Add usage to the totals
    pub fn add(&mut self, usage: &TokenUsage) {
        self.prompt += usage.prompt_tokens.unwrap_or(0);
        self.completion += usage.completion_tokens.unwrap_or(0);
        self.embedding += usage.embedding_tokens.unwrap_or(0);
        self.total += usage.total_tokens.unwrap_or(0);
    }

    /// Reset the totals
    pub fn reset(&mut self) {
        self.prompt = 0;
        self.completion = 0;
        self.embedding = 0;
        self.total = 0;
    }
}

/// Chat state
#[derive(Debug, Clone)]
pub struct ChatState {
    /// All messages in the conversation
    pub messages: Vec<Message>,
    /// Whether we're currently streaming a response
    pub is_streaming: bool,
    /// Current session ID (if any)
    pub current_session_id: Option<String>,
    /// Total token usage
    pub total_tokens: TokenTotals,
    /// ID of the message currently being streamed
    pub streaming_message_id: Option<String>,
    /// Current interaction mode
    pub mode: InteractionMode,
    /// Whether a session has been created
    pub session_created: bool,
}

impl Default for ChatState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            is_streaming: false,
            current_session_id: None,
            total_tokens: TokenTotals::default(),
            streaming_message_id: None,
            mode: InteractionMode::Agent,
            session_created: false,
        }
    }
}

impl ChatState {
    /// Create a new chat state
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a message to the conversation
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    /// Get a mutable reference to a message by ID
    pub fn get_message_mut(&mut self, message_id: &str) -> Option<&mut Message> {
        self.messages
            .iter_mut()
            .find(|m| m.message_id == message_id)
    }

    /// Update the content of a streaming message
    pub fn update_streaming_content(&mut self, message_id: &str, content: &str) {
        if let Some(msg) = self.get_message_mut(message_id) {
            if msg.streaming {
                msg.content.push_str(content);
            }
        }
    }

    /// Add a tool call to a message
    pub fn add_tool_call(&mut self, message_id: &str, tool_call: ToolCall) {
        if let Some(msg) = self.get_message_mut(message_id) {
            if msg.tool_calls.is_none() {
                msg.tool_calls = Some(Vec::new());
            }
            if let Some(ref mut calls) = msg.tool_calls {
                calls.push(tool_call);
            }
        }
    }

    /// Complete a tool call
    pub fn complete_tool_call(
        &mut self,
        message_id: &str,
        tool_key: &str,
        result: String,
        success: bool,
    ) {
        if let Some(msg) = self.get_message_mut(message_id)
            && let Some(ref mut calls) = msg.tool_calls
        {
            let by_id = calls
                .iter_mut()
                .find(|call| call.id == tool_key && call.is_running());
            if let Some(call) = by_id {
                call.complete(result, success);
                return;
            }
            if let Some(call) = calls
                .iter_mut()
                .find(|call| call.name == tool_key && call.is_running())
            {
                call.complete(result, success);
            }
        }
    }

    /// Finalize a streaming message
    pub fn finalize_message(&mut self, message_id: &str, usage: Option<TokenUsage>) {
        let mut add_usage = None;
        if let Some(msg) = self.get_message_mut(message_id) {
            let was_streaming = msg.streaming;
            msg.streaming = false;
            if was_streaming && let Some(ref usage) = usage {
                msg.token_usage = Some(usage.clone());
                add_usage = Some(usage.clone());
            }
            if msg.sources.is_empty()
                && let Some(parsed) = parse_sources_footer(&msg.content)
            {
                msg.sources = parsed
                    .into_iter()
                    .map(|citation| SourceCitation {
                        source_type: citation.source_type,
                        resource: citation.resource,
                    })
                    .collect();
            }
        }
        if let Some(usage) = add_usage {
            self.total_tokens.add(&usage);
        }
        self.is_streaming = false;
        self.streaming_message_id = None;
    }

    /// End the in-flight assistant row and record that the operator stopped it.
    ///
    /// Clears streaming flags even when the cancel request returns before any
    /// stream event arrives, so the composer is not left blocked.
    pub fn stop_generation(&mut self) {
        if let Some(message_id) = self.streaming_message_id.take() {
            self.update_streaming_content(&message_id, GENERATION_STOPPED_NOTICE);
            self.finalize_message(&message_id, None);
        }
        self.is_streaming = false;
        self.streaming_message_id = None;
    }

    /// Mark a message as an error
    pub fn mark_error(&mut self, message_id: &str, error_content: &str) {
        if let Some(msg) = self.get_message_mut(message_id) {
            if msg.content.is_empty() {
                msg.content = format!("**Error:** {}", error_content);
            } else {
                msg.content
                    .push_str(&format!("\n\n**Error:** {}", error_content));
            }
            msg.error = true;
            msg.streaming = false;
        }
        self.is_streaming = false;
        self.streaming_message_id = None;
    }

    /// Clear all messages
    pub fn clear(&mut self) {
        self.messages.clear();
        self.total_tokens.reset();
        self.is_streaming = false;
        self.streaming_message_id = None;
    }

    /// Get the conversation history for the API
    pub fn get_history(&self) -> Vec<crate::api::types::ChatMessage> {
        self.messages
            .iter()
            .filter(|m| !m.streaming && !m.error)
            .map(|m| crate::api::types::ChatMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect()
    }

    /// Set the current session ID
    pub fn set_session(&mut self, session_id: String) {
        self.current_session_id = Some(session_id);
        self.session_created = true;
    }

    /// Get the current session ID
    #[allow(
        dead_code,
        reason = "the desktop bin does not call it; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn session_id(&self) -> Option<&str> {
        self.current_session_id.as_deref()
    }

    /// Check if we have an active session
    pub fn has_session(&self) -> bool {
        self.current_session_id.is_some()
    }

    /// Set the interaction mode
    pub fn set_mode(&mut self, mode: InteractionMode) {
        self.mode = mode;
    }

    /// Add a system message (for command results)
    pub fn add_system_message(&mut self, content: impl Into<String>) {
        self.messages.push(Message::system(content));
    }

    /// Clear session (but keep the session ID)
    pub fn clear_conversation(&mut self) {
        self.messages.clear();
        self.total_tokens.reset();
        self.is_streaming = false;
        self.streaming_message_id = None;
        // Keep session_id and mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn streaming_message_with_id(message_id: &str) -> Message {
        let mut message = Message::streaming_assistant();
        message.message_id = message_id.to_string();
        message
    }

    fn usage(prompt: u32, completion: u32) -> TokenUsage {
        TokenUsage {
            prompt_tokens: Some(prompt),
            completion_tokens: Some(completion),
            total_tokens: Some(prompt + completion),
            embedding_tokens: None,
            cache_creation_tokens: None,
            cache_read_tokens: None,
        }
    }

    #[test]
    fn late_stream_chunks_do_not_mutate_finalized_message() {
        let mut chat = ChatState::default();
        chat.add_message(streaming_message_with_id("assistant-1"));
        chat.update_streaming_content("assistant-1", "hello");
        chat.finalize_message("assistant-1", None);

        chat.update_streaming_content("assistant-1", " late");

        assert_eq!(chat.messages[0].content, "hello");
        assert!(!chat.messages[0].streaming);
    }

    #[test]
    fn duplicate_finalize_does_not_double_count_usage() {
        let mut chat = ChatState::default();
        chat.add_message(streaming_message_with_id("assistant-1"));

        chat.finalize_message("assistant-1", Some(usage(3, 5)));
        chat.finalize_message("assistant-1", Some(usage(3, 5)));

        assert_eq!(chat.total_tokens.prompt, 3);
        assert_eq!(chat.total_tokens.completion, 5);
        assert_eq!(chat.total_tokens.total, 8);
        assert_eq!(chat.messages[0].token_usage, Some(usage(3, 5)));
    }

    #[test]
    fn clear_conversation_resets_in_flight_stream_state() {
        let mut chat = ChatState::default();
        chat.set_session("session-1".to_string());
        chat.add_message(streaming_message_with_id("assistant-1"));
        chat.is_streaming = true;
        chat.streaming_message_id = Some("assistant-1".to_string());
        chat.finalize_message("assistant-1", Some(usage(1, 2)));

        chat.clear_conversation();

        assert!(chat.messages.is_empty());
        assert!(!chat.is_streaming);
        assert!(chat.streaming_message_id.is_none());
        assert_eq!(chat.total_tokens.total, 0);
        assert_eq!(chat.session_id(), Some("session-1"));
    }

    #[test]
    fn stop_generation_clears_the_streaming_placeholder() {
        let mut chat = ChatState::default();
        chat.add_message(streaming_message_with_id("assistant-1"));
        chat.messages[0].content = "partial".to_string();
        chat.is_streaming = true;
        chat.streaming_message_id = Some("assistant-1".to_string());

        chat.stop_generation();

        assert!(!chat.is_streaming);
        assert!(chat.streaming_message_id.is_none());
        assert!(!chat.messages[0].streaming);
        assert!(chat.messages[0].content.contains("Generation stopped"));
        assert!(chat.messages[0].content.starts_with("partial"));
    }

    #[test]
    fn finalize_message_parses_sources_footer() {
        let mut chat = ChatState::default();
        let mut message = streaming_message_with_id("assistant-1");
        message.content = "Answer.\n\nSources:\n[1] https://example.com".to_string();
        chat.add_message(message);
        chat.finalize_message("assistant-1", None);

        assert_eq!(chat.messages[0].sources.len(), 1);
        assert_eq!(chat.messages[0].sources[0].resource, "https://example.com");
    }

    #[test]
    fn completing_one_of_two_same_name_calls_leaves_the_other_running() {
        let mut chat = ChatState::default();
        chat.add_message(streaming_message_with_id("m1"));
        chat.add_tool_call("m1", ToolCall::new("id-a", "bash"));
        chat.add_tool_call("m1", ToolCall::new("id-b", "bash"));

        chat.complete_tool_call("m1", "id-b", "done-b".to_string(), true);

        let calls = chat.messages[0].tool_calls.as_ref().expect("calls");
        assert!(calls[0].is_running(), "completing B must leave A running");
        assert_eq!(calls[0].id, "id-a");
        assert!(!calls[1].is_running());
        assert_eq!(calls[1].id, "id-b");
        assert_eq!(calls[1].result.as_deref(), Some("done-b"));
        assert_eq!(calls[1].success, Some(true));
    }

    #[test]
    fn legacy_result_without_an_id_completes_by_name() {
        let mut chat = ChatState::default();
        chat.add_message(streaming_message_with_id("m1"));
        chat.add_tool_call("m1", ToolCall::new("call-1", "bash"));

        chat.complete_tool_call("m1", "bash", "ok".to_string(), false);

        let call = &chat.messages[0].tool_calls.as_ref().expect("calls")[0];
        assert!(!call.is_running());
        assert_eq!(call.id, "call-1");
        assert_eq!(call.result.as_deref(), Some("ok"));
        assert_eq!(call.success, Some(false));
    }
}
