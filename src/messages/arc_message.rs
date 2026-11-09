//! Arc-wrapped message types for efficient sharing
//!
//! Provides zero-copy message sharing across system boundaries
//! while maintaining API compatibility with owned Message types.

use super::Message;
use std::sync::Arc;

/// Arc-wrapped message for efficient sharing
///
/// Prefer this over `Message` in hot paths where messages are
/// frequently cloned or shared across threads.
pub type ArcMessage = Arc<Message>;

/// Batch of messages with efficient conversion helpers
pub struct MessageBatch {
    messages: Vec<ArcMessage>,
}

impl MessageBatch {
    /// Create from owned messages
    pub fn from_messages(messages: Vec<Message>) -> Self {
        Self {
            messages: messages.into_iter().map(Arc::new).collect(),
        }
    }

    /// Create from Arc messages
    pub fn from_arc(messages: Vec<ArcMessage>) -> Self {
        Self { messages }
    }

    /// Convert to owned messages (clones data)
    pub fn into_messages(self) -> Vec<Message> {
        self.messages
            .into_iter()
            .map(|arc| (*arc).clone())
            .collect()
    }

    /// Get as slice of Arc messages
    pub fn as_slice(&self) -> &[ArcMessage] {
        &self.messages
    }

    /// Get iterator over Arc messages
    pub fn iter(&self) -> impl Iterator<Item = &ArcMessage> {
        self.messages.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// Extension trait for Message
pub trait MessageExt {
    /// Wrap in Arc for efficient sharing
    fn into_arc(self) -> ArcMessage;
}

impl MessageExt for Message {
    fn into_arc(self) -> ArcMessage {
        Arc::new(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_batch_from_messages_preserves_len() {
        let messages = vec![Message::user("Hello"), Message::assistant("Hi there")];

        let batch = MessageBatch::from_messages(messages);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn message_batch_into_messages_clones_text() {
        let messages = vec![Message::user("Hello"), Message::assistant("Hi there")];

        let batch = MessageBatch::from_messages(messages);
        let converted = batch.into_messages();

        assert_eq!(converted.len(), 2);
        assert_eq!(converted[0].extract_text(), "Hello");
    }

    #[test]
    fn message_ext_into_arc_shares_text() {
        let msg = Message::user("Hello");
        let arc_msg = msg.into_arc();

        assert_eq!(arc_msg.extract_text(), "Hello");
    }
}
