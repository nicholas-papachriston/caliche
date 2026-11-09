//! Helper functions for testing

use crate::messages::{Message, MessageContent, MessageRole};
use anyhow::Result;

/// Create a test message
pub fn test_message(role: MessageRole, content: &str) -> Message {
    Message {
        role,
        content: MessageContent::text(content.to_string()),
        metadata: None,
    }
}

/// Create a test user message
pub fn user_message(content: &str) -> Message {
    test_message(MessageRole::User, content)
}

/// Create a test assistant message
pub fn assistant_message(content: &str) -> Message {
    test_message(MessageRole::Assistant, content)
}

/// Create a test system message
pub fn system_message(content: &str) -> Message {
    test_message(MessageRole::System, content)
}

/// Generate random test data
pub fn random_string(len: usize) -> String {
    const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut rng = fastrand::Rng::new();
    (0..len)
        .map(|_| {
            let idx = rng.usize(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Wait for async operation with timeout
pub async fn with_timeout<F, T>(duration: std::time::Duration, future: F) -> Result<T>
where
    F: std::future::Future<Output = T>,
{
    tokio::time::timeout(duration, future)
        .await
        .map_err(|_| anyhow::anyhow!("Operation timed out"))
}
