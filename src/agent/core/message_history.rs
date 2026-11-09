//! Utilities for working with `Arc<Vec<Message>>` message history.
//!
//! Agent message history is stored as `Arc<Vec<Message>>` so that multiple
//! code paths can hold a reference cheaply (O(1) clone). This module provides
//! helpers that avoid unnecessary heap allocations when modifying history.

use crate::messages::{Message, MessageRole};
use std::sync::Arc;

/// Unwrap an `Arc<T>` if this is the sole owner; otherwise clone the inner value.
///
/// This is the canonical way to get an owned `T` from an `Arc<T>` without
/// always copying: when there is only one `Arc` referencing the allocation
/// (i.e. `Arc::strong_count == 1`), the inner value is moved out directly.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use caliche::agent::core::message_history::arc_unwrap_or_clone;
///
/// let arc = Arc::new(vec![1u32, 2, 3]);
/// let owned: Vec<u32> = arc_unwrap_or_clone(arc);
/// assert_eq!(owned, vec![1, 2, 3]);
/// ```
#[inline]
pub fn arc_unwrap_or_clone<T: Clone>(arc: Arc<T>) -> T {
    Arc::try_unwrap(arc).unwrap_or_else(|a| (*a).clone())
}

/// Prepend a single `Message` into an `Arc<Vec<Message>>` at position `pos`.
///
/// If this `Arc` is the sole owner, the message is inserted without copying the
/// entire vector. Otherwise a new `Vec` is allocated with the message prepended.
///
/// Returns the updated `Arc<Vec<Message>>`.
#[must_use]
pub fn arc_prepend_message(
    history: Arc<Vec<Message>>,
    pos: usize,
    msg: Message,
) -> Arc<Vec<Message>> {
    let mut vec = arc_unwrap_or_clone(history);
    vec.insert(pos.min(vec.len()), msg);
    Arc::new(vec)
}

/// Append a single `Message` to an `Arc<Vec<Message>>`.
///
/// If this `Arc` is the sole owner, the message is pushed without copying the
/// entire vector. Otherwise a new `Vec` is allocated with the message appended.
///
/// Returns the updated `Arc<Vec<Message>>`.
#[must_use]
pub fn arc_push_message(history: Arc<Vec<Message>>, msg: Message) -> Arc<Vec<Message>> {
    let mut vec = arc_unwrap_or_clone(history);
    vec.push(msg);
    Arc::new(vec)
}

fn is_persistent_system_message(message: &Message) -> bool {
    message
        .content
        .as_text()
        .is_some_and(|text| text.starts_with("[Previous conversation summary]"))
}

/// Remove request-scoped system context from reusable conversation history.
///
/// System prompts, generated workspace context, and tool descriptions are
/// rebuilt for each request. Keeping prior copies in memory causes duplicate
/// context and inflated prompt sizes on later turns. Auto-compaction summaries
/// are intentionally retained because they are durable conversation state.
pub fn sanitize_history_for_request<I>(history: I) -> Vec<Message>
where
    I: IntoIterator<Item = Message>,
{
    let filtered = history.into_iter().filter(|message| {
        !matches!(message.role, MessageRole::System) || is_persistent_system_message(message)
    });
    super::tool_history::repair_tool_history(filtered.collect())
}
