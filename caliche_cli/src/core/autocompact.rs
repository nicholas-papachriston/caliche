//! Auto-compact functionality for conversation management.
//!
//! Automatically summarizes conversation history when approaching context limits.
//! Preserves the existing memory strategy by clearing and repopulating it.

use crate::core::state::InteractiveState;
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::memory::maybe_auto_compact_memory;
use caliche::messages::Message;

/// Perform auto-compact if threshold is reached.
///
/// This function checks if auto-compact should trigger and if so, creates a new
/// session with a summary of the conversation history.
pub async fn maybe_auto_compact(
    state: &mut InteractiveState,
    service: &mut AgentService,
) -> Result<bool> {
    let should_compact = state.should_auto_compact();
    let Some(result) = maybe_auto_compact_memory(
        service,
        state.token_budget.max_tokens as usize,
        should_compact,
    )
    .await?
    else {
        return Ok(false);
    };

    state.messages.clear();
    state.messages.push(Message::system(format!(
        "[Previous conversation summary]\n{}",
        result.summary_text
    )));
    state.messages.extend(result.kept_messages.iter().cloned());

    let summary_tokens = (result.summary_text.len() / 4) as u64;
    let kept_tokens: u64 = result
        .kept_messages
        .iter()
        .map(|m| (m.content.extract_text().len() / 4) as u64)
        .sum();
    state.context_tokens = Some(summary_tokens + kept_tokens);

    Ok(true)
}
