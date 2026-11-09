//! Conversation compression for interactive mode.
//!
//! Summarizes older messages to reduce context size when approaching token limits.

use anyhow::Result;
use caliche::AgentService;

/// Compress conversation history by summarizing older messages.
///
/// Keeps the most recent exchanges and summarizes older ones.
/// Returns the summary text and number of tokens saved.
pub async fn compress_conversation_history(
    service: &mut AgentService,
    _provider: &str,
    _model: &str,
) -> Result<(String, u64)> {
    let memory_arc = service
        .memory()
        .ok_or_else(|| anyhow::anyhow!("No memory configured"))?;

    let history: Vec<caliche::messages::Message> = {
        let memory = memory_arc.read().await;
        memory.get_context().await?
    };

    if history.len() < 3 {
        return Ok((String::new(), 0));
    }

    let keep_count = 2.min(history.len());
    let to_summarize = &history[..history.len() - keep_count];

    let summarize_text: String = to_summarize
        .iter()
        .map(|m| format!("{:?}: {}", m.role, m.content.extract_text()))
        .collect::<Vec<_>>()
        .join("\n\n");

    let original_tokens = (summarize_text.len() / 4) as u64;

    let summary_prompt = format!(
        "Summarize this conversation concisely, preserving key decisions, requirements, and context. Be brief but complete:\n\n{}",
        summarize_text
    );

    let summary_response = service.chat(&summary_prompt).await?;
    let summary = summary_response.text().to_string();

    let summary_tokens = (summary.len() / 4) as u64;
    let saved_tokens = original_tokens.saturating_sub(summary_tokens);

    tracing::info!(
        "Compressed {} messages ({} chars) to summary ({} chars)",
        to_summarize.len(),
        summarize_text.len(),
        summary.len()
    );

    Ok((summary, saved_tokens))
}
