//! Context Window Compaction Example
//!
//! Demonstrates:
//! - Configuring compaction via config builder (CompactionConfig::builder()) for easy wiring
//! - Or via individual methods (compaction_strategy, compaction_trigger_threshold, etc.)
//! - Handling StreamEvent::Compaction when the context is compacted
//!
//! Run with:
//! ```bash
//! cargo run --example compaction --features "compaction,openai"
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::AgentBuilder;
use caliche::agent::core::compaction::{CompactionConfig, SlidingWindowCompaction};
use caliche::prelude::ProviderType;
use futures_util::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Context Window Compaction Example ===\n");

    let config = CompactionConfig::builder()
        .strategy(Arc::new(
            SlidingWindowCompaction::builder()
                .preserve_recent(6)
                .build(),
        ))
        .trigger_threshold(0.90)
        .target_threshold(0.70)
        .max_attempts(2)
        .build()?;

    let service = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o-mini".to_string())
        .compaction_config(config)
        .build()
        .await?;

    let mut stream = service.stream_chat("Say hello in one short sentence.");

    println!("Streaming (watch for Compaction if context grows):\n");
    while let Some(event) = stream.next().await {
        match event {
            Ok(ev) => match ev {
                caliche::messages::StreamEvent::TextChunk { content } => {
                    print!("{}", content);
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                }
                caliche::messages::StreamEvent::Compaction {
                    tokens_before,
                    tokens_after,
                    messages_removed,
                    summary,
                    report: _,
                } => {
                    println!(
                        "\n[Compaction] tokens {} -> {}, removed {} messages, summary: {:?}",
                        tokens_before, tokens_after, messages_removed, summary
                    );
                }
                caliche::messages::StreamEvent::LimitReached {
                    max_iterations,
                    usage,
                } => {
                    println!(
                        "\n[LimitReached] max_iterations={}, usage: {:?}",
                        max_iterations, usage
                    );
                    break;
                }
                caliche::messages::StreamEvent::Done => break,
                _ => {}
            },
            Err(e) => {
                println!("\nStream error: {}", e);
                break;
            }
        }
    }
    println!("\nDone.");
    Ok(())
}
