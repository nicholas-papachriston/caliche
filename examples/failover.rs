//! Model Failover and Auth Profile Example
//!
//! Demonstrates:
//! - Failover via config builder (FailoverConfig::builder()) for easy wiring into existing code
//! - Or via LlmClientBuilder methods (primary, fallback, cooldown_base, on_failover)
//! - Optional failover event subscription
//!
//! Run with:
//! ```bash
//! cargo run --example failover --features "failover,openai,anthropic"
//! ```
//! Set OPENAI_API_KEY and ANTHROPIC_API_KEY (or use .api_key() for primary).

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::llm::FailoverConfig;
use caliche::llm::FailoverEvent;
use caliche::prelude::{LlmClient, ProviderType};
use futures_util::StreamExt;
use tokio::sync::broadcast;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Failover Example ===\n");

    let (tx, _) = broadcast::channel::<FailoverEvent>(16);
    let mut rx = tx.subscribe();
    let task = tokio::spawn(async move {
        while let Ok(ev) = rx.recv().await {
            println!(
                "[Failover] {} -> {} | reason: {:?}",
                ev.from_provider, ev.to_provider, ev.reason
            );
        }
    });

    let config = FailoverConfig::builder()
        .primary(ProviderType::OpenAI, "gpt-4o-mini")
        .fallback(ProviderType::Anthropic, "claude-3-5-haiku-20241022")
        .cooldown_base(60)
        .on_failover(tx)
        .build()?;

    let client = LlmClient::builder().failover_config(config).build().await?;

    println!("Completion (will use primary; fallback if primary fails):");
    let response = client
        .complete_text("What is 2 + 2? Answer in one word.")
        .await?;
    println!("  Response: {}\n", response);

    println!("Streaming:");
    let mut stream = client.stream_text("Say 'hello' in one word.");
    while let Some(event) = stream.next().await {
        match event? {
            caliche::messages::StreamEvent::TextChunk { content } => {
                print!("{}", content);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            caliche::messages::StreamEvent::Done => break,
            _ => {}
        }
    }
    println!("\n");

    drop(stream);
    drop(client);
    let _ = task.await;

    println!("Done.");
    Ok(())
}
