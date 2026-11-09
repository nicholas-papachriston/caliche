//! Google Gemini Provider Example
//!
//! This example demonstrates:
//! - Using Google Gemini provider with AgentBuilder
//! - Model selection (gemini-2.5-flash, etc.)
//! - Streaming support with Google provider
//! - Comparison with other providers
//!
//! Best practices shown:
//! - Environment variable configuration
//! - Error handling
//! - Token usage tracking
//!
//! Run with:
//! ```bash
//! cargo run --example google
//! ```
//!
//! Prerequisites:
//! - Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::AgentBuilder;
use caliche::prelude::ProviderType;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // Example 1: Basic Google Gemini usage
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Google)
        .model("models/gemini-2.5-flash".to_string())
        .enable_tool("calculator")
        .build()
        .await?;

    let response = agent.chat("What is 15 multiplied by 23?").await?;

    println!("Query: What is 15 multiplied by 23?");
    println!("Response: {}", response.text());
    if let Some(total) = response.usage().total_tokens {
        println!("Tokens used: {}", total);
    }

    // Example 2: Streaming with Google provider
    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Google)
        .model("models/gemini-2.5-flash".to_string())
        .build()
        .await?;

    let agent = agent_service.into_agent();
    let mut stream = agent.stream_chat_with_tools("Write a haiku about programming");

    println!("\nStreaming response:");
    while let Some(event) = stream.next().await {
        match event {
            Ok(caliche::messages::StreamEvent::TextChunk { content }) => {
                print!("{}", content);
                std::io::Write::flush(&mut std::io::stdout())?;
            }
            Ok(caliche::messages::StreamEvent::Done) => break,
            Err(e) => {
                eprintln!("Stream error: {}", e);
                break;
            }
            _ => {}
        }
    }
    println!();

    // Example 3: Provider comparison
    println!("\nComparing providers:");

    let query = "Explain quantum computing in one sentence";

    // Google Gemini
    let mut google_agent = AgentBuilder::new()
        .provider(ProviderType::Google)
        .model("models/gemini-2.5-flash".to_string())
        .build()
        .await?;

    let google_response = google_agent.chat(query).await?;
    println!("Google Gemini: {}", google_response.text());

    Ok(())
}
