//! Cloudflare AI Gateway Provider Example
//!
//! Demonstrates non-streaming and streaming chat through the Cloudflare AI
//! Gateway, which exposes an OpenAI-compatible `/chat/completions` endpoint
//! that proxies requests to upstream providers (openai, anthropic, google,
//! workers-ai, etc.). Models use the `{provider}/{model}` format.
//!
//! Two authentication modes are supported:
//!
//! - BYOK (stored keys): set `CLOUDFLARE_AIG_API_KEY` to a Cloudflare API
//!   token. Cloudflare will inject the upstream provider key on your behalf.
//! - Passthrough: set `CLOUDFLARE_AIG_API_KEY` to the upstream provider's own
//!   API key, and optionally `CLOUDFLARE_AIG_AUTH_TOKEN` for an authenticated
//!   gateway (sent as the `cf-aig-authorization` header).
//!
//! Prerequisites:
//! - `CLOUDFLARE_AIG_API_KEY` (or `CLOUDFLARE_API_TOKEN`) environment variable.
//! - `CLOUDFLARE_ACCOUNT_ID` environment variable.
//! - Optional: `CLOUDFLARE_AIG_GATEWAY_ID` (defaults to `"default"`).
//! - Optional: `CLOUDFLARE_AIG_AUTH_TOKEN` for authenticated passthrough.
//! - `CLOUDFLARE_AIG_MODEL` in `{provider}/{model}` format.
//!
//! Run with:
//! ```bash
//! export CLOUDFLARE_AIG_API_KEY=...
//! export CLOUDFLARE_ACCOUNT_ID=...
//! cargo run --example cloudflare_ai_gateway --features cloudflare-ai-gateway
//! ```

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::messages::StreamEvent;
use caliche::prelude::ProviderType;
use caliche::{Agent, AgentBuilder};
use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Cloudflare AI Gateway Provider Examples ===\n");

    if std::env::var("CLOUDFLARE_AIG_API_KEY").is_err()
        && std::env::var("CLOUDFLARE_API_TOKEN").is_err()
    {
        eprintln!(
            "Skipping example: set CLOUDFLARE_AIG_API_KEY (or CLOUDFLARE_API_TOKEN) and CLOUDFLARE_ACCOUNT_ID to run."
        );
        return Ok(());
    }

    if std::env::var("CLOUDFLARE_ACCOUNT_ID").is_err() {
        eprintln!("Skipping example: set CLOUDFLARE_ACCOUNT_ID to run.");
        return Ok(());
    }

    example_basic_completion().await?;
    example_streaming_completion().await?;

    println!("\n=== Examples Complete ===");
    Ok(())
}

async fn example_basic_completion() -> Result<()> {
    println!("Example 1: Non-Streaming Chat via Cloudflare AI Gateway");
    println!("--------------------------------------------------------\n");

    let model = std::env::var("CLOUDFLARE_AIG_MODEL")
        .map_err(|_| anyhow::anyhow!("set CLOUDFLARE_AIG_MODEL in provider/model format"))?;

    let agent_service = AgentBuilder::new()
        .provider(ProviderType::CloudflareAIGateway)
        .model(model.clone())
        .build()
        .await?;

    let mut agent: Agent = agent_service.into_agent();

    let query = "In one sentence, what is Cloudflare AI Gateway?";
    println!("Model: {}", model);
    println!("Query: {}\n", query);

    let response = agent.chat(query).await?;
    println!("Response: {}", response.text());

    let usage = response.usage();
    if let Some(total) = usage.total_tokens {
        println!("\nTotal tokens: {}", total);
    }

    println!("\n");
    Ok(())
}

async fn example_streaming_completion() -> Result<()> {
    println!("Example 2: Streaming Chat via Cloudflare AI Gateway");
    println!("----------------------------------------------------\n");

    let model = std::env::var("CLOUDFLARE_AIG_MODEL")
        .map_err(|_| anyhow::anyhow!("set CLOUDFLARE_AIG_MODEL in provider/model format"))?;

    let agent_service = AgentBuilder::new()
        .provider(ProviderType::CloudflareAIGateway)
        .model(model.clone())
        .build()
        .await?;

    let agent: Agent = agent_service.into_agent();

    let query = "Write a short haiku about edge networks.";
    println!("Model: {}", model);
    println!("Query: {}\n", query);
    print!("Response: ");

    let mut stream = agent.stream_chat_with_tools(query);
    let mut total_tokens = 0u32;

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::TextChunk { content }) => {
                print!("{}", content);
                std::io::stdout().flush()?;
            }
            Ok(StreamEvent::Metadata { usage, .. }) => {
                if let Some(usage) = usage
                    && let Some(total) = usage.total_tokens
                {
                    total_tokens = total;
                }
            }
            Ok(StreamEvent::Done) => break,
            Ok(StreamEvent::LimitReached { .. }) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                break;
            }
        }
    }

    println!("\n\nTotal tokens: {}", total_tokens);
    println!("\n");
    Ok(())
}
