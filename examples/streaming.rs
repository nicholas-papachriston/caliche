//! Streaming Responses Example
//!
//! This example demonstrates how to use the Agent's streaming API directly
//! for real-time token-by-token output.
//!
//! Best practices shown:
//! - Proper stream event handling
//! - Real-time token display with flushing
//! - Tool call detection in streams
//! - Graceful error handling
//! - Token usage tracking
//!
//! Run with:
//! ```bash
//! cargo run --example streaming
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::prelude::ProviderType;
use caliche::{Agent, AgentBuilder};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // Example 1: Basic Streaming
    example_basic_streaming().await?;

    // Example 2: Streaming with Tool Calls
    example_streaming_with_tools().await?;

    Ok(())
}

async fn example_basic_streaming() -> Result<()> {
    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent: Agent = agent_service.into_agent();

    println!("Query: Write a haiku about programming");
    println!("Response (streaming):");

    let mut stream = agent.stream_chat_with_tools("Write a haiku about programming");

    while let Some(event) = stream.next().await {
        match event {
            Ok(chunk) => match chunk {
                caliche::messages::StreamEvent::TextChunk { content } => {
                    print!("{}", content);
                    use std::io::Write;
                    std::io::stdout().flush()?;
                }
                caliche::messages::StreamEvent::Metadata { usage, .. } => {
                    if let Some(u) = usage {
                        if let Some(total) = u.total_tokens {
                            println!("\n\nTokens: {}", total);
                        }
                    }
                }
                caliche::messages::StreamEvent::LimitReached {
                    max_iterations,
                    usage,
                } => {
                    println!("\n\nLimit reached: max_iterations={}", max_iterations);
                    if let Some(total) = usage.total_tokens {
                        println!("Total tokens used: {}", total);
                    }
                    break;
                }
                caliche::messages::StreamEvent::Done => {
                    break;
                }
                _ => {}
            },
            Err(e) => {
                println!("\nStream error: {}", e);
                break;
            }
        }
    }
    println!();

    Ok(())
}

async fn example_streaming_with_tools() -> Result<()> {
    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("calculator")
        .build()
        .await?;

    let agent: Agent = agent_service.into_agent();

    println!("\nQuery: What is 42 multiplied by 17?");
    println!("Response (streaming with tools):");

    let mut stream = agent.stream_chat_with_tools("What is 42 multiplied by 17?");

    while let Some(event) = stream.next().await {
        match event {
            Ok(chunk) => match chunk {
                caliche::messages::StreamEvent::TextChunk { content } => {
                    print!("{}", content);
                    use std::io::Write;
                    std::io::stdout().flush()?;
                }
                caliche::messages::StreamEvent::ToolCallStart {
                    name, arguments, ..
                } => {
                    println!("\n\nTool Call: {} with args: {}", name, arguments);
                }
                caliche::messages::StreamEvent::ToolCallResult { name, result, .. } => {
                    println!("\nTool Result ({}): {}", name, result);
                }
                caliche::messages::StreamEvent::Metadata { usage, .. } => {
                    if let Some(u) = usage {
                        if let Some(total) = u.total_tokens {
                            println!("\n\nTokens: {}", total);
                        }
                    }
                }
                caliche::messages::StreamEvent::LimitReached {
                    max_iterations,
                    usage,
                } => {
                    println!("\n\nLimit reached: max_iterations={}", max_iterations);
                    if let Some(total) = usage.total_tokens {
                        println!("Total tokens used: {}", total);
                    }
                    break;
                }
                caliche::messages::StreamEvent::Done => {
                    break;
                }
                _ => {}
            },
            Err(e) => {
                println!("\nStream error: {}", e);
                break;
            }
        }
    }
    println!();

    Ok(())
}
