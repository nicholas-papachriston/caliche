//! AWS Bedrock Provider Example
//!
//! This example demonstrates AWS Bedrock streaming capabilities with the fully
//! type-safe streaming implementation.
//!
//! Best practices shown:
//! - Bedrock provider configuration with AWS credentials
//! - Real-time streaming with proper event handling
//! - Token usage tracking from Bedrock streaming events
//! - Error handling for AWS-specific errors
//! - Multiple Bedrock model examples
//! - Tool calls in streaming mode
//!
//! Prerequisites:
//! - AWS credentials configured (via environment variables or IAM role)
//! - AWS_REGION set (e.g., "us-east-1")
//! - BEDROCK_MODEL_ID set (optional, defaults to Claude Sonnet)
//!
//! Run with:
//! ```bash
//! export AWS_REGION=us-east-1
//! export AWS_ACCESS_KEY_ID=your_key
//! export AWS_SECRET_ACCESS_KEY=your_secret
//! cargo run --example bedrock --features bedrock
//! ```
//!
//! Or use bearer token authentication:
//! ```bash
//! export AWS_REGION=us-east-1
//! export AWS_BEARER_TOKEN_BEDROCK=your_token
//! cargo run --example bedrock --features bedrock
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::prelude::ProviderType;
use caliche::{Agent, AgentBuilder};
use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== AWS Bedrock Streaming Examples ===\n");

    // Example 1: Basic Bedrock streaming
    example_basic_bedrock_streaming().await?;

    // Example 2: Streaming with different Bedrock models
    example_bedrock_models().await?;

    // Example 3: Streaming with tools
    example_bedrock_streaming_with_tools().await?;

    // Example 4: Long-form content streaming
    example_long_form_streaming().await?;

    Ok(())
}

async fn example_basic_bedrock_streaming() -> Result<()> {
    println!("--- Example 1: Basic Bedrock Streaming ---");

    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent: Agent = agent_service.into_agent();

    println!("Query: Explain quantum computing in one sentence");
    println!("Response (streaming):\n");

    let mut stream = agent.stream_chat_with_tools("Explain quantum computing in one sentence");

    while let Some(event) = stream.next().await {
        match event {
            Ok(chunk) => match chunk {
                caliche::messages::StreamEvent::TextChunk { content } => {
                    print!("{}", content);
                    std::io::stdout().flush()?;
                }
                caliche::messages::StreamEvent::Metadata {
                    usage,
                    finish_reason,
                    metrics: _,
                } => {
                    if let Some(u) = usage {
                        println!("\n\n--- Token Usage ---");
                        if let Some(prompt) = u.prompt_tokens {
                            println!("Prompt tokens: {}", prompt);
                        }
                        if let Some(completion) = u.completion_tokens {
                            println!("Completion tokens: {}", completion);
                        }
                        if let Some(total) = u.total_tokens {
                            println!("Total tokens: {}", total);
                        }
                    }
                    if let Some(reason) = finish_reason {
                        println!("Finish reason: {}", reason);
                    }
                }
                caliche::messages::StreamEvent::Done => {
                    println!("\n[Stream complete]");
                    break;
                }
                caliche::messages::StreamEvent::ToolCallStart { .. } => {
                    // Tool calls not used in this example
                }
                caliche::messages::StreamEvent::ToolCallResult { .. } => {
                    // Tool results not used in this example
                }
                caliche::messages::StreamEvent::LimitReached {
                    max_iterations,
                    usage,
                } => {
                    println!("\n\nLimit reached: max_iterations={}", max_iterations);
                    if let Some(total) = usage.total_tokens {
                        println!("Total tokens used: {}", total);
                    }
                }
                caliche::messages::StreamEvent::ThinkingChunk { .. }
                | caliche::messages::StreamEvent::AudioChunk { .. }
                | caliche::messages::StreamEvent::ImageGenerated { .. }
                | caliche::messages::StreamEvent::AttachmentProcessed { .. }
                | caliche::messages::StreamEvent::Compaction { .. }
                | caliche::messages::StreamEvent::PhaseTransition { .. }
                | caliche::messages::StreamEvent::Status { .. }
                | caliche::messages::StreamEvent::AgentStart
                | caliche::messages::StreamEvent::AgentEnd
                | caliche::messages::StreamEvent::TurnStart
                | caliche::messages::StreamEvent::TurnEnd => {}
            },
            Err(e) => {
                println!("\n\nStream error: {}", e);
                break;
            }
        }
    }
    println!("\n");

    Ok(())
}

async fn example_bedrock_models() -> Result<()> {
    println!("--- Example 2: Different Bedrock Models ---");

    // Try different Bedrock models
    let models = vec!["us.anthropic.claude-haiku-4-5-20251001-v1:0"];

    for model_id in models {
        println!("\nModel: {}", model_id);

        let agent_service = AgentBuilder::new()
            .provider(ProviderType::Bedrock)
            .model(model_id.to_string())
            .build()
            .await?;

        let agent: Agent = agent_service.into_agent();

        println!("Query: Write a haiku about Rust programming");
        println!("Response:\n");

        let mut stream = agent.stream_chat_with_tools("Write a haiku about Rust programming");

        while let Some(event) = stream.next().await {
            match event {
                Ok(chunk) => match chunk {
                    caliche::messages::StreamEvent::TextChunk { content } => {
                        print!("{}", content);
                        std::io::stdout().flush()?;
                    }
                    caliche::messages::StreamEvent::Metadata { usage, .. } => {
                        if let Some(u) = usage {
                            if let Some(total) = u.total_tokens {
                                println!("\n[Tokens used: {}]", total);
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
                    }
                    caliche::messages::StreamEvent::Done => {
                        break;
                    }
                    _ => {}
                },
                Err(e) => {
                    println!("\nError with model {}: {}", model_id, e);
                    break;
                }
            }
        }
        println!("\n");
    }

    Ok(())
}

async fn example_bedrock_streaming_with_tools() -> Result<()> {
    println!("--- Example 3: Bedrock Streaming with Tools ---");

    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tool("calculator")
        .build()
        .await?;

    let agent: Agent = agent_service.into_agent();

    println!("Query: Calculate 123 * 456 and explain the steps");
    println!("Response (streaming with tool calls):\n");

    let mut stream = agent.stream_chat_with_tools("Calculate 123 * 456 and explain the steps");

    while let Some(event) = stream.next().await {
        match event {
            Ok(chunk) => match chunk {
                caliche::messages::StreamEvent::TextChunk { content } => {
                    print!("{}", content);
                    std::io::stdout().flush()?;
                }
                caliche::messages::StreamEvent::ToolCallStart {
                    name, arguments, ..
                } => {
                    println!("\n\n[Tool Call: {}]", name);
                    println!("Arguments: {}", arguments);
                }
                caliche::messages::StreamEvent::ToolCallResult { name, result, .. } => {
                    println!("\n[Tool Result ({}): {}]", name, result);
                }
                caliche::messages::StreamEvent::LimitReached {
                    max_iterations,
                    usage,
                } => {
                    println!("\n\nLimit reached: max_iterations={}", max_iterations);
                    if let Some(total) = usage.total_tokens {
                        println!("Total tokens used: {}", total);
                    }
                }
                caliche::messages::StreamEvent::Metadata { usage, .. } => {
                    if let Some(u) = usage {
                        println!("\n\n--- Final Token Usage ---");
                        if let Some(prompt) = u.prompt_tokens {
                            println!("Prompt tokens: {}", prompt);
                        }
                        if let Some(completion) = u.completion_tokens {
                            println!("Completion tokens: {}", completion);
                        }
                        if let Some(total) = u.total_tokens {
                            println!("Total tokens: {}", total);
                        }
                    }
                }
                caliche::messages::StreamEvent::Done => {
                    println!("\n[Stream complete]");
                    break;
                }
                caliche::messages::StreamEvent::ThinkingChunk { .. }
                | caliche::messages::StreamEvent::AudioChunk { .. }
                | caliche::messages::StreamEvent::ImageGenerated { .. }
                | caliche::messages::StreamEvent::AttachmentProcessed { .. }
                | caliche::messages::StreamEvent::Compaction { .. }
                | caliche::messages::StreamEvent::PhaseTransition { .. }
                | caliche::messages::StreamEvent::Status { .. }
                | caliche::messages::StreamEvent::AgentStart
                | caliche::messages::StreamEvent::AgentEnd
                | caliche::messages::StreamEvent::TurnStart
                | caliche::messages::StreamEvent::TurnEnd => {}
            },
            Err(e) => {
                println!("\n\nStream error: {}", e);
                break;
            }
        }
    }
    println!("\n");

    Ok(())
}

async fn example_long_form_streaming() -> Result<()> {
    println!("--- Example 4: Long-Form Content Streaming ---");

    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await?;

    let agent: Agent = agent_service.into_agent();

    println!("Query: Write a short story about an AI agent exploring a codebase");
    println!("Response (streaming):\n");

    let mut stream =
        agent.stream_chat_with_tools("Write a short story about an AI agent exploring a codebase");

    let mut chunk_count = 0;
    let start_time = std::time::Instant::now();

    while let Some(event) = stream.next().await {
        match event {
            Ok(chunk) => match chunk {
                caliche::messages::StreamEvent::TextChunk { content } => {
                    print!("{}", content);
                    std::io::stdout().flush()?;
                    chunk_count += 1;
                }
                caliche::messages::StreamEvent::Metadata { usage, .. } => {
                    let elapsed = start_time.elapsed();
                    println!("\n\n--- Streaming Statistics ---");
                    println!("Chunks received: {}", chunk_count);
                    println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
                    if let Some(u) = usage {
                        if let Some(total) = u.total_tokens {
                            println!("Total tokens: {}", total);
                            let tokens_per_sec = total as f64 / elapsed.as_secs_f64();
                            println!("Tokens/second: {:.2}", tokens_per_sec);
                        }
                    }
                }
                caliche::messages::StreamEvent::Done => {
                    println!("\n\n[Stream complete]");
                    break;
                }
                caliche::messages::StreamEvent::ToolCallStart { .. } => {
                    // Tool calls not used in this example
                }
                caliche::messages::StreamEvent::ToolCallResult { .. } => {
                    // Tool results not used in this example
                }
                caliche::messages::StreamEvent::LimitReached {
                    max_iterations,
                    usage,
                } => {
                    println!("\n\nLimit reached: max_iterations={}", max_iterations);
                    if let Some(total) = usage.total_tokens {
                        println!("Total tokens used: {}", total);
                    }
                }
                caliche::messages::StreamEvent::ThinkingChunk { .. }
                | caliche::messages::StreamEvent::AudioChunk { .. }
                | caliche::messages::StreamEvent::ImageGenerated { .. }
                | caliche::messages::StreamEvent::AttachmentProcessed { .. }
                | caliche::messages::StreamEvent::Compaction { .. }
                | caliche::messages::StreamEvent::PhaseTransition { .. }
                | caliche::messages::StreamEvent::Status { .. }
                | caliche::messages::StreamEvent::AgentStart
                | caliche::messages::StreamEvent::AgentEnd
                | caliche::messages::StreamEvent::TurnStart
                | caliche::messages::StreamEvent::TurnEnd => {}
            },
            Err(e) => {
                println!("\n\nStream error: {}", e);
                break;
            }
        }
    }
    println!("\n");

    Ok(())
}
