//! Perplexity Provider Example
//!
//! This example demonstrates:
//! - Perplexity provider configuration
//! - Streaming with proper token usage tracking
//! - Tool usage with tool_choice parameter
//! - Different Perplexity models (sonar, sonar-pro)
//! - Error handling for Perplexity-specific errors
//! - Tool calling with forced tool usage
//!
//! Best practices shown:
//! - Proper API key configuration
//! - Token usage tracking from streaming events
//! - Tool choice configuration for forced tool usage
//! - Streaming event handling
//!
//! Prerequisites:
//! - PERPLEXITY_API_KEY environment variable set
//! - PERPLEXITY_MODEL set (optional, defaults to sonar)
//!
//! Run with:
//! ```bash
//! export PERPLEXITY_API_KEY=pplx-your-key
//! cargo run --example perplexity
//! ```

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::messages::{StreamEvent, ToolChoice};
use caliche::prelude::ProviderType;
use caliche::{Agent, AgentBuilder};
use futures_util::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Perplexity Provider Examples ===\n");

    // Example 1: Basic Perplexity agent
    example_basic_perplexity().await?;

    // Example 2: Streaming with token usage tracking
    example_streaming_with_tokens().await?;

    // Example 3: Tool usage with forced tool choice
    example_tool_usage().await?;

    println!("\n=== Examples Complete ===");

    Ok(())
}

/// Example 1: Basic Perplexity agent setup and chat
async fn example_basic_perplexity() -> Result<()> {
    println!("Example 1: Basic Perplexity Agent");
    println!("-----------------------------------\n");

    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Perplexity)
        .model("sonar".to_string())
        .build()
        .await?;

    let mut agent: Agent = agent_service.into_agent();

    let query = "Explain quantum computing in simple terms.";
    println!("Query: {}\n", query);

    let response = agent.chat(query).await?;
    let response_text = response.text();

    println!("Response: {}", response_text);

    let usage = response.usage();
    println!("\nToken Usage:");
    if let Some(prompt) = usage.prompt_tokens {
        println!("  Prompt tokens: {}", prompt);
    }
    if let Some(completion) = usage.completion_tokens {
        println!("  Completion tokens: {}", completion);
    }
    if let Some(total) = usage.total_tokens {
        println!("  Total tokens: {}", total);
    }

    println!("\n");
    Ok(())
}

/// Example 2: Streaming with proper token usage tracking
async fn example_streaming_with_tokens() -> Result<()> {
    println!("Example 2: Streaming with Token Usage Tracking");
    println!("-----------------------------------------------\n");

    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Perplexity)
        .model("sonar".to_string())
        .build()
        .await?;

    let agent: Agent = agent_service.into_agent();

    let query = "Write a haiku about programming.";
    println!("Query: {}\n", query);
    print!("Response: ");

    let mut stream = agent.stream_chat_with_tools(query);
    let mut full_response = String::new();
    let mut total_tokens = 0u32;
    let mut prompt_tokens = 0u32;
    let mut completion_tokens = 0u32;

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::TextChunk { content }) => {
                print!("{}", content);
                std::io::stdout().flush()?;
                full_response.push_str(&content);
            }
            Ok(StreamEvent::Metadata { usage, .. }) => {
                if let Some(usage) = usage {
                    if let Some(total) = usage.total_tokens {
                        total_tokens = total;
                    }
                    if let Some(prompt) = usage.prompt_tokens {
                        prompt_tokens = prompt;
                    }
                    if let Some(completion) = usage.completion_tokens {
                        completion_tokens = completion;
                    }
                }
            }
            Ok(StreamEvent::Done) => {
                break;
            }
            Ok(StreamEvent::ToolCallStart { .. }) => {
                // Tool calls are handled separately
            }
            Ok(StreamEvent::ToolCallResult { .. }) => {
                // Tool results are handled separately
            }
            Ok(StreamEvent::LimitReached {
                max_iterations,
                usage,
            }) => {
                eprintln!("\n\nLimit reached: max_iterations={}", max_iterations);
                if let Some(total) = usage.total_tokens {
                    println!("Total tokens used: {}", total);
                }
                break;
            }
            Ok(StreamEvent::ThinkingChunk { .. })
            | Ok(StreamEvent::AudioChunk { .. })
            | Ok(StreamEvent::ImageGenerated { .. })
            | Ok(StreamEvent::AttachmentProcessed { .. })
            | Ok(StreamEvent::Compaction { .. })
            | Ok(StreamEvent::PhaseTransition { .. })
            | Ok(StreamEvent::Status { .. })
            | Ok(StreamEvent::AgentStart)
            | Ok(StreamEvent::AgentEnd)
            | Ok(StreamEvent::TurnStart)
            | Ok(StreamEvent::TurnEnd) => {}
            Err(e) => {
                eprintln!("\nStream error: {}", e);
                break;
            }
        }
    }

    println!("\n\nToken Usage:");
    println!("  Prompt tokens: {}", prompt_tokens);
    println!("  Completion tokens: {}", completion_tokens);
    println!("  Total tokens: {}", total_tokens);

    println!("\n");
    Ok(())
}

/// Example 3: Tool usage with forced tool choice
async fn example_tool_usage() -> Result<()> {
    println!("Example 3: Tool Usage with Forced Tool Choice");
    println!("----------------------------------------------\n");

    let agent_service = AgentBuilder::new()
        .provider(ProviderType::Perplexity)
        .model("sonar".to_string())
        .enable_tool("calculator")
        .enable_tool("fs")
        .tool_choice(Some(ToolChoice::Specific(beef::lean::Cow::borrowed(
            "calculator",
        )))) // Force specific tool
        .system_prompt(
            "You are a helpful assistant that uses tools to solve problems. \
             When given a math problem, you MUST use the calculator tool. \
             The calculator tool accepts 'a', 'b', and 'operation' parameters. \
             For '15 * 23 + 42', first calculate 15 * 23 using the calculator, then add 42.",
        )
        .build()
        .await?;

    // Verify tools are registered
    let tools = agent_service.list_tools();
    println!(
        "Registered tools: {:?}",
        tools.iter().map(|(name, _, _)| name).collect::<Vec<_>>()
    );

    let agent: Agent = agent_service.into_agent();

    let query = "Calculate 15 * 23 + 42. Show your work.";
    println!("Query: {}\n", query);
    print!("Response: ");

    let mut stream = agent.stream_chat_with_tools(query);
    let mut tool_calls = 0;
    let mut tool_results = 0;
    let mut full_response = String::new();
    let mut has_error = false;

    while let Some(event) = stream.next().await {
        match event {
            Ok(StreamEvent::TextChunk { content }) => {
                print!("{}", content);
                std::io::stdout().flush()?;
                full_response.push_str(&content);
            }
            Ok(StreamEvent::ToolCallStart {
                name, arguments, ..
            }) => {
                tool_calls += 1;
                println!("\n\n[Tool Call #{}]", tool_calls);
                println!("  Tool: {}", name);
                println!("  Arguments: {}", arguments);
            }
            Ok(StreamEvent::ToolCallResult { name, result, .. }) => {
                tool_results += 1;
                println!("\n[Tool Result #{}]", tool_results);
                println!("  Tool: {}", name);
                println!("  Result: {}", result);
            }
            Ok(StreamEvent::Metadata {
                usage,
                finish_reason,
                ..
            }) => {
                if let Some(usage) = usage {
                    if let Some(total) = usage.total_tokens {
                        println!("\n\nToken Usage: {} tokens", total);
                    }
                }
                if let Some(reason) = finish_reason {
                    println!("\nFinish reason: {}", reason);
                    if reason.contains("error") || reason == "tool_use" {
                        has_error = reason.contains("error");
                        if reason == "tool_use" {
                            println!(
                                "Note: Model chose to use tools but no tool calls were received"
                            );
                        } else {
                            eprintln!("Error in finish_reason: {}", reason);
                        }
                    }
                }
            }
            Ok(StreamEvent::LimitReached {
                max_iterations,
                usage,
            }) => {
                eprintln!("\n\nLimit reached: max_iterations={}", max_iterations);
                if let Some(total) = usage.total_tokens {
                    println!("Total tokens used: {}", total);
                }
                break;
            }
            Ok(StreamEvent::Done) => {
                break;
            }
            Ok(StreamEvent::ThinkingChunk { .. })
            | Ok(StreamEvent::AudioChunk { .. })
            | Ok(StreamEvent::ImageGenerated { .. })
            | Ok(StreamEvent::AttachmentProcessed { .. })
            | Ok(StreamEvent::Compaction { .. })
            | Ok(StreamEvent::PhaseTransition { .. })
            | Ok(StreamEvent::Status { .. })
            | Ok(StreamEvent::AgentStart)
            | Ok(StreamEvent::AgentEnd)
            | Ok(StreamEvent::TurnStart)
            | Ok(StreamEvent::TurnEnd) => {}
            Err(e) => {
                has_error = true;
                eprintln!("\nStream error: {}", e);
                eprintln!("Error details: {:?}", e);
                break;
            }
        }
    }

    if full_response.is_empty() && !has_error {
        println!("\n(No text response received)");
    }

    println!("\n\nTool Usage Summary:");
    println!("  Tool calls: {}", tool_calls);
    println!("  Tool results: {}", tool_results);

    if tool_calls == 0 && full_response.is_empty() {
        println!("\nWarning: No tool calls and no text response received.");
        println!(
            "   This may indicate the model couldn't decide which tool to use with Required tool_choice."
        );
    }

    println!("\n");
    Ok(())
}
