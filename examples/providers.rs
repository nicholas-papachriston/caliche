//! Provider Examples - Grouped by Provider
//!
//! This example demonstrates all providers with comprehensive examples:
//! - Streaming generation with tools
//! - Non-streaming generation with tools
//! - Streaming generation without tools
//! - Non-streaming generation without tools
//!
//! Each provider section shows all 4 scenarios before moving to the next provider.
//!
//! Run with:
//! ```bash
//! cargo run --example providers --features providers-all
//! ```
//!
//! For detailed validation output:
//! ```bash
//! DEBUG=debug cargo run --example providers --features providers-all >> providers.txt
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::capabilities::providers::is_provider_available;
use caliche::messages::StreamEvent;
use caliche::output::ResponseFormat;
use caliche::prelude::ProviderType;
use caliche::{Agent, AgentBuilder};
use futures_util::StreamExt;
use std::time::Instant;

#[derive(Debug, Clone)]
struct ProviderConfig {
    provider: ProviderType,
    model: String,
    name: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // Initialize tracing for logging
    tracing_subscriber::fmt::init();

    println!("=== Provider Examples - Grouped by Provider ===\n");
    println!("Each provider demonstrates 4 scenarios:");
    println!("  1. Streaming generation with tools");
    println!("  2. Non-streaming generation with tools");
    println!("  3. Streaming generation without tools");
    println!("  4. Non-streaming generation without tools\n");

    let providers = get_provider_configs();

    for config in providers {
        println!("{}", "=".repeat(70));
        println!("Provider: {} ({})", config.name, config.model);
        println!("{}", "=".repeat(70));
        println!();

        // Example 1: Streaming with tools
        println!(
            "1. Streaming Generation with Tools: {} ({})",
            config.provider, config.model
        );
        println!("{}", "-".repeat(70));
        if let Err(e) = example_streaming_with_tools(&config).await {
            println!("  Failed: {}\n", e);
        }
        println!();

        // Example 2: Non-streaming with tools
        println!(
            "2. Non-Streaming Generation with Tools: {} ({})",
            config.provider, config.model
        );
        println!("{}", "-".repeat(70));
        if let Err(e) = example_non_streaming_with_tools(&config).await {
            println!("  Failed: {}\n", e);
        }
        println!();

        // Example 3: Streaming without tools
        println!(
            "3. Streaming Generation without Tools: {} ({})",
            config.provider, config.model
        );
        println!("{}", "-".repeat(70));
        if let Err(e) = example_streaming_without_tools(&config).await {
            println!("  Failed: {}\n", e);
        }
        println!();

        // Example 4: Non-streaming without tools
        println!(
            "4. Non-Streaming Generation without Tools: {} ({})",
            config.provider, config.model
        );
        println!("{}", "-".repeat(70));
        if let Err(e) = example_non_streaming_without_tools(&config).await {
            println!("  Failed: {}\n", e);
        }
        println!();
    }

    println!("{}", "=".repeat(70));
    println!("All examples complete!");
    println!("\nNote: Some examples may fail due to:");
    println!("  - Missing API keys");
    println!("  - Network issues");
    println!("  - Provider-specific configuration requirements");
    println!("  - Model availability");

    Ok(())
}

fn get_provider_configs() -> Vec<ProviderConfig> {
    let mut configs = Vec::new();

    // Check each provider and add default model if available
    let provider_types = vec![
        (ProviderType::OpenAI, "gpt-4o-mini"),
        (ProviderType::Anthropic, "claude-sonnet-4-5-20250929"),
        (
            ProviderType::Bedrock,
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
        (ProviderType::OpenRouter, "openai/gpt-4o-mini"),
        (ProviderType::Perplexity, "sonar"),
        (ProviderType::LMStudio, "openai/gpt-oss-20b"),
        (ProviderType::Google, "models/gemini-2.5-flash"),
        (ProviderType::Cohere, "command-r-plus"),
        (ProviderType::AzureOpenAI, "gpt-4o-mini"),
    ];

    for (provider, default_model) in provider_types {
        if is_provider_available(provider) {
            configs.push(ProviderConfig {
                provider,
                model: default_model.to_string(),
                name: format!("{}", provider),
            });
        } else {
            println!(
                "Skipping {} (not available - feature not enabled)",
                provider
            );
        }
    }

    configs
}

async fn example_streaming_with_tools(config: &ProviderConfig) -> Result<()> {
    let start = Instant::now();
    let query = "Calculate 15 * 23 and then echo back: 'Hello from streaming with tools!'";

    let mut builder = AgentBuilder::new()
        .provider(config.provider)
        .model(config.model.clone())
        .enable_tool("calculator")
        .enable_tool("fs")
        .system_prompt("You are a helpful assistant. When asked to calculate, use the calculator tool. When asked to echo, use the echo tool.");

    // Perplexity requires json_object response format for tool calling
    if config.provider == ProviderType::Perplexity {
        builder = builder.response_format(ResponseFormat::JsonObject);
    }

    let agent_service = builder.build().await?;
    let agent: Agent = agent_service.into_agent();

    let mut stream = agent.stream_chat_with_tools(query);
    let mut streamed_chunks = 0;
    let mut tool_calls = 0;
    let mut full_response = String::new();

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextChunk { content } => {
                streamed_chunks += 1;
                full_response.push_str(&content);
                print!("{}", content);
            }
            StreamEvent::ToolCallStart {
                name, arguments, ..
            } => {
                tool_calls += 1;
                println!("\n  [Tool Call: {} with args: {}]", name, arguments);
            }
            StreamEvent::ToolCallResult { name, result, .. } => {
                println!("  [Tool Result: {} -> {}]", name, result);
            }
            StreamEvent::Metadata { usage, .. } => {
                if let Some(usage) = usage {
                    if let Some(total) = usage.total_tokens {
                        println!("\n  [Tokens: {}]", total);
                    }
                }
            }
            StreamEvent::LimitReached {
                max_iterations,
                usage,
            } => {
                println!("\n  [Limit reached: max_iterations={}]", max_iterations);
                if let Some(total) = usage.total_tokens {
                    println!("  [Total tokens: {}]", total);
                }
                break;
            }
            StreamEvent::Done => break,
            StreamEvent::ThinkingChunk { .. }
            | StreamEvent::AudioChunk { .. }
            | StreamEvent::ImageGenerated { .. }
            | StreamEvent::AttachmentProcessed { .. }
            | StreamEvent::Compaction { .. }
            | StreamEvent::PhaseTransition { .. }
            | StreamEvent::Status { .. }
            | StreamEvent::AgentStart
            | StreamEvent::AgentEnd
            | StreamEvent::TurnStart
            | StreamEvent::TurnEnd => {}
        }
    }

    let duration_ms = start.elapsed().as_millis();
    println!(
        "\n  Completed in {}ms ({} chunks, {} tool calls)",
        duration_ms, streamed_chunks, tool_calls
    );
    Ok(())
}

async fn example_non_streaming_with_tools(config: &ProviderConfig) -> Result<()> {
    let start = Instant::now();
    let query = "Calculate 42 * 7 and then echo back: 'Hello from non-streaming with tools!'";

    let mut builder = AgentBuilder::new()
        .provider(config.provider)
        .model(config.model.clone())
        .enable_tool("calculator")
        .enable_tool("fs")
        .system_prompt("You are a helpful assistant. When asked to calculate, use the calculator tool. When asked to echo, use the echo tool.");

    // Perplexity requires json_object response format for tool calling
    if config.provider == ProviderType::Perplexity {
        builder = builder.response_format(ResponseFormat::JsonObject);
    }

    let agent_service = builder.build().await?;
    let mut agent: Agent = agent_service.into_agent();

    let response = agent.chat(query).await?;
    let duration_ms = start.elapsed().as_millis();

    println!("  Response: {}", response.text());
    if let Some(total) = response.usage().total_tokens {
        println!("  Tokens: {}", total);
    }
    println!("  Completed in {}ms", duration_ms);
    Ok(())
}

async fn example_streaming_without_tools(config: &ProviderConfig) -> Result<()> {
    let start = Instant::now();
    let query = "Explain the concept of recursion in programming in 2-3 sentences.";

    let builder = AgentBuilder::new()
        .provider(config.provider)
        .model(config.model.clone());

    let agent_service = builder.build().await?;
    let agent: Agent = agent_service.into_agent();

    let mut stream = agent.stream_chat_with_tools(query);
    let mut streamed_chunks = 0;
    let mut full_response = String::new();

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::TextChunk { content } => {
                streamed_chunks += 1;
                full_response.push_str(&content);
                print!("{}", content);
            }
            StreamEvent::Metadata { usage, .. } => {
                if let Some(usage) = usage {
                    if let Some(total) = usage.total_tokens {
                        println!("\n  [Tokens: {}]", total);
                    }
                }
            }
            StreamEvent::Done => break,
            _ => {}
        }
    }

    let duration_ms = start.elapsed().as_millis();
    println!(
        "\n  Completed in {}ms ({} chunks)",
        duration_ms, streamed_chunks
    );
    Ok(())
}

async fn example_non_streaming_without_tools(config: &ProviderConfig) -> Result<()> {
    let start = Instant::now();
    let query = "What are the key principles of object-oriented programming?";

    let builder = AgentBuilder::new()
        .provider(config.provider)
        .model(config.model.clone());

    let agent_service = builder.build().await?;
    let mut agent: Agent = agent_service.into_agent();

    let response = agent.chat(query).await?;
    let duration_ms = start.elapsed().as_millis();

    println!("  Response: {}", response.text());
    if let Some(total) = response.usage().total_tokens {
        println!("  Tokens: {}", total);
    }
    println!("  Completed in {}ms", duration_ms);
    Ok(())
}
