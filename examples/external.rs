//! External Provider Example
//!
//! This example demonstrates:
//! - External provider configuration with callbacks
//! - Streaming with proper token usage tracking
//! - Tool usage with tool_choice parameter
//! - Custom callback implementation for external LLM integration
//! - Error handling for external provider errors
//! - Tool calling with forced tool usage
//!
//! Best practices shown:
//! - Implementing ExternalCompletionCallback trait
//! - Token usage tracking from streaming events
//! - Tool choice configuration for forced tool usage
//! - Streaming event handling
//!
//! Note: External provider is typically used from Python/JS bindings to integrate
//! with LangChain or other external LLM libraries. This example shows how to
//! implement it in Rust for demonstration purposes.
//!
//! Run with:
//! ```bash
//! cargo run --example external
//! ```

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use async_stream::stream;
use async_trait::async_trait;
use caliche::Agent;
use caliche::agent::service::ProviderFactory;
use caliche::agent::service::ToolRegistryBuilder;
use caliche::llm::external::ExternalCompletionCallback;
use caliche::messages::{
    CompletionContent, CompletionRequest, CompletionResponse, StreamEvent, TokenUsage,
};
use caliche::prelude::ProviderType;
use futures_util::{Stream, StreamExt};
use std::io::Write;
use std::pin::Pin;
use std::sync::Arc;

// Mock callback implementation for demonstration
struct MockExternalCallback;

#[async_trait]
impl ExternalCompletionCallback for MockExternalCallback {
    async fn complete(&self, request: CompletionRequest) -> anyhow::Result<CompletionResponse> {
        // Mock implementation - in real usage, this would call an external LLM
        let response_text = format!(
            "Mock response to: {}. This is a demonstration of the External provider.",
            request
                .messages
                .last()
                .and_then(|m| m.content.as_text())
                .unwrap_or("query")
        );

        Ok(CompletionResponse {
            content: CompletionContent::Text {
                content: response_text,
            },
            usage: Some(TokenUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(20),
                total_tokens: Some(30),
                embedding_tokens: None,
                context_window: None,
                cache_creation_tokens: None,
                cache_read_tokens: None,
                last_prompt_tokens: None,
            }),
            finish_reason: Some(beef::lean::Cow::borrowed("stop")),
            reasoning_content: None,
        })
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamEvent>> + Send + '_>> {
        // Mock streaming implementation
        let query_text = request
            .messages
            .last()
            .and_then(|m| m.content.as_text())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "query".to_string());

        let words: Vec<String> = query_text
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let word_count = words.len() as u32;

        let stream = stream! {
                    // Stream text chunks
                    for word in words {
                        yield Ok(StreamEvent::TextChunk {
                            content: format!("{} ", word),
                        });
                    }

                    // Send metadata
                    yield Ok(StreamEvent::Metadata {
                        usage: Some(TokenUsage {
                            prompt_tokens: Some(10),
                            completion_tokens: Some(word_count),
                            total_tokens: Some(10 + word_count),
                            embedding_tokens: None,
                            context_window: None,
                            cache_creation_tokens: None,
                            cache_read_tokens: None,
                            last_prompt_tokens: None,
        }),
                        finish_reason: Some(beef::lean::Cow::borrowed("stop")),
                        metrics: None,
                    });

                    // Signal completion
                    yield Ok(StreamEvent::Done);
                };

        Box::pin(stream)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== External Provider Examples ===\n");

    // Example 1: Basic External agent
    example_basic_external().await?;

    // Example 2: Streaming with token usage tracking
    example_streaming_with_tokens().await?;

    // Example 3: Tool usage with forced tool choice
    example_tool_usage().await?;

    println!("\n=== Examples Complete ===");

    Ok(())
}

/// Example 1: Basic External agent setup and chat
async fn example_basic_external() -> Result<()> {
    println!("Example 1: Basic External Agent");
    println!("-----------------------------------\n");

    println!("Note: External provider requires callbacks and cannot be used directly");
    println!("with AgentBuilder. It's designed for Python/JS bindings to integrate");
    println!("with LangChain or other external LLM libraries.\n");

    // Create external provider with mock callback using ProviderFactory
    let callback = Arc::new(MockExternalCallback);
    let external_provider = ProviderFactory::create_with_external_callback(
        "external-mock",
        "external-mock".to_string(),
        callback,
        true, // supports_native_tools
        true, // supports_streaming
    );

    // Create tool registry
    let tool_registry = ToolRegistryBuilder::new(ProviderType::External)
        .build()
        .await?;

    // Create agent directly with the external provider
    let mut agent = Agent::new(external_provider, tool_registry);

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

    // Create external provider with mock callback
    let callback = Arc::new(MockExternalCallback);
    let external_provider = ProviderFactory::create_with_external_callback(
        "external-mock",
        "external-mock".to_string(),
        callback,
        true,
        true,
    );

    // Create tool registry
    let tool_registry = ToolRegistryBuilder::new(ProviderType::External)
        .build()
        .await?;

    let agent = Agent::new(external_provider, tool_registry);

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

    // Create external provider with mock callback
    let callback = Arc::new(MockExternalCallback);
    let external_provider = ProviderFactory::create_with_external_callback(
        "external-mock",
        "external-mock".to_string(),
        callback,
        true,
        true,
    );

    // Create tool registry with specific tools
    let tool_registry = ToolRegistryBuilder::new(ProviderType::External)
        .with_tool_names(Some(vec!["calculator".to_string(), "fs".to_string()]))
        .build()
        .await?;

    let agent = Agent::new(external_provider, tool_registry);

    println!("Registered tools: calculator, echo");

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
