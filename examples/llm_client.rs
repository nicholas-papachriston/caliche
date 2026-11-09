//! Non-Agentic LLM Client Example
//!
//! This example demonstrates:
//! - Using LlmClient for simple, single-shot LLM completions
//! - Non-streaming completions with messages
//! - Streaming completions
//! - Convenience methods for text-only prompts
//! - Builder pattern configuration
//!
//! Best practices shown:
//! - Direct LLM access without agent overhead
//! - Simple prompt/response workflows
//! - Streaming for real-time responses
//!
//! Run with:
//! ```bash
//! cargo run --example llm_client
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use beef::lean::Cow;
use caliche::messages::ParametersSchema;
use caliche::prelude::{LlmClient, Message, ProviderType, ToolChoice, ToolSchema};
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Non-Agentic LLM Client Examples ===\n");

    // Example 1: Basic usage with builder pattern
    println!("Example 1: Basic Non-Streaming Completion");
    let client = LlmClient::builder()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o".to_string())
        .temperature(0.7)
        .max_tokens(100)
        .build()
        .await?;

    let response = client
        .complete_text("What is 2+2? Answer in one sentence.")
        .await?;
    println!("  Response: {}\n", response);

    // Example 2: Completion with messages
    println!("Example 2: Completion with Message History");
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What is the capital of France?"),
    ];
    let response = client.complete(messages).await?;
    println!("  Response: {}", response.content.extract_text());
    if let Some(usage) = response.usage {
        println!("  Tokens used: {:?}\n", usage);
    }

    // Example 3: Streaming completion
    println!("Example 3: Streaming Completion");
    let mut stream =
        client.stream_text("Tell me a short story about a robot in exactly 3 sentences.");
    println!("  Streaming response: ");
    while let Some(event) = stream.next().await {
        match event? {
            caliche::messages::StreamEvent::TextChunk { content } => {
                print!("{}", content);
            }
            caliche::messages::StreamEvent::Done => {
                println!("\n  Stream completed\n");
                break;
            }
            _ => {}
        }
    }

    // Example 4: Streaming with messages
    println!("Example 4: Streaming with Message History");
    let messages = vec![
        Message::system("You are a creative writing assistant."),
        Message::user("Write a haiku about programming."),
    ];
    let mut stream = client.stream(messages);
    println!("  Streaming response: ");
    while let Some(event) = stream.next().await {
        match event? {
            caliche::messages::StreamEvent::TextChunk { content } => {
                print!("{}", content);
            }
            caliche::messages::StreamEvent::Done => {
                println!("\n  Stream completed\n");
                break;
            }
            _ => {}
        }
    }

    // Example 5: Different provider (Anthropic)
    println!("Example 5: Using Anthropic Provider");
    if let Ok(anthropic_client) = LlmClient::builder()
        .provider(ProviderType::Anthropic)
        .model("claude-haiku-4-5-20251001".to_string())
        .build()
        .await
    {
        println!("  Anthropic client created");
        let response = anthropic_client
            .complete_text("Say hello in one sentence.")
            .await?;
        println!("  Response: {}\n", response);
    } else {
        println!("  (Skipped: Anthropic credentials not configured)\n");
    }

    // Example 6: Custom timeout and retries
    println!("Example 6: Custom Configuration");
    let custom_client = LlmClient::builder()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o".to_string())
        .timeout(std::time::Duration::from_secs(30))
        .max_retries(3)
        .temperature(0.5)
        .build()
        .await?;
    println!("  Client created with custom timeout and retries");
    let response = custom_client
        .complete_text("What is the meaning of life? Answer briefly.")
        .await?;
    println!("  Response: {}\n", response);

    // Example 7: Using tools (function calling)
    println!("Example 7: Using Tools (Function Calling)");
    let tools = vec![ToolSchema {
        name: Cow::borrowed("get_weather"),
        description: Cow::borrowed("Get the current weather in a location"),
        parameters: ParametersSchema {
            schema_type: Cow::borrowed("object"),
            properties: serde_json::json!({
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature"
                }
            }),
            required: vec![beef::lean::Cow::borrowed("location")],
        },
    }];
    let tool_client = LlmClient::builder()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o".to_string())
        .tools(tools)
        .tool_choice(ToolChoice::Auto)
        .build()
        .await?;

    let messages = vec![Message::user("What's the weather like in San Francisco?")];
    let response = tool_client.complete(messages).await?;
    println!("  Response: {}", response.content.extract_text());

    // Check if there are tool calls in the response
    if response.content.has_tool_calls() {
        println!("  Tool calls detected:");
        for tool_call in response.content.as_tool_calls() {
            println!("    - {}: {}", tool_call.name, tool_call.arguments);
        }
        println!("  (Note: Tool calls are returned but not executed in this example)\n");
    } else {
        println!();
    }

    println!("=== All Examples Complete ===");
    Ok(())
}
