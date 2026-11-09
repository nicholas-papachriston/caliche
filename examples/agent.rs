//! Agent Creation and Configuration Example
//!
//! This example demonstrates:
//! - Creating agents with explicit configuration
//! - Runtime configuration with a single builder path
//! - Custom system prompts for specialized agents
//! - Explicit tool selection
//! - Multiple chat turns with context
//! - Shared state management
//!
//! Best practices shown:
//! - Proper error handling
//! - Token usage tracking
//! - Consistent builder-first configuration
//! - Type-safe agent creation
//!
//! Run with:
//! ```bash
//! cargo run --example agent --features=full
//! ```
//!
//! Set environment variables and consume them through explicit builder configuration:
//! ```bash
//! export OPENAI_API_KEY=...
//! cargo run --example agent --features=full
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::{Context, Result};
use caliche::prelude::ProviderType;
use caliche::{AgentBuilder, SharedState, agent::hooks::permission::PermissionMode};
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    // Example 1: Explicit configuration (recommended for production)
    println!("=== Example 1: Explicit Configuration ===");
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .build()
        .await
        .context("Failed to create agent. Check your credentials.")?;

    // Example 2: Chat with the agent
    let response = agent
        .chat("What is 15 * 23?")
        .await
        .context("Chat request failed")?;

    println!("Query: What is 15 * 23?");
    println!("Response: {}", response.text());
    if let Some(total) = response.usage().total_tokens {
        println!("Tokens: {}", total);
    }

    // Example 3: Multiple chat turns (maintains context)
    let response = agent
        .chat("Remember this number: 42. What is that number plus 10?")
        .await?;
    println!("\nQuery: Remember this number: 42. What is that number plus 10?");
    println!("Response: {}", response.text());

    let response = agent
        .chat("What was the number I asked you to remember?")
        .await?;
    println!("\nQuery: What was the number I asked you to remember?");
    println!("Response: {}", response.text());

    // Example 4: Runtime configuration through canonical builder path
    println!("\n=== Example 4: Runtime Configuration ===");
    let mut configured_agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .enable_tools(&["calculator", "fs"])
        .build()
        .await?;

    let response = configured_agent
        .chat("Echo this: Hello from builder-first runtime config!")
        .await?;
    println!("Query: Echo this: Hello from builder-first runtime config!");
    println!("Response: {}", response.text());

    // Example 5: Builder with custom system prompt
    println!("\n=== Example 5: Custom System Prompt ===");
    let mut coding_agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .system_prompt(
            r#"You are a Rust programming expert. 
When answering questions, focus on Rust best practices, 
memory safety, and performance considerations."#
                .to_string(),
        )
        .enable_tools(&["fs", "fs"])
        .build()
        .await?;

    let response = coding_agent
        .chat("What are the key principles of Rust memory safety?")
        .await?;
    println!("Query: What are the key principles of Rust memory safety?");
    println!("Response: {}", response.text());

    // Example 6: Agent with all default tools
    println!("\n=== Example 6: All Default Tools ===");
    let full_agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .enable_all_default_tools()
        .build()
        .await?;

    println!("Available tools:");
    for (name, _, _) in full_agent.list_tools() {
        println!("  - {}", name);
    }

    // Example 7: Permission modes and allowed tools
    println!("\n=== Example 7: Permission Modes and Allowed Tools ===");
    let mut restricted_agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .allowed_tools(&["calculator", "fs"])
        .permission_mode(PermissionMode::AcceptEdits)
        .build()
        .await?;

    println!("Agent configured with:");
    println!("  - Allowed tools: [calculator, echo]");
    println!("  - Permission mode: AcceptEdits (auto-accept file edits)");

    let response = restricted_agent.chat("What is 5 + 3?").await?;
    println!("Query: What is 5 + 3?");
    println!("Response: {}", response.text());

    // Example 8: Shared state management
    println!("\n=== Example 8: Shared State Management ===");
    let shared_state = Arc::new(SharedState::new());

    // Initialize state with some data
    shared_state.set("/order_id", json!("ORD-12345")).await?;
    shared_state.set("/customer/name", json!("Alice")).await?;
    shared_state
        .set("/customer/email", json!("alice@example.com"))
        .await?;

    println!("Initial state:");
    println!(
        "{}",
        serde_json::to_string_pretty(&shared_state.get_all().await)?
    );

    // Update state
    println!("\nUpdating order status...");
    shared_state
        .set("/order/status", json!("processing"))
        .await?;

    println!("Updated state:");
    println!(
        "{}",
        serde_json::to_string_pretty(&shared_state.get_all().await)?
    );

    // Demonstrate nested paths
    println!("\nSetting nested value...");
    shared_state
        .set("/order/items/0/name", json!("Widget"))
        .await?;
    shared_state
        .set("/order/items/0/quantity", json!(2))
        .await?;

    let item_name = shared_state.get("/order/items/0/name").await?;
    println!("Retrieved item name: {:?}", item_name);

    println!("\n=== Examples Complete ===");
    println!("\nTips:");
    println!("- Use one canonical path: AgentBuilder::new() + fluent config + build()");
    println!("- Keep provider/model explicit in application code");
    println!("- Shared state enables agents to coordinate and share data");
    println!("- Use permission modes and allowed_tools for security");
    println!(
        "- See examples/agent_controls.rs for lifecycle hooks, MCP servers, and session forking"
    );

    Ok(())
}
