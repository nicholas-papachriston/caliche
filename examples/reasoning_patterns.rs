//! Reasoning Patterns Example
//!
//! This example demonstrates:
//! - ReAct (Reason + Act) reasoning pattern
//! - Thought-Action-Observation loop
//! - Multi-step problem solving with tools
//!
//! Best practices shown:
//! - Iterative reasoning with tool use
//! - Step-by-step problem decomposition
//!
//! Run with:
//! ```bash
//! cargo run --example reasoning_patterns --features reasoning-patterns
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::prelude::ProviderType;
use caliche::{
    AgentBuilder,
    reasoning::{ReasoningPattern, react::ReActPattern},
};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    example_react().await?;

    Ok(())
}

async fn example_react() -> Result<()> {
    let agent = AgentBuilder::new()
        .provider(ProviderType::Bedrock)
        .model("us.anthropic.claude-haiku-4-5-20251001-v1:0".to_string())
        .system_prompt("You are a helpful AI assistant with access to tools.".to_string())
        .enable_tool("calculator")
        .enable_tool("http")
        .build()
        .await?;

    let react = ReActPattern::new(agent, 5);

    let query = "What is 15 multiplied by 23, and then add 42 to that result?";
    println!("Query: {}", query);

    let result = react.execute(query).await?;

    println!("\nReasoning Steps:");
    for (i, step) in result.steps.iter().enumerate() {
        match step.step_type {
            caliche::reasoning::StepType::Thought => {
                println!("  {}. Thought: {}", i + 1, step.content);
            }
            caliche::reasoning::StepType::Action => {
                println!("  {}. Action: {}", i + 1, step.content);
            }
            caliche::reasoning::StepType::Observation => {
                println!("  {}. Observation: {}", i + 1, step.content);
            }
            caliche::reasoning::StepType::Answer => {
                println!("  {}. Final Answer: {}", i + 1, step.content);
            }
            _ => {}
        }
    }

    println!("\nFinal Answer: {}", result.final_answer);

    if let Some(usage) = result.token_usage {
        if let Some(total) = usage.total_tokens {
            println!("Tokens: {}", total);
        }
    }

    Ok(())
}
