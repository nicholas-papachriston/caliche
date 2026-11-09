//! AgentFS Integration Example
//!
//! Demonstrates:
//! - AgentFS for session storage (single SQLite file per agent)
//! - Tool call audit trail (persistent, queryable)
//! - Session snapshot/restore via database file copy
//!
//! Run with:
//! ```bash
//! cargo run --example agentfs --features "agentfs,openai"
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::AgentBuilder;
use caliche::prelude::ProviderType;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== AgentFS Integration Example ===\n");

    let mut service = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o-mini".to_string())
        .with_agentfs("example-agent")
        .build()
        .await?;

    println!("AgentFS enabled: session and tool calls stored in .agentfs/example-agent.db\n");

    let response = service.chat("What is 2 + 2? Use the calculator.").await?;
    println!("Response: {}\n", response.text());

    println!("Session saved to AgentFS. Tool calls are audited in the database.");
    println!("Snapshot: cp .agentfs/example-agent.db example-agent-snapshot.db");

    Ok(())
}
