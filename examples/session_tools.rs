//! Session Tools Example (agent-to-agent)
//!
//! Demonstrates:
//! - AgentService with session-tools: sessions are registered in LiveSessionRegistry
//! - sessions_list: list active sessions (this service appears when running)
//! - sessions_send: send a message to another session (with optional reply_back)
//! - Optional SessionSpawner for sessions_spawn (custom spawner can be set via AgentBuilder::session_spawner)
//!
//! Run with:
//! ```bash
//! cargo run --example session_tools --features "session-tools,openai"
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::Result;
use caliche::prelude::ProviderType;
use caliche::{
    AgentService,
    agent::session::live_registry::{LiveSessionRegistry, SessionLiveStatus},
};

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== Session Tools Example ===\n");

    let mut service = AgentService::from_config(
        caliche::AgentServiceConfig::new()
            .with_provider(Some(ProviderType::OpenAI), Some("gpt-4o-mini".to_string()))
            .with_all_default_tools(),
    )
    .await?;

    let session_id = service.session().id.clone();
    println!("This agent's session_id: {}\n", session_id);

    let list = LiveSessionRegistry::global().list();
    println!("Live sessions (this agent is registered): {}", list.len());
    for s in &list {
        println!(
            "  - {} | model={} | status={:?}",
            s.session_id, s.model, s.status
        );
    }

    println!("\nChat: ask to list sessions (sessions_list tool):");
    let response = service
        .chat("Use the sessions_list tool to list all active sessions and tell me how many there are.")
        .await?;
    println!("Response: {}\n", response.text());

    let list_after = LiveSessionRegistry::global().list();
    let count = list_after
        .iter()
        .filter(|s| s.status == SessionLiveStatus::Active)
        .count();
    println!("Active sessions now: {}", count);

    println!("\nDone (AgentService drop will unregister this session).");
    Ok(())
}
