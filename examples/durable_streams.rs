//! Durable Streams Example
//!
//! This example shows how to configure an agent with a durable stream backend
//! so that events can be persisted for replay and resume (e.g. when using the API
//! or a harness with SSE and Last-Event-ID).
//!
//! Run with:
//! ```bash
//! cargo run --example durable_streams --features openai
//! cargo run --example durable_streams --features "openai,durable-streams-sqlite"   # SQLite
//! cargo run --example durable_streams --features "openai,durable-streams-postgres"  # Postgres
//! cargo run --example durable_streams --features "openai,durable-streams-s3"        # S3
//! cargo run --example durable_streams --features full   # includes Redis
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

    // In-memory durable stream: last 10_000 events per stream (ring buffer).
    // Use this when the agent is run behind a harness (e.g. API) that needs
    // to replay events on reconnection via Last-Event-ID.
    let mut agent = AgentBuilder::new()
        .provider(ProviderType::OpenAI)
        .model("gpt-4o-mini".to_string())
        .durable_stream_capacity(10_000)
        .build()
        .await?;

    println!("Agent built with in-memory durable stream (capacity 10_000)");
    let response = agent.chat("Say hello in one short sentence.").await?;
    println!("Response: {}", response.text());

    // Other backends, each behind its own feature: durable_stream_sqlite,
    // durable_stream_postgres, durable_stream_s3, and RedisDurableStreamBackend.

    Ok(())
}
