//! Sandbox Testing Example
//!
//! This example demonstrates:
//! - Creating sandbox environments for testing
//! - Mock providers with different behaviors
//! - Recording interactions for testing
//!
//! Best practices shown:
//! - Testing without real API calls
//! - Deterministic test scenarios
//!
//! Run with:
//! ```bash
//! cargo run --example sandbox --features testing
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

#[cfg(feature = "testing")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    use caliche::agent::service::sandbox::{AgentSandbox, MockBehavior};
    use caliche::prelude::ProviderType;

    let sandbox = AgentSandbox::new()
        .mock_provider(ProviderType::OpenAI, MockBehavior::Deterministic)
        .mock_provider(
            ProviderType::Anthropic,
            MockBehavior::Fixed("Mock Claude response".to_string()),
        )
        .record_interactions(true)
        .build();

    println!("Sandbox created with mock providers");
    println!("Mock providers available:");
    println!("  - OpenAI: Deterministic responses");
    println!("  - Anthropic: Fixed response");

    let interactions = sandbox.get_recorded_interactions().await;
    println!("Recorded interactions: {}", interactions.len());

    Ok(())
}

#[cfg(not(feature = "testing"))]
fn main() {
    println!("Testing feature not enabled. Build with --features testing");
}
