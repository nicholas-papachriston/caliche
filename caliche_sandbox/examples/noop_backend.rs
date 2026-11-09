// Run with:
// cargo run --example noop_backend

#![allow(
    clippy::print_stdout,
    reason = "examples print the sandbox result for the operator"
)]

use anyhow::Result;
use caliche_sandbox::{NamedPolicy, SandboxBackendType, create_backend, policy_to_config};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    let workspace = std::env::current_dir()?;
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );

    let backend = create_backend(&config).await?;
    println!("backend: {}", backend.capabilities().backend_name);

    let instance = backend.create(&config, &workspace).await?;
    let result = instance
        .exec(
            &["sh", "-c", "echo noop-backend-ok"],
            None,
            None,
            Duration::from_secs(5),
        )
        .await?;

    println!("stdout: {}", result.stdout.trim());
    println!("exit_code: {}", result.exit_code);
    instance.destroy().await?;
    Ok(())
}
