// Run with:
// cargo run --example sandbox_instance_methods

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
    let instance = backend.create(&config, &workspace).await?;

    println!("sandbox id: {}", instance.id());

    instance
        .write_file("examples-output/methods.txt", b"hello from sandbox")
        .await?;
    let bytes = instance.read_file("examples-output/methods.txt").await?;
    println!("read_file bytes: {}", bytes.len());

    let result = instance
        .exec(
            &["sh", "-c", "echo sandbox-methods-ok"],
            None,
            None,
            Duration::from_secs(5),
        )
        .await?;
    println!("exec stdout: {}", result.stdout.trim());
    println!("exec exit code: {}", result.exit_code);

    println!("is_alive before destroy: {}", instance.is_alive().await);

    match instance.reset().await {
        Ok(()) => println!("reset supported"),
        Err(err) => println!("reset not supported on this backend: {}", err),
    }

    instance.destroy().await?;
    println!("is_alive after destroy: {}", instance.is_alive().await);
    Ok(())
}
