// Run with:
// cargo run --example macos_seatbelt_backend --features sandbox-macos

#![allow(
    clippy::print_stdout,
    reason = "examples print the sandbox result for the operator"
)]

#[cfg(all(feature = "sandbox-macos", target_os = "macos"))]
use anyhow::Result;
#[cfg(all(feature = "sandbox-macos", target_os = "macos"))]
use caliche_sandbox::{NamedPolicy, SandboxBackendType, create_backend, policy_to_config};
#[cfg(all(feature = "sandbox-macos", target_os = "macos"))]
use std::time::Duration;

#[cfg(all(feature = "sandbox-macos", target_os = "macos"))]
#[tokio::main]
async fn main() -> Result<()> {
    let workspace = std::env::current_dir()?;
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );

    let backend = create_backend(&config).await?;
    println!("backend: {}", backend.capabilities().backend_name);
    let instance = backend.create(&config, &workspace).await?;
    let result = instance
        .exec(
            &["sh", "-c", "echo macos-seatbelt-backend-ok"],
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

#[cfg(not(all(feature = "sandbox-macos", target_os = "macos")))]
fn main() {
    println!(
        "This example requires macOS with sandbox-macos feature: cargo run --example macos_seatbelt_backend --features sandbox-macos"
    );
}
