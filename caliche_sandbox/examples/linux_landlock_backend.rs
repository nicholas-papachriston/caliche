// Run with:
// cargo run --example linux_landlock_backend --features sandbox-linux

#![allow(
    clippy::print_stdout,
    reason = "examples print the sandbox result for the operator"
)]

#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
use anyhow::Result;
#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
use caliche_sandbox::{NamedPolicy, SandboxBackendType, create_backend, policy_to_config};
#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
use std::time::Duration;

#[cfg(all(feature = "sandbox-linux", target_os = "linux"))]
#[tokio::main]
async fn main() -> Result<()> {
    let workspace = std::env::current_dir()?;
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::LinuxLandlock {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );

    let backend = create_backend(&config).await?;
    println!("backend: {}", backend.capabilities().backend_name);
    let instance = backend.create(&config, &workspace).await?;
    let result = instance
        .exec(
            &["sh", "-c", "echo linux-landlock-backend-ok"],
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

#[cfg(not(all(feature = "sandbox-linux", target_os = "linux")))]
fn main() {
    println!(
        "This example requires Linux with sandbox-linux feature: cargo run --example linux_landlock_backend --features sandbox-linux"
    );
}
