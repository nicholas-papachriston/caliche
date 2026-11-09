// Run with:
// cargo run --example windows_job_backend --features sandbox-windows

#![allow(
    clippy::print_stdout,
    reason = "examples print the sandbox result for the operator"
)]

#[cfg(all(feature = "sandbox-windows", target_os = "windows"))]
use anyhow::Result;
#[cfg(all(feature = "sandbox-windows", target_os = "windows"))]
use caliche_sandbox::{NamedPolicy, SandboxBackendType, create_backend, policy_to_config};
#[cfg(all(feature = "sandbox-windows", target_os = "windows"))]
use std::time::Duration;

#[cfg(all(feature = "sandbox-windows", target_os = "windows"))]
#[tokio::main]
async fn main() -> Result<()> {
    let workspace = std::env::current_dir()?;
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::WindowsJob {
            policy: NamedPolicy::WorkspaceWrite,
        },
    );

    let backend = create_backend(&config).await?;
    println!("backend: {}", backend.capabilities().backend_name);
    let instance = backend.create(&config, &workspace).await?;
    let result = instance
        .exec(
            &["cmd", "/c", "echo windows-job-backend-ok"],
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

#[cfg(not(all(feature = "sandbox-windows", target_os = "windows")))]
fn main() {
    println!(
        "This example requires Windows with sandbox-windows feature: cargo run --example windows_job_backend --features sandbox-windows"
    );
}
