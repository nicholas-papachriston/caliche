// Run with:
// cargo run --example agentfs_backend --features agentfs

#![allow(
    clippy::print_stdout,
    reason = "examples print the sandbox result for the operator"
)]

#[cfg(feature = "agentfs")]
use anyhow::{Context, Result};
#[cfg(feature = "agentfs")]
use caliche_sandbox::SandboxBackend;
#[cfg(feature = "agentfs")]
use caliche_sandbox::backends::agentfs::AgentFsSandboxBackend;
#[cfg(feature = "agentfs")]
use caliche_sandbox::{NamedPolicy, SandboxBackendType, policy_to_config};
#[cfg(feature = "agentfs")]
use std::sync::Arc;
#[cfg(feature = "agentfs")]
use std::time::Duration;

#[cfg(feature = "agentfs")]
#[tokio::main]
async fn main() -> Result<()> {
    let workspace = std::env::current_dir()?;
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Noop,
    );

    let options = agentfs_sdk::AgentFSOptions::with_path(workspace.to_string_lossy().into_owned());
    let agentfs = agentfs_sdk::AgentFS::open(options)
        .await
        .context("failed to open AgentFS at workspace path")?;
    let backend = AgentFsSandboxBackend::new(Arc::new(agentfs));
    println!("backend: {}", backend.capabilities().backend_name);

    let instance = backend.create(&config, &workspace).await?;
    instance
        .write_file(
            "examples-output/agentfs-backend.txt",
            b"agentfs backend file io",
        )
        .await?;
    let bytes = instance
        .read_file("examples-output/agentfs-backend.txt")
        .await?;
    println!("read_file bytes: {}", bytes.len());

    let result = instance
        .exec(
            &["sh", "-c", "echo agentfs-backend-ok"],
            None,
            None,
            Duration::from_secs(5),
        )
        .await?;
    println!("exec stdout: {}", result.stdout.trim());
    println!("exec exit_code: {}", result.exit_code);
    instance.destroy().await?;
    Ok(())
}

#[cfg(not(feature = "agentfs"))]
fn main() {
    println!("Enable agentfs feature: cargo run --example agentfs_backend --features agentfs");
}
