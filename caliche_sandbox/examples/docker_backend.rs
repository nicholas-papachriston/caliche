// Run with:
// cargo run --example docker_backend --features docker

#![allow(
    clippy::print_stdout,
    reason = "examples print the sandbox result for the operator"
)]

#[cfg(feature = "docker")]
use anyhow::Result;
#[cfg(feature = "docker")]
use caliche_sandbox::{NamedPolicy, SandboxBackendType, create_backend, policy_to_config};
#[cfg(feature = "docker")]
use std::time::Duration;

#[cfg(feature = "docker")]
#[tokio::main]
async fn main() -> Result<()> {
    let workspace = std::env::current_dir()?;
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::Docker {
            image: "caliche-sandbox:latest".to_string(),
            pool_size: 0,
        },
    );

    let backend = match create_backend(&config).await {
        Ok(backend) => backend,
        Err(err) => {
            println!("docker backend unavailable: {}", err);
            return Ok(());
        }
    };
    println!("backend: {}", backend.capabilities().backend_name);

    let instance = backend.create(&config, &workspace).await?;
    let result = instance
        .exec(
            &["sh", "-c", "echo docker-backend-ok"],
            None,
            None,
            Duration::from_secs(15),
        )
        .await?;
    println!("stdout: {}", result.stdout.trim());
    println!("exit_code: {}", result.exit_code);
    instance.destroy().await?;
    Ok(())
}

#[cfg(not(feature = "docker"))]
fn main() {
    println!("Enable docker feature: cargo run --example docker_backend --features docker");
}
