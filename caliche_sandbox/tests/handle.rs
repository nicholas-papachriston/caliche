use anyhow::Result;
use caliche_sandbox::backends::noop::NoopBackend;
use caliche_sandbox::traits::SandboxBackend;
use caliche_sandbox::{
    NamedPolicy, SandboxBackendType, SandboxHandle, SandboxMode, policy_to_config,
};
use std::path::Path;

async fn test_handle(mode: SandboxMode, is_main: bool) -> Result<SandboxHandle> {
    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        Path::new("/tmp"),
        None,
        SandboxBackendType::Noop,
    );
    let backend = NoopBackend;
    let instance = backend.create(&config, Path::new("/tmp")).await?;
    Ok(SandboxHandle::new(instance, config.tool_policy)
        .with_mode(mode)
        .with_main_agent(is_main))
}

#[tokio::test]
async fn sandbox_mode_off_is_inactive() -> Result<()> {
    assert!(!test_handle(SandboxMode::Off, true).await?.sandbox_active());
    Ok(())
}

#[tokio::test]
async fn sandbox_mode_all_is_active() -> Result<()> {
    assert!(test_handle(SandboxMode::All, true).await?.sandbox_active());
    Ok(())
}

#[tokio::test]
async fn sandbox_mode_non_main_only_for_subagents() -> Result<()> {
    assert!(
        !test_handle(SandboxMode::NonMain, true)
            .await?
            .sandbox_active()
    );
    assert!(
        test_handle(SandboxMode::NonMain, false)
            .await?
            .sandbox_active()
    );
    Ok(())
}
