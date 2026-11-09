//! The sandbox the CLI starts with does no work that grows with the
//! workspace, and a workspace the sandbox refuses reaches only the
//! sandboxed calls.

#![cfg(target_os = "macos")]

use caliche_cli::sandbox_startup::startup_handle;
use caliche_sandbox::paths::git_refusal;
use caliche_sandbox::{
    NamedPolicy, SANDBOX_UNAVAILABLE, SandboxBackendType, SandboxConfig, policy_to_config,
};
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::time::Duration;

fn workspace_write(root: &Path) -> SandboxConfig {
    policy_to_config(
        NamedPolicy::WorkspaceWrite,
        root,
        None,
        SandboxBackendType::MacOSSeatbelt {
            policy: NamedPolicy::WorkspaceWrite,
        },
    )
}

/// A loose git dir: what git discovery needs to accept `dir`.
fn git_dir_at(dir: &Path) {
    std::fs::create_dir_all(dir.join("objects")).expect("objects");
    std::fs::create_dir_all(dir.join("refs").join("heads")).expect("refs");
    std::fs::write(dir.join("HEAD"), "ref: refs/heads/main\n").expect("HEAD");
}

/// A workspace with a copied submodule gitfile whose git dir does not
/// exist (as `bun install` leaves under `node_modules/`), and a directory
/// nobody can list. The Linux and Docker scan for nested `.git` refuses
/// both, so the exec succeeding shows that the macOS backend reads the
/// index alone and does no scan.
#[tokio::test]
async fn startup_handle_runs_without_a_tree_walk_in_a_workspace_with_build_output() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().canonicalize().expect("canonical");
    git_dir_at(&root.join(".git"));
    let pkg = root.join("node_modules").join("pkg");
    std::fs::create_dir_all(&pkg).expect("pkg");
    std::fs::write(pkg.join(".git"), "gitdir: ../.git/modules/pkg\n").expect("gitfile");
    let locked = root.join("locked");
    std::fs::create_dir(&locked).expect("locked");
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).expect("chmod 000");

    let handle = startup_handle(&workspace_write(&root), &root).await;
    let output = match &handle {
        Ok(handle) => handle
            .exec(
                &["sh", "-c", "printf ok"],
                Some(root.to_str().expect("utf-8")),
                None,
                Duration::from_secs(5),
            )
            .await
            .map(|result| result.stdout)
            .map_err(|err| err.to_string()),
        Err(err) => Err(err.to_string()),
    };
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).expect("chmod back");

    assert_eq!(output, Ok("ok".to_owned()));
}

/// `.git` names `meta` inside the workspace, which the sandbox could
/// write. Startup succeeds; the sandboxed call returns the refusal, which
/// names the path.
#[tokio::test]
async fn startup_handle_refusal_reaches_only_sandboxed_calls() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().canonicalize().expect("canonical");
    git_dir_at(&root.join("meta"));
    std::fs::write(root.join(".git"), "gitdir: meta\n").expect("gitfile");

    let handle = startup_handle(&workspace_write(&root), &root)
        .await
        .map_err(|err| err.to_string());
    let handle = handle.expect("startup");
    let refused = handle
        .exec(
            &["sh", "-c", ": > ran.txt"],
            Some(root.to_str().expect("utf-8")),
            None,
            Duration::from_secs(5),
        )
        .await
        .map(|_| ())
        .map_err(|err| err.to_string());

    assert_eq!(
        refused,
        Err(format!(
            "{SANDBOX_UNAVAILABLE}: {}",
            git_refusal::git_dir_in_workspace(&root.join("meta"))
        ))
    );
    assert!(!root.join("ran.txt").exists());
}
