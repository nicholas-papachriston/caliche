#![cfg(feature = "docker")]
//! The workspace `.git` stays read-only in a Docker container.
//!
//! Runs only when `CLS_DOCKER_IMAGE` names a local image with `/bin/sh`
//! (for example `alpine:3.20`) and a Docker daemon is running.

use caliche_sandbox::paths::{
    CONFIG_FILE, GIT_DIR_NAME, GITFILE_PREFIX, GITLINK_MODE, INDEX_FILE, MODULES_DIR, git_refusal,
};
use caliche_sandbox::{NamedPolicy, SandboxBackend, SandboxBackendType, policy_to_config};
use std::fs;
use std::time::Duration;

mod git_shapes;

const IMAGE: &str = "CLS_DOCKER_IMAGE";

/// A version 2 index whose only entry is a submodule at `path`.
fn submodule_index(path: &str) -> Vec<u8> {
    let mut data = b"DIRC".to_vec();
    data.extend_from_slice(&2_u32.to_be_bytes());
    data.extend_from_slice(&1_u32.to_be_bytes());
    let start = data.len();
    data.extend_from_slice(&[0; 24]);
    data.extend_from_slice(&GITLINK_MODE.to_be_bytes());
    data.extend_from_slice(&[0; 12 + 20]);
    data.extend_from_slice(&u16::try_from(path.len()).unwrap_or(0).to_be_bytes());
    data.extend_from_slice(path.as_bytes());
    let padded = (data.len() - start + 8) & !7;
    data.resize(start + padded, 0);
    data.extend_from_slice(&[0; 20]);
    data
}

#[tokio::test]
async fn a_container_cannot_change_the_workspace_git_dir() {
    let Ok(image) = std::env::var(IMAGE) else {
        eprintln!("skipped: set {IMAGE} to a local image to run");
        return;
    };
    let dir = tempfile::tempdir().expect("workspace");
    let root = dir.path().canonicalize().expect("canonical");
    git_shapes::git_dir_at(&root.join(GIT_DIR_NAME));
    fs::create_dir_all(root.join(GIT_DIR_NAME).join("hooks")).expect("hooks dir");
    fs::write(root.join(GIT_DIR_NAME).join(CONFIG_FILE), b"[core]\n").expect("config");

    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &root,
        None,
        SandboxBackendType::Docker {
            image: image.clone(),
            pool_size: 0,
        },
    );
    let backend = caliche_sandbox::backends::docker::DockerBackend::new(&image)
        .await
        .expect("docker daemon");
    let instance = backend.create(&config, &root).await.expect("container");
    let script = "(echo x >> .git/config) 2>/dev/null && printf 'config '; \
                  (: > .git/hooks/pre-commit) 2>/dev/null && printf 'hook '; \
                  rm -rf .git 2>/dev/null; [ -d .git ] || printf 'delete '; \
                  (: > ok.txt) && printf 'workspace'";
    let result = instance
        .exec(
            &["/bin/sh", "-c", script],
            Some(root.to_str().expect("utf-8")),
            None,
            Duration::from_secs(30),
        )
        .await
        .expect("exec");
    instance.destroy().await.expect("destroy");

    assert_eq!(result.stdout, "workspace");
    assert_eq!(
        fs::read_to_string(root.join(GIT_DIR_NAME).join(CONFIG_FILE)).expect("config"),
        "[core]\n"
    );
    assert!(
        !root
            .join(GIT_DIR_NAME)
            .join("hooks")
            .join("pre-commit")
            .exists()
    );
}

/// A bind cannot keep a submodule's `.git` read-only (renaming its parent
/// moves the bind aside), so the instance is refused before a container
/// starts.
#[tokio::test]
async fn a_workspace_with_a_submodule_gitfile_is_refused() {
    let Ok(image) = std::env::var(IMAGE) else {
        eprintln!("skipped: set {IMAGE} to a local image to run");
        return;
    };
    let dir = tempfile::tempdir().expect("workspace");
    let root = dir.path().canonicalize().expect("canonical");
    git_shapes::git_dir_at(&root.join(GIT_DIR_NAME));
    fs::create_dir_all(root.join(GIT_DIR_NAME).join(MODULES_DIR).join("sub")).expect("modules");
    fs::create_dir(root.join("sub")).expect("sub");
    let gitfile = format!("{GITFILE_PREFIX} ../{GIT_DIR_NAME}/{MODULES_DIR}/sub\n");
    fs::write(root.join("sub/.git"), &gitfile).expect("gitfile");
    fs::write(
        root.join(GIT_DIR_NAME).join(INDEX_FILE),
        submodule_index("sub"),
    )
    .expect("index");

    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &root,
        None,
        SandboxBackendType::Docker {
            image: image.clone(),
            pool_size: 0,
        },
    );
    let backend = caliche_sandbox::backends::docker::DockerBackend::new(&image)
        .await
        .expect("docker daemon");
    let Err(err) = backend.create(&config, &root).await else {
        panic!("a container started with a writable submodule gitfile");
    };

    assert_eq!(err.to_string(), git_refusal::docker_nested("sub/.git"));
    assert_eq!(
        fs::read_to_string(root.join("sub/.git")).expect("gitfile"),
        gitfile
    );
}

/// A workspace without `.git` inside a linked worktree is refused before a
/// container starts: host git would find the worktree's git dir above it,
/// which a bind cannot keep read-only.
#[tokio::test]
async fn a_workspace_inside_a_worktree_is_refused() {
    let Ok(image) = std::env::var(IMAGE) else {
        eprintln!("skipped: set {IMAGE} to a local image to run");
        return;
    };
    let dir = tempfile::tempdir().expect("base");
    let base = dir.path().canonicalize().expect("canonical");
    let git_dir = caliche_policy::plant_linked_worktree(&base.join("m").join(GIT_DIR_NAME), "wt")
        .expect("linked worktree");
    let root = base.join("wt").join("sub");
    fs::create_dir_all(&root).expect("ws");
    fs::write(
        base.join("wt").join(GIT_DIR_NAME),
        format!("{GITFILE_PREFIX} {}\n", git_dir.display()),
    )
    .expect("gitfile");

    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &root,
        None,
        SandboxBackendType::Docker {
            image: image.clone(),
            pool_size: 0,
        },
    );
    let backend = caliche_sandbox::backends::docker::DockerBackend::new(&image)
        .await
        .expect("docker daemon");
    let Err(err) = backend.create(&config, &root).await else {
        panic!("a container started in a worktree without its own .git");
    };

    assert_eq!(err.to_string(), git_refusal::DOCKER_NO_ROOT_GIT);
}

/// The shapes Docker runs at all (the workspace has a `.git` entry) are
/// refused when that `.git` is not one git would use.
#[cfg(unix)]
#[tokio::test]
async fn create_refuses_every_planted_repository_shape_with_a_root_git() {
    let Ok(image) = std::env::var(IMAGE) else {
        eprintln!("skipped: set {IMAGE} to a local image to run");
        return;
    };
    let dir = tempfile::tempdir().expect("base");
    let base = dir.path().canonicalize().expect("canonical");
    let backend = caliche_sandbox::backends::docker::DockerBackend::new(&image)
        .await
        .expect("docker daemon");
    let mut wrong = Vec::new();

    for shape in git_shapes::planted_repository_shapes(&base)
        .into_iter()
        .filter(|shape| shape.workspace.join(GIT_DIR_NAME).exists())
    {
        let config = policy_to_config(
            NamedPolicy::WorkspaceWrite,
            &shape.writable,
            None,
            SandboxBackendType::Docker {
                image: image.clone(),
                pool_size: 0,
            },
        );
        let refused = backend
            .create(&config, &shape.workspace)
            .await
            .err()
            .map(|err| err.to_string());
        if refused.as_deref() != Some(shape.refusal.as_str()) {
            wrong.push((shape.name, refused));
        }
    }

    assert_eq!(wrong, Vec::<(&str, Option<String>)>::new());
}
