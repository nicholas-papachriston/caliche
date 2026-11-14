#![cfg(all(feature = "sandbox-wasm", unix))]

use caliche_sandbox::backends::wasm::WasmtimeBackend;
use caliche_sandbox::traits::SandboxBackend;
use caliche_sandbox::{NamedPolicy, SandboxBackendType, policy_to_config};
use std::collections::HashMap;
use std::os::unix::fs::PermissionsExt;
use std::time::Duration;

/// The smallest valid module: magic number and version.
const EMPTY_MODULE: &[u8] = b"\0asm\x01\0\0\0";

#[tokio::test]
async fn exec_never_runs_a_wasmtime_planted_on_its_path() {
    let workspace = tempfile::TempDir::new().expect("workspace");
    std::fs::write(workspace.path().join("empty.wasm"), EMPTY_MODULE).expect("write module");
    let planted = tempfile::TempDir::new().expect("planted dir");
    let marker = planted.path().join("ran");
    let fake = planted.path().join("wasmtime");
    std::fs::write(
        &fake,
        format!("#!/bin/sh\n/usr/bin/touch '{}'\n", marker.display()),
    )
    .expect("write planted wasmtime");
    std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).expect("chmod");

    let config = policy_to_config(
        NamedPolicy::ReadOnly,
        workspace.path(),
        None,
        SandboxBackendType::WasmWasmtime {
            policy: NamedPolicy::ReadOnly,
            modules: HashMap::new(),
        },
    );
    let instance = WasmtimeBackend::new()
        .expect("backend")
        .create(&config, workspace.path())
        .await
        .expect("wasm instance");
    let env = HashMap::from([("PATH".to_string(), planted.path().display().to_string())]);
    // Without a real wasmtime on the agent's PATH, exec fails; either way the
    // planted one must not run.
    let _ = instance
        .exec(&["empty.wasm"], None, Some(&env), Duration::from_secs(5))
        .await;
    assert!(!marker.exists(), "planted wasmtime ran");
}

/// A wasm instance that may write temp files gets its own temp dir; the
/// shared `/tmp`, where other programs and repositories live, is neither
/// readable nor writable through it.
#[tokio::test]
async fn the_shared_temp_dir_is_outside_a_wasm_instance() {
    let workspace = tempfile::TempDir::new().expect("workspace");
    let shared = tempfile::tempdir_in("/tmp").expect("shared");
    let secret = shared.path().join("secret.txt");
    std::fs::write(&secret, "SHARED-TMP-SECRET").expect("secret");
    let planted = shared.path().join("planted");
    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        workspace.path(),
        None,
        SandboxBackendType::WasmWasmtime {
            policy: NamedPolicy::WorkspaceWrite,
            modules: HashMap::new(),
        },
    );
    let instance = WasmtimeBackend::new()
        .expect("backend")
        .create(&config, workspace.path())
        .await
        .expect("wasm instance");

    let read = instance
        .read_file(&secret.display().to_string())
        .await
        .map_err(|err| err.to_string());
    let written = instance
        .write_file(&planted.display().to_string(), b"x")
        .await
        .map_err(|err| err.to_string());

    assert_eq!(
        (read, written),
        (
            Err(format!(
                "WASM sandbox denies read access outside allowed roots: {}",
                secret.display()
            )),
            Err(format!(
                "WASM sandbox denies write access outside allowed roots: {}",
                planted.display()
            )),
        )
    );
    assert!(!planted.exists());
}
