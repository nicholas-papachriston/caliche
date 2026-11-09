// Run with:
// cargo run --example wasm_wasmtime_backend --features sandbox-wasm
// Optional runtime override:
// cargo run --example wasm_wasmtime_backend --features sandbox-wasm -- --runtime ./path/to/js-runtime.wasm

#![allow(
    clippy::print_stdout,
    reason = "examples print the sandbox result for the operator"
)]

#[cfg(feature = "sandbox-wasm")]
use anyhow::{Context, Result};
#[cfg(feature = "sandbox-wasm")]
use caliche_sandbox::{NamedPolicy, SandboxBackendType, create_backend, policy_to_config};
#[cfg(feature = "sandbox-wasm")]
use std::collections::HashMap;
#[cfg(feature = "sandbox-wasm")]
use std::path::{Path, PathBuf};
#[cfg(feature = "sandbox-wasm")]
use std::time::Duration;

#[cfg(feature = "sandbox-wasm")]
#[tokio::main]
async fn main() -> Result<()> {
    let workspace = std::env::current_dir()?;
    let Some(runtime_path) = resolve_runtime_path(&workspace) else {
        println!("No JavaScript runtime wasm module found.");
        println!("Resolution order:");
        println!("  1) --runtime <path>");
        println!("  2) WASM_JS_RUNTIME environment variable");
        println!("  3) Default path (if present):");
        for candidate in runtime_default_candidates(&workspace) {
            println!("     - {}", candidate.display());
        }
        println!("Example:");
        println!(
            "  cargo run --example wasm_wasmtime_backend --features sandbox-wasm -- --runtime ./examples/runtimes/quickjs.wasm"
        );
        return Ok(());
    };
    println!("Using JS runtime: {}", runtime_path.display());

    let mut modules = HashMap::new();
    modules.insert("js".to_string(), runtime_path);

    let config = policy_to_config(
        NamedPolicy::WorkspaceWrite,
        &workspace,
        None,
        SandboxBackendType::WasmWasmtime {
            policy: NamedPolicy::WorkspaceWrite,
            modules,
        },
    );

    let backend = create_backend(&config).await?;
    println!("backend: {}", backend.capabilities().backend_name);

    let instance = backend.create(&config, &workspace).await?;
    let script_path = "examples-output/hello.js";
    let js_source = r#"
console.log("hello from sandboxed javascript");
console.log(2 + 3);
"#;

    instance
        .write_file(script_path, js_source.as_bytes())
        .await?;
    let script_bytes = instance
        .read_file(script_path)
        .await
        .context("failed to read generated js script")?;
    println!("script bytes: {}", script_bytes.len());

    match instance
        .exec(&["js", script_path], None, None, Duration::from_secs(10))
        .await
    {
        Ok(result) => {
            println!("javascript stdout:\n{}", result.stdout.trim());
            if !result.stderr.trim().is_empty() {
                println!("javascript stderr:\n{}", result.stderr.trim());
            }
            println!("exec exit_code: {}", result.exit_code);
        }
        Err(err) => {
            println!("javascript execution failed: {}", err);
            println!(
                "Ensure WASM_JS_RUNTIME points to a runtime wasm module that accepts a script path argv."
            );
        }
    }

    instance.destroy().await?;
    Ok(())
}

#[cfg(not(feature = "sandbox-wasm"))]
fn main() {
    println!(
        "Enable wasm backend feature: cargo run --example wasm_wasmtime_backend --features sandbox-wasm"
    );
}

#[cfg(feature = "sandbox-wasm")]
fn resolve_runtime_path(workspace: &Path) -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--runtime" {
            let value = args.next()?;
            let path = absolutize_candidate(workspace, &value);
            if path.exists() {
                return Some(path);
            }
        }
    }

    if let Ok(value) = std::env::var("WASM_JS_RUNTIME") {
        let path = absolutize_candidate(workspace, &value);
        if path.exists() {
            return Some(path);
        }
    }

    for default_path in runtime_default_candidates(workspace) {
        if default_path.exists() {
            return Some(default_path);
        }
    }
    None
}

#[cfg(feature = "sandbox-wasm")]
fn runtime_default_candidates(workspace: &Path) -> Vec<PathBuf> {
    vec![workspace.join("examples/runtimes/quickjs.wasm")]
}

#[cfg(feature = "sandbox-wasm")]
fn absolutize_candidate(workspace: &Path, candidate: &str) -> PathBuf {
    let path = Path::new(candidate);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}
