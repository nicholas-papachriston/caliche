//! Only `caliche_test_env` may call `std::env::set_var` or `remove_var`.
//!
//! Clippy's `disallowed-methods` covers the targets the merge gate lints. This
//! check also covers test files the gate does not lint yet, and code that
//! names the calls through an alias.

use std::path::{Path, PathBuf};

/// Text that changes the process environment outside the guard.
const RAW_ENV_MUTATION: [&str; 4] = [
    "env::set_var",
    "env::remove_var",
    "env::{set_var",
    "env::{remove_var",
];

/// Folders that hold no first-party Rust source, or hold the guard itself.
const SKIPPED_DIRS: [&str; 6] = [
    "target",
    "node_modules",
    "shotput",
    ".git",
    "caliche_test_env",
    "third_party",
];

fn workspace_root() -> PathBuf {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .ancestors()
        .nth(2)
        .map_or_else(|| crate_dir.to_path_buf(), Path::to_path_buf)
}

fn rust_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        if kind.is_dir() {
            let skipped = SKIPPED_DIRS.iter().any(|skip| name == *skip);
            if !skipped {
                rust_files(&path, found);
            }
        } else if kind.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
}

/// Every `path:line` in `root` that changes the environment without the guard.
fn raw_env_mutations(root: &Path) -> Vec<String> {
    let mut files = Vec::new();
    rust_files(root, &mut files);
    let mut hits = Vec::new();
    for file in files {
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        for (index, line) in text.lines().enumerate() {
            let compact: String = line.chars().filter(|c| !c.is_whitespace()).collect();
            if RAW_ENV_MUTATION
                .iter()
                .any(|pattern| compact.contains(pattern))
            {
                let shown = file
                    .strip_prefix(root)
                    .unwrap_or(&file)
                    .to_string_lossy()
                    .replace('\\', "/");
                hits.push(format!("{shown}:{}", index + 1));
            }
        }
    }
    hits.sort();
    hits
}

#[test]
fn no_code_outside_the_guard_changes_the_environment() {
    let root = workspace_root();
    assert!(
        root.join("Cargo.toml").is_file() && root.join("crates").is_dir(),
        "workspace root not found from {}",
        root.display()
    );
    let hits = raw_env_mutations(&root);
    assert_eq!(
        hits,
        Vec::<String>::new(),
        "use caliche_test_env::EnvGuard instead of a raw set_var or remove_var"
    );
}

#[test]
fn the_scan_finds_a_raw_call_and_an_aliased_import() {
    let dir = std::env::temp_dir().join(format!("caliche-test-env-scan-{}", std::process::id()));
    let src = dir.join("src");
    std::fs::create_dir_all(&src).ok();
    std::fs::write(
        src.join("a.rs"),
        "fn a() {\n    unsafe { std::env::set_var(\"K\", \"v\") };\n}\n",
    )
    .ok();
    std::fs::write(
        src.join("b.rs"),
        "use std::env::{ remove_var as clear };\nfn b() { let _ = 1; }\n",
    )
    .ok();
    std::fs::write(src.join("c.rs"), "fn c() { workflow.set_var(\"x\"); }\n").ok();
    let hits = raw_env_mutations(&dir);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(
        hits,
        vec!["src/a.rs:2".to_string(), "src/b.rs:1".to_string()]
    );
}
