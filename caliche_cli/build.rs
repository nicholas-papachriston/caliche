//! Embed `build_id` for `doctor` and `--version`.
//!
//! The value is tip literacy (PATH binary versus the checkout that produced
//! it). It is not a security property. Packagers without `.git` can set
//! `CALICHE_BUILD_ID` to the caliche commit.

#![allow(
    clippy::print_stdout,
    reason = "cargo build-script instructions are stdout key=value lines"
)]

use std::path::Path;
use std::process::Command;

#[path = "src/build_id_select.rs"]
mod build_id_select;

fn main() {
    println!("cargo:rerun-if-env-changed=CALICHE_BUILD_ID");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=src/build_id_select.rs");

    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let explicit = std::env::var("CALICHE_BUILD_ID").ok();
    let super_dir = superproject_dir(&manifest);
    if let Some(dir) = super_dir.as_deref() {
        watch_git_head(dir);
    }
    watch_git_head(&manifest);

    let superproject = super_dir.as_deref().and_then(git_candidate);
    let crate_head = git_candidate(&manifest);
    let id = build_id_select::select_build_id(
        explicit.as_deref(),
        superproject
            .as_ref()
            .map(|(sha, dirty)| (sha.as_str(), *dirty)),
        crate_head
            .as_ref()
            .map(|(sha, dirty)| (sha.as_str(), *dirty)),
    );
    println!("cargo:rustc-env=CALICHE_BUILD_ID={id}");
}

fn superproject_dir(manifest: &str) -> Option<String> {
    let text = git_stdout(manifest, &["rev-parse", "--show-superproject-working-tree"])?;
    let path = text.trim();
    if path.is_empty() {
        None
    } else {
        Some(path.to_owned())
    }
}

fn git_candidate(dir: &str) -> Option<(String, bool)> {
    let sha = git_stdout(dir, &["rev-parse", "HEAD"])?;
    let sha = sha.trim().to_owned();
    if sha.is_empty() {
        return None;
    }
    Some((sha, git_dirty(dir)))
}

fn git_dirty(dir: &str) -> bool {
    let Some(text) = git_stdout(dir, &["status", "--porcelain"]) else {
        return false;
    };
    !text.trim().is_empty()
}

fn watch_git_head(dir: &str) {
    let Some(git_dir) = git_stdout(dir, &["rev-parse", "--absolute-git-dir"]) else {
        return;
    };
    let git_dir = git_dir.trim();
    if git_dir.is_empty() || git_dir.contains('\n') {
        return;
    }
    let head = Path::new(git_dir).join("HEAD");
    println!("cargo:rerun-if-changed={}", head.display());
}

fn git_stdout(dir: &str, args: &[&str]) -> Option<String> {
    if dir.is_empty() {
        return None;
    }
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}
