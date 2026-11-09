//! Skill tree copy, hash, diff, and git checkout helpers.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail};
use similar::TextDiff;

use super::LOCK_FILE;

pub(super) fn skill_tree_diff(current: &Path, incoming: &Path) -> Result<String> {
    let current_files = list_files(current)?;
    let incoming_files = list_files(incoming)?;
    let mut out = String::new();
    let mut left = 0;
    let mut right = 0;
    while left < current_files.len() || right < incoming_files.len() {
        append_next_diff(
            &mut out,
            current,
            incoming,
            &current_files,
            &incoming_files,
            &mut left,
            &mut right,
        )?;
    }
    Ok(out)
}

fn list_files(dir: &Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        collect_hash_files(dir, dir, &mut files)?;
    }
    files.sort();
    Ok(files)
}

fn append_next_diff(
    out: &mut String,
    current: &Path,
    incoming: &Path,
    current_files: &[String],
    incoming_files: &[String],
    left: &mut usize,
    right: &mut usize,
) -> Result<()> {
    match (current_files.get(*left), incoming_files.get(*right)) {
        (Some(old), Some(new)) if old == new => {
            append_changed_file(out, &current.join(old), &incoming.join(new), old)?;
            *left += 1;
            *right += 1;
        }
        (Some(old), Some(new)) if old.as_str() < new.as_str() => {
            push_line(out, &format!("removed {old}"));
            *left += 1;
        }
        (Some(_), Some(new)) => {
            append_added_file(out, &incoming.join(new), new)?;
            *right += 1;
        }
        (Some(old), None) => {
            push_line(out, &format!("removed {old}"));
            *left += 1;
        }
        (None, Some(new)) => {
            append_added_file(out, &incoming.join(new), new)?;
            *right += 1;
        }
        (None, None) => {}
    }
    Ok(())
}

fn append_changed_file(out: &mut String, old: &Path, new: &Path, relative: &str) -> Result<()> {
    let old_bytes = fs::read(old).with_context(|| format!("read {}", old.display()))?;
    let new_bytes = fs::read(new).with_context(|| format!("read {}", new.display()))?;
    if old_bytes == new_bytes {
        return Ok(());
    }
    match (
        std::str::from_utf8(&old_bytes),
        std::str::from_utf8(&new_bytes),
    ) {
        (Ok(old_text), Ok(new_text)) => append_unified(out, relative, old_text, new_text),
        _ => push_line(out, &format!("binary file {relative} changed")),
    }
    Ok(())
}

fn append_added_file(out: &mut String, path: &Path, relative: &str) -> Result<()> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    match std::str::from_utf8(&bytes) {
        Ok(text) => append_unified(out, relative, "", text),
        Err(_) => push_line(out, &format!("added binary file {relative}")),
    }
    Ok(())
}

fn append_unified(out: &mut String, path: &str, old: &str, new: &str) {
    let text = TextDiff::from_lines(old, new)
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();
    out.push_str(&text);
    if !out.ends_with('\n') {
        out.push('\n');
    }
}

fn push_line(out: &mut String, line: &str) {
    out.push_str(line);
    out.push('\n');
}

pub(super) fn replace_tree(src: &Path, dest: &Path) -> Result<()> {
    if dest.exists() {
        fs::remove_dir_all(dest).with_context(|| format!("remove {}", dest.display()))?;
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    copy_tree(src, dest)
}

fn copy_tree(src: &Path, dest: &Path) -> Result<()> {
    fs::create_dir_all(dest)?;
    for entry in fs::read_dir(src).with_context(|| format!("read {}", src.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        if name == ".git" || name == LOCK_FILE {
            continue;
        }
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!("refusing to copy symlink {}", entry.path().display());
        }
        let target = dest.join(&name);
        if file_type.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), &target)
                .with_context(|| format!("copy {}", target.display()))?;
        }
    }
    Ok(())
}

pub(super) fn collect_hash_files(root: &Path, dir: &Path, out: &mut Vec<String>) -> Result<()> {
    for entry in fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let entry = entry?;
        let name = entry.file_name();
        if name == ".git" || name == LOCK_FILE {
            continue;
        }
        let file_type = entry.file_type()?;
        if file_type.is_symlink() {
            bail!("refusing to hash symlink {}", entry.path().display());
        }
        if file_type.is_dir() {
            collect_hash_files(root, &entry.path(), out)?;
        } else if file_type.is_file() {
            let relative = entry
                .path()
                .strip_prefix(root)
                .context("skill file escaped its directory")?
                .to_string_lossy()
                .replace('\\', "/");
            out.push(relative);
        }
    }
    Ok(())
}

static CHECKOUT_SEQ: AtomicU64 = AtomicU64::new(0);

pub(super) struct TempCheckout(pub(super) PathBuf);

impl TempCheckout {
    pub(super) fn create() -> Result<Self> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let seq = CHECKOUT_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "caliche-skill-{}-{nanos}-{seq}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).context("create skill checkout")?;
        Ok(Self(dir))
    }
}

impl Drop for TempCheckout {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub(super) fn clone_repo(url: &str, dest: &Path) -> Result<String> {
    let status = Command::new("git")
        .args(["clone", "--depth", "1", "--", url])
        .arg(dest)
        .status()
        .context("run git clone")?;
    if !status.success() {
        bail!("git clone failed for {url}");
    }
    git_head(dest)
}

pub(super) fn git_head(repo: &Path) -> Result<String> {
    let output = Command::new("git")
        .args(["-C"])
        .arg(repo)
        .args(["rev-parse", "HEAD"])
        .output()
        .context("run git rev-parse")?;
    if !output.status.success() {
        bail!("git rev-parse failed in {}", repo.display());
    }
    let sha = String::from_utf8(output.stdout).context("git sha is not utf-8")?;
    let sha = sha.trim();
    if sha.len() < 7 {
        bail!("git sha is empty");
    }
    Ok(sha.to_string())
}
