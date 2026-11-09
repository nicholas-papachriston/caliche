//! Path utilities.

use std::path::{Path, PathBuf};

/// Walk parent directories from `start` to find the git root (directory containing .git).
pub fn find_git_root(start: &Path) -> Option<PathBuf> {
    let mut current = start.canonicalize().ok()?;
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        current = current.parent()?.to_path_buf();
    }
}
