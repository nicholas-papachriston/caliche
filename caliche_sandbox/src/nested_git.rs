//! Nested `.git` entries under the writable roots, for the Linux and
//! Docker backends.
//!
//! Host `git` run inside a nested repository reads that repository's
//! `.git`, so a sandboxed write there (for example a hook) would run later
//! outside the sandbox. The macOS profile denies writes to a `.git` at any
//! depth by name; Landlock grants and Docker binds have no name rule, so
//! these backends need every nested `.git` as a target in advance.
//!
//! [`guarded_targets`] scans the workspace for them, and
//! [`refuse_git_outside`] the writable roots outside it. It runs where an
//! instance is created, which the CLI defers to the first sandboxed tool,
//! so startup does no work that grows with the workspace. The scan does
//! not follow symlinks and does not descend into a `.git` it finds. It is
//! bounded in entries, depth and time; a bound hit, or a directory it
//! cannot list, refuses the instance, so sandboxed tools fail closed.

use crate::git_targets::index_targets;
use crate::paths::git_refusal;
use anyhow::Result;
use caliche_policy::is_git_dir_name;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Limits of one scan, shared by every writable root it covers.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    /// Directory entries read.
    entries: usize,
    /// Directories below a writable root.
    depth: usize,
    time: Duration,
}

/// The limits the backends use.
const BOUNDS: Bounds = Bounds {
    entries: 1_000_000,
    depth: 64,
    time: Duration::from_secs(10),
};

/// The git targets of `workspace`, relative to it, sorted: the
/// [`index_targets`] plus every `.git` the scan finds in the workspace
/// when a writable root reaches it. `workspace` and `writable` are
/// canonical.
pub(crate) fn guarded_targets(workspace: &Path, writable: &[PathBuf]) -> Result<Vec<PathBuf>> {
    guarded_targets_within(workspace, writable, BOUNDS)
}

fn guarded_targets_within(
    workspace: &Path,
    writable: &[PathBuf],
    bounds: Bounds,
) -> Result<Vec<PathBuf>> {
    let mut targets = index_targets(workspace)?;
    if writable.iter().any(|root| reaches(workspace, root)) {
        targets.extend(Scan::new(bounds).git_entries(workspace)?);
    }
    targets.sort();
    targets.dedup();
    Ok(targets)
}

/// Refuse a `.git` under a writable root outside the workspace: no grant
/// or bind can leave it out. Runs after the backend's other checks, which
/// name the git dir more precisely when it belongs to the workspace.
pub(crate) fn refuse_git_outside(workspace: &Path, writable: &[PathBuf]) -> Result<()> {
    refuse_git_outside_within(workspace, writable, BOUNDS)
}

fn refuse_git_outside_within(workspace: &Path, writable: &[PathBuf], bounds: Bounds) -> Result<()> {
    let mut scan = Scan::new(bounds);
    for root in writable.iter().filter(|root| !reaches(workspace, root)) {
        if let Some(found) = scan.git_entries(root)?.first() {
            anyhow::bail!(git_refusal::git_dir_under_writable_root(
                &root.join(found),
                root
            ));
        }
    }
    Ok(())
}

/// Whether writable `root` is the workspace, inside it, or holds it.
fn reaches(workspace: &Path, root: &Path) -> bool {
    root.starts_with(workspace) || workspace.starts_with(root)
}

struct Scan {
    bounds: Bounds,
    started: Instant,
    entries: usize,
}

impl Scan {
    fn new(bounds: Bounds) -> Self {
        Self {
            bounds,
            started: Instant::now(),
            entries: 0,
        }
    }

    /// Every entry below `root` whose name is a git dir name, relative to
    /// `root`.
    fn git_entries(&mut self, root: &Path) -> Result<Vec<PathBuf>> {
        let mut found = Vec::new();
        let mut pending = vec![(PathBuf::new(), 0)];
        while let Some((dir, depth)) = pending.pop() {
            let path = root.join(&dir);
            let unlistable =
                |_: std::io::Error| anyhow::anyhow!(git_refusal::unlistable_dir(&path));
            for entry in std::fs::read_dir(&path).map_err(unlistable)? {
                let entry = entry.map_err(unlistable)?;
                self.count(root)?;
                let rel = dir.join(entry.file_name());
                if entry.file_name().to_str().is_some_and(is_git_dir_name) {
                    found.push(rel);
                    continue;
                }
                if !entry.file_type().map_err(unlistable)?.is_dir() {
                    continue;
                }
                if depth >= self.bounds.depth {
                    anyhow::bail!(git_refusal::scan_too_deep(
                        &root.join(&rel),
                        self.bounds.depth
                    ));
                }
                pending.push((rel, depth + 1));
            }
        }
        Ok(found)
    }

    fn count(&mut self, root: &Path) -> Result<()> {
        self.entries += 1;
        if self.entries > self.bounds.entries {
            anyhow::bail!(git_refusal::scan_too_many_entries(
                root,
                self.bounds.entries
            ));
        }
        if self.started.elapsed() > self.bounds.time {
            anyhow::bail!(git_refusal::scan_too_slow(root, self.bounds.time));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "nested_git_tests.rs"]
mod tests;
