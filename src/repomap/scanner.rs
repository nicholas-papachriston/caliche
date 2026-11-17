//! File scanner for RepoMap.

use std::path::{Path, PathBuf};

/// Extensions for source files to include in the codebase map.
pub const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "kt", "swift", "c", "cpp", "h", "hpp", "m",
    "mm", "sh", "bash", "zsh", "sql", "yaml", "yml", "toml", "json", "md", "vue", "svelte", "lua",
];

/// Scan the workspace and return paths to source files.
///
/// Respects .gitignore and .caliche_ignore.
pub fn scan(root: &Path) -> Vec<PathBuf> {
    let ext_set: std::collections::HashSet<&str> = SOURCE_EXTENSIONS.iter().copied().collect();

    let Ok(canonical) = root.canonicalize() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    // The walk follows no link, so a swapped directory cannot take it out
    // of the workspace. Paths keep the caller's spelling of `root`.
    let _ = crate::agent::tools::tree_walk::walk_tree(
        &canonical,
        crate::agent::tools::tree_walk::WalkOptions {
            ignores: crate::agent::tools::tree_walk::Ignores::RepoMap,
            ..Default::default()
        },
        &mut |entry| {
            if entry.kind == caliche_sandbox::EntryKind::File {
                let ext = entry
                    .path
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_lowercase)
                    .unwrap_or_default();
                if ext_set.contains(ext.as_str()) {
                    out.push(root.join(&entry.relative));
                }
            }
            std::ops::ControlFlow::Continue(())
        },
    );

    out
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// The map reads `.caliche_ignore`, and a `.gitignore` above the
    /// repository root does not apply, as the walker it replaced did.
    #[test]
    fn scan_applies_the_repo_map_ignore_rules() {
        let tmp = tempfile::tempdir().expect("tmp");
        let top = tmp.path().canonicalize().expect("top");
        let repo = top.join("repo");
        std::fs::create_dir_all(repo.join(".git")).expect("git dir");
        for (path, text) in [
            (top.join(".gitignore"), "above.rs\n"),
            (repo.join(".caliche_ignore"), "hidden.rs\n"),
            (repo.join("a.rs"), "a"),
            (repo.join("above.rs"), "b"),
            (repo.join("hidden.rs"), "h"),
        ] {
            std::fs::write(path, text).expect("write");
        }

        let mut files = scan(&repo);
        files.sort();

        assert_eq!(files, vec![repo.join("a.rs"), repo.join("above.rs")]);
    }
}
