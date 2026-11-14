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
        crate::agent::tools::tree_walk::WalkOptions::default(),
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
