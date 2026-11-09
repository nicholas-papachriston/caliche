//! File scanner for RepoMap.

use ignore::WalkBuilder;
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

    let mut walker = WalkBuilder::new(root);
    walker
        .standard_filters(true)
        .add_custom_ignore_filename(".caliche_ignore");

    let mut out = Vec::new();
    for result in walker.build() {
        let entry = match result {
            Ok(e) => e,
            Err(_) => continue,
        };

        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_lowercase())
            .unwrap_or_default();
        if ext_set.contains(ext.as_str()) {
            out.push(path.to_path_buf());
        }
    }

    out
}
