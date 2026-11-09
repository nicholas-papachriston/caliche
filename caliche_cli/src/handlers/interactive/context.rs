//! Context helpers for interactive mode.
//!
//! Handles @file, glob patterns, @recent, and related-file resolution.

use crate::core::ignore::should_ignore_file;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Find files matching a glob pattern.
pub async fn find_files_by_glob(pattern: &str) -> Result<Vec<PathBuf>> {
    let cwd = std::env::current_dir()?;
    let mut results = Vec::new();

    let full_pattern = if pattern.starts_with('/') || pattern.starts_with('.') {
        pattern.to_string()
    } else {
        format!("{}/{}", cwd.display(), pattern)
    };

    let project_root = std::env::current_dir()?;

    match glob::glob(&full_pattern) {
        Ok(paths) => {
            for entry in paths.flatten() {
                if entry.is_file() && !should_ignore_file(&entry, &project_root) {
                    results.push(entry);
                }
            }
        }
        Err(e) => return Err(anyhow::anyhow!("Invalid glob pattern: {}", e)),
    }

    results.sort();
    Ok(results)
}

/// Find recently modified files in the current directory.
pub async fn find_recent_files(
    count: usize,
) -> Result<Vec<(PathBuf, chrono::DateTime<chrono::Utc>)>> {
    use std::time::SystemTime;

    let cwd = std::env::current_dir()?;
    let mut files = Vec::new();

    let project_root = cwd.clone();
    fn visit_dir(
        path: &std::path::Path,
        files: &mut Vec<(PathBuf, SystemTime)>,
        depth: usize,
        project_root: &PathBuf,
    ) {
        if depth > 3 {
            return;
        }

        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let path = entry.path();

                // Check ignore file
                if should_ignore_file(&path, project_root) {
                    continue;
                }

                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with('.')
                        || name == "node_modules"
                        || name == "target"
                        || name == "dist"
                        || name == "build"
                        || name == "__pycache__"
                    {
                        continue;
                    }
                }

                if path.is_file() {
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            files.push((path, modified));
                        }
                    }
                } else if path.is_dir() {
                    visit_dir(&path, files, depth + 1, project_root);
                }
            }
        }
    }

    visit_dir(&cwd, &mut files, 0, &project_root);

    visit_dir(&cwd, &mut files, 0, &project_root);
    files.sort_by(|a, b| b.1.cmp(&a.1));

    let result: Vec<_> = files
        .into_iter()
        .take(count)
        .map(|(path, time)| (path, chrono::DateTime::<chrono::Utc>::from(time)))
        .collect();

    Ok(result)
}

/// Find files related to a given file (tests, sibling modules, etc.).
pub async fn find_related_files(file_path: &str) -> Result<Vec<(PathBuf, String)>> {
    let path = PathBuf::from(file_path);
    let mut related = Vec::new();

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));

    let test_patterns = related_test_names(stem, ext);

    for pattern in test_patterns {
        let test_path = parent.join(&pattern);
        if test_path.exists() {
            related.push((test_path, "test file".to_string()));
        }
    }

    push_sibling_modules(&mut related, parent, stem, ext);

    for index in ["mod.rs", "index.ts", "index.js", "__init__.py"] {
        let index_path = parent.join(index);
        if index_path.exists() && index_path != path {
            related.push((index_path, "module index".to_string()));
        }
    }

    Ok(related)
}

fn related_test_names(stem: &str, ext: &str) -> Vec<String> {
    match ext {
        "rs" => vec![
            format!("{stem}_test.rs"),
            format!("test_{stem}.rs"),
            format!("tests/{stem}.rs"),
        ],
        "ts" | "tsx" => vec![
            format!("{stem}.test.ts"),
            format!("{stem}.test.tsx"),
            format!("{stem}.spec.ts"),
            format!("__tests__/{stem}.ts"),
        ],
        "js" | "jsx" => vec![
            format!("{stem}.test.js"),
            format!("{stem}.test.jsx"),
            format!("{stem}.spec.js"),
            format!("__tests__/{stem}.js"),
        ],
        "py" => vec![
            format!("test_{stem}.py"),
            format!("{stem}_test.py"),
            format!("tests/test_{stem}.py"),
        ],
        "go" => vec![format!("{stem}_test.go")],
        _ => Vec::new(),
    }
}

fn push_sibling_modules(
    related: &mut Vec<(PathBuf, String)>,
    parent: &Path,
    stem: &str,
    ext: &str,
) {
    let Ok(entries) = std::fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let entry_path = entry.path();
        if !entry_path.is_file() {
            continue;
        }
        let Some(entry_stem) = entry_path.file_stem().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(entry_ext) = entry_path.extension().and_then(|name| name.to_str()) else {
            continue;
        };
        if entry_ext == ext && entry_stem != stem && is_sibling_stem(entry_stem) {
            related.push((entry_path, "sibling module".to_string()));
        }
    }
}

fn is_sibling_stem(stem: &str) -> bool {
    matches!(stem, "mod" | "index" | "types" | "utils")
}
