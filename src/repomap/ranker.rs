//! Heuristic ranking for RepoMap files.

use std::path::Path;

/// High-priority filenames that indicate entry points or key modules.
const PRIORITY_FILENAMES: &[&str] = &[
    "lib.rs",
    "main.rs",
    "mod.rs",
    "index.ts",
    "index.tsx",
    "index.js",
    "index.jsx",
    "__init__.py",
    "main.py",
    "main.go",
    "main.java",
    "App.tsx",
    "App.jsx",
    "app.tsx",
    "app.jsx",
];

/// Rank files by heuristic relevance. Lower score = higher priority.
pub fn rank(paths: &[std::path::PathBuf], _task: Option<&str>) -> Vec<std::path::PathBuf> {
    let task_words: Vec<String> = Vec::new(); // task keyword boost omitted in workspace tool for now

    let mut scored: Vec<_> = paths
        .iter()
        .map(|p| {
            let score = score_path(p, task_words.as_slice());
            (score, p.clone())
        })
        .collect();

    scored.sort_by_key(|(s, _)| *s);
    scored.into_iter().map(|(_, p)| p).collect()
}

fn score_path(path: &Path, task_words: &[String]) -> i32 {
    let mut score = 0i32;

    let depth = path.components().count();
    score += depth as i32 * 10;

    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if PRIORITY_FILENAMES.contains(&name) {
        score -= 50;
    }

    let path_str = path.to_string_lossy().to_lowercase();
    for word in task_words {
        if path_str.contains(word.as_str()) {
            score -= 20;
            break;
        }
    }

    score
}
