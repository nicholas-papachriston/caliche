use super::registry::logs_dir;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub(super) fn read_log_tail(path: &Path, lines: usize) -> Result<String> {
    let max_lines = lines.max(1);
    let content = std::fs::read_to_string(path)?;
    let rows: Vec<&str> = content.lines().collect();
    let start = rows.len().saturating_sub(max_lines);
    Ok(rows[start..].join("\n"))
}

pub(super) fn latest_log_file_path(prefix: &str) -> Option<String> {
    let dir = logs_dir().ok()?;
    let entries = std::fs::read_dir(dir).ok()?;
    let mut candidates: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix) && name.ends_with(".log"))
        })
        .collect();
    candidates.sort();
    candidates.last().map(|path| path.display().to_string())
}
