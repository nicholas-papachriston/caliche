//! Diff viewer for file changes.
//!
//! Provides colored inline diff display using the `similar` crate.

use similar::{ChangeTag, TextDiff};
use std::path::Path;

/// ANSI color codes
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

/// Generate a colored unified diff between two texts.
///
/// Produces standard unified-diff output with 3 lines of context, hunk
/// headers of the form `@@ -old_start,old_len +new_start,new_len @@`, and
/// ANSI colours applied per line. Uses `similar::TextDiff::grouped_ops` so
/// context handling is correct.
pub fn unified_diff(old: &str, new: &str, old_label: &str, new_label: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut output = String::new();

    output.push_str(&format!("{}--- {}{}\n", RED, old_label, RESET));
    output.push_str(&format!("{}+++ {}{}\n", GREEN, new_label, RESET));

    let groups = diff.grouped_ops(3);
    if groups.is_empty() {
        return output;
    }

    for group in groups {
        let Some(first) = group.first() else {
            continue;
        };
        let Some(last) = group.last() else {
            continue;
        };
        let (old_start, old_end) = (first.old_range().start, last.old_range().end);
        let (new_start, new_end) = (first.new_range().start, last.new_range().end);
        let old_len = old_end - old_start;
        let new_len = new_end - new_start;
        output.push_str(&format!(
            "{}@@ -{},{} +{},{} @@{}\n",
            CYAN,
            old_start + 1,
            old_len,
            new_start + 1,
            new_len,
            RESET
        ));

        for op in group {
            for change in diff.iter_changes(&op) {
                let value = change.value();
                let needs_newline = !value.ends_with('\n');
                match change.tag() {
                    ChangeTag::Delete => {
                        output.push_str(&format!("{}-{}{}", RED, value, RESET));
                    }
                    ChangeTag::Insert => {
                        output.push_str(&format!("{}+{}{}", GREEN, value, RESET));
                    }
                    ChangeTag::Equal => {
                        output.push_str(&format!(" {}", value));
                    }
                }
                if needs_newline {
                    output.push('\n');
                }
            }
        }
    }

    output
}

/// Generate a side-by-side diff (for wider terminals)
pub fn side_by_side_diff(old: &str, new: &str, width: usize) -> String {
    let diff = TextDiff::from_lines(old, new);
    let half_width = (width - 3) / 2; // Account for separator
    let mut output = String::new();

    // Header
    let left_header = format!("{:^width$}", "Old", width = half_width);
    let right_header = format!("{:^width$}", "New", width = half_width);
    output.push_str(&format!(
        "{}{}{} | {}{}{}\n",
        DIM, left_header, RESET, DIM, right_header, RESET
    ));
    output.push_str(&format!("{}{}{}\n", DIM, "-".repeat(width), RESET));

    for op in diff.ops() {
        for change in diff.iter_changes(op) {
            let line = change.value().trim_end_matches('\n');
            let truncated = if line.len() > half_width {
                format!(
                    "{}...",
                    caliche_contracts::text::truncate_utf8(line, half_width - 3)
                )
            } else {
                line.to_string()
            };

            match change.tag() {
                ChangeTag::Delete => {
                    let padded = format!("{:<width$}", truncated, width = half_width);
                    output.push_str(&format!(
                        "{}{}{} | {}{}{}\n",
                        RED,
                        padded,
                        RESET,
                        DIM,
                        " ".repeat(half_width),
                        RESET
                    ));
                }
                ChangeTag::Insert => {
                    output.push_str(&format!(
                        "{}{}{} | {}{}{}\n",
                        DIM,
                        " ".repeat(half_width),
                        RESET,
                        GREEN,
                        format!("{:<width$}", truncated, width = half_width),
                        RESET
                    ));
                }
                ChangeTag::Equal => {
                    let padded = format!("{:<width$}", truncated, width = half_width);
                    output.push_str(&format!("{} | {}\n", padded, padded));
                }
            }
        }
    }

    output
}

/// Generate a compact inline diff showing only changed lines
pub fn compact_diff(old: &str, new: &str) -> String {
    let diff = TextDiff::from_lines(old, new);
    let mut output = String::new();
    let mut has_changes = false;

    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Delete => {
                has_changes = true;
                output.push_str(&format!("{}-{}{}", RED, change.value(), RESET));
            }
            ChangeTag::Insert => {
                has_changes = true;
                output.push_str(&format!("{}+{}{}", GREEN, change.value(), RESET));
            }
            ChangeTag::Equal => {
                // Skip equal lines in compact mode
            }
        }
    }

    if !has_changes {
        output.push_str(&format!("{}No changes{}\n", DIM, RESET));
    }

    output
}

/// Statistics about the diff
#[derive(Debug, Clone, Default)]
pub struct DiffStats {
    pub additions: usize,
    pub deletions: usize,
    pub unchanged: usize,
}

impl DiffStats {
    /// Calculate diff statistics
    pub fn calculate(old: &str, new: &str) -> Self {
        let diff = TextDiff::from_lines(old, new);
        let mut stats = DiffStats::default();

        for change in diff.iter_all_changes() {
            match change.tag() {
                ChangeTag::Delete => stats.deletions += 1,
                ChangeTag::Insert => stats.additions += 1,
                ChangeTag::Equal => stats.unchanged += 1,
            }
        }

        stats
    }

    /// Format as a short summary string
    pub fn summary(&self) -> String {
        format!(
            "{}+{}{} {}-{}{}",
            GREEN, self.additions, RESET, RED, self.deletions, RESET
        )
    }
}

/// Display file diff between two files
pub async fn diff_files(old_path: &Path, new_path: &Path) -> anyhow::Result<String> {
    let old_content = tokio::fs::read_to_string(old_path).await?;
    let new_content = tokio::fs::read_to_string(new_path).await?;

    let old_label = old_path.display().to_string();
    let new_label = new_path.display().to_string();

    Ok(unified_diff(
        &old_content,
        &new_content,
        &old_label,
        &new_label,
    ))
}

/// Display diff for a file modification (old content vs new content)
pub fn diff_modification(path: &Path, old_content: &str, new_content: &str) -> String {
    let label = path.display().to_string();
    let old_label = format!("a/{}", label);
    let new_label = format!("b/{}", label);

    unified_diff(old_content, new_content, &old_label, &new_label)
}

/// Print a diff summary header
pub fn print_diff_header(path: &Path, stats: &DiffStats) {
    println!("{}{}{}  {}", BOLD, path.display(), RESET, stats.summary());
}

/// Interactive diff preview for file modifications
pub fn preview_diff(path: &Path, old_content: &str, new_content: &str) {
    let stats = DiffStats::calculate(old_content, new_content);
    print_diff_header(path, &stats);
    println!();
    print!(
        "{}",
        unified_diff(
            old_content,
            new_content,
            &format!("a/{}", path.display()),
            &format!("b/{}", path.display()),
        )
    );
}

/// Word-level diff for inline changes (shows character-level changes)
pub fn word_diff(old: &str, new: &str) -> String {
    let diff = TextDiff::from_words(old, new);
    let mut output = String::new();

    for change in diff.iter_all_changes() {
        match change.tag() {
            ChangeTag::Delete => {
                output.push_str(&format!("{}{}{}", RED, change.value(), RESET));
            }
            ChangeTag::Insert => {
                output.push_str(&format!("{}{}{}", GREEN, change.value(), RESET));
            }
            ChangeTag::Equal => {
                output.push_str(change.value());
            }
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_diff() {
        let old = "line1\nline2\nline3\n";
        let new = "line1\nmodified\nline3\n";

        let diff = unified_diff(old, new, "old.txt", "new.txt");
        assert!(diff.contains("--- old.txt"));
        assert!(diff.contains("+++ new.txt"));
        assert!(diff.contains("-line2"));
        assert!(diff.contains("+modified"));
    }

    #[test]
    fn test_diff_stats() {
        let old = "a\nb\nc\n";
        let new = "a\nx\ny\nc\n";

        let stats = DiffStats::calculate(old, new);
        assert_eq!(stats.deletions, 1);
        assert_eq!(stats.additions, 2);
        assert_eq!(stats.unchanged, 2);
    }

    #[test]
    fn test_compact_diff() {
        let old = "line1\nline2\nline3\n";
        let new = "line1\nmodified\nline3\n";

        let diff = compact_diff(old, new);
        assert!(!diff.contains("line1")); // Unchanged lines not shown
        assert!(diff.contains("line2")); // Deleted
        assert!(diff.contains("modified")); // Added
    }
}
