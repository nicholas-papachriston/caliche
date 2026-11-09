#[cfg(feature = "tools-grep")]
use crate::agent::tools::common::{params, path};
#[cfg(feature = "tools-grep")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-grep")]
use async_trait::async_trait;
#[cfg(feature = "tools-grep")]
use ignore::WalkBuilder;
#[cfg(feature = "tools-grep")]
use serde_json::json;

const NAME: &str = "glob";
const DESCRIPTION: &str = "Find files by glob-like name/path pattern, respecting gitignore.";
/// Banner unit when the character budget hides glob paths.
#[cfg(feature = "tools-grep")]
const PATHS_UNIT: &str = "paths";
/// Banner hint when the character budget hides glob paths.
#[cfg(feature = "tools-grep")]
const PAGE_HINT: &str = "use a narrower pattern or path to page";

#[derive(Debug)]
pub struct GlobTool;

#[cfg(feature = "tools-grep")]
impl ToolMeta for GlobTool {
    fn name(&self) -> &str {
        NAME
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory to search from.", "default": "." },
                "pattern": { "type": "string", "description": "Glob-like pattern, e.g. *.rs, **/*.ts, src/**/mod.rs." },
                "max_matches": { "type": "number", "description": "Maximum paths to return.", "default": 100 },
                "include_hidden": { "type": "boolean", "description": "Include hidden files and dirs.", "default": false }
            },
            "required": ["pattern"]
        })
    }
}

/// Return the character budget for glob output based on how many files matched.
///
/// Mirrors the tiers used in grep: fewer files → richer output; many files → lean.
#[cfg(feature = "tools-grep")]
fn glob_chars_budget(matched_count: usize) -> usize {
    if matched_count <= 3 {
        5_000
    } else if matched_count <= 8 {
        3_500
    } else {
        2_500
    }
}

#[cfg(feature = "tools-grep")]
#[async_trait]
impl Tool for GlobTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let pattern = match params::extract_string(&arguments, "pattern", None) {
            Ok(pattern) => pattern,
            Err(e) => return ToolResult::error(e),
        };
        let path_str = params::extract_string_or(&arguments, "path", ".");
        let max_matches =
            params::extract_number_or(&arguments, "max_matches", 100).clamp(1, 2_000) as usize;
        let include_hidden = params::extract_bool(&arguments, "include_hidden", false);

        let root = match path::validate_and_resolve_path(&path_str, Some(false)).await {
            Ok(root) => root,
            Err(e) => return ToolResult::error(e),
        };

        let mut perf = crate::observability::perf::span("glob.search");
        let pattern_for_worker = pattern.clone();
        let worker = tokio::task::spawn_blocking(move || {
            glob_walk(&root, &pattern_for_worker, max_matches, include_hidden)
        });
        let walked = match worker.await {
            Ok(walked) => walked,
            Err(e) => return ToolResult::error(format!("glob search worker failed: {e}")),
        };
        let mut matches = walked.matches;
        let scanned = walked.scanned;
        let truncated = walked.truncated;
        let errors = walked.errors;
        if let Some(ref mut span) = perf {
            span.set_file_count(scanned as u64);
            span.set_match_count(matches.len() as u64);
        }
        drop(perf);

        matches.sort();
        let total_count = matches.len();
        let budget = glob_chars_budget(total_count);

        // Walk matches once to find how many fit in the budget.
        let mut shown_count = 0usize;
        let mut chars_used = 0usize;
        for m in &matches {
            let entry_len = m.chars().count() + 1; // +1 for newline
            if shown_count > 0 && chars_used + entry_len > budget {
                break;
            }
            chars_used += entry_len;
            shown_count += 1;
        }

        let content = if matches.is_empty() {
            format!(
                "No files matched pattern '{}' under '{}' ({} file{} scanned).",
                pattern,
                path_str,
                scanned,
                if scanned == 1 { "" } else { "s" }
            )
        } else {
            let shown_paths = matches[..shown_count].join("\n");
            let mut content = format!(
                "Matched {} file{} for pattern '{}' under '{}':\n{}",
                total_count,
                if total_count == 1 { "" } else { "s" },
                pattern,
                path_str,
                shown_paths,
            );
            if shown_count < total_count {
                // Overflow is kept in the tool result. A later change can spill it to read_output.
                content.push_str(&super::grep::budget_banner(
                    shown_count,
                    total_count,
                    PATHS_UNIT,
                    PAGE_HINT,
                ));
            } else if truncated {
                content.push_str(&format!(
                    "\n\n[glob truncated at {max_matches} matches; narrow pattern or raise max_matches]"
                ));
            }
            if !errors.is_empty() {
                content.push_str(&format!(
                    "\n\n[walk warnings: {}]",
                    errors.into_iter().take(5).collect::<Vec<_>>().join("; ")
                ));
            }
            content
        };

        ToolResult::success_with_payload(
            content,
            json!({
                "op": "glob",
                "path": path_str,
                "pattern": pattern,
                "matches": matches,
                "match_count": total_count,
                "files_scanned": scanned,
                "truncated": truncated,
                "shown_count": shown_count,
                "chars_budget": budget,
            }),
        )
    }
}

#[cfg(feature = "tools-grep")]
struct GlobWalk {
    matches: Vec<String>,
    scanned: usize,
    truncated: bool,
    errors: Vec<String>,
}

#[cfg(feature = "tools-grep")]
fn glob_walk(
    root: &std::path::Path,
    pattern: &str,
    max_matches: usize,
    include_hidden: bool,
) -> GlobWalk {
    let compiled_pattern = GlobMatcher::new(pattern);
    let mut matches = Vec::new();
    let mut scanned = 0usize;
    let mut truncated = false;
    let mut errors = Vec::new();

    let mut walker = WalkBuilder::new(root);
    walker.hidden(!include_hidden);
    walker.git_ignore(true);
    walker.git_exclude(true);
    walker.ignore(true);

    for entry in walker.build() {
        let entry = match entry {
            Ok(entry) => entry,
            Err(e) => {
                errors.push(e.to_string());
                continue;
            }
        };
        if entry.file_type().is_some_and(|ft| ft.is_dir()) {
            continue;
        }
        scanned = scanned.saturating_add(1);
        let entry_path = entry.path();
        let rel = entry_path.strip_prefix(root).unwrap_or(entry_path);
        let rel_display = rel.to_string_lossy();
        if compiled_pattern.matches(&rel_display) {
            if matches.len() >= max_matches {
                truncated = true;
                break;
            }
            matches.push(rel_display.to_string());
        }
    }

    GlobWalk {
        matches,
        scanned,
        truncated,
        errors,
    }
}

#[cfg(feature = "tools-grep")]
struct GlobMatcher {
    pattern: String,
    basename_pattern: Option<String>,
}

#[cfg(feature = "tools-grep")]
impl GlobMatcher {
    fn new(pattern: &str) -> Self {
        let normalized = pattern.replace('\\', "/");
        let basename_pattern = (!normalized.contains('/')).then(|| normalized.clone());
        Self {
            pattern: normalized,
            basename_pattern,
        }
    }

    fn matches(&self, path: &str) -> bool {
        let path = path.replace('\\', "/");
        if wildcard_match(&self.pattern, &path) {
            return true;
        }
        if let Some(pattern) = self.pattern.strip_prefix("**/")
            && wildcard_match(pattern, &path)
        {
            return true;
        }
        if let Some(pattern) = self.basename_pattern.as_ref() {
            let file_name = path.rsplit('/').next().unwrap_or(path.as_str());
            return wildcard_match(pattern, file_name);
        }
        false
    }
}

#[cfg(feature = "tools-grep")]
fn wildcard_match(pattern: &str, text: &str) -> bool {
    let pattern = pattern.as_bytes();
    let text = text.as_bytes();
    let (mut pi, mut ti) = (0usize, 0usize);
    let mut star = None;
    let mut match_after_star = 0usize;

    while ti < text.len() {
        if pi < pattern.len() && (pattern[pi] == b'?' || pattern[pi] == text[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pattern.len() && pattern[pi] == b'*' {
            star = Some(pi);
            pi += 1;
            match_after_star = ti;
        } else if let Some(star_idx) = star {
            pi = star_idx + 1;
            match_after_star += 1;
            ti = match_after_star;
        } else {
            return false;
        }
    }

    while pi < pattern.len() && pattern[pi] == b'*' {
        pi += 1;
    }
    pi == pattern.len()
}

#[cfg(all(test, feature = "tools-grep"))]
mod tests {
    use super::{GlobMatcher, GlobTool, glob_chars_budget};
    use crate::agent::tools::{Tool, ToolResult};
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn glob_matcher_handles_common_patterns() {
        assert!(GlobMatcher::new("*.rs").matches("src/lib.rs"));
        assert!(GlobMatcher::new("**/*.rs").matches("src/agent/tools/fs_tool.rs"));
        assert!(GlobMatcher::new("src/**/fs_tool.rs").matches("src/agent/tools/fs_tool.rs"));
        assert!(!GlobMatcher::new("*.ts").matches("src/lib.rs"));
    }

    // ── chars_budget tiers ────────────────────────────────────────────────────

    #[test]
    fn chars_budget_tier_small() {
        assert_eq!(glob_chars_budget(0), 5_000);
        assert_eq!(glob_chars_budget(1), 5_000);
        assert_eq!(glob_chars_budget(3), 5_000);
    }

    #[test]
    fn chars_budget_tier_medium() {
        assert_eq!(glob_chars_budget(4), 3_500);
        assert_eq!(glob_chars_budget(8), 3_500);
    }

    #[test]
    fn chars_budget_tier_large() {
        assert_eq!(glob_chars_budget(9), 2_500);
        assert_eq!(glob_chars_budget(1_000), 2_500);
    }

    // ── output budget (end to end) ────────────────────────────────────────────

    const MANY_PATHS: usize = 60;

    /// 60 sorted paths of 88 chars each: 60 * 89 = 5_340 chars, over the
    /// 2_500-char budget for more than 8 matches.
    fn long_path_name(i: usize) -> String {
        format!("{i:03}_{}.txt", "p".repeat(80))
    }

    #[expect(
        clippy::expect_used,
        reason = "test fixture setup; a failure is a broken test env"
    )]
    async fn glob_many_long_paths() -> (TempDir, ToolResult) {
        let dir = TempDir::new().expect("tempdir");
        for i in 0..MANY_PATHS {
            std::fs::write(dir.path().join(long_path_name(i)), "x").expect("write");
        }
        let result = GlobTool
            .execute(json!({
                "path": dir.path().to_string_lossy(),
                "pattern": "*.txt",
            }))
            .await;
        assert!(result.success, "{:?}", result.error);
        (dir, result)
    }

    fn shown_paths(content: &str) -> Vec<&str> {
        content
            .lines()
            .filter(|line| line.ends_with(".txt"))
            .collect()
    }

    #[tokio::test]
    async fn glob_output_stops_at_char_budget() {
        let (_dir, result) = glob_many_long_paths().await;
        let shown = shown_paths(&result.content);
        let budget = glob_chars_budget(MANY_PATHS);
        let entry_len = long_path_name(0).chars().count() + 1;

        assert!(
            result
                .content
                .starts_with(&format!("Matched {MANY_PATHS} files")),
            "{}",
            result.content
        );
        assert_eq!(
            shown.len(),
            budget / entry_len,
            "budget must fill with whole paths and stop"
        );
        let expected: Vec<String> = (0..shown.len()).map(long_path_name).collect();
        assert_eq!(shown, expected, "the first paths in sorted order are shown");
        let used: usize = shown.iter().map(|p| p.chars().count() + 1).sum();
        assert!(used <= budget, "{used} chars of paths over {budget} budget");
    }

    #[tokio::test]
    async fn glob_budget_banner_reports_shown_and_total() {
        let (_dir, result) = glob_many_long_paths().await;
        assert_eq!(shown_paths(&result.content).len(), 28, "{}", result.content);
        assert!(
            result
                .content
                .ends_with("\n\n[showing 28/60 paths — use a narrower pattern or path to page]"),
            "{}",
            result.content
        );
    }

    #[tokio::test]
    async fn glob_has_no_banner_when_all_paths_fit() {
        let dir = TempDir::new().expect("tempdir");
        for name in ["a.rs", "b.rs"] {
            std::fs::write(dir.path().join(name), "x").expect("write");
        }
        let result = GlobTool
            .execute(json!({ "path": dir.path().to_string_lossy(), "pattern": "*.rs" }))
            .await;
        assert!(
            result.content.ends_with(":\na.rs\nb.rs"),
            "all paths and no banner expected: {}",
            result.content
        );
    }
}
