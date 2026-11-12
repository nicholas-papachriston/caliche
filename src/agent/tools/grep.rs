#[cfg(feature = "tools-grep")]
use crate::agent::tools::common::{params, path};
#[cfg(feature = "tools-grep")]
use crate::agent::tools::hashline;
#[cfg(feature = "tools-grep")]
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
#[cfg(feature = "tools-grep")]
use async_trait::async_trait;
#[cfg(feature = "tools-grep")]
use serde_json::json;
#[cfg(feature = "tools-grep")]
use std::collections::VecDeque;
#[cfg(feature = "tools-grep")]
use std::io::{BufRead, BufReader, ErrorKind, Read};
use std::path::{Path, PathBuf};

#[cfg(feature = "tools-grep")]
use grep_matcher::Matcher;
#[cfg(feature = "tools-grep")]
use grep_regex::RegexMatcher;
#[cfg(feature = "tools-grep")]
use ignore::WalkBuilder;

const NAME: &str = "grep";
const DESCRIPTION: &str =
    "Search for text patterns in files using regex. Supports recursive search and context lines.";
#[derive(Debug)]
pub struct GrepTool;

#[cfg(feature = "tools-grep")]
impl ToolMeta for GrepTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for. Use standard regex syntax. For literal strings, escape special characters."
                },
                "path": {
                    "type": "string",
                    "description": "The file or directory path to search in (default: current directory '.')",
                    "default": "."
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Whether to search recursively in subdirectories (default: true)",
                    "default": true
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional file filter: extension (`rs`, `.rs`, `*.rs`), basename (`lib.rs`), or path suffix (`src/lib.rs`)."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Whether the search is case-sensitive (default: true)",
                    "default": true
                },
                "max_matches": {
                    "type": "number",
                    "description": "Maximum number of matches to return (default: 100, prevents excessive output)",
                    "default": 100
                },
                "context_lines": {
                    "type": "number",
                    "description": "Number of context lines to show before and after each match (default: 2)",
                    "default": 2
                },
                "mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode.",
                    "default": "content"
                },
                "offset": {
                    "type": "number",
                    "description": "Skip this many matching files/results.",
                    "default": 0
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Search hidden files and directories (default: false). `.git` / `.hg` / `.svn` stay skipped.",
                    "default": false
                },
                "max_file_size": {
                    "type": "number",
                    "description": "Maximum file size in bytes to search (default: 10485760 = 10MB). Larger files will be skipped.",
                    "default": 10_485_760
                }
            },
            "required": ["pattern"]
        })
    }
}

#[async_trait]
impl Tool for GrepTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let pattern_str = match params::extract_string(&arguments, "pattern", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        let path_str = params::extract_string_or(&arguments, "path", ".");
        let recursive = params::extract_bool(&arguments, "recursive", true);
        let file_pattern = arguments["file_pattern"]
            .as_str()
            .map(std::string::ToString::to_string);
        let case_sensitive = params::extract_bool(&arguments, "case_sensitive", true);
        let max_matches = params::extract_number_or(&arguments, "max_matches", 100) as usize;
        let context_lines = params::extract_number_or(&arguments, "context_lines", 2) as usize;
        let mode = params::extract_string_or(&arguments, "mode", "content");
        let offset = params::extract_number_or(&arguments, "offset", 0) as usize;
        let include_hidden = params::extract_bool(&arguments, "include_hidden", false);
        let max_file_size = params::extract_number_or(&arguments, "max_file_size", 10_485_760);

        let attempts = match super::grep_pattern::grep_attempts(&pattern_str) {
            Ok(attempts) => attempts,
            Err(error) => return ToolResult::error(error),
        };

        // Validate and resolve path
        let normalized_path = match path::validate_and_resolve_path(&path_str, None).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(e),
        };

        let mut last_failure: Option<String> = None;
        for (index, attempt) in attempts.iter().enumerate() {
            let pattern_to_use = if case_sensitive {
                attempt.source.clone()
            } else {
                format!("(?i){}", attempt.source)
            };
            let compiled_pattern = match RegexMatcher::new(&pattern_to_use) {
                Ok(matcher) => matcher,
                Err(error) => {
                    last_failure = Some(format!(
                        "Invalid regex pattern '{pattern_str}': {error}. \
                         Quote the text for a literal search, or escape metacharacters."
                    ));
                    continue;
                }
            };
            let run = match run_grep_worker(GrepSearchJob {
                compiled_pattern,
                normalized_path: normalized_path.clone(),
                recursive,
                file_pattern: file_pattern.clone(),
                context_lines,
                max_file_size,
                max_matches,
                include_hidden,
            })
            .await
            {
                Ok(run) => run,
                Err(error) => return ToolResult::error(error),
            };
            let retry =
                index + 1 < attempts.len() && run.matches.is_empty() && run.walk_errors.is_empty();
            if retry {
                continue;
            }
            let mut formatted = format_grep_results(GrepFormatInput {
                path_str,
                pattern_str,
                mode,
                offset,
                max_matches,
                matches: run.matches,
                total_matches: run.total_matches,
                lines_scanned: run.lines_scanned,
                files_scanned: run.files_scanned,
                walk_errors: run.walk_errors,
            });
            if formatted.success
                && let Some(note) = attempt.note.as_deref()
            {
                formatted.content.push_str("\n\n[");
                formatted.content.push_str(note);
                formatted.content.push(']');
            }
            return formatted;
        }

        ToolResult::error(last_failure.unwrap_or_else(|| {
            format!(
                "Invalid regex pattern '{pattern_str}'. \
                 Quote the text for a literal search, or escape metacharacters."
            )
        }))
    }
}

#[cfg(feature = "tools-grep")]
async fn run_grep_worker(job: GrepSearchJob) -> Result<GrepRun, String> {
    let mut perf = crate::observability::perf::span("grep.search");
    let worker = tokio::task::spawn_blocking(move || run_grep_search(job));
    let run = match worker.await {
        Ok(Ok(run)) => run,
        Ok(Err(error)) => return Err(error),
        Err(error) => return Err(format!("grep search worker failed: {error}")),
    };
    if let Some(ref mut span) = perf {
        span.set_file_count(run.files_scanned as u64);
        span.set_match_count(run.total_matches as u64);
        span.set_line_count(run.lines_scanned as u64);
    }
    Ok(run)
}

#[cfg(feature = "tools-grep")]
struct GrepFormatInput {
    path_str: String,
    pattern_str: String,
    mode: String,
    offset: usize,
    max_matches: usize,
    matches: Vec<String>,
    total_matches: usize,
    lines_scanned: usize,
    files_scanned: usize,
    walk_errors: Vec<String>,
}

#[cfg(feature = "tools-grep")]
fn format_grep_results(mut input: GrepFormatInput) -> ToolResult {
    if !input.walk_errors.is_empty() && input.matches.is_empty() {
        return ToolResult::error(format!(
            "Error searching directory '{}': {}",
            input.path_str,
            input.walk_errors.join("; ")
        ));
    }
    if !input.walk_errors.is_empty() {
        input.matches.push(format!(
            "\n[Walk warnings — some files may have been skipped: {}]",
            input.walk_errors.join("; ")
        ));
    }
    if input.matches.is_empty() {
        return format_grep_empty(&input);
    }
    let matched_files = unique_match_files(&input.matches);
    if input.mode == "count" {
        return format_grep_count(&input, &matched_files);
    }
    if input.mode == "files_with_matches" {
        return format_grep_files(&input, &matched_files);
    }
    format_grep_content(input, matched_files)
}

#[cfg(feature = "tools-grep")]
fn format_grep_empty(input: &GrepFormatInput) -> ToolResult {
    ToolResult::success(format!(
        "No matches found for pattern '{}' in '{}' \
         ({} file{} searched, {} line{} scanned). \
         The path was valid and fully searched — the pattern is simply not present. \
         Try a broader or different pattern (e.g. a method name, class name, or substring).",
        input.pattern_str,
        input.path_str,
        input.files_scanned,
        if input.files_scanned == 1 { "" } else { "s" },
        input.lines_scanned,
        if input.lines_scanned == 1 { "" } else { "s" },
    ))
}

#[cfg(feature = "tools-grep")]
fn format_grep_count(input: &GrepFormatInput, matched_files: &[String]) -> ToolResult {
    ToolResult::success_with_payload(
        format!(
            "{} match{} across {} file{} for pattern '{}' in '{}'",
            input.total_matches,
            if input.total_matches == 1 { "" } else { "es" },
            matched_files.len(),
            if matched_files.len() == 1 { "" } else { "s" },
            input.pattern_str,
            input.path_str
        ),
        json!({
            "op": "grep",
            "path": input.path_str,
            "pattern": input.pattern_str,
            "mode": input.mode,
            "match_count": input.total_matches,
            "file_count": matched_files.len(),
            "files_scanned": input.files_scanned,
            "lines_scanned": input.lines_scanned,
        }),
    )
}

/// Longest grep output line, in chars, after match-centred trimming.
#[cfg(feature = "tools-grep")]
const LINE_LIMIT: usize = 300;

/// Marks the side of a grep line that match-centred trimming cut.
#[cfg(feature = "tools-grep")]
const TRIM_MARK: &str = "…";

/// Banner unit for `files_with_matches` output.
#[cfg(feature = "tools-grep")]
const FILES_UNIT: &str = "files";

/// Banner unit for `content` output.
#[cfg(feature = "tools-grep")]
const BLOCKS_UNIT: &str = "match blocks";

/// Banner appended when the character budget hides `total - shown` entries.
#[cfg(feature = "tools-grep")]
pub(super) fn budget_banner(shown: usize, total: usize, unit: &str, hint: &str) -> String {
    format!("\n\n[showing {shown}/{total} {unit} — {hint}]")
}

/// Paging hint for grep banners: the offset that shows the next entry.
#[cfg(feature = "tools-grep")]
fn offset_hint(next_offset: usize) -> String {
    format!("use offset={next_offset} to page, or narrow the pattern/path")
}

/// Return the character budget for grep/glob output based on how many files matched.
///
/// Fewer matched files means richer per-file context is more useful; more files
/// means we need to stay lean so the model's context does not flood.
#[cfg(feature = "tools-grep")]
fn chars_budget(matched_file_count: usize) -> usize {
    if matched_file_count <= 3 {
        5_000
    } else if matched_file_count <= 8 {
        3_500
    } else {
        2_500
    }
}

/// Trim a single grep output line to at most `line_limit` chars, windowed on
/// the first occurrence of `pattern` in the content portion of the line.
///
/// The line format is `path:lineno:hash: content` (with optional leading
/// `- ` / `+ ` for context lines). We preserve the prefix verbatim and
/// window only the content portion. A third of the budget goes before the
/// match and two thirds after, both cut on UTF-8 char boundaries, and each
/// cut is marked with `…`.
///
/// When `pattern` is not found in the content, the line is simply truncated
/// at `line_limit` chars from the right.
#[cfg(feature = "tools-grep")]
fn trim_line_to_window(line: &str, pattern: &str, line_limit: usize) -> String {
    if line.chars().count() <= line_limit {
        return line.to_owned();
    }

    // Find where the content portion starts (after the hashline prefix).
    // Hashline format: `path:lineno:hash: content` or `- path:...` / `+ path:...`
    // We locate the `: ` separator that ends the hash field.
    let content_start = line.find(": ").map(|i| i + 2).unwrap_or(0);
    let prefix = &line[..content_start];
    let content = &line[content_start..];

    // Locate the pattern match inside the content (byte offset).
    let match_byte = if pattern.is_empty() {
        0
    } else {
        content.find(pattern).unwrap_or(0)
    };

    // Budget: 1/3 before, 2/3 after.
    let prefix_char_len = prefix.chars().count();
    // Reserve one char for each cut mark so the result stays within `line_limit`.
    let content_budget = line_limit
        .saturating_sub(prefix_char_len)
        .saturating_sub(2 * TRIM_MARK.chars().count());
    let before_budget = content_budget / 3;
    let after_budget = content_budget.saturating_sub(before_budget);

    // Convert the match byte offset to a char offset.
    let match_char = content[..match_byte].chars().count();

    // Window start/end in char positions of `content`.
    let content_chars: usize = content.chars().count();
    let window_start_char = match_char.saturating_sub(before_budget);
    let window_end_char = (match_char + after_budget).min(content_chars);

    // Slice on char boundaries.
    let byte_start = content
        .char_indices()
        .nth(window_start_char)
        .map(|(b, _)| b)
        .unwrap_or(0);
    let byte_end = content
        .char_indices()
        .nth(window_end_char)
        .map(|(b, _)| b)
        .unwrap_or(content.len());

    let windowed = &content[byte_start..byte_end];
    let left_ellipsis = if window_start_char > 0 { TRIM_MARK } else { "" };
    let right_ellipsis = if window_end_char < content_chars {
        TRIM_MARK
    } else {
        ""
    };

    format!("{prefix}{left_ellipsis}{windowed}{right_ellipsis}")
}

#[cfg(feature = "tools-grep")]
fn format_grep_files(input: &GrepFormatInput, matched_files: &[String]) -> ToolResult {
    let all_files: Vec<&String> = matched_files.iter().skip(input.offset).collect();
    let total_count = all_files.len();
    let budget = chars_budget(total_count);

    let mut shown: Vec<&str> = Vec::new();
    let mut chars_used = 0usize;
    for file in &all_files {
        let entry_len = file.chars().count() + 1; // +1 for newline
        if !shown.is_empty() && chars_used + entry_len > budget {
            break;
        }
        shown.push(file.as_str());
        chars_used += entry_len;
    }
    let shown_count = shown.len();

    let mut body = format!(
        "{} file{} matched pattern '{}' in '{}':\n{}",
        total_count,
        if total_count == 1 { "" } else { "s" },
        input.pattern_str,
        input.path_str,
        shown.join("\n")
    );
    if shown_count < total_count {
        // Overflow is kept in the tool result. A later change can spill it to read_output.
        body.push_str(&budget_banner(
            shown_count,
            total_count,
            FILES_UNIT,
            &offset_hint(input.offset + shown_count),
        ));
    }

    ToolResult::success_with_payload(
        body,
        json!({
            "op": "grep",
            "path": input.path_str,
            "pattern": input.pattern_str,
            "mode": input.mode,
            "files": shown,
            "file_count": matched_files.len(),
            "files_scanned": input.files_scanned,
            "lines_scanned": input.lines_scanned,
            "offset": input.offset,
            "shown_count": shown_count,
            "chars_budget": budget,
        }),
    )
}

#[cfg(feature = "tools-grep")]
fn format_grep_content(input: GrepFormatInput, matched_files: Vec<String>) -> ToolResult {
    let content_matches: Vec<String> = if input.offset > 0 {
        input.matches.into_iter().skip(input.offset).collect()
    } else {
        input.matches
    };

    let file_count = matched_files.len();
    let budget = chars_budget(file_count);

    let mut output = format!(
        "Found {} match{} for pattern '{}' in '{}':\n\n",
        input.total_matches,
        if input.total_matches == 1 { "" } else { "es" },
        input.pattern_str,
        input.path_str
    );
    if input.total_matches > input.max_matches {
        output.push_str(&format!(
            "[truncated] Showing first {} matches (total: {})\n\n",
            input.max_matches, input.total_matches
        ));
    }

    let total_blocks = content_matches.len();
    let mut shown_count = 0usize;
    let mut chars_used = 0usize;
    let mut blocks: Vec<String> = Vec::with_capacity(total_blocks);

    for block in &content_matches {
        // Apply per-line window trimming inside this block.
        let trimmed_block: String = block
            .lines()
            .map(|line| trim_line_to_window(line, &input.pattern_str, LINE_LIMIT))
            .collect::<Vec<_>>()
            .join("\n");

        let block_len = trimmed_block.chars().count() + 2; // +2 for "\n\n" separator
        if shown_count > 0 && chars_used + block_len > budget {
            break;
        }
        chars_used += block_len;
        shown_count += 1;
        blocks.push(trimmed_block);
    }

    output.push_str(&blocks.join("\n\n"));

    if shown_count < total_blocks {
        // Overflow is kept in the tool result. A later change can spill it to read_output.
        output.push_str(&budget_banner(
            shown_count,
            total_blocks,
            BLOCKS_UNIT,
            &offset_hint(input.offset + shown_count),
        ));
    }

    ToolResult::success_with_payload(
        output,
        json!({
            "op": "grep",
            "path": input.path_str,
            "pattern": input.pattern_str,
            "mode": input.mode,
            "match_count": input.total_matches,
            "returned_matches": content_matches.len(),
            "files": matched_files,
            "files_scanned": input.files_scanned,
            "lines_scanned": input.lines_scanned,
            "offset": input.offset,
            "truncated": input.total_matches > input.max_matches,
            "shown_count": shown_count,
            "chars_budget": budget,
        }),
    )
}

#[cfg(feature = "tools-grep")]
struct GrepSearchJob {
    compiled_pattern: RegexMatcher,
    normalized_path: PathBuf,
    recursive: bool,
    file_pattern: Option<String>,
    context_lines: usize,
    max_file_size: u64,
    max_matches: usize,
    include_hidden: bool,
}

#[cfg(feature = "tools-grep")]
struct GrepRun {
    matches: Vec<String>,
    total_matches: usize,
    lines_scanned: usize,
    files_scanned: usize,
    walk_errors: Vec<String>,
}

#[cfg(feature = "tools-grep")]
fn run_grep_search(job: GrepSearchJob) -> Result<GrepRun, String> {
    let metadata = std::fs::metadata(&job.normalized_path).map_err(|e| {
        format!(
            "Failed to access path '{}': {e}",
            job.normalized_path.display()
        )
    })?;
    let is_file = metadata.is_file();
    if !is_file && !metadata.is_dir() {
        return Err(format!(
            "Path '{}' is neither a file nor a directory",
            job.normalized_path.display()
        ));
    }

    let mut total_matches = 0usize;
    let mut lines_scanned = 0usize;
    let mut files_scanned = 0usize;
    let mut matches = Vec::new();
    let mut walk_errors = Vec::new();

    if is_file {
        match GrepTool::search_file(
            None,
            &job.normalized_path,
            &job.compiled_pattern,
            job.context_lines,
            job.max_file_size,
            &mut total_matches,
            job.max_matches,
            &mut lines_scanned,
        ) {
            Ok(file_matches) => {
                files_scanned += 1;
                matches.extend(file_matches);
            }
            Err(e) => {
                return Err(format!(
                    "Error searching file '{}': {e}",
                    job.normalized_path.display()
                ));
            }
        }
    } else {
        let options = SearchDirectoryOptions {
            matcher: &job.compiled_pattern,
            recursive: job.recursive,
            file_pattern: job.file_pattern.as_deref(),
            context_lines: job.context_lines,
            max_file_size: job.max_file_size,
            max_matches: job.max_matches,
            include_hidden: job.include_hidden,
        };
        let mut counters = SearchCounters {
            total_matches: &mut total_matches,
            lines_scanned: &mut lines_scanned,
            files_scanned: &mut files_scanned,
        };
        match GrepTool::search_directory(&job.normalized_path, options, &mut counters) {
            Ok(dir_matches) => matches.extend(dir_matches),
            Err((dir_matches, errors)) => {
                matches.extend(dir_matches);
                walk_errors = errors;
            }
        }
    }

    Ok(GrepRun {
        matches,
        total_matches,
        lines_scanned,
        files_scanned,
        walk_errors,
    })
}

/// Match lines are `{path}:{line}:{hash}: {content}`. Split from the right so
/// Windows drive letters in `path` stay intact. Skip banners and context lines
/// (`[truncated]`, `- …`, `+ …`).
#[cfg(feature = "tools-grep")]
fn unique_match_files(matches: &[String]) -> Vec<String> {
    let mut files = std::collections::BTreeSet::new();
    for item in matches {
        for line in item.lines() {
            if let Some(path) = match_line_path(line) {
                files.insert(path.to_string());
                break;
            }
        }
    }
    files.into_iter().collect()
}

/// Extract `{path}` from a grep match line `{path}:{line}:{hash}: {content}`.
#[cfg(feature = "tools-grep")]
fn match_line_path(line: &str) -> Option<&str> {
    if line.is_empty() || line.starts_with('[') || line.starts_with("- ") || line.starts_with("+ ")
    {
        return None;
    }
    let mut parts = line.rsplitn(4, ':');
    let _content = parts.next()?;
    let hash = parts.next()?;
    let lineno = parts.next()?;
    let path = parts.next()?;
    if path.is_empty() {
        return None;
    }
    if !matches!(hash.len(), 2 | 4) || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    lineno.parse::<usize>().ok()?;
    Some(path)
}

/// `file_pattern` is an extension (`rs`, `.rs`, `*.rs`), a basename (`lib.rs`),
/// or a path suffix (`src/lib.rs`). Bare `rs` must not match a file named `yours`.
#[cfg(feature = "tools-grep")]
fn path_matches_file_pattern(path: &Path, pattern: &str) -> bool {
    if pattern.is_empty() {
        return true;
    }
    if let Some(rest) = pattern.strip_prefix("**/") {
        return path_matches_file_pattern(path, rest);
    }
    if pattern.contains('/') || pattern.contains('\\') {
        let normalized = path.to_string_lossy().replace('\\', "/");
        let want = pattern.replace('\\', "/");
        return normalized.ends_with(&want);
    }
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name == pattern {
        return true;
    }
    if let Some(ext) = pattern.strip_prefix("*.") {
        return path.extension().and_then(|e| e.to_str()) == Some(ext);
    }
    if let Some(ext) = pattern.strip_prefix('.')
        && !ext.is_empty()
        && !ext.contains('.')
    {
        return path.extension().and_then(|e| e.to_str()) == Some(ext);
    }
    if pattern.contains('.') {
        return name == pattern;
    }
    path.extension().and_then(|e| e.to_str()) == Some(pattern)
}

#[cfg(feature = "tools-grep")]
fn is_skipped_vcs_dir(entry: &ignore::DirEntry) -> bool {
    entry
        .file_type()
        .is_some_and(|file_type| file_type.is_dir())
        && matches!(entry.file_name().to_str(), Some(".git" | ".hg" | ".svn"))
}

#[cfg(feature = "tools-grep")]
struct SearchDirectoryOptions<'a> {
    matcher: &'a RegexMatcher,
    recursive: bool,
    file_pattern: Option<&'a str>,
    context_lines: usize,
    max_file_size: u64,
    max_matches: usize,
    include_hidden: bool,
}

#[cfg(feature = "tools-grep")]
struct SearchCounters<'a> {
    total_matches: &'a mut usize,
    lines_scanned: &'a mut usize,
    files_scanned: &'a mut usize,
}

#[cfg(feature = "tools-grep")]
impl GrepTool {
    /// Search `file_path`, opened below `root` without following a link.
    ///
    /// `root` is the walked directory, or `None` for a file the caller named
    /// and resolved, which is opened from the file system root.
    fn search_file(
        root: Option<&Path>,
        file_path: &Path,
        compiled_pattern: &RegexMatcher,
        context_lines: usize,
        max_file_size: u64,
        total_matches: &mut usize,
        max_matches: usize,
        lines_scanned: &mut usize,
    ) -> Result<Vec<String>, String> {
        let opened = match root {
            Some(root) => path::open_walked_file(root, file_path),
            None => path::open_resolved_file(file_path),
        };
        let file = match opened {
            Ok(file) => file,
            Err(e) if e.kind() == ErrorKind::InvalidData => return Ok(vec![]),
            Err(e) => return Err(format!("Failed to read file: {e}")),
        };
        let metadata = file
            .metadata()
            .map_err(|e| format!("Failed to get file metadata: {e}"))?;
        if metadata.len() > max_file_size {
            return Err(format!(
                "file too large to search ({} bytes, max {max_file_size}): {}",
                metadata.len(),
                file_path.display()
            ));
        }
        let path_str = file_path.display().to_string();
        search_reader(
            file,
            compiled_pattern,
            context_lines,
            total_matches,
            max_matches,
            lines_scanned,
            &path_str,
        )
    }

    fn search_directory(
        dir_path: &Path,
        options: SearchDirectoryOptions<'_>,
        counters: &mut SearchCounters<'_>,
    ) -> Result<Vec<String>, (Vec<String>, Vec<String>)> {
        let mut matches = Vec::new();
        let mut errors = Vec::new();

        let mut walker = WalkBuilder::new(dir_path);
        walker.max_filesize(Some(options.max_file_size));
        walker.hidden(!options.include_hidden);
        walker.git_ignore(true);
        walker.git_exclude(true);
        walker.ignore(true);

        // Skip VCS metadata trees even when include_hidden is true. File
        // extension filtering happens in the walk loop: ignore's `max_filesize`
        // short-circuits `filter_entry` for files.
        walker.filter_entry(|entry| !is_skipped_vcs_dir(entry));

        if !options.recursive {
            walker.max_depth(Some(1));
        }

        for result in walker.build() {
            if *counters.total_matches >= options.max_matches {
                break;
            }

            let entry = match result {
                Ok(e) => e,
                Err(e) => {
                    errors.push(format!("Walk error: {e}"));
                    continue;
                }
            };

            let entry_path = entry.path();
            // The walk does not follow links, so a link reports its own
            // type here and is skipped with directories and other non-files.
            if !entry.file_type().is_some_and(|ft| ft.is_file()) {
                continue;
            }
            if options
                .file_pattern
                .is_some_and(|pat| !path_matches_file_pattern(entry_path, pat))
            {
                continue;
            }

            match Self::search_file(
                Some(dir_path),
                entry_path,
                options.matcher,
                options.context_lines,
                options.max_file_size,
                counters.total_matches,
                options.max_matches,
                counters.lines_scanned,
            ) {
                Ok(file_matches) => {
                    *counters.files_scanned += 1;
                    matches.extend(file_matches);
                }
                Err(e) => {
                    errors.push(format!("Error searching '{}': {}", entry_path.display(), e));
                }
            }
        }

        if errors.is_empty() {
            Ok(matches)
        } else {
            Err((matches, errors))
        }
    }
}

#[cfg(feature = "tools-grep")]
fn strip_line_ending(buf: &mut Vec<u8>) {
    if buf.last().copied() == Some(b'\n') {
        buf.pop();
    }
    if buf.last().copied() == Some(b'\r') {
        buf.pop();
    }
}

#[cfg(feature = "tools-grep")]
fn search_reader<R: Read>(
    file: R,
    compiled_pattern: &RegexMatcher,
    context_lines: usize,
    total_matches: &mut usize,
    max_matches: usize,
    lines_scanned: &mut usize,
    path_str: &str,
) -> Result<Vec<String>, String> {
    let budget = max_matches.saturating_sub(*total_matches);
    if budget == 0 {
        return Ok(Vec::new());
    }

    let mut reader = BufReader::with_capacity(64 * 1024, file);
    let mut buf = Vec::with_capacity(256);
    let mut matches: Vec<String> = Vec::new();
    let mut line_num = 0usize;
    let mut before: VecDeque<(usize, String)> = VecDeque::with_capacity(context_lines.max(1));
    let mut pending_after: Vec<(usize, usize)> = Vec::new();
    let mut local_matches = 0usize;
    let mut local_lines = 0usize;

    loop {
        if local_matches >= budget && pending_after.is_empty() {
            break;
        }
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) if e.kind() == ErrorKind::InvalidData => return Ok(Vec::new()),
            Err(e) => return Err(format!("Failed to read file: {e}")),
        }
        strip_line_ending(&mut buf);
        let Ok(line) = std::str::from_utf8(&buf) else {
            // Binary / invalid UTF-8: skip the file without publishing a partial
            // match set or inflating the shared counters.
            return Ok(Vec::new());
        };
        line_num = line_num.saturating_add(1);
        local_lines = local_lines.saturating_add(1);

        for (match_idx, remaining) in &mut pending_after {
            if *remaining == 0 {
                continue;
            }
            matches[*match_idx].push('\n');
            matches[*match_idx].push_str("+ ");
            matches[*match_idx].push_str(&hashline::format_grep_hashline(path_str, line_num, line));
            *remaining = remaining.saturating_sub(1);
        }
        pending_after.retain(|(_, remaining)| *remaining > 0);

        let found = compiled_pattern
            .find(line.as_bytes())
            .ok()
            .flatten()
            .is_some();
        if found && local_matches < budget {
            local_matches = local_matches.saturating_add(1);
            let mut match_output = String::new();
            if context_lines > 0 {
                for (ctx_num, ctx_line) in &before {
                    match_output.push_str("- ");
                    match_output.push_str(&hashline::format_grep_hashline(
                        path_str, *ctx_num, ctx_line,
                    ));
                    match_output.push('\n');
                }
                pending_after.push((matches.len(), context_lines));
            }
            match_output.push_str(&hashline::format_grep_hashline(path_str, line_num, line));
            matches.push(match_output);
        }

        if context_lines > 0 {
            if before.len() == context_lines {
                before.pop_front();
            }
            before.push_back((line_num, line.to_string()));
        }
    }

    *total_matches = total_matches.saturating_add(local_matches);
    *lines_scanned = lines_scanned.saturating_add(local_lines);
    Ok(matches)
}

#[cfg(test)]
#[cfg(feature = "tools-grep")]
mod tests {
    use super::{
        GrepTool, match_line_path, path_matches_file_pattern, search_reader, unique_match_files,
    };
    use crate::agent::tools::hashline;
    use grep_matcher::Matcher;
    use grep_regex::RegexMatcher;
    use std::io::Cursor;
    use std::path::{Path, PathBuf};

    fn matcher(pattern: &str) -> RegexMatcher {
        RegexMatcher::new(pattern).expect("valid regex")
    }

    fn search(
        input: &[u8],
        pattern: &str,
        context_lines: usize,
        max_matches: usize,
    ) -> (Vec<String>, usize, usize) {
        let mut total_matches = 0usize;
        let mut lines_scanned = 0usize;
        let matches = search_reader(
            Cursor::new(input),
            &matcher(pattern),
            context_lines,
            &mut total_matches,
            max_matches,
            &mut lines_scanned,
            "t.rs",
        )
        .expect("search");
        (matches, total_matches, lines_scanned)
    }

    fn match_lines(blocks: &[String]) -> Vec<String> {
        blocks
            .iter()
            .flat_map(|block| block.lines())
            .filter(|line| match_line_path(line).is_some())
            .map(str::to_string)
            .collect()
    }

    fn slurp_match_lines(input: &str, pattern: &str, max_matches: usize) -> Vec<String> {
        let compiled = matcher(pattern);
        let mut hits = Vec::new();
        for (idx, line) in input.lines().enumerate() {
            if hits.len() >= max_matches {
                break;
            }
            let found = compiled.find(line.as_bytes()).ok().flatten().is_some();
            if found {
                hits.push(hashline::format_grep_hashline("t.rs", idx + 1, line));
            }
        }
        hits
    }

    #[test]
    fn unique_match_files_uses_path_not_path_and_line() {
        let matches = vec![
            "src/lib.rs:10:ab: fn main()".to_string(),
            "src/lib.rs:20:cd: let x = 1".to_string(),
            "src/main.rs:1:ef: hello".to_string(),
            "[truncated] Showing first 100".to_string(),
            "- src/lib.rs:9:aa: before".to_string(),
            "+ src/lib.rs:11:11: after".to_string(),
        ];
        assert_eq!(
            unique_match_files(&matches),
            vec!["src/lib.rs", "src/main.rs"]
        );
    }

    #[test]
    fn unique_match_files_reads_match_line_inside_context_block() {
        let block =
            "- src/lib.rs:9:aa: before\nsrc/lib.rs:10:ab: fn main()\n+ src/lib.rs:11:cc: after"
                .to_string();
        assert_eq!(unique_match_files(&[block]), vec!["src/lib.rs"]);
    }

    #[test]
    fn file_pattern_does_not_treat_rs_as_suffix_of_yours() {
        assert!(path_matches_file_pattern(Path::new("src/lib.rs"), "rs"));
        assert!(!path_matches_file_pattern(Path::new("src/yours"), "rs"));
        assert!(path_matches_file_pattern(Path::new("src/lib.rs"), "*.rs"));
        assert!(path_matches_file_pattern(Path::new("src/lib.rs"), ".rs"));
        assert!(path_matches_file_pattern(Path::new("src/lib.rs"), "lib.rs"));
        assert!(path_matches_file_pattern(
            Path::new("src/lib.rs"),
            "src/lib.rs"
        ));
        assert!(path_matches_file_pattern(
            Path::new("src/lib.rs"),
            "**/*.rs"
        ));
        assert!(path_matches_file_pattern(
            Path::new(".gitignore"),
            ".gitignore"
        ));
        assert!(!path_matches_file_pattern(Path::new("src/lib.ts"), "rs"));
    }

    #[test]
    fn streaming_search_matches_slurp_for_plain_hits() {
        let input = "alpha\nbeta token\ngamma\nsecond token\nomega\n";
        let (blocks, total, lines) = search(input.as_bytes(), "token", 0, 100);
        assert_eq!(total, 2);
        assert_eq!(lines, 5);
        assert_eq!(match_lines(&blocks), slurp_match_lines(input, "token", 100));
    }

    #[test]
    fn streaming_search_respects_max_matches() {
        let input = "token a\ntoken b\ntoken c\n";
        let (blocks, total, _) = search(input.as_bytes(), "token", 0, 2);
        assert_eq!(total, 2);
        assert_eq!(match_lines(&blocks).len(), 2);
        assert_eq!(match_lines(&blocks), slurp_match_lines(input, "token", 2));
    }

    #[test]
    fn streaming_search_emits_context_before_then_match_then_after() {
        let (blocks, total, _) = search(b"alpha\nbeta\nhit here\ngamma\ndelta\n", "hit", 2, 100);
        assert_eq!(total, 1);
        let lines: Vec<&str> = blocks[0].lines().collect();
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with("- "));
        assert!(lines[0].contains("alpha"));
        assert!(lines[1].starts_with("- "));
        assert!(lines[1].contains("beta"));
        assert!(match_line_path(lines[2]).is_some());
        assert!(lines[2].contains("hit here"));
        assert!(lines[3].starts_with("+ "));
        assert!(lines[3].contains("gamma"));
        assert!(lines[4].starts_with("+ "));
        assert!(lines[4].contains("delta"));
    }

    #[test]
    fn streaming_search_strips_crlf() {
        let (blocks, total, lines) = search(b"alpha\r\nhit here\r\nbeta\r\n", "hit", 0, 100);
        assert_eq!(total, 1);
        assert_eq!(lines, 3);
        assert_eq!(match_lines(&blocks).len(), 1);
        assert!(!blocks[0].contains('\r'));
    }

    #[test]
    fn invalid_utf8_does_not_inflate_match_count() {
        let mut total_matches = 7usize;
        let mut lines_scanned = 11usize;
        let matches = search_reader(
            Cursor::new(b"hit here\n\xff\nhit too\n"),
            &matcher("hit"),
            0,
            &mut total_matches,
            100,
            &mut lines_scanned,
            "bin.rs",
        )
        .expect("search");
        assert!(matches.is_empty());
        assert_eq!(total_matches, 7);
        assert_eq!(lines_scanned, 11);
    }

    #[test]
    fn oversized_file_is_error_not_match() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("grep-accuracy")
            .join("oversize");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).expect("fixture dir");
        let path = root.join("big.rs");
        std::fs::write(&path, vec![b'x'; 64]).expect("write");
        let mut total_matches = 0usize;
        let mut lines_scanned = 0usize;
        let root = root.canonicalize().expect("canonical fixture dir");
        let err = GrepTool::search_file(
            Some(&root),
            &root.join("big.rs"),
            &matcher("x"),
            0,
            16,
            &mut total_matches,
            100,
            &mut lines_scanned,
        )
        .expect_err("oversize must error");
        assert!(err.contains("too large"), "{err}");
        assert_eq!(total_matches, 0);
        assert_eq!(lines_scanned, 0);
    }

    // ── trim_line_to_window ───────────────────────────────────────────────────

    use super::{chars_budget, trim_line_to_window};

    /// A line shorter than the limit passes through unchanged.
    #[test]
    fn trim_short_line_unchanged() {
        let line = "src/lib.rs:10:ab12: fn hello()";
        assert_eq!(trim_line_to_window(line, "hello", 300), line);
    }

    /// When the line is long, the window is centred on the pattern.
    /// ~1/3 of the content budget goes before the match, ~2/3 after.
    #[test]
    fn trim_window_centred_on_match() {
        // Build: prefix (20 chars) + "before" padding (100 'a') + pattern + "after" padding (200 'z')
        let prefix = "path/to/file.rs:42:abcd: ";
        let before_pad = "a".repeat(100);
        let pattern = "TARGET";
        let after_pad = "z".repeat(200);
        let line = format!("{prefix}{before_pad}{pattern}{after_pad}");
        assert!(
            line.chars().count() > 300,
            "precondition: line must be long"
        );

        let result = trim_line_to_window(&line, pattern, 300);
        // The result must contain the pattern.
        assert!(result.contains(pattern), "pattern must appear in result");
        // Ellipsis on the left indicates before-context was trimmed.
        let content_part = &result[prefix.len()..];
        assert!(
            content_part.starts_with('…'),
            "left ellipsis expected, got: {content_part:?}"
        );
        // The result must be at most line_limit chars, cut marks included.
        assert!(
            result.chars().count() <= 300,
            "result too long: {} chars",
            result.chars().count()
        );
    }

    /// When the match is near the start, no left ellipsis is added.
    #[test]
    fn trim_window_no_left_ellipsis_when_match_at_start() {
        let prefix = "f.rs:1:ab: ";
        let pattern = "TARGET";
        let after_pad = "z".repeat(300);
        let line = format!("{prefix}{pattern}{after_pad}");
        let result = trim_line_to_window(&line, pattern, 300);
        let content_part = &result[prefix.len()..];
        assert!(
            !content_part.starts_with('…'),
            "no left ellipsis when match is at start: {content_part:?}"
        );
        assert!(result.contains(pattern));
    }

    /// Multibyte UTF-8 chars near the cut boundary must not break char boundary.
    #[test]
    fn trim_window_multibyte_utf8_near_cut() {
        // '→' is 3 bytes. Pack them so the cut lands inside a multi-byte sequence.
        let prefix = "f.rs:1:ab: ";
        // 120 copies of '→' (360 bytes) before the pattern
        let before = "→".repeat(120);
        let pattern = "MATCH";
        let after = "→".repeat(120);
        let line = format!("{prefix}{before}{pattern}{after}");
        // Must not panic on char-boundary violation.
        let result = trim_line_to_window(&line, pattern, 300);
        assert!(result.contains(pattern));
        // Result must be valid UTF-8 (no panic above already proves this, but be explicit).
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }

    /// A line of exactly 300 chars is at the limit and is not cut.
    #[test]
    fn trim_leaves_line_of_exactly_300_chars() {
        let line = format!("f.rs:1:ab: {}TARGET{}", "a".repeat(100), "z".repeat(183));
        assert_eq!(line.chars().count(), 300, "precondition");
        let result = super::trim_line_to_window(&line, "TARGET", super::LINE_LIMIT);
        assert_eq!(result, line);
        assert_eq!(result.chars().count(), 300);
    }

    /// One char over the limit cuts the line around the match.
    #[test]
    fn trim_cuts_line_of_301_chars() {
        let line = format!("f.rs:1:ab: {}TARGET{}", "a".repeat(100), "z".repeat(184));
        assert_eq!(line.chars().count(), 301, "precondition");
        let result = super::trim_line_to_window(&line, "TARGET", super::LINE_LIMIT);
        assert_eq!(
            result,
            format!("f.rs:1:ab: …{}TARGET{}", "a".repeat(95), "z".repeat(184))
        );
        assert_eq!(result.chars().count(), 297);
    }

    /// chars_budget returns the three expected tiers.
    #[test]
    fn chars_budget_tiers() {
        assert_eq!(chars_budget(0), 5_000);
        assert_eq!(chars_budget(3), 5_000);
        assert_eq!(chars_budget(4), 3_500);
        assert_eq!(chars_budget(8), 3_500);
        assert_eq!(chars_budget(9), 2_500);
        assert_eq!(chars_budget(100), 2_500);
    }

    // ── output budget and line trimming (end to end) ──────────────────────────

    use super::{LINE_LIMIT, TRIM_MARK};
    use crate::agent::tools::{Tool, ToolResult};
    use serde_json::json;
    use tempfile::TempDir;

    const MANY: usize = 60;

    /// Equal-length names that sort in index order.
    fn many_file_name(i: usize) -> String {
        format!("{i:03}_{}.txt", "q".repeat(80))
    }

    /// 60 matching files whose absolute paths together exceed the 2_500-char
    /// budget for more than 8 files.
    fn many_matching_files() -> TempDir {
        let dir = TempDir::new().expect("tempdir");
        for i in 0..MANY {
            std::fs::write(dir.path().join(many_file_name(i)), "needle\n").expect("write");
        }
        dir
    }

    async fn grep_needle(dir: &TempDir, extra: serde_json::Value) -> ToolResult {
        let mut args = json!({ "pattern": "needle", "path": dir.path().to_string_lossy() });
        if let (Some(args), Some(extra)) = (args.as_object_mut(), extra.as_object()) {
            args.extend(extra.clone());
        }
        let result = GrepTool.execute(args).await;
        assert!(result.success, "{:?}", result.error);
        result
    }

    fn listed_files(content: &str) -> Vec<&str> {
        content
            .lines()
            .filter(|line| line.ends_with(".txt"))
            .collect()
    }

    fn match_blocks<'a>(content: &'a str, file: &str) -> Vec<&'a str> {
        let marker = format!("{file}:");
        content
            .lines()
            .filter(|line| line.contains(&marker))
            .collect()
    }

    #[tokio::test]
    async fn grep_files_mode_stops_at_char_budget() {
        let dir = many_matching_files();
        let result = grep_needle(&dir, json!({ "mode": "files_with_matches" })).await;
        let listed = listed_files(&result.content);
        let budget = chars_budget(MANY);
        let entry_len = listed[0].chars().count() + 1;

        assert!(
            result.content.starts_with(&format!("{MANY} files matched")),
            "{}",
            result.content
        );
        assert_eq!(
            listed.len(),
            budget / entry_len,
            "budget must fill with whole paths and stop"
        );
        let used: usize = listed.iter().map(|p| p.chars().count() + 1).sum();
        assert!(used <= budget, "{used} chars of paths over {budget} budget");
    }

    /// 60 matching files under four 200-char directories. Each listed path
    /// is about 900 chars plus the temp-dir prefix, so exactly 2 fit the
    /// 2_500-char budget for any prefix shorter than about 350 chars.
    fn many_deep_matching_files() -> TempDir {
        let dir = TempDir::new().expect("tempdir");
        let deep = ["d", "e", "f", "g"]
            .iter()
            .fold(dir.path().to_path_buf(), |path, c| path.join(c.repeat(200)));
        std::fs::create_dir_all(&deep).expect("mkdir");
        for i in 0..MANY {
            std::fs::write(deep.join(many_file_name(i)), "needle\n").expect("write");
        }
        dir
    }

    #[tokio::test]
    async fn grep_files_mode_banner_offset_pages_to_next_file() {
        let dir = many_deep_matching_files();
        let first = grep_needle(&dir, json!({ "mode": "files_with_matches" })).await;
        assert_eq!(listed_files(&first.content).len(), 2, "{}", first.content);
        assert!(
            first.content.ends_with(
                "\n\n[showing 2/60 files — use offset=2 to page, or narrow the pattern/path]"
            ),
            "{}",
            first.content
        );

        let next = grep_needle(&dir, json!({ "mode": "files_with_matches", "offset": 2 })).await;
        let next_listed = listed_files(&next.content);
        assert!(
            next_listed[0].ends_with(&many_file_name(2)),
            "offset from the banner must start at the first hidden file: {}",
            next.content
        );
        assert!(
            next.content.ends_with(
                "\n\n[showing 2/58 files — use offset=4 to page, or narrow the pattern/path]"
            ),
            "{}",
            next.content
        );
    }

    /// Each match line is cut on both sides to exactly 300 chars, whatever the
    /// temp-dir prefix, so 16 blocks of 302 chars fit the 5_000-char budget.
    #[tokio::test]
    async fn grep_content_mode_stops_at_budget_with_offset_banner() {
        let dir = TempDir::new().expect("tempdir");
        let line = format!("{}needle{}", "a".repeat(400), "z".repeat(400));
        let body = vec![line.as_str(); MANY].join("\n");
        std::fs::write(dir.path().join("many.txt"), body).expect("write");

        let result = grep_needle(&dir, json!({ "context_lines": 0 })).await;
        let blocks = match_blocks(&result.content, "many.txt");
        assert_eq!(blocks.len(), 16, "{}", result.content);
        assert!(
            blocks.iter().all(|b| b.chars().count() == 300),
            "{}",
            result.content
        );
        assert!(
            result.content.ends_with(
                "\n\n[showing 16/60 match blocks — use offset=16 to page, or narrow the pattern/path]"
            ),
            "{}",
            result.content
        );
    }

    #[tokio::test]
    async fn grep_content_shows_first_block_even_over_budget() {
        let dir = TempDir::new().expect("tempdir");
        let filler = "c".repeat(400);
        let mut lines = vec![filler.as_str(); 20];
        lines.push("needle");
        lines.extend(vec![filler.as_str(); 20]);
        std::fs::write(dir.path().join("big.txt"), lines.join("\n")).expect("write");

        let result = grep_needle(&dir, json!({ "context_lines": 20 })).await;
        let block_chars: usize = match_blocks(&result.content, "big.txt")
            .iter()
            .map(|line| line.chars().count() + 1)
            .sum();
        assert!(
            block_chars > chars_budget(1),
            "precondition: block over budget"
        );
        assert!(result.content.contains("big.txt:21:"), "{}", result.content);
        assert!(!result.content.contains("[showing"), "{}", result.content);
    }

    #[tokio::test]
    async fn grep_content_trims_long_line_around_match() {
        let dir = TempDir::new().expect("tempdir");
        let line = format!("{}needle{}", "a".repeat(2_000), "z".repeat(2_000));
        std::fs::write(dir.path().join("long.txt"), line).expect("write");

        let result = grep_needle(&dir, json!({ "context_lines": 0 })).await;
        let hit = match_blocks(&result.content, "long.txt")[0];
        assert!(
            hit.chars().count() <= LINE_LIMIT,
            "{} chars over the {LINE_LIMIT} line limit",
            hit.chars().count()
        );
        let (_, content) = hit.split_once(": ").expect("hashline content");
        let window = content
            .strip_prefix(TRIM_MARK)
            .and_then(|rest| rest.strip_suffix(TRIM_MARK))
            .expect("both sides of the long line are cut and marked");
        let before = window.find("needle").expect("match kept in the window");
        let window_len = window.chars().count();
        assert!(
            before.abs_diff(window_len / 3) <= 1,
            "match at {before} of {window_len}: expected 1/3 before, 2/3 after"
        );
    }
}

#[cfg(all(test, unix, feature = "tools-grep"))]
mod link_tests {
    use super::GrepTool;
    use crate::agent::tools::Tool;
    use serde_json::json;

    /// File names named in a grep result, in result order.
    fn named_files(content: &str) -> Vec<String> {
        let mut names: Vec<String> = content
            .lines()
            .filter_map(|line| line.split(':').next())
            .filter_map(|path| std::path::Path::new(path).file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".txt"))
            .collect();
        names.dedup();
        names
    }

    #[tokio::test]
    async fn a_directory_grep_does_not_read_a_linked_file() {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.txt"), "needle outside\n").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        std::fs::write(ws.path().join("ok.txt"), "needle inside\n").expect("ok");
        std::os::unix::fs::symlink(
            outside.path().join("secret.txt"),
            ws.path().join("leak.txt"),
        )
        .expect("file link");
        std::os::unix::fs::symlink(outside.path(), ws.path().join("linked_dir")).expect("dir link");
        let root = ws.path().canonicalize().expect("canonical workspace");

        let result = GrepTool
            .execute(json!({ "pattern": "needle", "path": root.to_string_lossy() }))
            .await;

        assert!(result.success, "{:?}", result.error);
        assert_eq!(
            named_files(&result.content),
            ["ok.txt"],
            "{}",
            result.content
        );
        assert!(!result.content.contains("outside"), "{}", result.content);
    }
}

// When the `tools-grep` feature is disabled, `GrepTool` exists as a marker
// struct but is never registered in the tool registry, so no fallback `Tool`
// impl is needed. Callers that try to construct it without the feature still
// get a type-check error via the absent trait impls.
