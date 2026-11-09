use crate::agent::tools::names::{TOOL_BASH, TOOL_FILESYSTEM};

/// Shared tool output summarization/truncation utilities.
///
/// The goal is to keep tool outputs informative while preventing them from consuming
/// disproportionate amounts of prompt context.
#[derive(Debug, Clone, Copy)]
pub struct ToolOutputLimits {
    /// Hard cap (in characters) for tool output stored in memory / sent back to the model.
    pub max_chars: usize,
    /// Above this size (in characters), attempt tool-specific summarization before truncating.
    pub summarize_threshold: usize,
    /// Characters to keep from the head when doing head/tail truncation.
    pub head_chars: usize,
    /// Characters to keep from the tail when doing head/tail truncation.
    pub tail_chars: usize,
    /// Whether to add an annotation about summarization/truncation.
    pub annotate: bool,
}

const fn estimate_tokens_from_chars(chars: usize) -> usize {
    (chars.saturating_add(3)) / 4
}

pub use caliche_contracts::text::truncate_utf8;

fn truncate_tool_output(content: &str, max_chars: usize) -> String {
    if content.len() <= max_chars {
        return content.to_string();
    }
    let truncated = truncate_utf8(content, max_chars);
    let original_size = content.len();
    let truncated_size = truncated.len();
    let original_tokens = estimate_tokens_from_chars(original_size);
    let truncated_tokens = estimate_tokens_from_chars(truncated_size);
    format!(
        "{truncated}\n\n[Output truncated: {original_size} chars (est. {original_tokens} tokens) -> {truncated_size} chars (est. {truncated_tokens} tokens)]"
    )
}

fn truncate_with_head_tail(content: &str, head_chars: usize, tail_chars: usize) -> String {
    if content.len() <= head_chars.saturating_add(tail_chars) {
        return content.to_string();
    }
    let head_end = content.floor_char_boundary(head_chars.min(content.len()));
    let tail_start = content.floor_char_boundary(content.len().saturating_sub(tail_chars));
    let head = &content[..head_end];
    let tail = &content[tail_start..];
    let omitted = content
        .len()
        .saturating_sub(head_chars.saturating_add(tail_chars));
    format!("{head}\n[... {omitted} chars omitted ...]\n{tail}")
}

fn extract_command_summary_and_errors(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();
    let status = lines
        .iter()
        .find(|l| l.contains("exit") || l.contains("status") || l.contains("code"))
        .copied()
        .unwrap_or("");
    if n <= 25 {
        return content.to_string();
    }
    let head = lines[..12.min(n)].join("\n");
    let tail = lines[n.saturating_sub(8)..].join("\n");
    let mid = format!("[... {} lines omitted ...]", n.saturating_sub(20));
    if status.is_empty() {
        format!("{head}\n{mid}\n{tail}")
    } else {
        format!("{head}\n{mid}\n{tail}\n{status}")
    }
}

fn extract_file_structure_and_key_sections(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let n = lines.len();
    let preview = 15;
    if n <= preview * 2 {
        return content.to_string();
    }
    let first = lines[..preview].join("\n");
    let last = lines[n - preview..].join("\n");
    format!(
        "{}\n[... {} lines omitted ...]\n{}",
        first,
        n.saturating_sub(preview * 2),
        last
    )
}

/// Extract a structured summary from grep output.
/// Groups matches by file, shows match counts, and includes first few matches per file.
///
/// We require an unambiguous header (`Found N match[es] ...`) emitted by our own
/// grep tool before treating the content as grep output. The path-shaped heuristic
/// alone is too loose: a single line of code such as
/// `expect(result.attributes).toEqual({ count: "42", price: "19.99" });` contains
/// a `.` and two colons and was previously misclassified as a grep match,
/// turning a normal `fs op=read` of a code file into a bogus "Grep results: 1
/// matches" payload.
fn extract_grep_summary(content: &str, max_chars: usize) -> Option<String> {
    let lines: Vec<&str> = content.lines().collect();

    // Definitive grep marker emitted by `src/agent/tools/grep.rs`. Accept both
    // singular ("Found 1 match for ...") and plural ("Found 5 matches for ...").
    let has_grep_header = lines.iter().any(|l| {
        l.starts_with("Found ") && (l.contains(" match for ") || l.contains(" matches for "))
    });

    if !has_grep_header {
        return None;
    }

    // Parse file:line:content groups -- only after we've confirmed this is
    // really grep output, so the loose `path:line:content` heuristic is safe.
    use std::collections::HashMap;
    let mut files: HashMap<&str, Vec<&str>> = HashMap::new();

    for line in &lines {
        if let Some(first_colon) = line.find(':') {
            let potential_path = &line[..first_colon];
            if (potential_path.contains('/') || potential_path.contains('.'))
                && line[first_colon + 1..].contains(':')
            {
                let path = potential_path.trim();
                if !path.is_empty() {
                    files.entry(path).or_default().push(line.trim());
                }
            }
        }
    }

    // Also check for the "Found X matches (showing first Y)" line and extract total
    let total_matches: usize = lines
        .iter()
        .find_map(|l| {
            if l.starts_with("Found ") && l.contains("matches") {
                l.split_whitespace().nth(1).and_then(|n| n.parse().ok())
            } else {
                None
            }
        })
        .unwrap_or_else(|| files.values().map(std::vec::Vec::len).sum());

    let mut out = String::new();
    out.push_str(&format!(
        "Grep results: {} matches across {} files\n",
        total_matches,
        files.len()
    ));

    // Sort files by match count (descending) to prioritize most relevant
    let mut file_entries: Vec<(&&str, &Vec<&str>)> = files.iter().collect();
    file_entries.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (file, matches) in file_entries {
        // Add separator and file header
        out.push('\n');
        out.push_str(&format!("  {} ({} matches)\n", file, matches.len()));

        // Show first 3 matches for this file
        for m in matches.iter().take(3) {
            out.push_str(&format!("    {m}\n"));
        }
        if matches.len() > 3 {
            out.push_str(&format!("    ... {} more\n", matches.len() - 3));
        }

        if out.len() > max_chars {
            out.push_str("\n... (truncated for context)\n");
            break;
        }
    }

    Some(out)
}

/// Summarize tool output: tool-specific extraction for large payloads, then cap.
pub fn summarize_tool_output(content: &str, tool_name: &str, limits: ToolOutputLimits) -> String {
    if content.len() <= limits.max_chars {
        return content.to_string();
    }

    // If it's only moderately oversized, just truncate.
    if content.len() <= limits.summarize_threshold {
        return truncate_tool_output(content, limits.max_chars);
    }

    let summarized = match tool_name {
        TOOL_FILESYSTEM => extract_grep_summary(content, limits.max_chars)
            .unwrap_or_else(|| extract_file_structure_and_key_sections(content)),
        TOOL_BASH | "shell" | "execute_command" => extract_command_summary_and_errors(content),
        _ => truncate_with_head_tail(content, limits.head_chars, limits.tail_chars),
    };

    let capped = if summarized.len() <= limits.max_chars {
        summarized
    } else {
        truncate_tool_output(&summarized, limits.max_chars)
    };

    if limits.annotate {
        let mut result = format!(
            "[Tool output summarized: {} chars -> {} chars]\n{}",
            content.len(),
            capped.len(),
            capped
        );
        if tool_name == TOOL_FILESYSTEM || tool_name == "fs_read_internal" {
            result.push_str(
                "\n[Hint: use fs op=read with query=\"...\" or start_line/end_line \
                 to read targeted sections instead of the full file]",
            );
        }
        if matches!(tool_name, TOOL_BASH | "shell" | "execute_command")
            && !capped.contains("Full output:")
        {
            result.push_str(
                "\n[Hint: oversized bash output is spilled to a temp file; look for \
                 'Full output: <path>' and re-read with fs op=read]",
            );
        }
        result
    } else {
        capped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(max: usize) -> ToolOutputLimits {
        ToolOutputLimits {
            max_chars: max,
            summarize_threshold: max / 2,
            head_chars: max / 2,
            tail_chars: max / 4,
            annotate: false,
        }
    }

    #[test]
    fn extract_grep_summary_requires_explicit_header() {
        // A code line that contains a `.`, an opening brace, and two `:` would
        // previously be misclassified as grep output. With the header gate it
        // must return None instead.
        let code = "    expect(result.attributes).toEqual({ count: \"42\", price: \"19.99\" });";
        assert!(extract_grep_summary(code, 1024).is_none());
    }

    #[test]
    fn extract_grep_summary_accepts_real_grep_header() {
        let grep = "Found 2 matches for pattern 'foo' in 'src/':\n\nsrc/a.rs:10:foo bar\nsrc/b.rs:20:foo baz\n";
        let summary = extract_grep_summary(grep, 1024).expect("real grep should be detected");
        assert!(summary.contains("Grep results: 2 matches"));
        assert!(summary.contains("src/a.rs"));
        assert!(summary.contains("src/b.rs"));
    }

    #[test]
    fn extract_grep_summary_accepts_singular_match_header() {
        let grep = "Found 1 match for pattern 'foo' in 'src/lib.rs':\n\nsrc/lib.rs:42:foo bar\n";
        assert!(extract_grep_summary(grep, 1024).is_some());
    }

    #[test]
    fn summarize_tool_output_does_not_treat_code_as_grep() {
        // Build a payload large enough to trigger the summarizer (above max_chars)
        // but with no grep header. The fs branch should fall through to
        // `extract_file_structure_and_key_sections`, NOT pretend the file is grep.
        let mut content = String::new();
        for i in 0..2000 {
            content.push_str(&format!(
                "  expect(obj.method({{ key: \"v{i}\", other: \"x\" }})).toBe(true);\n",
            ));
        }
        let out = summarize_tool_output(&content, TOOL_FILESYSTEM, limits(20_000));
        assert!(
            !out.starts_with("Grep results:"),
            "code content must not be summarized as grep output, got: {}",
            &out[..out.len().min(120)]
        );
    }
}
