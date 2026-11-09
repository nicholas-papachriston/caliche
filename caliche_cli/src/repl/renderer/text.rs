//! Plain-text helpers for the REPL renderer.

use caliche::llm::prompt_cache::PromptCacheReport;
use caliche::observability::OutputVerbosity;

pub(super) use caliche_contracts::text::truncate_utf8 as truncate_str;

/// Dim ANSI (gray).
pub(super) const DIM: &str = "\x1b[90m";
pub(super) const CYAN: &str = "\x1b[36m";
pub(super) const GREEN: &str = "\x1b[32m";
pub(super) const RED: &str = "\x1b[31m";
pub(super) const YELLOW: &str = "\x1b[33m";
pub(super) const BLUE: &str = "\x1b[34m";
pub(super) const BOLD: &str = "\x1b[1m";
pub(super) const RESET: &str = "\x1b[0m";

/// Return a dim horizontal rule that spans the current terminal width (capped
/// at 80 columns so it doesn't overwhelm very wide terminals).
pub(super) fn turn_rule() -> String {
    let cols = crate::repl::terminal::terminal_size()
        .map(|(c, _)| c as usize)
        .unwrap_or(40)
        .min(80);
    format!("{}{}{}", DIM, "─".repeat(cols), RESET)
}

pub(super) fn render_wrapped_lines(text: &str, max_width: usize) -> Vec<String> {
    if max_width == 0 {
        return vec![text.to_string()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();

    for word in text.split_whitespace() {
        if current.is_empty() {
            if word.len() <= max_width {
                current.push_str(word);
            } else {
                // Hard-wrap long single tokens (URLs, identifiers).
                let mut rest = word;
                while rest.len() > max_width {
                    let mut end = max_width;
                    while end > 0 && !rest.is_char_boundary(end) {
                        end -= 1;
                    }
                    if end == 0 {
                        break;
                    }
                    lines.push(rest[..end].to_string());
                    rest = &rest[end..];
                }
                if !rest.is_empty() {
                    current.push_str(rest);
                }
            }
        } else if current.len() + 1 + word.len() <= max_width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(current);
            current = String::new();
            if word.len() <= max_width {
                current.push_str(word);
            } else {
                let mut rest = word;
                while rest.len() > max_width {
                    let mut end = max_width;
                    while end > 0 && !rest.is_char_boundary(end) {
                        end -= 1;
                    }
                    if end == 0 {
                        break;
                    }
                    lines.push(rest[..end].to_string());
                    rest = &rest[end..];
                }
                if !rest.is_empty() {
                    current.push_str(rest);
                }
            }
        }
    }

    if !current.is_empty() {
        lines.push(current);
    }

    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

pub(super) fn prompt_cache_usage_parts(
    usage: &caliche::messages::TokenUsage,
    verbosity: OutputVerbosity,
) -> Vec<String> {
    use crate::cost::format_tokens;

    let report = PromptCacheReport::from_usage(usage);
    if !report.saw_cache() && !verbosity.show_cache_detail() {
        return Vec::new();
    }

    let mut parts = Vec::new();
    if let Some(pct) = report.hit_pct() {
        parts.push(format!(
            "cache hit: {pct:.0}% ({})",
            report.outcome().as_str()
        ));
    } else {
        parts.push(format!("cache: {}", report.outcome().as_str()));
    }
    parts.push(format!(
        "cache read: {}",
        format_tokens(u64::from(report.cache_read))
    ));
    parts.push(format!(
        "cache write: {}",
        format_tokens(u64::from(report.cache_write))
    ));
    parts.push(format!(
        "uncached: {}",
        format_tokens(u64::from(report.uncached_tokens()))
    ));
    parts
}

#[cfg(test)]
mod tests {
    use super::truncate_str;

    #[test]
    fn truncate_stops_on_a_char_boundary() {
        let text = "\u{2550}\u{2550}";
        assert_eq!(truncate_str(text, 1), "");
        assert_eq!(truncate_str(text, 3), "\u{2550}");
        assert_eq!(truncate_str("hello", 10), "hello");
    }
}
