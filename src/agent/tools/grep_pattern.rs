//! Grep pattern planning.
//!
//! Models often wrap a literal snippet in quotes, or paste source text that is
//! not a valid regex (`foo(`, `a[b`). Those calls used to return `ok=false`.
//! The first attempt is the pattern as a regex when it compiles. Otherwise it
//! is searched once as a literal. If a quote-wrapped pattern matches nothing,
//! one follow-up search drops the wrapping quotes and searches the inside as a
//! literal.

use anyhow::{Result, bail};

/// One grep attempt.
///
/// `source` is always a valid regex (literals are escaped) for the in-process
/// matcher. `needle` is the text a fixed-string search should use (`grep -F`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepAttempt {
    /// Regex source passed to the matcher.
    pub source: String,
    /// Fixed-string needle. Equals `source` for a real regex.
    pub needle: String,
    /// `true` when this attempt is a literal search.
    pub literal: bool,
    /// Note to append when this attempt is the one that produced the result.
    pub note: Option<String>,
}

/// Build at most two search attempts for `raw`.
pub fn grep_attempts(raw: &str) -> Result<Vec<GrepAttempt>> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!(
            "pattern is empty. Pass the text or regex to find, for example pattern=hashline_delete."
        );
    }

    let mut attempts = Vec::with_capacity(2);
    if regex_compiles(trimmed) {
        attempts.push(GrepAttempt {
            source: trimmed.to_string(),
            needle: trimmed.to_string(),
            literal: false,
            note: None,
        });
    } else {
        attempts.push(GrepAttempt {
            source: escape_regex_literal(trimmed),
            needle: trimmed.to_string(),
            literal: true,
            note: Some(format!(
                "Pattern {trimmed:?} is not valid regex, so it was searched once as a literal. \
                 Escape metacharacters if you meant a regex."
            )),
        });
    }

    if let Some(inner) = strip_wrapping_quotes(trimmed) {
        let inner = inner.trim();
        if !inner.is_empty() {
            let literal = escape_regex_literal(inner);
            let already = attempts.iter().any(|attempt| attempt.source == literal);
            if !already {
                attempts.push(GrepAttempt {
                    source: literal,
                    needle: inner.to_string(),
                    literal: true,
                    note: Some(
                        "Wrapping quotes matched nothing. Retried once as a literal with the quotes removed."
                            .to_string(),
                    ),
                });
            }
        }
    }

    attempts.truncate(2);
    Ok(attempts)
}

fn strip_wrapping_quotes(value: &str) -> Option<&str> {
    let bytes = value.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let (open, close) = (bytes[0], bytes[bytes.len() - 1]);
    if (open == b'"' && close == b'"') || (open == b'\'' && close == b'\'') {
        return Some(&value[1..value.len() - 1]);
    }
    None
}

/// Escape a string so the Rust regex engine matches it literally.
#[must_use]
pub fn escape_regex_literal(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for ch in value.chars() {
        if matches!(
            ch,
            '\\' | '.' | '+' | '*' | '?' | '(' | ')' | '|' | '[' | ']' | '{' | '}' | '^' | '$'
        ) {
            escaped.push('\\');
        }
        escaped.push(ch);
    }
    escaped
}

fn regex_compiles(pattern: &str) -> bool {
    grep_regex::RegexMatcher::new(pattern).is_ok()
}

#[cfg(test)]
mod tests {
    use super::{escape_regex_literal, grep_attempts, strip_wrapping_quotes};

    #[test]
    fn plain_text_stays_a_regex() {
        let attempts = grep_attempts("hashline_delete").expect("attempts");
        assert_eq!(attempts.len(), 1);
        assert!(!attempts[0].literal);
        assert_eq!(attempts[0].source, "hashline_delete");
    }

    #[test]
    fn invalid_regex_becomes_a_literal() {
        let attempts = grep_attempts("foo(").expect("attempts");
        assert!(attempts[0].literal);
        assert_eq!(attempts[0].source, escape_regex_literal("foo("));
        assert!(
            attempts[0]
                .note
                .as_deref()
                .unwrap_or("")
                .contains("literal")
        );
    }

    #[test]
    fn quoted_pattern_adds_a_literal_retry_without_quotes() {
        let raw = "\"fn foo(bar)\"";
        let attempts = grep_attempts(raw).expect("attempts");
        assert_eq!(attempts.len(), 2);
        assert!(!attempts[0].literal);
        assert_eq!(attempts[0].source, raw);
        assert!(attempts[1].literal);
        assert_eq!(attempts[1].source, escape_regex_literal("fn foo(bar)"));
        assert!(attempts[1].note.as_deref().unwrap_or("").contains("quotes"));
    }

    #[test]
    fn strip_wrapping_quotes_keeps_inner_quotes() {
        assert_eq!(strip_wrapping_quotes("\"a\\\"b\""), Some("a\\\"b"));
        assert_eq!(strip_wrapping_quotes("hello"), None);
    }
}
