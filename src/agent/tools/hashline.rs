//! Content-addressed hashline tagging for file read/edit operations.
//!
//! Every line returned by fs(op=read) and fs(op=grep) is tagged with a short content hash
//! (`42:ab12| line content`). Edit operations reference that hash so a stale line number
//! cannot land on the wrong text.
//!
//! Matching order:
//! 1. Strict hash of the line with trailing whitespace and CR stripped (the hash `fs` prints).
//! 2. One retry that also strips leading whitespace, used only when the strict hash misses.
//!    The hinted line wins; otherwise the hash must be unique in range. Two loose matches refuse.
//! 3. A 2-hex hash is the low byte of the 4-hex hash and is accepted only on the hinted line.
//!    It is never relocated (a 2-hex collision is 1 in 256).
//!
//! The hash token must be exactly 2 or 4 hex digits (`ab12`, `42:ab12`, or `42:ab12| content`).
//! Punctuation and words are errors; a 2-hex run is not scraped out of them. Duplicate hashes
//! are not relocated to the nearest line; the error lists the candidates so one re-read can
//! pick the right one.

use anyhow::{Result, bail};

/// FNV-1a offset basis (32-bit)
const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;

/// FNV-1a prime multiplier
const FNV_PRIME: u32 = 0x0100_0193;

#[derive(Clone, Copy)]
enum HashMode {
    /// Trim trailing whitespace (including CR) before hashing. This is what `fs` prints.
    Strict,
    /// Also trim leading whitespace. Used for the single automatic retry.
    Loose,
}

fn hash_material(content: &str, mode: HashMode) -> &str {
    match mode {
        HashMode::Strict => content.trim_end(),
        HashMode::Loose => content.trim(),
    }
}

fn fnv1a32(content: &str) -> u32 {
    let mut hash: u32 = FNV_OFFSET_BASIS;
    for byte in content.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_value(content: &str, mode: HashMode) -> String {
    format!("{:04x}", fnv1a32(hash_material(content, mode)) & 0xFFFF)
}

/// Compute a 4-character hex content hash for a line.
///
/// Uses FNV-1a truncated to 16 bits. Trailing whitespace and CR are ignored so
/// CRLF files hash the same as LF. The low 8 bits match the previous 2-hex hash.
#[must_use]
pub fn compute_line_hash(content: &str) -> String {
    hash_value(content, HashMode::Strict)
}

/// Format a single line with hashline tag: "42:ab12| line content"
#[must_use]
pub fn format_hashline(line_num: usize, content: &str) -> String {
    format!("{line_num}:{}| {content}", compute_line_hash(content))
}

/// Format multiple lines with hashline tags, starting at `start_line` (1-indexed).
#[must_use]
pub fn format_hashlines(lines: &[&str], start_line: usize) -> String {
    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&format_hashline(start_line + i, line));
    }
    out
}

/// Format a single line for grep-style output: "path:42:a3: content"
#[must_use]
pub fn format_grep_hashline(path: &str, line_num: usize, content: &str) -> String {
    format!(
        "{path}:{line_num}:{}: {content}",
        compute_line_hash(content)
    )
}

fn hashes_equal(actual: &str, expected: &str) -> bool {
    if actual.eq_ignore_ascii_case(expected) {
        return true;
    }
    // Older reads printed only the low byte.
    expected.len() == 2
        && actual.len() == 4
        && actual
            .as_bytes()
            .get(2..)
            .is_some_and(|tail| tail.eq_ignore_ascii_case(expected.as_bytes()))
}

fn line_has_hash(content: &str, expected: &str, mode: HashMode) -> bool {
    hashes_equal(&hash_value(content, mode), expected)
}

/// Validate that the hash matches the actual content at the given line number.
/// Line numbers are 1-indexed.
/// Returns Ok(()) or Err with a descriptive message including the actual hash.
pub fn validate_line_hash(lines: &[String], line_num: usize, expected_hash: &str) -> Result<()> {
    if line_num == 0 {
        bail!("Line numbers must be 1-indexed");
    }
    let idx = line_num.saturating_sub(1);
    if idx >= lines.len() {
        bail!(
            "Line {} is beyond end of file ({} lines)",
            line_num,
            lines.len()
        );
    }
    let anchor = parse_hash_anchor(expected_hash)?;
    let actual_hash = compute_line_hash(&lines[idx]);
    if hashes_equal(&actual_hash, &anchor.hash) {
        Ok(())
    } else {
        bail!("{}", hash_miss_message(lines, line_num, &anchor.hash, &[]));
    }
}

/// A hash token copied from `fs` output, plus an optional line number when the
/// model pasted a `line:hash` anchor into the hash field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashAnchor {
    /// Lowercase 2- or 4-hex hash.
    pub hash: String,
    /// Line number from a `42:ab12` anchor, when present.
    pub line_hint: Option<usize>,
}

/// Parse `ab12`, `42:ab12`, or `42:ab12| line content` into a hash anchor.
pub fn parse_hash_anchor(raw: &str) -> Result<HashAnchor> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        bail!("Hash is empty. Copy the 4 hex characters from fs op=read, for example 42:ab12|.");
    }
    let head = trimmed
        .split_once('|')
        .map(|(head, _)| head.trim())
        .unwrap_or(trimmed);
    if let Some((line_part, hash_part)) = head.split_once(':')
        && let Ok(line) = line_part.trim().parse::<usize>()
        && line > 0
    {
        return Ok(HashAnchor {
            hash: normalize_hash_token(hash_part)?,
            line_hint: Some(line),
        });
    }
    Ok(HashAnchor {
        hash: normalize_hash_token(head)?,
        line_hint: None,
    })
}

fn normalize_hash_token(token: &str) -> Result<String> {
    let trimmed = token.trim();
    let normalized = trimmed.to_lowercase();
    if is_hash_token(&normalized) {
        return Ok(normalized);
    }
    bail!(
        "Invalid hash '{trimmed}'. Expected exactly 4 hex characters from fs op=read \
         (for example ab12), or the anchor 42:ab12. A legacy 2-hex hash is accepted only \
         at the hinted line. Do not wrap the hash in punctuation or words."
    )
}

fn is_hash_token(token: &str) -> bool {
    (token.len() == 2 || token.len() == 4) && token.chars().all(|ch| ch.is_ascii_hexdigit())
}

/// Locate a 1-indexed line by content hash.
///
/// Prefers `hint_line` when that line still matches. A 4-hex hash may then match
/// one other line in range (a line inserted above the hint). A 2-hex hash never
/// relocates. Duplicate hashes fail with the candidate lines.
pub fn locate_line_by_hash(
    lines: &[String],
    hint_line: usize,
    expected_hash: &str,
) -> Result<usize> {
    locate_line_by_hash_in_range(lines, hint_line, expected_hash, 1, lines.len())
}

fn locate_line_by_hash_in_range(
    lines: &[String],
    hint_line: usize,
    expected_hash: &str,
    min_line: usize,
    max_line: usize,
) -> Result<usize> {
    if hint_line == 0 {
        bail!("Line numbers must be 1-indexed");
    }
    let anchor = parse_hash_anchor(expected_hash)?;
    let min_line = min_line.max(1);
    let max_line = max_line.min(lines.len());
    if min_line > max_line {
        bail!("{}", hash_miss_message(lines, hint_line, &anchor.hash, &[]));
    }
    let hints = [hint_line, anchor.line_hint.unwrap_or(0)];
    if anchor.hash.len() == 2 {
        if let Some(line) = hinted_hash_match(&hints, lines, &anchor.hash, min_line, max_line) {
            return Ok(line);
        }
        let mut message = hash_miss_message(lines, hint_line, &anchor.hash, &[]);
        message.push_str(
            " Legacy 2-hex hashes are checked only at the hinted line and are not relocated.",
        );
        bail!("{message}");
    }
    let strict = collect_matches(lines, &anchor.hash, min_line, max_line, HashMode::Strict);
    if let Some(line) = hinted_match(&hints, &strict) {
        return Ok(line);
    }
    if let [only] = strict.as_slice() {
        return Ok(*only);
    }
    if strict.len() > 1 {
        bail!(
            "{}",
            hash_miss_message(lines, hint_line, &anchor.hash, &strict)
        );
    }
    // Strict hash missed everywhere. Retry once, also ignoring leading whitespace.
    let loose = collect_matches(lines, &anchor.hash, min_line, max_line, HashMode::Loose);
    if let Some(line) = hinted_match(&hints, &loose) {
        return Ok(line);
    }
    if let [only] = loose.as_slice() {
        return Ok(*only);
    }
    bail!(
        "{}",
        hash_miss_message(lines, hint_line, &anchor.hash, &loose)
    )
}

/// Strict match on a hinted line, then one leading-whitespace retry on that same line.
fn hinted_hash_match(
    hints: &[usize],
    lines: &[String],
    expected: &str,
    min_line: usize,
    max_line: usize,
) -> Option<usize> {
    for mode in [HashMode::Strict, HashMode::Loose] {
        for hint in hints {
            if (min_line..=max_line).contains(hint)
                && line_has_hash(&lines[hint - 1], expected, mode)
            {
                return Some(*hint);
            }
        }
    }
    None
}

fn hinted_match(hints: &[usize], matches: &[usize]) -> Option<usize> {
    hints
        .iter()
        .copied()
        .find(|hint| *hint > 0 && matches.contains(hint))
}

fn collect_matches(
    lines: &[String],
    expected: &str,
    min_line: usize,
    max_line: usize,
    mode: HashMode,
) -> Vec<usize> {
    let mut matches = Vec::new();
    for line_num in min_line..=max_line {
        if line_has_hash(&lines[line_num - 1], expected, mode) {
            matches.push(line_num);
        }
    }
    matches
}

fn preview_line(content: &str) -> String {
    let mut preview: String = content.chars().take(60).collect();
    if content.chars().count() > 60 {
        preview.push('…');
    }
    preview.replace('\r', "\\r")
}

fn hash_miss_message(
    lines: &[String],
    hint_line: usize,
    expected: &str,
    candidates: &[usize],
) -> String {
    let mut message = format!(
        "Hash mismatch for '{expected}' (hint line {hint_line}, {} lines in file). ",
        lines.len()
    );
    if hint_line > 0 && hint_line <= lines.len() {
        let actual = compute_line_hash(&lines[hint_line - 1]);
        message.push_str(&format!(
            "Line {hint_line} is now {actual}| {} . ",
            preview_line(&lines[hint_line - 1])
        ));
    }
    if candidates.len() > 1 {
        message.push_str(&format!(
            "Hash '{expected}' matches multiple lines, so it was not relocated. Candidates: "
        ));
        for (index, line_num) in candidates.iter().take(6).enumerate() {
            if index > 0 {
                message.push_str("; ");
            }
            let actual = compute_line_hash(&lines[line_num - 1]);
            message.push_str(&format!(
                "{line_num}:{actual}| {}",
                preview_line(&lines[line_num - 1])
            ));
        }
        if candidates.len() > 6 {
            message.push_str(&format!("; {} more", candidates.len() - 6));
        }
        message.push_str(". Use the 4-hex hash from a fresh fs op=read. ");
    } else if candidates.is_empty() {
        message.push_str(
            "No line has this hash after ignoring trailing whitespace and CR, \
             then retrying once with leading whitespace ignored. ",
        );
    }
    let window_start = hint_line.saturating_sub(5).max(1);
    let window_end = hint_line.saturating_add(5);
    message.push_str(&format!(
        "Re-read the slice with fs op=read start_line={window_start} end_line={window_end} \
         and retry using the printed `line:hash` tag (for example {hint_line}:abcd|)."
    ));
    message
}

/// Locate a 1-indexed inclusive range by start/end hashes after concurrent edits.
pub fn locate_line_range(
    lines: &[String],
    start_hint: usize,
    start_hash: &str,
    end_hint: usize,
    end_hash: &str,
) -> Result<(usize, usize)> {
    let start_anchor = parse_hash_anchor(start_hash)?;
    let end_anchor = parse_hash_anchor(end_hash)?;
    let start = locate_line_by_hash(lines, start_hint, start_hash)?;
    if start_hint == end_hint && hashes_equal(&start_anchor.hash, &end_anchor.hash) {
        return Ok((start, start));
    }
    if end_anchor.hash.len() == 2 {
        let end_hints = [end_hint, end_anchor.line_hint.unwrap_or(0)];
        let Some(end) = hinted_hash_match(&end_hints, lines, &end_anchor.hash, start, lines.len())
        else {
            let mut message = hash_miss_message(lines, end_hint, &end_anchor.hash, &[]);
            message.push_str(
                " Legacy 2-hex hashes are checked only at the hinted line and are not relocated.",
            );
            bail!("{message}");
        };
        if end < start {
            bail!("start_line ({start}) must be <= end_line ({end}) after hash relocation");
        }
        return Ok((start, end));
    }
    let span = end_hint.saturating_sub(start_hint);
    let aligned_end = start.saturating_add(span);
    if aligned_end >= start
        && aligned_end <= lines.len()
        && (line_has_hash(&lines[aligned_end - 1], &end_anchor.hash, HashMode::Strict)
            || line_has_hash(&lines[aligned_end - 1], &end_anchor.hash, HashMode::Loose))
    {
        return Ok((start, aligned_end));
    }
    let end = locate_line_by_hash_in_range(
        lines,
        start.saturating_add(span),
        end_hash,
        start,
        lines.len(),
    )?;
    if end < start {
        bail!("start_line ({start}) must be <= end_line ({end}) after hash relocation");
    }
    Ok((start, end))
}

/// Parse a `line:hash` anchor such as `42:ab12` into (`line_num`, hash).
pub fn parse_line_hash(anchor: &str) -> Result<(usize, String)> {
    let parsed = parse_hash_anchor(anchor)?;
    let Some(line_num) = parsed.line_hint else {
        bail!(
            "Invalid anchor '{anchor}': expected 'line:hash' (for example 42:ab12). \
             Copy it from the fs op=read tag `42:ab12| content`."
        );
    };
    Ok((line_num, parsed.hash))
}

#[cfg(test)]
mod tests {
    use super::{compute_line_hash, locate_line_by_hash, locate_line_range, parse_hash_anchor};

    #[test]
    fn locate_prefers_hint_then_nearest_shifted_line() {
        let lines = vec![
            "header".to_string(),
            "alpha_unique".to_string(),
            "body".to_string(),
        ];
        let hash = compute_line_hash("alpha_unique");
        assert_eq!(locate_line_by_hash(&lines, 2, &hash).expect("hint"), 2);

        let shifted = vec![
            "inserted".to_string(),
            "header".to_string(),
            "alpha_unique".to_string(),
            "body".to_string(),
        ];
        assert_eq!(locate_line_by_hash(&shifted, 2, &hash).expect("shifted"), 3);
    }

    #[test]
    fn locate_range_keeps_span_after_insert_above() {
        let lines = vec![
            "inserted".to_string(),
            "start_unique".to_string(),
            "middle".to_string(),
            "end_unique".to_string(),
        ];
        let start_hash = compute_line_hash("start_unique");
        let end_hash = compute_line_hash("end_unique");
        let (start, end) = locate_line_range(&lines, 1, &start_hash, 3, &end_hash).expect("range");
        assert_eq!((start, end), (2, 4));
    }

    #[test]
    fn locate_fails_when_duplicate_hashes_are_equidistant() {
        let token = "dup_line";
        let hash = compute_line_hash(token);
        let lines = vec![token.to_string(), "center".to_string(), token.to_string()];
        let err = locate_line_by_hash(&lines, 2, &hash).expect_err("ambiguous");
        assert!(err.to_string().contains("multiple lines"), "{err}");
        assert!(err.to_string().contains("dup_line"), "{err}");
    }

    #[test]
    fn crlf_and_trailing_space_hash_like_the_visible_line() {
        assert_eq!(compute_line_hash("alpha\r"), compute_line_hash("alpha"));
        assert_eq!(compute_line_hash("alpha  "), compute_line_hash("alpha"));
        assert_ne!(compute_line_hash("  alpha"), compute_line_hash("alpha"));
    }

    #[test]
    fn loose_retry_matches_hint_when_only_leading_whitespace_differs() {
        let lines = vec!["    alpha_unique  ".to_string(), "other".to_string()];
        let trimmed_hash = compute_line_hash("alpha_unique");
        assert_eq!(
            locate_line_by_hash(&lines, 1, &trimmed_hash).expect("loose hint"),
            1
        );
    }

    #[test]
    fn anchor_form_supplies_hash_and_line_hint() {
        let lines = vec!["header".to_string(), "alpha_unique".to_string()];
        let hash = compute_line_hash("alpha_unique");
        let anchor = format!("2:{hash}| alpha_unique");
        let parsed = parse_hash_anchor(&anchor).expect("anchor");
        assert_eq!(parsed.hash, hash);
        assert_eq!(parsed.line_hint, Some(2));
        assert_eq!(
            locate_line_by_hash(&lines, 9, &anchor).expect("anchor locate"),
            2
        );
    }

    #[test]
    fn legacy_two_hex_hash_matches_low_byte() {
        let lines = vec!["only_this_line".to_string()];
        let full = compute_line_hash("only_this_line");
        let legacy = &full[2..];
        assert_eq!(locate_line_by_hash(&lines, 1, legacy).expect("legacy"), 1);
    }

    #[test]
    fn mismatch_error_names_the_actual_hash_and_reread_window() {
        let lines = vec!["alpha".to_string(), "beta_unique".to_string()];
        let alpha = compute_line_hash("alpha");
        let actual = compute_line_hash("beta_unique");
        let wrong = ["0000", "0001", "0002"]
            .into_iter()
            .find(|candidate| *candidate != alpha.as_str() && *candidate != actual.as_str())
            .expect("unused hash");
        let err = locate_line_by_hash(&lines, 2, wrong).expect_err("miss");
        let text = err.to_string();
        assert!(text.contains(wrong), "{text}");
        assert!(text.contains(&actual), "{text}");
        assert!(text.contains("start_line="), "{text}");
        assert!(text.contains("Re-read"), "{text}");
    }

    #[test]
    fn malformed_hash_is_rejected_without_scraping_a_hex_run() {
        for raw in ["#12", "L42:ab", "line 42", "42-ab", "0xab12"] {
            let error = parse_hash_anchor(raw).expect_err(raw);
            let text = error.to_string();
            assert!(text.contains("Invalid hash"), "{raw}: {text}");
            assert!(text.contains(raw), "{raw}: {text}");
        }
    }

    #[test]
    fn legacy_two_hex_does_not_relocate_off_the_hint() {
        let hint = "changed_hint";
        let other = "other_line";
        let hint_hash = compute_line_hash(hint);
        let other_hash = compute_line_hash(other);
        let hint_low = &hint_hash[2..];
        let other_low = &other_hash[2..];
        let victim = (0..10_000)
            .map(|index| format!("victim-{index}"))
            .find(|text| {
                let hash = compute_line_hash(text);
                let low = &hash[2..];
                low != hint_low && low != other_low
            })
            .expect("victim");
        let legacy = compute_line_hash(&victim)[2..].to_string();
        let lines = vec![hint.to_string(), other.to_string(), victim];
        let error = locate_line_by_hash(&lines, 1, &legacy).expect_err("no relocate");
        let text = error.to_string();
        assert!(text.contains("not relocated"), "{text}");
        assert!(text.contains(&legacy), "{text}");
    }

    #[test]
    fn loose_retry_refuses_two_whitespace_equivalent_lines() {
        let lines = vec![
            "keep".to_string(),
            "    alpha".to_string(),
            "  alpha".to_string(),
        ];
        let hash = compute_line_hash("alpha");
        let error = locate_line_by_hash(&lines, 1, &hash).expect_err("ambiguous loose");
        let text = error.to_string();
        assert!(text.contains("multiple lines"), "{text}");
        assert!(text.contains("alpha"), "{text}");
    }
}
