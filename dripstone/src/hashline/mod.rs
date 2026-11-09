//! Content-addressed hashline tagging for vectorization and RAG result formatting.
//!
//! Provides compact content fingerprints (`line:hash`) for chunks, enabling
//! incremental indexing, content-addressed deduplication, and RAG result
//! formatting for line-tagged edit tools.

use std::fmt::Write;

/// FNV-1a offset basis (32-bit)
const FNV_OFFSET_BASIS: u32 = 0x811c_9dc5;

/// FNV-1a prime multiplier
const FNV_PRIME: u32 = 0x0100_0193;

/// Options for content hash computation (`hash_bytes`, normalization).
#[derive(Debug, Clone)]
pub struct HashlineOptions {
    /// Number of hash bytes: 1 = 2 hex chars, 2 = 4 hex chars
    pub hash_bytes: u8,
    /// Normalize trailing whitespace before hashing
    pub normalize_trailing_ws: bool,
}

impl Default for HashlineOptions {
    fn default() -> Self {
        Self {
            hash_bytes: 1,
            normalize_trailing_ws: true,
        }
    }
}

impl HashlineOptions {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_hash_bytes(mut self, bytes: u8) -> Self {
        self.hash_bytes = bytes.min(4);
        self
    }

    #[must_use]
    pub const fn with_normalize_trailing_ws(mut self, normalize: bool) -> Self {
        self.normalize_trailing_ws = normalize;
        self
    }
}

fn normalize_content(content: &str, normalize_ws: bool) -> &str {
    if normalize_ws {
        content.trim_end()
    } else {
        content
    }
}

fn compute_fnv1a(content: &str, normalize_ws: bool) -> u32 {
    let input = normalize_content(content, normalize_ws);
    let mut hash: u32 = FNV_OFFSET_BASIS;
    for byte in input.as_bytes() {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Compute content hash for a line or chunk. FNV-1a, deterministic.
#[must_use]
pub fn compute_content_hash(content: &str, options: &HashlineOptions) -> String {
    let hash = compute_fnv1a(content, options.normalize_trailing_ws);
    let bits = (options.hash_bytes as usize) * 8;
    let mask = if bits >= 32 {
        0xFFFF_FFFFu32
    } else {
        (1u32 << bits) - 1
    };
    let truncated = hash & mask;
    let hex_chars = (options.hash_bytes as usize) * 2;
    format!("{truncated:0>hex_chars$x}")
}

/// Compute line-by-line hashes for text. Returns (`line_num`, hash) pairs (1-indexed).
#[must_use]
pub fn compute_line_hashes(text: &str, options: &HashlineOptions) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .map(|(i, line)| (i + 1, compute_content_hash(line, options)))
        .collect()
}

/// Compute single hash for entire chunk (used for incremental indexing).
#[must_use]
pub fn compute_chunk_hash(content: &str, options: &HashlineOptions) -> String {
    compute_content_hash(content, options)
}

/// Format content with hashline tags for display. Used when returning RAG results.
/// `start_line` is the 1-indexed document line number of the first line in the content.
#[must_use]
pub fn format_content_with_hashlines(
    text: &str,
    start_line: usize,
    options: &HashlineOptions,
) -> String {
    let mut out = String::new();
    for (i, line) in text.lines().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let line_num = start_line + i;
        let hash = compute_content_hash(line, options);
        let _ = write!(out, "{line_num}:{hash}| {line}");
    }
    if text.ends_with('\n') && !text.is_empty() && !out.is_empty() {
        out.push('\n');
    }
    out
}

/// Returns the 1-indexed line number at the given character offset.
///
/// Offsets that land mid-UTF-8 scalar are floored to the previous char boundary
/// so slicing never panics.
#[must_use]
pub fn line_number_at_offset(content: &str, offset: usize) -> usize {
    if content.is_empty() {
        return 1;
    }
    let mut end = offset.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    content[..end].bytes().filter(|&byte| byte == b'\n').count() + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_content_hash() {
        let opts = HashlineOptions::default();
        let h1 = compute_content_hash("hello", &opts);
        let h2 = compute_content_hash("hello", &opts);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 2);

        let h3 = compute_content_hash("world", &opts);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_normalize_trailing_ws() {
        let opts = HashlineOptions::default();
        let h1 = compute_content_hash("line", &opts);
        let h2 = compute_content_hash("line  ", &opts);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_compute_line_hashes() {
        let opts = HashlineOptions::default();
        let text = "line one\nline two\nline three";
        let hashes = compute_line_hashes(text, &opts);
        assert_eq!(hashes.len(), 3);
        assert_eq!(hashes[0].0, 1);
        assert_eq!(hashes[1].0, 2);
        assert_eq!(hashes[2].0, 3);
        assert_eq!(hashes[0].1.len(), 2);
    }

    #[test]
    fn test_format_content_with_hashlines() {
        let opts = HashlineOptions::default();
        let text = "fn foo() {\n  return 1;\n}";
        let formatted = format_content_with_hashlines(text, 10, &opts);
        assert!(formatted.starts_with("10:"));
        assert!(formatted.contains("| fn foo() {"));
        assert!(formatted.contains("11:"));
        assert!(formatted.contains("|   return 1;"));
        assert!(formatted.contains("12:"));
    }

    #[test]
    fn test_line_number_at_offset() {
        let content = "line1\nline2\nline3";
        // 1-indexed: offset 0 is the first character of line 1.
        assert_eq!(line_number_at_offset(content, 0), 1);
        assert_eq!(line_number_at_offset(content, 5), 1); // newline after line1
        assert_eq!(line_number_at_offset(content, 6), 2); // first char of line2
        assert_eq!(line_number_at_offset(content, 12), 3);
        assert_eq!(line_number_at_offset(content, 100), 3);
        assert_eq!(line_number_at_offset("", 0), 1);

        // Mid-codepoint offset must not panic and floors to a char boundary.
        let utf8 = "a😀b";
        let mid_emoji = 2; // inside the 4-byte grinning face
        assert_eq!(line_number_at_offset(utf8, mid_emoji), 1);
    }
}
