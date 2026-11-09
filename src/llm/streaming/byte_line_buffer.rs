//! Byte-oriented line buffer for SSE streaming.
//!
//! Buffers raw bytes and decodes UTF-8 only when a complete line (ending with `\n`)
//! is available. This avoids "incomplete utf-8 byte sequence" errors when HTTP
//! chunks split multi-byte UTF-8 characters.

/// Byte line buffer that decodes UTF-8 only at line boundaries.
///
/// Use this instead of `LineBuffer` when reading from a raw byte stream (e.g.
/// `response.bytes_stream()`). Chunks may split multi-byte UTF-8 codepoints;
/// buffering bytes and decoding only complete lines prevents decode errors.
pub struct ByteLineBuffer {
    buffer: Vec<u8>,
    cursor: usize,
    compact_threshold: usize,
}

fn trim_ascii_whitespace(mut b: &[u8]) -> &[u8] {
    while b
        .first()
        .is_some_and(|&c| c == b' ' || c == b'\t' || c == b'\r')
    {
        b = &b[1..];
    }
    while b
        .last()
        .is_some_and(|&c| c == b' ' || c == b'\t' || c == b'\r')
    {
        b = &b[..b.len() - 1];
    }
    b
}

impl ByteLineBuffer {
    /// Create a new buffer with default 8KB capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(8192)
    }

    /// Create a buffer with the specified capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            cursor: 0,
            compact_threshold: capacity / 2,
        }
    }

    /// Append raw bytes. Never decode here; decoding happens only in `next_line`.
    #[inline]
    pub fn push_bytes(&mut self, b: &[u8]) {
        self.buffer.extend_from_slice(b);
    }

    /// Remove processed bytes from the front of the buffer.
    fn compact(&mut self) {
        if self.cursor > 0 {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }

    /// Return the next complete line as a `String`, or `None` if no full line yet.
    ///
    /// Only returns a line when a newline is present; avoids decoding incomplete UTF-8
    /// when chunk boundaries split multi-byte characters. Skips leading newlines.
    /// Trims ASCII whitespace (space, tab, CR) from each line.
    /// Returns `Some(Err(...))` if a complete line is not valid UTF-8 (e.g. corrupt API output).
    pub fn next_line(&mut self) -> Option<Result<String, std::str::Utf8Error>> {
        // Skip leading newlines (empty lines)
        while self.cursor < self.buffer.len() && self.buffer[self.cursor] == b'\n' {
            self.cursor += 1;
        }

        if self.cursor >= self.compact_threshold {
            self.compact();
        }

        let rest = &self.buffer[self.cursor..];

        // Only return a line when we see a newline. Treating "rest without newline" as a line
        // can decode bytes that end mid–multi-byte UTF-8 character and cause decode errors.
        let (line_bytes, advance) = if let Some(pos) = rest.iter().position(|&x| x == b'\n') {
            (&rest[..pos], pos + 1)
        } else {
            return None;
        };

        self.cursor += advance;
        let trimmed = trim_ascii_whitespace(line_bytes);
        let s = std::str::from_utf8(trimmed).map(std::borrow::ToOwned::to_owned);
        Some(s)
    }
}

impl Default for ByteLineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_line_utf8_boundary() {
        // Simulate chunk boundary splitting multi-byte UTF-8 (heart emoji = U+2764 = e2 9d a4)
        let mut buf = ByteLineBuffer::new();
        buf.push_bytes(b"data: {\"delta\":{\"content\":\"");
        assert!(buf.next_line().is_none());
        buf.push_bytes("\u{2764}".as_bytes()); // 3 bytes
        assert!(buf.next_line().is_none());
        buf.push_bytes(b"\"}}\n");
        let line = buf.next_line().expect("one line");
        let line = line.expect("valid utf-8");
        assert!(line.contains('\u{2764}'));
    }

    #[test]
    fn next_line_skips_empty() {
        let mut buf = ByteLineBuffer::new();
        buf.push_bytes(b"\n\ndata: [DONE]\n");
        let line = buf.next_line().expect("first non-empty");
        let line = line.expect("valid utf-8");
        assert_eq!(line, "data: [DONE]");
    }
}
