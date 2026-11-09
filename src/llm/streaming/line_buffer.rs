//! Zero-copy line buffer for SSE streaming
//!
//! Eliminates repeated allocations when parsing line-delimited streams
//! by using a cursor-based approach with periodic compaction.

/// Efficient line buffer for streaming parsers
///
/// # Example
/// ```
/// use caliche::llm::streaming::LineBuffer;
///
/// let mut buffer = LineBuffer::new();
/// buffer.push("data: {\"chunk\":1}\ndata:");
///
/// while let Some(line) = buffer.next_line() {
///     // Process line without allocation
///     println!("Line: {}", line);
/// }
///
/// buffer.compact(); // Remove processed data
/// ```
pub struct LineBuffer {
    buffer: String,
    cursor: usize,
    compact_threshold: usize,
}

impl LineBuffer {
    /// Create new buffer with default 8KB capacity
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(8192)
    }

    /// Create buffer with specified capacity
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buffer: String::with_capacity(capacity),
            cursor: 0,
            compact_threshold: capacity / 2,
        }
    }

    /// Add chunk to buffer
    #[inline]
    pub fn push(&mut self, s: &str) {
        self.buffer.push_str(s);
    }

    /// Get next complete line without allocation
    ///
    /// Returns `Some(&str)` pointing into internal buffer if line available,
    /// `None` if no complete line exists yet.
    ///
    /// If there's remaining data without a newline and we've already processed
    /// at least one line, returns the remaining data as a line.
    pub fn next_line(&mut self) -> Option<&str> {
        // Skip leading newlines (empty lines)
        while self.cursor < self.buffer.len() && self.buffer.as_bytes()[self.cursor] == b'\n' {
            self.cursor += 1;
        }

        // Auto-compact if threshold exceeded (before extracting line to keep indices valid)
        if self.cursor >= self.compact_threshold {
            self.compact();
        }

        let remaining = &self.buffer[self.cursor..];

        if let Some(newline_pos) = remaining.find('\n') {
            let line_start = self.cursor;
            let line_end = self.cursor + newline_pos;
            self.cursor = line_end + 1;

            Some(self.buffer[line_start..line_end].trim())
        } else if !remaining.is_empty() && self.cursor > 0 {
            // Return remaining data as a line if we've already processed at least one line
            // This handles the case where the last line doesn't have a trailing newline
            // Note: We advance the cursor to consume the line
            let line_start = self.cursor;
            self.cursor = self.buffer.len();
            Some(self.buffer[line_start..].trim())
        } else {
            None
        }
    }

    /// Manually compact buffer, removing processed lines
    ///
    /// This is automatically called when internal threshold is reached,
    /// but can be called manually for fine-grained control.
    pub fn compact(&mut self) {
        if self.cursor > 0 {
            self.buffer.drain(..self.cursor);
            self.cursor = 0;
        }
    }

    /// Check if buffer has unprocessed data
    #[must_use]
    pub const fn has_remaining(&self) -> bool {
        self.cursor < self.buffer.len()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
    }
}

impl Default for LineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_buffer_basic() {
        let mut buffer = LineBuffer::new();
        buffer.push("line1\nline2\nline3");

        assert_eq!(buffer.next_line(), Some("line1"));
        assert_eq!(buffer.next_line(), Some("line2"));
        assert_eq!(buffer.next_line(), Some("line3"));
        assert_eq!(buffer.next_line(), None);
    }

    #[test]
    fn test_line_buffer_partial() {
        let mut buffer = LineBuffer::new();
        buffer.push("line1\nline2");

        assert_eq!(buffer.next_line(), Some("line1"));
        assert_eq!(buffer.next_line(), Some("line2"));
        assert_eq!(buffer.next_line(), None);

        buffer.push("\nline3");
        assert_eq!(buffer.next_line(), Some("line3"));
    }

    #[test]
    fn test_line_buffer_trimming() {
        let mut buffer = LineBuffer::new();
        buffer.push("  line1  \n  line2  \n");

        assert_eq!(buffer.next_line(), Some("line1"));
        assert_eq!(buffer.next_line(), Some("line2"));
    }

    #[test]
    fn test_line_buffer_compact() {
        let mut buffer = LineBuffer::with_capacity(100);
        buffer.push("line1\nline2\nline3\n");

        buffer.next_line();
        buffer.next_line();

        // Manually compact
        buffer.compact();

        assert_eq!(buffer.next_line(), Some("line3"));
    }

    #[test]
    fn test_line_buffer_clear() {
        let mut buffer = LineBuffer::new();
        buffer.push("line1\nline2");
        buffer.clear();

        assert_eq!(buffer.next_line(), None);
        assert_eq!(buffer.has_remaining(), false);
    }

    #[test]
    fn test_line_buffer_has_remaining() {
        let mut buffer = LineBuffer::new();
        buffer.push("line1\nline2");

        assert_eq!(buffer.has_remaining(), true);
        buffer.next_line();
        assert_eq!(buffer.has_remaining(), true);
        buffer.next_line();
        assert_eq!(buffer.has_remaining(), false);
    }
}
