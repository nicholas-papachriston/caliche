//! Document chunking strategies for RAG applications.

use crate::errors::DripstoneError;
use serde::{Deserialize, Serialize};

/// A chunk of text with positional information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    /// The text content of the chunk
    pub content: String,
    /// Start position in the original document (character offset)
    pub start_offset: usize,
    /// End position in the original document (character offset)
    pub end_offset: usize,
    /// Optional metadata for the chunk
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Chunk {
    #[must_use]
    pub const fn new(content: String, start_offset: usize, end_offset: usize) -> Self {
        Self {
            content,
            start_offset,
            end_offset,
            metadata: None,
        }
    }

    #[must_use]
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Returns the length of the chunk in characters.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.content.len()
    }

    /// Returns true if the chunk is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.content.is_empty()
    }
}

/// Trait for document chunking strategies.
pub trait ChunkStrategy: Send + Sync {
    /// Chunks the given text into a vector of chunks.
    fn chunk(&self, text: &str) -> Vec<Chunk>;
}

/// Fixed-size chunking with configurable overlap.
pub struct FixedSizeChunker {
    /// Size of each chunk in characters
    chunk_size: usize,
    /// Overlap between consecutive chunks in characters
    overlap: usize,
}

impl FixedSizeChunker {
    /// Creates a new fixed-size chunker.
    ///
    /// # Arguments
    /// * `chunk_size` - Size of each chunk in characters (must be `> 0`).
    /// * `overlap` - Number of overlapping characters between chunks (must
    ///   be strictly less than `chunk_size`, otherwise the slider would never
    ///   advance).
    ///
    /// # Panics
    ///
    /// Panics if the arguments would produce a non-advancing slider. Use
    /// [`FixedSizeChunker::try_new`] if you want to propagate configuration
    /// errors as `Result` values instead.
    #[expect(
        clippy::panic,
        reason = "documented infallible constructor; use try_new for Result"
    )]
    #[must_use]
    pub fn new(chunk_size: usize, overlap: usize) -> Self {
        match Self::try_new(chunk_size, overlap) {
            Ok(c) => c,
            Err(e) => panic!("FixedSizeChunker::new: {e}"),
        }
    }

    /// Fallible counterpart to [`FixedSizeChunker::new`].
    ///
    /// Returns [`DripstoneError::configuration`] on invalid arguments so
    /// callers can surface the problem to their users without aborting the
    /// process.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn try_new(chunk_size: usize, overlap: usize) -> Result<Self, DripstoneError> {
        if chunk_size == 0 {
            return Err(DripstoneError::configuration(
                "FixedSizeChunker: chunk_size must be > 0",
            ));
        }
        if overlap >= chunk_size {
            return Err(DripstoneError::configuration(format!(
                "FixedSizeChunker: overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
            )));
        }
        Ok(Self {
            chunk_size,
            overlap,
        })
    }

    /// Creates a chunker with no overlap.
    #[must_use]
    pub fn with_size(chunk_size: usize) -> Self {
        Self::new(chunk_size, 0)
    }
}

impl ChunkStrategy for FixedSizeChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total_len = chars.len();

        if total_len == 0 {
            return chunks;
        }

        let step = self.chunk_size - self.overlap;
        let mut start = 0;

        while start < total_len {
            let end = (start + self.chunk_size).min(total_len);
            let chunk_chars = &chars[start..end];
            let content: String = chunk_chars.iter().collect();

            if !content.trim().is_empty() {
                chunks.push(Chunk::new(content, start, end));
            }

            if end == total_len {
                break;
            }

            start += step;
        }

        chunks
    }
}

/// Sentence-aware chunking that tries to split on sentence boundaries.
pub struct SentenceChunker {
    /// Approximate target size for chunks
    target_size: usize,
    /// Maximum size before forcing a split
    max_size: usize,
}

impl SentenceChunker {
    /// Creates a new sentence-aware chunker.
    ///
    /// # Panics
    ///
    /// Panics when `target_size` exceeds `max_size`.
    #[must_use]
    pub fn new(target_size: usize, max_size: usize) -> Self {
        assert!(target_size <= max_size, "target_size must be <= max_size");
        Self {
            target_size,
            max_size,
        }
    }

    /// Creates a chunker with default `max_size` (1.5x target).
    #[must_use]
    pub fn with_target(target_size: usize) -> Self {
        Self::new(target_size, target_size * 3 / 2)
    }

    fn split_into_sentences(text: &str) -> Vec<(usize, usize, String)> {
        // Simple sentence splitting on common terminators
        let mut sentences = Vec::new();
        let mut start = 0;
        let mut current = String::new();

        for (i, c) in text.char_indices() {
            current.push(c);

            if matches!(c, '.' | '!' | '?') {
                // Look ahead to see if this is really a sentence end
                let remaining = &text[i + c.len_utf8()..];
                if remaining.starts_with(char::is_whitespace) || remaining.is_empty() {
                    let trimmed = current.trim().to_string();
                    if !trimmed.is_empty() {
                        sentences.push((start, i + c.len_utf8(), trimmed));
                    }
                    current.clear();
                    start = i + c.len_utf8();
                }
            }
        }

        // Add any remaining text as a final sentence
        if !current.trim().is_empty() {
            sentences.push((start, text.len(), current.trim().to_string()));
        }

        sentences
    }
}

impl ChunkStrategy for SentenceChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let sentences = Self::split_into_sentences(text);
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_start = 0;
        let mut chunk_end = 0;

        for (sent_start, sent_end, sentence) in sentences {
            let sentence_len = sentence.len();

            if current_chunk.is_empty() {
                // Start a new chunk — move owned sentence, no allocation.
                current_chunk = sentence;
                chunk_start = sent_start;
                chunk_end = sent_end;
            } else if current_chunk.len() + sentence_len < self.target_size {
                // Add to current chunk
                current_chunk.push(' ');
                current_chunk.push_str(&sentence);
                chunk_end = sent_end;
            } else if current_chunk.len() + sentence_len < self.max_size {
                // Add to current chunk even though we exceeded target
                current_chunk.push(' ');
                current_chunk.push_str(&sentence);
                chunk_end = sent_end;

                // Finalize this chunk — take ownership instead of clone + clear.
                chunks.push(Chunk::new(
                    std::mem::take(&mut current_chunk),
                    chunk_start,
                    chunk_end,
                ));
            } else {
                // Finalize current chunk and start new one
                if !current_chunk.is_empty() {
                    chunks.push(Chunk::new(
                        std::mem::take(&mut current_chunk),
                        chunk_start,
                        chunk_end,
                    ));
                }
                // Move owned sentence, no allocation.
                current_chunk = sentence;
                chunk_start = sent_start;
                chunk_end = sent_end;
            }
        }

        // Add final chunk
        if !current_chunk.is_empty() {
            chunks.push(Chunk::new(current_chunk, chunk_start, chunk_end));
        }

        chunks
    }
}

/// Sliding window chunker with configurable stride.
pub struct SlidingWindowChunker {
    /// Window size in characters
    window_size: usize,
    /// Stride (step size) in characters
    stride: usize,
}

impl SlidingWindowChunker {
    /// Creates a new sliding window chunker.
    ///
    /// # Panics
    ///
    /// Panics when `stride` is zero or exceeds `window_size`.
    #[must_use]
    pub fn new(window_size: usize, stride: usize) -> Self {
        assert!(stride > 0, "stride must be greater than 0");
        assert!(
            stride <= window_size,
            "stride should be <= window_size for overlap"
        );
        Self {
            window_size,
            stride,
        }
    }

    /// Creates a chunker with 50% overlap (stride = `window_size` / 2).
    #[must_use]
    pub fn with_window(window_size: usize) -> Self {
        Self::new(window_size, window_size / 2)
    }
}

impl ChunkStrategy for SlidingWindowChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let total_len = chars.len();

        if total_len == 0 {
            return chunks;
        }

        let mut start = 0;

        while start < total_len {
            let end = (start + self.window_size).min(total_len);
            let chunk_chars = &chars[start..end];
            let content: String = chunk_chars.iter().collect();

            if !content.trim().is_empty() {
                chunks.push(Chunk::new(content, start, end));
            }

            if end == total_len {
                break;
            }

            start += self.stride;
        }

        chunks
    }
}

/// Sentence-aware chunker intended as the scaffold for a future semantic
/// chunker driven by embedding similarity.
///
/// Despite the name suggesting otherwise, the current implementation does
/// not compute embeddings and is structurally identical to
/// [`SentenceChunker`] once `min_chunk_size`/`max_chunk_size` are applied.
/// The `similarity_threshold` parameter is stored for API parity with a
/// future semantic implementation but is not consulted when chunking.
///
pub struct SentenceAwareChunker {
    min_chunk_size: usize,
    max_chunk_size: usize,
    _similarity_threshold: f32,
}

impl SentenceAwareChunker {
    /// Creates a new sentence-aware chunker.
    ///
    /// # Panics
    ///
    /// Panics if `min_chunk_size > max_chunk_size` or
    /// `similarity_threshold` is outside `0.0..=1.0`. Use
    /// [`SentenceAwareChunker::try_new`] for a fallible variant.
    #[expect(
        clippy::panic,
        reason = "documented infallible constructor; use try_new for Result"
    )]
    #[must_use]
    pub fn new(min_chunk_size: usize, max_chunk_size: usize, similarity_threshold: f32) -> Self {
        match Self::try_new(min_chunk_size, max_chunk_size, similarity_threshold) {
            Ok(c) => c,
            Err(e) => panic!("SentenceAwareChunker::new: {e}"),
        }
    }

    /// Fallible constructor variant.
    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn try_new(
        min_chunk_size: usize,
        max_chunk_size: usize,
        similarity_threshold: f32,
    ) -> Result<Self, DripstoneError> {
        if min_chunk_size > max_chunk_size {
            return Err(DripstoneError::configuration(format!(
                "SentenceAwareChunker: min_chunk_size ({min_chunk_size}) must be <= max_chunk_size ({max_chunk_size})"
            )));
        }
        if !(0.0..=1.0).contains(&similarity_threshold) {
            return Err(DripstoneError::configuration(format!(
                "SentenceAwareChunker: similarity_threshold ({similarity_threshold}) must be in [0.0, 1.0]"
            )));
        }
        Ok(Self {
            min_chunk_size,
            max_chunk_size,
            _similarity_threshold: similarity_threshold,
        })
    }

    /// Creates a chunker with default settings.
    #[must_use]
    pub fn with_threshold(similarity_threshold: f32) -> Self {
        Self::new(100, 1000, similarity_threshold)
    }
}

impl ChunkStrategy for SentenceAwareChunker {
    fn chunk(&self, text: &str) -> Vec<Chunk> {
        let sentence_chunker = SentenceChunker::new(self.min_chunk_size, self.max_chunk_size);
        sentence_chunker.chunk(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_size_chunker() {
        let text = "Hello world! This is a test document with some content.";
        let chunker = FixedSizeChunker::new(20, 5);
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.len() <= 20);
        }
    }

    #[test]
    fn test_fixed_size_no_overlap() {
        let text = "0123456789".repeat(10);
        let chunker = FixedSizeChunker::with_size(10);
        let chunks = chunker.chunk(&text);

        assert_eq!(chunks.len(), 10);
        assert_eq!(chunks[0].content.len(), 10);
    }

    #[test]
    fn test_sentence_chunker() {
        let text = "This is sentence one. This is sentence two. And here is three.";
        let chunker = SentenceChunker::with_target(30);
        let chunks = chunker.chunk(text);

        assert!(!chunks.is_empty());
        // Should try to keep sentences together
        for chunk in &chunks {
            assert!(chunk.content.contains('.'));
        }
    }

    #[test]
    fn test_sliding_window_chunker() {
        let text = "ABCDEFGHIJ";
        let chunker = SlidingWindowChunker::new(5, 3);
        let chunks = chunker.chunk(text);

        // Should have overlapping content
        assert!(chunks.len() > 1);
        for i in 1..chunks.len() {
            let prev = &chunks[i - 1].content;
            let curr = &chunks[i].content;
            // Check that there's some overlap
            assert!(prev[prev.len() - 2..].chars().any(|c| curr.contains(c)));
        }
    }

    #[test]
    fn test_empty_text() {
        let chunker = FixedSizeChunker::with_size(10);
        let chunks = chunker.chunk("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_metadata() {
        let chunk = Chunk::new("test".to_string(), 0, 4)
            .with_metadata(serde_json::json!({"source": "doc1"}));

        assert!(chunk.metadata.is_some());
    }
}
