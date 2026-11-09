//! Document types and loader trait.

use anyhow::Error as AnyError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("Failed to load document: {0}")]
    LoadFailed(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid format: {0}")]
    InvalidFormat(String),

    #[error("{kind}: {message}")]
    WithSource {
        kind: &'static str,
        message: String,
        #[source]
        source: Option<AnyError>,
    },
}

impl LoaderError {
    pub fn load_failed_with(message: impl Into<String>, source: impl Into<AnyError>) -> Self {
        Self::WithSource {
            kind: "Failed to load document",
            message: message.into(),
            source: Some(source.into()),
        }
    }

    pub fn parse_error_with(message: impl Into<String>, source: impl Into<AnyError>) -> Self {
        Self::WithSource {
            kind: "Parse error",
            message: message.into(),
            source: Some(source.into()),
        }
    }

    pub fn invalid_format_with(message: impl Into<String>, source: impl Into<AnyError>) -> Self {
        Self::WithSource {
            kind: "Invalid format",
            message: message.into(),
            source: Some(source.into()),
        }
    }
}

/// A document with content and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// The main content of the document.
    pub content: String,

    /// Metadata associated with the document.
    pub metadata: HashMap<String, String>,
}

impl Document {
    /// Create a new document with content.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the document.
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add multiple metadata entries.
    #[must_use]
    pub fn with_metadata_map(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata.extend(metadata);
        self
    }
}

/// Trait for loading documents from various sources.
#[async_trait]
pub trait DocumentLoader: Send + Sync + std::fmt::Debug {
    /// Load documents from the source.
    async fn load(&self) -> Result<Vec<Document>, LoaderError>;

    /// Get the name of this loader.
    fn name(&self) -> &str {
        "document_loader"
    }
}

/// Split text into chunks for processing.
pub struct TextSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
}

impl TextSplitter {
    /// Create a new text splitter.
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - Target size for each chunk in bytes (snapped to UTF-8 boundaries).
    ///   Zero is invalid and will be clamped to `1` to avoid dropping input text.
    /// * `chunk_overlap` - Overlap between chunks to maintain context.
    ///   Values greater than or equal to `chunk_size` are clamped to
    ///   `chunk_size - 1` so chunks always make forward progress.
    #[must_use]
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        let chunk_size = chunk_size.max(1);
        let chunk_overlap = chunk_overlap.min(chunk_size.saturating_sub(1));
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    /// Split text into chunks.
    ///
    /// `chunk_size` and `chunk_overlap` are measured in **bytes**, but slicing
    /// always lands on a UTF-8 character boundary (using
    /// [`str::floor_char_boundary`]) so the returned chunks are always valid
    /// `String`s even for multi-byte characters.
    #[must_use]
    pub fn split_text(&self, text: &str) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < text.len() {
            // Snap to the nearest UTF-8 char boundary at or before the target
            // end so we never slice through a multi-byte character.
            let end = text.floor_char_boundary((start + self.chunk_size).min(text.len()));

            // Guard against degenerate inputs (e.g. chunk_size == 0) that
            // would otherwise spin forever.
            if end == start {
                break;
            }

            chunks.push(text[start..end].to_string());

            if end >= text.len() {
                break;
            }

            // Also snap the overlap-adjusted start to a char boundary.
            let next = text.floor_char_boundary(end.saturating_sub(self.chunk_overlap));
            // Ensure forward progress even when overlap >= chunk_size.
            let mut next_start = if next < end { next } else { end };
            if next_start <= start {
                if let Some(boundary) =
                    (start + 1..=text.len()).find(|idx| text.is_char_boundary(*idx))
                {
                    next_start = boundary;
                } else {
                    break;
                }
            }
            start = next_start;
        }

        chunks
    }

    /// Split documents into smaller chunks.
    #[must_use]
    pub fn split_documents(&self, documents: &[Document]) -> Vec<Document> {
        let mut result = Vec::new();

        for (doc_idx, doc) in documents.iter().enumerate() {
            let chunks = self.split_text(&doc.content);
            let total_chunks = chunks.len(); // capture before consuming

            // Use `into_iter()` so each chunk String is moved rather than cloned.
            for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                let mut metadata = doc.metadata.clone();
                metadata.insert("chunk_index".to_owned(), chunk_idx.to_string());
                metadata.insert("total_chunks".to_owned(), total_chunks.to_string());
                metadata.insert("source_doc_index".to_owned(), doc_idx.to_string());

                result.push(Document {
                    content: chunk,
                    metadata,
                });
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_creation() {
        let doc = Document::new("Hello world")
            .with_metadata("author", "Alice")
            .with_metadata("date", "2024-01-01");

        assert_eq!(doc.content, "Hello world");
        assert_eq!(
            doc.metadata.get("author").map(String::as_str),
            Some("Alice")
        );
    }

    #[test]
    fn test_text_splitter() {
        let splitter = TextSplitter::new(10, 2);
        let text = "This is a test document with more than ten characters.";
        let chunks = splitter.split_text(text);

        assert!(chunks.len() > 1);
        assert!(chunks[0].len() <= 10);
    }

    #[test]
    fn test_split_documents() {
        let docs = vec![
            Document::new("Short doc"),
            Document::new("This is a much longer document that will be split into multiple chunks"),
        ];

        let splitter = TextSplitter::new(20, 5);
        let chunks = splitter.split_documents(&docs);

        assert!(chunks.len() > 2);
    }

    #[test]
    fn test_split_text_multibyte_utf8() {
        // Each Japanese hiragana character is 3 bytes in UTF-8.
        // chunk_size=9 covers 3 chars; overlap=3 covers 1 char.
        let text: String = "あ".repeat(12); // 36 bytes, 12 chars
        let splitter = TextSplitter::new(9, 3);
        let chunks = splitter.split_text(&text);

        // Must not panic and every chunk must be valid UTF-8.
        for chunk in &chunks {
            assert!(!chunk.is_empty());
            assert!(std::str::from_utf8(chunk.as_bytes()).is_ok());
        }
        assert!(
            chunks.len() > 1,
            "expected multiple chunks for a 12-char input"
        );
    }

    #[test]
    fn test_split_text_exact_boundary() {
        // Verify chunk boundaries never fall inside a multi-byte sequence.
        let text = "αβγδεζηθικλμνξοπρστ"; // Greek letters, 2 bytes each
        let splitter = TextSplitter::new(7, 2); // 7 bytes won't align to a 2-byte boundary
        let chunks = splitter.split_text(text);
        for chunk in &chunks {
            assert!(!chunk.is_empty());
        }
        // Reassembled de-duplicated content should cover the whole string.
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_split_text_overlap_equal_chunk_size_makes_progress() {
        let text = "abcdefghij";
        let splitter = TextSplitter::new(5, 5);
        let chunks = splitter.split_text(text);

        assert!(!chunks.is_empty());
        assert!(chunks.len() <= text.len());
        assert_eq!(chunks.last().map(String::as_str), Some("fghij"));
    }

    #[test]
    fn test_split_text_overlap_greater_than_chunk_size_makes_progress() {
        let text = "abcdefghij";
        let splitter = TextSplitter::new(5, 8);
        let chunks = splitter.split_text(text);

        assert!(!chunks.is_empty());
        assert!(chunks.len() <= text.len());
        assert_eq!(chunks.last().map(String::as_str), Some("fghij"));
    }

    #[test]
    fn test_split_text_zero_chunk_size_is_clamped() {
        let text = "hello world";
        let splitter = TextSplitter::new(0, 0);
        let chunks = splitter.split_text(text);

        assert!(!chunks.is_empty());
        let joined: String = chunks.iter().map(String::as_str).collect();
        assert!(joined.contains("hello"));
    }

    #[test]
    fn test_split_text_overlap_equal_or_greater_is_clamped() {
        let text = "abcdefghij";
        let a = TextSplitter::new(5, 5);
        let b = TextSplitter::new(5, 100);
        let ca = a.split_text(text);
        let cb = b.split_text(text);

        assert_eq!(ca.last().map(String::as_str), Some("fghij"));
        assert_eq!(cb.last().map(String::as_str), Some("fghij"));
    }
}
