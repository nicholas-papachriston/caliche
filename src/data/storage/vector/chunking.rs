//! Document chunking strategies re-exported from `Dripstone`.
//!
//! This module provides various text chunking strategies optimized for RAG applications:
//! - **`FixedSizeChunker`**: Simple fixed-size chunks with configurable overlap
//! - **`SentenceChunker`**: Sentence-aware chunking that respects sentence boundaries
//! - **`SentenceAwareChunker`**: Sentence-aware chunker with similarity-based merging
//! - **`SlidingWindowChunker`**: Sliding window with configurable overlap
//!
//! # Examples
//!
//! ```rust
//! use caliche::vecstore::chunking::{FixedSizeChunker, ChunkStrategy};
//!
//! let chunker = FixedSizeChunker::new(500, 50); // 500 chars, 50 overlap
//! let chunks = chunker.chunk("Your long document text here...");
//!
//! for chunk in chunks {
//!     println!("Chunk: {} (pos: {}-{})", chunk.content, chunk.start_offset, chunk.end_offset);
//! }
//! ```

// Re-export dripstone chunking types
pub use dripstone::rag::chunking::{
    Chunk, ChunkStrategy, FixedSizeChunker, SentenceAwareChunker, SentenceChunker,
    SlidingWindowChunker,
};
