//! RAG (Retrieval-Augmented Generation) utilities and features.

pub mod ast;
pub mod chunking;
pub mod citations;

#[cfg(feature = "rust-stemmers")]
pub mod hybrid;

#[cfg(feature = "embedding-ort")]
pub mod rerank;

pub use chunking::{
    Chunk, ChunkStrategy, FixedSizeChunker, SentenceAwareChunker, SentenceChunker,
    SlidingWindowChunker,
};
pub use citations::{CitationHelper, ContextReconstructor, SourceMetadata};

#[cfg(feature = "rust-stemmers")]
pub use hybrid::HybridSearchHelper;

#[cfg(feature = "embedding-ort")]
pub use rerank::{CrossEncoderConfig, CrossEncoderReranker};
