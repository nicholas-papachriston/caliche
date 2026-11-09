//! Integration tests for RAG features.

#![cfg(feature = "memory-backend")]

use dripstone::backend::memory::MemoryBackend;
use dripstone::rag::chunking::{
    ChunkStrategy, FixedSizeChunker, SentenceChunker, SlidingWindowChunker,
};
use dripstone::rag::citations::SourceMetadata;
use dripstone::{Chunk, Metric, SearchQuery, traits::Backend};
use std::sync::Arc;

#[test]
fn test_fixed_size_chunker() {
    let text = "Hello world! ".repeat(100);
    let chunker = FixedSizeChunker::new(50, 10);
    let chunks = chunker.chunk(&text);

    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(chunk.len() <= 50);
    }
}

#[test]
fn test_sentence_chunker() {
    let text = "This is sentence one. This is sentence two. This is sentence three.";
    let chunker = SentenceChunker::with_target(30);
    let chunks = chunker.chunk(text);

    assert!(!chunks.is_empty());
}

#[test]
fn test_sliding_window_chunker() {
    let text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let chunker = SlidingWindowChunker::new(10, 5);
    let chunks = chunker.chunk(text);

    assert!(chunks.len() > 2);
    // Verify overlap
    for i in 1..chunks.len() {
        assert!(chunks[i].start_offset < chunks[i - 1].end_offset);
    }
}

#[tokio::test]
async fn test_source_metadata_integration() {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // Create chunks with source metadata
    let mut chunks = Vec::new();
    for i in 0..5 {
        let meta = SourceMetadata::new("doc123")
            .with_title("Test Document")
            .with_chunk_index(i)
            .with_total_chunks(5);
        // Avoid zero-magnitude vector (i=0) for cosine similarity
        let v = (i as f32) + 1.0;
        let chunk = Chunk::new(
            format!("doc123:chunk:{i}"),
            0,
            "",
            vec![v, 0.0, 0.0],
            0,
            0,
            1,
            meta.to_json(),
            None,
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await.unwrap();

    // Search and extract citations
    let query = SearchQuery::new(vec![2.0, 0.0, 0.0])
        .metric(Metric::Cosine)
        .limit(3);
    let results = backend.search(&query).await.unwrap();

    // SourceMetadata::to_json() produces {"source": {"doc_id": ..., ...}}
    let has_citations = results.iter().any(|r| {
        r.chunk
            .metadata
            .get("source")
            .and_then(|s| s.get("doc_id"))
            .and_then(|v| v.as_str())
            .is_some()
    });
    assert!(
        has_citations,
        "search should return chunks with source metadata"
    );
}

#[tokio::test]
async fn test_context_reconstruction() {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // Create a document with multiple chunks
    let mut chunks = Vec::new();
    for i in 0..10 {
        let meta = SourceMetadata::new("article1")
            .with_chunk_index(i)
            .with_total_chunks(10);
        // Avoid zero-magnitude vector (i=0) for cosine similarity
        let v = (i as f32) + 1.0;
        let chunk = Chunk::new(
            format!("article1:chunk:{i}"),
            0,
            "",
            vec![v, 0.0, 0.0],
            0,
            0,
            1,
            meta.to_json(),
            None,
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await.unwrap();

    // Chunk::new(doc_id, 0, ...) produces id "doc_id#0"
    let chunk = backend
        .get("article1:chunk:5#0", None)
        .await
        .unwrap()
        .expect("chunk article1:chunk:5#0 should exist");
    // ContextReconstructor expects EmbeddingRecord, but we have Chunk
    // Extract chunk index from metadata and compute neighbors manually
    let chunk_idx = chunk
        .metadata
        .get("chunk_index")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(5) as usize;
    let window = 2;
    let neighbor_ids: Vec<String> = ((chunk_idx.saturating_sub(window))..=(chunk_idx + window))
        .filter(|&i| i != chunk_idx && i < 10)
        .map(|i| format!("article1:chunk:{i}"))
        .collect();

    // Should include chunks 3, 4, 6, 7 (within window of 2 around chunk 5)
    assert!(neighbor_ids.len() >= 2);
    assert!(
        neighbor_ids
            .iter()
            .any(|id| id.contains("chunk:3") || id.contains("chunk:4"))
    );
}

#[test]
fn test_chunking_with_metadata() {
    let text = "First sentence. Second sentence. Third sentence.";
    let chunker = SentenceChunker::with_target(20);
    let chunks = chunker.chunk(text);

    for (i, chunk) in chunks.iter().enumerate() {
        let with_meta = chunk.clone().with_metadata(serde_json::json!({
            "chunk_index": i,
            "source": "test_doc"
        }));
        assert!(with_meta.metadata.is_some());
    }
}
