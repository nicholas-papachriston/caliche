//! Example demonstrating a complete RAG workflow with chunking and citations.
macro_rules! outputln {
    () => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
    ($($arg:tt)+) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::{
    Chunk, MemoryBackend, Metric, SearchQuery, SearchResult,
    rag::chunking::{ChunkStrategy, SentenceChunker},
    rag::citations::SourceMetadata,
    traits::Backend,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // Sample document
    let document = "\
Machine learning is a method of data analysis that automates analytical model building. \
It is a branch of artificial intelligence based on the idea that systems can learn from data. \
Neural networks are computing systems inspired by biological neural networks. \
They are the foundation of deep learning algorithms. \
Natural language processing enables computers to understand and generate human language.";

    // Chunk the document
    let chunker = SentenceChunker::with_target(100);
    let chunks = chunker.chunk(document);

    outputln!("Document chunked into {} pieces\n", chunks.len());

    // Index chunks with source metadata
    for (i, chunk) in chunks.iter().enumerate() {
        // Create source metadata for citation tracking
        let source_meta = SourceMetadata::new("ml_intro_doc")
            .with_title("Introduction to Machine Learning")
            .with_chunk_index(i)
            .with_total_chunks(chunks.len())
            .with_source_url("https://example.com/ml-intro");

        // Simulate embedding generation (in production, use a real model)
        let embedding: Vec<f32> = (0..384)
            .map(|j| ((i * 50 + j) as f32 / 500.0).sin())
            .collect();

        let chunk = Chunk::new(
            format!("ml_intro_doc:chunk:{i}"),
            0,
            &chunk.content,
            embedding,
            0,
            chunk.content.len(),
            1,
            source_meta.to_json(),
            Some("knowledge-base".to_string()),
        );

        let content_preview = chunk.content.chars().take(60).collect::<String>();
        backend.upsert(&[chunk]).await?;

        outputln!("Indexed chunk {i}: {content_preview}");
    }

    outputln!("\n--- RAG Query ---\n");

    // Simulate a user query
    let query = "How do neural networks relate to deep learning?";
    outputln!("Query: \"{query}\"\n");

    // Generate query embedding (simulated)
    let query_embedding: Vec<f32> = (0..384).map(|j| (j as f32 / 500.0).cos()).collect();

    // Search for relevant chunks
    let query = SearchQuery::new(query_embedding)
        .metric(Metric::Cosine)
        .limit(3)
        .namespace("knowledge-base");
    let results = backend.search(&query).await?;

    outputln!("Top {} relevant chunks:\n", results.len());

    // Extract source metadata from chunks
    for (i, result) in results.iter().enumerate() {
        outputln!("{}. [Score: {:.4}]", i + 1, result.score);

        // Extract source metadata from chunk metadata
        if let Some(source) = result.chunk.metadata.get("source") {
            if let Some(doc_title) = source.get("doc_title").and_then(|v| v.as_str()) {
                outputln!("   Source: {doc_title}");
            }
            if let (Some(chunk_idx), Some(total)) = (
                source
                    .get("chunk_index")
                    .and_then(serde_json::value::Value::as_u64),
                source
                    .get("total_chunks")
                    .and_then(serde_json::value::Value::as_u64),
            ) {
                outputln!("   Chunk: {}/{}", chunk_idx + 1, total);
            }
        }

        // Get the actual text from chunk
        let content_preview = if result.chunk.content.len() > 60 {
            format!(
                "{}...",
                &result.chunk.content[..result.chunk.content.floor_char_boundary(60)]
            )
        } else {
            result.chunk.content.clone()
        };
        outputln!("   Content: {content_preview}");
        outputln!();
    }

    // Group by document
    outputln!("\n--- Citations Grouped by Document ---\n");
    let mut grouped: std::collections::HashMap<String, Vec<&SearchResult>> =
        std::collections::HashMap::new();
    for result in &results {
        if let Some(source) = result.chunk.metadata.get("source")
            && let Some(doc_id) = source.get("doc_id").and_then(|v| v.as_str())
        {
            grouped.entry(doc_id.to_string()).or_default().push(result);
        }
    }

    for (doc_id, doc_results) in grouped {
        outputln!("Document: {doc_id}");
        outputln!("  {} chunks found", doc_results.len());
        for result in doc_results {
            if let Some(source) = result.chunk.metadata.get("source")
                && let Some(chunk_idx) = source
                    .get("chunk_index")
                    .and_then(serde_json::value::Value::as_u64)
            {
                outputln!("    - Chunk {} (score: {:.4})", chunk_idx + 1, result.score);
            }
        }
    }

    Ok(())
}
