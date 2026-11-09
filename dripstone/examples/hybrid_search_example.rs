//! Example demonstrating hybrid search with dense and sparse vectors.
//! cargo run --example hybrid_search_example --features memory-backend

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::memory::MemoryBackend;
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::search::{Bm25Scorer, SimpleTokenizer};
use dripstone::traits::Backend;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create in-memory backend
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // Sample documents
    let documents = vec![
        "Machine learning is a subset of artificial intelligence".to_string(),
        "Deep learning uses neural networks with multiple layers".to_string(),
        "Natural language processing helps computers understand human language".to_string(),
        "Computer vision enables machines to interpret visual information".to_string(),
    ];

    // Initialize BM25 scorer
    let tokenizer = SimpleTokenizer::new().with_english_stopwords();
    let mut bm25 = Bm25Scorer::new();
    bm25.fit(&documents, &tokenizer);

    outputln!(
        "Indexing {} documents with hybrid vectors...\n",
        documents.len()
    );

    // Index documents with both dense (simulated) and sparse vectors
    let mut chunks = Vec::new();
    for (i, doc) in documents.iter().enumerate() {
        // In production, you'd generate dense embeddings with a real model
        // For this example, we simulate with random-ish values
        let dense_vector: Vec<f32> = (0..384)
            .map(|j| ((i * 100 + j) as f32 / 1000.0).sin())
            .collect();

        // Generate sparse vector using BM25
        let _sparse_vector = bm25.transform(doc, &tokenizer);

        let chunk = Chunk::new(
            format!("doc:{i}"),
            0,
            doc,
            dense_vector,
            0,
            doc.len(),
            1,
            serde_json::json!({
                "text": doc,
                "category": if i % 2 == 0 { "AI" } else { "ML" }
            }),
            None,
        );
        // Note: Sparse vectors would be stored in metadata or a separate field
        // For now, we'll store it in metadata
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;

    outputln!("Indexed successfully\n");

    // Perform hybrid search
    let query = "understanding language with neural networks";
    outputln!("Query: \"{query}\"\n");

    // Generate query vectors
    let query_dense: Vec<f32> = (0..384).map(|j| (j as f32 / 1000.0).sin()).collect();
    let query_sparse = bm25.transform(query, &tokenizer);

    // Search with balanced weights (0.5 dense, 0.5 sparse)
    outputln!("Results with balanced weighting (0.5 dense, 0.5 sparse):");
    let query = SearchQuery {
        vector: query_dense.clone().into(),
        limit: 3,
        metric: Metric::Cosine,
        namespace: None,
        filter: None,
        hybrid_weight: Some(0.5),
        sparse_vector: Some(query_sparse.clone()),
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    for (i, result) in results.iter().enumerate() {
        outputln!(
            "  {}. [Score: {:.4}] {}",
            i + 1,
            result.score,
            result.chunk.metadata["text"]
        );
    }

    // Search favoring dense vectors
    outputln!("\nResults favoring dense vectors (0.8 dense, 0.2 sparse):");
    let query = SearchQuery {
        vector: query_dense.clone().into(),
        limit: 3,
        metric: Metric::Cosine,
        namespace: None,
        filter: None,
        hybrid_weight: Some(0.8),
        sparse_vector: Some(query_sparse.clone()),
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    for (i, result) in results.iter().enumerate() {
        outputln!(
            "  {}. [Score: {:.4}] {}",
            i + 1,
            result.score,
            result.chunk.metadata["text"]
        );
    }

    // Search with metadata filter
    outputln!("\nResults filtered by category='AI':");
    use dripstone::search::Filter;
    let filter = Filter::Equal {
        field: "category".to_string(),
        value: serde_json::json!("AI"),
    };
    let query = SearchQuery {
        vector: query_dense.into(),
        limit: 5,
        metric: Metric::Cosine,
        namespace: None,
        filter: Some(filter),
        hybrid_weight: Some(0.5),
        sparse_vector: Some(query_sparse),
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };
    let results = backend.search(&query).await?;

    for (i, result) in results.iter().enumerate() {
        outputln!(
            "  {}. [Score: {:.4}] {}",
            i + 1,
            result.score,
            result.chunk.metadata["text"]
        );
    }

    Ok(())
}
