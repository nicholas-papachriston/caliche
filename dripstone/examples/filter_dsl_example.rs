//! Example demonstrating advanced metadata filtering DSL.

macro_rules! outputln {
    ($($arg:tt)*) => {{
        let mut output = std::io::stdout().lock();
        let _ = std::io::Write::write_fmt(&mut output, format_args!($($arg)*));
        let _ = std::io::Write::write_all(&mut output, b"\n");
    }};
}

use dripstone::backend::memory::MemoryBackend;
use dripstone::core::{Chunk, Metric, SearchQuery};
use dripstone::search::{Filter, FilterBuilder};
use dripstone::traits::Backend;
use serde_json::json;
use std::sync::Arc;

#[tokio::main]
#[expect(
    clippy::expect_used,
    reason = "example code panics on malformed inline fixtures"
)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Advanced Metadata Filtering DSL Example ===\n");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());

    // Create sample product records
    let products = [
        ("laptop", "Electronics", 1200, true),
        ("phone", "Electronics", 800, true),
        ("book", "Books", 20, true),
        ("tablet", "Electronics", 600, false),
        ("notebook", "Books", 15, true),
        ("headphones", "Electronics", 150, true),
        ("monitor", "Electronics", 300, false),
        ("novel", "Books", 25, true),
    ];

    outputln!("Indexing {} products...", products.len());
    let mut chunks = Vec::new();
    for (i, (name, category, price, in_stock)) in products.iter().enumerate() {
        let vector: Vec<f32> = (0..128).map(|j| (i + j) as f32 / 1000.0).collect();
        let chunk = Chunk::new(
            format!("product:{i}"),
            0,
            name.to_string(),
            vector,
            0,
            name.len(),
            1,
            json!({
                "name": name,
                "category": category,
                "price": price,
                "in_stock": in_stock
            }),
            None,
        );
        chunks.push(chunk);
    }
    backend.upsert(&chunks).await?;
    outputln!("Indexed successfully\n");

    let query_vector: Vec<f32> = (0..128).map(|j| j as f32 / 1000.0).collect();

    // Example 1: Simple equality filter
    outputln!("--- Example 1: Simple Equality Filter ---");
    outputln!("Find all Electronics products:\n");

    let filter1 = Filter::Equal {
        field: "category".to_string(),
        value: json!("Electronics"),
    };

    let query = SearchQuery {
        vector: query_vector.clone().into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: None,
        filter: Some(filter1),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    for result in &results {
        outputln!(
            "  - {} (${})",
            result.chunk.metadata["name"],
            result.chunk.metadata["price"]
        );
    }

    // Example 2: Complex AND filter
    outputln!("\n--- Example 2: Complex AND Filter ---");
    outputln!("Find Electronics products under $1000:\n");

    let filter2 = FilterBuilder::new()
        .equal("category", json!("Electronics"))
        .and(Filter::LessThan {
            field: "price".to_string(),
            value: json!(1000),
        })
        .build()
        .expect("filter builder should succeed");

    let query = SearchQuery {
        vector: query_vector.clone().into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: None,
        filter: Some(filter2),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    for result in &results {
        outputln!(
            "  - {} (${})",
            result.chunk.metadata["name"],
            result.chunk.metadata["price"]
        );
    }

    // Example 3: OR filter
    outputln!("\n--- Example 3: OR Filter ---");
    outputln!("Find products in Electronics OR Books category:\n");

    let filter3 = FilterBuilder::new()
        .in_values("category", vec![json!("Electronics"), json!("Books")])
        .build()
        .expect("filter builder should succeed");

    let query = SearchQuery {
        vector: query_vector.clone().into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: None,
        filter: Some(filter3),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    for result in &results {
        outputln!(
            "  - {} (${}, {})",
            result.chunk.metadata["name"],
            result.chunk.metadata["price"],
            result.chunk.metadata["category"]
        );
    }

    // Example 4: Complex query with multiple conditions
    outputln!("\n--- Example 4: Complex Query ---");
    outputln!("Find in-stock Electronics products under $500:\n");

    let filter4 = FilterBuilder::new()
        .equal("category", json!("Electronics"))
        .and(Filter::LessThan {
            field: "price".to_string(),
            value: json!(500),
        })
        .and(Filter::Equal {
            field: "in_stock".to_string(),
            value: json!(true),
        })
        .build()
        .expect("filter builder should succeed");

    let query = SearchQuery {
        vector: query_vector.clone().into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: None,
        filter: Some(filter4),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    for result in &results {
        outputln!(
            "  - {} (${}, in stock: {})",
            result.chunk.metadata["name"],
            result.chunk.metadata["price"],
            result.chunk.metadata["in_stock"]
        );
    }

    // Example 5: NOT filter
    outputln!("\n--- Example 5: NOT Filter ---");
    outputln!("Find products NOT in Electronics category:\n");

    let filter5 = Filter::Not {
        filter: Box::new(Filter::Equal {
            field: "category".to_string(),
            value: json!("Electronics"),
        }),
    };

    let query = SearchQuery {
        vector: query_vector.clone().into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: None,
        filter: Some(filter5),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    for result in &results {
        outputln!(
            "  - {} (${}, {})",
            result.chunk.metadata["name"],
            result.chunk.metadata["price"],
            result.chunk.metadata["category"]
        );
    }

    // Example 6: Range queries
    outputln!("\n--- Example 6: Range Queries ---");
    outputln!("Find products between $100 and $500:\n");

    let filter6 = FilterBuilder::new()
        .greater_than_or_equal("price", json!(100))
        .and(Filter::LessThanOrEqual {
            field: "price".to_string(),
            value: json!(500),
        })
        .build()
        .expect("filter builder should succeed");

    let query = SearchQuery {
        vector: query_vector.into(),
        limit: 10,
        metric: Metric::Cosine,
        namespace: None,
        filter: Some(filter6),
        hybrid_weight: None,
        sparse_vector: None,
        include_content: true,
        include_vectors: false,
        query_multi_vectors: None,
        maxsim: None,
    };

    let results = backend.search(&query).await?;

    for result in &results {
        outputln!(
            "  - {} (${})",
            result.chunk.metadata["name"],
            result.chunk.metadata["price"]
        );
    }

    Ok(())
}
