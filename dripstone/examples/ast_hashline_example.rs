//! Example demonstrating hashline tagging and pre-chunked ingestion in `Dripstone`.
//!
//! Shows:
//! - `PreChunk` and `ingest_chunks` for structure-aware chunking
//! - Hashline configuration and content-addressed tagging
//! - `format_content_with_hashlines` for result display
//!
//! Run with:
//! ```bash
//! cargo run -p dripstone --example ast_hashline_example --features="memory-backend,tokio"
//! ```

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

use dripstone::Dripstone;
use dripstone::backend::memory::MemoryBackend;
use dripstone::config::{ChunkStrategy, DripstoneConfig};
use dripstone::core::PreChunk;
use dripstone::hashline::{HashlineOptions, format_content_with_hashlines};
use dripstone::traits::{Backend, Embedder};
use std::sync::Arc;

/// Minimal embedder for demonstration (deterministic, no model required).
#[derive(Debug)]
struct DemoEmbedder {
    dim: usize,
}

impl DemoEmbedder {
    const fn new(dim: usize) -> Self {
        Self { dim }
    }

    fn embed_sync(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for (i, b) in text.bytes().enumerate() {
            v[i % self.dim] = (v[i % self.dim] + f32::from(b) / 256.0) % 1.0;
        }
        v
    }
}

#[async_trait::async_trait]
impl Embedder for DemoEmbedder {
    async fn embed(
        &self,
        texts: &[&str],
    ) -> Result<Vec<Vec<f32>>, dripstone::errors::DripstoneError> {
        Ok(texts.iter().map(|t| self.embed_sync(t)).collect())
    }

    async fn embed_query(
        &self,
        query: &str,
    ) -> Result<Vec<f32>, dripstone::errors::DripstoneError> {
        Ok(self.embed_sync(query))
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn name(&self) -> &'static str {
        "demo"
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    outputln!("=== Dripstone AST and Hashline Example ===\n");

    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let embedder = DemoEmbedder::new(128);

    let mut config = DripstoneConfig::default();
    config.chunking.strategy = ChunkStrategy::Sentence;
    config.hashline.enabled = true;
    config.hashline.line_level = true;
    config.hashline.format_results = true;

    let vf = Dripstone::with_config(backend, embedder, config);

    // Example 1: Hashline utilities
    outputln!("--- Hashline Utilities ---\n");

    let opts = HashlineOptions::default();
    let sample =
        "fn factorial(n: u32) -> u32 {\n    if n <= 1 { 1 } else { n * factorial(n - 1) }\n}";
    let hash = dripstone::hashline::compute_content_hash(sample, &opts);
    outputln!("Content hash (chunk): {hash}");

    let line_hashes = dripstone::hashline::compute_line_hashes(sample, &opts);
    outputln!("Line hashes:");
    for (line_num, h) in &line_hashes {
        outputln!("  Line {line_num}: {h}");
    }

    let formatted = format_content_with_hashlines(sample, 1, &opts);
    outputln!("\nFormatted with hashline tags:");
    outputln!("{formatted}");

    // Example 2: Pre-chunked ingestion (simulating AST boundaries)
    outputln!("\n--- Pre-Chunked Ingestion (AST-like boundaries) ---\n");

    let rust_code = r#"pub fn greet() {
    outputln!("Hello");
}

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

pub struct Point { x: f64, y: f64 }
"#;

    let pre_chunks = vec![
        PreChunk::new("pub fn greet() {\n    outputln!(\"Hello\");\n}", 0, 40),
        PreChunk::new("pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}", 42, 84),
        PreChunk::new("pub struct Point { x: f64, y: f64 }", 86, 119),
    ];

    outputln!(
        "Ingesting {} pre-computed chunks (function/struct boundaries)",
        pre_chunks.len()
    );
    let result = vf
        .ingest_chunks("rust_module", rust_code, pre_chunks)
        .namespace("demo")
        .await?;
    outputln!("Created {} chunks\n", result.chunks_created);

    // Example 3: Search and display with hashline formatting
    outputln!("--- Search Results ---\n");

    let results: Vec<dripstone::SearchResult> = vf
        .search("function that adds numbers")
        .namespace("demo")
        .limit(3)
        .await?;

    for (i, r) in results.iter().enumerate() {
        outputln!("{}. Score: {:.4}", i + 1, r.score);
        if let Some(hashline) = r.chunk.metadata.get("hashline") {
            if let Some(start_line) = hashline
                .get("start_line")
                .and_then(serde_json::value::Value::as_u64)
            {
                let opts = HashlineOptions::default();
                let formatted =
                    format_content_with_hashlines(&r.chunk.content, start_line as usize, &opts);
                outputln!("   {}", formatted.replace('\n', "\n   "));
            }
        } else {
            outputln!("   {}", r.chunk.content.replace('\n', "\n   "));
        }
        outputln!();
    }

    outputln!("Done.");
    Ok(())
}
