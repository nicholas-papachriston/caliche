//! AST-aware RAG with hashline tagging example.
//!
//! Demonstrates:
//! - AST parsing for Rust, JSON, and other supported formats
//! - Structure-aware chunking at function/class/object boundaries
//! - Hashline tagging for content-addressed result formatting
//! - RAG indexing and querying with AstAware chunk strategy
//!
//! Run with:
//! ```bash
//! cargo run --example ast_rag --features tools-rag,tools-ast,openai
//! ```
//! Requires OPENAI_API_KEY in environment.

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

use anyhow::{Context, Result};
use caliche::ast::{AstParser, chunk_at_ast_boundaries, extract_symbols};
use caliche::llm::OpenAIProvider;
use caliche::rag::{RagHashlineOptions, RagService};
use dripstone::backend::memory::MemoryBackend;
use std::path::Path;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();

    println!("=== AST-Aware RAG with Hashline Example ===\n");

    example_ast_parsing()?;
    example_ast_chunking()?;
    example_rag_with_ast_and_hashline().await?;

    Ok(())
}

/// Demonstrate AST parsing and symbol extraction.
fn example_ast_parsing() -> Result<()> {
    println!("--- AST Parsing ---\n");

    let json_str =
        r#"{"name": "caliche", "version": "0.1", "features": ["ast", "semantic_search"]}"#;
    let json_path = Path::new("package.json");
    let ast = AstParser::parse_source(
        json_str,
        caliche::ast::detect_language(json_path),
        &json_path.to_path_buf(),
    )
    .context("Parse JSON")?;

    let symbols = extract_symbols(&ast);
    println!("JSON symbols extracted: {}", symbols.len());
    for s in symbols.iter().take(5) {
        println!("  - {} ({:?})", s.name, s.kind);
    }

    let py_code = r#"
def factorial(n: int) -> int:
    return 1 if n <= 1 else n * factorial(n - 1)

class Counter:
    def __init__(self): self.value = 0
"#;
    let py_path = Path::new("example.py");
    if let Ok(ast) = AstParser::parse_source(
        py_code,
        caliche::ast::detect_language(py_path),
        &py_path.to_path_buf(),
    ) {
        let symbols = extract_symbols(&ast);
        println!("\nPython symbols extracted: {}", symbols.len());
        for s in &symbols {
            println!(
                "  - {} ({:?}) at line {}-{}",
                s.name, s.kind, s.line_start, s.line_end
            );
        }
    }

    println!();
    Ok(())
}

/// Demonstrate AST-based chunk boundary extraction.
fn example_ast_chunking() -> Result<()> {
    println!("--- AST Chunk Boundaries ---\n");

    let py_code = r#"
def greet(): print("Hi")
def add(a: int, b: int) -> int: return a + b
class Point: x: float; y: float
"#;
    let path = Path::new("module.py");
    let boundaries = chunk_at_ast_boundaries(py_code, path);
    println!("Python: {} chunks at AST boundaries", boundaries.len());
    for (i, b) in boundaries.iter().enumerate() {
        let preview: String = b.text.chars().take(50).collect();
        println!(
            "  {}. {}... (bytes {}-{})",
            i + 1,
            preview,
            b.start_offset,
            b.end_offset
        );
    }

    let json_str = r#"{
  "dependencies": {"serde": "1.0", "tokio": "1.0"},
  "dev-dependencies": {"criterion": "0.5"}
}"#;
    let json_path = Path::new("Cargo.json");
    let boundaries = chunk_at_ast_boundaries(json_str, json_path);
    println!("\nJSON: {} chunks at AST boundaries", boundaries.len());
    for (i, b) in boundaries.iter().enumerate() {
        let preview: String = b.text.chars().take(40).collect();
        println!("  {}. {}...", i + 1, preview);
    }

    let md_str = r#"# Introduction

This is a paragraph with **bold** text.

## Features
- Item one
- Item two

| Col A | Col B |
|-------|-------|
| a1    | b1    |
"#;
    let md_path = Path::new("README.md");
    let md_boundaries = chunk_at_ast_boundaries(md_str, md_path);
    println!(
        "\nMarkdown: {} chunks at AST boundaries",
        md_boundaries.len()
    );
    for (i, b) in md_boundaries.iter().enumerate() {
        let preview: String = b.text.chars().take(50).collect();
        println!(
            "  {}. {}... (bytes {}-{})",
            i + 1,
            preview,
            b.start_offset,
            b.end_offset
        );
    }

    println!();
    Ok(())
}

/// Demonstrate RAG with AST-aware chunking and hashline tagging.
async fn example_rag_with_ast_and_hashline() -> Result<()> {
    println!("--- RAG with AST + Hashline ---\n");

    let api_key = std::env::var("OPENAI_API_KEY").context("OPENAI_API_KEY required")?;
    let provider = Arc::new(OpenAIProvider::try_new(
        api_key,
        "text-embedding-3-small".to_string(),
    )?);

    let options = RagHashlineOptions::for_workspace(true, false)
        .chunking_strategy(dripstone::config::ChunkStrategy::AstAware);

    let backend = Arc::new(MemoryBackend::new()) as Arc<dyn dripstone::Backend>;
    let service = RagService::new_with_backend_and_options(provider, backend, false, options);

    let py_module = r#"
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

class User:
    def __init__(self, id: int, name: str): self.id = id; self.name = name
"#;

    let path = Path::new("src/lib.py");
    let (chunks, _) = service
        .index_document_with_path(py_module, "py_module", Some(path))
        .await?;
    println!("Indexed {} chunks (AST boundaries)", chunks);

    let (results, _top_sim, _usage): (
        Vec<String>,
        Option<f32>,
        Option<caliche::prelude::TokenUsage>,
    ) = service
        .query("function that concatenates a greeting", 3, 0.2)
        .await?;

    println!("\nQuery: 'function that returns a greeting string'");
    for (i, r) in results.iter().enumerate() {
        println!("{}. {}", i + 1, r);
    }

    println!("\nDone.");
    Ok(())
}
