//! Document Loaders Example
//!
//! This example demonstrates document loaders for various file formats.
//!
//! Run with:
//! ```bash
//! cargo run --example document_loaders --features document-loaders
//! ```

#![expect(
    clippy::print_stdout,
    reason = "examples print their results for the reader"
)]

#[cfg(feature = "document-loaders")]
use caliche::loaders::{Document, DocumentLoader, JsonLoader, TextSplitter};
use tokio::fs;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("Document Loader Examples\n");

    // Create temporary test files
    let temp_dir = std::env::temp_dir();

    // 1. JSON Loader
    println!("1. JSON Loader:");
    let json_file = temp_dir.join("test_data.json");
    let json_content = r#"[
        {"content": "First document", "author": "Alice", "date": "2024-01-01"},
        {"content": "Second document", "author": "Bob", "date": "2024-01-02"}
    ]"#;
    fs::write(&json_file, json_content).await?;

    let loader = JsonLoader::new(&json_file).with_content_key("content");
    let documents = loader.load().await?;
    println!("   Loaded {} documents from JSON", documents.len());
    for doc in &documents {
        println!("     - Content: {}", doc.content);
        println!("       Metadata: {:?}", doc.metadata);
    }
    println!();

    // 2. JSONL Loader
    println!("2. JSONL Loader:");
    let jsonl_file = temp_dir.join("test_data.jsonl");
    let jsonl_content = r#"{"text": "Line 1", "id": "1"}
{"text": "Line 2", "id": "2"}
{"text": "Line 3", "id": "3"}"#;
    fs::write(&jsonl_file, jsonl_content).await?;

    let loader = JsonLoader::new(&jsonl_file)
        .with_jsonl(true)
        .with_content_key("text");
    let documents = loader.load().await?;
    println!("   Loaded {} documents from JSONL", documents.len());
    println!();

    // 3. Text Splitter
    println!("3. Text Splitter:");
    let long_text = "This is a very long document that needs to be split into smaller chunks for processing. \
                     Each chunk should be a manageable size while maintaining some overlap to preserve context. \
                     The splitter will automatically handle this for us.";

    let doc = Document::new(long_text).with_metadata("source", "example");

    let splitter = TextSplitter::new(50, 10); // 50 char chunks, 10 char overlap
    let chunks = splitter.split_documents(&[doc]);

    println!("   Split into {} chunks:", chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        println!(
            "     Chunk {}: \"{}...\"",
            i + 1,
            chunk.content.chars().take(30).collect::<String>()
        );
        println!("       Metadata: {:?}", chunk.metadata);
    }
    println!();

    // 4. Document creation and manipulation
    println!("4. Document Creation:");
    let doc = Document::new("Sample content")
        .with_metadata("title", "My Document")
        .with_metadata("author", "Caliche")
        .with_metadata("version", "1.0");

    println!("   Created document:");
    println!("     Content: {}", doc.content);
    println!("     Metadata: {:?}", doc.metadata);
    println!();

    // Cleanup
    let _ = fs::remove_file(json_file).await;
    let _ = fs::remove_file(jsonl_file).await;

    println!("\nNote: CSV, Markdown, and HTML loaders work similarly.");

    Ok(())
}
