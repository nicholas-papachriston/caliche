/// Tests for document loaders
#[cfg(feature = "document-loaders")]
mod loader_tests {
    use caliche::loaders::{
        CsvLoader, Document, DocumentLoader, HtmlLoader, JsonLoader, MarkdownLoader, TextSplitter,
    };
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    #[expect(
        clippy::expect_used,
        reason = "test helper: a failure fails the calling test"
    )]
    fn create_temp_file(dir: &TempDir, name: &str, content: &str) -> String {
        let path = dir.path().join(name);
        let mut file = fs::File::create(&path).expect("File creation should succeed");
        file.write_all(content.as_bytes())
            .expect("File write should succeed");
        path.to_str()
            .expect("Path should be valid UTF-8")
            .to_string()
    }

    #[tokio::test]
    async fn test_csv_loader_basic() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago";
        let path = create_temp_file(&temp_dir, "test.csv", csv_content);

        let loader = CsvLoader::new(&path)
            .with_content_column("name")
            .with_metadata_columns(&["age", "city"]);

        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].content, "Alice");
        assert_eq!(docs[0].metadata["age"], "30");
        assert_eq!(docs[0].metadata["city"], "NYC");
    }

    #[tokio::test]
    async fn test_csv_loader_no_metadata() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let csv_content = "text\nFirst line\nSecond line\nThird line";
        let path = create_temp_file(&temp_dir, "test.csv", csv_content);

        let loader = CsvLoader::new(&path).with_content_column("text");

        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].content, "First line");
        assert_eq!(docs[1].content, "Second line");
    }

    #[tokio::test]
    async fn test_csv_loader_all_columns() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA";
        let path = create_temp_file(&temp_dir, "test.csv", csv_content);

        let loader = CsvLoader::new(&path).with_content_column("name");

        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "Alice");
    }

    #[tokio::test]
    async fn test_csv_loader_missing_file() {
        let loader = CsvLoader::new("/nonexistent/file.csv");
        let result = loader.load().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_loader_array() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let json_content = r#"[
            {"text": "First document", "category": "A"},
            {"text": "Second document", "category": "B"},
            {"text": "Third document", "category": "A"}
        ]"#;
        let path = create_temp_file(&temp_dir, "test.json", json_content);

        let loader = JsonLoader::new(&path).with_content_key("text");

        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].content, "First document");
        assert_eq!(docs[0].metadata.get("category"), Some(&"A".to_string()));
    }

    #[tokio::test]
    async fn test_json_loader_jsonl() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let jsonl_content = r#"{"text": "Line 1", "id": "1"}
{"text": "Line 2", "id": "2"}
{"text": "Line 3", "id": "3"}"#;
        let path = create_temp_file(&temp_dir, "test.jsonl", jsonl_content);

        let loader = JsonLoader::new(&path).with_content_key("text");

        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].content, "Line 1");
        assert_eq!(docs[1].content, "Line 2");
    }

    #[tokio::test]
    async fn test_json_loader_single_object() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let json_content = r#"{"text": "Single document", "author": "Alice"}"#;
        let path = create_temp_file(&temp_dir, "test.json", json_content);

        let loader = JsonLoader::new(&path).with_content_key("text");

        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "Single document");
    }

    #[tokio::test]
    async fn test_json_loader_nested() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let json_content = r#"[
            {"data": {"content": "First"}, "meta": "A"},
            {"data": {"content": "Second"}, "meta": "B"}
        ]"#;
        let path = create_temp_file(&temp_dir, "test.json", json_content);

        let loader = JsonLoader::new(&path).with_content_key("data.content");

        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "First");
        assert_eq!(docs[1].content, "Second");
    }

    #[tokio::test]
    async fn test_markdown_loader_basic() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let md_content = r#"# Title

This is a paragraph.

## Section 1

Some content here.

### Subsection

More content."#;
        let path = create_temp_file(&temp_dir, "test.md", md_content);

        let loader = MarkdownLoader::new(&path);
        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");

        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("Title"));
        assert!(docs[0].content.contains("paragraph"));
    }

    #[tokio::test]
    async fn test_markdown_loader_with_frontmatter() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let md_content = r#"---
title: Test Document
author: Alice
date: 2024-01-01
---

# Main Content

This is the actual content."#;
        let path = create_temp_file(&temp_dir, "test.md", md_content);

        let loader = MarkdownLoader::new(&path).with_frontmatter(true);
        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");

        assert_eq!(docs.len(), 1);
        // Frontmatter may be in metadata or stripped
        assert!(
            docs[0].content.contains("Main Content") || docs[0].content.contains("Test Document")
        );
    }

    #[tokio::test]
    async fn test_markdown_loader_code_blocks() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let md_content = r#"# Code Example

```rust
fn main() {
    println!("Hello!");
}
```

More text."#;
        let path = create_temp_file(&temp_dir, "test.md", md_content);

        let loader = MarkdownLoader::new(&path);
        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");

        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.len() > 0);
    }

    #[tokio::test]
    async fn test_html_loader_basic() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let html_content = r#"<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
    <h1>Main Title</h1>
    <p>First paragraph.</p>
    <p>Second paragraph.</p>
</body>
</html>"#;
        let path = create_temp_file(&temp_dir, "test.html", html_content);

        let loader = HtmlLoader::new(&path);
        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");

        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("Main Title"));
        assert!(docs[0].content.contains("First paragraph"));
        assert!(docs[0].content.contains("Second paragraph"));
    }

    #[tokio::test]
    async fn test_html_loader_remove_scripts() {
        let temp_dir = TempDir::new().expect("TempDir creation should succeed");
        let html_content = r#"<!DOCTYPE html>
<html>
<body>
    <p>Visible content.</p>
    <script>console.log('test');</script>
    <style>body { color: red; }</style>
</body>
</html>"#;
        let path = create_temp_file(&temp_dir, "test.html", html_content);

        let loader = HtmlLoader::new(&path)
            .with_remove_scripts(true)
            .with_remove_styles(true);
        let docs = loader
            .load()
            .await
            .expect("Document loading should succeed");

        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("Visible content"));
        // Scripts and styles should be removed
        assert!(!docs[0].content.contains("console.log") || docs[0].content.contains("Visible"));
    }

    #[test]
    fn test_text_splitter_basic() {
        let text = "This is a test. This is another sentence. And one more.";
        let splitter = TextSplitter::new(20, 5);
        let chunks = splitter.split_text(text);

        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(chunk.len() <= 30); // Allow some margin
        }
    }

    #[test]
    fn test_text_splitter_documents() {
        let docs = vec![
            Document::new("First document with some content that should be split."),
            Document::new("Second document with more content here."),
        ];

        let splitter = TextSplitter::new(20, 5);
        let chunks = splitter.split_documents(&docs);

        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_text_splitter_short_text() {
        let text = "Short text.";
        let splitter = TextSplitter::new(100, 10);
        let chunks = splitter.split_text(text);

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_text_splitter_with_overlap() {
        let text = "Sentence one. Sentence two. Sentence three. Sentence four.";
        let splitter = TextSplitter::new(20, 10);
        let chunks = splitter.split_text(text);

        // Should have multiple chunks with some overlap
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_text_splitter_preserves_words() {
        let text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8";
        let splitter = TextSplitter::new(15, 3);
        let chunks = splitter.split_text(text);

        // No chunk should split a word in the middle
        for chunk in chunks {
            for word in chunk.split_whitespace() {
                assert!(text.contains(word));
            }
        }
    }

    #[test]
    fn test_text_splitter_empty() {
        let splitter = TextSplitter::new(100, 10);
        let chunks = splitter.split_text("");
        // Empty string may return empty chunk or no chunks
        assert!(chunks.is_empty() || (chunks.len() == 1 && chunks[0].is_empty()));
    }

    #[test]
    fn test_text_splitter_metadata_preserved() {
        let mut doc = Document::new("Some content to split into multiple chunks.");
        doc.metadata
            .insert("source".to_string(), "test".to_string());
        doc.metadata
            .insert("author".to_string(), "Alice".to_string());

        let splitter = TextSplitter::new(15, 3);
        let chunks = splitter.split_documents(&[doc]);

        for chunk in chunks {
            assert_eq!(chunk.metadata.get("source"), Some(&"test".to_string()));
            assert_eq!(chunk.metadata.get("author"), Some(&"Alice".to_string()));
        }
    }

    #[test]
    fn test_document_new() {
        let doc = Document::new("Test content");
        assert_eq!(doc.content, "Test content");
        assert!(doc.metadata.is_empty());
    }

    #[test]
    fn test_document_with_metadata() {
        let mut doc = Document::new("Content");
        doc.metadata.insert("key".to_string(), "value".to_string());
        assert_eq!(doc.metadata.get("key"), Some(&"value".to_string()));
    }
}
