//! Markdown document loader with frontmatter support.

use super::document::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::fs;

#[cfg(feature = "document-loaders")]
use pulldown_cmark::{Event, Parser, Tag, TagEnd};

/// Load documents from Markdown files.
///
/// Supports YAML frontmatter for metadata.
#[derive(Debug, Clone)]
pub struct MarkdownLoader {
    path: PathBuf,
    extract_frontmatter: bool,
}

impl MarkdownLoader {
    /// Create a new Markdown loader.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            extract_frontmatter: true,
        }
    }

    /// Set whether to extract frontmatter as metadata.
    #[must_use]
    pub const fn with_frontmatter(mut self, extract: bool) -> Self {
        self.extract_frontmatter = extract;
        self
    }

    #[cfg(feature = "document-loaders")]
    fn parse_frontmatter(
        &self,
        content: &str,
    ) -> (std::collections::HashMap<String, String>, String) {
        let mut metadata = std::collections::HashMap::new();
        let normalized = content.replace("\r\n", "\n");

        if !self.extract_frontmatter || !normalized.starts_with("---\n") {
            return (metadata, content.to_string());
        }

        // Find end of frontmatter
        if let Some(end_idx) = normalized[4..].find("\n---\n") {
            let frontmatter = &normalized[4..end_idx + 4];
            let body = &normalized[end_idx + 8..];

            // Parse simple key: value pairs
            for line in frontmatter.lines() {
                if let Some((key, value)) = line.split_once(':') {
                    metadata.insert(key.trim().to_string(), value.trim().to_string());
                }
            }

            return (metadata, body.to_string());
        }

        (metadata, content.to_string())
    }

    #[cfg(feature = "document-loaders")]
    fn markdown_to_text(markdown: &str) -> String {
        let parser = Parser::new(markdown);
        let mut text = String::new();

        for event in parser {
            match event {
                Event::Text(t) | Event::Code(t) => {
                    text.push_str(&t);
                }
                Event::SoftBreak | Event::HardBreak => {
                    text.push(' ');
                }
                Event::End(TagEnd::Paragraph | TagEnd::Heading(_)) => {
                    text.push_str("\n\n");
                }
                Event::Start(Tag::CodeBlock(_)) => {
                    text.push('\n');
                }
                _ => {}
            }
        }

        text.trim().to_string()
    }
}

#[async_trait]
impl DocumentLoader for MarkdownLoader {
    #[cfg(feature = "document-loaders")]
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let content = fs::read_to_string(&self.path).await?;

        let (metadata, body) = self.parse_frontmatter(&content);
        let text = Self::markdown_to_text(&body);

        // Add file path as metadata
        let mut meta = metadata;
        meta.insert("source".to_owned(), self.path.display().to_string());

        Ok(vec![Document::new(text).with_metadata_map(meta)])
    }

    #[cfg(not(feature = "document-loaders"))]
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        Err(LoaderError::LoadFailed(
            "Markdown loader requires 'document-loaders' feature".to_string(),
        ))
    }

    fn name(&self) -> &'static str {
        "markdown"
    }
}
