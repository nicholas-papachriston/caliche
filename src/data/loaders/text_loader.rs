//! Plain text loader.

use super::document::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::fs;

/// Load documents from plain text files.
///
/// Reads the file as raw bytes and converts to UTF-8.  Invalid byte sequences
/// are replaced with the Unicode replacement character (`\u{FFFD}`) via
/// [`String::from_utf8_lossy`] so the loader never returns a hard error for
/// encoding issues.
#[derive(Debug, Clone)]
pub struct TextLoader {
    path: PathBuf,
}

impl TextLoader {
    /// Create a new text loader for the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl DocumentLoader for TextLoader {
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let bytes = fs::read(&self.path).await?;
        // `from_utf8` takes ownership; on failure the error wraps the original bytes so
        // we can recover them via `as_bytes()` without a second allocation.
        let content = String::from_utf8(bytes)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned());
        Ok(vec![Document::new(content)])
    }

    fn name(&self) -> &'static str {
        "text"
    }
}
