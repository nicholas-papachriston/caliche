//! PDF document loader.

use super::document::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::path::PathBuf;

#[derive(Debug, Clone)]
#[cfg_attr(
    not(feature = "document-loaders-extended"),
    expect(dead_code, reason = "only the extended loaders read the PDF path")
)]
pub struct PdfLoader {
    path: PathBuf,
}

impl PdfLoader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl DocumentLoader for PdfLoader {
    #[cfg(feature = "document-loaders-extended")]
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let bytes = tokio::fs::read(&self.path).await?;
        let text = pdf_extract::extract_text_from_mem(&bytes)
            .map_err(|e| LoaderError::ParseError(format!("PDF extraction failed: {}", e)))?;
        Ok(vec![Document::new(text)])
    }

    #[cfg(not(feature = "document-loaders-extended"))]
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        Err(LoaderError::InvalidFormat(
            "PDF extraction requires the `document-loaders-extended` feature".to_string(),
        ))
    }

    fn name(&self) -> &'static str {
        "pdf"
    }
}
