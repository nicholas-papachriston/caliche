//! CSV document loader.

use super::document::{Document, DocumentLoader, LoaderError};
use async_trait::async_trait;
use std::collections::HashSet;
use std::path::PathBuf;

#[cfg(feature = "document-loaders")]
use csv::Reader;

/// Load documents from CSV files.
///
/// Each row in the CSV becomes a document, with column values as metadata.
#[derive(Debug, Clone)]
pub struct CsvLoader {
    path: PathBuf,
    content_column: Option<String>,
    metadata_columns: HashSet<String>,
}

impl CsvLoader {
    /// Create a new CSV loader.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            content_column: None,
            metadata_columns: HashSet::new(),
        }
    }

    /// Set which column contains the main content.
    ///
    /// If not set, all columns will be concatenated.
    #[must_use]
    pub fn with_content_column(mut self, column: impl Into<String>) -> Self {
        self.content_column = Some(column.into());
        self
    }

    /// Set which columns to include as metadata.
    ///
    /// If not set, all columns (except content) will be included.
    #[must_use]
    pub fn with_metadata_columns(mut self, columns: &[impl AsRef<str>]) -> Self {
        self.metadata_columns = columns.iter().map(|s| s.as_ref().to_string()).collect();
        self
    }
}

#[async_trait]
impl DocumentLoader for CsvLoader {
    #[cfg(feature = "document-loaders")]
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        let mut reader = Reader::from_path(&self.path)
            .map_err(|e| LoaderError::LoadFailed(format!("Failed to open CSV: {e}")))?;

        let headers = reader
            .headers()
            .map_err(|e| LoaderError::ParseError(format!("Failed to read headers: {e}")))?
            .clone();

        let mut documents = Vec::new();

        for result in reader.records() {
            let record =
                result.map_err(|e| LoaderError::ParseError(format!("Failed to parse row: {e}")))?;

            let mut content = String::new();
            let mut metadata = std::collections::HashMap::new();

            for (idx, field) in record.iter().enumerate() {
                if let Some(header) = headers.get(idx) {
                    if Some(header) == self.content_column.as_deref() {
                        content = field.to_string();
                    } else if self.metadata_columns.is_empty()
                        || self.metadata_columns.contains(header)
                    {
                        metadata.insert(header.to_string(), field.to_string());
                    }

                    // If no content column specified, concatenate all
                    if self.content_column.is_none() {
                        if !content.is_empty() {
                            content.push_str(", ");
                        }
                        content.push_str(field);
                    }
                }
            }

            documents.push(Document::new(content).with_metadata_map(metadata));
        }

        Ok(documents)
    }

    #[cfg(not(feature = "document-loaders"))]
    async fn load(&self) -> Result<Vec<Document>, LoaderError> {
        Err(LoaderError::LoadFailed(
            "CSV loader requires 'document-loaders' feature".to_owned(),
        ))
    }

    fn name(&self) -> &'static str {
        "csv"
    }
}
