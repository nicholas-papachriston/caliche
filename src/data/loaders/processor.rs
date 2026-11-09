//! Unified document processor that dispatches by type.

use super::document::{Document, LoaderError};
use super::{
    CsvLoader, DocumentLoader, HtmlLoader, JsonLoader, MarkdownLoader, PdfLoader, TextLoader,
};
#[cfg(feature = "document-loaders-extended")]
use super::{DocxLoader, XlsxLoader};
use std::collections::HashMap;
use std::path::Path;
use tracing::warn;

/// The result of processing a document from any supported format.
#[derive(Debug, Clone)]
pub struct ProcessedDocument {
    /// Full text extracted from the document, with pages joined by `\n`.
    pub text: String,
    /// Key-value metadata extracted by the loader (title, author, source path, etc.).
    pub metadata: HashMap<String, String>,
    /// Lowercase file extension used to select the loader (e.g. `"pdf"`, `"csv"`).
    pub source_type: String,
}

/// Dispatches document loading based on file extension.
///
/// # Examples
///
/// ```rust,no_run
/// use caliche::loaders::DocumentProcessor;
/// use std::path::Path;
///
/// # async fn run() -> anyhow::Result<()> {
/// let processor = DocumentProcessor::default();
/// let doc = processor.process_file(Path::new("report.pdf")).await?;
/// println!("{}", doc.text);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct DocumentProcessor {
    max_file_size: usize,
}

impl Default for DocumentProcessor {
    fn default() -> Self {
        Self {
            max_file_size: 25 * 1024 * 1024,
        }
    }
}

impl DocumentProcessor {
    fn extension_from_media_type(media_type: &str) -> &str {
        match media_type {
            "application/pdf" => "pdf",
            "text/csv" => "csv",
            "text/html" => "html",
            "text/markdown" => "md",
            "application/json" => "json",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => "docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => "xlsx",
            "application/vnd.ms-excel" => "xls",
            _ => media_type.split('/').nth(1).unwrap_or("bin"),
        }
    }

    #[must_use]
    pub const fn with_max_file_size(mut self, max_file_size: usize) -> Self {
        self.max_file_size = max_file_size;
        self
    }

    /// # Errors
    ///
    /// Returns an error if the source cannot be read or parsed.
    pub async fn process_file(&self, path: &Path) -> Result<ProcessedDocument, LoaderError> {
        let metadata = tokio::fs::metadata(path).await?;
        if metadata.len() > self.max_file_size as u64 {
            return Err(LoaderError::InvalidFormat(format!(
                "File too large ({} bytes > {} bytes)",
                metadata.len(),
                self.max_file_size
            )));
        }

        let extension = path
            .extension()
            .and_then(std::ffi::OsStr::to_str)
            .unwrap_or_default()
            .to_ascii_lowercase();

        let mut loader_docs = load_by_extension(path, extension.as_str()).await?;
        stamp_source(&mut loader_docs, &path.display().to_string());
        let (merged_text, merged_metadata) = merge_loader_docs(loader_docs);

        Ok(ProcessedDocument {
            text: merged_text,
            metadata: merged_metadata,
            source_type: extension, // move directly — no intermediate clone needed
        })
    }

    /// # Errors
    ///
    /// Returns an error if the source cannot be read or parsed.
    pub async fn process_bytes(
        &self,
        data: &[u8],
        media_type: &str,
        filename: Option<&str>,
    ) -> Result<ProcessedDocument, LoaderError> {
        if data.len() > self.max_file_size {
            return Err(LoaderError::InvalidFormat(format!(
                "Input too large ({} bytes > {} bytes)",
                data.len(),
                self.max_file_size
            )));
        }

        let ext = filename
            .and_then(|f| {
                std::path::Path::new(f)
                    .extension()
                    .and_then(std::ffi::OsStr::to_str)
            })
            .unwrap_or_else(|| Self::extension_from_media_type(media_type));
        let temp_path = std::env::temp_dir().join(format!(
            "caliche-docproc-{}-{}.{}",
            uuid::Uuid::now_v7(),
            std::process::id(),
            ext
        ));
        tokio::fs::write(&temp_path, data).await?;
        let res = self.process_file(&temp_path).await;
        if let Err(e) = tokio::fs::remove_file(&temp_path).await {
            warn!(path = %temp_path.display(), error = %e, "failed to remove temp file after processing");
        }
        res
    }
}

async fn load_by_extension(path: &Path, extension: &str) -> Result<Vec<Document>, LoaderError> {
    match extension {
        "csv" => CsvLoader::new(path).load().await,
        "json" | "jsonl" => JsonLoader::new(path).load().await,
        "md" | "markdown" => MarkdownLoader::new(path).load().await,
        "html" | "htm" => HtmlLoader::new(path).load().await,
        "txt" | "log" | "rst" => TextLoader::new(path).load().await,
        "pdf" => PdfLoader::new(path).load().await,
        #[cfg(feature = "document-loaders-extended")]
        "docx" => DocxLoader::new(path).load().await,
        #[cfg(feature = "document-loaders-extended")]
        "xlsx" | "xls" | "ods" => XlsxLoader::new(path).load().await,
        _ => TextLoader::new(path).load().await,
    }
}

fn stamp_source(docs: &mut [Document], source: &str) {
    for doc in docs {
        doc.metadata
            .entry("source".to_owned())
            .or_insert_with(|| source.to_owned());
    }
}

fn merge_loader_docs(docs: Vec<Document>) -> (String, HashMap<String, String>) {
    let mut merged_text = String::new();
    let mut merged_metadata = HashMap::new();
    for doc in docs {
        if !merged_text.is_empty() {
            merged_text.push('\n');
        }
        merged_text.push_str(&doc.content);
        for (key, value) in doc.metadata {
            merged_metadata.entry(key).or_insert(value);
        }
    }
    (merged_text, merged_metadata)
}
