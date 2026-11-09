//! JavaScript bindings for Caliche Document Loaders
#![expect(
    clippy::needless_pass_by_value,
    reason = "NAPI must own values converted from JavaScript arguments at the ABI boundary"
)]

use caliche::loaders::{
    CsvLoader as RustCsvLoader, Document as RustDocument, DocumentLoader,
    HtmlLoader as RustHtmlLoader, JsonLoader as RustJsonLoader,
    MarkdownLoader as RustMarkdownLoader, TextSplitter as RustTextSplitter,
};
use napi_derive::napi;
use std::sync::Arc;

/// JavaScript wrapper for Document
#[napi]
pub struct Document {
    content: String,
    metadata: std::collections::HashMap<String, String>,
}

impl From<RustDocument> for Document {
    fn from(doc: RustDocument) -> Self {
        Self {
            content: doc.content,
            metadata: doc.metadata,
        }
    }
}

impl From<Document> for RustDocument {
    fn from(doc: Document) -> Self {
        let mut rust_doc = Self::new(doc.content);
        for (key, value) in doc.metadata {
            rust_doc = rust_doc.with_metadata(key, value);
        }
        rust_doc
    }
}

#[napi]
impl Document {
    /// Create a new Document
    #[napi(constructor, catch_unwind)]
    pub fn new(content: String) -> Self {
        Self {
            content,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata to a Document
    #[napi(catch_unwind)]
    pub fn with_metadata(&self, key: String, value: String) -> Self {
        let mut new_doc = Self {
            content: self.content.clone(),
            metadata: self.metadata.clone(),
        };
        new_doc.metadata.insert(key, value);
        new_doc
    }

    /// Get content
    #[napi(catch_unwind)]
    pub fn get_content(&self) -> String {
        self.content.clone()
    }

    /// Get metadata
    #[napi(catch_unwind)]
    pub fn get_metadata(&self) -> std::collections::HashMap<String, String> {
        self.metadata.clone()
    }
}

/// JavaScript wrapper for `TextSplitter`
#[napi]
pub struct TextSplitter {
    inner: RustTextSplitter,
}

#[napi]
impl TextSplitter {
    /// Create a new `TextSplitter`
    #[napi(constructor, catch_unwind)]
    pub fn new(chunk_size: u32, chunk_overlap: u32) -> Self {
        Self {
            inner: RustTextSplitter::new(chunk_size as usize, chunk_overlap as usize),
        }
    }

    /// Split text into chunks
    #[napi(catch_unwind)]
    pub fn split_text(&self, text: String) -> Vec<String> {
        self.inner.split_text(&text)
    }

    /// Split documents into chunks
    /// Takes an array of document objects with 'content' and 'metadata' fields
    #[napi(catch_unwind)]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub fn split_documents(
        &self,
        documents: Vec<serde_json::Value>,
    ) -> napi::Result<Vec<Document>> {
        let rust_docs: Vec<RustDocument> = documents
            .into_iter()
            .map(|doc_value| {
                let content = doc_value
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| napi::Error::from_reason("Document missing 'content' field"))?
                    .to_string();
                let mut rust_doc = RustDocument::new(content);
                if let Some(metadata) = doc_value.get("metadata").and_then(|v| v.as_object()) {
                    for (key, value) in metadata {
                        if let Some(value_str) = value.as_str() {
                            rust_doc = rust_doc.with_metadata(key.clone(), value_str.to_string());
                        }
                    }
                }
                Ok(rust_doc)
            })
            .collect::<napi::Result<Vec<_>>>()?;

        Ok(self
            .inner
            .split_documents(&rust_docs)
            .into_iter()
            .map(Into::into)
            .collect())
    }
}

/// JavaScript wrapper for `CsvLoader`
#[napi]
pub struct CsvLoader {
    inner: Arc<caliche::loaders::CsvLoader>,
}

#[napi]
impl CsvLoader {
    /// Create a new `CsvLoader`
    #[napi(constructor, catch_unwind)]
    pub fn new(path: String) -> Self {
        Self {
            inner: Arc::new(RustCsvLoader::new(path)),
        }
    }

    /// Set content column for CSV loader
    #[napi(catch_unwind)]
    pub fn with_content_column(&self, column: String) -> Self {
        Self {
            inner: Arc::new((*self.inner).clone().with_content_column(column)),
        }
    }

    /// Set metadata columns for CSV loader
    #[napi(catch_unwind)]
    pub fn with_metadata_columns(&self, columns: Vec<String>) -> Self {
        Self {
            inner: Arc::new((*self.inner).clone().with_metadata_columns(&columns)),
        }
    }

    /// Load documents from CSV
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn load(&self) -> napi::Result<Vec<Document>> {
        self.inner
            .load()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to load CSV: {e}")))
            .map(|docs| docs.into_iter().map(Into::into).collect())
    }

    /// Get CSV loader name
    #[napi(catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// JavaScript wrapper for `JsonLoader`
#[napi]
pub struct JsonLoader {
    inner: Arc<caliche::loaders::JsonLoader>,
}

#[napi]
impl JsonLoader {
    /// Create a new `JsonLoader`
    #[napi(constructor, catch_unwind)]
    pub fn new(path: String) -> Self {
        Self {
            inner: Arc::new(RustJsonLoader::new(path)),
        }
    }

    /// Set content key for JSON loader
    #[napi(catch_unwind)]
    pub fn with_content_key(&self, key: String) -> Self {
        Self {
            inner: Arc::new((*self.inner).clone().with_content_key(key)),
        }
    }

    /// Load documents from JSON
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn load(&self) -> napi::Result<Vec<Document>> {
        self.inner
            .load()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to load JSON: {e}")))
            .map(|docs| docs.into_iter().map(Into::into).collect())
    }

    /// Get JSON loader name
    #[napi(catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// JavaScript wrapper for `MarkdownLoader`
#[napi]
pub struct MarkdownLoader {
    inner: Arc<caliche::loaders::MarkdownLoader>,
}

#[napi]
impl MarkdownLoader {
    /// Create a new `MarkdownLoader`
    #[napi(constructor, catch_unwind)]
    pub fn new(path: String) -> Self {
        Self {
            inner: Arc::new(RustMarkdownLoader::new(path)),
        }
    }

    /// Load documents from Markdown
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn load(&self) -> napi::Result<Vec<Document>> {
        self.inner
            .load()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to load Markdown: {e}")))
            .map(|docs| docs.into_iter().map(Into::into).collect())
    }

    /// Get Markdown loader name
    #[napi(catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// JavaScript wrapper for `HtmlLoader`
#[napi]
pub struct HtmlLoader {
    inner: Arc<caliche::loaders::HtmlLoader>,
}

#[napi]
impl HtmlLoader {
    /// Create a new `HtmlLoader`
    #[napi(constructor, catch_unwind)]
    pub fn new(path: String) -> Self {
        Self {
            inner: Arc::new(RustHtmlLoader::new(path)),
        }
    }

    /// Load documents from HTML
    #[napi]
    ///
    /// # Errors
    ///
    /// Returns a NAPI error when validation, synchronization, serialization,
    /// or an underlying runtime operation fails.
    pub async fn load(&self) -> napi::Result<Vec<Document>> {
        self.inner
            .load()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Failed to load HTML: {e}")))
            .map(|docs| docs.into_iter().map(Into::into).collect())
    }

    /// Get HTML loader name
    #[napi(catch_unwind)]
    pub fn name(&self) -> String {
        self.inner.name().to_string()
    }
}
