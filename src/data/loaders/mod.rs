//! Document loaders for various file formats.
//!
//! This module provides loaders to read and parse documents from different formats,
//! making it easy to ingest data for RAG and other AI workflows.
//!
//! # Examples
//!
//! CSV loading:
//!
//! ```rust,no_run
//! use caliche::loaders::{CsvLoader, DocumentLoader};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let loader = CsvLoader::new("data.csv")
//!     .with_content_column("text")
//!     .with_metadata_columns(&["author", "date"]);
//!
//! let documents = loader.load().await?;
//! # Ok(())
//! # }
//! ```
//!
//! JSON loading:
//!
//! ```rust,no_run
//! use caliche::loaders::{JsonLoader, DocumentLoader};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let loader = JsonLoader::new("data.json")
//!     .with_content_key("content");
//!
//! let documents = loader.load().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Markdown loading:
//!
//! ```rust,no_run
//! use caliche::loaders::{MarkdownLoader, DocumentLoader};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let loader = MarkdownLoader::new("document.md");
//! let documents = loader.load().await?;
//! # Ok(())
//! # }
//! ```
//!
//! Text splitting:
//!
//! ```rust
//! use caliche::loaders::{TextSplitter, Document};
//!
//! let splitter = TextSplitter::new(500, 50); // 500 chars, 50 overlap
//! let documents = vec![Document::new("Long text...")];
//! let chunks = splitter.split_documents(&documents);
//! ```

pub mod csv_loader;
pub mod document;
#[cfg(feature = "document-loaders-extended")]
pub mod docx_loader;
pub mod html_loader;
pub mod json_loader;
pub mod markdown_loader;
pub mod pdf_loader;
pub mod processor;
pub mod text_loader;
#[cfg(feature = "document-loaders-extended")]
pub mod xlsx_loader;

pub use csv_loader::CsvLoader;
pub use document::{Document, DocumentLoader, LoaderError, TextSplitter};
#[cfg(feature = "document-loaders-extended")]
pub use docx_loader::DocxLoader;
pub use html_loader::HtmlLoader;
pub use json_loader::JsonLoader;
pub use markdown_loader::MarkdownLoader;
pub use pdf_loader::PdfLoader;
pub use processor::{DocumentProcessor, ProcessedDocument};
pub use text_loader::TextLoader;
#[cfg(feature = "document-loaders-extended")]
pub use xlsx_loader::XlsxLoader;
