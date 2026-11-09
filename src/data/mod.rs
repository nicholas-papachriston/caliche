//! Data management module - loaders, storage, and persistence.
//!
//! This module provides data handling capabilities:
//!
//! - **Document Loaders**: Load documents from various formats (CSV, JSON, Markdown, HTML)
//!   - Requires `document-loaders` feature
//!   - See [`loaders`] module for available loaders
//!
//! - **Storage**: Object storage backends (S3, Cloudflare R2, filesystem)
//!   - Requires `s3` feature for cloud storage
//!   - See [`storage`] module for storage backends
//!
//! - **Persistence**: Conversation and workflow state persistence
//!   - See [`persistence`] module for persistence backends
//!
//! # Examples
//!
//! ## Loading Documents
//!
//! ```rust,no_run
//! # #[cfg(feature = "document-loaders")]
//! use caliche::loaders::{JsonLoader, DocumentLoader};
//!
//! # #[cfg(feature = "document-loaders")]
//! # async fn example() -> anyhow::Result<()> {
//! let loader = JsonLoader::new("data.json");
//! let documents = loader.load().await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Object Storage
//!
//! ```rust,no_run
//! # #[cfg(feature = "s3")]
//! use caliche::storage::{ObjectStoreConfig, create_object_store};
//!
//! # #[cfg(feature = "s3")]
//! # async fn example() -> anyhow::Result<()> {
//! let config = ObjectStoreConfig::aws_s3(None, None, None, None);
//! let store = create_object_store(config).await?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "document-loaders")]
pub mod loaders;

pub mod storage;

pub mod persistence;
