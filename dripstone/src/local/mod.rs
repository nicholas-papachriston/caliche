//! Zero-config in-process store used by [`crate::Dripstone::open`].
//!
//! The hashing embedder and directory snapshot stay inside the `local`
//! feature. That feature is part of the default set and does not enable
//! remote backends, ONNX, or an HTTP client.

mod embedder;
mod store;

pub use embedder::{LOCAL_EMBED_DIMENSION, LOCAL_EMBEDDER_NAME, LocalEmbedder};
pub use store::OpenBackend;
