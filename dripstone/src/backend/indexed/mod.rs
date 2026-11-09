//! ANN-accelerated backend decorator.
//!
//! Wraps any [`Backend`] with a [`VectorIndex`] so that `search()` queries the
//! approximate nearest-neighbour index for candidate IDs, then fetches full
//! [`Chunk`] objects from the inner backend. All other operations delegate
//! directly to the inner backend.

mod backend;
mod config;

pub use backend::{IndexedBackend, ReindexError};
pub use config::{IndexUpdateMode, IndexedBackendConfig};

#[cfg(all(test, feature = "hnsw"))]
mod tests;
