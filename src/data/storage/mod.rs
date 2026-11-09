//! Storage module - object storage and vector storage

#[cfg(feature = "s3")]
pub mod object;

#[cfg(feature = "vector-memory")]
pub mod vector;
