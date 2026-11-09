//! Shared runtime and device selection for inference and embeddings.
//!
//! Provides a unified configuration and resolution policy so that both
//! inference and embedding pipelines can target MLX, CUDA, Metal, or CPU
//! in an extensible way.

mod capability;
mod config;

#[cfg(any(
    feature = "embedding-ort",
    feature = "embedding-candle",
    feature = "embedding-mlx",
    feature = "inference-mlx",
    feature = "model-download"
))]
pub use capability::*;
pub use config::{ResolvedRuntime, RuntimeConfig};
