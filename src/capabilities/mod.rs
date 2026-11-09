//! Capabilities module for feature availability detection.
//!
//! This module centralizes all feature flag logic, providing a single source
//! of truth for what features are available at compile time.

pub mod providers;
pub mod tools;
pub mod vector_backends;

/// Re-export all capability functions for convenience
pub use providers::*;
pub use tools::*;
pub use vector_backends::*;
