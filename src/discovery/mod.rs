//! Discovery of available embedding and LLM providers.
//!
//! Used for auto-detecting the best embedding provider on startup
//! when workspace embedding is enabled.

pub mod auto;
pub mod embedding;
pub mod local;
