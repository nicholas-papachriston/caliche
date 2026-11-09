//! Kernel contracts shared across core and orchestration layers.
//!
//! Types in this module are intentionally domain-level and must not depend on
//! orchestration-specific or adapter-specific implementations.

pub mod turn;

pub use turn::{Handoff, TurnResult};

/// Temporary shared-state bridge until kernel owns this type.
pub type SharedState = crate::orchestration::state::SharedState;
