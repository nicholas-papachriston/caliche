//! Re-export of the shared HTTP SSRF policy.
//!
//! The implementation lives in [`crate::security::http_policy`] so gateway
//! adapters, the HTTP tool, and scheduled delivery can use it without the
//! `plugins` feature.

pub use crate::security::http_policy::*;
