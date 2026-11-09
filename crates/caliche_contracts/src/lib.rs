//! Shared serde wire contracts for Caliche clients and servers.
//!
//! Prefer `core` / `alloc` imports over `std` when an equivalent exists so this
//! crate stays easy to embed in constrained hosts.

#![warn(clippy::std_instead_of_core, clippy::std_instead_of_alloc)]

#[cfg(feature = "acp")]
pub mod acp;
pub mod chat;
pub mod child_env;
pub mod context_ref;
pub mod credential_env;
pub mod mcp;
pub mod metrics;
pub mod models;
pub mod permission;
pub mod rpc;
pub mod sessions;
pub mod settings;
pub mod slash;
pub mod slash_intent;
pub mod stream;
pub mod text;
pub mod tools;
