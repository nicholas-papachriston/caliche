//! RepoMap: token-budgeted codebase overview.
//!
//! Scans workspace, ranks files by relevance, and produces a compact
//! file list that fits within a token budget. Used by the workspace tool's
//! codebase_context and the CLI code command.

mod builder;
mod ranker;
mod scanner;

pub use builder::build;
