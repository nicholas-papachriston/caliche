//! AST parsing and code graph module.
//!
//! Provides multi-language AST parsing via tree-sitter, graph structure exploration,
//! and integration with RAG for structure-aware code understanding.

pub mod types;

pub use types::{AstEdgeKind, AstNode, Language, NodeKind, Span};

#[cfg(feature = "ast")]
pub mod languages;
#[cfg(feature = "ast")]
pub mod parser;

#[cfg(feature = "ast")]
pub use languages::{LanguageRegistry, detect_language};
#[cfg(feature = "ast")]
pub use parser::AstParser;

#[cfg(all(feature = "ast", feature = "tools-graph"))]
pub mod graph;
#[cfg(all(feature = "ast", feature = "tools-graph"))]
pub use graph::CodeGraph;

#[cfg(feature = "ast")]
pub mod query;
#[cfg(feature = "ast")]
pub use query::AstQuery;

#[cfg(feature = "ast")]
pub mod analysis;
#[cfg(feature = "ast")]
pub mod chunking;
#[cfg(feature = "ast")]
pub use analysis::{
    CallGraph, ComplexityMetrics, ImportGraph, ScopeTree, Symbol, analyze_complexity,
    build_call_graph, build_import_graph, extract_scope_tree, extract_symbols,
};
#[cfg(feature = "ast")]
pub use chunking::{ChunkBoundary, chunk_at_ast_boundaries};
