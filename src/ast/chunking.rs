//! AST-based chunk boundaries for structure-aware RAG chunking.
//!
//! Produces chunk boundaries at function/class/object/array boundaries,
//! enabling semantic chunking for code and structured data.

use crate::ast::query::AstQuery;
use crate::ast::types::{Language, NodeKind};
use std::path::Path;

/// A chunk boundary with text and byte offsets in the source document.
#[derive(Debug, Clone)]
pub struct ChunkBoundary {
    pub text: String,
    pub start_offset: usize,
    pub end_offset: usize,
}

/// Compute chunk boundaries from AST for structure-aware RAG chunking.
///
/// For code: chunks at Function, Method, Class, Struct, Enum, Trait, Interface.
/// For JSON/YAML: chunks at top-level Object, Array, KeyValue.
/// Returns empty if parsing fails or language is unsupported.
pub fn chunk_at_ast_boundaries(source: &str, path: &Path) -> Vec<ChunkBoundary> {
    let language = crate::ast::detect_language(path);
    let Some(_) = crate::ast::LanguageRegistry::get_language(language) else {
        return vec![];
    };

    let path_buf = path.to_path_buf();
    let Ok(ast) = crate::ast::AstParser::parse_source(source, language, &path_buf) else {
        return vec![];
    };

    let chunk_kinds = chunk_kinds_for_language(language);
    let mut boundaries = Vec::new();

    for kind in chunk_kinds {
        for node in AstQuery::find_by_kind(&ast, &kind, None) {
            if node.text.is_empty() {
                continue;
            }
            boundaries.push(ChunkBoundary {
                text: node.text.clone(),
                start_offset: node.span.start_byte,
                end_offset: node.span.end_byte,
            });
        }
    }

    boundaries.sort_by_key(|b| b.start_offset);

    boundaries
}

fn chunk_kinds_for_language(language: Language) -> Vec<NodeKind> {
    match language {
        Language::Rust
        | Language::Python
        | Language::JavaScript
        | Language::TypeScript
        | Language::Go
        | Language::Java
        | Language::C
        | Language::Cpp
        | Language::Ruby => vec![
            NodeKind::Function,
            NodeKind::Method,
            NodeKind::Class,
            NodeKind::Struct,
            NodeKind::Enum,
            NodeKind::Trait,
            NodeKind::Interface,
        ],
        Language::Json | Language::Yaml => {
            vec![NodeKind::Object, NodeKind::Array, NodeKind::KeyValue]
        }
        Language::Html => vec![NodeKind::Object, NodeKind::Array],
        Language::Markdown => vec![
            NodeKind::Heading,
            NodeKind::Paragraph,
            NodeKind::List,
            NodeKind::ListItem,
            NodeKind::Table,
            NodeKind::TableRow,
            NodeKind::Block,
        ],
        Language::Csv => vec![NodeKind::Record, NodeKind::Field],
        _ => vec![],
    }
}
