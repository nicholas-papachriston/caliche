//! Structural queries over parsed AST.

use crate::ast::types::{AstNode, Language, NodeKind};
use anyhow::Result;

/// Structural query operations on AST.
pub struct AstQuery;

impl AstQuery {
    /// Find all nodes matching a kind with optional name pattern.
    pub fn find_by_kind<'a>(
        ast: &'a AstNode,
        kind: &NodeKind,
        name_pattern: Option<&str>,
    ) -> Vec<&'a AstNode> {
        let mut results = Vec::new();
        Self::find_by_kind_rec(ast, kind, name_pattern, &mut results);
        results
    }

    fn find_by_kind_rec<'a>(
        node: &'a AstNode,
        kind: &NodeKind,
        name_pattern: Option<&str>,
        results: &mut Vec<&'a AstNode>,
    ) {
        if &node.kind == kind {
            let matches_name = match (name_pattern, &node.name) {
                (None, _) => true,
                (Some(pat), Some(name)) => name.contains(pat),
                (Some(_), None) => false,
            };
            if matches_name {
                results.push(node);
            }
        }
        for child in &node.children {
            Self::find_by_kind_rec(child, kind, name_pattern, results);
        }
    }

    /// Find all references to a symbol name in the AST.
    /// Searches for identifier nodes and name properties that match.
    pub fn find_references<'a>(ast: &'a AstNode, symbol_name: &str) -> Vec<&'a AstNode> {
        let mut results = Vec::new();
        Self::find_references_rec(ast, symbol_name, &mut results);
        results
    }

    fn find_references_rec<'a>(
        node: &'a AstNode,
        symbol_name: &str,
        results: &mut Vec<&'a AstNode>,
    ) {
        if let Some(ref name) = node.name
            && name == symbol_name
        {
            results.push(node);
        }
        if node.text.trim() == symbol_name && node.syntax_kind.to_lowercase().contains("identifier")
        {
            results.push(node);
        }
        for child in &node.children {
            Self::find_references_rec(child, symbol_name, results);
        }
    }

    /// Run a tree-sitter S-expression query against a parsed AST.
    /// Note: Full tree-sitter query support requires tree-sitter Query API.
    /// This provides a simplified pattern match by kind.
    pub fn ts_query(_ast: &AstNode, _pattern: &str, _language: Language) -> Result<Vec<AstNode>> {
        anyhow::bail!("Tree-sitter S-expression queries not yet implemented - use find_by_kind")
    }
}
