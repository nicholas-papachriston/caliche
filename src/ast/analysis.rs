//! Higher-level analysis built on the parser and AST.

use crate::ast::query::AstQuery;
use crate::ast::types::{AstNode, NodeKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Extracted symbol with signature and location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: NodeKind,
    pub file_path: std::path::PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub signature: Option<String>,
    pub visibility: Option<String>,
}

/// Call graph: caller -> list of callees.
pub type CallGraph = HashMap<String, Vec<String>>;

/// Import graph: file -> list of imported modules/files.
pub type ImportGraph = HashMap<String, Vec<String>>;

/// Variable scope tree (simplified - maps scope id to bindings).
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ScopeTree {
    pub scopes: HashMap<String, Vec<String>>,
}

/// Code complexity metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: usize,
    pub nesting_depth: usize,
    pub line_count: usize,
    pub symbol_count: usize,
}

/// Extract all defined symbols from an AST.
pub fn extract_symbols(ast: &AstNode) -> Vec<Symbol> {
    let kinds = [
        NodeKind::Function,
        NodeKind::Method,
        NodeKind::Class,
        NodeKind::Struct,
        NodeKind::Enum,
        NodeKind::Trait,
        NodeKind::Interface,
        NodeKind::Variable,
        NodeKind::Constant,
        NodeKind::TypeAlias,
        NodeKind::Object,
        NodeKind::Array,
        NodeKind::KeyValue,
    ];
    let mut symbols = Vec::new();
    for kind in kinds {
        for node in AstQuery::find_by_kind(ast, &kind, None) {
            let name = node
                .name
                .clone()
                .unwrap_or_else(|| node.text.lines().next().unwrap_or("").trim().to_string());
            if !name.is_empty() && !name.starts_with('(') {
                symbols.push(Symbol {
                    name,
                    kind: node.kind.clone(),
                    file_path: node.file_path.clone(),
                    line_start: node.span.start_row,
                    line_end: node.span.end_row,
                    signature: Some(node.text.lines().take(3).fold(
                        String::new(),
                        |mut s, line| {
                            if !s.is_empty() {
                                s.push('\n');
                            }
                            s.push_str(line);
                            s
                        },
                    )),
                    visibility: None,
                });
            }
        }
    }
    symbols
}

/// Build call graph from AST (simplified - maps functions to their potential callees).
pub fn build_call_graph(ast: &AstNode) -> CallGraph {
    let mut graph = CallGraph::new();
    let symbols: std::collections::HashSet<_> =
        extract_symbols(ast).into_iter().map(|s| s.name).collect();
    for func in AstQuery::find_by_kind(ast, &NodeKind::Function, None)
        .into_iter()
        .chain(AstQuery::find_by_kind(ast, &NodeKind::Method, None))
    {
        let caller = func.name.as_ref().cloned().unwrap_or_default();
        if caller.is_empty() {
            continue;
        }
        let mut callees: Vec<String> = Vec::new();
        collect_calls_in_node(func, &symbols, &mut callees);
        if !callees.is_empty() {
            graph.insert(caller, callees);
        }
    }
    graph
}

fn collect_calls_in_node(
    node: &AstNode,
    symbols: &std::collections::HashSet<String>,
    callees: &mut Vec<String>,
) {
    if node.syntax_kind.to_lowercase().contains("call")
        && node.name.as_ref().is_some_and(|n| symbols.contains(n))
        && let Some(name) = &node.name
    {
        callees.push(name.clone());
    }
    for child in &node.children {
        collect_calls_in_node(child, symbols, callees);
    }
}

/// Build import/dependency graph from AST.
pub fn build_import_graph(ast: &AstNode) -> ImportGraph {
    let mut graph = ImportGraph::new();
    let file_key = ast.file_path.display().to_string();
    let mut imports = Vec::new();
    for node in AstQuery::find_by_kind(ast, &NodeKind::Import, None) {
        let import_path = node.text.trim().to_string();
        if !import_path.is_empty() && import_path.len() < 500 {
            imports.push(import_path);
        }
    }
    if !imports.is_empty() {
        graph.insert(file_key, imports);
    }
    graph
}

/// Placeholder scope-tree extraction. Returns an empty tree until scope analysis lands.
pub fn extract_scope_tree(_ast: &AstNode) -> ScopeTree {
    ScopeTree::default()
}

/// Analyze code complexity metrics.
pub fn analyze_complexity(ast: &AstNode) -> ComplexityMetrics {
    let symbol_count = extract_symbols(ast).len();
    let line_count = ast.span.end_row.saturating_sub(ast.span.start_row) + 1;
    let nesting_depth = max_nesting_depth(ast, 0);
    let cyclomatic_complexity = estimate_cyclomatic_complexity(ast);
    ComplexityMetrics {
        cyclomatic_complexity,
        nesting_depth,
        line_count,
        symbol_count,
    }
}

fn max_nesting_depth(ast: &AstNode, current: usize) -> usize {
    let mut max = current;
    for child in &ast.children {
        let child_depth = max_nesting_depth(child, current + 1);
        max = max.max(child_depth);
    }
    max
}

fn estimate_cyclomatic_complexity(ast: &AstNode) -> usize {
    let mut complexity = 1;
    let text_lower = ast.text.to_lowercase();
    for keyword in [
        "if", "else", "for", "while", "match", "case", "&&", "||", "?",
    ] {
        complexity += text_lower.matches(keyword).count();
    }
    for child in &ast.children {
        complexity += estimate_cyclomatic_complexity(child);
    }
    complexity
}
