//! Tree-sitter based AST parser.

use crate::ast::languages::{LanguageRegistry, detect_language};
use crate::ast::types::{AstNode, Language, NodeKind, Span};
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::fs;
use tree_sitter::Parser;

/// Parser for converting source code to unified AST representation.
pub struct AstParser;

impl AstParser {
    /// Parse a file from disk into an AstNode tree.
    pub async fn parse_file(path: &Path) -> Result<AstNode> {
        let path_buf = path.to_path_buf();
        let source = fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read file: {}", path.display()))?;
        let language = detect_language(path);
        Self::parse_source(&source, language, &path_buf)
    }

    /// Parse source code string into an AstNode tree.
    pub fn parse_source(source: &str, language: Language, file_path: &PathBuf) -> Result<AstNode> {
        let ts_lang = LanguageRegistry::get_language(language)
            .with_context(|| format!("No grammar available for language: {}", language))?;

        let mut parser = Parser::new();
        parser
            .set_language(&ts_lang)
            .map_err(|e| anyhow::anyhow!("Failed to set parser language: {}", e))?;

        let tree = parser
            .parse(source, None)
            .with_context(|| "Parsing failed")?;

        let root = tree.root_node();
        let text = source.to_string();
        Self::ts_node_to_ast(&root, &text, language, file_path)
    }

    /// Convert a tree-sitter Node to AstNode recursively.
    fn ts_node_to_ast(
        node: &tree_sitter::Node,
        source: &str,
        language: Language,
        file_path: &PathBuf,
    ) -> Result<AstNode> {
        let range = node.range();
        let start_byte = range.start_byte;
        let end_byte = range.end_byte;
        let text = source.get(start_byte..end_byte).unwrap_or("").to_string();

        let span = Span::new(
            start_byte,
            end_byte,
            range.start_point.row,
            range.end_point.row,
            range.start_point.column,
            range.end_point.column,
        );

        let id = AstNode::build_id(file_path, start_byte, end_byte);
        let syntax_kind = node.kind().to_string();
        let kind = Self::syntax_kind_to_node_kind(&syntax_kind, language);
        let name = Self::extract_name(node, source);

        let mut children = Vec::new();
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i as u32)
                && let Ok(ast_child) = Self::ts_node_to_ast(&child, source, language, file_path)
            {
                children.push(ast_child);
            }
        }

        let mut ast_node = AstNode::new(
            id,
            kind,
            syntax_kind,
            language,
            file_path.clone(),
            span,
            text,
        )
        .with_children(children);
        if let Some(n) = name {
            ast_node.name = Some(n);
        }
        Ok(ast_node)
    }

    /// Map tree-sitter syntax kind to normalized NodeKind.
    fn syntax_kind_to_node_kind(syntax_kind: &str, language: Language) -> NodeKind {
        let kind_lower = syntax_kind.to_lowercase();
        if language == Language::Markdown {
            return markdown_node_kind(&kind_lower, syntax_kind);
        }
        definition_node_kind(&kind_lower)
            .or_else(|| structural_node_kind(&kind_lower, syntax_kind))
            .unwrap_or_else(|| NodeKind::Other(syntax_kind.to_string()))
    }

    /// Extract identifier name from a node (e.g., function name, class name).
    fn extract_name(node: &tree_sitter::Node, source: &str) -> Option<String> {
        for i in 0..node.child_count() {
            if let Some(child) = node.child(i as u32)
                && (child.kind() == "identifier" || child.kind() == "name")
            {
                let range = child.range();
                let name = source.get(range.start_byte..range.end_byte)?;
                return Some(name.to_string());
            }
        }
        None
    }
}

fn markdown_node_kind(kind_lower: &str, syntax_kind: &str) -> NodeKind {
    match kind_lower {
        "atx_heading" | "setext_heading" => NodeKind::Heading,
        "paragraph" => NodeKind::Paragraph,
        "list" => NodeKind::List,
        "list_item" => NodeKind::ListItem,
        "pipe_table" => NodeKind::Table,
        "pipe_table_row" | "pipe_table_header" => NodeKind::TableRow,
        "section" | "document" => NodeKind::Block,
        "block_quote" | "fenced_code_block" | "indented_code_block" => NodeKind::Block,
        _ => NodeKind::Other(syntax_kind.to_string()),
    }
}

fn definition_node_kind(kind_lower: &str) -> Option<NodeKind> {
    if kind_lower.contains("function") && kind_lower.contains("definition") {
        Some(NodeKind::Function)
    } else if kind_lower.contains("function") || kind_lower == "function_declaration" {
        Some(NodeKind::Function)
    } else if kind_lower.contains("method") {
        Some(NodeKind::Method)
    } else if kind_lower.contains("class") && !kind_lower.contains("method") {
        Some(NodeKind::Class)
    } else if kind_lower.contains("struct") {
        Some(NodeKind::Struct)
    } else if kind_lower.contains("enum") {
        Some(NodeKind::Enum)
    } else if kind_lower.contains("trait") {
        Some(NodeKind::Trait)
    } else if kind_lower.contains("interface") {
        Some(NodeKind::Interface)
    } else {
        None
    }
}

fn structural_node_kind(kind_lower: &str, syntax_kind: &str) -> Option<NodeKind> {
    if kind_lower.contains("import") || kind_lower == "use_declaration" {
        Some(NodeKind::Import)
    } else if kind_lower.contains("parameter") || kind_lower == "formal_parameters" {
        Some(NodeKind::Parameter)
    } else if kind_lower.contains("identifier") && syntax_kind == "identifier" {
        Some(NodeKind::Variable)
    } else if kind_lower.contains("comment") {
        Some(NodeKind::Comment)
    } else if kind_lower.contains("block") || kind_lower == "block" {
        Some(NodeKind::Block)
    } else if kind_lower.contains("module") || kind_lower == "source_file" {
        Some(NodeKind::Module)
    } else if kind_lower == "object" || kind_lower == "dictionary" {
        Some(NodeKind::Object)
    } else if kind_lower == "array" || kind_lower == "sequence" {
        Some(NodeKind::Array)
    } else if kind_lower == "pair" || kind_lower == "key_value" || kind_lower.contains("key") {
        Some(NodeKind::KeyValue)
    } else {
        None
    }
}
