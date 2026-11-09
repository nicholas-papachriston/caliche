//! Core AST types for multi-language code representation.
//!
//! Provides a unified AST structure that can be converted to Dripstone graph types
//! for structural exploration and analysis.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Byte and line position in source code.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Span {
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_row: usize,
    pub end_row: usize,
    pub start_column: usize,
    pub end_column: usize,
}

impl Span {
    pub fn new(
        start_byte: usize,
        end_byte: usize,
        start_row: usize,
        end_row: usize,
        start_column: usize,
        end_column: usize,
    ) -> Self {
        Self {
            start_byte,
            end_byte,
            start_row,
            end_row,
            start_column,
            end_column,
        }
    }
}

/// Supported programming languages for AST parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    C,
    Cpp,
    Ruby,
    Toml,
    Json,
    Yaml,
    Html,
    Csv,
    Markdown,
    Unknown,
}

impl Language {
    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::JavaScript => "javascript",
            Language::TypeScript => "typescript",
            Language::Go => "go",
            Language::Java => "java",
            Language::C => "c",
            Language::Cpp => "cpp",
            Language::Ruby => "ruby",
            Language::Toml => "toml",
            Language::Json => "json",
            Language::Yaml => "yaml",
            Language::Html => "html",
            Language::Csv => "csv",
            Language::Markdown => "markdown",
            Language::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Normalized node kinds across languages.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Module,
    Function,
    Method,
    Class,
    Struct,
    Enum,
    Trait,
    Interface,
    Import,
    Variable,
    Constant,
    TypeAlias,
    Parameter,
    Block,
    Expression,
    Statement,
    Comment,
    Object,
    Array,
    KeyValue,
    // Markdown structure
    Heading,
    Paragraph,
    List,
    ListItem,
    Table,
    TableRow,
    // CSV/delimited structure
    Record,
    Field,
    Other(String),
}

impl NodeKind {
    pub fn as_str(&self) -> &str {
        if let Some(name) = self.declaration_name() {
            return name;
        }
        self.structure_name()
    }

    fn declaration_name(&self) -> Option<&str> {
        Some(match self {
            NodeKind::Module => "module",
            NodeKind::Function => "function",
            NodeKind::Method => "method",
            NodeKind::Class => "class",
            NodeKind::Struct => "struct",
            NodeKind::Enum => "enum",
            NodeKind::Trait => "trait",
            NodeKind::Interface => "interface",
            NodeKind::Import => "import",
            NodeKind::Variable => "variable",
            NodeKind::Constant => "constant",
            NodeKind::TypeAlias => "type_alias",
            NodeKind::Parameter => "parameter",
            NodeKind::Block => "block",
            NodeKind::Expression => "expression",
            NodeKind::Statement => "statement",
            _ => return None,
        })
    }

    fn structure_name(&self) -> &str {
        match self {
            NodeKind::Comment => "comment",
            NodeKind::Object => "object",
            NodeKind::Array => "array",
            NodeKind::KeyValue => "key_value",
            NodeKind::Heading => "heading",
            NodeKind::Paragraph => "paragraph",
            NodeKind::List => "list",
            NodeKind::ListItem => "list_item",
            NodeKind::Table => "table",
            NodeKind::TableRow => "table_row",
            NodeKind::Record => "record",
            NodeKind::Field => "field",
            NodeKind::Other(text) => text.as_str(),
            // Declaration kinds are named by `declaration_name`.
            _ => "module",
        }
    }
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// AST relationship types for graph edges.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AstEdgeKind {
    Contains,
    Calls,
    Imports,
    TypeReference,
    Implements,
    Inherits,
    DefinedIn,
    ReferencesSymbol,
}

impl AstEdgeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            AstEdgeKind::Contains => "CONTAINS",
            AstEdgeKind::Calls => "CALLS",
            AstEdgeKind::Imports => "IMPORTS",
            AstEdgeKind::TypeReference => "TYPE_REF",
            AstEdgeKind::Implements => "IMPLEMENTS",
            AstEdgeKind::Inherits => "INHERITS",
            AstEdgeKind::DefinedIn => "DEFINED_IN",
            AstEdgeKind::ReferencesSymbol => "REFERENCES",
        }
    }
}

/// A node in the parsed AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstNode {
    pub id: String,
    pub kind: NodeKind,
    pub syntax_kind: String,
    pub name: Option<String>,
    pub language: Language,
    pub file_path: PathBuf,
    pub span: Span,
    pub text: String,
    pub children: Vec<AstNode>,
    pub properties: HashMap<String, serde_json::Value>,
}

impl AstNode {
    pub fn new(
        id: String,
        kind: NodeKind,
        syntax_kind: String,
        language: Language,
        file_path: PathBuf,
        span: Span,
        text: String,
    ) -> Self {
        Self {
            id,
            kind,
            syntax_kind,
            name: None,
            language,
            file_path,
            span,
            text,
            children: Vec::new(),
            properties: HashMap::new(),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_children(mut self, children: Vec<AstNode>) -> Self {
        self.children = children;
        self
    }

    pub fn with_property(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.properties.insert(key.into(), value);
        self
    }

    /// Build a node ID from file path and span.
    pub fn build_id(file_path: &std::path::Path, start_byte: usize, end_byte: usize) -> String {
        format!("{}:{}:{}", file_path.display(), start_byte, end_byte)
    }
}
