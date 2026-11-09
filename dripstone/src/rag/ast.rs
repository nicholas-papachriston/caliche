//! Tree-sitter chunking for [`crate::config::ChunkStrategy::AstAware`].
//!
//! Ingest calls [`AstChunker`] when that strategy is selected. A parsed
//! function or method becomes one chunk and carries symbol and line metadata.
//! Unknown languages, a missing `ast` feature, and files with no function node
//! return [`AstChunking::Unavailable`] so the caller can fall back to sentence
//! chunking.

use serde_json::{Map, Value};

#[cfg(feature = "ast")]
use std::collections::HashMap;

/// JSON key for the programming language on a code chunk.
pub const META_LANGUAGE: &str = "language";
/// JSON key for a repository-relative path.
pub const META_PATH: &str = "path";
/// JSON key some callers use instead of [`META_PATH`].
pub const META_FILE: &str = "file";
/// JSON key for the function or method name.
pub const META_SYMBOL: &str = "symbol";
/// JSON key for the 1-based first line.
pub const META_START_LINE: &str = "start_line";
/// JSON key for the 1-based last line, inclusive.
pub const META_END_LINE: &str = "end_line";
/// JSON key for the chunk source, used by the code corpus `meta.jsonl`.
pub const META_TEXT: &str = "text";
/// JSON key for the dataset row id before AST splitting.
pub const META_SOURCE_ID: &str = "source_id";
/// JSON key for a dataset title, when the source has one.
pub const META_TITLE: &str = "title";

/// Language tags [`AstChunker`] can parse when the `ast` feature is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceLanguage {
    /// Python (`function_definition`).
    Python,
    /// Rust (`function_item`).
    Rust,
    /// JavaScript (`function_declaration`, methods).
    JavaScript,
    /// Go (`function_declaration`, `method_declaration`).
    Go,
    /// Java (`method_declaration`, `constructor_declaration`).
    Java,
}

impl SourceLanguage {
    /// Parse a language tag or file extension.
    #[must_use]
    pub fn parse(token: &str) -> Option<Self> {
        match token.trim().to_ascii_lowercase().as_str() {
            "python" | "py" => Some(Self::Python),
            "rust" | "rs" => Some(Self::Rust),
            "javascript" | "js" | "jsx" => Some(Self::JavaScript),
            "go" | "golang" => Some(Self::Go),
            "java" => Some(Self::Java),
            _ => None,
        }
    }

    /// Extension of the last path segment.
    #[must_use]
    pub fn from_path(path: &str) -> Option<Self> {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
        let (_, extension) = name.rsplit_once('.')?;
        Self::parse(extension)
    }

    /// Stable language tag stored on chunk metadata.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Python => "python",
            Self::Rust => "rust",
            Self::JavaScript => "javascript",
            Self::Go => "go",
            Self::Java => "java",
        }
    }

    #[cfg(feature = "ast")]
    const fn node_kinds(self) -> &'static [&'static str] {
        match self {
            Self::Python => &["function_definition"],
            Self::Rust => &["function_item"],
            Self::JavaScript => &[
                "function_declaration",
                "generator_function_declaration",
                "method_definition",
            ],
            Self::Go => &["function_declaration", "method_declaration"],
            Self::Java => &["method_declaration", "constructor_declaration"],
        }
    }
}

/// One function or method sliced out of a source file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstChunk {
    /// Source text of the node, including decorators when present.
    pub text: String,
    /// Start byte in the original source.
    pub start_offset: usize,
    /// End byte in the original source.
    pub end_offset: usize,
    /// Language tag from [`SourceLanguage::as_str`].
    pub language: Option<String>,
    /// Repository-relative path, when the caller passed one.
    pub path: Option<String>,
    /// Function, method, or constructor name.
    pub symbol: Option<String>,
    /// First line, 1-based.
    pub start_line: Option<u32>,
    /// Last line, 1-based and inclusive.
    pub end_line: Option<u32>,
}

impl AstChunk {
    /// Metadata object merged onto an ingested chunk.
    #[must_use]
    pub fn metadata(&self) -> Value {
        let mut object = Map::new();
        insert_string(&mut object, META_LANGUAGE, self.language.as_deref());
        insert_string(&mut object, META_PATH, self.path.as_deref());
        insert_string(&mut object, META_SYMBOL, self.symbol.as_deref());
        if let Some(line) = self.start_line {
            object.insert(META_START_LINE.to_string(), Value::from(line));
        }
        if let Some(line) = self.end_line {
            object.insert(META_END_LINE.to_string(), Value::from(line));
        }
        Value::Object(object)
    }
}

/// Result of [`AstChunker::chunk`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstChunking {
    /// At least one function or method node.
    Parsed(Vec<AstChunk>),
    /// No grammar, no language, a parse with no function nodes, or the `ast` feature is off.
    Unavailable,
}

/// Reuses one tree-sitter parser per language across calls.
#[derive(Default)]
pub struct AstChunker {
    #[cfg(feature = "ast")]
    parsers: HashMap<SourceLanguage, tree_sitter::Parser>,
}

impl std::fmt::Debug for AstChunker {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("AstChunker")
    }
}

impl AstChunker {
    /// Empty parser cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Split `source` on function and method boundaries.
    ///
    /// `language` is a tag such as `python` or `rs`. `path` is used when
    /// `language` is absent. Nested functions stay inside the outer function.
    pub fn chunk(
        &mut self,
        source: &str,
        language: Option<&str>,
        path: Option<&str>,
    ) -> AstChunking {
        let Some(language) = resolve_language(language, path) else {
            return AstChunking::Unavailable;
        };
        self.chunk_language(source, language, path)
    }

    fn chunk_language(
        &mut self,
        source: &str,
        language: SourceLanguage,
        path: Option<&str>,
    ) -> AstChunking {
        #[cfg(feature = "ast")]
        {
            chunk_with_parser(self, source, language, path)
        }
        #[cfg(not(feature = "ast"))]
        {
            let _ = (self, source, language, path);
            AstChunking::Unavailable
        }
    }
}

/// One-shot [`AstChunker::chunk`].
#[must_use]
pub fn chunk_code(source: &str, language: Option<&str>, path: Option<&str>) -> AstChunking {
    AstChunker::new().chunk(source, language, path)
}

fn resolve_language(language: Option<&str>, path: Option<&str>) -> Option<SourceLanguage> {
    language
        .and_then(SourceLanguage::parse)
        .or_else(|| path.and_then(SourceLanguage::from_path))
}

fn insert_string(object: &mut Map<String, Value>, key: &str, value: Option<&str>) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        object.insert(key.to_string(), Value::String(value.to_string()));
    }
}

#[cfg(feature = "ast")]
fn chunk_with_parser(
    chunker: &mut AstChunker,
    source: &str,
    language: SourceLanguage,
    path: Option<&str>,
) -> AstChunking {
    let parser = chunker
        .parsers
        .entry(language)
        .or_insert_with(tree_sitter::Parser::new);
    let grammar = grammar(language);
    if parser.set_language(&grammar).is_err() {
        return AstChunking::Unavailable;
    }
    let Some(tree) = parser.parse(source, None) else {
        return AstChunking::Unavailable;
    };
    let mut chunks = Vec::new();
    walk(
        tree.root_node(),
        language.node_kinds(),
        false,
        source,
        language,
        path,
        &mut chunks,
    );
    if chunks.is_empty() {
        AstChunking::Unavailable
    } else {
        AstChunking::Parsed(chunks)
    }
}

#[cfg(feature = "ast")]
fn grammar(language: SourceLanguage) -> tree_sitter::Language {
    match language {
        SourceLanguage::Python => tree_sitter_python::LANGUAGE.into(),
        SourceLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
        SourceLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        SourceLanguage::Go => tree_sitter_go::LANGUAGE.into(),
        SourceLanguage::Java => tree_sitter_java::LANGUAGE.into(),
    }
}

#[cfg(feature = "ast")]
fn walk(
    node: tree_sitter::Node<'_>,
    kinds: &[&str],
    ancestor_hit: bool,
    source: &str,
    language: SourceLanguage,
    path: Option<&str>,
    out: &mut Vec<AstChunk>,
) {
    let is_match = kinds.iter().any(|kind| *kind == node.kind());
    if is_match && !ancestor_hit {
        if let Some(chunk) = chunk_from_node(node, source, language, path) {
            out.push(chunk);
        }
        return;
    }
    let mut cursor = node.walk();
    if !cursor.goto_first_child() {
        return;
    }
    loop {
        walk(
            cursor.node(),
            kinds,
            ancestor_hit || is_match,
            source,
            language,
            path,
            out,
        );
        if !cursor.goto_next_sibling() {
            break;
        }
    }
}

#[cfg(feature = "ast")]
fn chunk_from_node(
    node: tree_sitter::Node<'_>,
    source: &str,
    language: SourceLanguage,
    path: Option<&str>,
) -> Option<AstChunk> {
    let symbol = symbol_name(node, source);
    let spanned = match node.parent() {
        Some(parent) if parent.kind() == "decorated_definition" => parent,
        _ => node,
    };
    let start_offset = spanned.start_byte();
    let end_offset = spanned.end_byte();
    let text = source.get(start_offset..end_offset)?.trim();
    if text.is_empty() {
        return None;
    }
    let (start_line, end_line) = line_bounds(spanned);
    Some(AstChunk {
        text: text.to_string(),
        start_offset,
        end_offset,
        language: Some(language.as_str().to_string()),
        path: path.map(str::to_string),
        symbol,
        start_line: Some(start_line),
        end_line: Some(end_line),
    })
}

#[cfg(feature = "ast")]
fn symbol_name(node: tree_sitter::Node<'_>, source: &str) -> Option<String> {
    let name = node.child_by_field_name("name")?;
    let text = source.get(name.start_byte()..name.end_byte())?.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

#[cfg(feature = "ast")]
fn line_bounds(node: tree_sitter::Node<'_>) -> (u32, u32) {
    let start_row = node.start_position().row;
    let mut end_row = node.end_position().row;
    if node.end_position().column == 0 && end_row > start_row {
        end_row -= 1;
    }
    let start = u32::try_from(start_row.saturating_add(1)).unwrap_or(u32::MAX);
    let end = u32::try_from(end_row.saturating_add(1)).unwrap_or(u32::MAX);
    (start, end.max(start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_language_is_unavailable() {
        let outcome = chunk_code("def alpha():\n    return 1\n", Some("ruby"), None);
        assert!(matches!(outcome, AstChunking::Unavailable));
    }

    #[test]
    fn path_extension_selects_python_when_ast_is_enabled() {
        let outcome = chunk_code("def alpha():\n    return 1\n", None, Some("src/demo.py"));
        assert_language_outcome(outcome);
    }

    #[cfg(feature = "ast")]
    #[test]
    fn python_keeps_outer_functions_and_names_them() {
        let source = "def alpha(x):\n    def inner(y):\n        return y\n    return inner(x)\n\ndef beta(y):\n    return y\n";
        let AstChunking::Parsed(chunks) = chunk_code(source, Some("python"), None) else {
            panic!("python parse produced no chunks");
        };
        let symbols: Vec<_> = chunks.iter().map(|chunk| chunk.symbol.as_deref()).collect();
        assert_eq!(symbols, vec![Some("alpha"), Some("beta")]);
        assert!(
            chunks
                .iter()
                .all(|chunk| chunk.language.as_deref() == Some("python"))
        );
        assert_eq!(chunks[0].start_line, Some(1));
        assert!(chunks[0].end_line.is_some_and(|line| line >= 1));
        assert!(chunks[0].metadata().get(META_SYMBOL).is_some());
    }

    #[cfg(feature = "ast")]
    #[test]
    fn supported_languages_emit_a_symbol() {
        let samples = [
            ("python", "def alpha():\n    return 1\n", "alpha"),
            ("rust", "fn alpha() -> i32 {\n    42\n}\n", "alpha"),
            (
                "javascript",
                "function alpha() {\n  return 1;\n}\n",
                "alpha",
            ),
            (
                "go",
                "package main\nfunc alpha() int {\n    return 1\n}\n",
                "alpha",
            ),
            (
                "java",
                "class T {\n  int alpha() {\n    return 1;\n  }\n}\n",
                "alpha",
            ),
        ];
        for (language, source, symbol) in samples {
            match chunk_code(source, Some(language), None) {
                AstChunking::Parsed(chunks) => {
                    assert!(
                        chunks
                            .iter()
                            .any(|chunk| chunk.symbol.as_deref() == Some(symbol)),
                        "{language} chunks: {chunks:?}"
                    );
                }
                AstChunking::Unavailable => panic!("{language} did not parse"),
            }
        }
    }

    fn assert_language_outcome(outcome: AstChunking) {
        #[cfg(feature = "ast")]
        {
            let AstChunking::Parsed(chunks) = outcome else {
                panic!("expected python chunks from a .py path");
            };
            assert_eq!(chunks[0].symbol.as_deref(), Some("alpha"));
        }
        #[cfg(not(feature = "ast"))]
        {
            assert!(matches!(outcome, AstChunking::Unavailable));
        }
    }
}
