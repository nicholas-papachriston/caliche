//! AST parsing and code graph tool for structural code understanding.
//!
//! Provides parse, symbols, query, `call_graph`, `import_graph`, `build_code_graph`,
//! `structural_search`, and complexity operations.

use crate::agent::tools::common::{params, path};
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::ast::CodeGraph;
use crate::ast::analysis::{
    analyze_complexity, build_call_graph, build_import_graph, extract_symbols,
};
use crate::ast::parser::AstParser;
use crate::ast::query::AstQuery;
use crate::ast::types::NodeKind;
use anyhow::{Result, bail};
use async_trait::async_trait;
use dripstone::graph::InMemoryGraphBackend;
use serde_json::json;
use std::sync::Arc;

pub struct AstTool {
    graph_backend: Arc<dyn dripstone::graph::GraphBackend + Send + Sync>,
}

impl std::fmt::Debug for AstTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AstTool").finish()
    }
}

impl AstTool {
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph_backend: Arc::new(InMemoryGraphBackend::new()),
        }
    }

    pub fn with_graph_backend(
        backend: Arc<dyn dripstone::graph::GraphBackend + Send + Sync>,
    ) -> Self {
        Self {
            graph_backend: backend,
        }
    }
}

impl Default for AstTool {
    fn default() -> Self {
        Self::new()
    }
}

const NAME: &str = "ast";
const DESCRIPTION: &str = "Parse and analyze code structure (requires a known file path). Use fs op=list or fs op=grep first to discover files. Operations: 'parse' for AST summary, 'symbols' for extracted symbols, \
         'query' for structural search by kind/name, 'call_graph' for function call relationships, \
         'import_graph' for import dependencies, 'build_code_graph' to store AST as graph for exploration, \
         'structural_search' to find code by pattern, 'complexity' for code complexity metrics.";

impl ToolMeta for AstTool {
    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn description_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(DESCRIPTION)
    }

    fn name_as_cow(&self) -> beef::lean::Cow<'static, str> {
        beef::lean::Cow::borrowed(NAME)
    }

    fn name(&self) -> &str {
        NAME
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["parse", "symbols", "query", "call_graph", "import_graph", "build_code_graph", "structural_search", "complexity"],
                    "description": "Operation: parse, symbols, query, call_graph, import_graph, build_code_graph, structural_search, complexity"
                },
                "path": {
                    "type": "string",
                    "description": "File path to parse (required for parse, symbols, call_graph, import_graph, complexity)"
                },
                "kind": {
                    "type": "string",
                    "description": "Node kind filter for query: function, class, struct, import, etc."
                },
                "name_pattern": {
                    "type": "string",
                    "description": "Optional name pattern to filter query results"
                },
                "root": {
                    "type": "string",
                    "description": "Root directory for build_code_graph"
                },
                "patterns": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "File extension patterns for build_code_graph (e.g. ['rs', 'py'])"
                }
            },
            "required": ["operation"]
        })
    }
}

#[async_trait]
impl Tool for AstTool {
    async fn execute(&self, arguments: serde_json::Value) -> ToolResult {
        let operation = match params::extract_string(&arguments, "operation", None) {
            Ok(op) => op,
            Err(e) => return ToolResult::error(e),
        };

        match operation.as_str() {
            "parse" => self.op_parse(&arguments).await,
            "symbols" => self.op_symbols(&arguments).await,
            "query" => self.op_query(&arguments).await,
            "call_graph" => self.op_call_graph(&arguments).await,
            "import_graph" => self.op_import_graph(&arguments).await,
            "build_code_graph" => self.op_build_code_graph(&arguments).await,
            "structural_search" => self.op_structural_search(&arguments).await,
            "complexity" => self.op_complexity(&arguments).await,
            _ => ToolResult::error(format!("Unknown operation: {operation}")),
        }
    }
}

impl AstTool {
    fn ensure_file_path(path_buf: &std::path::Path, original: &str) -> Result<()> {
        if path_buf.is_dir() {
            bail!(
                "Path '{original}' is a directory. The ast tool parse/symbol/query operations require a single source file path. Use fs op=list or fs op=grep to discover files first."
            );
        }
        Ok(())
    }

    async fn op_parse(&self, args: &serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(args, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Missing 'path' for parse: {e}")),
        };
        let path_buf = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid path: {e}")),
        };
        if let Err(e) = Self::ensure_file_path(&path_buf, &path_str) {
            return ToolResult::error(e.to_string());
        }
        let ast = match AstParser::parse_file(&path_buf).await {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Parse failed: {e}")),
        };
        let symbols = extract_symbols(&ast);
        let outline: Vec<String> = symbols
            .iter()
            .map(|s| {
                format!(
                    "  {} {} @ {}:{}",
                    s.kind.as_str(),
                    s.name,
                    s.line_start + 1,
                    s.line_end + 1
                )
            })
            .collect();
        let output = format!("## AST Outline for {}\n\n{}", path_str, outline.join("\n"));
        ToolResult::success(output)
    }

    async fn op_symbols(&self, args: &serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(args, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Missing 'path' for symbols: {e}")),
        };
        let path_buf = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid path: {e}")),
        };
        if let Err(e) = Self::ensure_file_path(&path_buf, &path_str) {
            return ToolResult::error(e.to_string());
        }
        let ast = match AstParser::parse_file(&path_buf).await {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Parse failed: {e}")),
        };
        let symbols = extract_symbols(&ast);
        let json = serde_json::to_string_pretty(&symbols).unwrap_or_else(|_| "[]".to_string());
        ToolResult::success(json)
    }

    async fn op_query(&self, args: &serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(args, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Missing 'path' for query: {e}")),
        };
        let kind_str = params::extract_string_or(args, "kind", "function");
        let name_pattern = args["name_pattern"].as_str().map(String::from);
        let path_buf = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid path: {e}")),
        };
        if let Err(e) = Self::ensure_file_path(&path_buf, &path_str) {
            return ToolResult::error(e.to_string());
        }
        let ast = match AstParser::parse_file(&path_buf).await {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Parse failed: {e}")),
        };
        let kind = match kind_str.as_str() {
            "function" => NodeKind::Function,
            "class" => NodeKind::Class,
            "struct" => NodeKind::Struct,
            "import" => NodeKind::Import,
            "method" => NodeKind::Method,
            "enum" => NodeKind::Enum,
            "trait" => NodeKind::Trait,
            _ => NodeKind::Other(kind_str.clone()),
        };
        let matches = AstQuery::find_by_kind(&ast, &kind, name_pattern.as_deref());
        let results: Vec<String> = matches
            .iter()
            .map(|n| {
                format!(
                    "{} {} @ {}:{} - {}",
                    n.kind,
                    n.name.as_deref().unwrap_or("(anonymous)"),
                    n.span.start_row + 1,
                    n.span.end_row + 1,
                    n.text.lines().next().unwrap_or("").trim()
                )
            })
            .collect();
        ToolResult::success(results.join("\n"))
    }

    async fn op_call_graph(&self, args: &serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(args, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Missing 'path' for call_graph: {e}")),
        };
        let path_buf = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid path: {e}")),
        };
        if let Err(e) = Self::ensure_file_path(&path_buf, &path_str) {
            return ToolResult::error(e.to_string());
        }
        let ast = match AstParser::parse_file(&path_buf).await {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Parse failed: {e}")),
        };
        let graph = build_call_graph(&ast);
        let json = serde_json::to_string_pretty(&graph).unwrap_or_else(|_| "{}".to_string());
        ToolResult::success(json)
    }

    async fn op_import_graph(&self, args: &serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(args, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Missing 'path' for import_graph: {e}")),
        };
        let path_buf = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid path: {e}")),
        };
        if let Err(e) = Self::ensure_file_path(&path_buf, &path_str) {
            return ToolResult::error(e.to_string());
        }
        let ast = match AstParser::parse_file(&path_buf).await {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Parse failed: {e}")),
        };
        let graph = build_import_graph(&ast);
        let json = serde_json::to_string_pretty(&graph).unwrap_or_else(|_| "{}".to_string());
        ToolResult::success(json)
    }

    async fn op_build_code_graph(&self, args: &serde_json::Value) -> ToolResult {
        let root = match params::extract_string(args, "root", None) {
            Ok(r) => r,
            Err(e) => {
                return ToolResult::error(format!("Missing 'root' for build_code_graph: {e}"));
            }
        };
        let path_buf = match path::validate_and_resolve_path(&root, Some(false)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid path: {e}")),
        };
        let patterns: Vec<&str> = args["patterns"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>())
            .unwrap_or_default();
        let backend = Arc::clone(&self.graph_backend);
        let code_graph = CodeGraph::new(backend, "code");
        let stats = match code_graph.ingest_project(&path_buf, &patterns).await {
            Ok(s) => s,
            Err(e) => return ToolResult::error(format!("Ingest failed: {e}")),
        };
        ToolResult::success(format!(
            "Code graph built: {} files parsed, {} nodes added",
            stats.files_parsed, stats.nodes_added
        ))
    }

    async fn op_structural_search(&self, args: &serde_json::Value) -> ToolResult {
        self.op_query(args).await
    }

    async fn op_complexity(&self, args: &serde_json::Value) -> ToolResult {
        let path_str = match params::extract_string(args, "path", None) {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Missing 'path' for complexity: {e}")),
        };
        let path_buf = match path::validate_and_resolve_path(&path_str, Some(true)).await {
            Ok(p) => p,
            Err(e) => return ToolResult::error(format!("Invalid path: {e}")),
        };
        if let Err(e) = Self::ensure_file_path(&path_buf, &path_str) {
            return ToolResult::error(e.to_string());
        }
        let ast = match AstParser::parse_file(&path_buf).await {
            Ok(a) => a,
            Err(e) => return ToolResult::error(format!("Parse failed: {e}")),
        };
        let metrics = analyze_complexity(&ast);
        let json = serde_json::to_string_pretty(&metrics).unwrap_or_else(|_| "{}".to_string());
        ToolResult::success(json)
    }
}
