//! AST-to-graph conversion for storing code structure in Dripstone graph backends.

use crate::ast::analysis::extract_symbols;
use crate::ast::query::AstQuery;
use crate::ast::types::{AstNode, NodeKind};
use anyhow::Result;
use dripstone::graph::{Edge, GraphBackend, Node};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tokio::fs;

/// Statistics from ingesting a project.
#[derive(Debug, Clone, Default)]
pub struct CodeGraphStats {
    pub files_parsed: usize,
    pub nodes_added: usize,
    pub edges_added: usize,
}

/// Converts AST structures to Dripstone graph storage for structural exploration.
pub struct CodeGraph {
    backend: Arc<dyn GraphBackend>,
    namespace: String,
}

impl CodeGraph {
    pub fn new(backend: Arc<dyn GraphBackend>, namespace: impl Into<String>) -> Self {
        Self {
            backend,
            namespace: namespace.into(),
        }
    }

    /// Ingest an AST into the graph backend.
    pub async fn ingest_ast(&self, ast: &AstNode) -> Result<()> {
        let node_ids = self.add_ast_nodes(ast).await?;
        self.add_containment_edges(ast, &node_ids).await?;
        Ok(())
    }

    async fn add_ast_nodes(&self, ast: &AstNode) -> Result<HashSet<String>> {
        let mut node_ids = HashSet::new();
        let significant = [
            NodeKind::Module,
            NodeKind::Function,
            NodeKind::Method,
            NodeKind::Class,
            NodeKind::Struct,
            NodeKind::Enum,
            NodeKind::Trait,
            NodeKind::Interface,
            NodeKind::Import,
            NodeKind::Object,
            NodeKind::Array,
            NodeKind::KeyValue,
            NodeKind::Heading,
            NodeKind::Paragraph,
            NodeKind::List,
            NodeKind::ListItem,
            NodeKind::Table,
            NodeKind::TableRow,
            NodeKind::Record,
            NodeKind::Field,
        ];
        for kind in significant {
            for node in AstQuery::find_by_kind(ast, &kind, None) {
                let gf_node = self.ast_node_to_graph_node(node)?;
                self.backend
                    .add_node(gf_node)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to add node: {}", e))?;
                node_ids.insert(node.id.clone());
            }
        }
        Ok(node_ids)
    }

    fn ast_node_to_graph_node(&self, ast: &AstNode) -> Result<Node> {
        let id = format!("{}:{}", self.namespace, ast.id);
        let mut props = std::collections::HashMap::new();
        props.insert(
            "file_path".to_string(),
            serde_json::json!(ast.file_path.display().to_string()),
        );
        props.insert(
            "line_start".to_string(),
            serde_json::json!(ast.span.start_row),
        );
        props.insert("line_end".to_string(), serde_json::json!(ast.span.end_row));
        props.insert(
            "snippet".to_string(),
            serde_json::json!(ast.text.lines().take(3).fold(String::new(), |mut s, line| {
                if !s.is_empty() {
                    s.push('\n');
                }
                s.push_str(line);
                s
            })),
        );
        props.insert(
            "language".to_string(),
            serde_json::json!(ast.language.as_str()),
        );
        if let Some(ref name) = ast.name {
            props.insert("name".to_string(), serde_json::json!(name));
        }
        let mut node = Node::new(id, ast.kind.as_str().to_owned());
        for (k, v) in props {
            node = node.with_property(k, v);
        }
        Ok(node)
    }

    async fn add_containment_edges(&self, ast: &AstNode, node_ids: &HashSet<String>) -> Result<()> {
        let _symbols = extract_symbols(ast);
        let parent_kinds = [
            NodeKind::Module,
            NodeKind::Class,
            NodeKind::Struct,
            NodeKind::Enum,
            NodeKind::Object,
            NodeKind::Array,
            NodeKind::List,
            NodeKind::Table,
            NodeKind::Block,
        ];
        for kind in parent_kinds {
            for parent in AstQuery::find_by_kind(ast, &kind, None) {
                let parent_id = format!("{}:{}", self.namespace, parent.id);
                if !node_ids.contains(&parent.id) {
                    continue;
                }
                for child in &parent.children {
                    let child_id = format!("{}:{}", self.namespace, child.id);
                    if node_ids.contains(&child.id) {
                        let edge = Edge::new(parent_id.clone(), child_id, "CONTAINS".to_string());
                        let _ = self.backend.add_edge(edge).await;
                    }
                }
            }
        }
        Ok(())
    }

    /// Ingest a whole project (multiple files).
    ///
    /// `root` is canonical. Only regular files directly in it are parsed: a
    /// link is skipped, and each file is opened without following a link.
    pub async fn ingest_project(&self, root: &Path, patterns: &[&str]) -> Result<CodeGraphStats> {
        let mut stats = CodeGraphStats::default();
        let mut entries = fs::read_dir(root)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read dir: {}", e))?;
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| anyhow::anyhow!("ReadDir error: {}", e))?
        {
            let path = entry.path();
            let is_file = entry.file_type().await.is_ok_and(|kind| kind.is_file());
            if is_file {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                let matches = patterns.is_empty()
                    || patterns
                        .iter()
                        .any(|p| *p == ext || path.to_str().is_some_and(|s| s.contains(p)));
                if matches
                    && let Ok(ast) = crate::ast::parser::AstParser::parse_file(&path).await
                    && self.ingest_ast(&ast).await.is_ok()
                {
                    stats.files_parsed += 1;
                    stats.nodes_added += extract_symbols(&ast).len();
                }
            }
        }
        Ok(stats)
    }
}

#[cfg(test)]
#[cfg(all(unix, feature = "ast-rust"))]
mod link_tests {
    use super::CodeGraph;
    use crate::ast::parser::AstParser;
    use std::sync::Arc;

    /// A canonical workspace with `ok.rs` and `leak.rs`, a link to a Rust
    /// file outside it.
    fn workspace_with_link() -> (tempfile::TempDir, tempfile::TempDir, std::path::PathBuf) {
        let outside = tempfile::tempdir().expect("outside dir");
        std::fs::write(outside.path().join("secret.rs"), "fn secret() {}\n").expect("secret");
        let ws = tempfile::tempdir().expect("workspace");
        let root = ws.path().canonicalize().expect("canonical workspace");
        std::fs::write(root.join("ok.rs"), "fn ok() {}\n").expect("ok");
        std::os::unix::fs::symlink(outside.path().join("secret.rs"), root.join("leak.rs"))
            .expect("file link");
        (outside, ws, root)
    }

    #[tokio::test]
    async fn a_project_ingest_parses_only_files_that_are_not_links() {
        let (_outside, _ws, root) = workspace_with_link();
        let graph = CodeGraph::new(
            Arc::new(dripstone::graph::InMemoryGraphBackend::new()),
            "code",
        );

        let stats = graph.ingest_project(&root, &["rs"]).await.expect("ingest");

        assert_eq!(stats.files_parsed, 1);
    }

    #[tokio::test]
    async fn parsing_refuses_a_file_that_is_a_link() {
        let (_outside, _ws, root) = workspace_with_link();

        let err = AstParser::parse_file(&root.join("leak.rs"))
            .await
            .expect_err("a link");

        assert_eq!(
            err.to_string(),
            format!("Failed to read file: {}", root.join("leak.rs").display())
        );
    }
}
