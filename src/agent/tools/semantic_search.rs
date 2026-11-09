//! Semantic Search Tool for codebase exploration.
//!
//! This tool enables semantic code search using embeddings, allowing the agent
//! to find relevant code by meaning rather than exact text matching.

use crate::agent::tools::common::params;
use crate::agent::tools::{Tool, ToolMeta, ToolResult};
use crate::llm::EmbeddingProvider;
use crate::utils::math::cosine_similarity;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for the semantic search tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSearchConfig {
    /// Workspace root directory
    pub workspace_root: PathBuf,
    /// File extensions to index
    pub file_extensions: Vec<String>,
    /// Maximum file size to index (in bytes)
    pub max_file_size: usize,
    /// Chunk size for splitting files
    pub chunk_size: usize,
    /// Chunk overlap
    pub chunk_overlap: usize,
    /// Number of results to return by default
    pub default_top_k: usize,
    /// Minimum similarity score (0.0 - 1.0)
    pub min_similarity: f32,
    /// Directories to ignore
    pub ignore_dirs: Vec<String>,
}

impl Default for SemanticSearchConfig {
    fn default() -> Self {
        Self {
            workspace_root: PathBuf::from("."),
            file_extensions: vec![
                "rs".to_string(),
                "py".to_string(),
                "js".to_string(),
                "ts".to_string(),
                "tsx".to_string(),
                "jsx".to_string(),
                "go".to_string(),
                "java".to_string(),
                "c".to_string(),
                "cpp".to_string(),
                "h".to_string(),
                "hpp".to_string(),
                "rb".to_string(),
                "php".to_string(),
                "swift".to_string(),
                "kt".to_string(),
                "scala".to_string(),
                "md".to_string(),
                "txt".to_string(),
            ],
            max_file_size: 1_000_000, // 1MB
            chunk_size: 1000,
            chunk_overlap: 200,
            default_top_k: 10,
            min_similarity: 0.3,
            ignore_dirs: vec![
                "target".to_string(),
                "node_modules".to_string(),
                ".git".to_string(),
                "__pycache__".to_string(),
                "dist".to_string(),
                "build".to_string(),
                ".next".to_string(),
                "vendor".to_string(),
            ],
        }
    }
}

/// A chunk of code with its embedding
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// File path relative to workspace root
    pub path: PathBuf,
    /// Start line number (1-indexed)
    pub start_line: usize,
    /// End line number (1-indexed)
    pub end_line: usize,
    /// The actual content
    pub content: String,
    /// Embedding vector
    pub embedding: Vec<f32>,
}

/// Search result
#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    /// File path
    pub path: String,
    /// Start line
    pub start_line: usize,
    /// End line
    pub end_line: usize,
    /// Content snippet
    pub content: String,
    /// Similarity score
    pub similarity: f32,
}

/// Codebase index for semantic search
#[derive(Default)]
pub struct CodebaseIndex {
    /// Indexed chunks
    chunks: Vec<CodeChunk>,
    /// Index state
    indexed_files: HashMap<PathBuf, chrono::DateTime<chrono::Utc>>,
}

impl CodebaseIndex {
    /// Create a new codebase index
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new codebase index from a config (config values are used by the parent tool)
    #[must_use]
    pub fn from_config(_config: &SemanticSearchConfig) -> Self {
        Self::new()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.indexed_files.clear();
    }

    /// Get index statistics
    #[must_use]
    pub fn stats(&self) -> (usize, usize) {
        (self.chunks.len(), self.indexed_files.len())
    }

    /// Add a chunk to the index
    pub fn add_chunk(&mut self, chunk: CodeChunk) {
        self.chunks.push(chunk);
    }

    /// Mark a file as indexed
    pub fn mark_indexed(&mut self, path: PathBuf) {
        self.indexed_files.insert(path, chrono::Utc::now());
    }

    /// Check if a file is indexed
    #[must_use]
    pub fn is_indexed(&self, path: &Path) -> bool {
        self.indexed_files.contains_key(path)
    }

    /// Search for similar chunks
    #[must_use]
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Vec<SearchResult> {
        let mut results: Vec<(f32, &CodeChunk)> = self
            .chunks
            .iter()
            .filter_map(|chunk| {
                let similarity = cosine_similarity(query_embedding, &chunk.embedding);
                if similarity >= min_similarity {
                    Some((similarity, chunk))
                } else {
                    None
                }
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k results
        results
            .into_iter()
            .take(top_k)
            .map(|(similarity, chunk)| SearchResult {
                path: chunk.path.display().to_string(),
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                content: truncate_content(&chunk.content, 500),
                similarity,
            })
            .collect()
    }
}

/// Truncate content for display
fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        let end = content.floor_char_boundary(max_len);
        format!("{}...", &content[..end])
    }
}

/// Semantic Search Tool
pub struct SemanticSearchTool {
    /// Embedding provider for generating embeddings
    embedding_provider: Arc<dyn EmbeddingProvider>,
    /// Codebase index
    index: Arc<RwLock<CodebaseIndex>>,
    /// Configuration
    config: SemanticSearchConfig,
}

impl SemanticSearchTool {
    /// Create a new semantic search tool
    pub fn new(
        embedding_provider: Arc<dyn EmbeddingProvider>,
        config: SemanticSearchConfig,
    ) -> Self {
        let index = Arc::new(RwLock::new(CodebaseIndex::from_config(&config)));
        Self {
            embedding_provider,
            index,
            config,
        }
    }

    /// Create with default configuration
    pub fn with_defaults(embedding_provider: Arc<dyn EmbeddingProvider>) -> Self {
        Self::new(embedding_provider, SemanticSearchConfig::default())
    }

    /// Set the workspace root
    #[must_use]
    pub fn with_workspace(mut self, workspace_root: PathBuf) -> Self {
        self.config.workspace_root = workspace_root;
        self
    }

    /// Index a single file
    async fn index_file(&self, path: &Path) -> Result<usize> {
        // Read file content
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read file '{}'", path.display()))?;

        if content.len() > self.config.max_file_size {
            bail!("File too large: {} bytes", content.len());
        }

        // Split into chunks
        let chunks = self.split_into_chunks(&content, path);
        if chunks.is_empty() {
            return Ok(0);
        }

        // Generate embeddings for all chunks
        let texts: Vec<String> = chunks.iter().map(|(_, _, _, text)| text.clone()).collect();

        let request = crate::llm::EmbeddingRequest { texts, model: None };

        let response = self
            .embedding_provider
            .embed(request)
            .await
            .context("Failed to generate embeddings")?;

        let embeddings = response.embeddings;

        // Store chunks with embeddings
        let mut index = self.index.write().await;
        let chunk_count = chunks.len();

        for ((start_line, end_line, content, _), embedding) in
            chunks.into_iter().zip(embeddings.into_iter())
        {
            index.add_chunk(CodeChunk {
                path: path.to_path_buf(),
                start_line,
                end_line,
                content,
                embedding,
            });
        }

        index.mark_indexed(path.to_path_buf());
        Ok(chunk_count)
    }

    /// Split content into chunks with line numbers
    fn split_into_chunks(
        &self,
        content: &str,
        _path: &Path,
    ) -> Vec<(usize, usize, String, String)> {
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();

        let chunk_lines = self.config.chunk_size / 80; // Approximate lines per chunk
        let overlap_lines = self.config.chunk_overlap / 80;

        let mut start = 0;
        while start < lines.len() {
            let end = (start + chunk_lines).min(lines.len());
            let chunk_content: String = lines[start..end].join("\n");

            // Create a version without line numbers for embedding
            let chunk_text = chunk_content.clone();

            chunks.push((start + 1, end, chunk_content, chunk_text));

            if end >= lines.len() {
                break;
            }

            start = end - overlap_lines;
        }

        chunks
    }

    /// Index the workspace
    async fn index_workspace(&self) -> Result<(usize, usize)> {
        let mut total_files = 0;
        let mut total_chunks = 0;

        let workspace = self.config.workspace_root.clone();
        let files = self.collect_files(&workspace).await?;

        for file in files {
            match self.index_file(&file).await {
                Ok(chunks) => {
                    total_files += 1;
                    total_chunks += chunks;
                    tracing::debug!("Indexed {} ({} chunks)", file.display(), chunks);
                }
                Err(e) => {
                    tracing::warn!("Failed to index {}: {}", file.display(), e);
                }
            }
        }

        Ok((total_files, total_chunks))
    }

    /// Collect files to index
    async fn collect_files(&self, dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        self.collect_files_recursive(dir, &mut files).await?;
        Ok(files)
    }

    fn collect_files_recursive<'a>(
        &'a self,
        dir: &'a Path,
        files: &'a mut Vec<PathBuf>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut entries = tokio::fs::read_dir(dir)
                .await
                .with_context(|| format!("Failed to read directory '{}'", dir.display()))?;

            while let Some(entry) = entries
                .next_entry()
                .await
                .context("Failed to read directory entry")?
            {
                let path = entry.path();
                let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                // Skip ignored directories
                if path.is_dir() {
                    if self.config.ignore_dirs.iter().any(|d| file_name == d) {
                        continue;
                    }
                    self.collect_files_recursive(&path, files).await?;
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                    && self.config.file_extensions.iter().any(|e| e == ext)
                {
                    files.push(path);
                }
            }

            Ok(())
        })
    }

    /// Search for similar code
    async fn search(
        &self,
        query: &str,
        top_k: Option<usize>,
        min_similarity: Option<f32>,
    ) -> Result<Vec<SearchResult>> {
        // Generate query embedding
        let request = crate::llm::EmbeddingRequest {
            texts: vec![query.to_string()],
            model: None,
        };

        let mut perf = crate::observability::perf::span("semantic_search.search");
        let response = self
            .embedding_provider
            .embed(request)
            .await
            .context("Failed to generate query embedding")?;

        let query_embedding = response
            .embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No embedding returned"))?;

        // Search index
        let index = self.index.read().await;
        let results = index.search(
            &query_embedding,
            top_k.unwrap_or(self.config.default_top_k),
            min_similarity.unwrap_or(self.config.min_similarity),
        );
        if let Some(ref mut span) = perf {
            span.set_match_count(results.len() as u64);
        }

        Ok(results)
    }
}

impl std::fmt::Debug for SemanticSearchTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SemanticSearchTool")
            .field("config", &self.config)
            .finish()
    }
}

const NAME: &str = "semantic_search";
const DESCRIPTION: &str = "Search the codebase semantically to find relevant code by meaning. \
Use this to explore unfamiliar codebases, find implementations, \
or understand how specific features work.";

impl ToolMeta for SemanticSearchTool {
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

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "Operation to perform: 'search', 'index', 'index_file', or 'stats'",
                    "enum": ["search", "index", "index_file", "stats"]
                },
                "query": {
                    "type": "string",
                    "description": "Search query describing what you're looking for (for 'search' operation)"
                },
                "file": {
                    "type": "string",
                    "description": "File path to index (for 'index_file' operation)"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 10)",
                    "default": 10
                },
                "min_similarity": {
                    "type": "number",
                    "description": "Minimum similarity score 0.0-1.0 (default: 0.3)",
                    "default": 0.3
                }
            },
            "required": ["operation"]
        })
    }
}

#[async_trait]
impl Tool for SemanticSearchTool {
    async fn execute(&self, args: Value) -> ToolResult {
        let operation = match params::extract_string(&args, "operation", Some("search")) {
            Ok(op) => op,
            Err(e) => return ToolResult::error(e),
        };

        match operation.as_str() {
            "search" => {
                let query = match args["query"].as_str() {
                    Some(q) => q,
                    None => {
                        return ToolResult::error(
                            "Query is required for search operation".to_string(),
                        );
                    }
                };

                let top_k = args["top_k"].as_u64().map(|n| n as usize);
                let min_similarity = args["min_similarity"].as_f64().map(|n| n as f32);

                match self.search(query, top_k, min_similarity).await {
                    Ok(results) => {
                        if results.is_empty() {
                            ToolResult::success("No results found matching the query.")
                        } else {
                            let output = serde_json::to_string_pretty(&json!({
                                "results": results,
                                "count": results.len()
                            }))
                            .unwrap_or_else(|_| "Failed to serialize results".to_string());
                            ToolResult::success(output)
                        }
                    }
                    Err(e) => ToolResult::error(e.to_string()),
                }
            }

            "index" => match self.index_workspace().await {
                Ok((files, chunks)) => {
                    ToolResult::success(format!("Indexed {files} files ({chunks} chunks)"))
                }
                Err(e) => ToolResult::error(format!("Failed to index workspace: {e}")),
            },

            "index_file" => {
                let file = match args["file"].as_str() {
                    Some(f) => PathBuf::from(f),
                    None => {
                        return ToolResult::error(
                            "File path is required for index_file operation".to_string(),
                        );
                    }
                };

                match self.index_file(&file).await {
                    Ok(chunks) => ToolResult::success(format!(
                        "Indexed {} ({} chunks)",
                        file.display(),
                        chunks
                    )),
                    Err(e) => ToolResult::error(format!("Failed to index file: {e}")),
                }
            }

            "stats" => {
                let index = self.index.read().await;
                let (chunks, files) = index.stats();
                ToolResult::success(format!("Index stats: {chunks} chunks from {files} files"))
            }

            _ => ToolResult::error(format!("Unknown operation: {operation}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.0001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.0001);

        let d = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &d) - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_truncate_content() {
        assert_eq!(truncate_content("hello", 10), "hello");
        assert_eq!(truncate_content("hello world", 8), "hello wo...");
    }
}
