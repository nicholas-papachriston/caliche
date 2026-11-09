/// Maximum number of results to return from RAG queries to prevent unbounded memory usage
pub(super) const MAX_TOP_K: usize = 100;

/// Result structure for directory indexing
#[derive(Debug)]
pub struct IndexDirectoryResult {
    pub files_indexed: usize,
    pub total_chunks: usize,
    pub failures: Vec<String>,
    pub token_usage: Option<crate::messages::TokenUsage>,
}
