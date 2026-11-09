//! Citation and source tracking utilities for RAG applications.

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::embedding::{EmbeddingRecord, SimilarityResult};

/// Standard metadata schema for source tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    /// Unique identifier for the source document
    pub doc_id: String,
    /// Human-readable title of the document
    #[serde(skip_serializing_if = "Option::is_none")]
    pub doc_title: Option<String>,
    /// Index of this chunk within the document
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_index: Option<usize>,
    /// Total number of chunks in the document
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_chunks: Option<usize>,
    /// Original URL or file path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    /// Additional arbitrary metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<serde_json::Value>,
}

impl SourceMetadata {
    /// Creates new source metadata with just a document ID.
    pub fn new(doc_id: impl Into<String>) -> Self {
        Self {
            doc_id: doc_id.into(),
            doc_title: None,
            chunk_index: None,
            total_chunks: None,
            source_url: None,
            extra: None,
        }
    }

    #[must_use]
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.doc_title = Some(title.into());
        self
    }

    #[must_use]
    pub const fn with_chunk_index(mut self, index: usize) -> Self {
        self.chunk_index = Some(index);
        self
    }

    #[must_use]
    pub const fn with_total_chunks(mut self, total: usize) -> Self {
        self.total_chunks = Some(total);
        self
    }

    #[must_use]
    pub fn with_source_url(mut self, url: impl Into<String>) -> Self {
        self.source_url = Some(url.into());
        self
    }

    #[must_use]
    pub fn with_extra(mut self, extra: serde_json::Value) -> Self {
        self.extra = Some(extra);
        self
    }

    /// Converts this metadata into a JSON value for embedding records.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "source": {
                "doc_id": self.doc_id,
                "doc_title": self.doc_title,
                "chunk_index": self.chunk_index,
                "total_chunks": self.total_chunks,
                "source_url": self.source_url,
                "extra": self.extra,
            }
        })
    }

    /// Attempts to extract source metadata from an embedding record.
    #[must_use]
    pub fn from_record(record: &EmbeddingRecord) -> Option<Self> {
        let metadata = record.metadata.as_ref()?;
        let source = metadata.get("source")?;
        serde_json::from_value(source.clone()).ok()
    }

    /// Builds source metadata from a dripstone Chunk, including hashline in extra when present.
    #[must_use]
    pub fn from_chunk(chunk: &crate::core::Chunk) -> Self {
        let mut meta = Self::new(&chunk.doc_id)
            .with_chunk_index(chunk.chunk_index)
            .with_total_chunks(chunk.total_chunks);

        if let Some(hashline) = chunk.metadata.get("hashline").cloned() {
            meta = meta.with_extra(serde_json::json!({ "hashline": hashline }));
        }

        meta
    }
}

/// Helper for extracting and formatting citations from search results.
pub struct CitationHelper;

impl CitationHelper {
    /// Extracts citation information from dripstone `SearchResult` (includes hashline in extra).
    #[must_use]
    pub fn extract_from_search_results(results: &[crate::core::SearchResult]) -> Vec<CitationInfo> {
        results
            .iter()
            .map(|result| {
                let source = SourceMetadata::from_chunk(&result.chunk);
                CitationInfo {
                    record_id: result.chunk.id.clone(),
                    score: result.score,
                    source,
                    content_preview: Self::create_preview(&result.chunk.content, 100),
                }
            })
            .collect()
    }

    /// Extracts citation information from search results.
    ///
    /// `content_preview` is derived from the record's metadata by looking at
    /// the conventional `content` / `text` / `body` keys (in that order).
    /// The record ID is only used as a last-resort fallback, since using it
    /// as a "preview" was actively misleading to end users.
    #[must_use]
    pub fn extract_citations(results: &[SimilarityResult]) -> Vec<CitationInfo> {
        results
            .iter()
            .filter_map(|result| {
                SourceMetadata::from_record(&result.record).map(|meta| {
                    let preview = Self::preview_from_record(&result.record);
                    CitationInfo {
                        record_id: result.record.id.to_string(),
                        score: result.score,
                        source: meta,
                        content_preview: preview,
                    }
                })
            })
            .collect()
    }

    /// Try to derive a human-readable preview for an `EmbeddingRecord` from
    /// its metadata. Falls back to an empty string when no recognizable
    /// content field is present; callers that need a strict guarantee of
    /// non-empty preview should use `extract_from_search_results` on
    /// `SearchResult` which always carries the chunk content.
    fn preview_from_record(record: &EmbeddingRecord) -> String {
        const CONTENT_KEYS: &[&str] = &["content", "text", "body"];
        let Some(metadata) = record.metadata.as_ref() else {
            return String::new();
        };
        for key in CONTENT_KEYS {
            if let Some(value) = metadata.get(*key)
                && let Some(s) = value.as_str()
                && !s.is_empty()
            {
                return Self::create_preview(s, 100);
            }
        }
        String::new()
    }

    /// Formats citations in a standard academic style.
    #[must_use]
    pub fn format_citations(citations: &[CitationInfo]) -> Vec<String> {
        citations
            .iter()
            .enumerate()
            .map(|(i, citation)| {
                let title = citation.source.doc_title.as_deref().unwrap_or("Untitled");
                let doc_id = &citation.source.doc_id;
                let chunk_info = citation
                    .source
                    .chunk_index
                    .map(|idx| format!(", chunk {idx}"))
                    .unwrap_or_default();

                format!("[{}] {} ({}{})", i + 1, title, doc_id, chunk_info)
            })
            .collect()
    }

    /// Groups citations by document.
    #[must_use]
    pub fn group_by_document(citations: &[CitationInfo]) -> Vec<(String, Vec<CitationInfo>)> {
        let mut grouped: Vec<(String, Vec<CitationInfo>)> = Vec::new();

        for citation in citations {
            let doc_id = citation.source.doc_id.clone();

            if let Some((_, list)) = grouped.iter_mut().find(|(id, _)| id == &doc_id) {
                list.push(citation.clone());
            } else {
                grouped.push((doc_id, vec![citation.clone()]));
            }
        }

        grouped
    }

    fn create_preview(content: &str, max_len: usize) -> String {
        if content.len() <= max_len {
            content.to_string()
        } else {
            format!("{}...", &content[..content.floor_char_boundary(max_len)])
        }
    }
}

/// Citation information extracted from a search result.
#[derive(Debug, Clone)]
pub struct CitationInfo {
    pub record_id: String,
    pub score: f32,
    pub source: SourceMetadata,
    pub content_preview: String,
}

/// Helper for reconstructing expanded context from chunked documents.
pub struct ContextReconstructor;

impl ContextReconstructor {
    /// Attempts to fetch neighboring chunks for a given result.
    ///
    /// This is a utility structure - actual implementation would need
    /// access to the embedding client to fetch additional chunks.
    #[must_use]
    pub fn get_neighbor_ids(record: &EmbeddingRecord, context_window: usize) -> Vec<String> {
        let Some(source_meta) = SourceMetadata::from_record(record) else {
            return Vec::new();
        };

        let Some(current_idx) = source_meta.chunk_index else {
            return Vec::new();
        };

        let doc_id = &source_meta.doc_id;
        let total_chunks = source_meta.total_chunks.unwrap_or(current_idx + 1);

        let start_idx = current_idx.saturating_sub(context_window);
        let end_idx = (current_idx + context_window + 1).min(total_chunks);

        (start_idx..end_idx)
            .filter(|&idx| idx != current_idx)
            .map(|idx| format!("{doc_id}:chunk:{idx}"))
            .collect()
    }

    /// Creates a context expansion plan for a set of results.
    #[must_use]
    pub fn plan_context_expansion(
        results: &[SimilarityResult],
        context_window: usize,
    ) -> Vec<String> {
        let mut all_ids = Vec::new();

        for result in results {
            let neighbor_ids = Self::get_neighbor_ids(&result.record, context_window);
            all_ids.extend(neighbor_ids);
        }

        // Deduplicate
        all_ids.sort();
        all_ids.dedup();
        all_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_metadata_creation() {
        let meta = SourceMetadata::new("doc123")
            .with_title("Test Document")
            .with_chunk_index(5)
            .with_total_chunks(10);

        assert_eq!(meta.doc_id, "doc123");
        assert_eq!(meta.doc_title, Some("Test Document".to_string()));
        assert_eq!(meta.chunk_index, Some(5));
        assert_eq!(meta.total_chunks, Some(10));
    }

    #[test]
    fn test_source_metadata_to_json() {
        let meta = SourceMetadata::new("doc123").with_title("Test");
        let json = meta.to_json();

        assert!(json.get("source").is_some());
        assert_eq!(json["source"]["doc_id"], "doc123");
    }

    #[test]
    fn test_get_neighbor_ids() {
        let mut record = EmbeddingRecord::new("test", vec![1.0]);
        let meta = SourceMetadata::new("doc1")
            .with_chunk_index(5)
            .with_total_chunks(10);
        record.metadata = Some(meta.to_json());

        let neighbors = ContextReconstructor::get_neighbor_ids(&record, 2);

        // Should include chunks 3,4,6,7 (within window of 2)
        assert!(!neighbors.is_empty());
        assert!(neighbors.iter().any(|id| id.contains("chunk:3")));
    }

    #[test]
    fn test_citation_helper_format() {
        let citations = vec![CitationInfo {
            record_id: "r1".to_string(),
            score: 0.95,
            source: SourceMetadata::new("doc1").with_title("Sample Doc"),
            content_preview: "Preview text".to_string(),
        }];

        let formatted = CitationHelper::format_citations(&citations);
        assert_eq!(formatted.len(), 1);
        assert!(formatted[0].contains("Sample Doc"));
    }

    #[test]
    fn test_group_by_document() {
        let citations = vec![
            CitationInfo {
                record_id: "r1".to_string(),
                score: 0.95,
                source: SourceMetadata::new("doc1"),
                content_preview: String::new(),
            },
            CitationInfo {
                record_id: "r2".to_string(),
                score: 0.90,
                source: SourceMetadata::new("doc1"),
                content_preview: String::new(),
            },
            CitationInfo {
                record_id: "r3".to_string(),
                score: 0.85,
                source: SourceMetadata::new("doc2"),
                content_preview: String::new(),
            },
        ];

        let grouped = CitationHelper::group_by_document(&citations);
        assert_eq!(grouped.len(), 2);
        assert_eq!(grouped[0].1.len(), 2); // doc1 has 2 citations
        assert_eq!(grouped[1].1.len(), 1); // doc2 has 1 citation
    }
}
