use serde::{Deserialize, Serialize};

pub type TermId = u32;
pub type DocId = u32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LexiconEntry {
    pub term_id: TermId,
    pub doc_freq: usize,
    pub postings_offset: usize,
    pub postings_len: usize,
    pub block_offset: usize,
    pub block_len: usize,
    pub max_term_score: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Posting {
    pub doc_id_delta: u32,
    pub term_freq: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PostingBlock {
    pub first_doc_id: DocId,
    pub last_doc_id: DocId,
    pub block_len: usize,
    pub block_max_score: f32,
    pub postings_offset: usize,
    pub postings_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDocumentMeta {
    pub doc_id: DocId,
    pub chunk_id: String,
    pub namespace: Option<String>,
    pub doc_len: u32,
    pub metadata_summary: serde_json::Value,
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextHit {
    pub doc_id: DocId,
    pub chunk_id: String,
    pub namespace: Option<String>,
    pub score: f32,
}
