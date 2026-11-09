use crate::errors::DripstoneError;
use crate::fts::index::InMemoryFullTextState;
use crate::fts::{DocId, PostingBlock, TermId};
use serde::{Deserialize, Serialize};

/// Durable lexical codec version reserved for future sqlite/mmap persistence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullTextCodecVersion {
    V1,
}

/// Durable lexicon entry shape for future on-disk lexical indexes.
#[derive(Debug, Clone)]
pub struct DurableLexiconEntryV1 {
    pub term_id: TermId,
    pub doc_freq: u32,
    pub postings_offset: u64,
    pub postings_len: u32,
    pub block_offset: u64,
    pub block_len: u32,
    pub max_term_score: f32,
}

/// Durable document metadata shape for future sqlite/mmap lexical sidecars.
#[derive(Debug, Clone)]
pub struct DurableDocumentMetaV1 {
    pub doc_id: DocId,
    pub chunk_id: String,
    pub namespace: Option<String>,
    pub doc_len: u32,
    pub metadata_json: String,
}

/// Future durable postings payload hook for sqlite/mmap sidecars.
#[derive(Debug, Clone)]
pub struct DurablePostingsBlockV1 {
    pub block: PostingBlock,
    pub encoded_postings: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DurableFullTextSnapshotV1 {
    state: InMemoryFullTextState,
}

pub(crate) fn encode_snapshot_v1(state: &InMemoryFullTextState) -> Result<Vec<u8>, DripstoneError> {
    serde_json::to_vec(&DurableFullTextSnapshotV1 {
        state: state.clone(),
    })
    .map_err(|e| DripstoneError::storage(format!("fts snapshot serialize: {e}")))
}

pub(crate) fn decode_snapshot_v1(bytes: &[u8]) -> Result<InMemoryFullTextState, DripstoneError> {
    serde_json::from_slice::<DurableFullTextSnapshotV1>(bytes)
        .map(|snapshot| snapshot.state)
        .map_err(|e| DripstoneError::storage(format!("fts snapshot deserialize: {e}")))
}
