use crate::errors::DripstoneError;
use crate::fts::{DocId, Posting, PostingBlock};

#[derive(Debug, Clone, Copy)]
pub struct DecodedPosting {
    pub doc_id: DocId,
    pub term_freq: u32,
    pub score: f32,
}

/// Decode delta-encoded postings into absolute (`DocId`, `term_freq`) pairs.
///
/// Returns an error when the cumulative `doc_id` would overflow `u32`,
/// which indicates a corrupted postings list. Silent saturation was
/// previously used here, which would collapse every subsequent doc into
/// `u32::MAX` and corrupt scoring.
///
/// # Errors
///
/// Returns an error when the operation cannot be completed.
pub fn decode_postings(postings: &[Posting]) -> Result<Vec<(DocId, u32)>, DripstoneError> {
    let mut decoded = Vec::with_capacity(postings.len());
    let mut doc_id: u32 = 0;
    for posting in postings {
        doc_id = doc_id.checked_add(posting.doc_id_delta).ok_or_else(|| {
            DripstoneError::storage(
                "FTS postings decode overflow: cumulative doc_id exceeded u32::MAX",
            )
        })?;
        decoded.push((doc_id, posting.term_freq));
    }
    Ok(decoded)
}

pub fn score_decoded_postings(
    decoded: Vec<(DocId, u32)>,
    mut scorer: impl FnMut(DocId, u32) -> f32,
) -> Vec<DecodedPosting> {
    let mut results = Vec::with_capacity(decoded.len());
    for (doc_id, term_freq) in decoded {
        results.push(DecodedPosting {
            doc_id,
            term_freq,
            score: scorer(doc_id, term_freq),
        });
    }
    results
}

#[must_use]
pub fn localize_blocks(blocks: &[PostingBlock], postings_offset: usize) -> Vec<PostingBlock> {
    blocks
        .iter()
        .map(|block| PostingBlock {
            postings_offset: block.postings_offset.saturating_sub(postings_offset),
            ..*block
        })
        .collect()
}
