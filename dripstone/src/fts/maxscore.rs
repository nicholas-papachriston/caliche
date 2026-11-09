use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

use crate::fts::postings::DecodedPosting;
use crate::fts::{DocId, FullTextHit, PostingBlock, TermId, TextDocumentMeta};

#[derive(Debug, Clone)]
pub struct MaxScoreTerm {
    pub term_id: TermId,
    pub max_term_score: f32,
    pub postings: Vec<DecodedPosting>,
    pub blocks: Vec<PostingBlock>,
}

#[derive(Debug)]
pub struct MaxScoreRequest<'a> {
    pub limit: usize,
    pub documents: &'a [TextDocumentMeta],
    pub terms: Vec<MaxScoreTerm>,
    pub allowed_docs: Option<&'a HashSet<DocId>>,
}

#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    score: f32,
    doc_id: DocId,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.doc_id == other.doc_id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.doc_id.cmp(&self.doc_id))
    }
}

#[derive(Debug, Clone)]
struct TermCursor {
    term_id: TermId,
    max_term_score: f32,
    postings: Vec<DecodedPosting>,
    blocks: Vec<PostingBlock>,
    pos: usize,
    block_pos: usize,
}

impl TermCursor {
    fn new(term: MaxScoreTerm) -> Self {
        Self {
            term_id: term.term_id,
            max_term_score: term.max_term_score,
            postings: term.postings,
            blocks: term.blocks,
            pos: 0,
            block_pos: 0,
        }
    }

    fn current_doc(&self) -> Option<DocId> {
        self.postings.get(self.pos).map(|posting| posting.doc_id)
    }

    fn current_score(&self) -> f32 {
        self.postings
            .get(self.pos)
            .map_or(0.0, |posting| posting.score)
    }

    fn current_max_score(&self) -> f32 {
        if self.current_doc().is_some() {
            self.max_term_score
        } else {
            0.0
        }
    }

    fn advance(&mut self) {
        self.pos = self.pos.saturating_add(1);
        self.refresh_block_pos();
    }

    fn advance_to(&mut self, target_doc: DocId) {
        if self.current_doc().is_none() {
            return;
        }
        if let Some(block_idx) = self.find_candidate_block(target_doc) {
            let block = self.blocks[block_idx];
            if block.postings_offset > self.pos {
                self.pos = block.postings_offset;
            }
            self.block_pos = block_idx;
        }
        while let Some(doc_id) = self.current_doc() {
            if doc_id >= target_doc {
                break;
            }
            self.pos += 1;
        }
        self.refresh_block_pos();
    }

    fn find_candidate_block(&self, target_doc: DocId) -> Option<usize> {
        if self.blocks.is_empty() {
            return None;
        }
        let mut low = self.block_pos;
        let mut high = self.blocks.len();
        while low < high {
            let mid = low + ((high - low) / 2);
            if self.blocks[mid].last_doc_id < target_doc {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        (low < self.blocks.len()).then_some(low)
    }

    fn refresh_block_pos(&mut self) {
        while self.block_pos + 1 < self.blocks.len() {
            let next = self.blocks[self.block_pos + 1];
            if self.pos < next.postings_offset {
                break;
            }
            self.block_pos += 1;
        }
    }
}

pub fn execute(request: MaxScoreRequest<'_>) -> Vec<FullTextHit> {
    if request.limit == 0 || request.terms.is_empty() || request.documents.is_empty() {
        return Vec::new();
    }

    let mut cursors: Vec<TermCursor> = request.terms.into_iter().map(TermCursor::new).collect();
    cursors.sort_by(|left, right| {
        left.max_term_score
            .partial_cmp(&right.max_term_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.term_id.cmp(&right.term_id))
    });

    let mut heap = BinaryHeap::new();
    let mut threshold = 0.0_f32;

    loop {
        let pivot = if heap.len() < request.limit {
            0
        } else {
            essential_start(&cursors, threshold)
        };
        let Some(candidate_doc) = next_candidate(&cursors[pivot..]) else {
            break;
        };

        if request
            .allowed_docs
            .is_some_and(|allowed_docs| !allowed_docs.contains(&candidate_doc))
        {
            score_doc(&mut cursors[pivot..], candidate_doc, 0.0);
            continue;
        }

        let optional_upper_bound: f32 = cursors[..pivot]
            .iter()
            .map(TermCursor::current_max_score)
            .sum();
        let essential_score = score_doc(&mut cursors[pivot..], candidate_doc, 0.0);

        if essential_score + optional_upper_bound <= threshold {
            continue;
        }

        let total_score = score_doc(&mut cursors[..pivot], candidate_doc, essential_score);

        if total_score <= threshold {
            continue;
        }

        offer(&mut heap, request.limit, total_score, candidate_doc);
        if heap.len() == request.limit {
            threshold = heap.peek().map_or(0.0, |entry| entry.score);
        }
    }

    ranked_hits(heap, request.documents)
}

/// Move each cursor past `doc`, adding the scores of the cursors that were on
/// it to `score` in cursor order.
fn score_doc(cursors: &mut [TermCursor], doc: DocId, mut score: f32) -> f32 {
    for cursor in cursors {
        cursor.advance_to(doc);
        if cursor.current_doc() == Some(doc) {
            score += cursor.current_score();
            cursor.advance();
        }
    }
    score
}

/// Keep `doc` if the heap has room or `score` beats its smallest entry.
fn offer(heap: &mut BinaryHeap<HeapEntry>, limit: usize, score: f32, doc_id: DocId) {
    if heap.len() < limit {
        heap.push(HeapEntry { score, doc_id });
    } else if let Some(mut smallest) = heap.peek_mut()
        && score > smallest.score
    {
        *smallest = HeapEntry { score, doc_id };
    }
}

fn ranked_hits(heap: BinaryHeap<HeapEntry>, documents: &[TextDocumentMeta]) -> Vec<FullTextHit> {
    let mut hits: Vec<_> = heap
        .into_sorted_vec()
        .into_iter()
        .filter_map(|entry| {
            documents
                .get(entry.doc_id as usize)
                .map(|meta| FullTextHit {
                    doc_id: entry.doc_id,
                    chunk_id: meta.chunk_id.clone(),
                    namespace: meta.namespace.clone(),
                    score: entry.score,
                })
        })
        .collect();
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.doc_id.cmp(&right.doc_id))
    });
    hits
}

fn essential_start(cursors: &[TermCursor], threshold: f32) -> usize {
    if cursors.is_empty() {
        return 0;
    }

    let mut suffix_sum = 0.0_f32;
    for (idx, cursor) in cursors.iter().enumerate().rev() {
        suffix_sum += cursor.current_max_score();
        if suffix_sum > threshold {
            return idx;
        }
    }
    cursors.len().saturating_sub(1)
}

fn next_candidate(cursors: &[TermCursor]) -> Option<DocId> {
    cursors.iter().filter_map(TermCursor::current_doc).min()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fts::postings::DecodedPosting;

    fn doc(doc_id: DocId, chunk_id: &str) -> TextDocumentMeta {
        TextDocumentMeta {
            doc_id,
            chunk_id: chunk_id.to_string(),
            namespace: None,
            doc_len: 4,
            metadata_summary: serde_json::json!({}),
            deleted: false,
        }
    }

    fn term(term_id: TermId, max_term_score: f32, postings: &[(DocId, f32)]) -> MaxScoreTerm {
        MaxScoreTerm {
            term_id,
            max_term_score,
            postings: postings
                .iter()
                .map(|(doc_id, score)| DecodedPosting {
                    doc_id: *doc_id,
                    term_freq: 1,
                    score: *score,
                })
                .collect(),
            blocks: vec![PostingBlock {
                first_doc_id: postings.first().map_or(0, |(doc_id, _)| *doc_id),
                last_doc_id: postings.last().map_or(0, |(doc_id, _)| *doc_id),
                block_len: postings.len(),
                block_max_score: max_term_score,
                postings_offset: 0,
                postings_len: postings.len(),
            }],
        }
    }

    #[test]
    fn maxscore_prefers_best_combined_document() {
        let documents = vec![doc(0, "doc0#0"), doc(1, "doc1#0"), doc(2, "doc2#0")];
        let terms = vec![
            term(0, 5.0, &[(0, 2.0), (1, 1.0), (2, 3.5)]),
            term(1, 4.0, &[(0, 1.0), (2, 4.0)]),
        ];

        let hits = execute(MaxScoreRequest {
            limit: 2,
            documents: &documents,
            terms,
            allowed_docs: None,
        });

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk_id, "doc2#0");
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn maxscore_respects_allowed_doc_filter() {
        let documents = vec![doc(0, "doc0#0"), doc(1, "doc1#0"), doc(2, "doc2#0")];
        let terms = vec![
            term(0, 5.0, &[(0, 2.0), (1, 1.0), (2, 3.5)]),
            term(1, 4.0, &[(0, 1.0), (2, 4.0)]),
        ];
        let allowed_docs = HashSet::from([0_u32, 1_u32]);

        let hits = execute(MaxScoreRequest {
            limit: 2,
            documents: &documents,
            terms,
            allowed_docs: Some(&allowed_docs),
        });

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk_id, "doc0#0");
        assert!(hits.iter().all(|hit| hit.chunk_id != "doc2#0"));
    }
}
