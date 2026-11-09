use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;

use memmap2::Mmap;

use crate::backend::mmap::format::{
    ArchivedMmapRowMetaV1, ParsedMmapHeaderV1, ParsedMmapHeaderV2, VectorEncoding,
};
use crate::core::{SearchQuery, SearchResult};
use crate::errors::DripstoneError;

use super::MmapBackend;

#[derive(Debug)]
struct ArchivedSearchCandidateV2<'a> {
    row: &'a ArchivedMmapRowMetaV1,
    row_index: usize,
    score: f32,
    metadata: Option<serde_json::Value>,
}

#[derive(Debug)]
struct HeapCandidate<T> {
    candidate: T,
    score: f32,
    higher_is_better: bool,
    ordinal: usize,
}

impl<T> PartialEq for HeapCandidate<T> {
    fn eq(&self, other: &Self) -> bool {
        self.ordinal == other.ordinal && self.score.to_bits() == other.score.to_bits()
    }
}

impl<T> Eq for HeapCandidate<T> {}

impl<T> PartialOrd for HeapCandidate<T> {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for HeapCandidate<T> {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        let score_cmp = if self.higher_is_better {
            other.score.total_cmp(&self.score)
        } else {
            self.score.total_cmp(&other.score)
        };

        score_cmp.then_with(|| self.ordinal.cmp(&other.ordinal))
    }
}

impl MmapBackend {
    pub(super) fn score_row_v2(
        scorer: &crate::simd::PreparedQueryScorer<'_>,
        mmap: &Mmap,
        header: &ParsedMmapHeaderV1,
        row_index: usize,
        candidate_bytes: &[u8],
    ) -> Result<f32, DripstoneError> {
        let candidate_norm = if scorer.metric() == crate::simd::DenseMetric::Cosine {
            Some(Self::vector_norm_v1(mmap, header, row_index)?)
        } else {
            None
        };
        match header.vector_encoding {
            VectorEncoding::F32 => {
                scorer.score_f32_le_bytes_with_candidate_norm(candidate_bytes, candidate_norm)
            }
            VectorEncoding::F16 => {
                scorer.score_f16_le_bytes_with_candidate_norm(candidate_bytes, candidate_norm)
            }
            VectorEncoding::Int8 | VectorEncoding::ProductQuantized => {
                Err(DripstoneError::storage(format!(
                    "mmap v1 encoding {:?} is reserved but not implemented",
                    header.vector_encoding
                )))
            }
        }
    }

    pub(super) fn score_row_v3(
        scorer: &crate::simd::PreparedQueryScorer<'_>,
        mmap: &Mmap,
        header: &ParsedMmapHeaderV2,
        row_index: usize,
        candidate_bytes: &[u8],
    ) -> Result<f32, DripstoneError> {
        let candidate_norm = if scorer.metric() == crate::simd::DenseMetric::Cosine {
            Some(Self::vector_norm_v2(mmap, header, row_index)?)
        } else {
            None
        };
        match header.vector_encoding {
            VectorEncoding::Int8 => scorer.score_i8_le_bytes_with_candidate_norm(
                candidate_bytes,
                candidate_norm,
                header.quant_scale,
                i8::try_from(header.quant_zero_point).map_err(|_| {
                    DripstoneError::storage("mmap v2 zero point does not fit in i8")
                })?,
            ),
            VectorEncoding::F32 | VectorEncoding::F16 | VectorEncoding::ProductQuantized => {
                Err(DripstoneError::storage(format!(
                    "mmap v2 encoding {:?} is not supported",
                    header.vector_encoding
                )))
            }
        }
    }

    pub(super) fn search_v1(
        mmap: &Mmap,
        header: &ParsedMmapHeaderV1,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let row_meta_table = Self::rkyv_row_meta_v1_unchecked(mmap, header);
        let scorer = query.metric.prepare(query.query())?;
        let higher_is_better = query.metric.higher_is_better();
        let mut heap: BinaryHeap<HeapCandidate<ArchivedSearchCandidateV2<'_>>> =
            BinaryHeap::with_capacity(query.limit);
        let mut ordinal = 0usize;

        for (row_index, row_meta) in row_meta_table.rows.iter().enumerate() {
            if !Self::namespace_matches_v1(row_meta, query.namespace.as_deref()) {
                continue;
            }

            let metadata = if let Some(ref filter) = query.filter {
                let metadata = Self::metadata_from_row_meta_v1(row_meta)
                    .map_err(|e| DripstoneError::storage(format!("mmap metadata json: {e}")))?;
                if !filter.matches(&metadata) {
                    continue;
                }
                Some(metadata)
            } else {
                None
            };

            let candidate_bytes = Self::vector_bytes_v2(mmap, header, row_index)?;
            let score = Self::score_row_v2(&scorer, mmap, header, row_index, candidate_bytes)?;
            let candidate = ArchivedSearchCandidateV2 {
                row: row_meta,
                row_index,
                score,
                metadata,
            };
            let heap_entry = HeapCandidate {
                candidate,
                score,
                higher_is_better,
                ordinal,
            };
            ordinal = ordinal.saturating_add(1);

            if heap.len() < query.limit {
                heap.push(heap_entry);
                continue;
            }

            let should_replace = heap.peek().is_some_and(|worst| {
                if higher_is_better {
                    score > worst.score
                } else {
                    score < worst.score
                }
            });
            if should_replace {
                let _ = heap.pop();
                heap.push(heap_entry);
            }
        }

        let finalists: Vec<ArchivedSearchCandidateV2<'_>> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|entry| entry.candidate)
            .collect();
        let mut results = Vec::with_capacity(finalists.len());
        for candidate in finalists {
            let chunk = Self::materialize_v1_chunk(
                mmap,
                header,
                candidate.row,
                candidate.row_index,
                candidate.metadata,
                query.include_content,
                query.include_vectors,
            )?;
            results.push(SearchResult::new(chunk, candidate.score, 0));
        }
        for (rank, result) in results.iter_mut().enumerate() {
            result.rank = rank;
        }
        Ok(results)
    }

    pub(super) fn search_v2(
        mmap: &Mmap,
        header: &ParsedMmapHeaderV2,
        query: &SearchQuery,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let row_meta_table = Self::rkyv_row_meta_v2_unchecked(mmap, header);
        let scorer = query.metric.prepare(query.query())?;
        let higher_is_better = query.metric.higher_is_better();
        let mut heap: BinaryHeap<HeapCandidate<ArchivedSearchCandidateV2<'_>>> =
            BinaryHeap::with_capacity(query.limit);
        let mut ordinal = 0usize;

        for (row_index, row_meta) in row_meta_table.rows.iter().enumerate() {
            if !Self::namespace_matches_v1(row_meta, query.namespace.as_deref()) {
                continue;
            }

            let metadata = if let Some(ref filter) = query.filter {
                let metadata = Self::metadata_from_row_meta_v1(row_meta)
                    .map_err(|e| DripstoneError::storage(format!("mmap metadata json: {e}")))?;
                if !filter.matches(&metadata) {
                    continue;
                }
                Some(metadata)
            } else {
                None
            };

            let candidate_bytes = Self::vector_bytes_v3(mmap, header, row_index)?;
            let score = Self::score_row_v3(&scorer, mmap, header, row_index, candidate_bytes)?;
            let candidate = ArchivedSearchCandidateV2 {
                row: row_meta,
                row_index,
                score,
                metadata,
            };
            let heap_entry = HeapCandidate {
                candidate,
                score,
                higher_is_better,
                ordinal,
            };
            ordinal = ordinal.saturating_add(1);

            if heap.len() < query.limit {
                heap.push(heap_entry);
                continue;
            }

            let should_replace = heap.peek().is_some_and(|worst| {
                if higher_is_better {
                    score > worst.score
                } else {
                    score < worst.score
                }
            });
            if should_replace {
                let _ = heap.pop();
                heap.push(heap_entry);
            }
        }

        let finalists: Vec<ArchivedSearchCandidateV2<'_>> = heap
            .into_sorted_vec()
            .into_iter()
            .map(|entry| entry.candidate)
            .collect();
        let mut results = Vec::with_capacity(finalists.len());
        for candidate in finalists {
            let chunk = Self::materialize_v2_chunk(
                mmap,
                header,
                candidate.row,
                candidate.row_index,
                candidate.metadata,
                query.include_content,
                query.include_vectors,
            )?;
            results.push(SearchResult::new(chunk, candidate.score, 0));
        }
        for (rank, result) in results.iter_mut().enumerate() {
            result.rank = rank;
        }
        Ok(results)
    }
}
