//! PLAID-style inverted index for late-interaction (ColBERT) candidate generation.
//!
//! Token embeddings are assigned to a shared codebook; each centroid owns a posting
//! list of document ids. Query-time probing unions those lists, then exact MaxSim
//! ranks the shortlist.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use parking_lot::RwLock;

use crate::core::{Chunk, Metric, SearchQuery, SearchResult};
use crate::errors::DripstoneError;
use crate::search::{MultiVectors, exact_maxsim_search, maxsim_score, multi_vectors_from_owned};
use crate::simd::{DenseMetric, PreparedQueryScorer};

/// Configuration for [`PlaidIndex`].
#[derive(Debug, Clone)]
pub struct PlaidConfig {
    /// Codebook size (centroids over token space).
    pub num_centroids: usize,
    /// Centroids probed per query token.
    pub probes: usize,
    /// Soft cap on unique candidate docs before MaxSim.
    pub candidate_budget: usize,
    /// Pairwise metric for centroid assignment and MaxSim.
    pub metric: Metric,
}

impl Default for PlaidConfig {
    fn default() -> Self {
        Self {
            num_centroids: 64,
            probes: 4,
            candidate_budget: 256,
            metric: Metric::DotProduct,
        }
    }
}

#[derive(Debug, Clone)]
struct PlaidDoc {
    id: String,
    namespace: Option<String>,
    multi_vectors: MultiVectors,
    chunk: Chunk,
}

#[derive(Debug, Default)]
struct PlaidState {
    dim: usize,
    centroids: Vec<Vec<f32>>,
    /// `postings[centroid]` = internal doc ids that have at least one token there.
    postings: Vec<Vec<u32>>,
    docs: Vec<PlaidDoc>,
    id_to_idx: HashMap<String, u32>,
}

/// Inverted late-interaction index (PLAID-inspired).
#[derive(Debug)]
pub struct PlaidIndex {
    config: PlaidConfig,
    state: RwLock<PlaidState>,
}

impl PlaidIndex {
    #[must_use]
    pub fn new(config: PlaidConfig) -> Self {
        Self {
            config,
            state: RwLock::new(PlaidState::default()),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.state.read().docs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Rebuild the codebook and postings from chunks that carry multi-vectors.
    ///
    /// # Errors
    ///
    /// Returns when scoring fails during codebook construction.
    pub fn rebuild(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let mut tokens: Vec<Vec<f32>> = Vec::new();
        let mut docs = Vec::new();
        let mut id_to_idx = HashMap::new();

        for chunk in chunks {
            let Some(ref mv) = chunk.multi_vectors else {
                continue;
            };
            if mv.is_empty() {
                continue;
            }
            let dim = mv[0].len();
            if dim == 0 {
                continue;
            }
            let idx = u32::try_from(docs.len())
                .map_err(|_| DripstoneError::pipeline("PLAID document count exceeds u32::MAX"))?;
            id_to_idx.insert(chunk.id.clone(), idx);
            for token in mv.iter() {
                if token.len() == dim {
                    tokens.push(token.to_vec());
                }
            }
            docs.push(PlaidDoc {
                id: chunk.id.clone(),
                namespace: chunk.namespace.clone(),
                multi_vectors: Arc::clone(mv),
                chunk: chunk.clone(),
            });
        }

        let dim = tokens.first().map_or(0, Vec::len);
        let mut centroids = sample_centroids(&tokens, self.config.num_centroids.max(1), dim);
        refine_centroids(&mut centroids, &tokens, self.config.metric, 4)?;

        let n_centroids = centroids.len().max(1);
        let mut postings = vec![Vec::new(); n_centroids];
        for (doc_idx, doc) in docs.iter().enumerate() {
            let mut seen = HashSet::new();
            for token in doc.multi_vectors.iter() {
                let c = nearest_centroid(token, &centroids, self.config.metric)?;
                if seen.insert(c) {
                    postings[c].push(doc_idx as u32);
                }
            }
        }

        *self.state.write() = PlaidState {
            dim,
            centroids,
            postings,
            docs,
            id_to_idx,
        };
        Ok(())
    }

    /// Upsert a single chunk into the index (rebuilds codebook when empty / dim changes).
    ///
    /// # Errors
    ///
    /// Returns when assignment fails.
    pub fn upsert_chunk(&self, chunk: &Chunk) -> Result<(), DripstoneError> {
        let Some(ref mv) = chunk.multi_vectors else {
            return Ok(());
        };
        if mv.is_empty() {
            return Ok(());
        }
        let dim = mv[0].len();
        {
            let state = self.state.read();
            if state.centroids.is_empty() || state.dim != dim {
                drop(state);
                // Cheap path for cold start: rebuild from just this chunk; callers
                // that batch should prefer [`Self::rebuild`].
                return self.rebuild(std::slice::from_ref(chunk));
            }
        }

        let mut state = self.state.write();
        if let Some(&idx) = state.id_to_idx.get(&chunk.id) {
            remove_doc_from_postings(&mut state, idx);
            state.docs[idx as usize] = PlaidDoc {
                id: chunk.id.clone(),
                namespace: chunk.namespace.clone(),
                multi_vectors: Arc::clone(mv),
                chunk: chunk.clone(),
            };
            assign_doc_to_postings(&mut state, idx, self.config.metric)?;
        } else {
            let idx = u32::try_from(state.docs.len())
                .map_err(|_| DripstoneError::pipeline("PLAID document count exceeds u32::MAX"))?;
            state.id_to_idx.insert(chunk.id.clone(), idx);
            state.docs.push(PlaidDoc {
                id: chunk.id.clone(),
                namespace: chunk.namespace.clone(),
                multi_vectors: Arc::clone(mv),
                chunk: chunk.clone(),
            });
            assign_doc_to_postings(&mut state, idx, self.config.metric)?;
        }
        Ok(())
    }

    /// Remove a chunk id from the index.
    pub fn remove_chunk(&self, id: &str) {
        let mut state = self.state.write();
        let Some(idx) = state.id_to_idx.remove(id) else {
            return;
        };
        remove_doc_from_postings(&mut state, idx);
        // Tombstone: clear multi-vectors so search skips it.
        if let Some(doc) = state.docs.get_mut(idx as usize) {
            doc.multi_vectors = multi_vectors_from_owned(vec![]);
            doc.id.clear();
        }
    }

    /// Probe postings + exact MaxSim for a multi-vector query.
    ///
    /// # Errors
    ///
    /// Returns when scoring fails or the query lacks multi-vectors.
    pub fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let Some(query_tokens) = query.query_multi_vectors.as_ref() else {
            return Err(DripstoneError::pipeline(
                "PlaidIndex::search requires query_multi_vectors",
            ));
        };
        let state = self.state.read();
        if state.docs.is_empty() || state.centroids.is_empty() {
            return Ok(Vec::new());
        }

        let probes = self.config.probes.max(1).min(state.centroids.len());
        let mut candidate_ids: HashSet<u32> = HashSet::new();
        for token in query_tokens.iter() {
            let nearest = top_centroids(token, &state.centroids, self.config.metric, probes)?;
            for c in nearest {
                for &doc_idx in &state.postings[c] {
                    candidate_ids.insert(doc_idx);
                    if candidate_ids.len() >= self.config.candidate_budget.max(1) {
                        break;
                    }
                }
                if candidate_ids.len() >= self.config.candidate_budget.max(1) {
                    break;
                }
            }
            if candidate_ids.len() >= self.config.candidate_budget.max(1) {
                break;
            }
        }

        let mut chunks = Vec::with_capacity(candidate_ids.len());
        for doc_idx in candidate_ids {
            let Some(doc) = state.docs.get(doc_idx as usize) else {
                continue;
            };
            if doc.id.is_empty() || doc.multi_vectors.is_empty() {
                continue;
            }
            if let Some(ref ns) = query.namespace
                && doc.namespace.as_deref() != Some(ns.as_str())
            {
                continue;
            }
            chunks.push(doc.chunk.clone());
        }
        drop(state);

        // Exact MaxSim over the posting shortlist (same scoring as Memory path).
        exact_maxsim_search(&chunks, query)
    }

    /// Score one query/doc pair (test / debug helper).
    ///
    /// # Errors
    ///
    /// Returns when MaxSim scoring fails.
    pub fn score_pair(
        &self,
        query_tokens: &MultiVectors,
        doc_id: &str,
    ) -> Result<Option<f32>, DripstoneError> {
        let state = self.state.read();
        let Some(&idx) = state.id_to_idx.get(doc_id) else {
            return Ok(None);
        };
        let doc = &state.docs[idx as usize];
        if doc.multi_vectors.is_empty() {
            return Ok(None);
        }
        Ok(Some(maxsim_score(
            query_tokens,
            &doc.multi_vectors,
            self.config.metric,
        )?))
    }
}

fn dense_metric(metric: Metric) -> Result<DenseMetric, DripstoneError> {
    match metric {
        Metric::Cosine => Ok(DenseMetric::Cosine),
        Metric::DotProduct => Ok(DenseMetric::Dot),
        Metric::Euclidean | Metric::Manhattan => Err(DripstoneError::pipeline(
            "PLAID codebook assignment only supports Cosine or DotProduct",
        )),
    }
}

fn sample_centroids(tokens: &[Vec<f32>], k: usize, dim: usize) -> Vec<Vec<f32>> {
    if tokens.is_empty() || dim == 0 {
        return vec![vec![0.0; dim.max(1)]; 1];
    }
    let k = k.min(tokens.len()).max(1);
    let step = (tokens.len() / k).max(1);
    let mut centroids = Vec::with_capacity(k);
    for i in 0..k {
        centroids.push(tokens[(i * step) % tokens.len()].clone());
    }
    centroids
}

fn refine_centroids(
    centroids: &mut [Vec<f32>],
    tokens: &[Vec<f32>],
    metric: Metric,
    iters: usize,
) -> Result<(), DripstoneError> {
    if centroids.is_empty() || tokens.is_empty() {
        return Ok(());
    }
    let dim = centroids[0].len();
    for _ in 0..iters {
        let mut acc = vec![vec![0.0_f32; dim]; centroids.len()];
        let mut counts = vec![0_u32; centroids.len()];
        for token in tokens {
            let c = nearest_centroid(token, centroids, metric)?;
            for (a, b) in acc[c].iter_mut().zip(token.iter()) {
                *a += *b;
            }
            counts[c] += 1;
        }
        for (i, centroid) in centroids.iter_mut().enumerate() {
            if counts[i] == 0 {
                continue;
            }
            let n = counts[i] as f32;
            for (a, b) in centroid.iter_mut().zip(acc[i].iter()) {
                *a = *b / n;
            }
        }
    }
    Ok(())
}

fn nearest_centroid(
    token: &[f32],
    centroids: &[Vec<f32>],
    metric: Metric,
) -> Result<usize, DripstoneError> {
    let tops = top_centroids(token, centroids, metric, 1)?;
    Ok(tops.first().copied().unwrap_or(0))
}

fn top_centroids(
    token: &[f32],
    centroids: &[Vec<f32>],
    metric: Metric,
    k: usize,
) -> Result<Vec<usize>, DripstoneError> {
    if centroids.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    let dense = dense_metric(metric)?;
    let prepared = PreparedQueryScorer::new(token, dense)?;
    let refs: Vec<&[f32]> = centroids.iter().map(Vec::as_slice).collect();
    let scores = prepared.score_dense_batch(&refs)?;
    let mut indexed: Vec<(usize, f32)> = scores.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    Ok(indexed.into_iter().take(k).map(|(i, _)| i).collect())
}

fn remove_doc_from_postings(state: &mut PlaidState, idx: u32) {
    for list in &mut state.postings {
        list.retain(|&d| d != idx);
    }
}

fn assign_doc_to_postings(
    state: &mut PlaidState,
    idx: u32,
    metric: Metric,
) -> Result<(), DripstoneError> {
    let doc = &state.docs[idx as usize];
    let mut seen = HashSet::new();
    for token in doc.multi_vectors.iter() {
        let c = nearest_centroid(token, &state.centroids, metric)?;
        if seen.insert(c) {
            state.postings[c].push(idx);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::{MaxSimOptions, multi_vectors_from_owned, pooled_vector_from_multi};
    use serde_json::json;

    fn chunk(doc: &str, tokens: Vec<Vec<f32>>) -> Chunk {
        let mv = multi_vectors_from_owned(tokens);
        let pooled = pooled_vector_from_multi(&mv);
        Chunk::new(
            doc,
            0,
            format!("c-{doc}"),
            pooled,
            0,
            1,
            1,
            json!({}),
            Some("ns".into()),
        )
        .with_multi_vectors(mv)
    }

    #[test]
    fn plaid_ranks_aligned_document_first() {
        let index = PlaidIndex::new(PlaidConfig {
            num_centroids: 8,
            probes: 4,
            candidate_budget: 32,
            metric: Metric::DotProduct,
        });
        let chunks = vec![
            chunk("good", vec![vec![1.0, 0.0], vec![0.0, 1.0]]),
            chunk("bad", vec![vec![-1.0, 0.0], vec![0.0, -1.0]]),
            chunk("mid", vec![vec![0.5, 0.5], vec![0.5, 0.5]]),
        ];
        index.rebuild(&chunks).expect("rebuild");

        let q = multi_vectors_from_owned(vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let query = SearchQuery::new(pooled_vector_from_multi(&q))
            .limit(1)
            .namespace("ns")
            .with_multi_vectors(q)
            .with_maxsim(MaxSimOptions::new(Metric::DotProduct));
        let hits = index.search(&query).expect("search");
        assert_eq!(hits[0].chunk.id, "good#0");
    }
}
