use std::collections::HashMap;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::core::{Chunk, FullTextQuery};
use crate::errors::DripstoneError;
use crate::fts::codec;
use crate::fts::maxscore::{MaxScoreRequest, MaxScoreTerm, execute as execute_maxscore};
use crate::fts::postings::{decode_postings, localize_blocks, score_decoded_postings};
use crate::fts::{
    DocId, FullTextHit, LexiconEntry, Posting, PostingBlock, TermId, TextDocumentMeta,
};
use crate::search::{EqualityTerm, NativeFilterPlan, SimpleTokenizer, Tokenizer};
use crate::utils::numeric::{u32_to_f32, usize_to_f32};

type TermOccurrences = HashMap<String, Vec<(DocId, u32, u32)>>;

#[async_trait]
pub trait FullTextIndex: Send + Sync {
    async fn build(&self, chunks: &[Chunk]) -> Result<(), DripstoneError>;
    async fn add_chunk(&self, chunk: Chunk) -> Result<(), DripstoneError>;
    async fn remove_chunk(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError>;
    async fn search_maxscore(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<FullTextHit>, DripstoneError>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clear(&self);
    fn is_dirty(&self) -> bool {
        false
    }
    async fn load_snapshot(&self, _path: &str) -> Result<bool, DripstoneError> {
        Ok(false)
    }
    async fn flush_snapshot(&self, _path: &str) -> Result<(), DripstoneError> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FullTextIndexConfig {
    pub tokenizer: SimpleTokenizer,
    pub block_size: usize,
}

impl Default for FullTextIndexConfig {
    fn default() -> Self {
        Self {
            tokenizer: SimpleTokenizer::new().with_english_stopwords(),
            block_size: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SourceDocument {
    chunk_id: String,
    namespace: Option<String>,
    text: String,
    metadata_summary: serde_json::Value,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub(crate) struct InMemoryFullTextState {
    documents: HashMap<String, SourceDocument>,
    doc_metas: Vec<TextDocumentMeta>,
    term_to_id: HashMap<String, TermId>,
    lexicon: HashMap<TermId, LexiconEntry>,
    postings: Vec<Posting>,
    block_index: Vec<PostingBlock>,
    avg_doc_len: f32,
    dirty: bool,
}

#[derive(Debug)]
pub struct InMemoryFullTextIndex {
    config: FullTextIndexConfig,
    state: RwLock<InMemoryFullTextState>,
}

impl InMemoryFullTextIndex {
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(FullTextIndexConfig::default())
    }

    #[must_use]
    pub fn with_config(config: FullTextIndexConfig) -> Self {
        Self {
            config,
            state: RwLock::new(InMemoryFullTextState::default()),
        }
    }

    fn composite_key(id: &str, namespace: Option<&str>) -> String {
        let namespace = namespace.unwrap_or("default");
        format!("{namespace}:{id}")
    }

    fn build_options(&self) -> crate::core::FullTextOptions {
        crate::core::FullTextOptions {
            block_size: self.config.block_size,
            ..crate::core::FullTextOptions::default()
        }
    }

    /// Replace the indexed documents.
    ///
    /// [`crate::Dripstone::open`] calls this while constructing the store,
    /// before a runtime can poll [`FullTextIndex::build`].
    pub(crate) fn replace_documents(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        let mut state = self.state.write();
        state.documents.clear();
        for chunk in chunks {
            state.documents.insert(
                Self::composite_key(&chunk.id, chunk.namespace.as_deref()),
                SourceDocument {
                    chunk_id: chunk.id.clone(),
                    namespace: chunk.namespace.clone(),
                    text: chunk.content.clone(),
                    metadata_summary: chunk.metadata.clone(),
                },
            );
        }
        state.dirty = true;
        self.rebuild_locked(&mut state, &self.build_options())?;
        drop(state);
        Ok(())
    }

    fn ensure_built(&self) -> Result<(), DripstoneError> {
        if !self.state.read().dirty {
            return Ok(());
        }

        let mut state = self.state.write();
        if state.dirty {
            self.rebuild_locked(&mut state, &self.build_options())?;
        }
        drop(state);
        Ok(())
    }

    fn snapshot_state(&self) -> Result<InMemoryFullTextState, DripstoneError> {
        self.ensure_built()?;
        Ok(self.state.read().clone())
    }

    fn rebuild_locked(
        &self,
        state: &mut InMemoryFullTextState,
        options: &crate::core::FullTextOptions,
    ) -> Result<(), DripstoneError> {
        state.doc_metas.clear();
        state.term_to_id.clear();
        state.lexicon.clear();
        state.postings.clear();
        state.block_index.clear();
        state.avg_doc_len = 0.0;
        state.dirty = false;

        if state.documents.is_empty() {
            return Ok(());
        }

        let mut documents: Vec<_> = state.documents.values().cloned().collect();
        documents.sort_by(|left, right| left.chunk_id.cmp(&right.chunk_id));

        let term_occurrences = self.collect_term_occurrences(state, &documents)?;
        let mut terms: Vec<_> = term_occurrences.into_iter().collect();
        terms.sort_by(|left, right| left.0.cmp(&right.0));

        for (term_idx, (term, mut postings_for_term)) in terms.into_iter().enumerate() {
            postings_for_term.sort_by_key(|(doc_id, _, _)| *doc_id);
            let term_id = TermId::try_from(term_idx)
                .map_err(|_| DripstoneError::storage("full-text term count exceeds u32"))?;
            state.term_to_id.insert(term, term_id);

            let postings_offset = state.postings.len();
            let block_offset = state.block_index.len();
            let doc_freq = postings_for_term.len();
            let mut prev_doc_id = 0_u32;
            let mut max_term_score = 0.0_f32;

            for block in postings_for_term.chunks(options.block_size.max(1)) {
                let block_postings_offset = state.postings.len();
                let mut block_max_score = 0.0_f32;
                let mut last_doc_id = block.first().map_or(0, |(doc_id, _, _)| *doc_id);

                for (doc_id, term_freq, doc_len) in block {
                    let delta = doc_id.saturating_sub(prev_doc_id);
                    prev_doc_id = *doc_id;
                    last_doc_id = *doc_id;
                    state.postings.push(Posting {
                        doc_id_delta: delta,
                        term_freq: *term_freq,
                    });

                    let score = bm25_score(
                        state.doc_metas.len(),
                        doc_freq,
                        *term_freq,
                        *doc_len,
                        state.avg_doc_len,
                        options.bm25_k1,
                        options.bm25_b,
                    );
                    max_term_score = max_term_score.max(score);
                    block_max_score = block_max_score.max(score);
                }

                state.block_index.push(PostingBlock {
                    first_doc_id: block.first().map_or(0, |(doc_id, _, _)| *doc_id),
                    last_doc_id,
                    block_len: block.len(),
                    block_max_score,
                    postings_offset: block_postings_offset,
                    postings_len: block.len(),
                });
            }

            state.lexicon.insert(
                term_id,
                LexiconEntry {
                    term_id,
                    doc_freq,
                    postings_offset,
                    postings_len: state.postings.len().saturating_sub(postings_offset),
                    block_offset,
                    block_len: state.block_index.len().saturating_sub(block_offset),
                    max_term_score,
                },
            );
        }
        Ok(())
    }

    fn collect_term_occurrences(
        &self,
        state: &mut InMemoryFullTextState,
        documents: &[SourceDocument],
    ) -> Result<TermOccurrences, DripstoneError> {
        let mut term_occurrences = TermOccurrences::new();
        let mut total_doc_len = 0_usize;
        for (doc_idx, document) in documents.iter().enumerate() {
            let doc_id = DocId::try_from(doc_idx)
                .map_err(|_| DripstoneError::storage("full-text document count exceeds u32"))?;
            let tokens = self.config.tokenizer.tokenize(&document.text);
            let doc_len = u32::try_from(tokens.len())
                .map_err(|_| DripstoneError::storage("full-text document length exceeds u32"))?;
            total_doc_len = total_doc_len
                .checked_add(usize::try_from(doc_len).unwrap_or(usize::MAX))
                .ok_or_else(|| DripstoneError::storage("full-text corpus length overflow"))?;
            let mut term_freqs: HashMap<String, u32> = HashMap::new();
            for token in tokens {
                *term_freqs.entry(token).or_insert(0) += 1;
            }
            state.doc_metas.push(TextDocumentMeta {
                doc_id,
                chunk_id: document.chunk_id.clone(),
                namespace: document.namespace.clone(),
                doc_len,
                metadata_summary: document.metadata_summary.clone(),
                deleted: false,
            });
            for (term, term_freq) in term_freqs {
                term_occurrences
                    .entry(term)
                    .or_default()
                    .push((doc_id, term_freq, doc_len));
            }
        }
        state.avg_doc_len = usize_to_f32(total_doc_len) / usize_to_f32(documents.len());
        Ok(term_occurrences)
    }

    fn query_terms(
        &self,
        state: &InMemoryFullTextState,
        query: &FullTextQuery,
    ) -> Result<Vec<MaxScoreTerm>, DripstoneError> {
        #[cfg(feature = "search-advanced")]
        let tokens = {
            let mut tokens = self.config.tokenizer.tokenize(&query.text);
            if query.options.query_expansion {
                let expanded = crate::search::expand_query_for_fts(&query.text);
                tokens.extend(expanded.keywords);
            }
            tokens
        };

        #[cfg(not(feature = "search-advanced"))]
        let tokens = self.config.tokenizer.tokenize(&query.text);

        let mut seen = std::collections::HashSet::new();
        let mut unique_terms = Vec::new();
        for token in tokens {
            if seen.insert(token.clone()) {
                unique_terms.push(token);
            }
        }

        if let Some(max_query_terms) = query.options.max_query_terms {
            unique_terms.truncate(max_query_terms);
        }

        let mut max_score_terms = Vec::with_capacity(unique_terms.len());
        for term in unique_terms {
            let Some(term_id) = state.term_to_id.get(&term).copied() else {
                continue;
            };
            let Some(entry) = state.lexicon.get(&term_id) else {
                continue;
            };
            let decoded = decode_postings(
                &state.postings[entry.postings_offset..entry.postings_offset + entry.postings_len],
            )?;
            let postings = score_decoded_postings(decoded, |doc_id, term_freq| {
                let doc_len = state
                    .doc_metas
                    .get(doc_id as usize)
                    .map(|meta| meta.doc_len)
                    .unwrap_or_default();
                bm25_score(
                    state.doc_metas.len(),
                    entry.doc_freq,
                    term_freq,
                    doc_len,
                    state.avg_doc_len,
                    query.options.bm25_k1,
                    query.options.bm25_b,
                )
            });
            let blocks = localize_blocks(
                &state.block_index[entry.block_offset..entry.block_offset + entry.block_len],
                entry.postings_offset,
            );
            max_score_terms.push(MaxScoreTerm {
                term_id,
                max_term_score: postings
                    .iter()
                    .map(|posting| posting.score)
                    .fold(0.0_f32, f32::max),
                postings,
                blocks,
            });
        }
        Ok(max_score_terms)
    }

    fn allowed_docs(
        state: &InMemoryFullTextState,
        query: &FullTextQuery,
    ) -> Option<std::collections::HashSet<DocId>> {
        let plan = crate::search::compile_native_filter_plan(
            query.namespace.as_deref(),
            query.filter.as_ref(),
        );
        match plan {
            NativeFilterPlan::MatchAll | NativeFilterPlan::Unsupported => None,
            NativeFilterPlan::EqualityConjunction(plan) => {
                Some(Self::allowed_docs_for_plan(state, &plan))
            }
            NativeFilterPlan::Disjunction(plans) => {
                let mut allowed_docs = std::collections::HashSet::new();
                for plan in &plans {
                    allowed_docs.extend(Self::allowed_docs_for_plan(state, plan));
                }
                Some(allowed_docs)
            }
        }
    }

    fn allowed_docs_for_plan(
        state: &InMemoryFullTextState,
        plan: &crate::search::EqualityConjunctionPlan,
    ) -> std::collections::HashSet<DocId> {
        state
            .doc_metas
            .iter()
            .filter(|meta| {
                if meta.deleted {
                    return false;
                }
                if plan.namespace.as_deref().is_some_and(|namespace| {
                    meta.namespace.as_deref().unwrap_or("default") != namespace
                }) {
                    return false;
                }
                plan.terms.iter().all(|term| match term {
                    EqualityTerm::Equal { field, value } => meta
                        .metadata_summary
                        .pointer(&json_pointer(field))
                        .is_some_and(|candidate| candidate == value),
                    EqualityTerm::In { field, values } => meta
                        .metadata_summary
                        .pointer(&json_pointer(field))
                        .is_some_and(|candidate| values.contains(candidate)),
                })
            })
            .map(|meta| meta.doc_id)
            .collect()
    }

    fn post_filter_hits(
        state: &InMemoryFullTextState,
        query: &FullTextQuery,
        hits: Vec<FullTextHit>,
    ) -> Vec<FullTextHit> {
        hits.into_iter()
            .filter(|hit| {
                state
                    .doc_metas
                    .get(hit.doc_id as usize)
                    .is_some_and(|meta| {
                        if meta.deleted {
                            return false;
                        }
                        if query.namespace.as_deref().is_some_and(|namespace| {
                            meta.namespace.as_deref().unwrap_or("default") != namespace
                        }) {
                            return false;
                        }
                        query
                            .filter
                            .as_ref()
                            .is_none_or(|filter| filter.matches(&meta.metadata_summary))
                    })
            })
            .collect()
    }
}

impl Default for InMemoryFullTextIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl FullTextIndex for InMemoryFullTextIndex {
    async fn build(&self, chunks: &[Chunk]) -> Result<(), DripstoneError> {
        self.replace_documents(chunks)
    }

    async fn add_chunk(&self, chunk: Chunk) -> Result<(), DripstoneError> {
        let mut state = self.state.write();
        state.documents.insert(
            Self::composite_key(&chunk.id, chunk.namespace.as_deref()),
            SourceDocument {
                chunk_id: chunk.id,
                namespace: chunk.namespace,
                text: chunk.content,
                metadata_summary: chunk.metadata,
            },
        );
        state.dirty = true;
        drop(state);
        Ok(())
    }

    async fn remove_chunk(&self, id: &str, namespace: Option<&str>) -> Result<(), DripstoneError> {
        let mut state = self.state.write();
        state.documents.remove(&Self::composite_key(id, namespace));
        state.dirty = true;
        drop(state);
        Ok(())
    }

    async fn search_maxscore(
        &self,
        query: &FullTextQuery,
    ) -> Result<Vec<FullTextHit>, DripstoneError> {
        self.ensure_built()?;
        let state = self.state.read();
        if state.doc_metas.is_empty() {
            return Ok(Vec::new());
        }
        let terms = self.query_terms(&state, query)?;
        if terms.is_empty() {
            return Ok(Vec::new());
        }

        let allowed_docs = Self::allowed_docs(&state, query);
        let mut hits = execute_maxscore(MaxScoreRequest {
            limit: query.limit,
            documents: &state.doc_metas,
            terms,
            allowed_docs: allowed_docs.as_ref(),
        });

        hits = Self::post_filter_hits(&state, query, hits);
        if hits.len() > query.limit {
            hits.truncate(query.limit);
        }
        drop(state);
        Ok(hits)
    }

    fn len(&self) -> usize {
        self.state.read().documents.len()
    }

    fn clear(&self) {
        *self.state.write() = InMemoryFullTextState::default();
    }

    fn is_dirty(&self) -> bool {
        self.state.read().dirty
    }

    async fn load_snapshot(&self, path: &str) -> Result<bool, DripstoneError> {
        if !std::path::Path::new(path).exists() {
            return Ok(false);
        }

        let bytes = std::fs::read(path)
            .map_err(|e| DripstoneError::storage_with_source("fts snapshot read", e))?;
        let state = codec::decode_snapshot_v1(&bytes)?;
        *self.state.write() = state;
        Ok(true)
    }

    async fn flush_snapshot(&self, path: &str) -> Result<(), DripstoneError> {
        let snapshot = self.snapshot_state()?;
        let bytes = codec::encode_snapshot_v1(&snapshot)?;
        if let Some(parent) = std::path::Path::new(path).parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| DripstoneError::storage_with_source("fts snapshot parent dir", e))?;
        }
        std::fs::write(path, bytes)
            .map_err(|e| DripstoneError::storage_with_source("fts snapshot write", e))
    }
}

fn bm25_score(
    num_docs: usize,
    doc_freq: usize,
    term_freq: u32,
    doc_len: u32,
    avg_doc_len: f32,
    k1: f32,
    b: f32,
) -> f32 {
    if num_docs == 0 || avg_doc_len <= f32::EPSILON || doc_freq == 0 {
        return 0.0;
    }

    let numerator = usize_to_f32(num_docs.saturating_sub(doc_freq)) + 0.5;
    let denominator = usize_to_f32(doc_freq) + 0.5;
    let idf = (numerator / denominator).ln_1p();
    let doc_norm = b.mul_add(u32_to_f32(doc_len) / avg_doc_len, 1.0 - b);
    let tf = u32_to_f32(term_freq);
    let normalized_tf = tf / doc_norm.max(f32::EPSILON);
    idf * (normalized_tf * (k1 + 1.0)) / (normalized_tf + k1)
}

fn json_pointer(field: &str) -> String {
    let mut pointer = String::new();
    for part in field.split('.') {
        pointer.push('/');
        pointer.push_str(part);
    }
    pointer
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::Filter;

    fn chunk(
        doc_id: &str,
        chunk_index: usize,
        content: &str,
        namespace: Option<&str>,
        metadata: serde_json::Value,
    ) -> Chunk {
        Chunk::new(
            doc_id,
            chunk_index,
            content,
            vec![1.0, 0.0],
            0,
            content.len(),
            1,
            metadata,
            namespace.map(ToOwned::to_owned),
        )
    }

    #[tokio::test]
    async fn in_memory_full_text_index_returns_ranked_hits() {
        let index = InMemoryFullTextIndex::new();
        let chunks = vec![
            chunk(
                "doc-a",
                0,
                "rust vector search and bm25 scoring",
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
            chunk(
                "doc-b",
                0,
                "rust rust rust full text retrieval",
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
            chunk(
                "doc-c",
                0,
                "python embeddings only",
                Some("notes"),
                serde_json::json!({"kind": "note"}),
            ),
        ];

        index.build(&chunks).await.expect("build should succeed");
        let hits = index
            .search_maxscore(&FullTextQuery::new("rust retrieval").limit(2))
            .await
            .expect("search should succeed");

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].chunk_id, "doc-b#0");
        assert!(hits[0].score >= hits[1].score);
    }

    #[tokio::test]
    async fn in_memory_full_text_index_applies_namespace_and_metadata_filters() {
        let index = InMemoryFullTextIndex::new();
        let chunks = vec![
            chunk(
                "doc-a",
                0,
                "rust vector search and bm25 scoring",
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
            chunk(
                "doc-b",
                0,
                "rust vector search examples",
                Some("docs"),
                serde_json::json!({"kind": "example"}),
            ),
            chunk(
                "doc-c",
                0,
                "rust vector search in another namespace",
                Some("notes"),
                serde_json::json!({"kind": "guide"}),
            ),
        ];

        index.build(&chunks).await.expect("build should succeed");
        let hits = index
            .search_maxscore(
                &FullTextQuery::new("rust search")
                    .namespace("docs")
                    .filter(Filter::Equal {
                        field: "kind".to_string(),
                        value: serde_json::json!("guide"),
                    })
                    .limit(5),
            )
            .await
            .expect("search should succeed");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, "doc-a#0");
    }

    #[tokio::test]
    async fn in_memory_full_text_index_removes_deleted_chunk() {
        let index = InMemoryFullTextIndex::new();
        let chunks = vec![
            chunk(
                "doc-a",
                0,
                "rust vector search and bm25 scoring",
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
            chunk(
                "doc-b",
                0,
                "rust rust rust full text retrieval",
                Some("docs"),
                serde_json::json!({"kind": "guide"}),
            ),
        ];

        index.build(&chunks).await.expect("build should succeed");
        index
            .remove_chunk("doc-b#0", Some("docs"))
            .await
            .expect("remove should succeed");

        let hits = index
            .search_maxscore(&FullTextQuery::new("retrieval rust").limit(5))
            .await
            .expect("search should succeed");

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].chunk_id, "doc-a#0");
    }
}
