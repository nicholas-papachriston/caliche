use std::collections::HashMap;

use super::{
    Backend, Chunk, ChunkText, Dripstone, DripstoneError, Embedder, IngestBuilder,
    IngestChunksBuilder, IngestResult, PreChunk, SentenceData, Serialize, cosine_similarity,
};
use crate::search::{multi_vectors_from_owned, tokens_for_late_interaction};

struct IngestHashPlan {
    skip_mask: Vec<bool>,
    hashes: std::collections::HashMap<usize, String>,
}

impl<B: Backend, E: Embedder> Dripstone<B, E> {
    async fn plan_ingest_hashes(
        &self,
        doc_id: &str,
        namespace: Option<&str>,
        chunks: &[ChunkText],
    ) -> Result<IngestHashPlan, DripstoneError> {
        let options = self.config.hashline.to_options();
        let mut plan = IngestHashPlan {
            skip_mask: vec![false; chunks.len()],
            hashes: std::collections::HashMap::new(),
        };
        if self.config.hashline.enabled && self.config.hashline.incremental_indexing {
            if let Some(store) = self.chunk_hash_store.as_ref() {
                let existing = store.get_doc_chunk_hashes(doc_id, namespace).await?;
                for (idx, chunk) in chunks.iter().enumerate() {
                    let hash = crate::hashline::compute_chunk_hash(&chunk.text, &options);
                    plan.skip_mask[idx] =
                        existing.get(&idx).is_some_and(|existing| existing == &hash);
                    plan.hashes.insert(idx, hash);
                }
            }
        } else if self.config.hashline.enabled {
            for (idx, chunk) in chunks.iter().enumerate() {
                plan.hashes.insert(
                    idx,
                    crate::hashline::compute_chunk_hash(&chunk.text, &options),
                );
            }
        }
        Ok(plan)
    }

    pub(super) async fn execute_ingest(
        &self,
        doc_id: &str,
        content: &str,
        chunks_text: Vec<ChunkText>,
        namespace: Option<String>,
        metadata: serde_json::Value,
    ) -> Result<IngestResult, DripstoneError> {
        if chunks_text.is_empty() {
            return Ok(IngestResult { chunks_created: 0 });
        }

        let total_chunks = chunks_text.len();
        let options = self.config.hashline.to_options();
        let hash_plan = self
            .plan_ingest_hashes(doc_id, namespace.as_deref(), &chunks_text)
            .await?;

        #[cfg(feature = "rust-stemmers")]
        let texts_for_bm25: Vec<String> = chunks_text.iter().map(|c| c.text.clone()).collect();

        let to_embed: Vec<(usize, ChunkText)> = chunks_text
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| !hash_plan.skip_mask[*idx])
            .collect();

        let texts: Vec<String> = to_embed.iter().map(|(_, c)| c.text.clone()).collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();
        let embeddings = self.embedder.embed(&text_refs).await?;

        let mut chunks: Vec<Chunk> = to_embed
            .into_iter()
            .zip(embeddings.into_iter())
            .map(|((idx, chunk_text), vector)| {
                let chunk_hash = hash_plan.hashes.get(&idx).cloned().unwrap_or_default();
                let mut chunk_metadata = metadata.clone();
                if let Some(extra) = chunk_text
                    .metadata
                    .as_ref()
                    .and_then(serde_json::Value::as_object)
                {
                    if let Some(object) = chunk_metadata.as_object_mut() {
                        for (key, value) in extra {
                            object.insert(key.clone(), value.clone());
                        }
                    }
                }
                if self.config.hashline.enabled {
                    let mut hashline_val = serde_json::json!({ "chunk_hash": chunk_hash });
                    if self.config.hashline.line_level {
                        let start_line = crate::hashline::line_number_at_offset(
                            content,
                            chunk_text.start_offset,
                        );
                        let line_hashes =
                            crate::hashline::compute_line_hashes(&chunk_text.text, &options);
                        let line_hashes_json: Vec<serde_json::Value> = line_hashes
                            .into_iter()
                            .map(|(ln, h)| serde_json::json!([ln, h]))
                            .collect();
                        hashline_val["start_line"] = serde_json::json!(start_line);
                        hashline_val["line_hashes"] = serde_json::json!(line_hashes_json);
                    }
                    if let Some(obj) = chunk_metadata.as_object_mut() {
                        obj.insert("hashline".into(), hashline_val);
                    }
                }
                Chunk::new(
                    doc_id,
                    idx,
                    chunk_text.text,
                    vector,
                    chunk_text.start_offset,
                    chunk_text.end_offset,
                    total_chunks,
                    chunk_metadata,
                    namespace.clone(),
                )
            })
            .collect();

        if self.config.search.late_interaction {
            self.attach_late_interaction_multi_vectors(&mut chunks)
                .await?;
        }

        let count = chunks.len();

        if !chunks.is_empty() {
            self.backend.upsert(&chunks).await?;
        }

        if self.config.hashline.enabled
            && let Some(store) = self.chunk_hash_store.as_ref()
        {
            store
                .set_doc_chunk_hashes(doc_id, namespace.as_deref(), hash_plan.hashes)
                .await?;
        }

        #[cfg(feature = "rust-stemmers")]
        if let Some(ref helper) = self.hybrid_helper {
            if self.config.hashline.line_level_bm25 {
                let chunk_lines: Vec<(String, Vec<String>)> = texts_for_bm25
                    .iter()
                    .enumerate()
                    .map(|(idx, text)| {
                        let chunk_id = format!("{doc_id}#{idx}");
                        let lines: Vec<String> = text
                            .split('\n')
                            .map(std::string::ToString::to_string)
                            .collect();
                        (chunk_id, lines)
                    })
                    .collect();
                helper.fit_lines(&chunk_lines);
            } else {
                helper.fit_dedup(&texts_for_bm25, self.config.hashline.bm25_dedup);
            }
        }

        Ok(IngestResult {
            chunks_created: count,
        })
    }

    async fn attach_late_interaction_multi_vectors(
        &self,
        chunks: &mut [Chunk],
    ) -> Result<(), DripstoneError> {
        let max_tokens = self.config.search.max_token_vectors.max(1);
        let mut unique = Vec::new();
        let mut index_of: HashMap<String, usize> = HashMap::new();
        let mut chunk_token_idx: Vec<Vec<usize>> = Vec::with_capacity(chunks.len());

        for chunk in chunks.iter() {
            let tokens = tokens_for_late_interaction(&chunk.content, max_tokens);
            let mut idxs = Vec::with_capacity(tokens.len());
            for token in tokens {
                if let Some(&existing) = index_of.get(&token) {
                    idxs.push(existing);
                } else {
                    let idx = unique.len();
                    index_of.insert(token.clone(), idx);
                    unique.push(token);
                    idxs.push(idx);
                }
            }
            chunk_token_idx.push(idxs);
        }

        if unique.is_empty() {
            return Ok(());
        }

        let refs: Vec<&str> = unique.iter().map(std::string::String::as_str).collect();
        let vectors = self.embedder.embed(&refs).await?;
        if vectors.len() != unique.len() {
            return Err(DripstoneError::pipeline(
                "late-interaction token embed count mismatch",
            ));
        }

        for (chunk, idxs) in chunks.iter_mut().zip(chunk_token_idx) {
            let owned: Vec<Vec<f32>> = idxs.into_iter().map(|i| vectors[i].clone()).collect();
            if !owned.is_empty() {
                chunk.multi_vectors = Some(multi_vectors_from_owned(owned));
            }
        }
        Ok(())
    }
}

impl<'a, B: Backend, E: Embedder> IngestChunksBuilder<'a, B, E> {
    pub(super) fn new(
        dripstone: &'a Dripstone<B, E>,
        doc_id: &'a str,
        content: &'a str,
        chunks: Vec<PreChunk>,
    ) -> Self {
        Self {
            dripstone,
            doc_id,
            content,
            chunks,
            namespace: None,
            metadata: serde_json::json!({}),
        }
    }

    #[must_use]
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    #[must_use]
    pub fn metadata<M: Serialize>(mut self, metadata: M) -> Self {
        match serde_json::to_value(metadata) {
            Ok(value) => self.metadata = value,
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "failed to serialize ingest chunk metadata; keeping existing metadata"
                );
            }
        }
        self
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn try_metadata<M: Serialize>(mut self, metadata: M) -> Result<Self, DripstoneError> {
        self.metadata = serde_json::to_value(metadata)?;
        Ok(self)
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn await_result(self) -> Result<IngestResult, DripstoneError> {
        let chunks_text: Vec<ChunkText> = self
            .chunks
            .into_iter()
            .map(|c| ChunkText::new(c.text, c.start_offset, c.end_offset))
            .collect();

        self.dripstone
            .execute_ingest(
                self.doc_id,
                self.content,
                chunks_text,
                self.namespace,
                self.metadata,
            )
            .await
    }
}

impl<'a, B: Backend, E: Embedder> std::future::IntoFuture for IngestChunksBuilder<'a, B, E> {
    type Output = Result<IngestResult, DripstoneError>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.await_result())
    }
}

impl<'a, B: Backend, E: Embedder> IngestBuilder<'a, B, E> {
    pub(super) fn new(dripstone: &'a Dripstone<B, E>, doc_id: &'a str, content: &'a str) -> Self {
        Self {
            dripstone,
            doc_id,
            content,
            namespace: None,
            metadata: serde_json::json!({}),
        }
    }

    #[must_use]
    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    #[must_use]
    pub fn metadata<M: Serialize>(mut self, metadata: M) -> Self {
        match serde_json::to_value(metadata) {
            Ok(value) => self.metadata = value,
            Err(error) => {
                tracing::warn!(
                    ?error,
                    "failed to serialize ingest metadata; keeping existing metadata"
                );
            }
        }
        self
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub fn try_metadata<M: Serialize>(mut self, metadata: M) -> Result<Self, DripstoneError> {
        self.metadata = serde_json::to_value(metadata)?;
        Ok(self)
    }

    ///
    /// # Errors
    ///
    /// Returns an error when the operation cannot be completed.
    pub async fn await_result(self) -> Result<IngestResult, DripstoneError> {
        let chunks_text = self.chunk_document().await;

        self.dripstone
            .execute_ingest(
                self.doc_id,
                self.content,
                chunks_text,
                self.namespace,
                self.metadata,
            )
            .await
    }

    async fn chunk_document(&self) -> Vec<ChunkText> {
        use crate::config::ChunkStrategy;

        match self.dripstone.config.chunking.strategy {
            ChunkStrategy::None => {
                vec![ChunkText::new(
                    self.content.to_string(),
                    0,
                    self.content.len(),
                )]
            }
            ChunkStrategy::Sentence | ChunkStrategy::Paragraph => self.chunk_by_sentences(),
            ChunkStrategy::AstAware => self.chunk_ast_aware(),
            ChunkStrategy::FixedTokens => self.chunk_fixed_size(),
            ChunkStrategy::Semantic => self.chunk_semantic().await,
        }
    }

    fn chunk_ast_aware(&self) -> Vec<ChunkText> {
        let language = json_string(&self.metadata, crate::rag::ast::META_LANGUAGE)
            .or_else(|| self.dripstone.config.chunking.code_language.clone());
        let path = json_string(&self.metadata, crate::rag::ast::META_PATH)
            .or_else(|| json_string(&self.metadata, crate::rag::ast::META_FILE));
        let mut chunker = crate::rag::ast::AstChunker::new();
        match chunker.chunk(self.content, language.as_deref(), path.as_deref()) {
            crate::rag::ast::AstChunking::Parsed(chunks) if !chunks.is_empty() => chunks
                .into_iter()
                .map(|chunk| {
                    let metadata = chunk.metadata();
                    ChunkText::new(chunk.text, chunk.start_offset, chunk.end_offset)
                        .with_metadata(metadata)
                })
                .collect(),
            crate::rag::ast::AstChunking::Parsed(_) | crate::rag::ast::AstChunking::Unavailable => {
                self.chunk_by_sentences()
            }
        }
    }

    fn chunk_by_sentences(&self) -> Vec<ChunkText> {
        let config = &self.dripstone.config.chunking;
        let target_chars = config.target_tokens * 4;
        let overlap_chars = config.overlap_tokens * 4;
        let min_chars = config.min_tokens * 4;

        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut chunk_start = 0;
        let mut last_sentence_end = 0;
        let sentences: Vec<&str> = self.content.split_inclusive(['.', '!', '?']).collect();

        let mut offset = 0;
        for sentence in sentences {
            let sentence_len = sentence.len();

            if current_chunk.len() + sentence_len > target_chars && !current_chunk.is_empty() {
                chunks.push(ChunkText::new(
                    current_chunk.trim().to_string(),
                    chunk_start,
                    offset,
                ));

                if overlap_chars > 0 && last_sentence_end > chunk_start {
                    let overlap_start = offset.saturating_sub(overlap_chars);
                    let overlap_start = overlap_start.min(offset).min(self.content.len());
                    let overlap_start = self.content.floor_char_boundary(overlap_start);
                    current_chunk = self.content[overlap_start..offset].to_string();
                    chunk_start = overlap_start;
                } else {
                    current_chunk = String::new();
                    chunk_start = offset;
                }
            }

            current_chunk.push_str(sentence);
            offset += sentence_len;
            last_sentence_end = offset;
        }

        if !current_chunk.is_empty() && current_chunk.trim().len() >= min_chars {
            chunks.push(ChunkText::new(
                current_chunk.trim().to_string(),
                chunk_start,
                self.content.len(),
            ));
        } else if !chunks.is_empty() && !current_chunk.is_empty() {
            if let Some(last) = chunks.last_mut() {
                last.text.push_str(&current_chunk);
                last.end_offset = self.content.len();
            }
        } else if chunks.is_empty() && !current_chunk.is_empty() {
            chunks.push(ChunkText::new(
                current_chunk.trim().to_string(),
                0,
                self.content.len(),
            ));
        }

        chunks
    }

    async fn chunk_semantic(&self) -> Vec<ChunkText> {
        let config = &self.dripstone.config.chunking;
        let min_chars = config.min_tokens * 4;
        let max_chars = config.max_tokens * 4;
        let similarity_threshold = config.semantic_threshold;
        let sentence_data = self.split_sentences_with_offsets();

        if sentence_data.is_empty() {
            return vec![];
        }

        if sentence_data.len() == 1 {
            return vec![ChunkText::new(
                sentence_data[0].text.clone(),
                sentence_data[0].start_offset,
                sentence_data[0].end_offset,
            )];
        }

        let sentence_texts: Vec<&str> = sentence_data.iter().map(|s| s.text.as_str()).collect();
        let Ok(embeddings) = self.dripstone.embedder.embed(&sentence_texts).await else {
            return self.chunk_by_sentences();
        };

        let mut chunks = Vec::new();
        let mut current_chunk_sentences = vec![0];
        let mut current_start = sentence_data[0].start_offset;

        for i in 1..sentence_data.len() {
            let similarity = cosine_similarity(&embeddings[i - 1], &embeddings[i]);
            let current_size: usize = current_chunk_sentences
                .iter()
                .map(|&idx| sentence_data[idx].text.len())
                .sum();
            let would_exceed_max = current_size + sentence_data[i].text.len() > max_chars;
            let similarity_drop = similarity < similarity_threshold;

            if (similarity_drop || would_exceed_max) && current_size >= min_chars {
                let chunk_text = current_chunk_sentences
                    .iter()
                    .map(|&idx| sentence_data[idx].text.as_str())
                    .collect::<Vec<_>>()
                    .join("");

                let chunk_end = current_chunk_sentences
                    .last()
                    .and_then(|&idx| sentence_data.get(idx))
                    .map_or(sentence_data[i].end_offset, |s| s.end_offset);

                chunks.push(ChunkText::new(chunk_text, current_start, chunk_end));

                current_chunk_sentences.clear();
                current_chunk_sentences.push(i);
                current_start = sentence_data[i].start_offset;
            } else {
                current_chunk_sentences.push(i);
            }
        }

        if !current_chunk_sentences.is_empty() {
            let chunk_text = current_chunk_sentences
                .iter()
                .map(|&idx| sentence_data[idx].text.as_str())
                .collect::<Vec<_>>()
                .join("");

            let chunk_end = current_chunk_sentences
                .last()
                .and_then(|&idx| sentence_data.get(idx))
                .map_or(self.content.len(), |s| s.end_offset);

            chunks.push(ChunkText::new(chunk_text, current_start, chunk_end));
        }

        let mut merged_chunks = Vec::new();
        let mut pending_chunk: Option<ChunkText> = None;

        for chunk in chunks {
            if chunk.text.len() < min_chars {
                if let Some(mut prev) = pending_chunk.take() {
                    prev.text.push_str(&chunk.text);
                    prev.end_offset = chunk.end_offset;
                    pending_chunk = Some(prev);
                } else {
                    pending_chunk = Some(chunk);
                }
            } else if let Some(prev) = pending_chunk.take() {
                let mut merged = prev;
                merged.text.push_str(&chunk.text);
                merged.end_offset = chunk.end_offset;
                merged_chunks.push(merged);
            } else {
                merged_chunks.push(chunk);
            }
        }

        if let Some(chunk) = pending_chunk {
            if merged_chunks.is_empty() {
                merged_chunks.push(chunk);
            } else if let Some(last) = merged_chunks.last_mut() {
                last.text.push_str(&chunk.text);
                last.end_offset = chunk.end_offset;
            }
        }

        merged_chunks
    }

    fn split_sentences_with_offsets(&self) -> Vec<SentenceData> {
        let mut sentences = Vec::new();
        let mut start = 0;
        let mut current = String::new();

        for (i, ch) in self.content.char_indices() {
            current.push(ch);

            if matches!(ch, '.' | '!' | '?') {
                let next_is_whitespace_or_end = self.content[i + ch.len_utf8()..]
                    .chars()
                    .next()
                    .is_none_or(char::is_whitespace);

                if next_is_whitespace_or_end {
                    let trimmed = current.trim();
                    if !trimmed.is_empty() {
                        sentences.push(SentenceData {
                            text: trimmed.to_string(),
                            start_offset: start,
                            end_offset: i + ch.len_utf8(),
                        });
                    }
                    current.clear();
                    start = i + ch.len_utf8();
                }
            }
        }

        let trimmed = current.trim();
        if !trimmed.is_empty() {
            sentences.push(SentenceData {
                text: trimmed.to_string(),
                start_offset: start,
                end_offset: self.content.len(),
            });
        }

        sentences
    }

    fn chunk_fixed_size(&self) -> Vec<ChunkText> {
        let config = &self.dripstone.config.chunking;
        let target_chars = config.target_tokens * 4;
        let overlap_chars = config.overlap_tokens * 4;

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < self.content.len() {
            let end = (start + target_chars).min(self.content.len());
            let end = self.content.floor_char_boundary(end);
            chunks.push(ChunkText::new(
                self.content[start..end].to_string(),
                start,
                end,
            ));

            if end >= self.content.len() {
                break;
            }

            start = end.saturating_sub(overlap_chars);
            start = start.min(end);
            start = self.content.floor_char_boundary(start);
        }

        chunks
    }
}

fn json_string(value: &serde_json::Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

impl<'a, B: Backend, E: Embedder> std::future::IntoFuture for IngestBuilder<'a, B, E> {
    type Output = Result<IngestResult, DripstoneError>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.await_result())
    }
}
