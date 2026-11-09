use super::{
    Backend, Dripstone, DripstoneError, Embedder, SearchBuilder, SearchQuery, SearchResult,
};
#[cfg(feature = "rust-stemmers")]
use crate::search::fusion::fuse_dense_with_sparse_scores;
use crate::search::fusion::{FusionConfig, fuse_search_results};

impl<'a, B: Backend, E: Embedder> SearchBuilder<'a, B, E> {
    pub(super) fn new(dripstone: &'a Dripstone<B, E>, query: &'a str) -> Self {
        Self {
            dripstone,
            query,
            search_mode: dripstone.config.search.default_mode,
            limit: None,
            namespace: None,
            filter: None,
            hybrid_weight: None,
            hybrid_full_text_weight: None,
            hybrid_full_text_fusion: None,
            full_text_options: dripstone.config.search.full_text_options.clone(),
            rerank: dripstone.config.search.rerank_enabled,
            hashline_format: None,
            mmr_config: None,
            #[cfg(feature = "search-advanced")]
            temporal_decay_config: None,
            query_multi_vectors: None,
            maxsim: None,
        }
    }

    pub const fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub fn filter(mut self, filter: crate::search::Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Rank with dense similarity and skip BM25 fusion.
    ///
    /// [`crate::Dripstone::open`] stores use dense plus BM25 reciprocal-rank
    /// fusion unless the caller sets this. [`crate::Dripstone::new`] and the
    /// builder already start on dense search.
    #[must_use]
    pub const fn dense_only(mut self) -> Self {
        self.search_mode = crate::core::SearchMode::Dense;
        self.hybrid_weight = None;
        self.hybrid_full_text_weight = None;
        self.hybrid_full_text_fusion = None;
        self
    }

    pub const fn hybrid(mut self, dense_weight: f32) -> Self {
        self.search_mode = crate::core::SearchMode::Dense;
        self.hybrid_weight = Some(dense_weight.clamp(0.0, 1.0));
        self
    }

    pub const fn full_text(mut self) -> Self {
        self.search_mode = crate::core::SearchMode::FullText;
        self.hybrid_weight = None;
        self.hybrid_full_text_weight = None;
        self
    }

    pub const fn hybrid_full_text(mut self, dense_weight: f32) -> Self {
        self.search_mode = crate::core::SearchMode::HybridDenseFullText;
        self.hybrid_full_text_weight = Some(dense_weight.clamp(0.0, 1.0));
        self.hybrid_full_text_fusion = None;
        self.hybrid_weight = None;
        self
    }

    pub const fn hybrid_full_text_fusion(mut self, fusion: FusionConfig) -> Self {
        self.search_mode = crate::core::SearchMode::HybridDenseFullText;
        self.hybrid_full_text_fusion = Some(fusion);
        self.hybrid_full_text_weight = None;
        self.hybrid_weight = None;
        self
    }

    pub const fn hybrid_full_text_rrf(mut self, rrf_k: f32) -> Self {
        self.search_mode = crate::core::SearchMode::HybridDenseFullText;
        self.hybrid_full_text_fusion =
            Some(crate::search::FusionConfig::reciprocal_rank().with_rrf_k(rrf_k));
        self.hybrid_full_text_weight = None;
        self.hybrid_weight = None;
        self
    }

    pub const fn query_expansion(mut self, enabled: bool) -> Self {
        self.full_text_options.query_expansion = enabled;
        self
    }

    pub const fn rerank(mut self, enabled: bool) -> Self {
        self.rerank = enabled;
        self
    }

    pub const fn hashline_format(mut self, enabled: bool) -> Self {
        self.hashline_format = Some(enabled);
        self
    }

    pub const fn mmr(mut self, config: crate::search::MmrConfig) -> Self {
        self.mmr_config = Some(config);
        self
    }

    #[cfg(feature = "search-advanced")]
    pub fn temporal_decay(mut self, config: crate::search::TemporalDecayConfig) -> Self {
        self.temporal_decay_config = Some(config);
        self
    }

    /// Attach late-interaction query token vectors for MaxSim.
    pub fn with_multi_vectors(mut self, tokens: crate::search::MultiVectors) -> Self {
        self.query_multi_vectors = Some(tokens);
        self
    }

    /// Configure MaxSim options (used with [`Self::with_multi_vectors`]).
    pub const fn with_maxsim(mut self, opts: crate::search::MaxSimOptions) -> Self {
        self.maxsim = Some(opts);
        self
    }

    fn limit_and_fetch_limit(&self) -> (usize, usize) {
        let limit = self
            .limit
            .unwrap_or(self.dripstone.config.search.default_limit);
        let mut fetch_limit = if self.rerank {
            limit.saturating_mul(self.dripstone.config.search.rerank_candidates_multiplier)
        } else {
            limit
        };
        if self.query_multi_vectors.is_some() {
            let overfetch = self.maxsim.map_or(5, |o| o.overfetch.max(1));
            fetch_limit = fetch_limit.saturating_mul(overfetch).max(limit);
        }
        (limit, fetch_limit)
    }

    fn build_search_query(&self, query_vector: Vec<f32>, fetch_limit: usize) -> SearchQuery {
        let mut search_query = SearchQuery::new(query_vector).limit(fetch_limit);

        if let Some(ref namespace) = self.namespace {
            search_query = search_query.namespace(namespace.clone());
        }

        if let Some(ref filter) = self.filter {
            search_query = search_query.filter(filter.clone());
        }

        if let Some(ref mv) = self.query_multi_vectors {
            search_query = search_query.with_multi_vectors(std::sync::Arc::clone(mv));
        }
        if let Some(opts) = self.maxsim {
            search_query = search_query.with_maxsim(opts);
        }

        search_query
    }

    fn build_full_text_query(&self, fetch_limit: usize) -> crate::core::FullTextQuery {
        let mut lexical_query = crate::core::FullTextQuery::new(self.query)
            .limit(fetch_limit)
            .options(self.full_text_options.clone());

        if let Some(ref namespace) = self.namespace {
            lexical_query = lexical_query.namespace(namespace.clone());
        }

        if let Some(ref filter) = self.filter {
            lexical_query = lexical_query.filter(filter.clone());
        }

        lexical_query
    }

    async fn execute_search(
        &self,
        search_query: SearchQuery,
        query_dimension: usize,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        if let Some(weight) = self.hybrid_weight {
            #[cfg(feature = "rust-stemmers")]
            let use_line_level = self.dripstone.config.hashline.line_level_bm25
                && self
                    .dripstone
                    .hybrid_helper
                    .as_ref()
                    .is_some_and(|h| h.has_line_corpus());

            #[cfg(not(feature = "rust-stemmers"))]
            let use_line_level = false;

            if use_line_level {
                return self
                    .execute_line_level_hybrid_search(search_query, weight)
                    .await;
            }

            return self
                .execute_chunk_level_search(search_query, query_dimension, weight)
                .await;
        }

        self.dripstone.backend.search(&search_query).await
    }

    async fn execute_chunk_level_search(
        &self,
        mut search_query: SearchQuery,
        query_dimension: usize,
        weight: f32,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let sparse_vector = {
            #[cfg(feature = "rust-stemmers")]
            {
                self.dripstone
                    .hybrid_helper
                    .as_ref()
                    .and_then(|h| h.generate_sparse_vector(self.query))
            }
            #[cfg(not(feature = "rust-stemmers"))]
            {
                None
            }
        };

        if self.dripstone.backend.capabilities().native_hybrid && sparse_vector.is_some() {
            search_query.hybrid_weight = Some(weight);
            search_query.sparse_vector = sparse_vector;
            return self.dripstone.backend.search(&search_query).await;
        }

        if let Some(sparse_vec) = sparse_vector {
            let mut dense_query = search_query.clone();
            dense_query.sparse_vector = None;
            dense_query.hybrid_weight = None;
            let dense_results = self.dripstone.backend.search(&dense_query).await?;
            let sparse_results = self
                .search_sparse_candidates(&search_query, sparse_vec, query_dimension)
                .await?;
            let fusion_config = FusionConfig::weighted_sum().with_alpha(weight);
            return fuse_search_results(dense_results, sparse_results, &fusion_config);
        }

        search_query.hybrid_weight = None;
        self.dripstone.backend.search(&search_query).await
    }

    async fn search_sparse_candidates(
        &self,
        search_query: &SearchQuery,
        sparse_vec: crate::sparse::SparseVector,
        query_dimension: usize,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let mut sparse_vec_with_dim = sparse_vec.clone();
        if sparse_vec_with_dim.dimension.is_none() {
            sparse_vec_with_dim.dimension = Some(query_dimension);
        }
        if let Ok(dense_from_sparse) = sparse_vec_with_dim.to_dense() {
            let mut sparse_query = search_query.clone();
            sparse_query.vector = dense_from_sparse.into();
            sparse_query.sparse_vector = None;
            sparse_query.hybrid_weight = None;
            self.dripstone.backend.search(&sparse_query).await
        } else {
            Ok(Vec::new())
        }
    }

    #[cfg_attr(
        not(feature = "rust-stemmers"),
        expect(
            clippy::unused_async,
            reason = "await is used when rust-stemmers enables the hybrid helper search path"
        )
    )]
    async fn execute_line_level_hybrid_search(
        &self,
        search_query: SearchQuery,
        weight: f32,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        #[cfg(feature = "rust-stemmers")]
        {
            let Some(helper) = self.dripstone.hybrid_helper.as_ref() else {
                return Err(DripstoneError::configuration(
                    "line-level hybrid search requires a configured hybrid helper",
                ));
            };
            let mut dense_query = search_query.clone();
            dense_query.sparse_vector = None;
            dense_query.hybrid_weight = None;
            let dense_results = self.dripstone.backend.search(&dense_query).await?;
            let sparse_scores = helper.score_chunks_by_lines(self.query);
            fuse_dense_with_sparse_scores(dense_results, &sparse_scores, weight)
        }
        #[cfg(not(feature = "rust-stemmers"))]
        {
            let _ = (search_query, weight);
            Err(DripstoneError::configuration(
                "line-level hybrid search requires the 'rust-stemmers' feature",
            ))
        }
    }

    fn apply_post_processing(
        &self,
        mut results: Vec<SearchResult>,
        limit: usize,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        #[cfg(feature = "search-advanced")]
        if let Some(ref config) = self.temporal_decay_config {
            crate::search::apply_temporal_decay(&mut results, config, chrono::Utc::now());
        }

        if self.rerank {
            #[cfg(feature = "embedding-ort")]
            {
                let reranker = self.dripstone.reranker.as_ref().ok_or_else(|| {
                    DripstoneError::configuration(
                        "reranking requested, but no ONNX reranker is configured; set search.rerank_model_path or disable reranking",
                    )
                })?;
                results = reranker.rerank(self.query, results)?;
            }
            #[cfg(not(feature = "embedding-ort"))]
            {
                return Err(DripstoneError::configuration(
                    "reranking requested, but Dripstone was built without the 'embedding-ort' feature",
                ));
            }
        }

        if let Some(ref config) = self.mmr_config {
            results = crate::search::mmr_rerank(&results, config);
        }

        results.truncate(limit);

        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        self.apply_hashline_formatting(&mut results);

        Ok(results)
    }

    fn apply_hashline_formatting(&self, results: &mut [SearchResult]) {
        let should_format = self.hashline_format == Some(true)
            || ((self.dripstone.config.hashline.format_results
                || self.dripstone.config.search.hashline_in_results)
                && self.hashline_format != Some(false));
        if !should_format {
            return;
        }

        let options = self.dripstone.config.hashline.to_options();
        for result in results.iter_mut() {
            if let Some(hashline) = result
                .chunk
                .metadata
                .get("hashline")
                .and_then(|h| h.as_object())
                && let Some(start_line) = hashline
                    .get("start_line")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|line| usize::try_from(line).ok())
            {
                result.chunk.content = crate::hashline::format_content_with_hashlines(
                    &result.chunk.content,
                    start_line,
                    &options,
                );
            }
        }
    }

    /// Executes the configured search.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] when embedding, backend retrieval, fusion,
    /// reranking, or post-processing fails.
    pub async fn await_result(self) -> Result<Vec<SearchResult>, DripstoneError> {
        let (limit, fetch_limit) = self.limit_and_fetch_limit();
        let results = match self.search_mode {
            crate::core::SearchMode::Dense => self.search_dense(limit, fetch_limit).await?,
            crate::core::SearchMode::FullText => self.search_full_text(fetch_limit).await?,
            crate::core::SearchMode::HybridDenseFullText => {
                self.search_hybrid_dense_full_text(fetch_limit).await?
            }
        };
        self.apply_post_processing(results, limit)
    }

    async fn search_dense(
        &self,
        limit: usize,
        fetch_limit: usize,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let query_vector = if let Some(ref mv) = self.query_multi_vectors {
            crate::search::pooled_vector_from_multi(mv).to_vec()
        } else {
            self.dripstone.embedder.embed_query(self.query).await?
        };
        let query_dimension = query_vector.len();
        if self.query_multi_vectors.is_some() && self.hybrid_weight.is_none() {
            let search_query = self.build_search_query(query_vector, limit);
            return crate::traits::search_with_maxsim_fallback(
                self.dripstone.backend.as_ref(),
                &search_query,
            )
            .await;
        }
        let search_query = self.build_search_query(query_vector, fetch_limit);
        let mut results = self
            .execute_search(search_query.clone(), query_dimension)
            .await?;
        if self.query_multi_vectors.is_some()
            && !self.dripstone.backend.capabilities().native_maxsim
        {
            let mut final_query = search_query;
            final_query.limit = limit;
            results = crate::search::rerank_with_maxsim(results, &final_query)?;
        }
        Ok(results)
    }

    async fn search_full_text(
        &self,
        fetch_limit: usize,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let lexical_query = self.build_full_text_query(fetch_limit);
        if !self.dripstone.backend.capabilities().native_full_text {
            return Err(DripstoneError::pipeline(
                "backend does not support native full-text search",
            ));
        }
        self.dripstone
            .backend
            .search_full_text(&lexical_query)
            .await
    }

    async fn search_hybrid_dense_full_text(
        &self,
        fetch_limit: usize,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let query_vector = self.dripstone.embedder.embed_query(self.query).await?;
        let dense_query = self.build_search_query(query_vector, fetch_limit);
        let lexical_query = self.build_full_text_query(fetch_limit);
        let fusion_config = if let Some(weight) = self.hybrid_full_text_weight {
            crate::search::FusionConfig::weighted_sum().with_alpha(weight)
        } else {
            self.hybrid_full_text_fusion
                .clone()
                .unwrap_or_else(|| self.dripstone.config.search.hybrid_full_text_fusion.clone())
        };

        if self
            .dripstone
            .backend
            .capabilities()
            .native_hybrid_full_text
        {
            return self
                .dripstone
                .backend
                .search_hybrid_full_text(&dense_query, &lexical_query, &fusion_config)
                .await;
        }
        if self.dripstone.backend.capabilities().native_full_text {
            let dense_results = self.dripstone.backend.search(&dense_query).await?;
            let lexical_results = self
                .dripstone
                .backend
                .search_full_text(&lexical_query)
                .await?;
            return fuse_search_results(dense_results, lexical_results, &fusion_config);
        }
        Err(DripstoneError::pipeline(
            "backend does not support native full-text search",
        ))
    }
}

impl<'a, B: Backend, E: Embedder> std::future::IntoFuture for SearchBuilder<'a, B, E> {
    type Output = Result<Vec<SearchResult>, DripstoneError>;
    type IntoFuture =
        std::pin::Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(self.await_result())
    }
}
