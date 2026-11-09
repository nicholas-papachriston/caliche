use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::core::SearchQuery;
use crate::core::{Chunk, Metric, SearchResult};
#[cfg(feature = "tokio")]
use crate::embedding::SimilarityMetric;
use crate::errors::DripstoneError;
#[cfg(feature = "tokio")]
use crate::index::{IndexSearchRequest, VectorIndex};
use crate::search::Filter;
#[cfg(feature = "tokio")]
use crate::search::compile_native_filter_plan;
use crate::search::mmr::{MmrConfig, mmr_rerank};
use crate::traits::Backend;

#[cfg(feature = "search-advanced")]
use crate::search::temporal_decay::{TemporalDecayConfig, apply_temporal_decay};
#[cfg(feature = "search-advanced")]
use chrono::Utc;

/// Represents the immutable inputs shared across search pipeline stages.
#[must_use]
pub struct PipelineContext<'a> {
    query: &'a [f32],
    namespace: Option<&'a str>,
    filter: Option<&'a Filter>,
    metric: Metric,
}

impl<'a> PipelineContext<'a> {
    pub const fn new(
        query: &'a [f32],
        namespace: Option<&'a str>,
        filter: Option<&'a Filter>,
        metric: Metric,
    ) -> Self {
        Self {
            query,
            namespace,
            filter,
            metric,
        }
    }

    #[must_use]
    pub const fn query(&self) -> &'a [f32] {
        self.query
    }

    #[must_use]
    pub const fn namespace(&self) -> Option<&'a str> {
        self.namespace
    }

    #[must_use]
    pub const fn filter(&self) -> Option<&'a Filter> {
        self.filter
    }

    #[must_use]
    pub const fn metric(&self) -> Metric {
        self.metric
    }
}

#[async_trait]
pub trait PipelineStage<B>: Send + Sync
where
    B: Backend,
{
    async fn execute(
        &self,
        backend: &B,
        context: &PipelineContext<'_>,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError>;
}

#[must_use]
pub struct SearchPipeline<B>
where
    B: Backend,
{
    metric: Metric,
    stages: Vec<Arc<dyn PipelineStage<B> + Send + Sync>>,
    final_limit: Option<usize>,
}

impl<B> Clone for SearchPipeline<B>
where
    B: Backend,
{
    fn clone(&self) -> Self {
        Self {
            metric: self.metric,
            stages: self.stages.clone(), // Arc::clone is cheap
            final_limit: self.final_limit,
        }
    }
}

impl<B> SearchPipeline<B>
where
    B: Backend,
{
    /// Creates a builder for a pipeline using the given similarity metric.
    pub const fn builder(metric: Metric) -> SearchPipelineBuilder<B> {
        SearchPipelineBuilder::new(metric)
    }

    /// Executes every stage without a metadata filter.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] if any pipeline stage fails.
    pub async fn execute(
        &self,
        backend: &B,
        query: &[f32],
        namespace: Option<&str>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        self.execute_with_filter(backend, query, namespace, None)
            .await
    }

    /// Executes every stage with an optional metadata filter.
    ///
    /// # Errors
    ///
    /// Returns [`DripstoneError`] if any pipeline stage fails.
    pub async fn execute_with_filter(
        &self,
        backend: &B,
        query: &[f32],
        namespace: Option<&str>,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let context = PipelineContext::new(query, namespace, filter, self.metric);
        let mut results = Vec::new();
        for stage in &self.stages {
            results = stage.execute(backend, &context, results).await?;
        }
        if let Some(limit) = self.final_limit
            && results.len() > limit
        {
            results.truncate(limit);
        }
        Ok(results)
    }
}

#[must_use]
pub struct SearchPipelineBuilder<B>
where
    B: Backend,
{
    metric: Metric,
    stages: Vec<Arc<dyn PipelineStage<B> + Send + Sync>>,
    final_limit: Option<usize>,
}

impl<B> SearchPipelineBuilder<B>
where
    B: Backend,
{
    /// Creates an empty search pipeline builder.
    pub const fn new(metric: Metric) -> Self {
        Self {
            metric,
            stages: Vec::new(),
            final_limit: None,
        }
    }

    /// Truncates the final result set to `limit`.
    pub const fn with_final_limit(mut self, limit: usize) -> Self {
        self.final_limit = Some(limit);
        self
    }

    pub fn coarse_ann(mut self, limit: usize) -> Self {
        self.stages
            .push(Arc::new(CoarseAnnStage::new(limit)) as Arc<dyn PipelineStage<B> + Send + Sync>);
        self
    }

    /// ANN retrieval stage backed by a [`VectorIndex`].
    ///
    /// Queries the index for `limit` candidates, converts `SimilarityResult`s
    /// into `SearchResult`s by fetching full chunks from the backend, and passes
    /// them downstream for further filtering/reranking.
    #[cfg(feature = "tokio")]
    pub fn index_ann(mut self, index: Arc<dyn VectorIndex>, limit: usize) -> Self {
        self.stages.push(
            Arc::new(IndexAnnStage { index, limit }) as Arc<dyn PipelineStage<B> + Send + Sync>
        );
        self
    }

    pub fn metadata_filter(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.stages
            .push(Arc::new(MetadataFilterStage::new(key.into(), value))
                as Arc<dyn PipelineStage<B> + Send + Sync>);
        self
    }

    pub fn normalize_scores(mut self) -> Self {
        self.stages
            .push(Arc::new(ScoreNormalizationStage) as Arc<dyn PipelineStage<B> + Send + Sync>);
        self
    }

    pub fn diversity_rerank(mut self, config: DiversityConfig) -> Self {
        self.stages
            .push(Arc::new(DiversityRerankStage::new(config))
                as Arc<dyn PipelineStage<B> + Send + Sync>);
        self
    }

    /// MMR diversity reranking (lambda * relevance - (1-lambda) * similarity to selected).
    pub fn mmr_rerank(mut self, config: MmrConfig) -> Self {
        self.stages
            .push(Arc::new(MmrRerankStage::new(config)) as Arc<dyn PipelineStage<B> + Send + Sync>);
        self
    }

    /// Temporal decay: down-weight older results (requires `search-advanced`).
    #[cfg(feature = "search-advanced")]
    pub fn temporal_decay(mut self, config: TemporalDecayConfig) -> Self {
        self.stages
            .push(Arc::new(TemporalDecayStage::new(config))
                as Arc<dyn PipelineStage<B> + Send + Sync>);
        self
    }

    pub fn predicate<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&SearchResult) -> bool + Send + Sync + 'static,
    {
        self.stages
            .push(Arc::new(CustomPredicateStage::new(predicate))
                as Arc<dyn PipelineStage<B> + Send + Sync>);
        self
    }

    pub fn post_process<F>(mut self, processor: F) -> Self
    where
        F: Fn(Vec<SearchResult>) -> Vec<SearchResult> + Send + Sync + 'static,
    {
        self.stages
            .push(Arc::new(PostProcessStage::new(processor))
                as Arc<dyn PipelineStage<B> + Send + Sync>);
        self
    }

    /// Builds the configured search pipeline.
    pub fn build(self) -> SearchPipeline<B> {
        SearchPipeline {
            metric: self.metric,
            stages: self.stages,
            final_limit: self.final_limit,
        }
    }
}

struct CoarseAnnStage {
    limit: usize,
}

impl CoarseAnnStage {
    const fn new(limit: usize) -> Self {
        Self { limit }
    }
}

#[async_trait]
impl<B> PipelineStage<B> for CoarseAnnStage
where
    B: Backend,
{
    async fn execute(
        &self,
        backend: &B,
        context: &PipelineContext<'_>,
        _results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let query = SearchQuery {
            vector: context.query().to_vec().into(),
            limit: self.limit,
            metric: context.metric(),
            namespace: context.namespace().map(str::to_string),
            filter: context.filter().cloned(),
            hybrid_weight: None,
            sparse_vector: None,
            query_multi_vectors: None,
            maxsim: None,
            include_content: true,
            include_vectors: false,
        };

        let results = backend.search(&query).await?;
        Ok(results)
    }
}

#[cfg(feature = "tokio")]
struct IndexAnnStage {
    index: Arc<dyn VectorIndex>,
    limit: usize,
}

#[cfg(feature = "tokio")]
#[async_trait]
impl<B> PipelineStage<B> for IndexAnnStage
where
    B: Backend,
{
    async fn execute(
        &self,
        backend: &B,
        context: &PipelineContext<'_>,
        _results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let metric: SimilarityMetric = context.metric().into();
        let overfetch = self.limit.saturating_mul(2).max(self.limit);
        let native_filter_plan = compile_native_filter_plan(context.namespace(), context.filter());
        let candidates = self
            .index
            .search_request(IndexSearchRequest {
                query: context.query(),
                limit: overfetch,
                metric,
                native_filter: native_filter_plan
                    .is_supported()
                    .then_some(&native_filter_plan),
            })
            .await?;

        let mut results: Vec<SearchResult> = Vec::with_capacity(candidates.len());
        for candidate in &candidates {
            let id: &str = &candidate.record.id;
            let ns = candidate.record.namespace.as_deref();
            if let Some(ns_filter) = context.namespace() {
                let candidate_ns = ns.unwrap_or("default");
                if candidate_ns != ns_filter {
                    continue;
                }
            }
            if let Some(chunk) = backend.get(id, ns).await? {
                if context
                    .filter()
                    .is_some_and(|filter| !filter.matches(&chunk.metadata))
                {
                    continue;
                }
                results.push(SearchResult::new(chunk, candidate.score, results.len()));
            }
        }
        results.truncate(self.limit);
        Ok(results)
    }
}

struct MetadataFilterStage {
    key: String,
    value: serde_json::Value,
}

impl MetadataFilterStage {
    const fn new(key: String, value: serde_json::Value) -> Self {
        Self { key, value }
    }
}

#[async_trait]
impl<B> PipelineStage<B> for MetadataFilterStage
where
    B: Backend,
{
    async fn execute(
        &self,
        _backend: &B,
        _context: &PipelineContext<'_>,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        results.retain(|result| result.chunk.metadata.get(&self.key) == Some(&self.value));
        Ok(results)
    }
}

struct ScoreNormalizationStage;

#[async_trait]
impl<B> PipelineStage<B> for ScoreNormalizationStage
where
    B: Backend,
{
    async fn execute(
        &self,
        _backend: &B,
        context: &PipelineContext<'_>,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        if results.is_empty() {
            return Ok(results);
        }

        let metric = context.metric();
        let mut min_score = f32::MAX;
        let mut max_score = f32::MIN;
        for result in &results {
            let score = result.score;
            if score < min_score {
                min_score = score;
            }
            if score > max_score {
                max_score = score;
            }
        }

        if min_score.total_cmp(&max_score).is_eq() {
            for result in &mut results {
                result.score = if metric.higher_is_better() { 1.0 } else { 0.0 };
            }
            return Ok(results);
        }

        for result in &mut results {
            if metric.higher_is_better() {
                let normalized = (result.score - min_score) / (max_score - min_score);
                result.score = normalized.clamp(0.0, 1.0);
            } else {
                // For distance metrics, lower is better, so invert the normalization
                let normalized = 1.0 - (result.score - min_score) / (max_score - min_score);
                result.score = normalized.clamp(0.0, 1.0);
            }
        }
        Ok(results)
    }
}

#[derive(Clone)]
#[must_use]
pub struct DiversityConfig {
    limit: Option<usize>,
    metadata_key: Option<String>,
    max_per_value: Option<usize>,
    max_similarity: Option<f32>,
    min_distance: Option<f32>,
}

impl DiversityConfig {
    /// Creates a diversity configuration with no gates.
    pub const fn new() -> Self {
        Self {
            limit: None,
            metadata_key: None,
            max_per_value: None,
            max_similarity: None,
            min_distance: None,
        }
    }

    /// Sets the maximum number of selected results.
    pub const fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_metadata_key(mut self, key: impl Into<String>) -> Self {
        self.metadata_key = Some(key.into());
        self
    }

    /// Sets the maximum number of results per metadata value.
    pub const fn with_max_per_value(mut self, max: usize) -> Self {
        self.max_per_value = Some(max);
        self
    }

    /// Rejects candidates above this pairwise similarity.
    pub const fn with_max_similarity(mut self, value: f32) -> Self {
        self.max_similarity = Some(value);
        self
    }

    /// Rejects candidates below this pairwise distance.
    pub const fn with_min_distance(mut self, value: f32) -> Self {
        self.min_distance = Some(value);
        self
    }
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self::new()
    }
}

struct DiversityRerankStage {
    config: DiversityConfig,
}

impl DiversityRerankStage {
    const fn new(config: DiversityConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl<B> PipelineStage<B> for DiversityRerankStage
where
    B: Backend,
{
    async fn execute(
        &self,
        _backend: &B,
        context: &PipelineContext<'_>,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        if results.len() <= 1 {
            return Ok(results);
        }

        let metric = context.metric();
        if self.config.max_similarity.is_some() && !metric.higher_is_better() {
            return Err(DripstoneError::pipeline(
                "max_similarity gate requires a metric where higher scores denote better matches",
            ));
        }
        if self.config.min_distance.is_some() && metric.higher_is_better() {
            return Err(DripstoneError::pipeline(
                "min_distance gate requires a metric where lower scores denote better matches",
            ));
        }
        let mut selected: Vec<SearchResult> = Vec::new();
        let mut per_value_counter: HashMap<String, usize> = HashMap::new();
        let diversity_limit = self.config.limit.unwrap_or(usize::MAX);
        let max_per_value = self.config.max_per_value.unwrap_or(1);
        if max_per_value == 0 {
            return Err(DripstoneError::pipeline(
                "max_per_value must be greater than zero",
            ));
        }
        let max_similarity = self.config.max_similarity;
        let min_distance = self.config.min_distance;

        // Iterate over results by value to avoid cloning until necessary
        for candidate in results {
            if selected.len() >= diversity_limit {
                break;
            }

            let bucket_cache = if let Some(key) = &self.config.metadata_key {
                let bucket = derive_metadata_bucket(&candidate.chunk, key);
                if per_value_counter
                    .get(&bucket)
                    .is_some_and(|count| *count >= max_per_value)
                {
                    continue;
                }
                Some(bucket)
            } else {
                None
            };

            if let Some(threshold) = max_similarity
                && metric.higher_is_better()
                && violates_similarity_gate(metric, &selected, &candidate, threshold)?
            {
                continue;
            }

            if let Some(threshold) = min_distance
                && !metric.higher_is_better()
                && violates_distance_gate(metric, &selected, &candidate, threshold)?
            {
                continue;
            }

            if let Some(bucket) = bucket_cache {
                per_value_counter
                    .entry(bucket)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }

            selected.push(candidate);
        }

        Ok(selected)
    }
}

struct MmrRerankStage {
    config: MmrConfig,
}

impl MmrRerankStage {
    const fn new(config: MmrConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl<B> PipelineStage<B> for MmrRerankStage
where
    B: Backend,
{
    async fn execute(
        &self,
        _backend: &B,
        _context: &PipelineContext<'_>,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        Ok(mmr_rerank(&results, &self.config))
    }
}

#[cfg(feature = "search-advanced")]
struct TemporalDecayStage {
    config: TemporalDecayConfig,
}

#[cfg(feature = "search-advanced")]
impl TemporalDecayStage {
    const fn new(config: TemporalDecayConfig) -> Self {
        Self { config }
    }
}

#[cfg(feature = "search-advanced")]
#[async_trait]
impl<B> PipelineStage<B> for TemporalDecayStage
where
    B: Backend,
{
    async fn execute(
        &self,
        _backend: &B,
        _context: &PipelineContext<'_>,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        apply_temporal_decay(&mut results, &self.config, Utc::now());
        Ok(results)
    }
}

fn derive_metadata_bucket(chunk: &Chunk, key: &str) -> String {
    chunk
        .metadata
        .get(key)
        .map_or_else(|| String::from("null"), std::string::ToString::to_string)
}

fn violates_similarity_gate(
    metric: Metric,
    selected: &[SearchResult],
    candidate: &SearchResult,
    threshold: f32,
) -> Result<bool, DripstoneError> {
    for existing in selected {
        let similarity = metric.compute(&existing.chunk.vector, &candidate.chunk.vector)?;
        if similarity > threshold {
            return Ok(true);
        }
    }
    Ok(false)
}

fn violates_distance_gate(
    metric: Metric,
    selected: &[SearchResult],
    candidate: &SearchResult,
    threshold: f32,
) -> Result<bool, DripstoneError> {
    for existing in selected {
        let distance = metric.compute(&existing.chunk.vector, &candidate.chunk.vector)?;
        if distance < threshold {
            return Ok(true);
        }
    }
    Ok(false)
}

struct CustomPredicateStage {
    predicate: Arc<dyn Fn(&SearchResult) -> bool + Send + Sync>,
}

impl CustomPredicateStage {
    fn new<F>(predicate: F) -> Self
    where
        F: Fn(&SearchResult) -> bool + Send + Sync + 'static,
    {
        Self {
            predicate: Arc::new(predicate),
        }
    }
}

#[async_trait]
impl<B> PipelineStage<B> for CustomPredicateStage
where
    B: Backend,
{
    async fn execute(
        &self,
        _backend: &B,
        _context: &PipelineContext<'_>,
        mut results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        results.retain(|result| (self.predicate)(result));
        Ok(results)
    }
}

struct PostProcessStage {
    processor: Arc<dyn Fn(Vec<SearchResult>) -> Vec<SearchResult> + Send + Sync>,
}

impl PostProcessStage {
    fn new<F>(processor: F) -> Self
    where
        F: Fn(Vec<SearchResult>) -> Vec<SearchResult> + Send + Sync + 'static,
    {
        Self {
            processor: Arc::new(processor),
        }
    }
}

#[async_trait]
impl<B> PipelineStage<B> for PostProcessStage
where
    B: Backend,
{
    async fn execute(
        &self,
        _backend: &B,
        _context: &PipelineContext<'_>,
        results: Vec<SearchResult>,
    ) -> Result<Vec<SearchResult>, DripstoneError> {
        let processed = (self.processor)(results);
        Ok(processed)
    }
}
