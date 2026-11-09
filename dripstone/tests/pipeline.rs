use std::cmp::Ordering;
use std::sync::Arc;

use async_trait::async_trait;
use dripstone::core::{Chunk, Metric, SearchQuery, SearchResult};
use dripstone::embedding::{EmbeddingRecord, SimilarityResult};
use dripstone::errors::DripstoneError;
use dripstone::index::{IndexSearchRequest, VectorIndex};
use dripstone::pipeline::{DiversityConfig, SearchPipelineBuilder};
use dripstone::search::NativeFilterPlan;
use dripstone::traits::Backend;
use serde_json::json;

#[derive(Debug)]
struct TestBackend {
    chunks: Vec<Chunk>,
}

impl TestBackend {
    const fn new(chunks: Vec<Chunk>) -> Self {
        Self { chunks }
    }

    fn filtered_chunks<'a>(&'a self, namespace: Option<&str>) -> impl Iterator<Item = &'a Chunk> {
        self.chunks.iter().filter(move |chunk| match namespace {
            Some(ns) => chunk.namespace.as_deref() == Some(ns),
            None => true,
        })
    }
}

#[async_trait]
impl Backend for TestBackend {
    fn capabilities(&self) -> dripstone::core::Capabilities {
        dripstone::core::Capabilities::memory()
    }

    async fn upsert(&self, _chunks: &[Chunk]) -> Result<(), DripstoneError> {
        Ok(())
    }

    async fn get(
        &self,
        id: &str,
        namespace: Option<&str>,
    ) -> Result<Option<Chunk>, DripstoneError> {
        for chunk in self.filtered_chunks(namespace) {
            if chunk.id == id {
                return Ok(Some(chunk.clone()));
            }
        }
        Ok(None)
    }

    async fn delete(&self, _id: &str, _namespace: Option<&str>) -> Result<(), DripstoneError> {
        Ok(())
    }

    async fn delete_document(
        &self,
        _doc_id: &str,
        _namespace: Option<&str>,
    ) -> Result<usize, DripstoneError> {
        Ok(0)
    }

    async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>, DripstoneError> {
        let mut results = Vec::new();
        for chunk in self.filtered_chunks(query.namespace.as_deref()) {
            let score = query.metric.compute(&query.vector, &chunk.vector)?;
            results.push(SearchResult::new(chunk.clone(), score, 0));
        }

        results.sort_by(|a, b| {
            if query.metric.higher_is_better() {
                partial_cmp_f32(b.score, a.score).unwrap_or(Ordering::Equal)
            } else {
                partial_cmp_f32(a.score, b.score).unwrap_or(Ordering::Equal)
            }
        });

        if results.len() > query.limit {
            results.truncate(query.limit);
        }

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i;
        }

        Ok(results)
    }

    async fn namespaces(&self) -> Result<Vec<String>, DripstoneError> {
        Ok(Vec::new())
    }

    async fn delete_namespace(&self, _namespace: &str) -> Result<(), DripstoneError> {
        Ok(())
    }

    async fn count(&self, namespace: Option<&str>) -> Result<usize, DripstoneError> {
        Ok(self.filtered_chunks(namespace).count())
    }

    async fn list(
        &self,
        namespace: Option<&str>,
        _limit: Option<usize>,
    ) -> Result<Vec<Chunk>, DripstoneError> {
        Ok(self.filtered_chunks(namespace).cloned().collect())
    }
}

fn build_chunk(doc_id: &str, chunk_index: usize, vector: Vec<f32>, category: &str) -> Chunk {
    Chunk::new(
        doc_id,
        chunk_index,
        "",
        vector,
        0,
        0,
        1,
        json!({ "category": category }),
        None,
    )
}

fn backend_with_chunks(chunks: Vec<Chunk>) -> Arc<TestBackend> {
    Arc::new(TestBackend::new(chunks))
}

#[derive(Debug, Default)]
struct CapturingIndex {
    last_plan: std::sync::Mutex<Option<NativeFilterPlan>>,
    last_query_len: std::sync::Mutex<Option<usize>>,
}

#[async_trait]
impl VectorIndex for CapturingIndex {
    async fn build(&self, _records: &[EmbeddingRecord]) -> Result<(), DripstoneError> {
        Ok(())
    }

    #[expect(
        clippy::expect_used,
        reason = "test capture mutexes are never poisoned; panicking is desired"
    )]
    async fn search_request(
        &self,
        request: IndexSearchRequest<'_>,
    ) -> Result<Vec<SimilarityResult>, DripstoneError> {
        *self
            .last_plan
            .lock()
            .expect("capturing index plan mutex should not be poisoned") =
            request.native_filter.cloned();
        *self
            .last_query_len
            .lock()
            .expect("capturing index query mutex should not be poisoned") =
            Some(request.query.len());
        Ok(vec![
            SimilarityResult::new(
                Arc::new(
                    EmbeddingRecord::new("a#0", vec![1.0, 0.0])
                        .with_metadata(json!({"category": "alpha"})),
                ),
                0.99,
            ),
            SimilarityResult::new(
                Arc::new(
                    EmbeddingRecord::new("b#0", vec![0.9, 0.1])
                        .with_metadata(json!({"category": "beta"})),
                ),
                0.95,
            ),
        ])
    }

    async fn add(&self, _record: EmbeddingRecord) -> Result<(), DripstoneError> {
        Ok(())
    }

    async fn remove(&self, _id: &str) -> Result<(), DripstoneError> {
        Ok(())
    }

    async fn iter_records(&self) -> Result<Vec<Arc<EmbeddingRecord>>, DripstoneError> {
        Ok(Vec::new())
    }

    async fn iter_records_by_namespace(
        &self,
        _namespace: Option<&str>,
    ) -> Result<Vec<Arc<EmbeddingRecord>>, DripstoneError> {
        Ok(Vec::new())
    }

    fn len(&self) -> usize {
        2
    }

    fn clear(&self) {}
}

#[tokio::test]
async fn pipeline_filters_metadata_after_ann() {
    let chunks = vec![
        build_chunk("a", 0, vec![1.0, 0.0], "alpha"),
        build_chunk("b", 0, vec![0.9, 0.1], "beta"),
        build_chunk("c", 0, vec![0.8, 0.2], "beta"),
    ];
    let backend = backend_with_chunks(chunks);
    let pipeline = SearchPipelineBuilder::new(Metric::Cosine)
        .coarse_ann(3)
        .metadata_filter("category", json!("beta"))
        .with_final_limit(1)
        .build();

    let results = pipeline
        .execute(&*backend, &[1.0, 0.0], None)
        .await
        .expect("pipeline search should succeed");

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.id, "b#0");
    assert_eq!(
        results[0]
            .chunk
            .metadata
            .get("category")
            .and_then(|value| value.as_str()),
        Some("beta")
    );
}

#[tokio::test]
async fn pipeline_index_ann_propagates_supported_filter_plan() {
    let chunks = vec![
        build_chunk("a", 0, vec![1.0, 0.0], "alpha"),
        build_chunk("b", 0, vec![0.9, 0.1], "beta"),
    ];
    let backend = backend_with_chunks(chunks);
    let index = Arc::new(CapturingIndex::default());
    let pipeline = SearchPipelineBuilder::new(Metric::Cosine)
        .index_ann(index.clone(), 2)
        .build();
    let filter = dripstone::search::Filter::Or {
        left: Box::new(dripstone::search::Filter::Equal {
            field: "category".to_string(),
            value: json!("alpha"),
        }),
        right: Box::new(dripstone::search::Filter::Equal {
            field: "category".to_string(),
            value: json!("beta"),
        }),
    };

    let results = pipeline
        .execute_with_filter(&*backend, &[1.0, 0.0], None, Some(&filter))
        .await
        .expect("pipeline search should succeed");

    assert_eq!(results.len(), 2);

    let captured_plan = index
        .last_plan
        .lock()
        .expect("capturing index plan mutex should not be poisoned")
        .clone()
        .expect("native filter plan should be captured");
    assert!(matches!(captured_plan, NativeFilterPlan::Disjunction(ref plans) if plans.len() == 2));
}

#[tokio::test]
async fn pipeline_index_ann_keeps_exact_filter_fallback() {
    let chunks = vec![
        build_chunk("a", 0, vec![1.0, 0.0], "alpha"),
        build_chunk("b", 0, vec![0.9, 0.1], "beta"),
    ];
    let backend = backend_with_chunks(chunks);
    let index = Arc::new(CapturingIndex::default());
    let pipeline = SearchPipelineBuilder::new(Metric::Cosine)
        .index_ann(index.clone(), 2)
        .build();
    let filter = dripstone::search::Filter::GreaterThan {
        field: "score".to_string(),
        value: json!(0),
    };

    let results = pipeline
        .execute_with_filter(&*backend, &[1.0, 0.0], None, Some(&filter))
        .await
        .expect("pipeline search should succeed");

    assert!(results.is_empty());

    let captured_plan = index
        .last_plan
        .lock()
        .expect("capturing index plan mutex should not be poisoned")
        .clone();
    assert!(
        captured_plan.is_none(),
        "unsupported filters should not be forwarded as native index plans"
    );
}

#[tokio::test]
async fn pipeline_normalizes_scores_into_unit_range() {
    let chunks = vec![
        build_chunk("a", 0, vec![1.0, 0.0], "alpha"),
        build_chunk("b", 0, vec![0.0, 1.0], "beta"),
        build_chunk("c", 0, vec![0.5, 0.5], "gamma"),
    ];
    let backend = backend_with_chunks(chunks);
    let pipeline = SearchPipelineBuilder::new(Metric::Cosine)
        .coarse_ann(3)
        .normalize_scores()
        .build();

    let results = pipeline
        .execute(&*backend, &[1.0, 0.0], None)
        .await
        .expect("pipeline search should succeed");

    assert_eq!(results.len(), 3);
    for result in &results {
        assert!(
            (0.0..=1.0).contains(&result.score),
            "score not normalized: {}",
            result.score
        );
    }

    let max_score = results
        .iter()
        .map(|result| result.score)
        .fold(f32::MIN, f32::max);
    let min_score = results
        .iter()
        .map(|result| result.score)
        .fold(f32::MAX, f32::min);
    assert!(
        (max_score - 1.0).abs() < f32::EPSILON,
        "expected a normalized score of 1.0"
    );
    assert!(
        min_score.abs() < f32::EPSILON,
        "expected a normalized score of 0.0"
    );
}

#[tokio::test]
async fn pipeline_enforces_diversity_limits() {
    let chunks = vec![
        build_chunk("a", 0, vec![1.0, 0.0], "alpha"),
        build_chunk("b", 0, vec![0.9, 0.1], "alpha"),
        build_chunk("c", 0, vec![0.8, 0.2], "beta"),
        build_chunk("d", 0, vec![0.7, 0.3], "gamma"),
    ];
    let backend = backend_with_chunks(chunks);
    let pipeline = SearchPipelineBuilder::new(Metric::Cosine)
        .coarse_ann(4)
        .diversity_rerank(
            DiversityConfig::new()
                .with_limit(3)
                .with_metadata_key("category")
                .with_max_per_value(1),
        )
        .build();

    let results = pipeline
        .execute(&*backend, &[1.0, 0.0], None)
        .await
        .expect("pipeline search should succeed");

    assert_eq!(results.len(), 3);
    let mut categories = Vec::new();
    for result in &results {
        let category = result
            .chunk
            .metadata
            .get("category")
            .and_then(|value| value.as_str())
            .expect("category metadata should exist");
        assert!(
            !categories.contains(&category),
            "category {category} appears more than once"
        );
        categories.push(category);
    }
}

fn partial_cmp_f32(left: f32, right: f32) -> Option<Ordering> {
    if left.is_nan() || right.is_nan() {
        return None;
    }
    left.partial_cmp(&right)
}
