//! Speed benchmark cases (see parent `speed_benchmarks.rs`).

use std::sync::Arc;
use std::time::Instant;
use time::OffsetDateTime;

use crate::common::*;
use crate::datasets;
#[cfg(feature = "memory-backend")]
use crate::harness::PercentileStats;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use crate::speed_bench_report;
#[cfg(feature = "memory-backend")]
use dripstone::backend::memory::MemoryBackend as MemBackend;
use dripstone::{Chunk, DripstoneError, SearchQuery, traits::Backend};
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serde_json::json;
#[cfg(any(feature = "memory-backend", feature = "mmap-backend"))]
use serial_test::serial;

#[cfg(feature = "memory-backend")]
use super::helpers::measure_search_series;

/// Captures supported OR-of-equality filters against indexed ANN backends.
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_native_filter_index_paths() -> Result<(), DripstoneError> {
    use dripstone::index::spfresh::{CentroidIndex, SPFreshConfig};
    use dripstone::{
        Filter, HnswConfig, HnswIndex, IndexedBackend, IndexedBackendConfig, VectorIndex,
    };

    let corpus = datasets::load_integration_corpus().await?;
    let dim = corpus.dimension;
    let n_cap = corpus
        .base
        .len()
        .min(4096)
        .min(dripstone_bench_max_points());
    let n = n_cap.max(256).min(corpus.base.len().max(1));
    let query_count = corpus.queries.len().min(30).max(10);
    let nq = corpus.queries.len().max(1);
    let base = format!(
        "native-filter-bench-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );

    let supported_filter = Filter::Or {
        left: Box::new(Filter::And {
            left: Box::new(Filter::Equal {
                field: "domain".to_string(),
                value: json!("docs"),
            }),
            right: Box::new(Filter::In {
                field: "bucket".to_string(),
                values: vec![json!(0), json!(1)],
            }),
        }),
        right: Box::new(Filter::And {
            left: Box::new(Filter::Equal {
                field: "lang".to_string(),
                value: json!("rust"),
            }),
            right: Box::new(Filter::In {
                field: "bucket".to_string(),
                values: vec![json!(4), json!(5)],
            }),
        }),
    };
    let fallback_filter = Filter::Or {
        left: Box::new(Filter::And {
            left: Box::new(Filter::GreaterThanOrEqual {
                field: "tier".to_string(),
                value: json!(6),
            }),
            right: Box::new(Filter::Equal {
                field: "domain".to_string(),
                value: json!("docs"),
            }),
        }),
        right: Box::new(Filter::And {
            left: Box::new(Filter::Equal {
                field: "lang".to_string(),
                value: json!("rust"),
            }),
            right: Box::new(Filter::In {
                field: "bucket".to_string(),
                values: vec![json!(4), json!(5)],
            }),
        }),
    };

    let mut rows = Vec::new();
    let backends: Vec<(&str, Box<dyn VectorIndex>)> = vec![
        (
            "hnsw",
            Box::new(HnswIndex::new(
                HnswConfig {
                    ef_search: 64,
                    rerank_overfetch: 4,
                    ..Default::default()
                },
                dim,
            )),
        ),
        (
            "spfresh",
            Box::new(CentroidIndex::new(
                SPFreshConfig {
                    num_centroids: 64,
                    max_iterations: 20,
                    convergence_threshold: 0.001,
                    num_probes: 8,
                    metric: dripstone::embedding::SimilarityMetric::Cosine,
                    max_cluster_records: None,
                },
                dim,
            )),
        ),
    ];

    for (subject, index) in backends {
        let namespace = format!("{base}-{subject}");
        let chunks: Vec<Chunk> = (0..n)
            .map(|i| {
                let vector: Arc<[f32]> = Arc::from(corpus.base[i].clone().into_boxed_slice());
                let domain = match i % 4 {
                    0 | 1 => "docs",
                    2 => "code",
                    _ => "other",
                };
                let lang = if i % 2 == 0 { "rust" } else { "python" };
                Chunk::new(
                    format!("filter-doc-{i}"),
                    0,
                    format!("{domain} {lang} vector benchmark chunk {i}"),
                    vector,
                    0,
                    0,
                    1,
                    json!({
                        "domain": domain,
                        "lang": lang,
                        "bucket": i % 8,
                        "tier": i % 10,
                    }),
                    Some(namespace.clone()),
                )
            })
            .collect();

        let backend = IndexedBackend::new(
            MemBackend::new(),
            index,
            IndexedBackendConfig {
                auto_rebuild_threshold: 0,
                fallback_to_brute_force: false,
                overfetch_factor: 4,
                ..Default::default()
            },
        );
        backend.upsert(&chunks).await?;

        let mut variant_rows = Vec::new();
        for (variant, filter) in [
            ("no_filter", None),
            ("native_or_disjunction", Some(supported_filter.clone())),
            ("fallback_range_or", Some(fallback_filter.clone())),
        ] {
            let filter_for_query = filter.clone();
            let query_namespace = namespace.clone();
            let queries = &corpus.queries;
            let metric = corpus.search_metric;
            let (p50, qps, total_results) =
                measure_search_series(&backend, query_count, move |i| {
                    let mut query = SearchQuery::new(Arc::from(
                        queries[(i * 11 + 3) % nq].clone().into_boxed_slice(),
                    ))
                    .metric(metric)
                    .limit(10)
                    .namespace(query_namespace.as_str());
                    if let Some(filter) = filter_for_query.clone() {
                        query = query.filter(filter);
                    }
                    query
                })
                .await?;
            variant_rows.push(json!({
                "variant": variant,
                "search_p50_ms": speed_bench_report::duration_ms(p50),
                "qps": qps,
                "avg_results": total_results as f64 / query_count as f64,
            }));
        }

        rows.push(json!({
            "index": subject,
            "rows": variant_rows,
        }));
        backend.delete_namespace(&namespace).await?;
    }

    speed_bench_report::record(json!({
        "test": "test_native_filter_index_paths",
        "n": n,
        "dim": dim,
        "query_count": query_count,
        "rows": rows,
    }));
    Ok(())
}

/// Captures lexical rebuild/snapshot paths plus weighted-sum vs RRF hybrid fusion.
#[cfg(feature = "memory-backend")]
#[tokio::test]
#[serial]
async fn test_full_text_snapshot_and_hybrid_paths() -> Result<(), DripstoneError> {
    use dripstone::fts::InMemoryFullTextIndex;
    use dripstone::search::FusionConfig;
    use dripstone::{FullTextIndexedBackend, FullTextIndexedBackendConfig, FullTextQuery};
    use tempfile::tempdir;

    let corpus = datasets::load_integration_corpus().await?;
    let n_cap = corpus
        .base
        .len()
        .min(1500)
        .min(dripstone_bench_max_points());
    let n = n_cap.max(128).min(corpus.base.len().max(1));
    let nq = corpus.queries.len().max(1);
    let query_count = corpus.queries.len().min(24).max(8);
    let namespace = format!(
        "fts-bench-{}",
        OffsetDateTime::now_utc().unix_timestamp_nanos()
    );
    let dir = tempdir().map_err(|e| DripstoneError::storage(format!("tempdir: {e}")))?;
    let snapshot_path = dir.path().join("fts_snapshot.json");
    let snapshot_path_string = snapshot_path.to_string_lossy().to_string();
    let config = FullTextIndexedBackendConfig {
        auto_rebuild_threshold: 0,
        snapshot_path: Some(snapshot_path_string.clone()),
        rebuild_on_search: true,
    };

    let chunks: Vec<Chunk> = (0..n)
        .map(|i| {
            let topic = match i % 4 {
                0 => "rust vector search benchmark",
                1 => "python full text retrieval",
                2 => "hybrid ranking fusion bm25",
                _ => "snapshot rebuild index",
            };
            let modifier = match i % 3 {
                0 => "rust rust fusion",
                1 => "lexical recall bm25",
                _ => "dense ann mmap",
            };
            Chunk::new(
                format!("fts-doc-{i}"),
                0,
                format!("{topic} {modifier} corpus item {i}"),
                Arc::from(corpus.base[i].clone().into_boxed_slice()),
                0,
                0,
                1,
                json!({ "topic": topic, "group": i % 6 }),
                Some(namespace.clone()),
            )
        })
        .collect();

    let backend = FullTextIndexedBackend::new(
        MemBackend::new(),
        Arc::new(InMemoryFullTextIndex::new()),
        config.clone(),
    );
    backend.upsert(&chunks).await?;

    let lexical_query = FullTextQuery::new("rust vector fusion snapshot")
        .limit(10)
        .namespace(namespace.as_str());
    let dirty_search_start = Instant::now();
    let dirty_search_results = backend.search_full_text(&lexical_query).await?;
    let dirty_search_rebuild = dirty_search_start.elapsed();

    let mut warm_lexical = PercentileStats::new();
    for i in 0..query_count {
        let lexical = FullTextQuery::new(if i % 2 == 0 {
            "rust vector fusion"
        } else {
            "bm25 snapshot index"
        })
        .limit(10)
        .namespace(namespace.as_str());
        let t0 = Instant::now();
        let _ = backend.search_full_text(&lexical).await?;
        warm_lexical.add(t0.elapsed());
    }
    let warm_lexical_p50 = warm_lexical.calculate().p50;

    let flush_start = Instant::now();
    backend.flush().await?;
    let flush_snapshot = flush_start.elapsed();
    let snapshot_bytes = std::fs::metadata(&snapshot_path)
        .map_err(|e| DripstoneError::storage_with_source("fts snapshot metadata", e))?
        .len();

    let reloaded_inner = MemBackend::new();
    reloaded_inner.upsert(&chunks).await?;
    let reloaded = FullTextIndexedBackend::new(
        reloaded_inner,
        Arc::new(InMemoryFullTextIndex::new()),
        config,
    );
    let reload_start = Instant::now();
    let snapshot_results = reloaded.search_full_text(&lexical_query).await?;
    let reload_from_snapshot = reload_start.elapsed();

    let mut hybrid_rows = Vec::new();
    for (variant, fusion) in [
        ("weighted_sum", FusionConfig::weighted_sum().with_alpha(0.5)),
        ("reciprocal_rank", FusionConfig::reciprocal_rank()),
    ] {
        let mut stats = PercentileStats::new();
        let wall = Instant::now();
        let mut total_results = 0_usize;
        for i in 0..query_count {
            let dense = SearchQuery::new(Arc::from(
                corpus.queries[(i * 13 + 5) % nq].clone().into_boxed_slice(),
            ))
            .metric(corpus.search_metric)
            .limit(10)
            .namespace(namespace.as_str());
            let lexical = FullTextQuery::new(if i % 2 == 0 {
                "rust vector fusion"
            } else {
                "hybrid bm25 ranking"
            })
            .limit(10)
            .namespace(namespace.as_str());
            let t0 = Instant::now();
            let results = reloaded
                .search_hybrid_full_text(&dense, &lexical, &fusion)
                .await?;
            stats.add(t0.elapsed());
            total_results += results.len();
        }
        let qps = query_count as f64 / wall.elapsed().as_secs_f64().max(1e-9);
        hybrid_rows.push(json!({
            "variant": variant,
            "search_p50_ms": speed_bench_report::duration_ms(stats.calculate().p50),
            "qps": qps,
            "avg_results": total_results as f64 / query_count as f64,
        }));
    }

    speed_bench_report::record(json!({
        "test": "test_full_text_snapshot_and_hybrid_paths",
        "n": n,
        "query_count": query_count,
        "dirty_search_rebuild_ms": speed_bench_report::duration_ms(dirty_search_rebuild),
        "warm_lexical_p50_ms": speed_bench_report::duration_ms(warm_lexical_p50),
        "flush_snapshot_ms": speed_bench_report::duration_ms(flush_snapshot),
        "reload_from_snapshot_ms": speed_bench_report::duration_ms(reload_from_snapshot),
        "snapshot_bytes": snapshot_bytes,
        "dirty_search_results": dirty_search_results.len(),
        "snapshot_results": snapshot_results.len(),
        "hybrid_rows": hybrid_rows,
    }));

    backend.delete_namespace(&namespace).await?;
    reloaded.delete_namespace(&namespace).await?;
    Ok(())
}
