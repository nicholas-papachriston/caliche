use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{
    DictionaryArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;

use super::catalog::{CorpusName, CorpusSelector};
use super::env::{CorpusRequest, resolve_cap};
use super::layout::{CorpusManifest, CorpusRowPolicy, MaterializedRows, write_layout};
use super::normalize::{layout_reusable, prepare_corpus_blocking};
use super::parquet::cohere_shard_relative;
use super::sanitize::apply_row_policy;
use super::{
    EMB_COLUMN, INTEGRATION_QUERY_MAX, INTEGRATION_TRAIN_MAX, MIN_L2_NORM_FOR_COSINE,
    SOAK_QUERY_MAX, SOAK_TRAIN_MAX,
};
use crate::core::Metric;

#[test]
fn catalog_names_and_english_shard_are_stable() {
    assert_eq!(
        CorpusName::parse("nytimes-256-angular").expect("nytimes"),
        CorpusName::Nytimes256Angular
    );
    assert_eq!(
        CorpusName::parse("GIST").expect("gist"),
        CorpusName::Gist960Euclidean
    );
    assert_eq!(
        CorpusName::from_url("https://ann-benchmarks.com/sift-128-euclidean.hdf5"),
        Some(CorpusName::Sift128Euclidean)
    );
    assert_eq!(cohere_shard_relative(0), "en/0000.parquet");
    assert!(
        CorpusName::parse("simple")
            .expect_err("simple english")
            .to_string()
            .contains("en/")
    );
    assert_eq!(
        CorpusName::CohereWikiEn.language(),
        Some(super::catalog::COHERE_LANG_PREFIX)
    );
    assert_eq!(CorpusName::parse("cosqa").expect("cosqa"), CorpusName::Code);
    assert_eq!(CorpusName::parse("coir").expect("coir"), CorpusName::Code);
    assert_eq!(
        CorpusName::Code.dimension(),
        Some(super::CODE_FEATURE_DIMENSION)
    );
    assert_eq!(CorpusName::Code.revision(), "cosqa-0846fa3b");
    assert_eq!(CorpusName::Code.metric(), Metric::Cosine);
    let corpus_url = super::parquet::huggingface_dataset_url(
        super::COSQA_REPO,
        super::COSQA_REVISION,
        super::COSQA_CORPUS_FILE,
    );
    assert!(corpus_url.contains("CoIR-Retrieval/cosqa"));
    assert!(corpus_url.contains("0846fa3b963a21bead36e9fab61451fc83b777a6"));
    assert!(corpus_url.ends_with("corpus/corpus-00000-of-00001.parquet"));
}

#[test]
fn caps_prefer_corpus_env_over_integration_alias() {
    assert_eq!(resolve_cap(Some(10), Some(4_000), 99), 10);
    assert_eq!(resolve_cap(None, Some(4_000), 99), 4_000);
    assert_eq!(resolve_cap(None, None, INTEGRATION_TRAIN_MAX), 4_000);
    assert_eq!(resolve_cap(Some(0), None, SOAK_TRAIN_MAX), SOAK_TRAIN_MAX);
    assert_eq!(INTEGRATION_QUERY_MAX, 64);
    assert_eq!(SOAK_QUERY_MAX, 128);
    assert_eq!(SOAK_TRAIN_MAX, 10_000);
}

#[test]
fn parquet_normalize_writes_capped_layout_and_reuses_it() {
    let dir = tempfile::tempdir().expect("temp");
    let parquet = dir.path().join("en.parquet");
    write_fixture(&parquet);

    let cache = dir.path().join("cache");
    let request = parquet_request(&cache, &parquet, 4, 2);
    let prepared = prepare_corpus_blocking(&request).expect("normalize");
    assert_eq!(prepared.loaded.dimension, 2);
    assert_eq!(prepared.loaded.base.len(), 4);
    assert_eq!(prepared.loaded.queries.len(), 2);
    assert_eq!(prepared.loaded.base[0], vec![1.0, 0.0]);
    assert_eq!(prepared.loaded.ids[0], "doc0");
    assert_eq!(prepared.loaded.queries[0], vec![0.0, 0.0]);
    assert!(prepared.manifest().has_meta);
    assert!(!prepared.manifest().has_neighbors);
    assert!(prepared.normalized.root.join("manifest.json").is_file());
    assert!(prepared.normalized.root.join("vectors.f32.bin").is_file());
    assert!(prepared.normalized.root.join("ids.jsonl").is_file());
    assert!(prepared.normalized.root.join("queries.f32.bin").is_file());
    assert!(prepared.normalized.root.join("meta.jsonl").is_file());

    let batches: Vec<_> = prepared
        .normalized
        .train_batches(2, 4)
        .expect("batches")
        .collect::<Result<Vec<_>, _>>()
        .expect("batch rows");
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].vectors.len(), 2);
    assert_eq!(batches[1].offset, 2);
    assert_eq!(batches[1].ids[0], "doc2");

    let again = prepare_corpus_blocking(&request).expect("reuse");
    assert_eq!(again.loaded.base, prepared.loaded.base);
    assert_eq!(again.manifest().train_count, 4);
    assert_eq!(again.manifest().row_policy, CorpusRowPolicy::Keep);
    assert_eq!(again.manifest().dropped_train_rows, 0);
    assert_eq!(again.manifest().dropped_query_rows, 0);
    assert_eq!(again.loaded.queries[0], vec![0.0, 0.0]);

    let smaller = parquet_request(&cache, &parquet, 2, 1);
    let sliced = prepare_corpus_blocking(&smaller).expect("smaller cap");
    assert_eq!(sliced.loaded.base.len(), 2);
    assert_eq!(sliced.loaded.base[0], vec![1.0, 0.0]);
    assert_eq!(sliced.loaded.queries, vec![vec![0.0, 0.0]]);
    assert_eq!(sliced.manifest().train_count, 4);
    assert_eq!(sliced.manifest().scanned_train_rows, 4);
}

#[test]
fn profiles_are_distinct_caps() {
    assert_ne!(INTEGRATION_TRAIN_MAX, SOAK_TRAIN_MAX);
    assert_ne!(INTEGRATION_QUERY_MAX, SOAK_QUERY_MAX);
}

#[test]
fn row_policy_follows_metric() {
    assert_eq!(
        CorpusRowPolicy::for_metric(Metric::Cosine),
        CorpusRowPolicy::DropBelowMinL2
    );
    assert_eq!(
        CorpusRowPolicy::for_metric(Metric::Euclidean),
        CorpusRowPolicy::Keep
    );
    assert_eq!(
        CorpusRowPolicy::for_metric(Metric::DotProduct),
        CorpusRowPolicy::Keep
    );
    assert_eq!(
        CorpusRowPolicy::for_metric(Metric::Manhattan),
        CorpusRowPolicy::Keep
    );
}

#[test]
fn cosine_policy_drops_zero_and_near_zero_rows() {
    let mut rows = degenerate_rows();
    let report = apply_row_policy(&mut rows, CorpusRowPolicy::DropBelowMinL2).expect("cosine");
    assert_eq!(report.scanned_train_rows, 4);
    assert_eq!(report.scanned_query_rows, 3);
    assert_eq!(report.dropped_train_rows, 2);
    assert_eq!(report.dropped_query_rows, 1);
    assert_eq!(rows.train, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(rows.train_ids, vec!["doc0".to_string(), "doc3".to_string()]);
    assert_eq!(
        rows.train_meta,
        vec![
            Some(serde_json::json!({ "idx": 0 })),
            Some(serde_json::json!({ "idx": 3 })),
        ]
    );
    assert_eq!(
        rows.queries,
        vec![vec![1.0, 1.0], vec![MIN_L2_NORM_FOR_COSINE, 0.0]]
    );
    assert_eq!(rows.query_ids, vec!["q1".to_string(), "q2".to_string()]);
    assert_eq!(
        rows.neighbors.expect("neighbors"),
        vec![vec![0, 1, -1], vec![1, 0, -1]]
    );

    let mut empty = degenerate_rows();
    empty.train = vec![vec![0.0, 0.0], vec![MIN_L2_NORM_FOR_COSINE * 0.5, 0.0]];
    empty.train_ids = vec!["doc0".to_string(), "doc1".to_string()];
    empty.train_meta = vec![None, None];
    let error =
        apply_row_policy(&mut empty, CorpusRowPolicy::DropBelowMinL2).expect_err("all zero");
    assert!(error.to_string().contains("training"));
}

#[test]
fn cosine_policy_remaps_graded_judgments_with_the_neighbor_rows() {
    let mut rows = degenerate_rows();
    rows.judgments = Some(vec![
        super::QueryJudgments {
            query_id: "q0".to_string(),
            neighbors: vec![super::GradedNeighbor {
                index: 1,
                class: super::RelevanceClass::Support,
                score: None,
            }],
        },
        super::QueryJudgments {
            query_id: "q1".to_string(),
            neighbors: vec![
                super::GradedNeighbor {
                    index: 0,
                    class: super::RelevanceClass::Answer,
                    score: None,
                },
                super::GradedNeighbor {
                    index: 1,
                    class: super::RelevanceClass::Support,
                    score: None,
                },
                super::GradedNeighbor {
                    index: 3,
                    class: super::RelevanceClass::Answer,
                    score: None,
                },
            ],
        },
        super::QueryJudgments {
            query_id: "q2".to_string(),
            neighbors: vec![],
        },
    ]);
    apply_row_policy(&mut rows, CorpusRowPolicy::DropBelowMinL2).expect("cosine");
    let judgments = rows.judgments.expect("judgments");
    assert_eq!(judgments.len(), 2);
    assert_eq!(judgments[0].query_id, "q1");
    assert_eq!(super::positive_indices(&judgments[0].neighbors), vec![0, 1]);
    assert!(
        judgments[0]
            .neighbors
            .iter()
            .all(|neighbor| neighbor.class == super::RelevanceClass::Answer)
    );
    assert!(judgments[1].neighbors.is_empty());
}

#[test]
fn euclidean_policy_keeps_zero_rows() {
    let mut rows = degenerate_rows();
    let report = apply_row_policy(&mut rows, CorpusRowPolicy::Keep).expect("euclidean");
    assert_eq!(report.dropped_train_rows, 0);
    assert_eq!(report.dropped_query_rows, 0);
    assert_eq!(report.scanned_train_rows, 4);
    assert_eq!(rows.train.len(), 4);
    assert_eq!(rows.train[1], vec![0.0, 0.0]);
    assert_eq!(rows.queries[0], vec![0.0, 0.0]);
    assert_eq!(rows.neighbors.expect("neighbors")[1], vec![0, 1, 3]);
}

#[test]
fn smaller_cap_reuses_sanitized_cosine_prefix() {
    let dir = tempfile::tempdir().expect("temp");
    let mut rows = degenerate_rows();
    let report = apply_row_policy(&mut rows, CorpusRowPolicy::DropBelowMinL2).expect("sanitize");
    let full = angle_request(
        dir.path(),
        report.scanned_train_rows,
        report.scanned_query_rows,
    );
    let manifest = manifest_for(&full, &rows, &report);
    write_layout(&full.root(), manifest, &rows).expect("layout");

    let smaller = angle_request(dir.path(), 1, 1);
    let sliced = prepare_corpus_blocking(&smaller).expect("smaller cap reuses");
    assert!(!smaller.root().join("source").exists());
    assert_eq!(sliced.loaded.base, vec![vec![1.0, 0.0]]);
    assert_eq!(sliced.loaded.ids, vec!["doc0".to_string()]);
    assert_eq!(sliced.loaded.queries, vec![vec![1.0, 1.0]]);
    assert_eq!(
        sliced.loaded.neighbors.as_ref().expect("neighbors"),
        &vec![vec![0, 1, -1]]
    );
    assert_eq!(sliced.manifest().dropped_train_rows, 2);
    assert_eq!(sliced.manifest().dropped_query_rows, 1);
    assert_eq!(sliced.manifest().train_count, 2);
    assert_eq!(sliced.manifest().scanned_train_rows, 4);
    assert!(
        sliced
            .loaded
            .base
            .iter()
            .all(|vector| super::row_ok_for_cosine(vector))
    );
    assert!(
        sliced
            .loaded
            .queries
            .iter()
            .all(|vector| super::row_ok_for_cosine(vector))
    );

    let again = prepare_corpus_blocking(&full).expect("same scanned cap");
    assert_eq!(again.loaded.base.len(), 2);
    assert_eq!(again.loaded.queries.len(), 2);
    assert_eq!(again.loaded.ids[1], "doc3");
    assert!(!full.root().join("source").exists());
}

#[test]
fn legacy_cosine_layout_is_not_reused() {
    let dir = tempfile::tempdir().expect("temp");
    let request = angle_request(dir.path(), 1, 1);
    let mut manifest = manifest_for(
        &request,
        &degenerate_rows(),
        &super::sanitize::RowPolicyReport {
            policy: CorpusRowPolicy::Unspecified,
            scanned_train_rows: 0,
            scanned_query_rows: 0,
            dropped_train_rows: 0,
            dropped_query_rows: 0,
        },
    );
    manifest.row_policy = CorpusRowPolicy::Unspecified;
    manifest.train_count = 10_000;
    manifest.query_count = 128;
    manifest.scanned_train_rows = 0;
    manifest.scanned_query_rows = 0;
    assert!(
        !layout_reusable(&manifest, &request),
        "a pre-sanitize cosine cache must be rebuilt even for a cap below the first zero row"
    );

    manifest.metric = Metric::Euclidean;
    manifest.row_policy = CorpusRowPolicy::Unspecified;
    let euclidean = CorpusRequest {
        selector: CorpusSelector::Hdf5Url {
            name: "sift-fixture".to_string(),
            url: "http://127.0.0.1:1/sift-128-euclidean.hdf5".to_string(),
        },
        ..request
    };
    manifest.name = euclidean.layout_name();
    manifest.revision = euclidean.revision();
    manifest.metric = euclidean.default_metric();
    assert!(layout_reusable(&manifest, &euclidean));
}

/// A cache is reusable only when `grade_policy` matches.
///
/// Caps and row policy already cover this layout. Removing
/// `manifest.grade_policy == request.grade_policy` from `same_source` makes
/// the mismatch asserts fail.
#[test]
fn mismatched_grade_policy_is_not_a_reusable_layout() {
    let dir = tempfile::tempdir().expect("temp");
    let (stored, binary) = euclidean_layout(dir.path(), super::GradePolicy::BinaryAnswer);
    assert!(
        layout_reusable(&stored, &binary),
        "matching binary_answer reuses the layout"
    );

    let mismatches = [
        super::GradePolicy::Explicit,
        super::GradePolicy::Threshold {
            answer_min: super::DEFAULT_ANSWER_MIN_SCORE,
        },
    ];
    for policy in mismatches {
        let request = CorpusRequest {
            grade_policy: policy,
            ..binary.clone()
        };
        assert!(
            !layout_reusable(&stored, &request),
            "grade_policy {policy:?} must not reuse a binary_answer layout"
        );
    }

    let (threshold_layout, threshold) = euclidean_layout(
        dir.path(),
        super::GradePolicy::Threshold {
            answer_min: super::DEFAULT_ANSWER_MIN_SCORE,
        },
    );
    assert!(
        layout_reusable(&threshold_layout, &threshold),
        "matching threshold reuses the layout"
    );
    let tighter = CorpusRequest {
        grade_policy: super::GradePolicy::Threshold {
            answer_min: super::DEFAULT_ANSWER_MIN_SCORE + 1,
        },
        ..threshold
    };
    assert!(
        !layout_reusable(&threshold_layout, &tighter),
        "a different answer_min must not reuse the threshold layout"
    );
}

#[test]
fn code_jsonl_normalizes_ast_chunks_qrels_and_reuses_the_layout() {
    let dir = tempfile::tempdir().expect("temp");
    let staged = dir.path().join("staged");
    std::fs::create_dir_all(&staged).expect("staged");
    write_lines(
        &staged.join("corpus.jsonl"),
        &[
            r#"{"_id":"c1","language":"python","title":"pair","text":"def alpha(x):\n    return x\n\ndef beta(y):\n    return y\n"}"#,
            r#"{"_id":"c2","language":"python","title":"gamma","text":"def gamma(z):\n    return z\n"}"#,
            r#"{"_id":"c3","language":"python","title":"blank","text":"   "}"#,
        ],
    );
    write_lines(
        &staged.join("queries.jsonl"),
        &[
            r#"{"_id":"q1","text":"find alpha"}"#,
            r#"{"_id":"q2","text":"find gamma"}"#,
            r#"{"_id":"q3","text":"unused filler query"}"#,
        ],
    );
    write_lines(
        &staged.join("qrels.jsonl"),
        &[
            r#"{"query-id":"q1","corpus-id":"c1","score":1}"#,
            r#"{"query-id":"q2","corpus-id":"c2","score":1}"#,
        ],
    );

    let cache = dir.path().join("cache");
    let request = code_request(&cache, &staged, 10, 2);
    let prepared = prepare_corpus_blocking(&request).expect("normalize");
    assert_eq!(prepared.loaded.dimension, super::CODE_FEATURE_DIMENSION);
    assert_eq!(prepared.manifest().metric, Metric::Cosine);
    assert_eq!(prepared.manifest().format, super::CorpusFormat::Code);
    assert_eq!(
        prepared.manifest().row_policy,
        CorpusRowPolicy::DropBelowMinL2
    );
    assert!(prepared.manifest().dropped_train_rows >= 1);
    assert_eq!(prepared.loaded.base.len(), 3);
    assert_eq!(
        prepared.loaded.ids,
        vec!["c1#0".to_string(), "c1#1".to_string(), "c2".to_string()]
    );
    let symbols: Vec<_> = prepared
        .loaded
        .train_meta
        .iter()
        .map(|row| {
            row.as_ref()
                .and_then(|value| value.get("symbol"))
                .and_then(|value| value.as_str())
                .unwrap_or("")
                .to_string()
        })
        .collect();
    assert_eq!(symbols, vec!["alpha", "beta", "gamma"]);
    assert!(prepared.manifest().has_meta);
    assert!(prepared.manifest().has_qrels);
    assert_eq!(prepared.loaded.queries.len(), 2);
    assert_eq!(
        prepared.loaded.neighbors.as_deref(),
        Some(&[vec![0, 1], vec![2, -1]][..])
    );
    let judgments = prepared.loaded.judgments.as_ref().expect("judgments");
    assert!(
        judgments
            .iter()
            .flat_map(|row| row.neighbors.iter())
            .all(|neighbor| neighbor.class == super::RelevanceClass::Answer)
    );
    assert!(judgments.iter().all(|row| {
        row.neighbors
            .iter()
            .all(|neighbor| neighbor.class != super::RelevanceClass::Support)
    }));
    assert!(prepared.normalized.root.join(super::QRELS_FILE).is_file());
    let norm: f32 = prepared.loaded.base[0]
        .iter()
        .map(|value| value * value)
        .sum();
    assert!((norm - 1.0).abs() < 1e-4, "feature norm squared {norm}");

    let again = prepare_corpus_blocking(&request).expect("reuse");
    assert_eq!(again.loaded.base, prepared.loaded.base);
    assert_eq!(
        again.manifest().train_count,
        prepared.manifest().train_count
    );

    let sliced = prepare_corpus_blocking(&code_request(&cache, &staged, 1, 1)).expect("slice");
    assert_eq!(sliced.loaded.base.len(), 1);
    assert_eq!(sliced.loaded.ids[0], "c1#0");
    assert_eq!(
        sliced.manifest().train_count,
        prepared.manifest().train_count
    );
}

#[test]
fn code_select_keeps_a_late_labeled_row_inside_the_cap() {
    let dir = tempfile::tempdir().expect("temp");
    let staged = dir.path().join("staged");
    std::fs::create_dir_all(&staged).expect("staged");
    write_lines(
        &staged.join("corpus.jsonl"),
        &[
            r#"{"_id":"filler-a","language":"python","text":"def filler_a():\n    return 0\n"}"#,
            r#"{"_id":"filler-b","language":"python","text":"def filler_b():\n    return 0\n"}"#,
            r#"{"_id":"target","language":"python","text":"def target():\n    return 1\n"}"#,
        ],
    );
    write_lines(
        &staged.join("queries.jsonl"),
        &[r#"{"_id":"q-target","text":"the target function"}"#],
    );
    write_lines(
        &staged.join("qrels.jsonl"),
        &[r#"{"query-id":"q-target","corpus-id":"target","score":1}"#],
    );
    let prepared = prepare_corpus_blocking(&code_request(&dir.path().join("cache"), &staged, 2, 1))
        .expect("normalize");
    assert_eq!(
        prepared.loaded.ids,
        vec!["target".to_string(), "filler-a".to_string()]
    );
    assert_eq!(prepared.loaded.neighbors.as_deref(), Some(&[vec![0]][..]));
}

#[test]
fn explicit_support_qrel_stays_out_of_the_answer_class() {
    let dir = tempfile::tempdir().expect("temp");
    let staged = dir.path().join("staged");
    std::fs::create_dir_all(&staged).expect("staged");
    write_lines(
        &staged.join("corpus.jsonl"),
        &[
            r#"{"_id":"answer-doc","language":"python","text":"def answer():\n    return 1\n"}"#,
            r#"{"_id":"support-doc","language":"python","text":"def support():\n    return 2\n"}"#,
        ],
    );
    write_lines(
        &staged.join("queries.jsonl"),
        &[r#"{"_id":"q","text":"find the answer"}"#],
    );
    write_lines(
        &staged.join("qrels.jsonl"),
        &[
            r#"{"query-id":"q","corpus-id":"answer-doc","score":1,"class":"answer"}"#,
            r#"{"query-id":"q","corpus-id":"support-doc","score":1,"role":"support"}"#,
        ],
    );
    let mut request = code_request(&dir.path().join("cache"), &staged, 4, 1);
    request.grade_policy = super::GradePolicy::Explicit;
    let prepared = prepare_corpus_blocking(&request).expect("normalize");
    assert_eq!(
        prepared.loaded.neighbors.as_deref(),
        Some(&[vec![0, 1]][..])
    );
    let row = &prepared.loaded.judgments.as_ref().expect("judgments")[0];
    assert_eq!(row.neighbors[0].class, super::RelevanceClass::Answer);
    assert_eq!(row.neighbors[1].class, super::RelevanceClass::Support);
    let graded = super::graded_recall_at_k(&[1], &row.neighbors, 1);
    assert!((graded.answer_at_k.expect("answer") - 0.0).abs() < 1e-9);
    assert!((graded.support_at_k.expect("support") - 1.0).abs() < 1e-9);
}

#[test]
fn threshold_policy_splits_low_scores_into_support() {
    let dir = tempfile::tempdir().expect("temp");
    let staged = dir.path().join("staged");
    std::fs::create_dir_all(&staged).expect("staged");
    write_lines(
        &staged.join("corpus.jsonl"),
        &[
            r#"{"_id":"high","language":"python","text":"def high():\n    return 1\n"}"#,
            r#"{"_id":"low","language":"python","text":"def low():\n    return 0\n"}"#,
        ],
    );
    write_lines(
        &staged.join("queries.jsonl"),
        &[r#"{"_id":"q","text":"graded query"}"#],
    );
    write_lines(
        &staged.join("qrels.jsonl"),
        &[
            r#"{"query-id":"q","corpus-id":"low","score":1}"#,
            r#"{"query-id":"q","corpus-id":"high","score":2}"#,
        ],
    );
    let mut request = code_request(&dir.path().join("cache-threshold"), &staged, 4, 1);
    request.grade_policy = super::GradePolicy::Threshold { answer_min: 2 };
    let prepared = prepare_corpus_blocking(&request).expect("normalize");
    let row = &prepared.loaded.judgments.as_ref().expect("judgments")[0];
    assert_eq!(row.neighbors[0].class, super::RelevanceClass::Support);
    assert_eq!(row.neighbors[1].class, super::RelevanceClass::Answer);
    assert_eq!(prepared.manifest().grade_policy, request.grade_policy);
}

#[test]
fn code_parquet_fixture_uses_the_hugging_face_paths() {
    let dir = tempfile::tempdir().expect("temp");
    let staged = dir.path().join("staged");
    write_code_parquet(
        &staged.join("corpus/corpus-00000-of-00001.parquet"),
        &["_id", "text", "title", "language"],
        &[(
            "d1",
            "def alpha():\n    return 1\n",
            Some("alpha"),
            Some("python"),
        )],
    );
    write_code_parquet(
        &staged.join("queries/queries-00000-of-00001.parquet"),
        &["_id", "text", "title", "language"],
        &[("q1", "return one", None, None)],
    );
    write_qrel_parquet(&staged.join("data/test-00000-of-00001.parquet"));
    let prepared = prepare_corpus_blocking(&code_request(&dir.path().join("cache"), &staged, 4, 1))
        .expect("parquet");
    assert_eq!(prepared.loaded.ids, vec!["d1".to_string()]);
    assert_eq!(prepared.loaded.query_ids, vec!["q1".to_string()]);
    assert_eq!(prepared.manifest().has_qrels, true);
    assert_eq!(
        prepared.loaded.train_meta[0]
            .as_ref()
            .and_then(|value| value.get("symbol"))
            .and_then(|value| value.as_str()),
        Some("alpha")
    );
}

#[test]
#[ignore = "set DRIPSTONE_COSQA_STAGE to a CoIR CoSQA tree (corpus/, queries/, data/)"]
fn code_staged_cosqa_sample_normalizes() {
    let root = std::env::var("DRIPSTONE_COSQA_STAGE").expect("DRIPSTONE_COSQA_STAGE");
    let dir = tempfile::tempdir().expect("temp");
    let prepared = prepare_corpus_blocking(&code_request(
        &dir.path().join("cache"),
        Path::new(&root),
        32,
        4,
    ))
    .expect("cosqa");
    assert_eq!(prepared.loaded.dimension, super::CODE_FEATURE_DIMENSION);
    assert!(!prepared.loaded.ids.is_empty());
    assert!(prepared.loaded.ids.len() <= 32);
    assert_eq!(prepared.loaded.queries.len(), 4);
    let has_symbol = prepared.loaded.train_meta.iter().any(|row| {
        row.as_ref()
            .and_then(|value| value.get("symbol"))
            .and_then(|value| value.as_str())
            .is_some_and(|symbol| !symbol.is_empty())
    });
    assert!(
        has_symbol,
        "expected at least one AST symbol in the CoSQA sample"
    );
}

#[test]
fn code_dictionary_parquet_strings_decode() {
    let dir = tempfile::tempdir().expect("temp");
    let staged = dir.path().join("staged");
    write_dictionary_code_parquet(&staged.join("corpus.parquet"));
    write_lines(
        &staged.join("queries.jsonl"),
        &[r#"{"_id":"q-dict","text":"find delta"}"#],
    );
    let prepared = prepare_corpus_blocking(&code_request(&dir.path().join("cache"), &staged, 4, 1))
        .expect("dictionary parquet");
    assert_eq!(prepared.loaded.ids, vec!["d-dict".to_string()]);
    assert_eq!(
        prepared.loaded.train_meta[0]
            .as_ref()
            .and_then(|value| value.get("symbol"))
            .and_then(|value| value.as_str()),
        Some("delta")
    );
}

trait ManifestView {
    fn manifest(&self) -> &super::CorpusManifest;
}

impl ManifestView for super::PreparedCorpus {
    fn manifest(&self) -> &super::CorpusManifest {
        &self.loaded.manifest
    }
}

fn degenerate_rows() -> MaterializedRows {
    let near_zero = MIN_L2_NORM_FOR_COSINE * 0.5;
    MaterializedRows {
        train: vec![
            vec![1.0, 0.0],
            vec![0.0, 0.0],
            vec![near_zero, 0.0],
            vec![0.0, 1.0],
        ],
        train_ids: vec![
            "doc0".to_string(),
            "doc1".to_string(),
            "doc2".to_string(),
            "doc3".to_string(),
        ],
        train_meta: vec![
            Some(serde_json::json!({ "idx": 0 })),
            Some(serde_json::json!({ "idx": 1 })),
            Some(serde_json::json!({ "idx": 2 })),
            Some(serde_json::json!({ "idx": 3 })),
        ],
        queries: vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![MIN_L2_NORM_FOR_COSINE, 0.0],
        ],
        query_ids: vec!["q0".to_string(), "q1".to_string(), "q2".to_string()],
        neighbors: Some(vec![vec![0, 1, 3], vec![0, 1, 3], vec![3, 0, 1]]),
        neighbors_width: 3,
        dimension: 2,
        source_exhausted: false,
        source_url: None,
        source_path: None,
        has_qrels: false,
        judgments: None,
    }
}

fn angle_request(cache: &Path, train_max: usize, query_max: usize) -> CorpusRequest {
    CorpusRequest {
        selector: CorpusSelector::Hdf5Url {
            name: "angle-fixture".to_string(),
            url: "http://127.0.0.1:1/angle-fixture.hdf5".to_string(),
        },
        caps: super::CorpusCaps {
            train_max,
            query_max,
        },
        neighbors_k: 3,
        cache_dir: cache.to_path_buf(),
        emb_column: EMB_COLUMN.to_string(),
        parquet_recursive: false,
        shard_max: 1,
        grade_policy: super::GradePolicy::BinaryAnswer,
    }
}

/// Euclidean layout whose caps and row policy already allow reuse.
fn euclidean_layout(cache: &Path, policy: super::GradePolicy) -> (CorpusManifest, CorpusRequest) {
    let request = CorpusRequest {
        selector: CorpusSelector::Hdf5Url {
            name: "sift-fixture".to_string(),
            url: "http://127.0.0.1:1/sift-128-euclidean.hdf5".to_string(),
        },
        grade_policy: policy,
        ..angle_request(cache, 1, 1)
    };
    let mut manifest = manifest_for(
        &request,
        &degenerate_rows(),
        &super::sanitize::RowPolicyReport {
            policy: CorpusRowPolicy::Unspecified,
            scanned_train_rows: 0,
            scanned_query_rows: 0,
            dropped_train_rows: 0,
            dropped_query_rows: 0,
        },
    );
    manifest.row_policy = CorpusRowPolicy::Unspecified;
    manifest.train_count = 10_000;
    manifest.query_count = 128;
    manifest.scanned_train_rows = 0;
    manifest.scanned_query_rows = 0;
    (manifest, request)
}

fn manifest_for(
    request: &CorpusRequest,
    rows: &MaterializedRows,
    report: &super::sanitize::RowPolicyReport,
) -> CorpusManifest {
    CorpusManifest {
        name: request.layout_name(),
        revision: request.revision(),
        format: request.format(),
        dimension: rows.dimension,
        metric: request.default_metric(),
        train_count: rows.train.len(),
        query_count: rows.queries.len(),
        row_policy: report.policy,
        scanned_train_rows: report.scanned_train_rows,
        scanned_query_rows: report.scanned_query_rows,
        dropped_train_rows: report.dropped_train_rows,
        dropped_query_rows: report.dropped_query_rows,
        train_cap: request.caps.train_max,
        query_cap: request.caps.query_max,
        neighbors_k: 0,
        neighbors_width: rows.neighbors_width,
        source_url: None,
        source_path: None,
        has_meta: false,
        has_neighbors: false,
        has_qrels: false,
        grade_policy: request.grade_policy,
        source_exhausted: rows.source_exhausted,
        emb_column: None,
        language: None,
    }
}

fn parquet_request(
    cache: &std::path::Path,
    parquet: &std::path::Path,
    train_max: usize,
    query_max: usize,
) -> CorpusRequest {
    CorpusRequest {
        selector: CorpusSelector::ParquetDir {
            name: "fixture".to_string(),
            path: parquet.to_path_buf(),
        },
        caps: super::CorpusCaps {
            train_max,
            query_max,
        },
        neighbors_k: 10,
        cache_dir: cache.to_path_buf(),
        emb_column: EMB_COLUMN.to_string(),
        parquet_recursive: false,
        shard_max: 1,
        grade_policy: super::GradePolicy::BinaryAnswer,
    }
}

fn write_fixture(path: &std::path::Path) {
    let values = Float32Array::from(vec![
        1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
    ]);
    let item = Arc::new(Field::new("item", DataType::Float32, false));
    let embeddings =
        FixedSizeListArray::try_new(Arc::clone(&item), 2, Arc::new(values), None).expect("list");
    let schema = Arc::new(Schema::new(vec![
        Field::new("_id", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, true),
        Field::new("emb", DataType::FixedSizeList(item, 2), false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e", "f"])),
            Arc::new(StringArray::from(vec![
                Some("t0"),
                Some("t1"),
                None,
                Some("t3"),
                Some("t4"),
                Some("t5"),
            ])),
            Arc::new(embeddings),
        ],
    )
    .expect("batch");
    let file = File::create(path).expect("create");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}

fn code_request(
    cache: &std::path::Path,
    staged: &std::path::Path,
    train_max: usize,
    query_max: usize,
) -> CorpusRequest {
    CorpusRequest {
        selector: CorpusSelector::CodeDir {
            name: "code".to_string(),
            path: staged.to_path_buf(),
        },
        caps: super::CorpusCaps {
            train_max,
            query_max,
        },
        neighbors_k: 10,
        cache_dir: cache.to_path_buf(),
        emb_column: EMB_COLUMN.to_string(),
        parquet_recursive: false,
        shard_max: 1,
        grade_policy: super::GradePolicy::BinaryAnswer,
    }
}

fn write_lines(path: &std::path::Path, lines: &[&str]) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent");
    }
    std::fs::write(path, lines.join("\n")).expect("lines");
}

fn write_code_parquet(
    path: &std::path::Path,
    _columns: &[&str],
    rows: &[(&str, &str, Option<&str>, Option<&str>)],
) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent");
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("_id", DataType::Utf8, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("title", DataType::Utf8, true),
        Field::new("language", DataType::Utf8, true),
    ]));
    let ids: Vec<&str> = rows.iter().map(|row| row.0).collect();
    let texts: Vec<&str> = rows.iter().map(|row| row.1).collect();
    let titles: Vec<Option<&str>> = rows.iter().map(|row| row.2).collect();
    let languages: Vec<Option<&str>> = rows.iter().map(|row| row.3).collect();
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(texts)),
            Arc::new(StringArray::from(titles)),
            Arc::new(StringArray::from(languages)),
        ],
    )
    .expect("code batch");
    let file = File::create(path).expect("create");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}

fn write_dictionary_code_parquet(path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent");
    }
    let dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let schema = Arc::new(Schema::new(vec![
        Field::new("_id", dict.clone(), false),
        Field::new("text", dict.clone(), false),
        Field::new("title", dict.clone(), true),
        Field::new("language", dict, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(dict_utf8(&["d-dict"], &[0])),
            Arc::new(dict_utf8(&["def delta():\n    return 1\n"], &[0])),
            Arc::new(dict_utf8(&["delta"], &[0])),
            Arc::new(dict_utf8(&["PYTHON"], &[0])),
        ],
    )
    .expect("dictionary batch");
    let file = File::create(path).expect("create");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}

fn dict_utf8(values: &[&str], keys: &[i32]) -> DictionaryArray<Int32Type> {
    DictionaryArray::<Int32Type>::try_new(
        Int32Array::from(keys.to_vec()),
        Arc::new(StringArray::from(values.to_vec())),
    )
    .expect("dictionary")
}

fn write_qrel_parquet(path: &std::path::Path) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent");
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("query-id", DataType::Utf8, false),
        Field::new("corpus-id", DataType::Utf8, false),
        Field::new("score", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(StringArray::from(vec!["q1"])),
            Arc::new(StringArray::from(vec!["d1"])),
            Arc::new(Int64Array::from(vec![1_i64])),
        ],
    )
    .expect("qrel batch");
    let file = File::create(path).expect("create");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("writer");
    writer.write(&batch).expect("write");
    writer.close().expect("close");
}
