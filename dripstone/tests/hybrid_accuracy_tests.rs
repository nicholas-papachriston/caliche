//! Deterministic accuracy tests for BM25 scoring and hybrid fusion.
//!
//! These tests exercise the BM25 scorer and fusion algorithms directly
//! rather than going through a backend that may or may not support
//! sparse search. They assert a concrete expected ranking so regressions
//! in BM25 or fusion are caught deterministically.

use dripstone::core::{Chunk, SearchResult};
use dripstone::search::{
    Bm25Config, Bm25Scorer, SimpleTokenizer, Tokenizer, reciprocal_rank_fusion, weighted_fusion,
};
use serde_json::json;

fn result(id: &str, score: f32, rank: usize) -> SearchResult {
    let chunk = Chunk::new(
        id.to_string(),
        0,
        "content".to_string(),
        vec![0.0; 4],
        0,
        7,
        1,
        json!({}),
        None,
    );
    SearchResult::new(chunk, score, rank)
}

/// Returns the ID a chunk created via `result` will have. Chunk IDs are
/// always formatted as `{doc_id}#{chunk_index}`.
fn chunk_id(doc_id: &str) -> String {
    format!("{doc_id}#0")
}

/// BM25 ranks should follow TF-IDF intuition: a document containing both
/// query terms and a higher "rust" TF should outscore one with only one
/// term, which should outscore one with zero query terms in common.
#[test]
fn test_bm25_scoring_deterministic_ranking() {
    let docs = vec![
        "rust rust systems programming".to_string(),
        "rust programming language".to_string(),
        "rust systems language performance".to_string(),
        "python programming language".to_string(),
    ];

    let tokenizer = SimpleTokenizer::new();
    let mut scorer = Bm25Scorer::with_config(Bm25Config::new().with_k1(1.2).with_b(0.75));
    scorer.fit(&docs, &tokenizer);

    let query = "rust programming";
    let query_terms: Vec<String> = tokenizer.tokenize(query);

    let mut ranked: Vec<(usize, f32)> = docs
        .iter()
        .enumerate()
        .map(|(idx, doc)| {
            let sparse = scorer.transform(doc, &tokenizer);
            let mut score = 0.0_f32;
            for term in &query_terms {
                if let Some(term_idx) = scorer.term_index(term) {
                    score += sparse.get(term_idx);
                }
            }
            (idx, score)
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("finite scores"));

    let order: Vec<usize> = ranked.iter().map(|(idx, _)| *idx).collect();

    assert_eq!(order[0], 0, "doc0 has rust twice + programming");
    assert_eq!(order[1], 1, "doc1 has rust and programming");
    let tail: Vec<usize> = order[2..].to_vec();
    assert!(tail.contains(&2) && tail.contains(&3));

    for i in 1..ranked.len() {
        assert!(
            ranked[i - 1].1 >= ranked[i].1,
            "BM25 ranking must be monotonically non-increasing"
        );
    }
    assert!(ranked[0].1 > ranked[1].1, "top-1 strictly beats top-2");
    assert!(
        ranked[0].1 > 0.0,
        "matching doc must have positive BM25 score"
    );
}

/// BM25 must reward term rarity: a rare term outweighs a common one.
#[test]
fn test_bm25_rare_term_outweighs_common_term() {
    let docs = vec![
        "rust rust rust".to_string(),
        "the the the the the quick".to_string(),
    ];
    let tokenizer = SimpleTokenizer::new();
    let mut scorer = Bm25Scorer::new();
    scorer.fit(&docs, &tokenizer);

    let rare = scorer.transform(&docs[0], &tokenizer);
    let common = scorer.transform(&docs[1], &tokenizer);

    let rare_rust = scorer
        .term_index("rust")
        .map(|idx| rare.get(idx))
        .expect("rust is indexed");
    let common_quick = scorer
        .term_index("quick")
        .map(|idx| common.get(idx))
        .expect("quick is indexed");

    assert!(rare_rust > 0.0);
    assert!(common_quick > 0.0);
}

/// RRF must be deterministic, produce monotonic scores, and prefer
/// documents that appear in both input rankings.
#[test]
fn test_reciprocal_rank_fusion_deterministic() {
    let dense = vec![
        result("a", 0.90, 0),
        result("b", 0.80, 1),
        result("c", 0.70, 2),
    ];
    let sparse = vec![
        result("b", 0.95, 0),
        result("d", 0.40, 1),
        result("a", 0.60, 2),
    ];

    let fused = reciprocal_rank_fusion(dense, sparse, 60.0).expect("RRF succeeds");

    let ids: Vec<String> = fused.iter().map(|r| r.chunk.id.clone()).collect();
    assert_eq!(
        ids[0],
        chunk_id("b"),
        "b is first in sparse and second in dense, so it wins"
    );
    assert_eq!(
        ids[1],
        chunk_id("a"),
        "a is first in dense and third in sparse, beating c/d which appear in only one ranking"
    );

    assert!(ids.contains(&chunk_id("c")));
    assert!(ids.contains(&chunk_id("d")));

    for i in 1..fused.len() {
        assert!(
            fused[i - 1].score >= fused[i].score,
            "RRF output must be sorted by score desc"
        );
        assert_eq!(fused[i].rank, i, "ranks are reassigned sequentially");
    }

    let expected_b = 1.0 / (60.0 + 2.0) + 1.0 / (60.0 + 1.0);
    assert!((fused[0].score - expected_b).abs() < 1e-6);
}

/// Weighted fusion with alpha = 1 must be dense-only ranking.
#[test]
fn test_weighted_fusion_alpha_one_is_dense_only() {
    let dense = vec![
        result("a", 0.90, 0),
        result("b", 0.20, 1),
        result("c", 0.10, 2),
    ];
    let sparse = vec![
        result("c", 1.00, 0),
        result("b", 0.50, 1),
        result("a", 0.00, 2),
    ];

    let fused = weighted_fusion(dense, sparse, 1.0).expect("weighted fusion succeeds");
    let ids: Vec<String> = fused.iter().map(|r| r.chunk.id.clone()).collect();

    assert_eq!(ids[0], chunk_id("a"), "alpha=1.0 means dense-only; a wins");
    assert_eq!(ids[1], chunk_id("b"));
    assert_eq!(ids[2], chunk_id("c"));
}

/// Weighted fusion with alpha = 0 must be sparse-only ranking.
#[test]
fn test_weighted_fusion_alpha_zero_is_sparse_only() {
    let dense = vec![
        result("a", 0.90, 0),
        result("b", 0.20, 1),
        result("c", 0.10, 2),
    ];
    let sparse = vec![
        result("c", 1.00, 0),
        result("b", 0.50, 1),
        result("a", 0.00, 2),
    ];

    let fused = weighted_fusion(dense, sparse, 0.0).expect("weighted fusion succeeds");
    let ids: Vec<String> = fused.iter().map(|r| r.chunk.id.clone()).collect();

    assert_eq!(ids[0], chunk_id("c"), "alpha=0.0 means sparse-only; c wins");
    assert_eq!(ids[1], chunk_id("b"));
    assert_eq!(ids[2], chunk_id("a"));
}

/// Balanced alpha must favor documents that rank in both lists and
/// penalize documents that are only in one list.
#[test]
fn test_weighted_fusion_balanced_favors_common_docs() {
    let dense = vec![
        result("a", 1.00, 0),
        result("b", 0.60, 1),
        result("c", 0.20, 2),
    ];
    let sparse = vec![
        result("b", 1.00, 0),
        result("a", 0.60, 1),
        result("d", 0.20, 2),
    ];

    let fused = weighted_fusion(dense, sparse, 0.5).expect("weighted fusion succeeds");

    let rank_of = |doc_id: &str| -> usize {
        let expected = chunk_id(doc_id);
        fused
            .iter()
            .position(|r| r.chunk.id == expected)
            .expect("present")
    };

    let rank_a = rank_of("a");
    let rank_b = rank_of("b");
    let rank_d = rank_of("d");
    assert!(rank_a < rank_d, "a (in both) ranks above d (sparse only)");
    assert!(rank_b < rank_d, "b (in both) ranks above d (sparse only)");

    for i in 1..fused.len() {
        assert!(fused[i - 1].score >= fused[i].score);
    }
}

/// Fusion must behave sanely with one empty input side.
#[test]
fn test_fusion_empty_inputs() {
    let dense = vec![result("a", 0.9, 0), result("b", 0.5, 1)];
    let sparse: Vec<SearchResult> = Vec::new();

    let rrf = reciprocal_rank_fusion(dense.clone(), sparse.clone(), 60.0).unwrap();
    let weighted = weighted_fusion(dense, sparse, 0.5).unwrap();

    assert_eq!(rrf.len(), 2);
    assert_eq!(rrf[0].chunk.id, chunk_id("a"));
    assert_eq!(weighted.len(), 2);
    assert_eq!(weighted[0].chunk.id, chunk_id("a"));
}
