#![expect(
    dead_code,
    reason = "each test binary includes this module and uses a different subset of it"
)]

use dripstone::core::{Chunk, SearchResult};
use serde_json::{Value, json};

/// Deterministic, non-normalized corpus generator with low collision rates so ANN recall is
/// dominated by index quality instead of top-k ties from repeated vectors.
pub fn build_vector(seed: usize, dimension: usize) -> Vec<f32> {
    let mut vector = Vec::with_capacity(dimension);
    let seed_f = seed as f32 + 1.0;
    for offset in 0..dimension {
        let position = offset as f32 + 1.0;
        let phase = seed_f * 0.013 + position * 0.017;
        let envelope = seed_f * 0.007 - position * 0.011;
        let bias = (seed % 11) as f32 * 0.001;
        vector.push(0.5f32.mul_add(envelope.cos(), phase.sin()) + bias);
    }
    vector
}

pub fn build_chunk(id: String, namespace: &str, seed: usize, dimension: usize) -> Chunk {
    let vector = build_vector(seed, dimension);
    let content = format!("Content for {id}");
    let metadata = json!({
        "segment": if seed.is_multiple_of(2) { "alpha" } else { "beta" },
        "ordinal": seed,
    });

    Chunk::new(
        id,
        0,
        &content,
        vector,
        0,
        content.len(),
        1,
        metadata,
        Some(namespace.to_string()),
    )
}

/// Build edge case vectors for testing
pub fn build_edge_case_vectors() -> Vec<(String, Vec<f32>)> {
    vec![
        ("single_element".to_string(), vec![0.5]),
        ("all_zeros".to_string(), vec![0.0; 128]),
        ("all_ones".to_string(), vec![1.0; 128]),
        (
            "alternating".to_string(),
            (0..128)
                .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
                .collect(),
        ),
        ("very_small".to_string(), vec![f32::EPSILON; 128]),
        ("large_dimension".to_string(), vec![0.5; 2048]),
    ]
}

/// Build special metadata for edge case testing
pub fn build_special_metadata() -> Vec<(String, Value)> {
    vec![
        (
            "unicode".to_string(),
            json!({"text": "Hello 世界", "emoji": "test"}),
        ),
        (
            "nested".to_string(),
            json!({"level1": {"level2": {"level3": "deep"}}}),
        ),
        ("null_value".to_string(), json!({"nullable": null})),
        ("empty_string".to_string(), json!({"empty": ""})),
        (
            "large_string".to_string(),
            json!({"data": "x".repeat(10000)}),
        ),
        ("array".to_string(), json!({"items": [1, 2, 3, 4, 5]})),
        (
            "mixed_types".to_string(),
            json!({"str": "text", "num": 42, "bool": true, "arr": [1, 2]}),
        ),
        (
            "special_chars".to_string(),
            json!({"chars": "!@#$%^&*()[]{}|\\:;\"'<>,.?/~`"}),
        ),
    ]
}

/// Build concurrent chunks for parallel testing
pub fn build_concurrent_chunks(
    count: usize,
    namespace: &str,
    dimension: usize,
    base_seed: usize,
) -> Vec<Chunk> {
    (0..count)
        .map(|i| {
            build_chunk(
                format!("concurrent-{i}"),
                namespace,
                base_seed + i,
                dimension,
            )
        })
        .collect()
}

/// Assert that two chunks are equivalent
#[allow(
    dead_code,
    reason = "shared test helper used by selected integration targets"
)]
pub fn assert_chunks_equivalent(a: &Chunk, b: &Chunk, tolerance: f32) {
    assert_eq!(a.id, b.id, "Chunk IDs should match");
    assert_eq!(a.namespace, b.namespace, "Namespaces should match");
    assert_eq!(
        a.vector.len(),
        b.vector.len(),
        "Vector dimensions should match"
    );

    for (i, (va, vb)) in a.vector.iter().zip(b.vector.iter()).enumerate() {
        assert!(
            (va - vb).abs() < tolerance,
            "Vector element {i} differs: {va} vs {vb} (tolerance: {tolerance})"
        );
    }

    assert_eq!(a.metadata, b.metadata, "Metadata should match");
}

/// Assert that search results are similar across backends
pub fn assert_search_results_similar(
    results_a: &[SearchResult],
    results_b: &[SearchResult],
    score_tolerance: f32,
) {
    assert_eq!(
        results_a.len(),
        results_b.len(),
        "Search result counts should match"
    );

    for (i, (ra, rb)) in results_a.iter().zip(results_b.iter()).enumerate() {
        assert_eq!(
            ra.chunk.id, rb.chunk.id,
            "Result {} ID mismatch: {} vs {}",
            i, ra.chunk.id, rb.chunk.id
        );

        assert!(
            (ra.score - rb.score).abs() < score_tolerance,
            "Result {} score differs: {} vs {} (tolerance: {})",
            i,
            ra.score,
            rb.score,
            score_tolerance
        );
    }
}

/// Build special character IDs for testing
pub fn build_special_ids() -> Vec<String> {
    vec![
        "simple-id".to_string(),
        "id_with_underscores".to_string(),
        "id-with-dashes".to_string(),
        "id.with.dots".to_string(),
        "id:with:colons".to_string(),
        "id/with/slashes".to_string(),
        "unicode-世界-id".to_string(),
        "emoji-test-id".to_string(),
        "spaces in id".to_string(),
        "very-long-".to_string() + &"x".repeat(500),
    ]
}

// -----------------------------------------------------------------------------
// Docker Compose stack credentials (`docker-compose.example.yml` defaults)
// -----------------------------------------------------------------------------
// Set `EMBEDDING_*_API_KEY=` (empty) or `STACK_*` to empty to talk to services without auth.
// Each integration test binary uses only a subset of these helpers.

/// Redis / Valkey / Sentinel / Cluster password (`STACK_REDIS_PASSWORD`, default `dripstone_dev_redis`).
pub fn stack_redis_password() -> String {
    std::env::var("STACK_REDIS_PASSWORD").unwrap_or_else(|_| "dripstone_dev_redis".into())
}

/// Compose `redis` service: `redis://:password@127.0.0.1:6379/0`.
pub fn redis_url_local() -> String {
    let p = stack_redis_password();
    format!("redis://:{p}@127.0.0.1:6379/0")
}

/// Compose `valkey` service: `redis://:password@127.0.0.1:6380/0`.
pub fn valkey_url_local() -> String {
    let p = stack_redis_password();
    format!("redis://:{p}@127.0.0.1:6380/0")
}

/// Compose `postgres` on port 5433 unless `POSTGRES_URL` / `EMBEDDING_PG_URL` is set.
pub fn postgres_url_local() -> String {
    if let Ok(url) = std::env::var("POSTGRES_URL") {
        return url;
    }
    if let Ok(url) = std::env::var("EMBEDDING_PG_URL") {
        return url;
    }
    let pw = std::env::var("STACK_POSTGRES_PASSWORD").unwrap_or_else(|_| "dripstone_dev_pg".into());
    format!("postgres://postgres:{pw}@127.0.0.1:5433/embeddings")
}

/// Qdrant API key; empty `EMBEDDING_QDRANT_API_KEY` disables the header.
pub fn qdrant_api_key() -> Option<String> {
    if let Ok(s) = std::env::var("EMBEDDING_QDRANT_API_KEY") {
        let t = s.trim();
        if t.is_empty() {
            return None;
        }
        return Some(s);
    }
    Some(std::env::var("STACK_QDRANT_API_KEY").unwrap_or_else(|_| "dripstone_dev_qdrant".into()))
}

/// Weaviate bearer token. Only set when the server has API key (or OIDC) enabled: Weaviate returns
/// 401 if anonymous access is allowed but an `Authorization` header is still sent.
/// For local `docker-compose.example.yml` Weaviate, export `EMBEDDING_WEAVIATE_API_KEY` or `STACK_WEAVIATE_API_KEY`.
pub fn weaviate_api_key() -> Option<String> {
    if let Ok(s) = std::env::var("EMBEDDING_WEAVIATE_API_KEY") {
        let t = s.trim();
        if t.is_empty() {
            return None;
        }
        return Some(s);
    }
    std::env::var("STACK_WEAVIATE_API_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

/// Chroma token header when `STACK_CHROMA_API_KEY` / `EMBEDDING_CHROMA_API_KEY` is set.
pub fn chroma_api_key() -> Option<String> {
    if let Ok(s) = std::env::var("EMBEDDING_CHROMA_API_KEY") {
        let t = s.trim();
        if t.is_empty() {
            return None;
        }
        return Some(s);
    }
    std::env::var("STACK_CHROMA_API_KEY")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

// -----------------------------------------------------------------------------
// Dripstone speed benchmark harness (`tests/speed_benchmarks.rs`)
// -----------------------------------------------------------------------------

/// Long comparison tier: larger corpora and concurrency sweeps (`DRIPSTONE_BENCH_LONG=1`).
pub fn dripstone_bench_long() -> bool {
    matches!(
        std::env::var("DRIPSTONE_BENCH_LONG").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

/// Cap corpus rows for long cases (default `50_000`).
pub fn dripstone_bench_max_points() -> usize {
    std::env::var("DRIPSTONE_BENCH_MAX_POINTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(50_000)
}

/// Brute vs HNSW at 10K / 50K / 100K (`DRIPSTONE_BENCH_HNSW_SCALE=1`).
/// For the full 100K row, set `DRIPSTONE_BENCH_MAX_POINTS` to at least `100_000` (e.g. `120000`).
pub fn dripstone_bench_hnsw_scale() -> bool {
    matches!(
        std::env::var("DRIPSTONE_BENCH_HNSW_SCALE").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

/// HNSW recall/latency sweep across multiple parameter combinations (`DRIPSTONE_BENCH_HNSW_SWEEP=1`).
pub fn dripstone_bench_hnsw_sweep() -> bool {
    matches!(
        std::env::var("DRIPSTONE_BENCH_HNSW_SWEEP").as_deref(),
        Ok("1" | "true" | "yes")
    )
}

/// Comma-separated backend names for optional matrix (`DRIPSTONE_BENCH_BACKENDS`).
/// Empty means the matrix test is skipped.
pub fn dripstone_bench_backends_wanted() -> std::collections::HashSet<String> {
    let v = std::env::var("DRIPSTONE_BENCH_BACKENDS").unwrap_or_default();
    v.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_lowercase)
        .collect()
}
