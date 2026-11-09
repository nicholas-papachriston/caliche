use super::{
    MmapBackend, MmapBackendConfig, MmapVectorEncoding, write_dripstone_mmap,
    write_dripstone_mmap_with_encoding,
};
use crate::core::{Chunk, Metric, SearchQuery};
use crate::traits::Backend;
use std::io::Write;
use tempfile::tempdir;

fn test_chunks(namespace: Option<String>) -> Vec<Chunk> {
    vec![
        Chunk::new(
            "d1",
            0,
            "hello",
            vec![1.0_f32, 0.0, 0.0],
            0,
            5,
            1,
            serde_json::json!({"kind":"x"}),
            namespace.clone(),
        ),
        Chunk::new(
            "d2",
            0,
            "world",
            vec![0.0_f32, 1.0, 0.0],
            0,
            5,
            1,
            serde_json::json!({"kind":"y"}),
            namespace,
        ),
    ]
}

fn test_query() -> SearchQuery {
    SearchQuery::new(vec![1.0_f32, 0.0, 0.0])
        .metric(Metric::Cosine)
        .limit(10)
        .namespace("ns1")
}

fn expected_default_encoding() -> MmapVectorEncoding {
    #[cfg(target_arch = "aarch64")]
    {
        MmapVectorEncoding::F32
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        MmapVectorEncoding::F16
    }
}

#[test]
fn mmap_v1_roundtrip_search() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("idx.vf1");
    write_dripstone_mmap_with_encoding(
        &path,
        &test_chunks(Some("ns1".to_string())),
        MmapVectorEncoding::F32,
    )
    .expect("write");

    let backend =
        MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), 3)).expect("open");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    let results = rt.block_on(backend.search(&test_query())).expect("search");
    assert_eq!(results.len(), 2);
    assert!(results[0].score >= results[1].score);
}

#[test]
fn mmap_v1_f32_roundtrip_search_and_get() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("idx.vf1");
    write_dripstone_mmap_with_encoding(
        &path,
        &test_chunks(Some("ns1".to_string())),
        MmapVectorEncoding::F32,
    )
    .expect("write");

    let backend =
        MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), 3)).expect("open");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    let results = rt.block_on(backend.search(&test_query())).expect("search");
    assert_eq!(results.len(), 2);
    assert!(results[0].score >= results[1].score);

    let chunk = rt
        .block_on(backend.get("d1#0", Some("ns1")))
        .expect("get")
        .expect("chunk");
    assert_eq!(chunk.content, "hello");
    assert_eq!(chunk.vector.as_ref(), &[1.0_f32, 0.0, 0.0]);
}

#[test]
fn mmap_v1_f16_roundtrip_search_has_small_score_drift() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("idx_f16.vf1");
    write_dripstone_mmap(&path, &test_chunks(Some("ns1".to_string()))).expect("write");

    let backend =
        MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), 3)).expect("open");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    let results = rt.block_on(backend.search(&test_query())).expect("search");
    assert_eq!(results.len(), 2);
    assert!(results[0].score >= results[1].score);
    assert!(
        (results[0].score - 1.0).abs() < 0.01,
        "score drift too high"
    );
}

#[test]
fn mmap_v2_int8_roundtrip_search_and_get() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("idx_int8.vf2");
    write_dripstone_mmap_with_encoding(
        &path,
        &test_chunks(Some("ns1".to_string())),
        MmapVectorEncoding::Int8,
    )
    .expect("write");

    let backend =
        MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), 3)).expect("open");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    let results = rt.block_on(backend.search(&test_query())).expect("search");
    assert_eq!(results.len(), 2);
    assert!(results[0].score >= results[1].score);
    assert!(
        (results[0].score - 1.0).abs() < 0.05,
        "int8 score drift too high"
    );

    let chunk = rt
        .block_on(backend.get("d1#0", Some("ns1")))
        .expect("get")
        .expect("chunk");
    assert_eq!(chunk.content, "hello");
    assert!((chunk.vector[0] - 1.0).abs() < 0.05);
    assert!(chunk.vector[1].abs() < 0.05);
}

#[test]
fn mmap_product_quantized_remains_explicitly_unsupported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("idx_pq.vf");
    let err = write_dripstone_mmap_with_encoding(
        &path,
        &test_chunks(Some("ns1".to_string())),
        MmapVectorEncoding::ProductQuantized,
    )
    .expect_err("pq should remain unsupported");
    let message = err.to_string();
    assert!(
        message.contains("reserved") || message.contains("not implemented"),
        "unexpected error: {message}"
    );
}

#[test]
fn mmap_v1_default_writer_and_config_use_arch_default() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("idx_default.vf1");
    write_dripstone_mmap(&path, &test_chunks(Some("ns1".to_string()))).expect("write");

    let backend = MmapBackend::new(
        MmapBackendConfig::new(path.to_string_lossy(), 3)
            .with_read_write(true)
            .with_flush_every(0),
    )
    .expect("open rw");
    assert_eq!(backend.config.vector_encoding, expected_default_encoding());
}

#[test]
fn mmap_dimension_mismatch_mentions_format_version() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("idx.vf1");
    let chunks = vec![Chunk::new(
        "d1",
        0,
        "hello",
        vec![1.0_f32, 0.0, 0.0],
        0,
        5,
        1,
        serde_json::json!({}),
        None,
    )];
    write_dripstone_mmap_with_encoding(&path, &chunks, MmapVectorEncoding::F32).expect("write");
    let err = MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), 99))
        .expect_err("wrong dim");
    let msg = err.to_string();
    assert!(
        msg.contains("format version") && msg.contains("dimension"),
        "unexpected error: {msg}"
    );
}

#[test]
fn mmap_unknown_magic_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unknown.vf");
    let mut file = std::fs::File::create(&path).expect("create");
    file.write_all(b"NOTFORGE").expect("write magic");
    file.write_all(&[0u8; 48]).expect("pad header");
    drop(file);

    let err = MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), 3))
        .expect_err("unknown magic should fail");
    let msg = err.to_string();
    assert!(
        msg.contains("DRIPST01") && msg.contains("DRIPST02"),
        "unexpected error: {msg}"
    );
}

#[test]
fn mmap_read_write_upsert_flush_and_snapshot_len() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rw.vf1");
    let backend = MmapBackend::new(
        MmapBackendConfig::new(path.to_string_lossy(), 3)
            .with_read_write(true)
            .with_flush_every(1)
            .with_vector_encoding(MmapVectorEncoding::F16),
    )
    .expect("open rw");
    let chunk = Chunk::new(
        "a",
        0,
        "x",
        vec![0.0_f32, 1.0, 0.0],
        0,
        1,
        1,
        serde_json::json!({}),
        Some("n1".to_string()),
    );
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("rt");
    rt.block_on(async {
        backend
            .upsert(std::slice::from_ref(&chunk))
            .await
            .expect("upsert");
        backend.flush().await.expect("flush");
    });
    assert!(backend.last_snapshot_len() > 0);

    let reopened =
        MmapBackend::new(MmapBackendConfig::new(path.to_string_lossy(), 3)).expect("reopen");
    let result = rt
        .block_on(reopened.get("a#0", Some("n1")))
        .expect("get")
        .expect("chunk");
    assert_eq!(result.vector.as_ref(), &[0.0_f32, 1.0, 0.0]);
}
