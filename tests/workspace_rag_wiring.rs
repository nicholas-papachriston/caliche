//! Workspace RAG wiring via shared_rag_tool in AgentService::from_config.

#![cfg(feature = "tools-rag")]

use async_trait::async_trait;
use caliche::agent::tools::{RagTool, Tool};
use caliche::llm::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
use caliche::rag::{
    RagFusionMode, RagHashlineOptions, RagSearchMode, RagSearchOptions, RagService,
    WorkspaceContextQuery,
};
use caliche::traits::credentials::{Credentials, StaticCredentialProvider};
use caliche::vecstore::{Backend, Filter, MemoryBackend};
use caliche::workspace::{PersistenceConfig, WorkspaceIndexerConfig};
use caliche::{AgentService, AgentServiceConfig, ProviderType};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct StubEmbeddingProvider;

#[async_trait]
impl EmbeddingProvider for StubEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        Ok(EmbeddingResponse {
            embeddings: request.texts.iter().map(|_| vec![0.1_f32; 8]).collect(),
            usage: None,
            model: "stub".to_string(),
        })
    }

    fn name(&self) -> &str {
        "stub"
    }
}

#[derive(Debug, Default)]
struct CountingEmbeddingProvider {
    texts: AtomicUsize,
}

#[async_trait]
impl EmbeddingProvider for CountingEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        self.texts.fetch_add(request.texts.len(), Ordering::SeqCst);
        Ok(EmbeddingResponse {
            embeddings: request.texts.iter().map(|_| vec![0.1_f32; 8]).collect(),
            usage: None,
            model: "counting-stub".to_string(),
        })
    }

    fn name(&self) -> &str {
        "counting-stub"
    }
}

fn test_config_with_shared_rag(rag_tool: Arc<RagTool>) -> AgentServiceConfig {
    let mut config = AgentServiceConfig::new()
        .with_provider(
            Some(ProviderType::Anthropic),
            Some("claude-3-5-sonnet-20241022".to_string()),
        )
        .with_context_window(Some(8_192))
        .with_credential_provider(Arc::new(StaticCredentialProvider::new(
            Credentials::ApiKey("test-key".to_string()),
        )))
        .with_tool_names(vec!["calculator".to_string()]);
    config.shared_rag_tool = Some(rag_tool);
    config
}

#[tokio::test]
async fn from_config_wires_shared_rag_tool_to_workspace_context() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let rag_tool = Arc::new(RagTool::new(embedding, Some("test-ns")));

    let service = AgentService::from_config(test_config_with_shared_rag(rag_tool))
        .await
        .expect("service should build with shared_rag_tool");

    assert!(
        service.has_workspace_rag(),
        "shared_rag_tool should enable workspace RAG on the agent"
    );
    assert!(
        service.rag_service().is_some(),
        "rag_service should be available from shared tool"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn enable_workspace_embedding_after_build_starts_background_index() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let mut service = AgentService::from_config(
        AgentServiceConfig::new()
            .with_provider(
                Some(ProviderType::Anthropic),
                Some("claude-3-5-sonnet-20241022".to_string()),
            )
            .with_credential_provider(Arc::new(StaticCredentialProvider::new(
                Credentials::ApiKey("test-key".to_string()),
            )))
            .with_tool_names(vec!["calculator".to_string()]),
    )
    .await
    .expect("service should build without workspace RAG");

    assert!(
        !service.has_workspace_rag(),
        "workspace RAG should be absent until enable_workspace_embedding"
    );

    let root = std::env::temp_dir().join(format!("caliche-deferred-embed-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("temp workspace");
    let config = WorkspaceIndexerConfig::new(root.clone()).persistence(PersistenceConfig::InMemory);

    let indexed_root = service
        .enable_workspace_embedding(config, embedding)
        .await
        .expect("deferred workspace embedding should attach");

    assert_eq!(indexed_root, root);
    assert!(
        service.has_workspace_rag(),
        "enable_workspace_embedding should wire workspace RAG after build"
    );
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn index_batch_with_paths_indexes_every_document_concurrently() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let rag = caliche::rag::RagService::new(embedding);
    let documents: Vec<(String, String, Option<std::path::PathBuf>)> = (0..12)
        .map(|i| {
            (
                format!("doc-{i}"),
                format!("workspace document number {i} about indexing"),
                None,
            )
        })
        .collect();
    let (chunks, docs, failures, _) = rag
        .index_batch_with_paths(documents)
        .await
        .expect("concurrent batch");
    assert!(failures.is_empty(), "failures: {failures:?}");
    assert_eq!(docs, 12);
    assert!(chunks >= 12);
}

#[tokio::test]
async fn query_context_hybrid_returns_formatted_hits() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let rag = caliche::rag::RagService::new(embedding);
    rag.index_document("hybrid retrieval should surface this rust indexer", "idx")
        .await
        .expect("index");
    let opts = caliche::rag::WorkspaceContextQuery {
        top_k: 4,
        min_similarity: 0.0,
        line_level_bm25: true,
        search_options: None,
    };
    let (results, _, _) = rag
        .query_context("rust indexer", &opts)
        .await
        .expect("hybrid context query");
    assert!(
        results.iter().any(|hit| hit.contains("Similarity")),
        "expected formatted similarity hits, got {results:?}"
    );
}

#[tokio::test]
async fn query_context_full_text_hybrid_ranks_identifier_over_prose() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let rag = RagService::new(embedding);
    rag.index_document("hello world narrative text about driving tours", "prose")
        .await
        .expect("index prose");
    rag.index_document("fn unique_identifier_xyz() {}", "code")
        .await
        .expect("index code");

    let opts = WorkspaceContextQuery {
        top_k: 4,
        min_similarity: 0.0,
        line_level_bm25: false,
        search_options: None,
    };
    let (results, _, _) = rag
        .query_context("unique_identifier_xyz", &opts)
        .await
        .expect("fts hybrid context query");
    assert!(
        !results.is_empty(),
        "expected at least one injected hit, got {results:?}"
    );
    assert!(
        results[0].contains("unique_identifier_xyz"),
        "stub embeddings are identical, so dense cannot distinguish docs; FTS hybrid should rank the identifier first, got {results:?}"
    );
}

#[tokio::test]
async fn wrapped_backend_full_text_query_does_not_fall_back_to_dense() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let rag = RagService::new(embedding);
    rag.index_document("fn unique_identifier_xyz() {}", "code")
        .await
        .expect("index");

    let (results, _, _, warnings, _) = rag
        .enhanced_query(
            "unique_identifier_xyz",
            4,
            0.0,
            RagSearchMode::FullText,
            RagFusionMode::Weighted,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("full-text query");
    assert!(
        warnings
            .iter()
            .all(|warning| !warning.contains("falling back")),
        "FullTextIndexedBackend wrap should advertise native full-text, warnings: {warnings:?}"
    );
    assert!(
        results
            .iter()
            .any(|hit| hit.chunk.content.contains("unique_identifier_xyz")),
        "full-text search should return the identifier document, got {results:?}"
    );
}

#[tokio::test]
async fn incremental_hashline_skips_unchanged_embeddings_and_clear_resets_hashes() {
    let counting = Arc::new(CountingEmbeddingProvider::default());
    let embedding: Arc<dyn EmbeddingProvider> = Arc::clone(&counting) as Arc<dyn EmbeddingProvider>;
    let rag = RagTool::new_with_hashline_options(
        embedding,
        None::<String>,
        RagHashlineOptions::for_workspace(true, false),
    )
    .service();

    let source = "workspace document about incremental hashline indexing";
    rag.index_document(source, "doc-1")
        .await
        .expect("first index");
    let after_first = counting.texts.load(Ordering::SeqCst);
    assert!(
        after_first > 0,
        "first ingest should embed at least one chunk"
    );

    rag.index_document(source, "doc-1")
        .await
        .expect("second index of unchanged content");
    let after_second = counting.texts.load(Ordering::SeqCst);
    assert_eq!(
        after_second, after_first,
        "unchanged hashline chunks should skip re-embedding"
    );

    rag.index_document(
        "workspace document about incremental hashline indexing changed",
        "doc-1",
    )
    .await
    .expect("changed content should re-embed");
    let after_change = counting.texts.load(Ordering::SeqCst);
    assert!(
        after_change > after_second,
        "changed content should embed again"
    );

    rag.clear().await.expect("clear index and hash store");
    rag.index_document(source, "doc-1")
        .await
        .expect("reindex after clear");
    let after_clear = counting.texts.load(Ordering::SeqCst);
    assert!(
        after_clear > after_change,
        "clear must drop incremental hashes so a full re-embed happens"
    );
}

#[tokio::test]
async fn default_query_context_threshold_still_ranks_identifier() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let rag = RagService::new(embedding);
    rag.index_document("hello world narrative text about driving tours", "prose")
        .await
        .expect("index prose");
    rag.index_document("fn unique_identifier_xyz() {}", "code")
        .await
        .expect("index code");

    let opts = WorkspaceContextQuery {
        min_similarity: 0.25,
        ..WorkspaceContextQuery::default()
    };
    let (results, _, _) = rag
        .query_context("unique_identifier_xyz", &opts)
        .await
        .expect("default-threshold fts hybrid");
    assert!(
        results
            .first()
            .is_some_and(|hit| hit.contains("unique_identifier_xyz")),
        "weighted FTS hybrid scores should remain above the 0.25 injection floor, got {results:?}"
    );
}

#[tokio::test]
async fn rag_tool_default_query_uses_hybrid_full_text() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let tool = RagTool::new(embedding, None::<String>);
    let index_prose = tool
        .execute(serde_json::json!({
            "operation": "index",
            "doc_id": "prose",
            "document": "hello world narrative text about driving tours"
        }))
        .await;
    assert!(index_prose.success, "{}", index_prose.content);
    let index_code = tool
        .execute(serde_json::json!({
            "operation": "index",
            "doc_id": "code",
            "document": "fn unique_identifier_xyz() {}"
        }))
        .await;
    assert!(index_code.success, "{}", index_code.content);

    let result = tool
        .execute(serde_json::json!({
            "operation": "query",
            "query": "unique_identifier_xyz",
            "top_k": 4,
            "min_similarity": 0.0
        }))
        .await;
    assert!(result.success, "{}", result.content);
    assert!(
        result.content.contains("unique_identifier_xyz"),
        "default RAG tool query should use FTS hybrid, got {}",
        result.content
    );
    assert!(
        result.content.contains("hybrid_full_text"),
        "tool output should report hybrid_full_text, got {}",
        result.content
    );
}

#[tokio::test]
async fn temporal_decay_path_dates_rank_newer_doc_first() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let rag = RagService::new(embedding);
    rag.index_document(
        "unique_identifier_xyz old journal notes",
        "journal/2020-01-01.md",
    )
    .await
    .expect("index old");
    rag.index_document(
        "unique_identifier_xyz new journal notes",
        "journal/2026-08-01.md",
    )
    .await
    .expect("index new");

    let opts = WorkspaceContextQuery {
        top_k: 4,
        min_similarity: 0.0,
        line_level_bm25: false,
        search_options: Some(RagSearchOptions {
            use_temporal_decay: Some(true),
            temporal_decay_half_life_days: Some(30.0),
            ..RagSearchOptions::default()
        }),
    };
    let (results, _, _) = rag
        .query_context("unique_identifier_xyz", &opts)
        .await
        .expect("temporal context query");
    assert!(
        results
            .first()
            .is_some_and(|hit| hit.contains("new journal notes")),
        "path_date_pattern temporal decay should rank 2026 ahead of 2020, got {results:?}"
    );
}

#[tokio::test]
async fn path_metadata_filter_keeps_only_matching_file() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let rag = RagService::new(embedding);
    let root = std::env::temp_dir().join(format!("caliche-rag-path-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("temp dir");
    let keep_path = root.join("keep.rs");
    let drop_path = root.join("drop.rs");
    std::fs::write(&keep_path, "fn unique_keep_token_aaa() {}").expect("write keep");
    std::fs::write(&drop_path, "fn unique_drop_token_bbb() {}").expect("write drop");

    rag.index_document_with_path(
        "fn unique_keep_token_aaa() {}",
        keep_path.to_str().unwrap_or("keep.rs"),
        Some(keep_path.as_path()),
    )
    .await
    .expect("index keep");
    rag.index_document_with_path(
        "fn unique_drop_token_bbb() {}",
        drop_path.to_str().unwrap_or("drop.rs"),
        Some(drop_path.as_path()),
    )
    .await
    .expect("index drop");

    let keep_path_str = keep_path.to_string_lossy().into_owned();
    let (results, _, _, warnings, _) = rag
        .enhanced_query(
            "unique_keep_token_aaa unique_drop_token_bbb",
            8,
            0.0,
            RagSearchMode::HybridDenseFullText,
            RagFusionMode::Weighted,
            None,
            None,
            false,
            None,
            Some(Filter::Equal {
                field: "path".to_owned(),
                value: serde_json::Value::String(keep_path_str),
            }),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("filtered query");
    let _ = std::fs::remove_dir_all(root);
    assert!(
        warnings
            .iter()
            .all(|warning| !warning.contains("falling back")),
        "filter should run on wrapped FTS, warnings: {warnings:?}"
    );
    assert!(
        results
            .iter()
            .any(|hit| hit.chunk.content.contains("unique_keep_token_aaa")),
        "filtered search should keep the matching path, got {results:?}"
    );
    assert!(
        results
            .iter()
            .all(|hit| !hit.chunk.content.contains("unique_drop_token_bbb")),
        "filtered search should drop the other path, got {results:?}"
    );
}

#[tokio::test]
async fn namespace_isolation_hides_secret_chunks_from_other_namespace() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(StubEmbeddingProvider);
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let secret = RagService::new_with_backend(
        Arc::clone(&embedding),
        Arc::clone(&backend),
        false,
        false,
        false,
    )
    .with_namespace("secret");
    let public = RagService::new_with_backend(embedding, backend, false, false, false)
        .with_namespace("public");

    secret
        .index_document("fn unique_secret_token_ns() {}", "secret-doc")
        .await
        .expect("index secret");
    public
        .index_document("fn unique_public_token_ns() {}", "public-doc")
        .await
        .expect("index public");

    let opts = WorkspaceContextQuery {
        top_k: 8,
        min_similarity: 0.0,
        ..WorkspaceContextQuery::default()
    };
    let (secret_hits, _, _) = secret
        .query_context("unique_secret_token_ns", &opts)
        .await
        .expect("secret query");
    let (public_hits, _, _) = public
        .query_context("unique_secret_token_ns", &opts)
        .await
        .expect("public query for secret token");

    assert!(
        secret_hits
            .iter()
            .any(|hit| hit.contains("unique_secret_token_ns")),
        "secret namespace should see its own identifier, got {secret_hits:?}"
    );
    assert!(
        public_hits
            .iter()
            .all(|hit| !hit.contains("unique_secret_token_ns")),
        "public namespace must not leak secret chunks, got {public_hits:?}"
    );
}

#[derive(Debug)]
struct TokenSensitiveEmbeddingProvider;

fn token_embedding(text: &str) -> Vec<f32> {
    let mut vector = vec![0.0_f32; 8];
    match text {
        "alpha" => vector[0] = 1.0,
        "beta" => vector[1] = 1.0,
        "decoy" => {
            vector[0] = 0.7;
            vector[1] = 0.7;
        }
        _ => {
            for (index, byte) in text.bytes().enumerate() {
                vector[index % 8] += f32::from(byte) / 255.0;
            }
            let norm = vector
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .max(1e-8);
            for value in &mut vector {
                *value /= norm;
            }
        }
    }
    vector
}

#[async_trait]
impl EmbeddingProvider for TokenSensitiveEmbeddingProvider {
    async fn embed(&self, request: EmbeddingRequest) -> anyhow::Result<EmbeddingResponse> {
        Ok(EmbeddingResponse {
            embeddings: request
                .texts
                .iter()
                .map(|text| token_embedding(text))
                .collect(),
            usage: None,
            model: "token-sensitive-stub".to_string(),
        })
    }

    fn name(&self) -> &str {
        "token-sensitive-stub"
    }
}

#[tokio::test]
async fn late_interaction_maxsim_ranks_aligned_tokens_ahead_of_decoy() {
    let embedding: Arc<dyn EmbeddingProvider> = Arc::new(TokenSensitiveEmbeddingProvider);
    let backend: Arc<dyn Backend> = Arc::new(MemoryBackend::new());
    let rag = RagService::new_with_backend_and_options(
        embedding,
        backend,
        false,
        RagHashlineOptions::disabled().late_interaction(true),
    );

    assert!(
        rag.late_interaction_enabled(),
        "hashline late_interaction should enable MaxSim / PLAID"
    );
    assert!(
        rag.dripstone().backend().capabilities().native_maxsim,
        "PLAID wrap should advertise native MaxSim"
    );

    rag.index_document("alpha beta", "aligned")
        .await
        .expect("index aligned");
    rag.index_document("decoy decoy", "decoy")
        .await
        .expect("index decoy");

    let stored = rag
        .dripstone()
        .backend()
        .get("aligned#0", Some("rag"))
        .await
        .expect("get aligned")
        .expect("aligned chunk");
    let token_count = stored
        .multi_vectors
        .as_ref()
        .map_or(0, |vectors| vectors.len());
    assert_eq!(
        token_count, 2,
        "late-interaction ingest should store one vector per unique token"
    );

    let (results, _, _, warnings, _) = rag
        .enhanced_query(
            "alpha beta",
            2,
            0.0,
            RagSearchMode::Dense,
            RagFusionMode::Weighted,
            None,
            None,
            false,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .await
        .expect("maxsim query");
    assert!(
        warnings
            .iter()
            .all(|warning| !warning.contains("Late-interaction MaxSim skipped")),
        "MaxSim query embed should succeed, warnings: {warnings:?}"
    );
    assert!(
        results
            .first()
            .is_some_and(|hit| hit.chunk.content.contains("alpha beta")),
        "MaxSim should rank the token-aligned document first, got {results:?}"
    );
}
