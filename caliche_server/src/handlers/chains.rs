//! Chain API handlers.

use crate::AppState;
use crate::models::*;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use caliche::agent::service::AgentService;
use caliche::chains::*;
use caliche::prompts::PromptTemplate;
use caliche::rag::{RagFusionMode, RagSearchMode};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::Instrument;
use uuid::Uuid;

fn resolve_api_search_mode(
    requested: Option<ApiSearchMode>,
    default: RagSearchMode,
) -> RagSearchMode {
    requested.map(ApiSearchMode::to_rag).unwrap_or(default)
}

fn resolve_api_fusion_mode(requested: Option<ApiFusionMode>) -> RagFusionMode {
    requested
        .map(ApiFusionMode::to_rag)
        .unwrap_or(RagFusionMode::Weighted)
}

/// POST /api/chains/llm - Execute LLM chain
pub async fn llm_chain_handler(
    State(state): State<AppState>,
    Json(request): Json<LLMChainRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "chain_llm", request_id = %request_id);

    async move {
        tracing::info!("Received LLM chain request");

        let service =
            create_service(&state, request.provider, request.model, request.tools).await?;
        let prompt = PromptTemplate::new(&request.prompt_template);

        let chain = LLMChain::new(Arc::new(Mutex::new(service)), prompt);
        let result = chain
            .run(&request.variables)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(ChainResponse {
            output: result.output,
            intermediate_steps: result.context.intermediate_steps,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/chains/sequential - Execute sequential chain
pub async fn sequential_chain_handler(
    State(state): State<AppState>,
    Json(request): Json<SequentialChainRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "chain_sequential", request_id = %request_id);

    async move {
        tracing::info!("Received sequential chain request");

        let base_service =
            create_service(&state, request.provider, request.model, request.tools).await?;

        let mut sequential_chain = ComposableChain::new(LLMChain::new(
            Arc::new(Mutex::new(base_service.clone_for_request())),
            PromptTemplate::new(&request.chains[0].prompt_template),
        )) >> ComposableChain::new(LLMChain::new(
            Arc::new(Mutex::new(base_service.clone_for_request())),
            PromptTemplate::new(&request.chains[1].prompt_template),
        ));

        for step in request.chains {
            sequential_chain = sequential_chain
                >> ComposableChain::new(LLMChain::new(
                    Arc::new(Mutex::new(base_service.clone_for_request())),
                    PromptTemplate::new(&step.prompt_template),
                ));
        }

        let result = sequential_chain
            .run(&request.input)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(ChainResponse {
            output: result.output,
            intermediate_steps: result.context.intermediate_steps,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/chains/map-reduce - Execute map-reduce chain
pub async fn map_reduce_handler(
    State(state): State<AppState>,
    Json(request): Json<MapReduceRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "chain_map_reduce", request_id = %request_id);

    async move {
        tracing::info!("Received map-reduce chain request");

        let base_service =
            create_service(&state, request.provider, request.model, request.tools).await?;

        let map_prompt = PromptTemplate::new(&request.map_template);
        let reduce_prompt = PromptTemplate::new(&request.reduce_template);

        let map_chain = Arc::new(LLMChain::new(
            Arc::new(Mutex::new(base_service.clone_for_request())),
            map_prompt,
        ));
        let reduce_chain = Arc::new(LLMChain::new(
            Arc::new(Mutex::new(base_service.clone_for_request())),
            reduce_prompt,
        ));

        let chain = MapReduceChain::new(map_chain, reduce_chain);

        let input = serde_json::json!({
            "items": request.items
        });

        let result = chain
            .run(&input)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(ChainResponse {
            output: result.output,
            intermediate_steps: result.context.intermediate_steps,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/chains/retrieval - Execute RAG retrieval chain
#[cfg(feature = "retrieval-chains")]
pub async fn retrieval_chain_handler(
    State(state): State<AppState>,
    Json(request): Json<RetrievalChainRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "chain_retrieval", request_id = %request_id);

    async move {
        tracing::info!("Received retrieval chain request");

        #[cfg(feature = "tools-rag")]
        let rag_tool = state.shared_rag_tool.as_ref().ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "RAG tool not available. Configure RAG tool in AppState.".to_string(),
            )
        })?;

        #[cfg(not(feature = "tools-rag"))]
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
        ));

        let rag_service = rag_tool.service();

        let top_k = request.top_k.unwrap_or(5);
        let namespace = request.collection.as_ref().map(|s| s.as_str()).unwrap_or("rag");
        let search_mode = resolve_api_search_mode(request.search_mode, RagSearchMode::Dense);
        let fusion_mode = resolve_api_fusion_mode(request.fusion_mode);

        let (retrieved_docs, _top_similarity, _token_usage, warnings, _context_stats) = rag_service
            .enhanced_query(
                &request.query,
                top_k,
                0.0,
                search_mode,
                fusion_mode,
                request.dense_weight,
                request.sparse_weight,
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
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("RAG query failed: {}", e)))?;

        let context_docs: Vec<String> = retrieved_docs
            .into_iter()
            .map(|doc| doc.chunk.content)
            .collect();

        // Create LLM service and chain
        let llm_service = Arc::new(Mutex::new(
            create_service(&state, request.provider, request.model, request.tools).await?,
        ));

        let prompt_template = format!(
            "Answer the question based on this context:\n\n{{context}}\n\nQuestion: {{query}}\n\nAnswer:"
        );
        let prompt = PromptTemplate::new(&prompt_template);
        let llm_chain = Arc::new(LLMChain::new(llm_service, prompt));

        // Create retrieval chain using the backend (for future direct backend access)
        // For now, we've already retrieved, so we'll manually construct the augmented input
        let context_text = if context_docs.is_empty() {
            "No relevant context found.".to_string()
        } else {
            let mut buffer = String::with_capacity(context_docs.len() * 100);
            for (i, doc) in context_docs.iter().enumerate() {
                if i > 0 {
                    buffer.push_str("\n\n");
                }
                buffer.push_str("[Document ");
                buffer.push_str(&(i + 1).to_string());
                buffer.push_str("]\n");
                buffer.push_str(doc);
            }
            buffer
        };

        let augmented_input = serde_json::json!({
            "query": request.query,
            "context": context_text
        });

        // Run LLM chain with augmented input
        let result = llm_chain
            .run(&augmented_input)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        // Add retrieval step to intermediate steps
        let mut intermediate_steps = result.context.intermediate_steps;
        intermediate_steps.insert(0, (
            "retrieval".to_string(),
            serde_json::json!({
                "query": request.query,
                "num_results": context_docs.len(),
                "namespace": namespace,
                "top_k": top_k,
                "search_mode": search_mode.as_str(),
                "fusion_mode": fusion_mode.as_str(),
                "warnings": warnings
            }),
        ));

        Ok(Json(ChainResponse {
            output: result.output,
            intermediate_steps,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// Stub for retrieval chain when feature is disabled
#[cfg(not(feature = "retrieval-chains"))]
pub async fn retrieval_chain_handler(
    _state: State<AppState>,
    _request: Json<RetrievalChainRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "Retrieval chains feature not enabled. Compile with --features retrieval-chains"
            .to_string(),
    ))
}

/// POST /api/chains/hybrid-search - Execute hybrid search chain
#[cfg(feature = "retrieval-chains")]
pub async fn hybrid_search_handler(
    State(state): State<AppState>,
    Json(request): Json<HybridSearchRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span =
        tracing::span!(tracing::Level::INFO, "chain_hybrid_search", request_id = %request_id);

    async move {
        tracing::info!("Received hybrid search chain request");

        #[cfg(feature = "tools-rag")]
        let rag_tool = state.shared_rag_tool.as_ref().ok_or_else(|| {
            (
                StatusCode::BAD_REQUEST,
                "RAG tool not available. Configure RAG tool in AppState.".to_string(),
            )
        })?;

        #[cfg(not(feature = "tools-rag"))]
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
        ));

        let rag_service = rag_tool.service();

        let top_k = request.top_k.unwrap_or(10);
        let dense_weight = request.dense_weight.or(request.vector_weight);
        let sparse_weight = request.sparse_weight.or(request.keyword_weight);
        let search_mode =
            resolve_api_search_mode(request.search_mode, RagSearchMode::HybridDenseSparse);
        let fusion_mode = resolve_api_fusion_mode(request.fusion_mode);

        let (search_results, _top_similarity, _token_usage, warnings, _context_stats) = rag_service
            .enhanced_query(
                &request.query,
                top_k,
                0.0,
                search_mode,
                fusion_mode,
                dense_weight,
                sparse_weight,
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
            .map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Hybrid search failed: {}", e),
                )
            })?;

        // Extract document content from search results
        let context_docs: Vec<String> = search_results
            .into_iter()
            .map(|result| result.chunk.content)
            .collect();

        // Create LLM service and chain
        let llm_service = Arc::new(Mutex::new(
            create_service(&state, request.provider, request.model, request.tools).await?,
        ));

        let prompt_template =
            format!("Based on these documents:\n\n{{context}}\n\nAnswer: {{query}}");
        let prompt = PromptTemplate::new(&prompt_template);
        let llm_chain = Arc::new(LLMChain::new(llm_service, prompt));

        // Format context from retrieved documents
        let context_text = if context_docs.is_empty() {
            "No relevant documents found.".to_string()
        } else {
            let mut buffer = String::with_capacity(context_docs.len() * 100);
            for (i, doc) in context_docs.iter().enumerate() {
                if i > 0 {
                    buffer.push_str("\n\n");
                }
                buffer.push_str("[Document ");
                buffer.push_str(&(i + 1).to_string());
                buffer.push_str("]\n");
                buffer.push_str(doc);
            }
            buffer
        };

        let augmented_input = serde_json::json!({
            "query": request.query,
            "context": context_text
        });

        // Run LLM chain with augmented input
        let result = llm_chain
            .run(&augmented_input)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        // Add hybrid search step to intermediate steps
        let mut intermediate_steps = result.context.intermediate_steps;
        intermediate_steps.insert(
            0,
            (
                "hybrid_search".to_string(),
                serde_json::json!({
                    "query": request.query,
                    "search_mode": search_mode.as_str(),
                    "fusion_mode": fusion_mode.as_str(),
                    "dense_weight": dense_weight,
                    "sparse_weight": sparse_weight,
                    "final_results": context_docs.len(),
                    "warnings": warnings
                }),
            ),
        );

        Ok(Json(ChainResponse {
            output: result.output,
            intermediate_steps,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// Stub for hybrid search when feature is disabled
#[cfg(not(feature = "retrieval-chains"))]
pub async fn hybrid_search_handler(
    _state: State<AppState>,
    _request: Json<HybridSearchRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "Hybrid search feature not enabled. Compile with --features retrieval-chains".to_string(),
    ))
}

/// POST /api/chains/document-map-reduce - Execute document map-reduce
#[cfg(feature = "retrieval-chains")]
pub async fn document_map_reduce_handler(
    State(state): State<AppState>,
    Json(request): Json<DocumentMapReduceRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span =
        tracing::span!(tracing::Level::INFO, "chain_doc_map_reduce", request_id = %request_id);

    async move {
        tracing::info!("Received document map-reduce chain request");

        let base_service =
            create_service(&state, request.provider, request.model, request.tools).await?;

        let map_prompt = PromptTemplate::new(&request.map_template);
        let reduce_prompt = PromptTemplate::new(&request.reduce_template);

        let map_chain = Arc::new(LLMChain::new(
            Arc::new(Mutex::new(base_service.clone_for_request())),
            map_prompt,
        ));
        let reduce_chain = Arc::new(LLMChain::new(
            Arc::new(Mutex::new(base_service.clone_for_request())),
            reduce_prompt,
        ));

        let mut chain = DocumentMapReduceChain::new(map_chain, reduce_chain);

        if let Some(chunk_size) = request.chunk_size {
            chain = chain.with_chunking(
                caliche::chains::document_map_reduce::ChunkingStrategy::Sentence {
                    max_size: chunk_size,
                },
            );
        }

        let input = serde_json::json!({
            "document": request.document
        });

        let result = chain
            .run(&input)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(ChainResponse {
            output: result.output,
            intermediate_steps: result.context.intermediate_steps,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// Stub for document map-reduce when feature is disabled
#[cfg(not(feature = "retrieval-chains"))]
pub async fn document_map_reduce_handler(
    _state: State<AppState>,
    _request: Json<DocumentMapReduceRequest>,
) -> Result<Json<ChainResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "Document map-reduce feature not enabled. Compile with --features retrieval-chains"
            .to_string(),
    ))
}

// Helper function to create service
async fn create_service(
    state: &AppState,
    provider: Option<String>,
    model: Option<String>,
    tools: Option<Vec<String>>,
) -> Result<AgentService, (StatusCode, String)> {
    let provider_type = provider
        .as_ref()
        .map(|p| p.parse())
        .transpose()
        .map_err(|e: String| (StatusCode::BAD_REQUEST, format!("Invalid provider: {}", e)))?;

    {
        let mut config = caliche::AgentServiceConfig::new()
            .with_provider(provider_type, model)
            .with_tool_names_or_defaults(tools);
        #[cfg(feature = "tools-rag")]
        {
            config.shared_rag_tool = state.shared_rag_tool.clone();
        }
        AgentService::from_config(config).await
    }
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create service: {}", e),
        )
    })
}
