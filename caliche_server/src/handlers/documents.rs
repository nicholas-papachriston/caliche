//! Document processing API handlers.

use crate::AppState;
use crate::models::*;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::loaders::*;
use caliche::rag::{RagFusionMode, RagSearchMode, filter_dto::filter_from_value};
use tracing::Instrument;
use uuid::Uuid;

fn resolve_document_search_mode(request: &DocumentSearchRequest) -> RagSearchMode {
    request
        .search_mode
        .map(ApiSearchMode::to_rag)
        .unwrap_or(RagSearchMode::Dense)
}

fn resolve_document_fusion_mode(request: &DocumentSearchRequest) -> RagFusionMode {
    request
        .fusion_mode
        .map(ApiFusionMode::to_rag)
        .unwrap_or(RagFusionMode::Weighted)
}

fn resolve_document_filter(
    filter: Option<&serde_json::Value>,
) -> Result<Option<caliche::vecstore::Filter>, String> {
    match filter {
        None => Ok(None),
        Some(value) => {
            let is_filter_dsl = value
                .as_object()
                .is_some_and(|object| object.contains_key("type"));
            if is_filter_dsl {
                filter_from_value(value)
                    .map(Some)
                    .map_err(|e| e.to_string())
            } else {
                Ok(None)
            }
        }
    }
}

/// POST /api/documents/load - Load documents from various formats
#[cfg(feature = "document-loaders")]
pub async fn load_documents_handler(
    Json(request): Json<DocumentLoadRequest>,
) -> Result<Json<DocumentLoadResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "documents_load", request_id = %request_id);

    async move {
        tracing::info!("Received document load request");

        let documents = load_documents_from_source(request.source).await?;

        let document_dtos: Vec<DocumentDto> = documents.into_iter().map(Into::into).collect();
        let count = document_dtos.len();

        Ok(Json(DocumentLoadResponse {
            documents: document_dtos,
            count,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// Stub when document-loaders feature is disabled
#[cfg(not(feature = "document-loaders"))]
pub async fn load_documents_handler(
    _request: Json<DocumentLoadRequest>,
) -> Result<Json<DocumentLoadResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "Document loaders feature not enabled. Compile with --features document-loaders"
            .to_string(),
    ))
}

/// POST /api/documents/split - Split documents into chunks
pub async fn split_documents_handler(
    Json(request): Json<DocumentSplitRequest>,
) -> Result<Json<DocumentSplitResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "documents_split", request_id = %request_id);

    async move {
        tracing::info!("Received document split request");

        let chunk_overlap = request.chunk_overlap.unwrap_or(50);
        let splitter = TextSplitter::new(request.chunk_size, chunk_overlap);

        // Convert DTOs to Documents
        let documents: Vec<Document> = request
            .documents
            .iter()
            .map(|dto| Document {
                content: dto.content.clone(),
                metadata: dto.metadata.clone(),
            })
            .collect();

        let chunks = splitter.split_documents(&documents);

        let chunk_dtos: Vec<ChunkDto> = chunks
            .into_iter()
            .enumerate()
            .map(|(idx, doc)| ChunkDto {
                content: doc.content,
                metadata: doc.metadata,
                chunk_index: idx,
            })
            .collect();

        let total_chunks = chunk_dtos.len();

        Ok(Json(DocumentSplitResponse {
            chunks: chunk_dtos,
            total_chunks,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/documents/index - Index documents into RAG vector store
#[cfg(feature = "tools-rag")]
pub async fn index_documents_handler(
    State(state): State<AppState>,
    Json(request): Json<DocumentIndexRequest>,
) -> Result<Json<DocumentIndexResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "documents_index", request_id = %request_id);

    async move {
        tracing::info!(
            request_id = %request_id,
            collection = %request.collection_name,
            document_count = request.documents.len(),
            vector_storage = ?request.vector_storage,
            "Received document index request"
        );

        #[cfg(feature = "tools-rag")]
        let rag_tool = state.shared_rag_tool.as_ref().ok_or_else(|| {
            tracing::error!(
                request_id = %request_id,
                "RAG tool not available for document indexing"
            );
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "RAG tool not available. Ensure OPENAI_API_KEY or BEDROCK_EMBEDDING_MODEL_ID is set."
                    .to_string(),
            )
        })?;
        #[cfg(not(feature = "tools-rag"))]
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
        ));

        let mut ids = Vec::new();
        let mut indexed_count = 0;

        for (idx, doc) in request.documents.iter().enumerate() {
            use caliche::agent::tools::Tool;

            let doc_id = format!("{}_{}", request.collection_name, idx);
            let args = serde_json::json!({
                "operation": "index",
                "document": doc.content,
                "doc_id": doc_id
            });

            match Tool::execute(rag_tool.as_ref(), args).await {
                result if result.success => {
                    ids.push(doc_id);
                    indexed_count += 1;
                }
                result => {
                    tracing::warn!(
                        request_id = %request_id,
                        collection = %request.collection_name,
                        document_index = idx,
                        doc_id = %doc_id,
                        error = ?result.error,
                        "Failed to index document"
                    );
                }
            }
        }

        // Register collection in document_collections registry
        if indexed_count > 0 {
            let collection_name = request.collection_name.clone();
            let now = chrono::Utc::now();
            if let Some(mut entry) = state.document_collections.get_mut(&collection_name) {
                // Update existing collection
                entry.document_count += indexed_count;
                entry.updated_at = now;
            } else {
                // Create new collection entry
                state.document_collections.insert(
                    collection_name.clone(),
                    crate::CollectionMetadata {
                        name: collection_name,
                        document_count: indexed_count,
                        created_at: now,
                        updated_at: now,
                    },
                );
            }
        }

        Ok(Json(DocumentIndexResponse {
            collection_name: request.collection_name,
            indexed_count,
            ids,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/rag/index-batch - Batch index documents into RAG vector store
#[cfg(feature = "tools-rag")]
pub async fn batch_index_documents_handler(
    State(state): State<AppState>,
    Json(request): Json<DocumentBatchIndexRequest>,
) -> Result<Json<DocumentBatchIndexResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "rag_batch_index", request_id = %request_id);

    async move {
        use serde_json::json;

        use caliche::agent::tools::Tool;

        tracing::info!(
            "Received batch index request for {} documents",
            request.documents.len()
        );

        #[cfg(feature = "tools-rag")]
        let rag_tool = state.shared_rag_tool.as_ref().ok_or_else(|| {
            tracing::error!(
                request_id = %request_id,
                "RAG tool not available for document indexing"
            );
            (
                StatusCode::SERVICE_UNAVAILABLE,
                "RAG tool not available. Ensure OPENAI_API_KEY or BEDROCK_EMBEDDING_MODEL_ID is set."
                    .to_string(),
            )
        })?;
        #[cfg(not(feature = "tools-rag"))]
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
        ));

        let args = serde_json::json!({
            "operation": "index_batch",
            "documents": request.documents.iter().map(|item| {
                json!({
                    "doc_id": item.doc_id,
                    "document": item.document
                })
            }).collect::<Vec<_>>()
        });

        match rag_tool.execute(args).await {
            result if result.success => {
                // Parse the result JSON
                let result_json: serde_json::Value = serde_json::from_str(&result.content)
                    .map_err(|e| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to parse result: {}", e),
                        )
                    })?;

                let total_chunks = result_json["total_chunks"].as_u64().unwrap_or(0) as usize;
                let total_documents = result_json["total_documents"].as_u64().unwrap_or(0) as usize;
                let failures: Vec<String> = result_json["failure_details"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                let token_usage = result.usage;

                Ok(Json(DocumentBatchIndexResponse {
                    total_chunks,
                    total_documents,
                    failures,
                    token_usage,
                    request_id,
                }))
            }
            result => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                result
                    .error
                    .unwrap_or_else(|| "Batch indexing failed".to_string()),
            )),
        }
    }
    .instrument(span)
    .await
}

/// Stub when tools-rag feature is disabled
#[cfg(not(feature = "tools-rag"))]
pub async fn batch_index_documents_handler(
    _state: State<AppState>,
    _request: Json<DocumentBatchIndexRequest>,
) -> Result<Json<DocumentBatchIndexResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
    ))
}

/// Stub when tools-rag feature is disabled
#[cfg(not(feature = "tools-rag"))]
pub async fn index_documents_handler(
    _state: State<AppState>,
    _request: Json<DocumentIndexRequest>,
) -> Result<Json<DocumentIndexResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
    ))
}

/// GET /api/documents/collections - List available collections
#[cfg(feature = "tools-rag")]
pub async fn list_collections_handler(
    State(state): State<AppState>,
) -> Result<Json<CollectionsResponse>, (StatusCode, String)> {
    let _span = tracing::span!(tracing::Level::INFO, "documents_list_collections");

    // Start with collections from app state registry
    let collections_map: std::collections::HashMap<String, CollectionInfo> = state
        .document_collections
        .iter()
        .map(|entry| {
            let name = entry.value().name.clone();
            let info = CollectionInfo {
                name: name.clone(),
                document_count: entry.value().document_count,
                created_at: Some(entry.value().created_at.to_rfc3339()),
                updated_at: Some(entry.value().updated_at.to_rfc3339()),
            };
            (name, info)
        })
        .collect();

    // Convert to vector and return
    let collections: Vec<CollectionInfo> = collections_map.into_values().collect();

    Ok(Json(CollectionsResponse { collections }))
}

/// Stub when tools-rag feature is disabled
#[cfg(not(feature = "tools-rag"))]
pub async fn list_collections_handler(
    _state: State<AppState>,
) -> Result<Json<CollectionsResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
    ))
}

/// DELETE /api/documents/collections/:name - Delete a collection
#[cfg(feature = "tools-rag")]
pub async fn delete_collection_handler(
    State(state): State<AppState>,
    Path(name): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let _span =
        tracing::span!(tracing::Level::INFO, "documents_delete_collection", collection = %name);

    // Remove from app state registry
    state.document_collections.remove(&name);

    Ok(StatusCode::NO_CONTENT)
}

/// Stub when tools-rag feature is disabled
#[cfg(not(feature = "tools-rag"))]
pub async fn delete_collection_handler(
    _state: State<AppState>,
    _name: Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
    ))
}

/// POST /api/documents/search - Search documents in collection
#[cfg(feature = "tools-rag")]
pub async fn search_documents_handler(
    State(state): State<AppState>,
    Json(request): Json<DocumentSearchRequest>,
) -> Result<Json<DocumentSearchResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();

    tracing::info!(
        request_id = %request_id,
        collection = %request.collection_name,
        query_length = request.query.len(),
        top_k = ?request.top_k,
        vector_storage = ?request.vector_storage,
        "Received document search request"
    );

    let rag_tool = state.shared_rag_tool.as_ref().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "RAG tool not available. Configure RAG tool in AppState.".to_string(),
        )
    })?;

    let rag_service = rag_tool.service();
    let top_k = request.top_k.unwrap_or(5);
    let min_similarity = request.min_similarity.unwrap_or(0.0);
    let search_mode = resolve_document_search_mode(&request);
    let fusion_mode = resolve_document_fusion_mode(&request);
    let filter = resolve_document_filter(request.filter.as_ref())
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid filter: {e}")))?;

    let (retrieved_docs, _top_similarity, _token_usage, warnings, _context_stats) = rag_service
        .enhanced_query(
            &request.query,
            top_k,
            min_similarity,
            search_mode,
            fusion_mode,
            request.dense_weight,
            request.sparse_weight,
            false,
            request.filter.as_ref(),
            filter,
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
                format!("RAG query failed: {}", e),
            )
        })?;

    let mut results = Vec::new();
    for doc in retrieved_docs {
        results.push(SearchResult {
            content: doc.chunk.content,
            score: doc.score,
            metadata: std::collections::HashMap::new(),
        });
    }

    Ok(Json(DocumentSearchResponse {
        results,
        warnings,
        search_mode: Some(search_mode.as_str().to_string()),
        fusion_mode: Some(fusion_mode.as_str().to_string()),
        request_id,
    }))
}

/// Stub when tools-rag feature is disabled
#[cfg(not(feature = "tools-rag"))]
pub async fn search_documents_handler(
    _state: State<AppState>,
    _request: Json<DocumentSearchRequest>,
) -> Result<Json<DocumentSearchResponse>, (StatusCode, String)> {
    Err((
        StatusCode::NOT_IMPLEMENTED,
        "RAG tool feature not enabled. Compile with --features tools-rag".to_string(),
    ))
}

#[cfg(feature = "document-loaders")]
async fn load_documents_from_source(
    source: DocumentSource,
) -> Result<Vec<Document>, (StatusCode, String)> {
    match source {
        DocumentSource::Csv {
            location,
            content_column,
            metadata_columns,
        } => load_csv_documents(location, content_column, metadata_columns).await,
        DocumentSource::Json {
            location,
            content_key,
        } => load_json_documents(location, content_key).await,
        DocumentSource::Markdown { location } => load_markdown_documents(location).await,
        DocumentSource::Html { location } => load_html_documents(location).await,
    }
}

#[cfg(feature = "document-loaders")]
fn write_temp_upload(
    content: &str,
    extension: &str,
) -> Result<std::path::PathBuf, (StatusCode, String)> {
    let temp_file = std::env::temp_dir().join(format!("upload_{}.{extension}", Uuid::now_v7()));
    std::fs::write(&temp_file, content).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to write temp file: {}", e),
        )
    })?;
    Ok(temp_file)
}

#[cfg(feature = "document-loaders")]
fn temp_file_path_str(temp_file: &std::path::Path) -> Result<&str, (StatusCode, String)> {
    temp_file.to_str().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to convert temp file path to string".to_string(),
        )
    })
}

#[cfg(feature = "document-loaders")]
fn map_load_err(format: &str, error: impl std::fmt::Display) -> (StatusCode, String) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("Failed to load {format}: {error}"),
    )
}

#[cfg(feature = "document-loaders")]
async fn load_csv_documents(
    location: DocumentLocation,
    content_column: Option<String>,
    metadata_columns: Option<Vec<String>>,
) -> Result<Vec<Document>, (StatusCode, String)> {
    let content = get_content_from_location(location).await?;
    let temp_file = write_temp_upload(&content, "csv")?;
    let mut loader = CsvLoader::new(temp_file_path_str(&temp_file)?);
    if let Some(col) = content_column {
        loader = loader.with_content_column(&col);
    }
    if let Some(cols) = metadata_columns {
        loader = loader.with_metadata_columns(&cols.iter().map(String::as_str).collect::<Vec<_>>());
    }
    let docs = loader.load().await.map_err(|e| map_load_err("CSV", e))?;
    let _ = std::fs::remove_file(temp_file);
    Ok(docs)
}

#[cfg(feature = "document-loaders")]
async fn load_json_documents(
    location: DocumentLocation,
    content_key: Option<String>,
) -> Result<Vec<Document>, (StatusCode, String)> {
    let content = get_content_from_location(location).await?;
    let temp_file = write_temp_upload(&content, "json")?;
    let mut loader = JsonLoader::new(temp_file_path_str(&temp_file)?);
    if let Some(key) = content_key {
        loader = loader.with_content_key(&key);
    }
    let docs = loader.load().await.map_err(|e| map_load_err("JSON", e))?;
    let _ = std::fs::remove_file(temp_file);
    Ok(docs)
}

#[cfg(feature = "document-loaders")]
async fn load_markdown_documents(
    location: DocumentLocation,
) -> Result<Vec<Document>, (StatusCode, String)> {
    let content = get_content_from_location(location).await?;
    let temp_file = write_temp_upload(&content, "md")?;
    let loader = MarkdownLoader::new(temp_file_path_str(&temp_file)?);
    let docs = loader
        .load()
        .await
        .map_err(|e| map_load_err("Markdown", e))?;
    let _ = std::fs::remove_file(temp_file);
    Ok(docs)
}

#[cfg(feature = "document-loaders")]
async fn load_html_documents(
    location: DocumentLocation,
) -> Result<Vec<Document>, (StatusCode, String)> {
    let content = get_content_from_location(location).await?;
    let temp_file = write_temp_upload(&content, "html")?;
    let loader = HtmlLoader::new(temp_file_path_str(&temp_file)?);
    let docs = loader.load().await.map_err(|e| map_load_err("HTML", e))?;
    let _ = std::fs::remove_file(temp_file);
    Ok(docs)
}

#[cfg(feature = "document-loaders")]
async fn get_content_from_location(
    location: DocumentLocation,
) -> Result<String, (StatusCode, String)> {
    match location {
        DocumentLocation::Path { path } => {
            let workspace = crate::handlers::files::get_workspace_root();
            tokio::task::spawn_blocking(move || {
                read_document_text(&workspace, std::path::Path::new(&path))
            })
            .await
            .unwrap_or_else(|join| {
                Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Document read task failed: {join}"),
                ))
            })
        }
        DocumentLocation::Content { content } => Ok(content),
    }
}

/// Read a document inside `workspace` as UTF-8 text (blocking).
///
/// The read is the file browser's: no link below the workspace is followed,
/// a FIFO or other special file is refused without blocking, and the size is
/// capped.
#[cfg(feature = "document-loaders")]
fn read_document_text(
    workspace: &std::path::Path,
    path: &std::path::Path,
) -> Result<String, (StatusCode, String)> {
    let (target, bytes) = crate::handlers::files::read_workspace_bytes(workspace, path)?;
    String::from_utf8(bytes).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to read file {}: {e}", target.display()),
        )
    })
}

#[cfg(all(test, feature = "document-loaders"))]
mod tests {
    use super::read_document_text;
    use axum::http::StatusCode;
    use std::path::{Path, PathBuf};

    fn workspace() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("workspace");
        let root = dir.path().canonicalize().expect("canonical workspace");
        (dir, root)
    }

    #[test]
    fn a_document_in_the_workspace_is_read() {
        let (_dir, root) = workspace();
        std::fs::write(root.join("doc.md"), "# Title\n").expect("doc");

        assert_eq!(
            read_document_text(&root, Path::new("doc.md")),
            Ok("# Title\n".to_string())
        );
    }

    #[test]
    fn a_document_over_the_limit_is_400_too_large() {
        let (_dir, root) = workspace();
        std::fs::write(root.join("big.md"), vec![b'a'; 1024 * 1024 + 1]).expect("big");

        assert_eq!(
            read_document_text(&root, Path::new("big.md")),
            Err((
                StatusCode::BAD_REQUEST,
                "File is too large (over 1048576 bytes). Maximum is 1048576 bytes.".to_string()
            ))
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_is_refused_without_blocking() {
        let (_dir, root) = workspace();
        let made = std::process::Command::new("mkfifo")
            .arg(root.join("pipe"))
            .status()
            .expect("mkfifo");
        assert!(made.success(), "mkfifo failed");

        let (sender, receiver) = std::sync::mpsc::channel();
        let ws = root.clone();
        std::thread::spawn(move || {
            let _ = sender.send(read_document_text(&ws, Path::new("pipe")));
        });
        let result = receiver
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("a FIFO must be refused, not read until a writer appears");

        assert_eq!(
            result,
            Err((StatusCode::BAD_REQUEST, "Path is not a file".to_string()))
        );
    }

    #[cfg(unix)]
    #[test]
    fn a_parent_swapped_for_a_symlink_is_403() {
        let (_dir, root) = workspace();
        let outside = tempfile::tempdir().expect("outside");
        std::fs::write(outside.path().join("doc.md"), "secret").expect("outside doc");
        std::fs::create_dir(root.join("sub")).expect("sub");
        std::fs::write(root.join("sub/doc.md"), "inside").expect("inside doc");

        let ws = root.clone();
        let target = outside.path().to_path_buf();
        crate::handlers::files::AFTER_VALIDATE.with(|cell| {
            *cell.borrow_mut() = Some(Box::new(move |_validated: &Path| {
                std::fs::rename(ws.join("sub"), ws.join("sub.moved")).expect("move sub");
                std::os::unix::fs::symlink(&target, ws.join("sub")).expect("symlink");
            }));
        });

        assert_eq!(
            read_document_text(&root, Path::new("sub/doc.md")),
            Err((
                StatusCode::FORBIDDEN,
                format!(
                    "Failed to open file: refusing to follow symlink at 'sub' in '{}'",
                    root.join("sub/doc.md").display()
                )
            ))
        );
    }
}
