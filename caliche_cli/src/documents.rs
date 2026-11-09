//! CLI handlers for document processing.

use crate::commands::{RagCliFusionMode, RagCliSearchMode};
use caliche::loaders::*;
use caliche::rag::{RagFusionMode, RagHashlineOptions, RagSearchMode, RagService};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct CliSearchResult {
    pub content: String,
    pub score: f32,
}

pub async fn build_rag_tool_for_collection(
    service: &caliche::AgentService,
    collection: &str,
    hashline_enabled: bool,
    hashline_format_results: bool,
) -> anyhow::Result<caliche::agent::tools::RagTool> {
    let embedding_provider = if let Some(provider) = service.embedding_provider() {
        provider
    } else if let Some(discovered) =
        Box::pin(caliche::discovery::embedding::discover_best_embedding_provider()).await
    {
        discovered.provider
    } else {
        anyhow::bail!(
            "No embedding provider available for RAG. Configure a provider with embeddings or set OPENAI_API_KEY / other embedding-capable provider env vars."
        );
    };

    let namespace = Some(collection.to_string());
    let hashline_options =
        RagHashlineOptions::for_edit_tools(hashline_enabled, hashline_format_results);

    if std::env::var("VECTOR_BACKEND").is_ok() {
        let config = caliche::vecstore::VectorBackendConfig::from_env()?;
        let backend = config.build_backend().await?;
        Ok(
            caliche::agent::tools::RagTool::with_backend_and_hashline_options(
                embedding_provider,
                backend,
                namespace,
                false,
                hashline_options,
            ),
        )
    } else {
        Ok(caliche::agent::tools::RagTool::new_with_hashline_options(
            embedding_provider,
            namespace,
            hashline_options,
        ))
    }
}

/// Load documents from a file
pub async fn load_documents(path: &str, format: &str) -> anyhow::Result<Vec<Document>> {
    match format.to_lowercase().as_str() {
        "csv" => {
            let loader = CsvLoader::new(path);
            loader.load().await.map_err(|e| anyhow::anyhow!("{}", e))
        }
        "json" => {
            let loader = JsonLoader::new(path);
            loader.load().await.map_err(|e| anyhow::anyhow!("{}", e))
        }
        "markdown" | "md" => {
            let loader = MarkdownLoader::new(path);
            loader.load().await.map_err(|e| anyhow::anyhow!("{}", e))
        }
        "html" => {
            let loader = HtmlLoader::new(path);
            loader.load().await.map_err(|e| anyhow::anyhow!("{}", e))
        }
        _ => anyhow::bail!("Unsupported document format: {}", format),
    }
}

/// Split documents into chunks
pub fn split_documents(
    documents: Vec<Document>,
    chunk_size: usize,
    chunk_overlap: usize,
) -> Vec<Document> {
    let splitter = TextSplitter::new(chunk_size, chunk_overlap);
    splitter.split_documents(&documents)
}

/// Index documents into RAG
pub async fn index_documents(
    rag_tool: &caliche::agent::tools::RagTool,
    _collection: &str,
    documents: Vec<Document>,
) -> anyhow::Result<Vec<String>> {
    if documents.is_empty() {
        return Ok(Vec::new());
    }

    // Convert documents to the format expected by index_batch: Vec<(doc_id, document_content)>
    let mut doc_pairs: Vec<(String, String)> = Vec::with_capacity(documents.len());
    let mut doc_ids: Vec<String> = Vec::with_capacity(documents.len());

    for (idx, doc) in documents.iter().enumerate() {
        // Try to get doc_id from metadata, otherwise generate one
        let doc_id = doc
            .metadata
            .get("doc_id")
            .or_else(|| doc.metadata.get("id"))
            .cloned()
            .unwrap_or_else(|| format!("doc_{}", idx));

        doc_ids.push(doc_id.clone());
        doc_pairs.push((doc_id, doc.content.clone()));
    }

    // Index documents using batch operation
    let (_total_chunks, _total_documents, failures, _token_usage) =
        rag_tool.index_batch(doc_pairs).await?;

    // Extract failed doc_ids from failure messages (format: "doc_id: error message")
    let failed_ids: std::collections::HashSet<String> = failures
        .iter()
        .filter_map(|failure| {
            // Failure format is "doc_id: error message"
            failure.split(':').next().map(|s| s.trim().to_string())
        })
        .collect();

    // Return all doc_ids that were not in the failures list
    let successful_ids: Vec<String> = doc_ids
        .into_iter()
        .filter(|doc_id| !failed_ids.contains(doc_id))
        .collect();

    Ok(successful_ids)
}

pub fn rag_search_mode(mode: Option<&RagCliSearchMode>, default: RagSearchMode) -> RagSearchMode {
    match mode {
        Some(RagCliSearchMode::Dense) => RagSearchMode::Dense,
        Some(RagCliSearchMode::Hybrid) => RagSearchMode::HybridDenseSparse,
        Some(RagCliSearchMode::FullText) => RagSearchMode::FullText,
        Some(RagCliSearchMode::HybridFullText) => RagSearchMode::HybridDenseFullText,
        None => default,
    }
}

pub fn rag_fusion_mode(mode: Option<&RagCliFusionMode>) -> RagFusionMode {
    match mode {
        Some(RagCliFusionMode::Weighted) | None => RagFusionMode::Weighted,
        Some(RagCliFusionMode::Rrf) => RagFusionMode::ReciprocalRank,
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn query_documents(
    rag_service: Arc<RagService>,
    query: &str,
    top_k: usize,
    min_similarity: f32,
    search_mode: RagSearchMode,
    fusion_mode: RagFusionMode,
    dense_weight: Option<f32>,
    sparse_weight: Option<f32>,
    use_mmr: Option<bool>,
    mmr_lambda: Option<f64>,
    use_temporal_decay: Option<bool>,
    temporal_decay_half_life_days: Option<f64>,
) -> anyhow::Result<(Vec<CliSearchResult>, Vec<String>)> {
    let (results, _top_similarity, _token_usage, warnings, _context_stats) = rag_service
        .enhanced_query(
            query,
            top_k,
            min_similarity,
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
            use_mmr,
            mmr_lambda,
            use_temporal_decay,
            temporal_decay_half_life_days,
        )
        .await?;
    let mapped = results
        .into_iter()
        .map(|result| CliSearchResult {
            content: result.chunk.content,
            score: result.score,
        })
        .collect();
    Ok((mapped, warnings))
}

/// Display document processing results
pub fn display_documents(documents: &[Document], limit: Option<usize>) {
    let display_count = limit.unwrap_or(documents.len()).min(documents.len());

    println!("\n=== Loaded {} Documents ===", documents.len());
    for (idx, doc) in documents.iter().take(display_count).enumerate() {
        println!("\n--- Document {} ---", idx + 1);
        println!(
            "Content preview: {}...",
            doc.content.chars().take(200).collect::<String>()
        );
        println!("Metadata: {:?}", doc.metadata);
    }

    if documents.len() > display_count {
        println!(
            "\n... and {} more documents",
            documents.len() - display_count
        );
    }
}

/// Display indexing results
pub fn display_indexing_result(collection: &str, ids: &[String]) {
    println!("\n=== Indexing Complete ===");
    println!("Collection: {}", collection);
    if std::env::var("VECTOR_BACKEND").is_err() {
        println!("Storage: process-local memory (set VECTOR_BACKEND for persistence)");
    }
    println!("Indexed {} documents", ids.len());
    if !ids.is_empty() && ids.len() <= 10 {
        println!("Document IDs: {:?}", ids);
    }
}

pub fn display_query_result(
    collection: &str,
    query: &str,
    search_mode: RagSearchMode,
    fusion_mode: RagFusionMode,
    results: &[CliSearchResult],
    warnings: &[String],
) {
    println!("\n=== Retrieval Results ===");
    println!("Collection: {}", collection);
    println!("Query: {}", query);
    println!("Search mode: {}", search_mode.as_str());
    println!("Fusion mode: {}", fusion_mode.as_str());
    println!("Results: {}", results.len());

    if !warnings.is_empty() {
        println!("Warnings:");
        for warning in warnings {
            println!("- {}", warning);
        }
    }

    for (idx, result) in results.iter().enumerate() {
        println!("\n--- Result {} ---", idx + 1);
        println!("Score: {:.4}", result.score);
        println!("{}", result.content);
    }
}
