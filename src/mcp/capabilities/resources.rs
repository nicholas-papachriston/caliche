//! Resources capability implementation for MCP.

#[cfg(feature = "tools-rag")]
use crate::agent::tools::RagTool;
use crate::mcp::protocol::*;
use anyhow::{Context, Result};
use caliche_mcp::pagination::{DEFAULT_PAGE_SIZE, cmp_name, paginate_owned};
use dashmap::DashMap;
use std::sync::Arc;

pub(crate) fn collect_resources(
    document_collections: Option<&Arc<DashMap<String, crate::mcp::CollectionMetadata>>>,
) -> Vec<McpResource> {
    let mut resources = Vec::new();

    if let Some(collections) = document_collections {
        for entry in collections.iter() {
            let collection_name: &String = entry.key();
            let metadata: &crate::mcp::CollectionMetadata = entry.value();

            resources.push(McpResource {
                uri: format!("rag://{}/", collection_name),
                name: Some(collection_name.clone()),
                title: None,
                description: Some(format!(
                    "Document collection '{}' with {} documents",
                    collection_name, metadata.document_count
                )),
                mime_type: Some("application/json".to_string()),
                size: None,
                icons: None,
                meta: None,
            });
        }
    }
    resources
}

/// Handle `resources/list` with deterministic cursor pagination.
pub async fn handle_resources_list(
    document_collections: Option<&Arc<DashMap<String, crate::mcp::CollectionMetadata>>>,
    params: Option<ResourcesListParams>,
) -> Result<ResourcesListResult> {
    let mut resources = collect_resources(document_collections);

    resources.sort_by(|a, b| {
        cmp_name(
            a.name.as_deref().unwrap_or(a.uri.as_str()),
            b.name.as_deref().unwrap_or(b.uri.as_str()),
        )
    });
    let cursor = params.as_ref().and_then(|p| p.cursor.as_deref());
    let page = paginate_owned(resources, cursor, Some(DEFAULT_PAGE_SIZE))
        .context("invalid resources/list cursor")?;

    Ok(ResourcesListResult {
        resources: page.items,
        next_cursor: page.next_cursor,
    })
}

pub(crate) fn collect_resource_templates() -> Vec<ResourceTemplate> {
    #[cfg(feature = "tools-rag")]
    let templates = vec![
        ResourceTemplate {
            uri_template: "rag://{collection}/".to_string(),
            name: Some("RAG collection summary".to_string()),
            title: None,
            description: Some("Metadata for a RAG document collection (JSON blob)".to_string()),
            mime_type: Some("application/json".to_string()),
            icons: None,
            meta: None,
        },
        ResourceTemplate {
            uri_template: "rag://{collection}/{chunk_id}".to_string(),
            name: Some("RAG chunk text".to_string()),
            title: None,
            description: Some(
                "Plain-text contents of a single chunk from a RAG collection".to_string(),
            ),
            mime_type: Some("text/plain".to_string()),
            icons: None,
            meta: None,
        },
    ];
    #[cfg(not(feature = "tools-rag"))]
    let templates: Vec<ResourceTemplate> = Vec::new();
    templates
}

/// Handle `resources/templates/list`.
pub async fn handle_resources_templates_list(
    params: Option<ResourceTemplatesListParams>,
) -> Result<ResourceTemplatesListResult> {
    let mut templates = collect_resource_templates();

    templates.sort_by(|a, b| {
        cmp_name(
            a.name.as_deref().unwrap_or(a.uri_template.as_str()),
            b.name.as_deref().unwrap_or(b.uri_template.as_str()),
        )
    });
    let cursor = params.as_ref().and_then(|p| p.cursor.as_deref());
    let page = paginate_owned(templates, cursor, Some(DEFAULT_PAGE_SIZE))
        .context("invalid resources/templates/list cursor")?;
    Ok(ResourceTemplatesListResult {
        resource_templates: page.items,
        next_cursor: page.next_cursor,
    })
}

/// Handle resources/read request
#[cfg(feature = "tools-rag")]
pub async fn handle_resources_read(
    rag_tool: Option<&Arc<RagTool>>,
    params: ResourcesReadParams,
) -> Result<ResourcesReadResult> {
    let rag_tool = rag_tool.ok_or_else(|| {
        anyhow::anyhow!("RAG resources require tools-rag and an initialized RagTool")
    })?;

    if !params.uri.starts_with("rag://") {
        return Err(anyhow::anyhow!(
            "Invalid resource URI: must start with 'rag://'"
        ));
    }

    let path = params
        .uri
        .strip_prefix("rag://")
        .ok_or_else(|| anyhow::anyhow!("Failed to strip 'rag://' prefix from URI"))?;
    let parts: Vec<&str> = path.split('/').collect();

    if parts.is_empty() {
        return Err(anyhow::anyhow!(
            "Invalid resource URI: missing collection name"
        ));
    }

    let collection_name = parts[0];

    if parts.len() == 1 || (parts.len() == 2 && parts[1].is_empty()) {
        return Ok(ResourcesReadResult {
            contents: vec![ResourceContents {
                uri: params.uri.clone(),
                mime_type: Some("application/json".to_string()),
                text: Some(format!(
                    r#"{{"collection": "{}", "type": "collection"}}"#,
                    collection_name
                )),
                blob: None,
            }],
        });
    }

    let chunk_id = parts
        .get(1)
        .copied()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow::anyhow!("Invalid resource URI: missing chunk id"))?;

    let chunk_text = rag_tool
        .service()
        .get_chunk_text(chunk_id)
        .await
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Resource not found: no chunk '{}' in collection '{}'",
                chunk_id,
                collection_name
            )
        })?;

    Ok(ResourcesReadResult {
        contents: vec![ResourceContents {
            uri: params.uri.clone(),
            mime_type: Some("text/plain".to_string()),
            text: Some(chunk_text),
            blob: None,
        }],
    })
}
