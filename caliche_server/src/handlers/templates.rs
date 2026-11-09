//! API handlers for template management.

use crate::AppState;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::templates::{
    AgentTemplate, TemplateRegistry, TemplateSearchFilters, TemplateValidator,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Template list response
#[derive(Serialize)]
pub struct TemplateListResponse {
    templates: Vec<TemplateInfo>,
    count: usize,
}

/// Template information (simplified for API)
#[derive(Serialize)]
pub struct TemplateInfo {
    id: String,
    name: String,
    version: String,
    description: String,
    author: Option<String>,
    tags: Vec<String>,
    provider: String,
    model: Option<String>,
}

impl From<&AgentTemplate> for TemplateInfo {
    fn from(template: &AgentTemplate) -> Self {
        Self {
            id: template.id(),
            name: template.name.clone(),
            version: template.version.clone(),
            description: template.description.clone(),
            author: template.author.clone(),
            tags: template.tags.clone(),
            provider: template.config.provider.clone(),
            model: template.config.model.clone(),
        }
    }
}

/// Template detail response
#[derive(Serialize)]
pub struct TemplateDetailResponse {
    template: AgentTemplate,
}

/// Template validation request
#[derive(Deserialize)]
pub struct ValidateTemplateRequest {
    template: AgentTemplate,
}

/// Template validation response
#[derive(Serialize)]
pub struct ValidateTemplateResponse {
    valid: bool,
    errors: Vec<String>,
}

/// Template instantiation request
#[derive(Deserialize)]
pub struct InstantiateTemplateRequest {
    template_id: String,
    #[serde(default)]
    overrides: serde_json::Value,
    #[serde(default)]
    tools: Option<Vec<String>>,
}

/// Template instantiation response
#[derive(Serialize)]
pub struct InstantiateTemplateResponse {
    agent_id: String,
    provider: String,
    model: String,
}

/// List all templates
pub async fn list_templates_handler(
    State(_state): State<AppState>,
) -> Result<Json<TemplateListResponse>, (StatusCode, String)> {
    let mut registry = TemplateRegistry::new();
    let dir = PathBuf::from("templates");

    if dir.exists() {
        registry.add_template_dir(&dir);
    }

    let count = registry
        .load_templates()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let templates: Vec<TemplateInfo> = registry
        .list()
        .iter()
        .map(|t| TemplateInfo::from(t.as_ref()))
        .collect();

    Ok(Json(TemplateListResponse { templates, count }))
}

/// Get template details
pub async fn get_template_handler(
    State(_state): State<AppState>,
    Path(template_id): Path<String>,
) -> Result<Json<TemplateDetailResponse>, (StatusCode, String)> {
    let mut registry = TemplateRegistry::new();
    let dir = PathBuf::from("templates");

    if dir.exists() {
        registry.add_template_dir(&dir);
    }

    registry
        .load_templates()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let arc = registry.get(&template_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Template not found: {}", template_id),
        )
    })?;

    Ok(Json(TemplateDetailResponse {
        template: arc.as_ref().clone(),
    }))
}

/// Validate a template
pub async fn validate_template_handler(
    State(_state): State<AppState>,
    Json(request): Json<ValidateTemplateRequest>,
) -> Result<Json<ValidateTemplateResponse>, (StatusCode, String)> {
    let validator = TemplateValidator::new();

    match validator.validate(&request.template) {
        Ok(_) => Ok(Json(ValidateTemplateResponse {
            valid: true,
            errors: Vec::new(),
        })),
        Err(e) => Ok(Json(ValidateTemplateResponse {
            valid: false,
            errors: vec![e.to_string()],
        })),
    }
}

/// Instantiate an agent from a template
pub async fn instantiate_template_handler(
    State(_state): State<AppState>,
    Json(request): Json<InstantiateTemplateRequest>,
) -> Result<Json<InstantiateTemplateResponse>, (StatusCode, String)> {
    let mut registry = TemplateRegistry::new();
    let dir = PathBuf::from("templates");

    if dir.exists() {
        registry.add_template_dir(&dir);
    }

    registry
        .load_templates()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let arc = registry.get(&request.template_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Template not found: {}", request.template_id),
        )
    })?;

    let mut template = arc.as_ref().clone();

    // Apply overrides if provided
    if !request.overrides.is_null() {
        // Simple override application - could be enhanced
        if let Some(model) = request.overrides.get("model").and_then(|v| v.as_str()) {
            template.config.model = Some(model.to_string());
        }
        if let Some(provider) = request.overrides.get("provider").and_then(|v| v.as_str()) {
            template.config.provider = provider.to_string();
        }
    }

    // Apply tools override if provided (takes precedence over template tools)
    if let Some(ref tools) = request.tools {
        template.config.tools = Some(tools.clone());
    }

    let builder = template.instantiate().await.map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to instantiate template: {}", e),
        )
    })?;

    // Build the agent (this creates it but doesn't store it)
    let agent = Box::pin(builder.build()).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to build agent: {}", e),
        )
    })?;

    // Generate a simple agent ID
    let agent_id = uuid::Uuid::now_v7().to_string();

    Ok(Json(InstantiateTemplateResponse {
        agent_id,
        provider: agent.provider().to_string(),
        model: agent.model().to_string(),
    }))
}

/// Upload a custom template
pub async fn upload_template_handler(
    State(_state): State<AppState>,
    Json(template): Json<AgentTemplate>,
) -> Result<Json<TemplateDetailResponse>, (StatusCode, String)> {
    let validator = TemplateValidator::new();
    validator
        .validate(&template)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid template: {}", e)))?;

    // In a real implementation, you would save this to a database or file system
    // For now, we just return the validated template

    Ok(Json(TemplateDetailResponse { template }))
}

/// Delete a template
pub async fn delete_template_handler(
    State(_state): State<AppState>,
    Path(_template_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    // In a real implementation, you would delete from database or file system
    // For now, we just return success

    Ok(StatusCode::NO_CONTENT)
}

/// Get template categories
#[derive(Serialize)]
pub struct CategoriesResponse {
    categories: Vec<String>,
}

pub async fn get_categories_handler(
    State(_state): State<AppState>,
) -> Result<Json<CategoriesResponse>, (StatusCode, String)> {
    let mut registry = TemplateRegistry::new();
    let dir = PathBuf::from("templates");

    if dir.exists() {
        registry.add_template_dir(&dir);
    }

    registry
        .load_templates()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let categories = registry.get_categories();

    Ok(Json(CategoriesResponse { categories }))
}

/// Search templates
#[derive(Deserialize)]
pub struct SearchQuery {
    q: Option<String>,
    tags: Option<String>,
    category: Option<String>,
    provider: Option<String>,
    min_rating: Option<f64>,
}

pub async fn search_templates_handler(
    State(_state): State<AppState>,
    Query(query): Query<SearchQuery>,
) -> Result<Json<TemplateListResponse>, (StatusCode, String)> {
    let mut registry = TemplateRegistry::new();
    let dir = PathBuf::from("templates");

    if dir.exists() {
        registry.add_template_dir(&dir);
    }

    registry
        .load_templates()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let filters = TemplateSearchFilters {
        query: query.q,
        tags: query
            .tags
            .map(|t| t.split(',').map(|s| s.trim().to_string()).collect()),
        category: query.category,
        provider: query.provider,
        min_rating: query.min_rating,
    };

    let templates: Vec<TemplateInfo> = registry
        .search(&filters)
        .iter()
        .map(|t| TemplateInfo::from(t.as_ref()))
        .collect();

    let count = templates.len();

    Ok(Json(TemplateListResponse { templates, count }))
}

/// Fork a template
#[derive(Deserialize)]
pub struct ForkTemplateRequest {
    template_id: String,
    new_name: Option<String>,
    new_version: Option<String>,
}

pub async fn fork_template_handler(
    State(_state): State<AppState>,
    Json(request): Json<ForkTemplateRequest>,
) -> Result<Json<TemplateDetailResponse>, (StatusCode, String)> {
    let mut registry = TemplateRegistry::new();
    let dir = PathBuf::from("templates");

    if dir.exists() {
        registry.add_template_dir(&dir);
    }

    registry
        .load_templates()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let forked = registry
        .fork_template(&request.template_id, request.new_name, request.new_version)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(TemplateDetailResponse { template: forked }))
}

/// Export template as YAML
pub async fn export_template_handler(
    State(_state): State<AppState>,
    Path(template_id): Path<String>,
) -> Result<String, (StatusCode, String)> {
    let mut registry = TemplateRegistry::new();
    let dir = PathBuf::from("templates");

    if dir.exists() {
        registry.add_template_dir(&dir);
    }

    registry
        .load_templates()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let arc = registry.get(&template_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Template not found: {}", template_id),
        )
    })?;

    let yaml = serde_norway::to_string(arc.as_ref()).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to serialize template: {}", e),
        )
    })?;

    Ok(yaml)
}

/// Import template from YAML/JSON
#[derive(Deserialize)]
pub struct ImportTemplateRequest {
    content: String,
    format: Option<String>, // "yaml" or "json"
}

pub async fn import_template_handler(
    State(_state): State<AppState>,
    Json(request): Json<ImportTemplateRequest>,
) -> Result<Json<TemplateDetailResponse>, (StatusCode, String)> {
    use caliche::templates::TemplateLoader;

    let loader = TemplateLoader::new();
    let format = request.format.as_deref().unwrap_or("yaml");

    let template = if format == "json" || request.content.trim_start().starts_with('{') {
        loader
            .load_template_from_str(&request.content, None)
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to parse template: {}", e),
                )
            })?
    } else {
        loader
            .load_template_from_str(&request.content, None)
            .map_err(|e| {
                (
                    StatusCode::BAD_REQUEST,
                    format!("Failed to parse template: {}", e),
                )
            })?
    };

    Ok(Json(TemplateDetailResponse { template }))
}

/// Instantiate template with variables
#[derive(Deserialize)]
pub struct InstantiateWithVariablesRequest {
    template_id: String,
    variables: Option<HashMap<String, serde_json::Value>>,
    #[serde(default)]
    tools: Option<Vec<String>>,
}

pub async fn instantiate_with_variables_handler(
    State(_state): State<AppState>,
    Json(request): Json<InstantiateWithVariablesRequest>,
) -> Result<Json<InstantiateTemplateResponse>, (StatusCode, String)> {
    let mut registry = TemplateRegistry::new();
    let dir = PathBuf::from("templates");

    if dir.exists() {
        registry.add_template_dir(&dir);
    }

    registry
        .load_templates()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let arc = registry.get(&request.template_id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Template not found: {}", request.template_id),
        )
    })?;

    let mut template = arc.as_ref().clone();

    // Apply tools override if provided (takes precedence over template tools)
    if let Some(ref tools) = request.tools {
        template.config.tools = Some(tools.clone());
    }

    let builder = template
        .instantiate_with_variables(request.variables)
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("Failed to instantiate template: {}", e),
            )
        })?;

    let agent = Box::pin(builder.build()).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to build agent: {}", e),
        )
    })?;

    let agent_id = uuid::Uuid::now_v7().to_string();

    Ok(Json(InstantiateTemplateResponse {
        agent_id,
        provider: agent.provider().to_string(),
        model: agent.model().to_string(),
    }))
}
