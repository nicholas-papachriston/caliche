//! Plugin authoring and catalog types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginSummary {
    pub name: String,
    pub version: String,
    pub description: String,
    pub plugin_type: String,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginListResponse {
    pub plugins: Vec<PluginSummary>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginGuide {
    pub plugin_type: String,
    pub title: String,
    pub summary: String,
    pub when_to_use: String,
    #[serde(default)]
    pub notes: Vec<String>,
    pub starter_definition: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginCatalogResponse {
    pub guides: Vec<PluginGuide>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginValidationResponse {
    pub valid: bool,
    pub normalized_definition: serde_json::Value,
    #[serde(default)]
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginSaveResponse {
    pub created: bool,
    pub summary: PluginSummary,
    pub definition: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PluginExportResponse {
    pub name: String,
    pub definition: String,
}
