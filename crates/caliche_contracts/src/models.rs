use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCatalogEntryDto {
    pub id: String,
    pub display_name: String,
    pub configured: bool,
    pub local: bool,
    #[serde(default)]
    pub models: Vec<ModelCatalogEntryDto>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCatalogEntryDto {
    pub id: String,
    pub name: String,
    pub provider_id: String,
    pub provider_display_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelsCatalogResponseDto {
    #[serde(default)]
    pub providers: Vec<ProviderCatalogEntryDto>,
    #[serde(default)]
    pub models: Vec<ModelCatalogEntryDto>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderModelsResponseDto {
    pub provider: ProviderCatalogEntryDto,
    #[serde(default)]
    pub models: Vec<ModelCatalogEntryDto>,
}
