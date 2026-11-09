//! Template registry for managing and discovering templates.

#[cfg(feature = "s3")]
use crate::data::storage::object::StorageConfig;
#[cfg(feature = "s3")]
use crate::data::storage::object::backend::StorageBackend;
use crate::templates::{AgentTemplate, TemplateLoader};
use anyhow::Result;
use dashmap::DashMap;
use std::cmp::Ordering;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Template values are stored as `Arc<AgentTemplate>` so that cloning a reference
// out of the `DashMap` (which must hold a shard lock) is a cheap atomic
// reference-count increment rather than a full deep copy of the struct.

/// Compare two version strings (simple semantic versioning)
fn compare_versions(v1: &str, v2: &str) -> Ordering {
    let parts1: Vec<u32> = v1
        .split('.')
        .filter_map(|s| s.parse::<u32>().ok())
        .collect();
    let parts2: Vec<u32> = v2
        .split('.')
        .filter_map(|s| s.parse::<u32>().ok())
        .collect();

    for (p1, p2) in parts1.iter().zip(parts2.iter()) {
        match p1.cmp(p2) {
            Ordering::Equal => continue,
            other => return other,
        }
    }

    parts1.len().cmp(&parts2.len())
}

/// Registry for managing agent templates.
///
/// Templates are stored as `Arc<AgentTemplate>` so that every lookup or list
/// operation returns a cheaply-cloned `Arc` rather than a deep copy of the
/// template struct.  Callers that need an owned value can do `(*arc).clone()`.
pub struct TemplateRegistry {
    templates: Arc<DashMap<String, Arc<AgentTemplate>>>,
    loader: TemplateLoader,
    template_dirs: Vec<PathBuf>,
    #[cfg(feature = "s3")]
    storage_backends: Vec<(Arc<dyn StorageBackend>, String)>, // (backend, prefix)
}

/// Search filters for templates
#[derive(Debug, Clone, Default)]
pub struct TemplateSearchFilters {
    pub query: Option<String>,
    pub tags: Option<Vec<String>>,
    pub category: Option<String>,
    pub provider: Option<String>,
    pub min_rating: Option<f64>,
}

impl TemplateRegistry {
    /// Create a new template registry
    pub fn new() -> Self {
        Self {
            templates: Arc::new(DashMap::new()),
            loader: TemplateLoader::new(),
            template_dirs: Vec::new(),
            #[cfg(feature = "s3")]
            storage_backends: Vec::new(),
        }
    }

    /// Create a new template registry with storage configuration
    #[cfg(feature = "s3")]
    pub async fn with_storage(config: StorageConfig) -> Result<Self> {
        use crate::templates::storage::create_storage_backend;

        let backend = create_storage_backend(&config).await?;
        let mut registry = Self::new();
        registry
            .storage_backends
            .push((backend, "templates/".to_string()));
        Ok(registry)
    }

    /// Add a template directory to scan
    pub fn add_template_dir<P: AsRef<Path>>(&mut self, dir: P) {
        self.template_dirs.push(dir.as_ref().to_path_buf());
    }

    /// Add a storage source (backend + prefix)
    #[cfg(feature = "s3")]
    pub fn add_storage_source(
        &mut self,
        backend: Arc<dyn StorageBackend>,
        prefix: impl Into<String>,
    ) {
        self.storage_backends.push((backend, prefix.into()));
    }

    /// Load templates from configured directories and storage backends
    pub async fn load_templates(&self) -> Result<usize> {
        let mut count = 0;

        // Load from filesystem directories
        for dir in &self.template_dirs {
            if dir.exists() {
                let templates = self.loader.load_templates_from_dir(dir).await?;
                for template in templates {
                    let id = template.id();
                    self.templates.insert(id, Arc::new(template));
                    count += 1;
                }
            }
        }

        // Load from storage backends
        #[cfg(all(feature = "s3", feature = "templates"))]
        {
            for (backend, prefix) in &self.storage_backends {
                let templates = self
                    .loader
                    .load_templates_from_storage(backend.as_ref(), Some(prefix))
                    .await?;
                for template in templates {
                    let id = template.id();
                    self.templates.insert(id, Arc::new(template));
                    count += 1;
                }
            }
        }

        Ok(count)
    }

    /// Save a template to storage
    #[cfg(all(feature = "s3", feature = "templates"))]
    pub async fn save_template_to_storage(
        &self,
        template: &AgentTemplate,
        backend: &dyn StorageBackend,
        bucket_or_prefix: Option<&str>,
    ) -> Result<()> {
        use crate::templates::storage::{save_template_to_storage, template_storage_key};

        let key = template_storage_key(&template.name, &template.version);
        let full_key = if let Some(prefix) = bucket_or_prefix {
            format!("{}/{}", prefix.trim_end_matches('/'), key)
        } else {
            key
        };

        save_template_to_storage(backend, template, &full_key).await
    }

    /// Register a template.
    pub fn register(&self, template: AgentTemplate) {
        let id = template.id();
        self.templates.insert(id, Arc::new(template));
    }

    /// Get a template by ID (supports versioned lookups like "name:version" or just "name" for latest).
    ///
    /// Returns an `Arc` to the stored template; clone the `Arc` to share the
    /// reference cheaply, or call `(*arc).clone()` for an owned copy.
    pub fn get(&self, id: &str) -> Option<Arc<AgentTemplate>> {
        // If ID contains colon, it's a versioned lookup
        if let Some((name, version)) = id.split_once(':') {
            // Exact version match
            let versioned_id = format!("{}:{}", name.to_lowercase().replace(' ', "_"), version);
            if let Some(template) = self.templates.get(&versioned_id) {
                return Some(Arc::clone(template.value()));
            }
        }

        // Try exact match first
        if let Some(template) = self.templates.get(id) {
            return Some(Arc::clone(template.value()));
        }

        // Try to find latest version by name
        let name_lower = id.to_lowercase().replace(' ', "_");
        let mut matching: Vec<Arc<AgentTemplate>> = self
            .templates
            .iter()
            .filter(|entry| entry.key().starts_with(&format!("{}:", name_lower)))
            .map(|entry| Arc::clone(entry.value()))
            .collect();

        if matching.is_empty() {
            return None;
        }

        // Sort by version (semantic versioning) and return latest
        matching.sort_by(|a, b| compare_versions(&a.version, &b.version));

        matching.pop()
    }

    /// Get a specific version of a template.
    pub fn get_version(&self, name: &str, version: &str) -> Option<Arc<AgentTemplate>> {
        let id = format!("{}:{}", name.to_lowercase().replace(' ', "_"), version);
        self.templates
            .get(&id)
            .map(|entry| Arc::clone(entry.value()))
    }

    /// List all versions of a template.
    pub fn list_versions(&self, name: &str) -> Vec<Arc<AgentTemplate>> {
        let name_lower = name.to_lowercase().replace(' ', "_");
        let prefix = format!("{}:", name_lower);
        let mut matching: Vec<Arc<AgentTemplate>> = self
            .templates
            .iter()
            .filter(|entry| entry.key().starts_with(&prefix))
            .map(|entry| Arc::clone(entry.value()))
            .collect();

        // Sort by version
        matching.sort_by(|a, b| compare_versions(&a.version, &b.version));
        matching
    }

    /// List all templates.
    pub fn list(&self) -> Vec<Arc<AgentTemplate>> {
        self.templates
            .iter()
            .map(|entry| Arc::clone(entry.value()))
            .collect()
    }

    /// Search templates by tags.
    pub fn search_by_tags(&self, tags: &[String]) -> Vec<Arc<AgentTemplate>> {
        self.templates
            .iter()
            .filter(|entry| {
                let template_tags = &entry.value().tags;
                tags.iter().any(|tag| template_tags.contains(tag))
            })
            .map(|entry| Arc::clone(entry.value()))
            .collect()
    }

    /// Search templates by provider.
    pub fn search_by_provider(&self, provider: &str) -> Vec<Arc<AgentTemplate>> {
        self.templates
            .iter()
            .filter(|entry| entry.value().config.provider == provider)
            .map(|entry| Arc::clone(entry.value()))
            .collect()
    }

    /// Search templates by category.
    pub fn search_by_category(&self, category: &str) -> Vec<Arc<AgentTemplate>> {
        self.templates
            .iter()
            .filter(|entry| {
                entry
                    .value()
                    .category
                    .as_ref()
                    .map(|c| c.eq_ignore_ascii_case(category))
                    .unwrap_or(false)
            })
            .map(|entry| Arc::clone(entry.value()))
            .collect()
    }

    /// Advanced search with filters.
    pub fn search(&self, filters: &TemplateSearchFilters) -> Vec<Arc<AgentTemplate>> {
        self.templates
            .iter()
            .filter(|entry| {
                let template = entry.value();

                // Query search (name, description)
                if let Some(ref query) = filters.query {
                    let query_lower = query.to_lowercase();
                    let matches_name = template.name.to_lowercase().contains(&query_lower);
                    let matches_desc = template.description.to_lowercase().contains(&query_lower);
                    if !matches_name && !matches_desc {
                        return false;
                    }
                }

                // Tag filter
                if let Some(ref tags) = filters.tags
                    && !tags.iter().any(|tag| template.tags.contains(tag))
                {
                    return false;
                }

                // Category filter
                if let Some(ref category) = filters.category
                    && template
                        .category
                        .as_ref()
                        .map(|c| !c.eq_ignore_ascii_case(category))
                        .unwrap_or(true)
                {
                    return false;
                }

                // Provider filter
                if let Some(ref provider) = filters.provider
                    && template.config.provider != *provider
                {
                    return false;
                }

                // Rating filter
                if let Some(min_rating) = filters.min_rating {
                    if let Some(ref stats) = template.usage_stats {
                        if let Some(rating) = stats.average_rating {
                            if rating < min_rating {
                                return false;
                            }
                        } else {
                            return false; // No rating means doesn't meet minimum
                        }
                    } else {
                        return false; // No stats means doesn't meet minimum
                    }
                }

                true
            })
            .map(|entry| Arc::clone(entry.value()))
            .collect()
    }

    /// Get all categories
    pub fn get_categories(&self) -> Vec<String> {
        let mut categories: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entry in self.templates.iter() {
            if let Some(ref category) = entry.value().category {
                categories.insert(category.clone());
            }
        }
        let mut categories: Vec<String> = categories.into_iter().collect();
        categories.sort();
        categories
    }

    /// Get all tags
    pub fn get_all_tags(&self) -> Vec<String> {
        let mut tags: std::collections::HashSet<String> = std::collections::HashSet::new();
        for entry in self.templates.iter() {
            for tag in &entry.value().tags {
                tags.insert(tag.clone());
            }
        }
        let mut tags: Vec<String> = tags.into_iter().collect();
        tags.sort();
        tags
    }

    /// Fork a template (create a copy with modifications).
    ///
    /// Returns an owned `AgentTemplate`; the clone is made from the `Arc` held in
    /// the registry so the original remains shared.
    pub fn fork_template(
        &self,
        id: &str,
        new_name: Option<String>,
        new_version: Option<String>,
    ) -> Result<AgentTemplate> {
        let arc = self
            .get(id)
            .ok_or_else(|| anyhow::anyhow!("Template not found: {}", id))?;

        let mut forked = (*arc).clone();
        if let Some(name) = new_name {
            forked.name = name;
        }
        if let Some(version) = new_version {
            forked.version = version;
        }

        Ok(forked)
    }

    /// Remove a template
    pub fn remove(&self, id: &str) -> bool {
        self.templates.remove(id).is_some()
    }

    /// Get template count
    pub fn count(&self) -> usize {
        self.templates.len()
    }
}

impl Default for TemplateRegistry {
    fn default() -> Self {
        Self::new()
    }
}
