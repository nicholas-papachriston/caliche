//! In-memory plugin index with persistence coordination.

use crate::plugins::persistence::PluginStore;
use crate::plugins::schema::PluginDefinition;
use crate::plugins::validation::{ValidationContext, validate_plugin};
use anyhow::{Context, Result, anyhow};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry that holds loaded plugin definitions and persists them via PluginStore.
#[derive(Debug)]
pub struct PluginRegistry {
    plugins: RwLock<HashMap<String, Arc<PluginDefinition>>>,
    store: Arc<PluginStore>,
}

impl PluginRegistry {
    pub fn new(store: Arc<PluginStore>) -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
            store,
        }
    }

    /// Register a new plugin (validate, persist, then add to in-memory index).
    pub async fn register(&self, def: PluginDefinition, ctx: &ValidationContext) -> Result<()> {
        validate_plugin(&def, ctx).context("plugin validation failed")?;
        self.store.put(&def).await.context("persist plugin")?;
        let arc = Arc::new(def);
        self.plugins.write().insert(arc.name.clone(), arc);
        Ok(())
    }

    /// Update an existing plugin (validate, persist, then replace in index).
    pub async fn update(&self, def: PluginDefinition, ctx: &ValidationContext) -> Result<()> {
        validate_plugin(&def, ctx).context("plugin validation failed")?;
        let name = def.name.clone();
        self.store.put(&def).await.context("persist plugin")?;
        self.plugins.write().insert(name, Arc::new(def));
        Ok(())
    }

    /// Delete a plugin by name (from store and in-memory).
    pub async fn delete(&self, name: &str) -> Result<()> {
        self.store
            .delete_by_name(name)
            .await
            .context("delete from store")?;
        self.plugins.write().remove(name);
        Ok(())
    }

    /// Get a plugin by name.
    pub fn get(&self, name: &str) -> Option<Arc<PluginDefinition>> {
        self.plugins.read().get(name).cloned()
    }

    /// List all loaded plugins.
    pub fn list(&self) -> Vec<Arc<PluginDefinition>> {
        self.plugins.read().values().cloned().collect()
    }

    /// Export a plugin as JSON string.
    #[expect(
        clippy::unused_async,
        reason = "public API already awaited by API/CLI plugin handlers"
    )]
    pub async fn export(&self, name: &str) -> Result<String> {
        let def = self
            .get(name)
            .ok_or_else(|| anyhow!("plugin not found: {}", name))?;
        serde_json::to_string_pretty(def.as_ref()).context("serialize plugin")
    }

    /// Load all plugins from the store into memory (e.g. on startup).
    pub async fn load_all(&self) -> Result<Vec<Arc<PluginDefinition>>> {
        let defs = self.store.load_all().await.context("load from store")?;
        let mut guard = self.plugins.write();
        guard.clear();
        let mut out = Vec::with_capacity(defs.len());
        for def in defs {
            let arc = Arc::new(def);
            out.push(Arc::clone(&arc));
            guard.insert(arc.name.clone(), arc);
        }
        Ok(out)
    }
}
