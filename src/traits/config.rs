//! Configuration source abstraction for layered config loading.
//!
//! This module provides a unified interface for loading configuration from
//! multiple sources. Implement the `ConfigSource` trait to add custom
//! configuration sources like remote stores, databases, or custom file formats.

use anyhow::Result;
use async_trait::async_trait;
use serde::de::DeserializeOwned;
use std::sync::Arc;

/// Trait for configuration sources that can retrieve configuration values
#[async_trait]
pub trait ConfigSource: Send + Sync + std::fmt::Debug {
    /// Get a configuration value as a string
    ///
    /// # Arguments
    /// * `key` - Configuration key (e.g., "api.timeout", "model.name")
    ///
    /// # Returns
    /// * `Ok(Some(String))` - Configuration value if found
    /// * `Ok(None)` - Key not found
    /// * `Err` - Error retrieving configuration
    async fn get_value(&self, key: &str) -> Result<Option<String>>;

    /// Get the name of this configuration source (for debugging/logging)
    fn source_name(&self) -> &str;
}

/// Helper trait for typed configuration (auto-implemented for all ConfigSource implementations)
pub trait TypedConfigSource: ConfigSource {
    /// Get a typed configuration value
    ///
    /// # Type Parameters
    /// * `T` - Type to deserialize (must implement `DeserializeOwned`)
    ///
    /// # Arguments
    /// * `key` - Configuration key
    ///
    /// # Returns
    /// * `Ok(Some(T))` - Configuration value if found and deserializable
    /// * `Ok(None)` - Key not found
    /// * `Err` - Error retrieving or deserializing
    fn get_typed<T: DeserializeOwned>(
        &self,
        key: &str,
    ) -> impl std::future::Future<Output = Result<Option<T>>> + Send {
        async move {
            if let Some(value_str) = self.get_value(key).await? {
                match serde_json::from_str(&value_str) {
                    Ok(value) => Ok(Some(value)),
                    Err(e) => Err(anyhow::anyhow!(
                        "Failed to deserialize config value for key '{}': {}",
                        key,
                        e
                    )),
                }
            } else {
                Ok(None)
            }
        }
    }
}

// Auto-implement for all ConfigSource implementations
impl<T: ConfigSource> TypedConfigSource for T {}

/// Configuration source that reads from environment variables
#[derive(Debug, Clone)]
pub struct EnvironmentConfigSource;

impl EnvironmentConfigSource {
    /// Create a new environment configuration source
    pub fn new() -> Self {
        Self
    }
}

impl Default for EnvironmentConfigSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConfigSource for EnvironmentConfigSource {
    async fn get_value(&self, key: &str) -> Result<Option<String>> {
        Ok(caliche_contracts::credential_env::var(key).ok())
    }

    fn source_name(&self) -> &str {
        "environment"
    }
}

/// Configuration source that reads from JSON/YAML files
#[derive(Debug, Clone)]
pub struct FileConfigSource {
    path: String,
    format: ConfigFileFormat,
}

#[derive(Debug, Clone, Copy)]
pub enum ConfigFileFormat {
    Json,
    Yaml,
}

impl FileConfigSource {
    /// Create a new file configuration source
    ///
    /// # Arguments
    /// * `path` - Path to the configuration file
    /// * `format` - File format (JSON or YAML)
    pub fn new(path: impl Into<String>, format: ConfigFileFormat) -> Self {
        Self {
            path: path.into(),
            format,
        }
    }

    /// Create a JSON file configuration source
    pub fn json(path: impl Into<String>) -> Self {
        Self::new(path, ConfigFileFormat::Json)
    }

    /// Create a YAML file configuration source
    pub fn yaml(path: impl Into<String>) -> Self {
        Self::new(path, ConfigFileFormat::Yaml)
    }
}

#[async_trait]
impl ConfigSource for FileConfigSource {
    async fn get_value(&self, key: &str) -> Result<Option<String>> {
        let content = tokio::fs::read_to_string(&self.path)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to read config file '{}': {}", self.path, e))?;

        let value: serde_json::Value = match self.format {
            ConfigFileFormat::Json => serde_json::from_str(&content)
                .map_err(|e| anyhow::anyhow!("Failed to parse JSON config file: {}", e))?,
            ConfigFileFormat::Yaml => {
                #[cfg(feature = "templates")]
                {
                    serde_norway::from_str(&content)
                        .map_err(|e| anyhow::anyhow!("Failed to parse YAML config file: {}", e))?
                }
                #[cfg(not(feature = "templates"))]
                {
                    return Err(anyhow::anyhow!(
                        "YAML support requires the 'templates' feature to be enabled"
                    ));
                }
            }
        };

        // Support dot notation for nested keys (e.g., "api.timeout")
        let mut current = &value;
        for part in key.split('.') {
            match current.get(part) {
                Some(v) => current = v,
                None => return Ok(None),
            }
        }

        // Convert to string
        match current {
            serde_json::Value::String(s) => Ok(Some(s.clone())),
            serde_json::Value::Number(n) => Ok(Some(n.to_string())),
            serde_json::Value::Bool(b) => Ok(Some(b.to_string())),
            serde_json::Value::Null => Ok(None),
            _ => Ok(Some(serde_json::to_string(current)?)),
        }
    }

    fn source_name(&self) -> &str {
        "file"
    }
}

/// Configuration source that chains multiple sources, trying each in order
#[derive(Debug)]
pub struct ChainedConfigSource {
    sources: Vec<Arc<dyn ConfigSource>>,
}

impl ChainedConfigSource {
    /// Create a new chained configuration source
    pub fn new(sources: Vec<Arc<dyn ConfigSource>>) -> Self {
        Self { sources }
    }
}

#[async_trait]
impl ConfigSource for ChainedConfigSource {
    async fn get_value(&self, key: &str) -> Result<Option<String>> {
        for source in &self.sources {
            match source.get_value(key).await {
                Ok(Some(value)) => return Ok(Some(value)),
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(
                        "Config source '{}' failed for key '{}': {}",
                        source.source_name(),
                        key,
                        e
                    );
                    continue;
                }
            }
        }
        Ok(None)
    }

    fn source_name(&self) -> &str {
        "chained"
    }
}

/// Configuration source wrapper that caches values with TTL
#[derive(Debug)]
pub struct CachedConfigSource {
    inner: Arc<dyn ConfigSource>,
    cache: tokio::sync::Mutex<std::collections::HashMap<String, (String, std::time::SystemTime)>>,
    ttl: std::time::Duration,
}

impl CachedConfigSource {
    /// Create a new cached configuration source
    ///
    /// # Arguments
    /// * `inner` - Inner configuration source to wrap
    /// * `ttl` - Time-to-live for cached values
    pub fn new(inner: Arc<dyn ConfigSource>, ttl: std::time::Duration) -> Self {
        Self {
            inner,
            cache: tokio::sync::Mutex::new(std::collections::HashMap::new()),
            ttl,
        }
    }
}

#[async_trait]
impl ConfigSource for CachedConfigSource {
    async fn get_value(&self, key: &str) -> Result<Option<String>> {
        let now = std::time::SystemTime::now();

        // Check cache
        {
            let cache = self.cache.lock().await;
            if let Some((value, cached_at)) = cache.get(key)
                && now.duration_since(*cached_at).unwrap_or(self.ttl) < self.ttl
            {
                return Ok(Some(value.clone()));
            }
        }

        // Fetch from inner source
        let value = self.inner.get_value(key).await?;

        // Update cache if value found
        if let Some(ref v) = value {
            let mut cache = self.cache.lock().await;
            cache.insert(key.to_string(), (v.clone(), now));
        }

        Ok(value)
    }

    fn source_name(&self) -> &str {
        "cached"
    }
}

/// No-op configuration source that always returns None (useful for testing)
#[derive(Debug, Clone)]
pub struct NoOpConfigSource;

impl NoOpConfigSource {
    /// Create a new no-op configuration source
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpConfigSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConfigSource for NoOpConfigSource {
    async fn get_value(&self, _key: &str) -> Result<Option<String>> {
        Ok(None)
    }

    fn source_name(&self) -> &str {
        "no-op"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_no_op_config_source() {
        let source = NoOpConfigSource::new();
        let value = source
            .get_value("any_key")
            .await
            .expect("get_value should succeed");
        assert_eq!(value, None);
    }
}
