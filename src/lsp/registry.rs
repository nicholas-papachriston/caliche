//! LSP server registry and management.

use crate::lsp::config::{LspConfig, ServerConfig};
use std::collections::HashMap;
use std::path::Path;

/// Registry of LSP servers by language.
#[derive(Debug, Clone)]
pub struct LspServerRegistry {
    /// Language → server configuration mapping.
    servers: HashMap<String, ServerConfig>,
}

impl LspServerRegistry {
    /// Create a new registry with default servers.
    pub fn new() -> Self {
        Self {
            servers: LspConfig::default_servers(),
        }
    }

    /// Create a registry from LSP configuration.
    pub fn from_config(config: &LspConfig) -> Self {
        Self {
            servers: config.servers.clone(),
        }
    }

    /// Get server configuration for a file extension.
    pub fn get_server_for_ext(&self, ext: &str) -> Option<(&str, &ServerConfig)> {
        for (lang, config) in &self.servers {
            if config.enabled && config.extensions.iter().any(|e| e == ext) {
                return Some((lang, config));
            }
        }
        None
    }

    /// Get server configuration for a language.
    pub fn get_server(&self, lang: &str) -> Option<&ServerConfig> {
        self.servers.get(lang).filter(|c| c.enabled)
    }

    /// Get language for a file path.
    pub fn get_language_for_path(&self, path: &Path) -> Option<&str> {
        let ext = path.extension()?.to_str()?;
        let ext = format!(".{}", ext);
        self.get_server_for_ext(&ext).map(|(lang, _)| lang)
    }

    /// Register a new server.
    pub fn register(&mut self, lang: String, config: ServerConfig) {
        self.servers.insert(lang, config);
    }

    /// Get all registered servers.
    pub fn servers(&self) -> &HashMap<String, ServerConfig> {
        &self.servers
    }

    /// Check if a server is enabled for a language.
    pub fn is_enabled(&self, lang: &str) -> bool {
        self.servers.get(lang).map(|c| c.enabled).unwrap_or(false)
    }
}

impl Default for LspServerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_default() {
        let registry = LspServerRegistry::new();
        assert!(registry.get_server("rust").is_some());
        assert!(registry.get_server("python").is_some());
    }

    #[test]
    fn test_get_server_for_ext() {
        let registry = LspServerRegistry::new();
        let (lang, _) = registry
            .get_server_for_ext(".rs")
            .expect("should find rust");
        assert_eq!(lang, "rust");
    }

    #[test]
    fn test_get_language_for_path() {
        let registry = LspServerRegistry::new();
        let lang = registry
            .get_language_for_path(Path::new("main.rs"))
            .expect("should find rust");
        assert_eq!(lang, "rust");

        let lang = registry
            .get_language_for_path(Path::new("script.py"))
            .expect("should find python");
        assert_eq!(lang, "python");
    }

    #[test]
    fn test_is_enabled() {
        let registry = LspServerRegistry::new();
        assert!(registry.is_enabled("rust"));
        assert!(registry.is_enabled("python"));
        assert!(!registry.is_enabled("unknown"));
    }
}
