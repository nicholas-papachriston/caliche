//! LSP configuration parsing and management.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LSP configuration from TOML or direct builder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspConfig {
    /// Whether LSP is enabled globally.
    #[serde(default = "default_enabled")]
    pub enabled: bool,

    /// Per-language server configurations.
    #[serde(default)]
    pub servers: HashMap<String, ServerConfig>,

    /// Timeout for LSP requests in seconds.
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,

    /// Timeout for server startup in seconds.
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,

    /// Maximum number of cached diagnostics per file.
    #[serde(default = "default_max_diagnostics")]
    pub max_diagnostics_per_file: usize,

    /// Maximum number of files to keep diagnostics for.
    #[serde(default = "default_max_cached_files")]
    pub max_cached_files: usize,

    /// Settle time (ms) to wait for diagnostics after change.
    #[serde(default = "default_settle_time_ms")]
    pub settle_time_ms: u64,
}

fn default_enabled() -> bool {
    false
}

fn default_request_timeout() -> u64 {
    5
}

fn default_startup_timeout() -> u64 {
    10
}

fn default_max_diagnostics() -> usize {
    100
}

fn default_max_cached_files() -> usize {
    100
}

fn default_settle_time_ms() -> u64 {
    500
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
            servers: Self::default_servers(),
            request_timeout_secs: default_request_timeout(),
            startup_timeout_secs: default_startup_timeout(),
            max_diagnostics_per_file: default_max_diagnostics(),
            max_cached_files: default_max_cached_files(),
            settle_time_ms: default_settle_time_ms(),
        }
    }
}

impl LspConfig {
    /// Create a new LSP configuration with default servers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse LSP config from TOML.
    pub fn from_toml(content: &str) -> anyhow::Result<Self> {
        let config: LspConfig = toml::from_str(content)?;
        Ok(config)
    }

    /// Get default server configurations for common languages.
    pub fn default_servers() -> HashMap<String, ServerConfig> {
        let mut servers = HashMap::new();

        // Rust
        servers.insert(
            "rust".to_string(),
            ServerConfig {
                command: "rust-analyzer".to_string(),
                args: vec![],
                extensions: vec![".rs".to_string()],
                root_patterns: vec!["Cargo.toml".to_string()],
                env: HashMap::new(),
                enabled: true,
            },
        );

        // Python
        servers.insert(
            "python".to_string(),
            ServerConfig {
                command: "pyright-langserver".to_string(),
                args: vec!["--stdio".to_string()],
                extensions: vec![".py".to_string()],
                root_patterns: vec![
                    "pyproject.toml".to_string(),
                    "setup.py".to_string(),
                    "requirements.txt".to_string(),
                ],
                env: HashMap::new(),
                enabled: true,
            },
        );

        // TypeScript/JavaScript
        servers.insert(
            "typescript".to_string(),
            ServerConfig {
                command: "typescript-language-server".to_string(),
                args: vec!["--stdio".to_string()],
                extensions: vec![
                    ".ts".to_string(),
                    ".tsx".to_string(),
                    ".js".to_string(),
                    ".jsx".to_string(),
                ],
                root_patterns: vec!["package.json".to_string(), "tsconfig.json".to_string()],
                env: HashMap::new(),
                enabled: true,
            },
        );

        // Go
        servers.insert(
            "go".to_string(),
            ServerConfig {
                command: "gopls".to_string(),
                args: vec![],
                extensions: vec![".go".to_string()],
                root_patterns: vec!["go.mod".to_string()],
                env: HashMap::new(),
                enabled: true,
            },
        );

        // C/C++
        servers.insert(
            "cpp".to_string(),
            ServerConfig {
                command: "clangd".to_string(),
                args: vec![],
                extensions: vec![
                    ".c".to_string(),
                    ".cpp".to_string(),
                    ".h".to_string(),
                    ".hpp".to_string(),
                ],
                root_patterns: vec!["compile_commands.json".to_string(), ".clangd".to_string()],
                env: HashMap::new(),
                enabled: true,
            },
        );

        servers
    }

    /// Get server config for a file extension.
    pub fn get_server_for_ext(&self, ext: &str) -> Option<(&str, &ServerConfig)> {
        for (lang, config) in &self.servers {
            if config.enabled && config.extensions.iter().any(|e| e == ext) {
                return Some((lang, config));
            }
        }
        None
    }

    /// Get server config for a language.
    pub fn get_server(&self, lang: &str) -> Option<&ServerConfig> {
        self.servers.get(lang).filter(|c| c.enabled)
    }
}

/// Configuration for a single LSP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Command to start the server (e.g., "rust-analyzer").
    pub command: String,

    /// Arguments passed to the command.
    #[serde(default)]
    pub args: Vec<String>,

    /// File extensions this server handles (e.g., [".rs"]).
    pub extensions: Vec<String>,

    /// Files that indicate the workspace root.
    #[serde(default)]
    pub root_patterns: Vec<String>,

    /// Environment variables for the server process.
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Whether this server is enabled.
    #[serde(default = "default_server_enabled")]
    pub enabled: bool,
}

fn default_server_enabled() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LspConfig::default();
        assert!(!config.enabled);
        assert!(config.servers.contains_key("rust"));
        assert!(config.servers.contains_key("python"));
        assert!(config.servers.contains_key("typescript"));
    }

    #[test]
    fn test_get_server_for_ext() {
        let config = LspConfig::default();
        let (lang, _) = config.get_server_for_ext(".rs").expect("should find rust");
        assert_eq!(lang, "rust");

        let (lang, _) = config
            .get_server_for_ext(".py")
            .expect("should find python");
        assert_eq!(lang, "python");

        assert!(config.get_server_for_ext(".unknown").is_none());
    }

    #[test]
    fn test_from_toml() {
        let toml = r#"
enabled = true
request_timeout_secs = 10

[servers.rust]
command = "rust-analyzer"
args = []
extensions = [".rs"]
root_patterns = ["Cargo.toml"]
enabled = true
"#;
        let config = LspConfig::from_toml(toml).expect("should parse");
        assert!(config.enabled);
        assert_eq!(config.request_timeout_secs, 10);
        assert!(config.servers.contains_key("rust"));
    }
}
