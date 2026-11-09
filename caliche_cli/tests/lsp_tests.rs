//! LSP configuration and language detection tests.

use caliche_cli::lsp::{LspConfig, LspManager};
use std::path::Path;

#[test]
fn lsp_config_default_includes_common_languages() {
    let config = LspConfig::default();
    assert!(!config.enabled);
    assert!(config.servers.contains_key("rust"));
    assert!(config.servers.contains_key("typescript"));
    assert!(config.servers.contains_key("python"));
}

#[test]
fn registry_detects_language_from_path() {
    let manager = LspManager::new(LspConfig::default());
    let registry = manager.registry();

    assert_eq!(
        registry.get_language_for_path(Path::new("test.rs")),
        Some("rust")
    );
    assert_eq!(
        registry.get_language_for_path(Path::new("test.ts")),
        Some("typescript")
    );
    assert_eq!(
        registry.get_language_for_path(Path::new("test.py")),
        Some("python")
    );
    assert_eq!(
        registry.get_language_for_path(Path::new("test.unknown")),
        None
    );
}
