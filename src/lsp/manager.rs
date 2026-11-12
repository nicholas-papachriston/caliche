//! LSP manager for coordinating multiple language servers.

use crate::lsp::client::{LspClient, file_uri};
use crate::lsp::config::LspConfig;
use crate::lsp::registry::LspServerRegistry;
use crate::lsp::types::{
    Diagnostic, FileLocation, HoverInfo, Location, Position, SymbolInfo, SymbolKind,
};
use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::time::{Duration, Instant};
use tracing::debug;

/// Workspace key for identifying a unique workspace + language combination.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct WorkspaceKey {
    workspace: PathBuf,
    language: String,
}

struct DocumentState {
    language: String,
    version: AtomicI32,
    opened: AtomicBool,
}

#[derive(Debug, Clone)]
struct CachedDiagnostics {
    diagnostics: Vec<Diagnostic>,
    updated_at: Instant,
}

/// LSP manager for coordinating language servers.
pub struct LspManager {
    /// Server registry.
    registry: LspServerRegistry,
    /// Active LSP clients by workspace + language.
    clients: DashMap<WorkspaceKey, Arc<tokio::sync::Mutex<LspClient>>>,
    startup_locks: DashMap<WorkspaceKey, Arc<tokio::sync::Mutex<()>>>,
    documents: DashMap<PathBuf, DocumentState>,
    /// Configuration.
    config: LspConfig,
    /// Diagnostics cache by file path.
    diagnostics_cache: DashMap<PathBuf, CachedDiagnostics>,
}

impl std::fmt::Debug for LspManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LspManager")
            .field("client_count", &self.clients.len())
            .field("diagnostics_cached_files", &self.diagnostics_cache.len())
            .field("enabled", &self.config.enabled)
            .finish()
    }
}

impl LspManager {
    /// Create a new LSP manager.
    pub fn new(config: LspConfig) -> Self {
        let registry = LspServerRegistry::from_config(&config);
        Self {
            registry,
            clients: DashMap::new(),
            startup_locks: DashMap::new(),
            documents: DashMap::new(),
            config,
            diagnostics_cache: DashMap::new(),
        }
    }

    /// Get or start a language server for a workspace.
    pub async fn get_or_start_server(
        &self,
        workspace: &Path,
        language: &str,
    ) -> Result<Arc<tokio::sync::Mutex<LspClient>>> {
        if !self.is_enabled() {
            return Err(anyhow!("LSP is disabled in configuration"));
        }

        let key = WorkspaceKey {
            workspace: workspace.to_path_buf(),
            language: language.to_string(),
        };

        // Check if client already exists
        if let Some(client) = self.clients.get(&key) {
            return Ok(Arc::clone(client.value()));
        }

        let startup_lock = self
            .startup_locks
            .entry(key.clone())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone();
        let _guard = startup_lock.lock().await;

        if let Some(client) = self.clients.get(&key) {
            return Ok(Arc::clone(client.value()));
        }

        // Get server config
        let server_config = self
            .registry
            .get_server(language)
            .ok_or_else(|| anyhow!("No LSP server configured for language: {}", language))?;

        // Start new server
        debug!(
            "Starting LSP server for {}: {}",
            language, server_config.command
        );
        let mut client = LspClient::new(
            server_config,
            workspace.to_path_buf(),
            Duration::from_secs(self.config.request_timeout_secs),
        )
        .context(format!(
            "Failed to start LSP server for language: {}",
            language
        ))?;

        // Initialize
        client.initialize().await.context(format!(
            "Failed to initialize LSP server for language: {}",
            language
        ))?;

        let client = Arc::new(tokio::sync::Mutex::new(client));
        self.clients.insert(key, Arc::clone(&client));

        Ok(client)
    }

    /// Open a document in the language server.
    pub async fn open_document(&self, path: &Path, language: &str, content: &str) -> Result<()> {
        let workspace = self.workspace_root_for(path, language);
        let client_arc = self.get_or_start_server(&workspace, language).await?;
        let mut client = client_arc.lock().await;

        let canonical_path = normalize_path(path)?;
        let uri = file_uri(&canonical_path)?.to_string();
        let state = self
            .documents
            .entry(canonical_path.clone())
            .or_insert_with(|| DocumentState {
                language: language.to_string(),
                version: AtomicI32::new(0),
                opened: AtomicBool::new(false),
            });
        state.version.store(1, Ordering::SeqCst);
        state.opened.store(true, Ordering::SeqCst);
        client
            .notify(
                "textDocument/didOpen",
                serde_json::json!({
                    "textDocument": {
                        "uri": uri,
                        "languageId": language,
                        "version": 1,
                        "text": content,
                    }
                }),
            )
            .await?;

        debug!("Opened document: {}", path.display());
        Ok(())
    }

    /// Change a document in the language server.
    pub async fn change_document(&self, path: &Path, language: &str, content: &str) -> Result<()> {
        let workspace = self.workspace_root_for(path, language);
        let client_arc = self.get_or_start_server(&workspace, language).await?;
        let mut client = client_arc.lock().await;

        let canonical_path = normalize_path(path)?;
        let uri = file_uri(&canonical_path)?.to_string();
        let version = self.next_document_version(&canonical_path, language);
        client
            .notify(
                "textDocument/didChange",
                serde_json::json!({
                    "textDocument": {
                        "uri": uri,
                        "version": version,
                    },
                    "contentChanges": [
                        {
                            "text": content,
                        }
                    ]
                }),
            )
            .await?;

        // Wait for diagnostics to settle
        self.collect_notifications(&mut client).await;

        debug!("Changed document: {}", path.display());
        Ok(())
    }

    /// Save a document in the language server.
    pub async fn save_document(&self, path: &Path, language: &str) -> Result<Vec<Diagnostic>> {
        let workspace = self.workspace_root_for(path, language);
        let client_arc = self.get_or_start_server(&workspace, language).await?;
        let mut client = client_arc.lock().await;

        let canonical_path = normalize_path(path)?;
        let uri = file_uri(&canonical_path)?.to_string();
        client
            .notify(
                "textDocument/didSave",
                serde_json::json!({
                    "textDocument": {
                        "uri": uri,
                    }
                }),
            )
            .await?;

        // Wait for diagnostics
        self.collect_notifications(&mut client).await;

        // Return cached diagnostics if available
        if let Some(diags) = self.diagnostics_cache.get(&canonical_path) {
            Ok(diags.diagnostics.clone())
        } else {
            Ok(vec![])
        }
    }

    /// Open a document and return diagnostics after the configured settle window.
    pub async fn diagnostics_for_path(&self, path: &Path) -> Result<Vec<Diagnostic>> {
        let language = self
            .registry
            .get_language_for_path(path)
            .ok_or_else(|| anyhow!("No LSP server configured for file: {}", path.display()))?
            .to_string();
        let content = read_document(path).await?;
        self.open_document(path, &language, &content).await?;
        self.save_document(path, &language).await
    }

    /// Request hover information for a file position.
    pub async fn hover(&self, path: &Path, position: Position) -> Result<Option<HoverInfo>> {
        let language = self
            .registry
            .get_language_for_path(path)
            .ok_or_else(|| anyhow!("No LSP server configured for file: {}", path.display()))?
            .to_string();
        let content = read_document(path).await?;
        self.open_document(path, &language, &content).await?;
        let workspace = self.workspace_root_for(path, &language);
        let client_arc = self.get_or_start_server(&workspace, &language).await?;
        let mut client = client_arc.lock().await;
        let result = client
            .request(
                "textDocument/hover",
                json!({
                    "textDocument": { "uri": file_uri(&normalize_path(path)?)?.to_string() },
                    "position": position,
                }),
            )
            .await?;
        self.collect_notifications(&mut client).await;
        parse_hover(result)
    }

    /// Query workspace symbols.
    pub async fn workspace_symbols(
        &self,
        query: &str,
        scope: Option<&Path>,
    ) -> Result<Vec<SymbolInfo>> {
        let base = scope.unwrap_or_else(|| Path::new("."));
        let language = if base.is_file() {
            self.registry
                .get_language_for_path(base)
                .ok_or_else(|| anyhow!("No LSP server configured for file: {}", base.display()))?
                .to_string()
        } else {
            self.registry
                .servers()
                .keys()
                .next()
                .ok_or_else(|| anyhow!("No LSP servers configured"))?
                .to_string()
        };
        let workspace = if base.is_file() {
            self.workspace_root_for(base, &language)
        } else {
            base.to_path_buf()
        };
        let client_arc = self.get_or_start_server(&workspace, &language).await?;
        let mut client = client_arc.lock().await;
        let result = client
            .request("workspace/symbol", json!({ "query": query }))
            .await?;
        self.collect_notifications(&mut client).await;
        parse_symbols(result)
    }

    /// Get cached diagnostics for a file.
    pub fn get_diagnostics(&self, path: &Path) -> Vec<Diagnostic> {
        self.diagnostics_cache
            .get(path)
            .map(|diags| diags.diagnostics.clone())
            .unwrap_or_default()
    }

    /// Update cached diagnostics for a file.
    pub fn set_diagnostics(&self, path: PathBuf, diagnostics: Vec<Diagnostic>) {
        // Enforce max diagnostics per file
        let diags = if diagnostics.len() > self.config.max_diagnostics_per_file {
            diagnostics[..self.config.max_diagnostics_per_file].to_vec()
        } else {
            diagnostics
        };

        self.diagnostics_cache.insert(
            path,
            CachedDiagnostics {
                diagnostics: diags,
                updated_at: Instant::now(),
            },
        );

        // Enforce max cached files (simple LRU-ish eviction)
        if self.diagnostics_cache.len() > self.config.max_cached_files
            && let Some(oldest) = self
                .diagnostics_cache
                .iter()
                .min_by_key(|entry| entry.value().updated_at)
                .map(|entry| entry.key().clone())
        {
            self.diagnostics_cache.remove(&oldest);
        }
    }

    /// Clear all cached diagnostics.
    pub fn clear_diagnostics(&self) {
        self.diagnostics_cache.clear();
    }

    /// Check if LSP is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get configuration.
    pub fn config(&self) -> &LspConfig {
        &self.config
    }

    /// Get registry.
    pub fn registry(&self) -> &LspServerRegistry {
        &self.registry
    }

    pub fn supported_languages(&self) -> Vec<String> {
        self.registry.servers().keys().cloned().collect()
    }

    pub fn is_server_available(&self, language: &str) -> bool {
        self.registry
            .get_server(language)
            .map(|server| command_available(&server.command))
            .unwrap_or(false)
    }

    fn workspace_root_for(&self, path: &Path, language: &str) -> PathBuf {
        let Some(config) = self.registry.get_server(language) else {
            return path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
        };
        let mut current = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        };
        loop {
            if config
                .root_patterns
                .iter()
                .any(|pattern| current.join(pattern).exists())
            {
                return current;
            }
            if !current.pop() {
                return path
                    .parent()
                    .unwrap_or_else(|| Path::new("."))
                    .to_path_buf();
            }
        }
    }

    fn next_document_version(&self, path: &Path, language: &str) -> i32 {
        let entry = self
            .documents
            .entry(path.to_path_buf())
            .or_insert_with(|| DocumentState {
                language: language.to_string(),
                version: AtomicI32::new(1),
                opened: AtomicBool::new(true),
            });
        let _ = &entry.language;
        entry.version.fetch_add(1, Ordering::SeqCst) + 1
    }

    async fn collect_notifications(&self, client: &mut LspClient) {
        let deadline = Instant::now() + Duration::from_millis(self.config.settle_time_ms);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let Some(notification) = client.recv_notification(remaining).await else {
                break;
            };
            if notification.method == "textDocument/publishDiagnostics" {
                self.handle_publish_diagnostics(notification.params);
            }
        }
    }

    fn handle_publish_diagnostics(&self, params: Value) {
        let Some(uri) = params.get("uri").and_then(Value::as_str) else {
            return;
        };
        let Ok(path) = path_from_uri(uri) else {
            return;
        };
        let diagnostics = params
            .get("diagnostics")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|value| serde_json::from_value::<Diagnostic>(value.clone()).ok())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        self.set_diagnostics(path, diagnostics);
    }
}

impl Default for LspManager {
    fn default() -> Self {
        Self::new(LspConfig::default())
    }
}

/// The text of `path` for the language server: resolved inside the
/// workspace as a tool path is, and read following no link, so a link
/// swapped in after the policy gate decided is not read.
async fn read_document(path: &Path) -> Result<String> {
    use crate::agent::tools::common::path::{read_resolved_text_async, validate_and_resolve_path};
    let resolved = validate_and_resolve_path(&path.to_string_lossy(), Some(true))
        .await
        .with_context(|| format!("Failed to read file for LSP: {}", path.display()))?;
    read_resolved_text_async(&resolved)
        .await
        .with_context(|| format!("Failed to read file for LSP: {}", path.display()))
}

fn normalize_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn path_from_uri(uri: &str) -> Result<PathBuf> {
    let url = url::Url::parse(uri).context("invalid LSP file URI")?;
    url.to_file_path()
        .map_err(|()| anyhow!("failed to convert LSP URI to path: {}", uri))
}

fn parse_hover(result: Value) -> Result<Option<HoverInfo>> {
    if result.is_null() {
        return Ok(None);
    }
    let contents = result
        .get("contents")
        .map(hover_contents_to_string)
        .unwrap_or_default();
    let range = result
        .get("range")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .context("failed to parse hover range")?;
    Ok(Some(HoverInfo { contents, range }))
}

fn hover_contents_to_string(value: &Value) -> String {
    if let Some(text) = value.as_str() {
        return text.to_string();
    }
    if let Some(text) = value
        .as_object()
        .and_then(|object| object.get("value"))
        .and_then(Value::as_str)
    {
        return text.to_string();
    }
    if let Some(items) = value.as_array() {
        return items
            .iter()
            .map(hover_contents_to_string)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
    }
    value.to_string()
}

fn parse_symbols(result: Value) -> Result<Vec<SymbolInfo>> {
    let Some(items) = result.as_array() else {
        return Ok(vec![]);
    };
    items
        .iter()
        .map(|item| {
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let kind = item
                .get("kind")
                .and_then(Value::as_u64)
                .map(SymbolKind::from_lsp_value)
                .unwrap_or(SymbolKind::Variable);
            let location = serde_json::from_value::<Location>(
                item.get("location").cloned().unwrap_or(Value::Null),
            )
            .context("failed to parse symbol location")?;
            let container_name = item
                .get("containerName")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            Ok(SymbolInfo {
                name,
                kind,
                location,
                container_name,
            })
        })
        .collect()
}

pub fn location_to_file_location(location: &Location) -> Result<FileLocation> {
    let file = path_from_uri(&location.uri)?;
    Ok(FileLocation {
        file,
        line: location.range.start.line + 1,
        column: location.range.start.character + 1,
        end_line: Some(location.range.end.line + 1),
        end_column: Some(location.range.end.character + 1),
    })
}

fn command_available(command: &str) -> bool {
    if command.contains(std::path::MAIN_SEPARATOR) {
        Path::new(command).exists()
    } else {
        std::env::var_os("PATH")
            .and_then(|paths| {
                std::env::split_paths(&paths)
                    .map(|path| path.join(command))
                    .find(|candidate| candidate.exists())
            })
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_key() {
        let key1 = WorkspaceKey {
            workspace: PathBuf::from("/home/user/project"),
            language: "rust".to_string(),
        };
        let key2 = WorkspaceKey {
            workspace: PathBuf::from("/home/user/project"),
            language: "rust".to_string(),
        };
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_lsp_manager_new() {
        let manager = LspManager::default();
        assert!(!manager.is_enabled());
        assert!(!manager.get_diagnostics(Path::new("test.rs")).is_empty() == false);
    }

    #[test]
    fn test_set_and_get_diagnostics() {
        let manager = LspManager::default();
        let path = PathBuf::from("test.rs");
        let diags = vec![];
        manager.set_diagnostics(path.clone(), diags);
        assert!(manager.get_diagnostics(&path).is_empty());
    }
}
