//! Where Caliche stores per-user files.
//!
//! [`CONFIG_DIR_ENV`], when set to a non-empty path, is that directory.
//! `server.lock`, `servers/`, `config.toml`, `mcp.json`, `plugins.db`, and the
//! API task ledger then live under it. When it is unset, locks and config stay
//! in the platform config directory and the plugin store stays in the data
//! directory.
//!
//! CLI logs, `endpoints.json`, `gateway-cluster.db`, the CLI session database,
//! and the CLI task ledger keep their own roots. `caliche --server` writes
//! `tasks.db` under `~/.caliche` from `HOME`.

use std::path::PathBuf;

/// Override for the `caliche` config directory.
///
/// This is the directory itself, not its parent. On Windows `dirs` ignores
/// `XDG_CONFIG_HOME`, so tests and operators set this to keep records,
/// `server.lock`, and `plugins.db` out of the real profile.
pub const CONFIG_DIR_ENV: &str = "CALICHE_CONFIG_DIR";

/// `CALICHE_CONFIG_DIR` when it is set and not empty.
#[must_use]
pub fn config_dir_override() -> Option<PathBuf> {
    let value = std::env::var_os(CONFIG_DIR_ENV)?;
    if value.is_empty() {
        return None;
    }
    Some(PathBuf::from(value))
}

/// `server.lock` beside the registry.
#[must_use]
pub fn server_lock_path() -> PathBuf {
    if let Some(dir) = config_dir_override() {
        return dir.join("server.lock");
    }
    config_root().join("caliche").join("server.lock")
}

/// User `config.toml`.
#[must_use]
pub fn user_config_path() -> PathBuf {
    if let Some(dir) = config_dir_override() {
        return dir.join("config.toml");
    }
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("caliche")
        .join("config.toml")
}

/// MCP server list. Durable settings sit beside this file.
#[must_use]
pub fn mcp_config_path() -> PathBuf {
    if let Some(dir) = config_dir_override() {
        return dir.join("mcp.json");
    }
    dirs::config_dir()
        .map(|dir| dir.join("caliche").join("mcp.json"))
        .unwrap_or_else(|| PathBuf::from(".").join(".caliche_mcp.json"))
}

/// SQLite plugin store.
#[must_use]
pub fn plugin_store_path() -> PathBuf {
    if let Some(dir) = config_dir_override() {
        return dir.join("plugins.db");
    }
    dirs::data_dir()
        .map(|dir| dir.join("caliche").join("plugins.db"))
        .unwrap_or_else(|| PathBuf::from(".").join("caliche_plugins.db"))
}

/// Directory for `tasks.db` opened by the API server.
///
/// The override is the directory itself. Otherwise this is the platform data
/// directory plus `caliche`.
#[must_use]
pub fn task_ledger_dir() -> Option<PathBuf> {
    if let Some(dir) = config_dir_override() {
        return Some(dir);
    }
    dirs::data_dir().map(|dir| dir.join("caliche"))
}

fn config_root() -> PathBuf {
    dirs::config_dir().unwrap_or_else(home_config_fallback)
}

fn home_config_fallback() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".config")
}
