//! CLI storage module for SQLite-backed configuration and session persistence

use anyhow::{Context, Result};
use std::path::PathBuf;

pub mod credentials;
mod credentials_env;
pub mod database;

pub mod env_scanner;
pub mod history;
#[cfg(target_os = "macos")]
pub(crate) mod keychain_touchid;
pub mod schema;
pub mod sessions;

pub use credentials::{
    KeychainCredentialProvider, ScopeInfo, ScopeRecord, ScopeSourceEntry, ScopeStatus,
    SecureCredentialManager, compute_scope_hash, keyring_prompts_allowed,
    session_credential_provider,
};
pub use credentials_env::{record_shell_env_keys, shell_env_keys};
pub use database::Database;
pub use env_scanner::DetectedCredential;
pub use history::HistoryStore;
pub use sessions::SessionStore;

pub(crate) fn cli_home_dir() -> Result<PathBuf> {
    if let Some(home) = std::env::var_os("HOME")
        && !home.is_empty()
    {
        return Ok(PathBuf::from(home));
    }
    dirs::home_dir().context("Could not determine home directory")
}
