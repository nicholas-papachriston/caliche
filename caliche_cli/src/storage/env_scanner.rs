//! Environment file scanner for auto-detecting provider credentials.
//!
//! Scans `.env` files in the current directory (and optionally parent directories)
//! for known LLM provider API keys and returns them for optional keychain ingestion.

use crate::credentials::known_providers;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// A credential detected from a `.env` file on disk.
#[derive(Debug, Clone)]
pub struct DetectedCredential {
    /// The environment variable name (e.g. `OPENAI_API_KEY`)
    pub env_var: String,
    /// The raw value from the `.env` file
    pub value: String,
    /// The `.env` file path where this was found
    pub source_path: PathBuf,
    /// The logical provider name (e.g. `openai`)
    pub provider_name: String,
}

/// File names that are scanned for credentials (in preference order per directory).
const ENV_FILE_NAMES: &[&str] = &[
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.staging",
];

/// Build the set of known env var names -> provider name from [`known_providers`](crate::credentials::known_providers).
fn build_provider_env_map() -> HashMap<&'static str, &'static str> {
    known_providers()
        .iter()
        .filter(|(_, env_var, _)| !env_var.is_empty())
        .map(|(name, env_var, _)| (*env_var, *name))
        .collect()
}

/// Parse a single `.env` file and return all key-value pairs that match known provider env vars.
///
/// Uses `dotenvy`'s iterator to parse without modifying the process environment.
fn parse_env_file(path: &Path, env_map: &HashMap<&str, &str>) -> Vec<DetectedCredential> {
    let iter = match dotenvy::from_filename_iter(path) {
        Ok(it) => it,
        Err(_) => return Vec::new(),
    };

    iter.filter_map(|item| {
        let (key, value) = item.ok()?;
        let provider_name = env_map.get(key.as_str())?;
        if value.trim().is_empty() {
            return None;
        }
        Some(DetectedCredential {
            env_var: key,
            value,
            source_path: path.to_path_buf(),
            provider_name: provider_name.to_string(),
        })
    })
    .collect()
}

/// Scan a directory for `.env` files and return detected provider credentials.
///
/// If `check_parents` is true, also walks up the directory tree until the filesystem
/// root, collecting credentials from each level. Credentials found in closer directories
/// take priority (duplicates from parent dirs are ignored).
pub fn scan_for_credentials(dir: &Path, check_parents: bool) -> Vec<DetectedCredential> {
    let env_map = build_provider_env_map();

    // Track which env vars we have already found (closer dir wins over parent)
    let mut seen_vars: HashMap<String, DetectedCredential> = HashMap::new();

    let dirs_to_scan: Vec<PathBuf> = if check_parents {
        let mut dirs = Vec::new();
        let mut current = Some(dir.to_path_buf());
        while let Some(d) = current {
            dirs.push(d.clone());
            current = d.parent().map(|p| p.to_path_buf());
        }
        dirs
    } else {
        vec![dir.to_path_buf()]
    };

    for dir_path in dirs_to_scan {
        for file_name in ENV_FILE_NAMES {
            let env_path = dir_path.join(file_name);
            if !env_path.exists() {
                continue;
            }

            let found = parse_env_file(&env_path, &env_map);
            for cred in found {
                // Only keep the first (closest) occurrence of each env var
                seen_vars.entry(cred.env_var.clone()).or_insert(cred);
            }
        }
    }

    seen_vars.into_values().collect()
}

/// Filter detected credentials to only those not already present in the current process
/// environment. This avoids re-prompting for credentials the user already has set.
pub fn filter_new_credentials(detected: Vec<DetectedCredential>) -> Vec<DetectedCredential> {
    detected
        .into_iter()
        .filter(|c| caliche_contracts::credential_env::var(&c.env_var).is_err())
        .collect()
}
