//! Secure credential management using system keyring with directory-scoped credential vaults.
//!
//! # Architecture
//!
//! Credentials are stored in the OS keychain under a single entry so the OS shows at most
//! one access prompt per run (e.g. "Always Allow" once per install).
//!
//! - `caliche` / `credentials:data`  -- single entry holding all scopes (global + per-directory)
//!
//! Per-scope `<hash>` is a deterministic 16-character hex string derived from the canonical
//! directory path (used as a key inside the unified blob, not as a separate keyring username).
//!
//! ## Resolution order (highest → lowest priority)
//!
//! 1. Process environment variables (already in the shell)
//! 2. CWD-scoped keychain credentials
//! 3. Linked scope credentials (other directories whose creds have been shared)
//! 4. Global keychain credentials
//! 5. Database defaults (default provider / model)

use crate::credentials::{GATEWAY_CREDENTIALS, known_providers, normalize_provider_name};
use anyhow::{Context, Result};
use caliche::persistence::{CredentialMetadata, CredentialStoreBackend, SqliteCredentialStore};
use chrono::Utc;
use sqlx::SqlitePool;
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use std::collections::HashMap;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, OnceLock};

/// Whether the OS keyring may show an access prompt.
///
/// Off-TTY runs (CI, agents, pipes) skip the keyring so startup cannot hang
/// waiting for a GUI prompt that never appears. Set `CALICHE_NO_KEYRING=1`
/// to force skip, or `CALICHE_USE_KEYRING=1` to force enable.
#[must_use]
pub fn keyring_prompts_allowed() -> bool {
    match std::env::var("CALICHE_NO_KEYRING") {
        Ok(value) if value != "0" && !value.is_empty() => return false,
        _ => {}
    }
    match std::env::var("CALICHE_USE_KEYRING") {
        Ok(value) if value == "1" || value.eq_ignore_ascii_case("true") => return true,
        Ok(value) if value == "0" || value.eq_ignore_ascii_case("false") => return false,
        _ => {}
    }
    std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

const SERVICE_NAME: &str = "caliche";
/// Single keyring entry for all scopes so the OS only prompts once (e.g. "Always Allow" once)
const UNIFIED_ENTRY: &str = "credentials:data";

/// True when credentials should use macOS data-protection keychain (Touch ID / passcode).
/// On macOS this is the default; set CALICHE_KEYCHAIN_USE_TOUCH_ID=false to use the regular keychain.
#[cfg(all(target_os = "macos", not(test)))]
fn use_touch_id_keychain() -> bool {
    std::env::var("CALICHE_KEYCHAIN_USE_TOUCH_ID").as_deref() != Ok("false")
}

/// Unit tests use the in-memory mock store only, never the OS keychain.
#[cfg(any(not(target_os = "macos"), test))]
fn use_touch_id_keychain() -> bool {
    false
}

static KEYRING_STORE_INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Serializes unified keyring blob read-modify-write across per-command manager instances.
static KEYRING_BLOB_LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();

fn keyring_blob_lock() -> &'static std::sync::Mutex<()> {
    KEYRING_BLOB_LOCK.get_or_init(|| std::sync::Mutex::new(()))
}

fn keyring_entry(service: &str, user: &str) -> Result<keyring_core::Entry> {
    ensure_default_keyring_store()?;
    keyring_core::Entry::new(service, user).context("Failed to create keyring entry")
}

fn ensure_default_keyring_store() -> Result<()> {
    match KEYRING_STORE_INIT
        .get_or_init(|| install_default_keyring_store().map_err(|err| err.to_string()))
    {
        Ok(()) => Ok(()),
        Err(err) => Err(anyhow::anyhow!(
            "Failed to initialize platform keyring store: {err}"
        )),
    }
}

/// Unit tests keep credentials in memory so they never read or write the OS keychain.
#[cfg(test)]
fn install_default_keyring_store() -> Result<()> {
    keyring_core::set_default_store(keyring_core::mock::Store::new()?);
    Ok(())
}

#[cfg(all(target_os = "macos", not(test)))]
fn install_default_keyring_store() -> Result<()> {
    keyring_core::set_default_store(apple_native_keyring_store::keychain::Store::new()?);
    Ok(())
}

#[cfg(all(windows, not(test)))]
fn install_default_keyring_store() -> Result<()> {
    keyring_core::set_default_store(windows_native_keyring_store::store::Store::new()?);
    Ok(())
}

#[cfg(all(target_os = "linux", not(test)))]
fn install_default_keyring_store() -> Result<()> {
    keyring_core::set_default_store(linux_keyutils_keyring_store::Store::new()?);
    Ok(())
}

#[cfg(all(
    not(target_os = "macos"),
    not(windows),
    not(target_os = "linux"),
    not(test)
))]
fn install_default_keyring_store() -> Result<()> {
    Err(anyhow::anyhow!(
        "No keyring-core credential store is configured for this platform"
    ))
}

// ---------------------------------------------------------------------------
// Scope helpers
// ---------------------------------------------------------------------------

/// Information about a credential scope (a directory whose credentials are stored).
#[derive(Debug, Clone)]
pub struct ScopeInfo {
    /// Canonical filesystem path of the scope directory
    pub path: String,
    /// 16-char deterministic hex hash of `path` used as the keyring entry suffix
    pub hash: String,
}

impl ScopeInfo {
    /// Derive scope info from a filesystem path.
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        let canonical = path
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| path.as_ref().to_path_buf());
        let path_str = canonical.to_string_lossy().to_string();
        let hash = compute_scope_hash(&path_str);
        Self {
            path: path_str,
            hash,
        }
    }
}

/// Compute a deterministic 16-char hex hash for a path string.
///
/// Uses djb2 polynomial hashing -- not cryptographic but stable across platforms
/// and Rust versions since it relies only on primitive arithmetic.
pub fn compute_scope_hash(path: &str) -> String {
    let mut h: u64 = 5381;
    for b in path.bytes() {
        h = h.wrapping_mul(33).wrapping_add(u64::from(b));
    }
    format!("{h:016x}")
}

// ---------------------------------------------------------------------------
// SecureCredentialManager
// ---------------------------------------------------------------------------

/// Secure credential manager that stores API keys in the OS keychain.
///
/// Credentials are organized into *scopes*:
/// - **global** -- available from any directory
/// - **directory-scoped** -- available only when `caliche` is run from that directory
///   (or from a directory that has linked to it)
pub struct SecureCredentialManager {
    credential_store: Arc<SqliteCredentialStore>,
    /// Pool used for scope metadata tables (`credential_scopes`, `credential_scope_links`)
    pool: SqlitePool,
    /// Active scope derived from the CWD at construction time; `None` means global-only
    current_scope: Option<ScopeInfo>,
    /// Cache: scope_hash -> provider -> api_key. Filled once from the single keyring entry.
    cache: std::sync::RwLock<HashMap<String, HashMap<String, String>>>,
    /// True after we have read the unified keyring entry (so we only prompt once per run).
    unified_loaded: std::sync::RwLock<bool>,
    /// Cached `list_configured` result to avoid repeated DB/keyring round-trips on startup
    configured_cache: std::sync::RwLock<Option<Vec<crate::credentials::ProviderStatus>>>,
}

impl SecureCredentialManager {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Create a new manager with the CWD detected automatically as the scope.
    pub async fn new() -> Result<Self> {
        let cwd = std::env::current_dir().ok();
        let scope = cwd.map(ScopeInfo::from_path);
        Self::new_internal(scope).await
    }

    /// Create a manager with a specific scope directory.
    pub async fn new_with_scope(dir: Option<&Path>) -> Result<Self> {
        let scope = dir.map(ScopeInfo::from_path);
        Self::new_internal(scope).await
    }

    async fn new_internal(current_scope: Option<ScopeInfo>) -> Result<Self> {
        let db_path = Self::default_db_path()?;
        crate::storage::schema::ensure_schema(&db_path).await?;

        let credential_store = SqliteCredentialStore::from_path(&db_path).await?;
        let pool = Self::open_pool(&db_path).await?;
        Self::ensure_scope_tables(&pool).await?;

        Ok(Self {
            credential_store: Arc::new(credential_store),
            pool,
            current_scope,
            cache: std::sync::RwLock::new(HashMap::new()),
            unified_loaded: std::sync::RwLock::new(false),
            configured_cache: std::sync::RwLock::new(None),
        })
    }

    // ------------------------------------------------------------------
    // DB / pool helpers
    // ------------------------------------------------------------------

    fn default_db_path() -> Result<PathBuf> {
        let home = crate::storage::cli_home_dir()?;
        Ok(home.join(".caliche").join("caliche.db"))
    }

    async fn open_pool(db_path: &Path) -> Result<SqlitePool> {
        let url = format!("sqlite:{}", db_path.display());
        let opts = SqliteConnectOptions::from_str(&url)?.create_if_missing(true);
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(opts)
            .await
            .context("Failed to open scope pool")
    }

    /// Idempotent: create scope tables if they don't exist yet.
    async fn ensure_scope_tables(pool: &SqlitePool) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS credential_scopes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_path TEXT NOT NULL,
                scope_hash TEXT NOT NULL UNIQUE,
                label TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )",
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS credential_scope_links (
                source_hash TEXT NOT NULL,
                target_hash TEXT NOT NULL,
                provider TEXT NOT NULL DEFAULT '*',
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_hash, target_hash, provider)
            )",
        )
        .execute(pool)
        .await?;

        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_scope_links_target ON credential_scope_links(target_hash)",
        )
        .execute(pool)
        .await
        .ok(); // index creation is best-effort

        Ok(())
    }

    // ------------------------------------------------------------------
    // Scope helpers
    // ------------------------------------------------------------------

    /// The hash of the currently active scope, or `"global"` if none.
    pub fn current_scope_hash(&self) -> &str {
        self.current_scope
            .as_ref()
            .map(|s| s.hash.as_str())
            .unwrap_or("global")
    }

    /// A reference to the current scope, if any.
    pub fn current_scope(&self) -> Option<&ScopeInfo> {
        self.current_scope.as_ref()
    }

    // ------------------------------------------------------------------
    // Keyring I/O (single entry = one keychain prompt per run)
    // ------------------------------------------------------------------

    fn read_blob_from_keyring_unlocked() -> HashMap<String, HashMap<String, String>> {
        if !keyring_prompts_allowed() {
            return HashMap::new();
        }
        let json_opt: Option<String> = if use_touch_id_keychain() {
            #[cfg(target_os = "macos")]
            {
                crate::storage::keychain_touchid::get(SERVICE_NAME, UNIFIED_ENTRY)
                    .ok()
                    .flatten()
                    .or_else(|| {
                        keyring_entry(SERVICE_NAME, UNIFIED_ENTRY)
                            .ok()
                            .and_then(|entry| entry.get_password().ok())
                    })
            }
            #[cfg(not(target_os = "macos"))]
            None
        } else if let Ok(entry) = keyring_entry(SERVICE_NAME, UNIFIED_ENTRY) {
            entry.get_password().ok()
        } else {
            None
        };
        if let Some(json) = json_opt {
            if let Ok(blob) =
                serde_json::from_str::<HashMap<String, HashMap<String, String>>>(&json)
            {
                return blob;
            }
        }
        HashMap::new()
    }

    fn write_blob_to_keyring_unlocked(
        blob: &HashMap<String, HashMap<String, String>>,
    ) -> Result<()> {
        if !keyring_prompts_allowed() {
            anyhow::bail!("Keyring writes are disabled for non-interactive sessions");
        }
        let json = serde_json::to_string(blob).context("Failed to serialize credentials")?;
        let entry = keyring_entry(SERVICE_NAME, UNIFIED_ENTRY)?;

        if use_touch_id_keychain() {
            #[cfg(target_os = "macos")]
            {
                if let Err(e) =
                    crate::storage::keychain_touchid::set(SERVICE_NAME, UNIFIED_ENTRY, &json)
                {
                    tracing::warn!(
                        "Touch ID keychain unavailable ({}), using standard keychain",
                        e
                    );
                    entry
                        .set_password(&json)
                        .context("Failed to store credentials in keyring")?;
                }
                return Ok(());
            }
        }

        entry
            .set_password(&json)
            .context("Failed to store credentials in keyring")?;
        Ok(())
    }

    /// Hold one keyring lock across load → mutate → save to avoid RMW races.
    fn with_blob_mut<R>(
        &self,
        mutate: impl FnOnce(&mut HashMap<String, HashMap<String, String>>) -> Result<R>,
    ) -> Result<R> {
        let _guard = match keyring_blob_lock().lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let mut blob = Self::read_blob_from_keyring_unlocked();
        let result = mutate(&mut blob)?;
        Self::write_blob_to_keyring_unlocked(&blob)?;
        if let Ok(mut guard) = self.cache.write() {
            *guard = blob;
        }
        if let Ok(mut loaded) = self.unified_loaded.write() {
            *loaded = true;
        }
        Ok(result)
    }

    /// Ensure the unified keyring entry has been read once and the cache is populated.
    ///
    /// This is a read. It does not write the blob back. A write on every launch
    /// asks the OS for a second keychain prompt before the prompt or the command.
    fn ensure_unified_loaded(&self) {
        if self.unified_loaded.read().map(|g| *g).unwrap_or(false) {
            return;
        }
        let _guard = match keyring_blob_lock().lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if self.unified_loaded.read().map(|g| *g).unwrap_or(false) {
            return;
        }
        let blob = Self::read_blob_from_keyring_unlocked();
        if let Ok(mut guard) = self.cache.write() {
            *guard = blob;
        }
        if let Ok(mut loaded) = self.unified_loaded.write() {
            *loaded = true;
        }
    }

    /// Load credentials for a specific scope hash from the keyring.
    /// Uses the single unified entry so we only touch the keychain once.
    fn load_scope_from_keyring(&self, scope_hash: &str) -> HashMap<String, String> {
        self.ensure_unified_loaded();
        if let Ok(guard) = self.cache.read() {
            if let Some(creds) = guard.get(scope_hash) {
                return creds.clone();
            }
        }
        HashMap::new()
    }

    /// Save credentials for a specific scope hash to the keyring.
    fn save_scope_to_keyring(
        &self,
        scope_hash: &str,
        creds: &HashMap<String, String>,
    ) -> Result<()> {
        self.ensure_unified_loaded();
        self.with_blob_mut(|blob| {
            if creds.is_empty() {
                blob.remove(scope_hash);
            } else {
                blob.insert(scope_hash.to_string(), creds.clone());
            }
            Ok(())
        })
    }

    /// Load credentials from the keyring cache, populating it on first access.
    pub(super) fn get_cached_scope(&self, scope_hash: &str) -> HashMap<String, String> {
        // Fast path: already cached
        if let Ok(guard) = self.cache.read()
            && let Some(creds) = guard.get(scope_hash)
        {
            return creds.clone();
        }
        // Slow path: load from keyring
        let creds = self.load_scope_from_keyring(scope_hash);
        if let Ok(mut guard) = self.cache.write() {
            guard.insert(scope_hash.to_string(), creds.clone());
        }
        creds
    }

    /// Update the in-memory cache for a scope and persist to keyring.
    fn update_scope(&self, scope_hash: &str, creds: HashMap<String, String>) -> Result<()> {
        self.save_scope_to_keyring(scope_hash, &creds)
    }

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Set an API key, storing it in the **global** scope.
    ///
    /// Explicit `config set` commands always write globally so they are
    /// reachable from any directory.
    pub async fn set_api_key(&self, provider: &str, api_key: &str) -> Result<()> {
        self.set_api_key_in_scope(provider, api_key, "global").await
    }

    /// Set an API key in a specific scope.
    pub async fn set_api_key_in_scope(
        &self,
        provider: &str,
        api_key: &str,
        scope_hash: &str,
    ) -> Result<()> {
        let provider = normalize_provider_name(provider);
        let now = Utc::now();

        // Load current creds for this scope
        let mut creds = self.get_cached_scope(scope_hash);
        creds.insert(provider.clone(), api_key.to_string());
        self.update_scope(scope_hash, creds)?;

        // Record metadata in the provider DB (global metadata table)
        let metadata = CredentialMetadata {
            provider: provider.clone(),
            source: "keyring".to_string(),
            is_default: false,
            default_model: None,
            created_at: now,
            updated_at: now,
        };
        if let Err(e) = self.credential_store.set_credential(&metadata).await {
            tracing::warn!("Failed to save credential metadata to database: {e}");
        }

        // Invalidate configured_cache
        if let Ok(mut guard) = self.configured_cache.write() {
            *guard = None;
        }

        super::credentials_env::publish(self, &provider).await;
        Ok(())
    }

    /// Ingest a slice of env-file-detected credentials into the current CWD scope.
    ///
    /// Also registers the scope in the `credential_scopes` table if not present.
    pub async fn ingest_env_credentials(
        &self,
        detected: &[crate::storage::env_scanner::DetectedCredential],
    ) -> Result<()> {
        if detected.is_empty() {
            return Ok(());
        }

        let scope_hash = self.current_scope_hash();

        // Register scope in DB if we have an actual scope (not global)
        if let Some(scope) = &self.current_scope {
            self.upsert_scope_record(&scope.path, &scope.hash, None)
                .await?;
        }

        let mut creds = self.get_cached_scope(scope_hash);
        for det in detected {
            creds.insert(det.provider_name.clone(), det.value.clone());

            // Also record metadata so list_configured() knows about these
            let now = Utc::now();
            let metadata = CredentialMetadata {
                provider: det.provider_name.clone(),
                source: format!("env-file:{scope_hash}"),
                is_default: false,
                default_model: None,
                created_at: now,
                updated_at: now,
            };
            if let Err(e) = self.credential_store.set_credential(&metadata).await {
                tracing::warn!("Failed to save env-file credential metadata: {e}");
            }
        }
        self.update_scope(scope_hash, creds)?;

        // Invalidate configured_cache
        if let Ok(mut guard) = self.configured_cache.write() {
            *guard = None;
        }

        for det in detected {
            super::credentials_env::publish(self, &det.provider_name).await;
        }
        Ok(())
    }

    /// Get an API key for a provider, checking all scopes in priority order.
    pub async fn get_api_key(&self, provider: &str) -> Option<String> {
        let provider_lower = normalize_provider_name(provider);

        if let Some(key) = self.get_api_key_from_env(&provider_lower) {
            return Some(key);
        }

        self.resolve_from_scope_chain(&provider_lower).await
    }

    /// Get API key from environment variables only.
    fn get_api_key_from_env(&self, provider: &str) -> Option<String> {
        if let Some((_, env_var, _)) = known_providers().iter().find(|(p, _, _)| *p == provider)
            && !env_var.is_empty()
            && let Ok(key) = caliche_contracts::credential_env::var(env_var)
        {
            return Some(key);
        }
        if let Some((_, env_var, _)) = GATEWAY_CREDENTIALS.iter().find(|(p, _, _)| *p == provider)
            && let Ok(key) = caliche_contracts::credential_env::var(env_var)
        {
            return Some(key);
        }
        let env_var = format!("{}_API_KEY", provider.to_uppercase());
        caliche_contracts::credential_env::var(&env_var).ok()
    }

    /// Walk the scope resolution chain and return the first matching key.
    ///
    /// Order: CWD scope → linked scopes → global.
    pub(super) async fn resolve_from_scope_chain(&self, provider: &str) -> Option<String> {
        // 1. CWD scope
        if let Some(scope) = &self.current_scope {
            let creds = self.get_cached_scope(&scope.hash);
            if let Some(key) = creds.get(provider) {
                return Some(key.clone());
            }
        }

        // 2. Linked scopes
        let target_hash = self.current_scope_hash().to_string();
        if let Ok(links) = self.get_linked_sources(&target_hash).await {
            for (source_hash, link_provider) in links {
                let applies = link_provider == "*" || link_provider == provider;
                if applies {
                    let creds = self.get_cached_scope(&source_hash);
                    if let Some(key) = creds.get(provider) {
                        return Some(key.clone());
                    }
                }
            }
        }

        // 3. Global
        let global = self.get_cached_scope("global");
        global.get(provider).cloned()
    }

    /// Remove an API key for a provider (removes from all scopes).
    pub async fn remove_api_key(&self, provider: &str) -> Result<bool> {
        let provider = normalize_provider_name(provider);

        if self
            .credential_store
            .get_credential(&provider)
            .await?
            .is_none()
        {
            return Ok(false);
        }

        // Remove from all known scopes' caches and keyring entries
        let scope_hashes = self.all_scope_hashes().await;
        for hash in &scope_hashes {
            let mut creds = self.get_cached_scope(hash);
            if creds.remove(&provider).is_some() {
                self.update_scope(hash, creds)?;
            }
        }

        self.credential_store.delete_credential(&provider).await?;

        // Invalidate configured_cache
        if let Ok(mut guard) = self.configured_cache.write() {
            *guard = None;
        }

        Ok(true)
    }

    /// Collect all known scope hashes (global + all registered directory scopes).
    async fn all_scope_hashes(&self) -> Vec<String> {
        let mut hashes = vec!["global".to_string()];
        let rows: Vec<String> = sqlx::query_scalar("SELECT scope_hash FROM credential_scopes")
            .fetch_all(&self.pool)
            .await
            .unwrap_or_default();
        hashes.extend(rows);
        hashes
    }

    // ------------------------------------------------------------------
    // Default provider / model
    // ------------------------------------------------------------------

    /// Set the default provider.
    pub async fn set_default_provider(&self, provider: &str) -> Result<()> {
        self.credential_store.set_default_provider(provider).await?;
        Ok(())
    }

    /// Get the default provider.
    pub async fn default_provider(&self) -> Option<String> {
        self.stored_default_provider()
            .await
            .or_else(|| std::env::var("AGENT_PROVIDER").ok())
    }

    /// Stored default only. Does not fall back to `AGENT_PROVIDER`.
    pub async fn stored_default_provider(&self) -> Option<String> {
        self.credential_store
            .get_default_provider()
            .await
            .ok()
            .flatten()
            .filter(|value| !value.trim().is_empty())
    }

    /// Set the default model for a provider.
    pub async fn set_default_model(&self, provider: &str, model: &str) -> Result<()> {
        let provider = normalize_provider_name(provider);
        let now = Utc::now();
        let mut metadata = self
            .credential_store
            .get_credential(&provider)
            .await?
            .unwrap_or_else(|| CredentialMetadata {
                provider: provider.clone(),
                source: "env".to_string(),
                is_default: false,
                default_model: None,
                created_at: now,
                updated_at: now,
            });
        metadata.default_model = Some(model.to_string());
        metadata.updated_at = now;
        self.credential_store.set_credential(&metadata).await?;
        Ok(())
    }

    /// Get the default model for a provider.
    pub async fn get_default_model(&self, provider: &str) -> Option<String> {
        self.credential_store
            .get_credential(provider)
            .await
            .ok()
            .flatten()
            .and_then(|m| m.default_model)
    }

    // ------------------------------------------------------------------
    // List configured providers
    // ------------------------------------------------------------------

    /// List all providers and their configuration status (cached).
    pub async fn list_configured(&self) -> Vec<crate::credentials::ProviderStatus> {
        if let Ok(guard) = self.configured_cache.read()
            && let Some(ref cached) = *guard
        {
            return cached.clone();
        }
        let results = self.compute_list_configured().await;
        if let Ok(mut guard) = self.configured_cache.write() {
            *guard = Some(results.clone());
        }
        results
    }

    async fn compute_list_configured(&self) -> Vec<crate::credentials::ProviderStatus> {
        let all_credentials = self
            .credential_store
            .list_credentials()
            .await
            .unwrap_or_default();
        let cred_map: HashMap<String, CredentialMetadata> = all_credentials
            .into_iter()
            .map(|c| (c.provider.clone(), c))
            .collect();
        let default_prov = self.default_provider().await;

        let mut results = Vec::new();
        for (name, env_var, description) in known_providers() {
            // The same lookup that fills the session's credential store: a
            // metadata row without a readable key is not configured.
            let has_config = self.resolve_from_scope_chain(name).await.is_some();
            let has_env_key =
                !env_var.is_empty() && caliche_contracts::credential_env::var(env_var).is_ok();
            let default_model = cred_map.get(*name).and_then(|c| c.default_model.clone());
            let is_default = default_prov.as_deref() == Some(*name);

            results.push(crate::credentials::ProviderStatus {
                name: name.to_string(),
                description: description.to_string(),
                has_config_key: has_config,
                has_env_key,
                is_configured: has_config
                    || has_env_key
                    || *name == "lmstudio"
                    || *name == "ollama"
                    || *name == "local-inference",
                default_model,
                is_default,
                is_running: None,
                available_models: None,
            });
        }
        results
    }

    // ------------------------------------------------------------------
    // Load into the in-process credential store
    // ------------------------------------------------------------------

    /// Load resolved credentials into the in-process credential store
    /// ([`caliche_contracts::credential_env`]) in this priority order:
    ///
    /// 1. Shell env vars (set before `caliche` was launched) — never overridden
    /// 2. CWD-scoped keychain credentials
    /// 3. Linked scope credentials
    /// 4. Global keychain credentials
    /// 5. `--env-file`-sourced vars (in the process env but not in `shell_env_keys`)
    ///    — these are **overrideable** by keychain so the keychain is the true source of truth
    ///
    /// `shell_env_keys` should be the set of env var *names* that were in the process
    /// environment **before** the `--env-file` load ran.  Passing an empty set degrades to
    /// the old "never override anything already set" behaviour and is safe as a default.
    ///
    /// The process environment is never written.
    pub async fn apply_to_credential_store_with_priority(
        &self,
        shell_env_keys: &std::collections::HashSet<String>,
    ) {
        super::credentials_env::apply_to_credential_store_with_priority(self, shell_env_keys).await;
    }

    /// Put what the scope chain resolves now for every provider into the
    /// in-process store, replacing older stored values. Names the shell
    /// exported keep their values.
    pub async fn publish_to_credential_store(&self) {
        super::credentials_env::publish_all(self).await;
    }

    /// Load credentials into the in-process credential store.
    ///
    /// Every name that already has a value, in the process environment or in the
    /// store, keeps it: the keychain fills only names that are still unset.  Use
    /// this only when the pre-launch shell snapshot is not available (e.g., in
    /// config subcommand handlers).
    pub async fn apply_to_credential_store(&self) {
        let mut already_set: std::collections::HashSet<String> =
            std::env::vars().map(|(k, _)| k).collect();
        already_set.extend(
            caliche_contracts::credential_env::snapshot()
                .into_iter()
                .map(|(k, _)| k),
        );
        self.apply_to_credential_store_with_priority(&already_set)
            .await;
    }

    // ------------------------------------------------------------------
    // Scope management
    // ------------------------------------------------------------------

    /// Register or update a scope record in the database.
    pub async fn upsert_scope_record(
        &self,
        scope_path: &str,
        scope_hash: &str,
        label: Option<&str>,
    ) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO credential_scopes (scope_path, scope_hash, label, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?4)
             ON CONFLICT(scope_hash) DO UPDATE SET
                label = COALESCE(?3, label),
                updated_at = ?4",
        )
        .bind(scope_path)
        .bind(scope_hash)
        .bind(label)
        .bind(&now)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Link a source scope's credentials to the current (target) scope.
    ///
    /// `provider` is `"*"` to share all providers, or a specific provider name.
    pub async fn link_scope(&self, source_path: &str, provider: &str) -> Result<()> {
        let source_scope = ScopeInfo::from_path(source_path);
        let target_hash = self.current_scope_hash().to_string();

        // Ensure source scope exists in DB
        self.upsert_scope_record(source_path, &source_scope.hash, None)
            .await?;

        // Ensure target scope exists in DB
        if let Some(scope) = &self.current_scope {
            self.upsert_scope_record(&scope.path, &scope.hash, None)
                .await?;
        }

        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT OR REPLACE INTO credential_scope_links
                (source_hash, target_hash, provider, created_at)
             VALUES (?1, ?2, ?3, ?4)",
        )
        .bind(&source_scope.hash)
        .bind(&target_hash)
        .bind(provider)
        .bind(&now)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Remove a scope link from the current (target) scope.
    pub async fn unlink_scope(&self, source_path: &str, provider: Option<&str>) -> Result<bool> {
        let source_scope = ScopeInfo::from_path(source_path);
        let target_hash = self.current_scope_hash();

        let rows_affected = if let Some(prov) = provider {
            sqlx::query(
                "DELETE FROM credential_scope_links
                 WHERE source_hash = ?1 AND target_hash = ?2 AND provider = ?3",
            )
            .bind(&source_scope.hash)
            .bind(target_hash)
            .bind(prov)
            .execute(&self.pool)
            .await?
            .rows_affected()
        } else {
            // Remove all links from this source to this target
            sqlx::query(
                "DELETE FROM credential_scope_links
                 WHERE source_hash = ?1 AND target_hash = ?2",
            )
            .bind(&source_scope.hash)
            .bind(target_hash)
            .execute(&self.pool)
            .await?
            .rows_affected()
        };

        Ok(rows_affected > 0)
    }

    /// Promote a scoped credential to the global scope, then remove it from the CWD scope.
    pub async fn promote_to_global(&self, provider: &str) -> Result<bool> {
        let provider = normalize_provider_name(provider);
        let scope_hash = self.current_scope_hash().to_string();

        if scope_hash == "global" {
            return Ok(false); // already global
        }

        let mut cwd_creds = self.get_cached_scope(&scope_hash);
        let key = match cwd_creds.remove(&provider) {
            Some(k) => k,
            None => return Ok(false),
        };

        // Write to global
        let mut global_creds = self.get_cached_scope("global");
        global_creds.insert(provider.clone(), key);
        self.update_scope("global", global_creds)?;

        // Remove from CWD scope
        self.update_scope(&scope_hash, cwd_creds)?;

        // Update DB metadata source to "keyring"
        if let Ok(Some(mut meta)) = self.credential_store.get_credential(&provider).await {
            meta.source = "keyring".to_string();
            meta.updated_at = Utc::now();
            let _ = self.credential_store.set_credential(&meta).await;
        }

        // Invalidate caches
        if let Ok(mut guard) = self.cache.write() {
            guard.remove("global");
            guard.remove(&scope_hash);
        }
        if let Ok(mut guard) = self.configured_cache.write() {
            *guard = None;
        }

        Ok(true)
    }

    /// List all registered scopes with their stored providers.
    pub async fn list_scopes(&self) -> Vec<ScopeRecord> {
        let rows = sqlx::query_as::<_, (String, String, Option<String>, String)>(
            "SELECT scope_path, scope_hash, label, updated_at FROM credential_scopes ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await
        .unwrap_or_default();

        let mut records = Vec::new();
        for (path, hash, label, updated_at) in rows {
            let creds = self.get_cached_scope(&hash);
            let providers: Vec<String> = creds.into_keys().collect();
            records.push(ScopeRecord {
                path,
                hash,
                label,
                providers,
                updated_at,
            });
        }

        // Include global scope if it has credentials
        let global_creds = self.get_cached_scope("global");
        if !global_creds.is_empty() {
            let providers: Vec<String> = global_creds.into_keys().collect();
            records.insert(
                0,
                ScopeRecord {
                    path: "global".to_string(),
                    hash: "global".to_string(),
                    label: Some("Global (all directories)".to_string()),
                    providers,
                    updated_at: String::new(),
                },
            );
        }

        records
    }

    /// Get all credentials for a specific scope hash (returns masked keys for display).
    pub fn get_scope_credentials(&self, scope_hash: &str) -> HashMap<String, String> {
        self.get_cached_scope(scope_hash)
    }

    /// Get all links for the current target scope.
    ///
    /// Returns `Vec<(source_hash, provider)>`.
    pub async fn get_linked_sources(&self, target_hash: &str) -> Result<Vec<(String, String)>> {
        let rows = sqlx::query_as::<_, (String, String)>(
            "SELECT source_hash, provider FROM credential_scope_links WHERE target_hash = ?1",
        )
        .bind(target_hash)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows)
    }

    /// Return a human-readable status view of all credentials accessible from the CWD.
    pub async fn status_for_current_scope(&self) -> ScopeStatus {
        let mut sources: Vec<ScopeSourceEntry> = Vec::new();

        // CWD scope
        if let Some(scope) = &self.current_scope {
            let creds = self.get_cached_scope(&scope.hash);
            if !creds.is_empty() {
                sources.push(ScopeSourceEntry {
                    label: format!("Directory: {}", scope.path),
                    hash: scope.hash.clone(),
                    providers: creds.into_keys().collect(),
                    is_linked: false,
                });
            }
        }

        // Linked scopes
        let target_hash = self.current_scope_hash().to_string();
        if let Ok(links) = self.get_linked_sources(&target_hash).await {
            for (source_hash, link_provider) in links {
                let creds = self.get_cached_scope(&source_hash);
                let path = self.scope_path_for_hash(&source_hash).await;
                let providers: Vec<String> = if link_provider == "*" {
                    creds.into_keys().collect()
                } else {
                    creds.into_keys().filter(|p| p == &link_provider).collect()
                };
                if !providers.is_empty() {
                    sources.push(ScopeSourceEntry {
                        label: format!("Linked: {}", path.unwrap_or_else(|| source_hash.clone())),
                        hash: source_hash,
                        providers,
                        is_linked: true,
                    });
                }
            }
        }

        // Global scope
        let global = self.get_cached_scope("global");
        if !global.is_empty() {
            sources.push(ScopeSourceEntry {
                label: "Global".to_string(),
                hash: "global".to_string(),
                providers: global.into_keys().collect(),
                is_linked: false,
            });
        }

        // Env vars
        let env_providers: Vec<String> = known_providers()
            .iter()
            .filter(|(_, env_var, _)| {
                !env_var.is_empty() && caliche_contracts::credential_env::var(env_var).is_ok()
            })
            .map(|(name, _, _)| name.to_string())
            .collect();

        ScopeStatus {
            cwd: self
                .current_scope
                .as_ref()
                .map(|s| s.path.clone())
                .unwrap_or_else(|| "global".to_string()),
            sources,
            env_providers,
        }
    }

    async fn scope_path_for_hash(&self, hash: &str) -> Option<String> {
        sqlx::query_scalar::<_, String>(
            "SELECT scope_path FROM credential_scopes WHERE scope_hash = ?1",
        )
        .bind(hash)
        .fetch_optional(&self.pool)
        .await
        .ok()
        .flatten()
    }
}

// ---------------------------------------------------------------------------
// Display types
// ---------------------------------------------------------------------------

/// A scope entry returned by `list_scopes`.
#[derive(Debug, Clone)]
pub struct ScopeRecord {
    pub path: String,
    pub hash: String,
    pub label: Option<String>,
    pub providers: Vec<String>,
    pub updated_at: String,
}

/// An individual source of credentials for the current CWD.
#[derive(Debug, Clone)]
pub struct ScopeSourceEntry {
    pub label: String,
    pub hash: String,
    pub providers: Vec<String>,
    pub is_linked: bool,
}

/// Full status view for the current working directory.
#[derive(Debug, Clone)]
pub struct ScopeStatus {
    pub cwd: String,
    pub sources: Vec<ScopeSourceEntry>,
    /// Providers found in process environment variables (not from keychain)
    pub env_providers: Vec<String>,
}

// ---------------------------------------------------------------------------
// KeychainCredentialProvider
// ---------------------------------------------------------------------------

/// A [`CredentialProvider`] backed directly by the OS keychain.
///
/// Reads credentials from the scoped keychain (CWD → linked scopes → global)
/// at construction time, so it does not see keys saved later.
///
/// Structured credentials stored as JSON (e.g. `{"access_key":"…","secret_key":"…"}`)
/// are automatically normalized to the appropriate `Credentials` variant via
/// [`Credentials::normalize`].
///
/// [`session_credential_provider`] chains it after the in-process store.
#[derive(Debug)]
pub struct KeychainCredentialProvider {
    /// Fully-resolved credential map: lowercase provider name → raw stored value.
    /// Values may be plain API key strings or JSON-encoded structured credential maps.
    resolved: std::collections::HashMap<String, String>,
    /// Human-readable label for `source_name()`.
    scope_label: String,
}

impl KeychainCredentialProvider {
    /// Build a provider by resolving the full scope chain from a `SecureCredentialManager`.
    ///
    /// Resolution order (lowest priority first so later entries overwrite earlier ones):
    /// global → linked scopes → CWD scope
    pub async fn from_manager(manager: &SecureCredentialManager) -> Self {
        let mut resolved: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();

        // Global (lowest priority)
        let global = manager.get_cached_scope("global");
        resolved.extend(global);

        // Linked scopes (middle priority)
        let target_hash = manager.current_scope_hash().to_string();
        if let Ok(links) = manager.get_linked_sources(&target_hash).await {
            for (source_hash, link_provider) in links {
                let mut creds = manager.get_cached_scope(&source_hash);
                creds.retain(|provider, _| link_provider == "*" || *provider == link_provider);
                resolved.extend(creds);
            }
        }

        // CWD scope (highest priority in keychain)
        if let Some(scope) = manager.current_scope() {
            let creds = manager.get_cached_scope(&scope.hash);
            resolved.extend(creds);
        }

        let scope_label = manager
            .current_scope()
            .map(|s| s.path.clone())
            .unwrap_or_else(|| "global".to_string());

        Self {
            resolved,
            scope_label,
        }
    }
}

impl std::fmt::Display for KeychainCredentialProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "KeychainCredentialProvider({})", self.scope_label)
    }
}

#[async_trait::async_trait]
impl caliche::traits::credentials::CredentialProvider for KeychainCredentialProvider {
    async fn get_credentials(
        &self,
        provider: &str,
    ) -> anyhow::Result<caliche::traits::credentials::Credentials> {
        use caliche::traits::credentials::Credentials;

        let provider_lower = normalize_provider_name(provider);
        let Some(value) = self.resolved.get(&provider_lower) else {
            anyhow::bail!(
                "No keychain credentials for provider '{}' (scope: {})",
                provider,
                self.scope_label
            );
        };

        let raw = value.trim();
        if raw.is_empty() {
            anyhow::bail!("Empty credential value for provider '{}'", provider);
        }

        // Structured credential stored as JSON — parse and normalize
        if raw.starts_with('{') {
            if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, String>>(raw)
            {
                if provider_lower == "cloudflare-ai-gateway" {
                    return Ok(Credentials::Custom(map));
                }
                return Ok(Credentials::Custom(map).normalize());
            }
        }

        // Plain API key (the common case)
        Ok(Credentials::ApiKey(value.clone()))
    }

    fn supports_provider(&self, provider: &str) -> bool {
        self.resolved
            .contains_key(&normalize_provider_name(provider))
    }

    fn source_name(&self) -> &str {
        "keychain"
    }
}

/// The credential provider CLI sessions and catalogs build model providers
/// with.
///
/// The in-process store comes first. Startup fills it from the shell and the
/// keychain scopes in priority order, and each key saved during the session
/// replaces its older value there, so `/provider` sees what `/creds` and
/// setup report as configured. The keychain values read when this is built
/// are the fallback for a process whose store was never filled.
pub async fn session_credential_provider(
    manager: Option<&SecureCredentialManager>,
) -> Arc<dyn caliche::traits::credentials::CredentialProvider> {
    use caliche::traits::credentials::{
        ChainedCredentialProvider, CredentialProvider, EnvironmentCredentialProvider,
    };

    let environment: Arc<dyn CredentialProvider> = Arc::new(EnvironmentCredentialProvider::new());
    let Some(manager) = manager else {
        return environment;
    };
    let keychain: Arc<dyn CredentialProvider> =
        Arc::new(KeychainCredentialProvider::from_manager(manager).await);
    Arc::new(ChainedCredentialProvider::new(vec![environment, keychain]))
}

#[cfg(any(test, feature = "test-utils"))]
impl SecureCredentialManager {
    /// Test helper: in-memory SQLite credential manager without keyring seeding.
    pub async fn new_for_test(db_path: &Path) -> Result<Self> {
        crate::storage::schema::ensure_schema(&db_path.to_path_buf()).await?;

        let credential_store = SqliteCredentialStore::from_path(db_path).await?;
        let pool = Self::open_pool(db_path).await?;
        Self::ensure_scope_tables(&pool).await?;

        Ok(Self {
            credential_store: Arc::new(credential_store),
            pool,
            current_scope: None,
            cache: std::sync::RwLock::new(HashMap::new()),
            unified_loaded: std::sync::RwLock::new(false),
            configured_cache: std::sync::RwLock::new(None),
        })
    }

    /// Test helper: whether a provider row exists in SQLite metadata.
    pub async fn has_db_credential(&self, provider: &str) -> Result<bool> {
        Ok(self
            .credential_store
            .get_credential(provider)
            .await?
            .is_some())
    }

    /// Test helper: seed the in-memory scope cache without touching the keyring.
    pub fn test_seed_scope_cache(&self, scope_hash: &str, provider: &str, api_key: &str) {
        if let Ok(mut guard) = self.cache.write() {
            guard
                .entry(scope_hash.to_string())
                .or_default()
                .insert(provider.to_string(), api_key.to_string());
        }
        if let Ok(mut loaded) = self.unified_loaded.write() {
            *loaded = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::keyring_prompts_allowed;

    #[test]
    #[serial_test::serial(env)]
    fn no_keyring_env_disables_prompts() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.set("CALICHE_NO_KEYRING", "1");
        env.remove("CALICHE_USE_KEYRING");
        assert!(!keyring_prompts_allowed());
    }

    /// Reads the flag only. It does not open the OS keyring.
    #[test]
    #[serial_test::serial(env)]
    fn use_keyring_env_forces_prompts_on() {
        let mut env = crate::test_env::EnvGuard::acquire();
        env.remove("CALICHE_NO_KEYRING");
        env.set("CALICHE_USE_KEYRING", "1");
        assert!(keyring_prompts_allowed());
    }
}
