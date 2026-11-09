//! Load keychain credentials into the in-process credential store with shell-first priority.
//!
//! Nothing here writes the process environment; readers use
//! [`caliche_contracts::credential_env::var`].

use super::credentials::SecureCredentialManager;
use crate::credentials::{GATEWAY_CREDENTIALS, known_providers};
use caliche_contracts::credential_env;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Names in the process environment before `--env-file` was loaded.
static SHELL_ENV_KEYS: OnceLock<HashSet<String>> = OnceLock::new();

/// Record the names the shell exported. Call once, before `--env-file` is
/// loaded into the process environment.
pub fn record_shell_env_keys() {
    let _ = SHELL_ENV_KEYS.set(std::env::vars().map(|(k, _)| k).collect());
}

/// The names [`record_shell_env_keys`] recorded, or every name in the
/// process environment when nothing was recorded.
pub fn shell_env_keys() -> HashSet<String> {
    SHELL_ENV_KEYS
        .get()
        .cloned()
        .unwrap_or_else(|| std::env::vars().map(|(k, _)| k).collect())
}

/// Put the key `manager` resolves for `provider` into the in-process store,
/// replacing any value stored earlier, so a key saved during a session is
/// the one the session uses. A name the shell exported keeps its value.
pub(super) async fn publish(manager: &SecureCredentialManager, provider: &str) {
    let Some(key) = manager.resolve_from_scope_chain(provider).await else {
        return;
    };
    let mut locked = locked_env_vars(&shell_env_keys());
    apply_provider_key(provider, &key, &mut locked);
}

/// [`publish`] every provider and gateway credential: the store then holds
/// what the scope chain resolves now, for example after a scope link.
pub(super) async fn publish_all(manager: &SecureCredentialManager) {
    for (provider, _, _) in known_providers().iter().chain(GATEWAY_CREDENTIALS) {
        publish(manager, provider).await;
    }
}

pub(super) async fn apply_to_credential_store_with_priority(
    manager: &SecureCredentialManager,
    shell_env_keys: &HashSet<String>,
) {
    let mut applied = locked_env_vars(shell_env_keys);

    if let Some(scope) = manager.current_scope() {
        let creds = manager.get_cached_scope(&scope.hash);
        apply_credential_map(&creds, &mut applied);
    }

    let target_hash = manager.current_scope_hash().to_string();
    if let Ok(links) = manager.get_linked_sources(&target_hash).await {
        for (source_hash, link_provider) in links {
            let mut creds = manager.get_cached_scope(&source_hash);
            creds.retain(|provider, _| link_provider == "*" || *provider == link_provider);
            apply_credential_map(&creds, &mut applied);
        }
    }

    let global = manager.get_cached_scope("global");
    apply_credential_map(&global, &mut applied);
}

fn locked_env_vars(shell_env_keys: &HashSet<String>) -> HashSet<String> {
    let mut locked = HashSet::new();
    for (_, env_var, _) in known_providers() {
        if !env_var.is_empty() && shell_env_keys.contains(*env_var) {
            locked.insert((*env_var).to_string());
        }
    }
    for (_, env_var, _) in GATEWAY_CREDENTIALS {
        if shell_env_keys.contains(*env_var) {
            locked.insert((*env_var).to_string());
        }
    }
    locked
}

fn set_env_if_available(env_var: &str, value: &str, applied: &mut HashSet<String>) {
    if !applied.contains(env_var) {
        credential_env::set(env_var, value);
        applied.insert(env_var.to_string());
    }
}

fn apply_cloudflare_gateway_values(
    values: &HashMap<String, String>,
    applied: &mut HashSet<String>,
) {
    if let Some(api_key) = values.get("api_key") {
        set_env_if_available("CLOUDFLARE_AIG_API_KEY", api_key, applied);
    }
    if let Some(auth_token) = values.get("cf_aig_authorization") {
        set_env_if_available("CLOUDFLARE_AIG_AUTH_TOKEN", auth_token, applied);
    }
    if let Some(account_id) = values.get("account_id") {
        set_env_if_available("CLOUDFLARE_ACCOUNT_ID", account_id, applied);
    }
    if let Some(gateway_id) = values.get("gateway_id") {
        set_env_if_available("CLOUDFLARE_AIG_GATEWAY_ID", gateway_id, applied);
    }
}

fn apply_bedrock_values(values: &HashMap<String, String>, applied: &mut HashSet<String>) {
    if let Some(access_key) = values.get("access_key") {
        set_env_if_available("AWS_ACCESS_KEY_ID", access_key, applied);
    }
    if let Some(secret_key) = values.get("secret_key") {
        set_env_if_available("AWS_SECRET_ACCESS_KEY", secret_key, applied);
    }
    if let Some(session_token) = values.get("session_token") {
        set_env_if_available("AWS_SESSION_TOKEN", session_token, applied);
    }
    if let Some(region) = values.get("region") {
        set_env_if_available("AWS_REGION", region, applied);
    }
}

fn apply_json_provider_values(
    provider: &str,
    values: &HashMap<String, String>,
    applied: &mut HashSet<String>,
) -> bool {
    if provider == "cloudflare-ai-gateway" {
        apply_cloudflare_gateway_values(values, applied);
        return true;
    }
    if provider == "bedrock" {
        apply_bedrock_values(values, applied);
        return true;
    }
    false
}

fn apply_provider_key(provider: &str, key: &str, applied: &mut HashSet<String>) {
    if key.trim_start().starts_with('{')
        && let Ok(values) = serde_json::from_str::<HashMap<String, String>>(key)
        && apply_json_provider_values(provider, &values, applied)
    {
        return;
    }

    if let Some((_, env_var, _)) = known_providers().iter().find(|(p, _, _)| *p == provider)
        && !env_var.is_empty()
        && !applied.contains(*env_var)
    {
        credential_env::set(*env_var, key);
        applied.insert((*env_var).to_string());
    }
    if let Some((_, env_var, _)) = GATEWAY_CREDENTIALS.iter().find(|(p, _, _)| *p == provider)
        && !applied.contains(*env_var)
    {
        credential_env::set(*env_var, key);
        applied.insert((*env_var).to_string());
    }
}

fn apply_credential_map(creds: &HashMap<String, String>, applied: &mut HashSet<String>) {
    for (provider, key) in creds {
        apply_provider_key(provider, key, applied);
    }
}
