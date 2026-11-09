//! Credential loading and setup.

use crate::credentials::provider_has_env_credentials;
use crate::storage::SecureCredentialManager;
use std::collections::HashSet;

/// Load credentials from the secure credential store.
pub async fn load_credentials() -> Option<SecureCredentialManager> {
    SecureCredentialManager::new().await.ok()
}

/// Load keychain credentials into the in-process credential store during CLI startup.
///
/// Values the shell exported before launch keep priority. The process
/// environment is not written.
pub async fn load_startup_credentials(
    cred_manager: &SecureCredentialManager,
    shell_env_keys: &HashSet<String>,
) {
    cred_manager
        .apply_to_credential_store_with_priority(shell_env_keys)
        .await;
}

/// Set up credentials if needed for the given provider.
///
/// Returns true if credentials were set up, false if they were already available.
pub async fn setup_credentials_if_needed(
    cred_manager: Option<&SecureCredentialManager>,
    provider_name: &str,
) -> anyhow::Result<bool> {
    // Check if credentials are already available
    if provider_has_env_credentials(provider_name) {
        return Ok(false);
    }

    let has_stored_creds = if let Some(cm) = cred_manager {
        let configured = cm.list_configured().await;
        let provider = crate::credentials::normalize_provider_name(provider_name);
        configured
            .iter()
            .find(|p| p.name == provider)
            .map(|p| p.is_configured)
            .unwrap_or(false)
    } else {
        false
    };

    if has_stored_creds {
        return Ok(false);
    }

    // Need to set up credentials
    println!(
        "\n\x1b[33mNo API credentials configured for {}.\x1b[0m",
        provider_name
    );
    println!("Let's set up your provider.\n");
    // Each key is in the in-process store as soon as setup saves it.
    let setup = crate::credentials::interactive_setup().await;
    crate::credentials::report_startup_setup(&setup);
    Ok(setup.is_ok_and(|report| report.outcome == crate::credentials::SetupOutcome::Completed))
}
