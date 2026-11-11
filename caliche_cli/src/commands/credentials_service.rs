//! Shared credential operations for CLI handlers and HTTP parity routes.
//!
//! Centralizes keyring-backed provider/gateway/vault logic so `handlers/*` and
//! `server/routes/*` stay thin adapters over one implementation.

use crate::credentials::{GATEWAY_CREDENTIALS, ProviderStatus, known_providers};
use crate::storage::{DetectedCredential, ScopeRecord, ScopeStatus, SecureCredentialManager};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const SECRET_CONFIRM_FLAG: &str = "--yes-i-understand-this-prints-secrets";

pub const VAULT_IMPORT_CONFIRM_MESSAGE: &str =
    "vault import requires confirm: true (acknowledges writing detected secrets to the keychain)";

/// Facade over [`SecureCredentialManager`] with shared credential UX helpers.
pub struct CredentialService {
    manager: SecureCredentialManager,
}

impl CredentialService {
    /// HTTP/JSON vault import must explicitly acknowledge writing detected secrets.
    pub fn validate_vault_import_confirm(confirm: Option<bool>) -> Result<()> {
        if confirm.unwrap_or(false) {
            Ok(())
        } else {
            bail!(VAULT_IMPORT_CONFIRM_MESSAGE)
        }
    }

    pub async fn new() -> Result<Self> {
        Ok(Self {
            manager: SecureCredentialManager::new().await?,
        })
    }

    pub async fn new_with_scope(dir: Option<&Path>) -> Result<Self> {
        Ok(Self {
            manager: SecureCredentialManager::new_with_scope(dir).await?,
        })
    }

    pub fn manager(&self) -> &SecureCredentialManager {
        &self.manager
    }

    /// Mask stored secrets for display (never reveal partial key material).
    pub fn mask_for_display(_value: &str) -> String {
        "****".to_string()
    }

    pub fn format_credential_value(value: &str, unmask: bool) -> String {
        if unmask {
            value.to_string()
        } else {
            Self::mask_for_display(value)
        }
    }

    pub fn require_secret_confirmation(confirmed: bool, action: &str) -> Result<()> {
        if confirmed {
            Ok(())
        } else {
            bail!(
                "{action} prints secrets to stdout. Re-run with {SECRET_CONFIRM_FLAG} if you really need raw output."
            )
        }
    }

    pub fn secret_confirmation_message(action: &str) -> String {
        format!(
            "{action} prints secrets. Set yes_i_understand_this_prints_secrets=true if raw output is required."
        )
    }

    pub async fn set_provider_key(&self, provider: &str, api_key: &str) -> Result<()> {
        if api_key.is_empty() {
            bail!("API key cannot be empty");
        }
        self.manager.set_api_key(provider, api_key).await
    }

    pub async fn get_provider_key(&self, provider: &str) -> Option<String> {
        self.manager.get_api_key(provider).await
    }

    pub async fn remove_provider_key(&self, provider: &str) -> Result<bool> {
        self.manager.remove_api_key(provider).await
    }

    pub async fn list_provider_status(&self, configured_only: bool) -> Vec<ProviderStatus> {
        let providers = self.manager.list_configured().await;
        if configured_only {
            providers
                .into_iter()
                .filter(|status| status.is_configured)
                .collect()
        } else {
            providers
        }
    }

    pub async fn default_provider(&self) -> Option<String> {
        self.manager.default_provider().await
    }

    pub async fn set_default_provider(&self, provider: &str) -> Result<()> {
        self.manager.set_default_provider(provider).await
    }

    pub async fn set_default_model(&self, provider: &str, model: &str) -> Result<()> {
        self.manager.set_default_model(provider, model).await
    }

    pub fn normalize_gateway_name(name: &str) -> String {
        name.to_ascii_lowercase()
    }

    pub fn is_known_gateway(name: &str) -> bool {
        let normalized = Self::normalize_gateway_name(name);
        GATEWAY_CREDENTIALS
            .iter()
            .any(|(credential, _, _)| *credential == normalized)
    }

    pub async fn set_gateway_credential(&self, name: &str, value: &str) -> Result<()> {
        let cred_name = Self::normalize_gateway_name(name);
        if !Self::is_known_gateway(&cred_name) {
            bail!(
                "Unknown gateway credential '{name}'. Run 'caliche config gateway list' to see supported names."
            );
        }
        if value.is_empty() {
            bail!("Credential value cannot be empty");
        }
        self.manager.set_api_key(&cred_name, value).await
    }

    pub async fn gateway_status_rows(
        &self,
        configured_only: bool,
    ) -> Vec<(&'static str, &'static str, &'static str, bool)> {
        let mut rows = Vec::new();
        for (name, env_var, description) in GATEWAY_CREDENTIALS {
            let configured = self.manager.get_api_key(name).await.is_some();
            if configured_only && !configured {
                continue;
            }
            rows.push((*name, *env_var, *description, configured));
        }
        rows
    }

    pub async fn vault_status(&self) -> ScopeStatus {
        self.manager.status_for_current_scope().await
    }

    pub async fn vault_list_scopes(&self) -> Vec<ScopeRecord> {
        self.manager.list_scopes().await
    }

    pub fn current_scope_hash(&self) -> &str {
        self.manager.current_scope_hash()
    }

    pub fn current_scope_path(&self) -> String {
        self.manager
            .current_scope()
            .map(|scope| scope.path.clone())
            .unwrap_or_else(|| "global".to_string())
    }

    pub fn vault_scope_credentials(&self, scope_hash: &str) -> HashMap<String, String> {
        self.manager.get_scope_credentials(scope_hash)
    }

    pub fn scope_hash_for_path(path: &str) -> String {
        crate::storage::ScopeInfo::from_path(path).hash
    }

    pub async fn vault_link(&self, source_dir: &str, provider: &str) -> Result<()> {
        self.manager.link_scope(source_dir, provider).await
    }

    pub async fn vault_unlink(&self, source_dir: &str, provider: Option<&str>) -> Result<bool> {
        self.manager.unlink_scope(source_dir, provider).await
    }

    pub async fn vault_import_global(&self, detected: &[DetectedCredential]) -> Result<usize> {
        let mut stored = 0usize;
        for credential in detected {
            self.manager
                .set_api_key(&credential.provider_name, &credential.value)
                .await?;
            stored += 1;
        }
        Ok(stored)
    }

    pub async fn vault_import_directory(&self, detected: &[DetectedCredential]) -> Result<usize> {
        self.manager.ingest_env_credentials(detected).await?;
        Ok(detected.len())
    }

    pub async fn vault_promote(&self, provider: &str) -> Result<bool> {
        self.manager.promote_to_global(provider).await
    }

    pub async fn build_export_lines(&self, shell: &str) -> Result<Vec<String>> {
        Self::validate_shell(shell)?;
        let mut lines = Vec::new();

        for (provider, env_var, _) in known_providers() {
            if !env_var.is_empty()
                && let Some(key) = self.manager.get_api_key(provider).await
            {
                lines.push(Self::shell_export_line(shell, env_var, &key)?);
            }
        }
        for (name, env_var, _) in GATEWAY_CREDENTIALS {
            if let Some(key) = self.manager.get_api_key(name).await {
                lines.push(Self::shell_export_line(shell, env_var, &key)?);
            }
        }
        if let Some(provider) = self.manager.default_provider().await
            && provider != "env"
        {
            lines.push(Self::shell_export_line(shell, "AGENT_PROVIDER", &provider)?);
        }
        Ok(lines)
    }

    pub fn resolve_dir(path_str: &str) -> Result<PathBuf> {
        let expanded: PathBuf = if let Some(suffix) = path_str.strip_prefix("~/") {
            dirs::home_dir()
                .ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?
                .join(suffix)
        } else if path_str == "~" {
            dirs::home_dir().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?
        } else {
            PathBuf::from(path_str)
        };
        if expanded.is_absolute() {
            Ok(expanded)
        } else {
            Ok(std::env::current_dir()?.join(expanded))
        }
    }

    pub fn validate_shell(shell: &str) -> Result<()> {
        match shell {
            "bash" | "zsh" | "sh" | "fish" | "powershell" | "pwsh" => Ok(()),
            _ => bail!("Unknown shell type: {shell}. Supported: bash, zsh, fish, powershell"),
        }
    }

    pub fn default_export_path(shell: &str) -> PathBuf {
        let extension = match shell {
            "fish" => "fish",
            "powershell" | "pwsh" => "ps1",
            _ => "sh",
        };
        std::env::temp_dir().join(format!("caliche-env-{}.{}", std::process::id(), extension))
    }

    pub fn shell_export_line(shell: &str, env_var: &str, value: &str) -> Result<String> {
        match shell {
            "bash" | "zsh" | "sh" => Ok(format!("export {env_var}={}", Self::posix_quote(value))),
            "fish" => Ok(format!("set -gx {env_var} {}", Self::posix_quote(value))),
            "powershell" | "pwsh" => Ok(format!(
                "$env:{env_var} = {}",
                Self::powershell_quote(value)
            )),
            _ => bail!("Unknown shell type: {shell}"),
        }
    }

    pub fn posix_quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "'\\''"))
    }

    pub fn powershell_quote(value: &str) -> String {
        format!("'{}'", value.replace('\'', "''"))
    }

    pub fn provider_description(name: &str) -> &'static str {
        known_providers()
            .iter()
            .find(|(provider, _, _)| *provider == name)
            .map(|(_, _, desc)| *desc)
            .unwrap_or("Custom provider")
    }

    pub fn env_file_names_display() -> &'static str {
        ".env, .env.local, .env.development, .env.production, .env.staging"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A build without bedrock or azure-openai still exports their keys:
    /// export reads every known provider, not the offered ones.
    #[tokio::test]
    #[serial_test::serial(env)]
    async fn export_keeps_keys_for_providers_this_build_left_out() {
        let home = tempfile::tempdir().expect("temp home");
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("HOME", home.path());
        env.set("CALICHE_NO_KEYRING", "1");
        env.set("AWS_ACCESS_KEY_ID", "fake-aws-key");
        env.set("AZURE_OPENAI_API_KEY", "fake-azure-key");
        let service = CredentialService {
            manager: SecureCredentialManager::new_for_test(&home.path().join("caliche.db"))
                .await
                .expect("credential manager"),
        };

        let lines = service.build_export_lines("bash").await.expect("export");
        let cloud: Vec<&str> = lines
            .iter()
            .map(String::as_str)
            .filter(|line| line.contains("AWS_") || line.contains("AZURE_"))
            .collect();

        assert_eq!(
            cloud,
            vec![
                "export AWS_ACCESS_KEY_ID='fake-aws-key'",
                "export AZURE_OPENAI_API_KEY='fake-azure-key'",
            ]
        );
    }
}
