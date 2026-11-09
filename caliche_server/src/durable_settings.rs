//! Durable provider, model, and sampling settings beside `mcp.json`.
//!
//! `--server` reads this file on boot. `PUT /api/settings` writes it. The
//! system prompt is not a field: project instructions stay in `AGENTS.md` and
//! the built-in prompt, which startup assembles. A settings save must not
//! freeze the live prompt into the next process.

use caliche::AgentSettings;
use serde::{Deserialize, Serialize};
use std::path::Path;

const SETTINGS_FILE_NAME: &str = "settings.json";

/// Provider, model, and sampling values that survive a process restart.
///
/// Precedence at boot is CLI flag, then this record, then env or the provider
/// default. `--provider` selects the provider and still keeps this record's
/// model when the saved provider matches. `--model` and `--temperature`
/// replace the saved values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DurableSettings {
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
}

impl DurableSettings {
    /// Copy the fields this file is allowed to store.
    ///
    /// `system_prompt` is omitted on purpose. The snapshot used by
    /// `GET /api/settings` includes the effective prompt for inspection.
    #[must_use]
    pub fn from_agent_settings(settings: &AgentSettings) -> Self {
        Self {
            provider: settings.provider.to_string(),
            model: settings.model.clone(),
            temperature: settings.temperature,
            max_tokens: settings.max_tokens,
        }
    }
}

/// `settings.json` in the same directory as the MCP config file.
#[must_use]
pub fn path_beside_mcp(mcp_config_path: &Path) -> std::path::PathBuf {
    mcp_config_path.with_file_name(SETTINGS_FILE_NAME)
}

/// Default path: `dirs::config_dir()/caliche/settings.json`.
#[must_use]
pub fn default_path() -> std::path::PathBuf {
    path_beside_mcp(&crate::bootstrap::default_mcp_config_path())
}

/// Load a settings file. Missing or unreadable files yield `None`.
pub async fn load(path: &Path) -> Option<DurableSettings> {
    let content = tokio::fs::read_to_string(path).await.ok()?;
    match serde_json::from_str::<DurableSettings>(&content) {
        Ok(settings) => Some(settings),
        Err(err) => {
            tracing::warn!(
                path = %path.display(),
                error = %err,
                "ignoring unreadable durable settings"
            );
            None
        }
    }
}

/// Write the settings file. Parent directories are created when missing.
pub async fn save(path: &Path, settings: &DurableSettings) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent).await?;
    }
    let content = serde_json::to_string_pretty(settings)?;
    tokio::fs::write(path, content).await?;
    Ok(())
}

/// Choose a sampling value.
///
/// CLI flag wins. A value already applied from project config stays when no
/// flag is set. Otherwise the durable record fills the gap.
#[must_use]
pub fn sampling_value<T>(cli: Option<T>, saved: Option<T>, already_set: bool) -> Option<T> {
    if cli.is_some() {
        return cli;
    }
    if already_set {
        return None;
    }
    saved
}

#[cfg(test)]
mod tests {
    use super::*;
    use caliche::llm::ProviderType;

    fn sample_agent_settings() -> AgentSettings {
        AgentSettings {
            provider: ProviderType::Anthropic,
            model: "claude-sonnet-4-5-20250929".to_string(),
            system_prompt: Some(
                "IDENTITY:\n# Project Context (from AGENTS.md)\nMARKER".to_string(),
            ),
            temperature: Some(0.5),
            max_tokens: Some(256),
            context_window: Some(8_000),
            tools: vec!["fs".to_string()],
            mode: None,
            embedding_provider: None,
            embedding_model: None,
            show_sources: Some(true),
        }
    }

    #[test]
    fn from_agent_settings_omits_system_prompt() {
        let record = DurableSettings::from_agent_settings(&sample_agent_settings());
        let json = serde_json::to_string(&record).expect("serialize");
        assert!(!json.contains("system_prompt"));
        assert!(!json.contains("AGENTS.md"));
        assert!(!json.contains("MARKER"));
        assert!(json.contains("claude-sonnet-4-5-20250929"));
        assert_eq!(record.temperature, Some(0.5));
        assert_eq!(record.max_tokens, Some(256));
    }

    #[test]
    fn path_beside_mcp_uses_the_mcp_directory() {
        let path = path_beside_mcp(Path::new("/tmp/caliche/mcp.json"));
        assert_eq!(path, Path::new("/tmp/caliche/settings.json"));
    }

    #[tokio::test]
    async fn save_and_load_roundtrip_without_prompt() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.json");
        let record = DurableSettings::from_agent_settings(&sample_agent_settings());
        save(&path, &record).await.expect("save");
        let raw = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(!raw.contains("system_prompt"));
        let loaded = load(&path).await.expect("load");
        assert_eq!(loaded, record);
    }

    #[tokio::test]
    async fn load_ignores_a_system_prompt_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.json");
        let raw = r#"{
            "provider": "anthropic",
            "model": "claude-sonnet-4-5-20250929",
            "temperature": 0.5,
            "system_prompt": "do not keep this"
        }"#;
        tokio::fs::write(&path, raw).await.expect("write");
        let loaded = load(&path).await.expect("load");
        assert_eq!(loaded.model, "claude-sonnet-4-5-20250929");
        assert_eq!(loaded.temperature, Some(0.5));
        save(&path, &loaded).await.expect("save");
        let rewritten = tokio::fs::read_to_string(&path).await.expect("read");
        assert!(!rewritten.contains("system_prompt"));
        assert!(!rewritten.contains("do not keep this"));
    }

    #[test]
    fn sampling_value_prefers_cli_then_saved_then_existing() {
        assert_eq!(sampling_value(Some(0.2_f32), Some(0.5), true), Some(0.2));
        assert_eq!(sampling_value(None, Some(0.5), true), None);
        assert_eq!(sampling_value(None, Some(0.5), false), Some(0.5));
        assert_eq!(sampling_value::<f32>(None, None, false), None);
    }
}
