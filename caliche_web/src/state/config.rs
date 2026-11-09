//! Configuration state management.

use crate::api::AgentSettings;
use serde::{Deserialize, Serialize};

fn default_show_sources() -> bool {
    true
}

/// Application configuration state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigState {
    /// Base URL for the API server
    pub api_base_url: String,
    /// Selected provider (if any)
    pub provider: Option<String>,
    /// Selected model (if any)
    pub model: Option<String>,
    /// Selected tools (if any)
    pub selected_tools: Vec<String>,
    /// Selected embedding provider (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_provider: Option<String>,
    /// Selected embedding model (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    /// Whether to display source citations on assistant messages
    #[serde(default = "default_show_sources")]
    pub show_sources: bool,
    /// Font size preference
    pub font_size: FontSize,
    /// Active reasoning pattern (None = server default)
    pub reasoning_pattern: Option<String>,
    /// This session changed provider or model after the server pair was shown.
    ///
    /// Not persisted. A reload adopts the server pair again.
    #[serde(skip)]
    pub user_edited_live_settings: bool,
    /// `settings.get` has returned a provider or model in this session.
    ///
    /// Not persisted. The header ignores select events until this is set and
    /// the control has painted, so the browser cannot write the first catalog
    /// id over the server pair.
    #[serde(skip)]
    pub server_settings_seen: bool,
}

/// Font size options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum FontSize {
    Small,
    #[default]
    Medium,
    Large,
}

impl FontSize {
    /// Get the CSS font size value
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn as_css(&self) -> &'static str {
        match self {
            FontSize::Small => "14px",
            FontSize::Medium => "16px",
            FontSize::Large => "18px",
        }
    }
}

impl Default for ConfigState {
    fn default() -> Self {
        Self {
            api_base_url: "http://localhost:3000".to_string(),
            provider: None,
            model: None,
            selected_tools: Vec::new(),
            embedding_provider: None,
            embedding_model: None,
            show_sources: true,
            font_size: FontSize::default(),
            reasoning_pattern: None,
            user_edited_live_settings: false,
            server_settings_seen: false,
        }
    }
}

impl ConfigState {
    /// Set the provider
    pub fn set_provider(&mut self, provider: Option<String>) {
        self.provider = provider;
    }

    /// Set the model
    pub fn set_model(&mut self, model: Option<String>) {
        self.model = model;
    }

    /// Add a tool to the selection
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn add_tool(&mut self, tool: impl Into<String>) {
        let tool = tool.into();
        if !self.selected_tools.contains(&tool) {
            self.selected_tools.push(tool);
        }
    }

    /// Remove a tool from the selection
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn remove_tool(&mut self, tool: &str) {
        self.selected_tools.retain(|t| t != tool);
    }

    /// Toggle a tool selection
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn toggle_tool(&mut self, tool: impl Into<String>) {
        let tool = tool.into();
        if self.selected_tools.contains(&tool) {
            self.selected_tools.retain(|t| t != &tool);
        } else {
            self.selected_tools.push(tool);
        }
    }

    /// Clear all selected tools
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn clear_tools(&mut self) {
        self.selected_tools.clear();
    }

    /// Set the font size
    #[allow(
        dead_code,
        reason = "nothing calls it yet; lib.rs and main.rs both compile this module, so expect cannot hold in both"
    )]
    pub fn set_font_size(&mut self, size: FontSize) {
        self.font_size = size;
    }

    /// Apply provider, model, tools, embedding, and source-visibility from server settings.
    pub fn apply_from_settings(&mut self, settings: &AgentSettings) {
        self.provider = if settings.provider.is_empty() {
            None
        } else {
            Some(settings.provider.clone())
        };
        self.model = if settings.model.is_empty() {
            None
        } else {
            Some(settings.model.clone())
        };
        self.selected_tools = settings.tools.clone();
        self.embedding_provider = settings
            .embedding_provider
            .as_ref()
            .filter(|provider| !provider.is_empty())
            .cloned();
        self.embedding_model = settings
            .embedding_model
            .as_ref()
            .filter(|model| !model.is_empty())
            .cloned();
        if let Some(show_sources) = settings.show_sources {
            self.show_sources = show_sources;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::InteractionMode;

    fn settings(
        provider: &str,
        model: &str,
        tools: &[&str],
        embedding_provider: Option<&str>,
        show_sources: Option<bool>,
    ) -> AgentSettings {
        AgentSettings {
            provider: provider.to_string(),
            model: model.to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: tools.iter().map(ToString::to_string).collect(),
            mode: Some(InteractionMode::Agent),
            embedding_provider: embedding_provider.map(ToString::to_string),
            embedding_model: None,
            show_sources,
            vector_backend: None,
            vector_backend_state: None,
        }
    }

    #[test]
    fn add_tool_is_idempotent() {
        let mut config = ConfigState::default();

        config.add_tool("read_file");
        config.add_tool("read_file");

        assert_eq!(config.selected_tools, vec!["read_file"]);
    }

    #[test]
    fn toggle_tool_adds_and_removes_without_duplicates() {
        let mut config = ConfigState::default();

        config.toggle_tool("search");
        config.toggle_tool("search");
        config.toggle_tool("search");

        assert_eq!(config.selected_tools, vec!["search"]);
    }

    #[test]
    fn apply_from_settings_clears_empty_provider_and_model() {
        let mut config = ConfigState {
            provider: Some("openai".to_string()),
            model: Some("gpt-4o".to_string()),
            selected_tools: vec!["read_file".to_string()],
            ..ConfigState::default()
        };

        config.apply_from_settings(&settings("", "", &[], None, None));

        assert_eq!(config.provider, None);
        assert_eq!(config.model, None);
        assert!(config.selected_tools.is_empty());
    }

    #[test]
    fn apply_from_settings_replaces_tools_instead_of_appending() {
        let mut config = ConfigState {
            selected_tools: vec!["old_tool".to_string()],
            ..ConfigState::default()
        };

        config.apply_from_settings(&settings("anthropic", "claude", &["new_tool"], None, None));
        config.apply_from_settings(&settings("anthropic", "claude", &["new_tool"], None, None));

        assert_eq!(config.selected_tools, vec!["new_tool"]);
    }

    #[test]
    fn apply_from_settings_persists_embedding_and_show_sources() {
        let mut config = ConfigState::default();
        config.apply_from_settings(&settings(
            "openai",
            "gpt-4o",
            &[],
            Some("openai"),
            Some(false),
        ));

        assert_eq!(config.embedding_provider.as_deref(), Some("openai"));
        assert!(!config.show_sources);
    }

    #[test]
    fn apply_from_settings_clears_embedding_when_server_omits_it() {
        let mut config = ConfigState {
            embedding_provider: Some("openai".to_string()),
            embedding_model: Some("text-embedding-3-small".to_string()),
            ..ConfigState::default()
        };

        config.apply_from_settings(&settings("openai", "gpt-4o", &[], None, None));

        assert_eq!(config.embedding_provider, None);
        assert_eq!(config.embedding_model, None);
    }

    #[test]
    fn missing_show_sources_in_saved_json_defaults_to_true() {
        let parsed: ConfigState =
            serde_json::from_str(r#"{"api_base_url":"http://localhost:3000","provider":null,"model":null,"selected_tools":[],"font_size":"Medium"}"#)
                .expect("deserialize");
        assert!(parsed.show_sources);
        assert_eq!(parsed.embedding_provider, None);
    }
}
