//! Merge persisted CLI settings into provider/model discovery lists.

use crate::api::{AgentSettings, ModelInfo, ProviderCatalogEntry};
use crate::state::ConfigState;

/// "(not set)" is only a placeholder while no name is committed.
///
/// `PUT /api/settings` parses `provider` as a provider id. An empty string is
/// rejected, so a committed provider cannot be cleared from the select.
#[must_use]
pub fn show_uncommitted_option(current: &str) -> bool {
    current.trim().is_empty()
}

/// Whether the header should replace its provider and model with the server pair.
///
/// The connection loop used to do this only on the disconnected-to-connected
/// edge. A `settings.get` that arrives on a later poll (or a select that
/// paints the first catalog id before that payload is applied) then left the
/// header on the wrong provider until the settings tab mounted and applied it.
/// The server pair wins until the operator edits a control in this session.
#[must_use]
pub fn should_adopt_server_settings(
    local_provider: &str,
    local_model: &str,
    server_provider: &str,
    server_model: &str,
    user_edited: bool,
) -> bool {
    if user_edited {
        return false;
    }
    let server_provider = server_provider.trim();
    let server_model = server_model.trim();
    if server_provider.is_empty() && server_model.is_empty() {
        return false;
    }
    local_provider.trim() != server_provider || local_model.trim() != server_model
}

/// Whether a header select event is an operator change.
///
/// Events are ignored until the server pair has been seen and the control has
/// painted, and when the event repeats the value already committed. That stops
/// the browser from persisting the first catalog option while options mount.
#[must_use]
pub fn accept_header_selection(server_seen: bool, armed: bool, current: &str, next: &str) -> bool {
    server_seen && armed && current != next
}

/// Whether a live header/settings change should be written.
///
/// An empty provider is not a clear. `PUT /api/settings` parses `provider` as
/// a provider id and returns 400 for an empty string, with or without a model.
#[must_use]
pub fn should_put_live_settings(provider: &str, _model: &str) -> bool {
    !provider.trim().is_empty()
}

/// Build a settings payload from the current UI config.
#[must_use]
pub fn settings_from_config(config: &ConfigState) -> AgentSettings {
    AgentSettings {
        provider: config.provider.clone().unwrap_or_default(),
        model: config.model.clone().unwrap_or_default(),
        system_prompt: None,
        temperature: None,
        max_tokens: None,
        context_window: None,
        tools: config.selected_tools.clone(),
        mode: None,
        embedding_provider: config.embedding_provider.clone(),
        embedding_model: config.embedding_model.clone(),
        show_sources: Some(config.show_sources),
        vector_backend: None,
        vector_backend_state: None,
    }
}

/// Ensure the active CLI provider/model appear in dropdown options when discovery omits them.
pub fn merge_current_settings_into_model_options(
    providers: &mut Vec<ProviderCatalogEntry>,
    models: &mut Vec<ModelInfo>,
    settings: &mut Option<AgentSettings>,
) {
    let Some(settings) = settings.as_mut() else {
        return;
    };
    if settings.provider.trim().is_empty() {
        return;
    }

    let provider_id = settings.provider.trim().to_ascii_lowercase();
    settings.provider = provider_id.clone();

    let provider_display_name = providers
        .iter()
        .find(|provider| provider.id == provider_id)
        .map(|provider| provider.display_name.clone())
        .unwrap_or_else(|| provider_id.clone());

    if !providers.iter().any(|provider| provider.id == provider_id) {
        providers.push(ProviderCatalogEntry {
            id: provider_id.clone(),
            display_name: provider_display_name.clone(),
            configured: true,
            local: false,
            models: Vec::new(),
        });
    }

    providers.sort_by(|left, right| left.id.cmp(&right.id));
    if settings.model.trim().is_empty() {
        return;
    }

    let has_current_model = models
        .iter()
        .any(|model| model.provider_id == provider_id && model.id == settings.model);
    if !has_current_model {
        let model = ModelInfo {
            id: settings.model.clone(),
            name: settings.model.clone(),
            provider_id: provider_id.clone(),
            provider_display_name,
            description: Some("current CLI setting".to_string()),
            context_length: None,
        };
        if let Some(provider) = providers
            .iter_mut()
            .find(|provider| provider.id == provider_id)
        {
            provider.models.push(model.clone());
            provider
                .models
                .sort_by(|left, right| left.id.cmp(&right.id));
        }
        models.push(model);
    }
    models.sort_by(|left, right| {
        left.provider_id
            .cmp(&right.provider_id)
            .then_with(|| left.id.cmp(&right.id))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::InteractionMode;

    #[test]
    fn settings_provider_model_are_added_to_options_when_discovery_omits_them() {
        let mut providers = Vec::new();
        let mut models = Vec::new();
        let mut settings = Some(AgentSettings {
            provider: "openrouter".to_string(),
            model: "openai/gpt-4.1".to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: Vec::new(),
            mode: Some(InteractionMode::Agent),
            embedding_provider: None,
            embedding_model: None,
            show_sources: None,
            vector_backend: None,
            vector_backend_state: None,
        });

        merge_current_settings_into_model_options(&mut providers, &mut models, &mut settings);

        assert_eq!(settings.as_ref().unwrap().provider, "openrouter");
        assert_eq!(providers[0].id, "openrouter");
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].provider_id, "openrouter");
        assert_eq!(models[0].id, "openai/gpt-4.1");
    }

    #[test]
    fn settings_provider_uses_discovered_provider_id() {
        let mut providers = vec![ProviderCatalogEntry {
            id: "anthropic".to_string(),
            display_name: "Anthropic".to_string(),
            configured: true,
            local: false,
            models: Vec::new(),
        }];
        let mut models = vec![ModelInfo {
            id: "claude-sonnet-4-5".to_string(),
            name: "claude-sonnet-4-5".to_string(),
            provider_id: "anthropic".to_string(),
            provider_display_name: "Anthropic".to_string(),
            description: None,
            context_length: None,
        }];
        let mut settings = Some(AgentSettings {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-5".to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: Vec::new(),
            mode: Some(InteractionMode::Agent),
            embedding_provider: None,
            embedding_model: None,
            show_sources: None,
            vector_backend: None,
            vector_backend_state: None,
        });

        merge_current_settings_into_model_options(&mut providers, &mut models, &mut settings);

        assert_eq!(settings.as_ref().unwrap().provider, "anthropic");
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn merging_current_settings_is_idempotent_across_reconnects() {
        let mut providers = Vec::new();
        let mut models = Vec::new();
        let mut settings = Some(AgentSettings {
            provider: "OPENROUTER".to_string(),
            model: "openai/gpt-4.1".to_string(),
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            context_window: None,
            tools: Vec::new(),
            mode: Some(InteractionMode::Agent),
            embedding_provider: None,
            embedding_model: None,
            show_sources: None,
            vector_backend: None,
            vector_backend_state: None,
        });

        merge_current_settings_into_model_options(&mut providers, &mut models, &mut settings);
        merge_current_settings_into_model_options(&mut providers, &mut models, &mut settings);

        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].id, "openrouter");
        assert_eq!(providers[0].models.len(), 1);
        assert_eq!(models.len(), 1);
        assert_eq!(settings.as_ref().unwrap().provider, "openrouter");
    }

    #[test]
    fn settings_from_config_includes_embedding_and_show_sources() {
        let mut config = crate::state::ConfigState::default();
        config.provider = Some("openai".into());
        config.model = Some("gpt-4o".into());
        config.embedding_provider = Some("openai".into());
        config.embedding_model = Some("text-embedding-3-small".into());
        config.show_sources = false;

        let settings = settings_from_config(&config);
        assert_eq!(settings.provider, "openai");
        assert_eq!(settings.embedding_provider.as_deref(), Some("openai"));
        assert_eq!(settings.show_sources, Some(false));
    }

    #[test]
    fn empty_provider_is_a_placeholder_not_a_clear() {
        assert!(show_uncommitted_option(""));
        assert!(show_uncommitted_option("  "));
        assert!(!show_uncommitted_option("anthropic"));
        assert!(!should_put_live_settings("", ""));
        assert!(should_put_live_settings("anthropic", ""));
        assert!(!should_put_live_settings("", "claude"));

        let settings = settings_from_config(&crate::state::ConfigState::default());
        assert!(settings.provider.is_empty());
        assert!(!should_put_live_settings(
            &settings.provider,
            &settings.model
        ));
    }

    #[test]
    fn header_adopts_server_provider_without_opening_settings() {
        let server_provider = "openrouter";
        let server_model = "anthropic/claude-sonnet-4.6";
        assert!(should_adopt_server_settings(
            "",
            "",
            server_provider,
            server_model,
            false
        ));
        assert!(should_adopt_server_settings(
            "bedrock",
            "",
            server_provider,
            server_model,
            false
        ));
        assert!(!should_adopt_server_settings(
            "bedrock",
            "",
            server_provider,
            server_model,
            true
        ));
        assert!(!should_adopt_server_settings(
            server_provider,
            server_model,
            server_provider,
            server_model,
            false
        ));
        assert!(!should_adopt_server_settings("", "", "", "", false));
        assert!(!accept_header_selection(false, true, "", "bedrock"));
        assert!(!accept_header_selection(true, false, "", "bedrock"));
        assert!(!accept_header_selection(
            true,
            true,
            server_provider,
            server_provider
        ));
        assert!(accept_header_selection(
            true,
            true,
            server_provider,
            "bedrock"
        ));
    }
}
