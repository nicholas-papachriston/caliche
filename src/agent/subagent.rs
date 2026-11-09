//! Subagent model and provider selection.

use crate::llm::ProviderType;
use std::str::FromStr;
use std::sync::{Arc, RwLock};

/// User or config override for the model a subagent should run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubagentModelSelection {
    /// Provider override. When unset, the profile spec or the parent provider is used.
    pub provider: Option<ProviderType>,
    /// Model override. May be a bare id or `provider/model`.
    pub model: Option<String>,
}

/// Interior-mutable subagent model choice shared by spawners and slash commands.
#[derive(Debug, Clone, Default)]
pub struct SharedSubagentSelection {
    inner: Arc<RwLock<SubagentModelSelection>>,
}

impl SharedSubagentSelection {
    /// Wrap a selection so spawners and hosts can update it later.
    #[must_use]
    pub fn new(selection: SubagentModelSelection) -> Self {
        Self {
            inner: Arc::new(RwLock::new(selection)),
        }
    }

    /// Read the current selection.
    #[must_use]
    pub fn get(&self) -> SubagentModelSelection {
        self.inner
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Replace the current selection.
    pub fn set(&self, selection: SubagentModelSelection) {
        *self
            .inner
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = selection;
    }
}

impl SubagentModelSelection {
    /// Build a selection from optional provider and model strings.
    #[must_use]
    pub fn from_parts(provider: Option<&str>, model: Option<&str>) -> Self {
        Self {
            provider: provider.and_then(|value| ProviderType::from_str(value.trim()).ok()),
            model: nonempty(model),
        }
    }

    /// Resolve the provider and model a spawned subagent should use.
    ///
    /// Precedence: this selection, then the profile `model:` spec, then the parent.
    #[must_use]
    pub fn resolve(
        &self,
        profile_model: Option<&str>,
        fallback_provider: ProviderType,
        fallback_model: &str,
    ) -> (ProviderType, String) {
        let parsed_override = self.model.as_deref().map(parse_model_spec);
        let parsed_profile = profile_model.map(parse_model_spec);

        let model = parsed_override
            .as_ref()
            .map(|parsed| parsed.1.clone())
            .or_else(|| parsed_profile.as_ref().map(|parsed| parsed.1.clone()))
            .unwrap_or_else(|| fallback_model.to_owned());

        let provider = self
            .provider
            .or_else(|| parsed_override.and_then(|parsed| parsed.0))
            .or_else(|| parsed_profile.and_then(|parsed| parsed.0))
            .unwrap_or(fallback_provider);

        (provider, model)
    }
}

/// Split `provider/model` when the left side is a known provider name.
#[must_use]
pub fn parse_model_spec(spec: &str) -> (Option<ProviderType>, String) {
    let spec = spec.trim();
    if spec.is_empty() {
        return (None, String::new());
    }
    if let Some((left, right)) = spec.split_once('/') {
        if !right.is_empty()
            && let Ok(provider) = ProviderType::from_str(left)
        {
            return (Some(provider), right.to_owned());
        }
    }
    (None, spec.to_owned())
}

fn nonempty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::{SharedSubagentSelection, SubagentModelSelection, parse_model_spec};
    use crate::llm::ProviderType;

    #[test]
    fn parse_model_spec_splits_known_provider() {
        let (provider, model) = parse_model_spec("anthropic/claude-sonnet-4-5");
        assert_eq!(provider, Some(ProviderType::Anthropic));
        assert_eq!(model, "claude-sonnet-4-5");
    }

    #[test]
    fn parse_model_spec_keeps_bare_id() {
        let (provider, model) = parse_model_spec("gpt-4o");
        assert_eq!(provider, None);
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn resolve_prefers_selection_over_profile_and_parent() {
        let selection = SubagentModelSelection::from_parts(Some("anthropic"), Some("claude-haiku"));
        let (provider, model) = selection.resolve(
            Some("openai/gpt-4o"),
            ProviderType::OpenRouter,
            "fallback-model",
        );
        assert_eq!(provider, ProviderType::Anthropic);
        assert_eq!(model, "claude-haiku");
    }

    #[test]
    fn resolve_uses_profile_provider_model_when_selection_empty() {
        let selection = SubagentModelSelection::default();
        let (provider, model) = selection.resolve(
            Some("openai/gpt-4o-mini"),
            ProviderType::Anthropic,
            "parent",
        );
        assert_eq!(provider, ProviderType::OpenAI);
        assert_eq!(model, "gpt-4o-mini");
    }

    #[test]
    fn resolve_falls_back_to_parent() {
        let selection = SubagentModelSelection::default();
        let (provider, model) = selection.resolve(None, ProviderType::OpenRouter, "parent-model");
        assert_eq!(provider, ProviderType::OpenRouter);
        assert_eq!(model, "parent-model");
    }

    #[test]
    fn shared_selection_updates_are_visible() {
        let shared = SharedSubagentSelection::new(SubagentModelSelection::default());
        shared.set(SubagentModelSelection::from_parts(
            Some("anthropic"),
            Some("claude-haiku"),
        ));
        let current = shared.get();
        assert_eq!(current.provider, Some(ProviderType::Anthropic));
        assert_eq!(current.model.as_deref(), Some("claude-haiku"));
    }
}
