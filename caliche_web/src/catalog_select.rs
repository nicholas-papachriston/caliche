//! Model ids offered in the header and settings pickers.
//!
//! OpenRouter-style routing aliases (`:batch`, `:nitro`, and the same family)
//! stay out of the list. The committed id is still shown when settings already
//! use one, so the picker matches `settings.get`.

use crate::api::ModelInfo;

const HIDDEN_MODEL_SUFFIXES: &[&str] = &[
    "batch", "nitro", "floor", "exacto", "free", "extended", "online",
];

/// True when `id` is a model the operator can pick, not a routing alias.
#[must_use]
pub fn is_selectable_model_id(id: &str) -> bool {
    let Some((_, suffix)) = id.rsplit_once(':') else {
        return true;
    };
    let suffix = suffix.to_ascii_lowercase();
    !HIDDEN_MODEL_SUFFIXES.iter().any(|hidden| suffix == *hidden)
}

/// Models for one provider, narrowed by `query`, with `selected_id` kept visible.
#[must_use]
pub fn models_for_picker<'a>(
    models: &'a [ModelInfo],
    provider_id: &str,
    query: &str,
    selected_id: Option<&str>,
) -> Vec<&'a ModelInfo> {
    let query = query.trim().to_ascii_lowercase();
    let mut picked: Vec<&ModelInfo> = models
        .iter()
        .filter(|model| {
            provider_matches(provider_id, &model.provider_id)
                && is_selectable_model_id(&model.id)
                && query_matches(&query, model)
        })
        .collect();
    if let Some(selected_id) = selected_id.filter(|id| !id.is_empty()) {
        let already = picked.iter().any(|model| model.id == selected_id);
        if !already
            && let Some(selected) = models.iter().find(|model| {
                model.id == selected_id && provider_matches(provider_id, &model.provider_id)
            })
        {
            picked.insert(0, selected);
        }
    }
    picked
}

/// First pickable model for `provider_id`, skipping routing aliases.
#[must_use]
pub fn first_selectable_model_id(models: &[ModelInfo], provider_id: &str) -> Option<String> {
    models
        .iter()
        .find(|model| {
            provider_matches(provider_id, &model.provider_id) && is_selectable_model_id(&model.id)
        })
        .map(|model| model.id.clone())
}

fn provider_matches(selected: &str, model_provider: &str) -> bool {
    selected.is_empty() || selected == model_provider
}

fn query_matches(query: &str, model: &ModelInfo) -> bool {
    if query.is_empty() {
        return true;
    }
    model.id.to_ascii_lowercase().contains(query) || model.name.to_ascii_lowercase().contains(query)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(id: &str, provider: &str) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            name: id.to_string(),
            provider_id: provider.to_string(),
            provider_display_name: provider.to_string(),
            description: None,
            context_length: None,
        }
    }

    #[test]
    fn batch_and_routing_suffixes_are_hidden() {
        assert!(is_selectable_model_id("anthropic/claude-sonnet-4"));
        assert!(is_selectable_model_id("amazon.titan-embed-text-v2:0"));
        assert!(!is_selectable_model_id("openai/gpt-4o:batch"));
        assert!(!is_selectable_model_id("openai/gpt-4o:nitro"));
        assert!(!is_selectable_model_id("openai/gpt-4o:floor"));
        assert!(!is_selectable_model_id("openai/gpt-4o:exacto"));
        assert!(!is_selectable_model_id("openai/gpt-4o:free"));
        assert!(!is_selectable_model_id("openai/gpt-4o:extended"));
        assert!(!is_selectable_model_id("openai/gpt-4o:online"));
    }

    #[test]
    fn routing_suffixes_are_hidden_on_a_non_openrouter_catalog() {
        for suffix in [
            "batch", "nitro", "floor", "exacto", "free", "extended", "online",
        ] {
            assert!(!is_selectable_model_id(&format!(
                "amazon.nova-pro-v1:{suffix}"
            )));
        }
        assert!(is_selectable_model_id("amazon.nova-pro-v1:0"));
        let models = vec![
            model("amazon.nova-pro-v1:0", "bedrock"),
            model("amazon.nova-pro-v1:batch", "bedrock"),
            model("anthropic.claude-sonnet:online", "bedrock"),
            model("meta.llama3:free", "bedrock"),
        ];
        let picked = models_for_picker(&models, "bedrock", "", None);
        let ids: Vec<&str> = picked.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(ids, vec!["amazon.nova-pro-v1:0"]);
    }

    #[test]
    fn picker_keeps_the_committed_alias_and_filters_the_rest() {
        let models = vec![
            model("openai/gpt-4o", "openrouter"),
            model("openai/gpt-4o:batch", "openrouter"),
            model("anthropic/claude", "anthropic"),
        ];
        let picked = models_for_picker(&models, "openrouter", "", Some("openai/gpt-4o:batch"));
        let ids: Vec<&str> = picked.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(ids, vec!["openai/gpt-4o:batch", "openai/gpt-4o"]);
    }

    #[test]
    fn query_narrows_the_list() {
        let models = vec![
            model("openai/gpt-4o", "openrouter"),
            model("anthropic/claude-haiku", "openrouter"),
        ];
        let picked = models_for_picker(&models, "openrouter", "haiku", None);
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].id, "anthropic/claude-haiku");
    }

    #[test]
    fn first_selectable_skips_batch() {
        let models = vec![
            model("openai/gpt-4o:batch", "openrouter"),
            model("openai/gpt-4o", "openrouter"),
        ];
        assert_eq!(
            first_selectable_model_id(&models, "openrouter").as_deref(),
            Some("openai/gpt-4o")
        );
    }
}
