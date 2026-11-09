//! Provider selection for `caliche --server` and `caliche doctor`.
//!
//! The rule lives in [`caliche_server::boot_provider`] so the standalone
//! server binary uses it too. This module adds the CLI's writes to the
//! durable `settings.json` record.

pub(crate) use caliche_server::boot_provider::{
    PersistedBoot, ServerBootProviderInput, apply_server_boot_provider, load_persisted_boot,
    select_server_boot_provider,
};
#[cfg(test)]
pub(crate) use caliche_server::boot_provider::{ServerBootProvider, merge_persisted_boot};
use caliche_server::durable_settings::DurableSettings;
use std::path::Path;

/// Provider and model replace the durable record. Sampling stays when the
/// live service has no value of its own.
#[must_use]
pub(crate) fn durable_settings_after_selection(
    existing: Option<&DurableSettings>,
    provider: &str,
    model: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> DurableSettings {
    DurableSettings {
        provider: provider.to_owned(),
        model: model.to_owned(),
        temperature: temperature.or_else(|| existing.and_then(|settings| settings.temperature)),
        max_tokens: max_tokens.or_else(|| existing.and_then(|settings| settings.max_tokens)),
    }
}

/// Write the provider and model the next process will boot.
///
/// Empty provider or model leaves the file unchanged. An existing temperature
/// or max token count stays when the caller passes `None`.
pub(crate) async fn save_selected_provider_model_at(
    path: &Path,
    provider: &str,
    model: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> anyhow::Result<()> {
    let provider = provider.trim();
    let model = model.trim();
    if provider.is_empty() || model.is_empty() {
        return Ok(());
    }
    let existing = caliche_server::durable_settings::load(path).await;
    let record = durable_settings_after_selection(
        existing.as_ref(),
        provider,
        model,
        temperature,
        max_tokens,
    );
    caliche_server::durable_settings::save(path, &record).await
}

/// Write `settings.json` beside `mcp.json`.
pub(crate) async fn save_selected_provider_model(
    provider: &str,
    model: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> anyhow::Result<()> {
    save_selected_provider_model_at(
        &caliche_server::durable_settings::default_path(),
        provider,
        model,
        temperature,
        max_tokens,
    )
    .await
}

/// Record `provider` and `model` as the selection the next launch starts
/// with: `settings.json`, and the SQLite settings when `database` is open.
/// Failures are logged; the session goes on.
pub(crate) async fn save_last_selection(
    database: Option<&crate::storage::Database>,
    provider: &str,
    model: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) {
    if let Some(db) = database {
        if let Err(e) = db.save_setting("last_provider", provider).await {
            tracing::warn!("Failed to save last_provider: {}", e);
        }
        if let Err(e) = db.save_setting("last_model", model).await {
            tracing::warn!("Failed to save last_model: {}", e);
        }
    }
    if let Err(e) = save_selected_provider_model(provider, model, temperature, max_tokens).await {
        tracing::warn!(error = %e, "failed to persist provider and model");
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ServerBootProviderInput, apply_server_boot_provider, durable_settings_after_selection,
        merge_persisted_boot, save_selected_provider_model_at, select_server_boot_provider,
    };
    use caliche::llm::ProviderType;
    use caliche_server::durable_settings::DurableSettings;

    fn choose<'a>(
        explicit: Option<&'a str>,
        explicit_model: bool,
        agent: Option<&'a str>,
        default_env: Option<&'a str>,
        last: Option<&'a str>,
        last_model: Option<&'a str>,
    ) -> super::ServerBootProvider {
        select_server_boot_provider(ServerBootProviderInput {
            explicit_provider: explicit,
            explicit_model,
            agent_provider: agent,
            default_provider_env: default_env,
            last_provider: last,
            last_model,
        })
    }

    #[test]
    fn last_wins_over_env_for_server_boot() {
        let choice = choose(
            None,
            false,
            Some("openai"),
            Some("anthropic"),
            Some("google"),
            Some("gemini-2.5-flash"),
        );
        assert_eq!(choice.source, "last");
        assert_eq!(choice.provider, "google");
    }

    #[test]
    fn flag_wins_over_last_and_env() {
        let choice = choose(
            Some(" openai "),
            false,
            Some("anthropic"),
            None,
            Some("google"),
            Some("gemini-2.5-flash"),
        );
        assert_eq!(choice.source, "flag");
        assert_eq!(choice.provider, "openai");
        let mixed_case = choose(Some("OpenAI"), false, None, None, None, None);
        assert_eq!(mixed_case.provider, "openai");
    }

    #[test]
    fn explicit_model_skips_last_and_uses_env() {
        let choice = choose(
            None,
            true,
            None,
            Some(" anthropic "),
            Some("google"),
            Some("gemini-2.5-flash"),
        );
        assert_eq!(choice.source, "env");
        assert_eq!(choice.provider, "anthropic");
    }

    #[test]
    fn invalid_last_falls_through_to_env() {
        let choice = choose(
            None,
            false,
            Some("openai"),
            None,
            Some("not-a-provider"),
            Some("some-model"),
        );
        assert_eq!(choice.source, "env");
        assert_eq!(choice.provider, "openai");
    }

    #[test]
    fn missing_last_model_falls_through_to_default() {
        let choice = choose(None, false, Some("  "), None, Some("google"), Some("  "));
        assert_eq!(choice.source, "default");
        assert_eq!(choice.provider, "openrouter");
    }

    #[test]
    fn agent_provider_beats_default_provider_env() {
        let choice = choose(None, false, Some("openai"), Some("anthropic"), None, None);
        assert_eq!(choice.source, "env");
        assert_eq!(choice.provider, "openai");
    }

    #[test]
    fn apply_last_sets_provider_and_model() {
        let choice = choose(
            None,
            false,
            Some("openai"),
            None,
            Some("ollama"),
            Some("llama3.2"),
        );
        let mut provider = None;
        let mut model = None;
        let used_last = apply_server_boot_provider(
            &choice,
            Some("ollama"),
            Some("llama3.2"),
            &mut provider,
            &mut model,
        )
        .expect("ollama parses");
        assert!(used_last);
        assert_eq!(provider, Some(ProviderType::Ollama));
        assert_eq!(model.as_deref(), Some("llama3.2"));
    }

    #[test]
    fn apply_default_leaves_provider_unset() {
        let choice = choose(None, false, None, None, None, None);
        let mut provider = None;
        let mut model = Some("keep".to_string());
        let used_last = apply_server_boot_provider(&choice, None, None, &mut provider, &mut model)
            .expect("default");
        assert!(!used_last);
        assert!(provider.is_none());
        assert_eq!(model.as_deref(), Some("keep"));
    }

    #[test]
    fn flag_keeps_saved_model_when_provider_matches() {
        let choice = choose(
            Some("anthropic"),
            false,
            Some("openai"),
            None,
            Some("anthropic"),
            Some("claude-sonnet-4-5-20250929"),
        );
        assert_eq!(choice.source, "flag");
        let mut provider = Some(ProviderType::Anthropic);
        let mut model = None;
        let used_last = apply_server_boot_provider(
            &choice,
            Some("anthropic"),
            Some("claude-sonnet-4-5-20250929"),
            &mut provider,
            &mut model,
        )
        .expect("anthropic parses");
        assert!(used_last);
        assert_eq!(provider, Some(ProviderType::Anthropic));
        assert_eq!(model.as_deref(), Some("claude-sonnet-4-5-20250929"));
    }

    #[test]
    fn flag_drops_saved_model_when_provider_differs() {
        let choice = choose(
            Some("openai"),
            false,
            None,
            None,
            Some("anthropic"),
            Some("claude-sonnet-4-5-20250929"),
        );
        let mut provider = Some(ProviderType::OpenAI);
        let mut model = None;
        let used_last = apply_server_boot_provider(
            &choice,
            Some("anthropic"),
            Some("claude-sonnet-4-5-20250929"),
            &mut provider,
            &mut model,
        )
        .expect("openai parses");
        assert!(!used_last);
        assert!(model.is_none());
    }

    #[test]
    fn explicit_model_is_not_replaced_by_saved_model() {
        let choice = choose(
            Some("anthropic"),
            true,
            None,
            None,
            Some("anthropic"),
            Some("claude-sonnet-4-5-20250929"),
        );
        let mut provider = Some(ProviderType::Anthropic);
        let mut model = Some("claude-haiku-4-5-20251001".to_string());
        let used_last = apply_server_boot_provider(
            &choice,
            Some("anthropic"),
            Some("claude-sonnet-4-5-20250929"),
            &mut provider,
            &mut model,
        )
        .expect("anthropic parses");
        assert!(!used_last);
        assert_eq!(model.as_deref(), Some("claude-haiku-4-5-20251001"));
    }

    #[test]
    fn merge_prefers_settings_file_over_sqlite() {
        let file = DurableSettings {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-5-20250929".to_string(),
            temperature: Some(0.5),
            max_tokens: Some(256),
        };
        let boot = merge_persisted_boot(Some(&file), Some("google"), Some("gemini-2.5-flash"));
        assert_eq!(boot.provider.as_deref(), Some("anthropic"));
        assert_eq!(boot.model.as_deref(), Some("claude-sonnet-4-5-20250929"));
        assert_eq!(boot.temperature, Some(0.5));
        assert_eq!(boot.max_tokens, Some(256));
    }

    #[test]
    fn selection_keeps_saved_sampling_when_the_service_has_none() {
        let existing = DurableSettings {
            provider: "anthropic".to_string(),
            model: "claude-haiku-4-5-20251001".to_string(),
            temperature: Some(0.37),
            max_tokens: Some(321),
        };
        let next = durable_settings_after_selection(
            Some(&existing),
            "openrouter",
            "anthropic/claude-sonnet-4.5",
            None,
            None,
        );
        assert_eq!(next.provider, "openrouter");
        assert_eq!(next.model, "anthropic/claude-sonnet-4.5");
        assert_eq!(next.temperature, Some(0.37));
        assert_eq!(next.max_tokens, Some(321));
    }

    #[tokio::test]
    async fn save_selected_provider_model_updates_the_boot_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("settings.json");
        let existing = DurableSettings {
            provider: "anthropic".to_string(),
            model: "claude-haiku-4-5-20251001".to_string(),
            temperature: Some(0.37),
            max_tokens: Some(321),
        };
        caliche_server::durable_settings::save(&path, &existing)
            .await
            .expect("seed");
        save_selected_provider_model_at(&path, "google", "gemini-2.5-flash", None, None)
            .await
            .expect("save");
        let loaded = caliche_server::durable_settings::load(&path)
            .await
            .expect("load");
        assert_eq!(loaded.provider, "google");
        assert_eq!(loaded.model, "gemini-2.5-flash");
        assert_eq!(loaded.temperature, Some(0.37));
        assert_eq!(loaded.max_tokens, Some(321));
    }

    #[test]
    fn merge_uses_sqlite_when_settings_file_is_absent() {
        let boot = merge_persisted_boot(None, Some("anthropic"), Some("claude-haiku-4-5-20251001"));
        assert_eq!(boot.provider.as_deref(), Some("anthropic"));
        assert_eq!(boot.model.as_deref(), Some("claude-haiku-4-5-20251001"));
        assert_eq!(boot.temperature, None);
    }
}
