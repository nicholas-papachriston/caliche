use super::helpers::*;
use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use crate::cost::format_tokens;
use anyhow::Result;
use caliche::AgentService;
use caliche::agent::service::ProviderFactory;
use caliche::llm::ProviderType;

/// Switch the session to `provider` with `model`, or with the provider's
/// default model when `model` is `None`.
///
/// On failure the session keeps its provider and model, and the error is one
/// line: the provider, what is missing, and the command that fixes it.
pub(crate) async fn switch_session_provider(
    state: &mut InteractiveState,
    service: &mut AgentService,
    provider: ProviderType,
    model: Option<String>,
) -> std::result::Result<(), String> {
    let current = format!("{}/{}", service.provider(), service.model());
    let switched = async {
        let model = match model {
            Some(model) => model,
            None => {
                caliche::agent::service::resolve_model_for_provider_with_discovery(provider, None)
                    .await?
            }
        };
        let mode = state.interaction_mode;
        apply_runtime_settings(state, service, Some(provider), Some(model), mode).await
    }
    .await;
    match switched {
        Ok(()) => {
            sync_runtime_metadata(state, service);
            Ok(())
        }
        Err(err) => Err(switch_failure_line(provider, &err, &current)),
    }
}

fn switch_failure_line(provider: ProviderType, err: &anyhow::Error, current: &str) -> String {
    let detail = anyhow::anyhow!("{err:#}");
    if caliche::llm::unconfigured::is_missing_provider_credentials(&detail) {
        let missing = crate::credentials::missing_key_line(&provider.to_string());
        return format!("{missing} Still using {current}.");
    }
    format!("Could not switch to {provider}: {err:#}. Still using {current}.")
}

const EMBEDDING_USAGE: &str =
    "Usage: /embedding <provider>[/<model>] or /embedding <provider> <model> or /embedding clear";

enum EmbeddingSpecChoice {
    Unchanged,
    ShowCurrent,
    Value(String),
}

impl SlashCommandExecutor {
    pub(crate) fn handle_embeddings() -> Result<SlashCommandResult> {
        Ok(SlashCommandResult::ShowList {
            title: "Available Embedding Providers".to_string(),
            items: crate::credentials::offered_embedding_providers()
                .iter()
                .map(|(id, _)| (*id).to_string())
                .collect(),
            descriptions: crate::credentials::offered_embedding_providers()
                .iter()
                .map(|(_, description)| Some((*description).to_string()))
                .collect(),
        })
    }

    pub(crate) async fn handle_embedding(
        state: &mut InteractiveState,
        service: &mut AgentService,
        spec: Option<&str>,
    ) -> Result<SlashCommandResult> {
        match Self::resolve_embedding_spec(spec)? {
            EmbeddingSpecChoice::Unchanged => Ok(SlashCommandResult::StatusMessage(
                "Embedding provider unchanged.".to_string(),
            )),
            EmbeddingSpecChoice::ShowCurrent => Ok(Self::current_embedding_status(state, service)),
            EmbeddingSpecChoice::Value(spec) => {
                Self::apply_resolved_embedding_spec(state, service, spec.trim()).await
            }
        }
    }

    fn resolve_embedding_spec(spec: Option<&str>) -> Result<EmbeddingSpecChoice> {
        if let Some(value) = spec {
            return Ok(EmbeddingSpecChoice::Value(value.to_string()));
        }
        if !supports_interactive_selection() {
            return Ok(EmbeddingSpecChoice::ShowCurrent);
        }
        match Self::prompt_embedding_spec()? {
            Some(value) => Ok(EmbeddingSpecChoice::Value(value)),
            None => Ok(EmbeddingSpecChoice::Unchanged),
        }
    }

    fn prompt_embedding_spec() -> Result<Option<String>> {
        let labels: Vec<String> = crate::credentials::offered_embedding_providers()
            .iter()
            .map(|(id, description)| format!("{id} - {description}"))
            .collect();
        let Some(index) = prompt_select_index("Select embedding provider", &labels, 0)? else {
            return Ok(None);
        };
        Ok(Some(
            crate::credentials::offered_embedding_providers()
                .get(index)
                .map_or("openai", |(id, _)| *id)
                .to_string(),
        ))
    }

    fn current_embedding_status(
        state: &InteractiveState,
        service: &AgentService,
    ) -> SlashCommandResult {
        let provider_label = service
            .embedding_provider_id()
            .map(str::to_owned)
            .or_else(|| {
                service
                    .embedding_provider()
                    .map(|emb| emb.name().to_string())
            })
            .or_else(|| state.embedding_provider.clone());
        let model_label = service
            .embedding_model_id()
            .map(str::to_owned)
            .or_else(|| state.embedding_model.clone())
            .unwrap_or_else(|| "(default)".to_string());
        let current = if let Some(provider) = provider_label {
            format!("Embedding provider: {provider} / {model_label}\n{EMBEDDING_USAGE}")
        } else {
            format!("No embedding provider configured\n{EMBEDDING_USAGE}")
        };
        SlashCommandResult::StatusMessage(current)
    }

    async fn apply_resolved_embedding_spec(
        state: &mut InteractiveState,
        service: &mut AgentService,
        spec_str: &str,
    ) -> Result<SlashCommandResult> {
        if spec_str.is_empty() {
            return Ok(Self::current_embedding_status(state, service));
        }
        if spec_str.eq_ignore_ascii_case("clear") {
            state.embedding_provider = None;
            state.embedding_model = None;
            service.set_embedding_provider(None);
            return Ok(SlashCommandResult::StatusMessage(
                "Embedding provider cleared.".to_string(),
            ));
        }
        Self::apply_embedding_spec(state, service, spec_str).await
    }

    fn parse_provider_model_spec(spec_str: &str) -> (&str, Option<String>) {
        // Whitespace form is checked first so routed model IDs like
        // `openai/text-embedding-3-small` remain intact.
        if let Some((provider, model)) = spec_str.split_once(char::is_whitespace) {
            let model = model.trim();
            return (
                provider.trim(),
                if model.is_empty() {
                    None
                } else {
                    Some(model.to_string())
                },
            );
        }
        if let Some(pos) = spec_str.find('/') {
            let (provider, model) = spec_str.split_at(pos);
            return (provider.trim(), Some(model[1..].trim().to_string()));
        }
        (spec_str.trim(), None)
    }

    async fn apply_embedding_spec(
        state: &mut InteractiveState,
        service: &mut AgentService,
        spec_str: &str,
    ) -> Result<SlashCommandResult> {
        let (provider_str, model_opt) = Self::parse_provider_model_spec(spec_str);
        let provider_type = match ProviderType::parse_in_build(provider_str) {
            Ok(provider) => provider,
            Err(err) => return Ok(SlashCommandResult::Error(err)),
        };
        let model = match model_opt {
            Some(model) => model,
            None => match caliche::agent::service::default_embedding_model(provider_type) {
                Some(model) => model,
                None => {
                    return Ok(SlashCommandResult::Error(format!(
                        "Provider '{provider_str}' does not support embeddings.\nSupported: {}",
                        crate::credentials::offered_embedding_providers()
                            .iter()
                            .map(|(name, _)| *name)
                            .collect::<Vec<_>>()
                            .join(", ")
                    )));
                }
            },
        };
        let cred_provider = cli_model_catalog_options().await.credential_provider;
        let embedding_provider =
            ProviderFactory::create_embedding_provider(provider_type, model.clone(), cred_provider)
                .await
                .map_err(|err| anyhow::anyhow!("Failed to create embedding provider: {err}"))?;
        if embedding_provider.is_none() {
            return Ok(SlashCommandResult::Error(format!(
                "Provider '{provider_str}' does not support embeddings"
            )));
        }
        service.set_embedding_selection(
            embedding_provider,
            Some(provider_type.to_string()),
            Some(model.clone()),
        );
        state.embedding_provider = Some(provider_type.to_string());
        state.embedding_model = Some(model.clone());
        Ok(SlashCommandResult::StatusMessage(format!(
            "Switched to embedding provider: {} with model: {}{}",
            provider_str,
            model,
            if service.has_workspace_rag() {
                "\n\nNote: Existing workspace embeddings were created with the previous provider.\nThey may not be compatible with the new embedding model.\nRun /reindex to rebuild the workspace index."
            } else {
                "\n\nRun /reindex to index the current workspace with this provider."
            }
        )))
    }

    pub(crate) async fn handle_image_generators(
        provider: Option<&str>,
    ) -> Result<SlashCommandResult> {
        use caliche::llm::ProviderType;

        let provider = provider.map(str::trim).filter(|value| !value.is_empty());

        match provider {
            Some("openai") => {
                Self::handle_image_generators_for_provider(ProviderType::OpenAI).await
            }
            Some("google") | Some("gemini") => {
                Self::handle_image_generators_for_provider(ProviderType::Google).await
            }
            Some("openrouter") => {
                Self::handle_image_generators_for_provider(ProviderType::OpenRouter).await
            }
            Some("cloudflare") | Some("cloudflare-ai-gateway") | Some("cf-aig") => {
                Self::handle_image_generators_for_provider(ProviderType::CloudflareAIGateway).await
            }
            Some(other) => Ok(SlashCommandResult::Error(format!(
                "Provider '{other}' does not support image generation.\nSupported: openai, google, openrouter, cloudflare-ai-gateway"
            ))),
            None => {
                let items = vec![
                    "openai".to_string(),
                    "google".to_string(),
                    "openrouter".to_string(),
                    "cloudflare-ai-gateway".to_string(),
                ];
                let descriptions = vec![
                    Some("OpenAI image models, discovered via /v1/models".to_string()),
                    Some("Google Gemini image models, discovered via v1beta/models".to_string()),
                    Some("OpenRouter image models, discovered via /api/v1/models".to_string()),
                    Some("Cloudflare AI Gateway routed image models".to_string()),
                ];
                Ok(SlashCommandResult::ShowList {
                    title: "Available Image Generation Providers".to_string(),
                    items,
                    descriptions,
                })
            }
        }
    }

    pub(crate) async fn handle_image_generators_for_provider(
        provider_type: caliche::llm::ProviderType,
    ) -> Result<SlashCommandResult> {
        let models = caliche::llm::discover_image_generation_models(
            provider_type,
            cli_model_catalog_options().await.credential_provider,
        )
        .await?;
        let descriptions = models
            .iter()
            .map(|model| Some(image_model_description(provider_type, model).to_string()))
            .collect::<Vec<_>>();
        Ok(SlashCommandResult::ShowList {
            title: format!("Available Image Generation Models ({provider_type})"),
            items: models,
            descriptions,
        })
    }

    pub(crate) async fn handle_image_generation(
        state: &mut InteractiveState,
        service: &mut AgentService,
        spec: Option<&str>,
    ) -> Result<SlashCommandResult> {
        use caliche::agent::service::ProviderFactory;
        use caliche::llm::{ProviderType, resolve_image_generation_model};
        use caliche::traits::credentials::EnvironmentCredentialProvider;
        use std::sync::Arc;

        let spec = spec.map(str::trim).filter(|value| !value.is_empty());
        let Some(spec) = spec else {
            let configured = service
                .image_generation_provider()
                .map(|provider| format!("{} / {}", provider.name(), provider.model()))
                .or_else(|| {
                    state.image_generation_provider.as_ref().map(|provider| {
                        format!(
                            "{} / {}",
                            provider,
                            state
                                .image_generation_model
                                .as_deref()
                                .unwrap_or("(default)")
                        )
                    })
                })
                .unwrap_or_else(|| "(not configured)".to_string());
            return Ok(SlashCommandResult::StatusMessage(format!(
                "Image generation: {configured}\nUsage: /image-generation <openai|google|openrouter|cloudflare-ai-gateway>[/<model>] or /image-generation <provider> <model> or /image-generation clear"
            )));
        };

        if spec.eq_ignore_ascii_case("clear") {
            state.image_generation_provider = None;
            state.image_generation_model = None;
            service.set_image_generation_provider(None);
            return Ok(SlashCommandResult::StatusMessage(
                "Image generation cleared; generate_image tool disabled.".to_string(),
            ));
        }

        let (provider_str, model_opt) =
            if let Some((provider, model)) = spec.split_once(char::is_whitespace) {
                let model = model.trim();
                (
                    provider.trim(),
                    if model.is_empty() {
                        None
                    } else {
                        Some(model.to_string())
                    },
                )
            } else if let Some(pos) = spec.find('/') {
                let (provider, model) = spec.split_at(pos);
                (provider.trim(), Some(model[1..].trim().to_string()))
            } else {
                (spec.trim(), None)
            };
        let provider_type: ProviderType = provider_str
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid image generation provider: {}", e))?;
        if !matches!(
            provider_type,
            ProviderType::OpenAI
                | ProviderType::Google
                | ProviderType::OpenRouter
                | ProviderType::CloudflareAIGateway
        ) {
            return Ok(SlashCommandResult::Error(format!(
                "Provider '{}' does not support image generation.\nSupported: openai, google, openrouter, cloudflare-ai-gateway",
                provider_str
            )));
        }
        let model =
            resolve_image_generation_model(provider_type, model_opt.clone()).ok_or_else(|| {
                anyhow::anyhow!("No default image generation model for {provider_str}")
            })?;

        if service.artifact_store().is_none() {
            service.set_artifact_store(Some(Arc::new(caliche::ArtifactStore::for_workspace(
                &state.working_directory,
                state.artifact_dir.as_deref(),
            ))));
        }

        let cred_provider = Arc::new(EnvironmentCredentialProvider::new());
        let provider = ProviderFactory::create_image_generation_provider(
            provider_type,
            Some(model.clone()),
            cred_provider,
        )
        .await?
        .ok_or_else(|| anyhow::anyhow!("{provider_str} does not support image generation"))?;
        service.set_image_generation_provider(Some(provider));
        state.image_generation_provider = Some(provider_type.to_string());
        state.image_generation_model = Some(model.clone());

        Ok(SlashCommandResult::StatusMessage(format!(
            "Image generation set to: {} with model: {}\nThe generate_image tool is now available and saves outputs under {}.",
            provider_type,
            model,
            service
                .artifact_store()
                .map(|store| store.root().display().to_string())
                .unwrap_or_else(|| ".caliche/artifacts".to_string())
        )))
    }

    pub(crate) fn handle_artifact_dir(
        state: &mut InteractiveState,
        service: &mut AgentService,
        path: Option<&std::path::Path>,
    ) -> Result<SlashCommandResult> {
        match path {
            None => {
                let current = service
                    .artifact_store()
                    .map(|store| store.root().display().to_string())
                    .unwrap_or_else(|| caliche::artifacts::DEFAULT_ARTIFACT_DIR.to_string());
                Ok(SlashCommandResult::StatusMessage(format!(
                    "Artifact directory: {current}\nUsage: /artifact-dir <path> or /artifact-dir clear"
                )))
            }
            Some(path) if path.to_str().is_some_and(|value| value == "clear") => {
                state.artifact_dir = None;
                let store = caliche::ArtifactStore::for_workspace(&state.working_directory, None);
                service.set_artifact_store(Some(std::sync::Arc::new(store)));
                Ok(SlashCommandResult::StatusMessage(format!(
                    "Artifact directory reset to {}",
                    service
                        .artifact_store()
                        .map(|store| store.root().display().to_string())
                        .unwrap_or_else(|| caliche::artifacts::DEFAULT_ARTIFACT_DIR.to_string())
                )))
            }
            Some(path) => {
                state.artifact_dir = Some(path.to_path_buf());
                let store = caliche::ArtifactStore::for_workspace(
                    &state.working_directory,
                    state.artifact_dir.as_deref(),
                );
                service.set_artifact_store(Some(std::sync::Arc::new(store)));
                Ok(SlashCommandResult::StatusMessage(format!(
                    "Artifact directory set to: {} (persists across sessions)",
                    service
                        .artifact_store()
                        .map(|store| store.root().display().to_string())
                        .unwrap_or_else(|| path.display().to_string())
                )))
            }
        }
    }

    pub(crate) async fn handle_providers() -> Result<SlashCommandResult> {
        let catalog =
            caliche::llm::discover_provider_catalog(&cli_model_catalog_options().await).await?;
        let items = catalog
            .iter()
            .map(|provider| provider.id.clone())
            .collect::<Vec<_>>();
        let descriptions = catalog
            .iter()
            .map(|provider| {
                Some(format!(
                    "{} ({} model{})",
                    provider.display_name,
                    provider.models.len(),
                    if provider.models.len() == 1 { "" } else { "s" }
                ))
            })
            .collect::<Vec<_>>();

        Ok(SlashCommandResult::ShowList {
            title: "Configured Providers".to_string(),
            items,
            descriptions,
        })
    }

    pub(crate) async fn handle_provider(
        state: &mut InteractiveState,
        service: &mut AgentService,
        name: Option<&str>,
    ) -> Result<SlashCommandResult> {
        let provider_name_owned = match name {
            Some(n) => n.to_string(),
            None => {
                if !supports_interactive_selection() {
                    return Ok(SlashCommandResult::StatusMessage(format!(
                        "Current provider: {}\nUsage: /provider <provider_name>",
                        service.provider()
                    )));
                }

                let current_provider = service.provider().to_string();
                let catalog =
                    caliche::llm::discover_provider_catalog(&cli_model_catalog_options().await)
                        .await?;
                if catalog.is_empty() {
                    return Ok(SlashCommandResult::StatusMessage(
                        "No configured providers discovered. Run `caliche config setup` first."
                            .to_string(),
                    ));
                }
                let mut default_index = 0usize;
                let items: Vec<String> = catalog
                    .iter()
                    .enumerate()
                    .map(|(idx, provider)| {
                        if provider.id == current_provider {
                            default_index = idx;
                        }
                        let current_marker = if provider.id == current_provider {
                            " [current]"
                        } else {
                            ""
                        };
                        format!(
                            "{} - {} ({} model{}){}",
                            provider.id,
                            provider.display_name,
                            provider.models.len(),
                            if provider.models.len() == 1 { "" } else { "s" },
                            current_marker
                        )
                    })
                    .collect();

                let Some(index) = prompt_select_index("Select provider", &items, default_index)?
                else {
                    return Ok(SlashCommandResult::StatusMessage(format!(
                        "Provider unchanged: {}",
                        current_provider
                    )));
                };

                catalog[index].id.clone()
            }
        };
        let provider_name = provider_name_owned.as_str();

        let new_provider_type = match ProviderType::parse_in_build(provider_name) {
            Ok(provider) => provider,
            Err(err) => return Ok(SlashCommandResult::Error(err)),
        };
        let previous_model = service.model().to_string();
        let model = if new_provider_type == service.provider() {
            Some(previous_model.clone())
        } else {
            None
        };
        if let Err(line) = switch_session_provider(state, service, new_provider_type, model).await {
            return Ok(SlashCommandResult::Error(line));
        }
        let new_provider = service.provider().to_string();
        let new_model = service.model().to_string();
        let model_note = if new_model != previous_model {
            format!(" Model resolved to {}.", new_model)
        } else {
            String::new()
        };

        Ok(SlashCommandResult::StatusMessage(format!(
            "Switched to provider: {} with model: {}. Preserved {} messages and active mode.{}",
            new_provider,
            new_model,
            state.messages.len(),
            model_note
        )))
    }

    pub(crate) async fn handle_model(
        state: &mut InteractiveState,
        service: &mut AgentService,
        name: Option<&str>,
    ) -> Result<SlashCommandResult> {
        let model_name_owned = match name {
            Some(n) => n.to_string(),
            None => {
                if !supports_interactive_selection() {
                    return Ok(SlashCommandResult::StatusMessage(format!(
                        "Current model: {}\nUsage: /model <model_name>",
                        service.model()
                    )));
                }

                let models_result = Self::handle_models_for_provider(service.provider()).await?;
                let (items, descriptions) = match models_result {
                    SlashCommandResult::ShowList {
                        items,
                        descriptions,
                        ..
                    } => (items, descriptions),
                    _ => (Vec::new(), Vec::new()),
                };
                if items.is_empty() {
                    return Ok(SlashCommandResult::StatusMessage(format!(
                        "No models available for provider '{}'",
                        service.provider()
                    )));
                }

                let current_model = service.model().to_string();
                let mut default_index = 0usize;
                let display_items: Vec<String> = items
                    .iter()
                    .enumerate()
                    .map(|(idx, model)| {
                        if model == &current_model {
                            default_index = idx;
                        }
                        let desc = descriptions
                            .get(idx)
                            .and_then(|d| d.as_deref())
                            .unwrap_or("");
                        let suffix = if model == &current_model {
                            " [current]"
                        } else {
                            ""
                        };
                        if desc.is_empty() {
                            format!("{model}{suffix}")
                        } else {
                            format!("{model} - {desc}{suffix}")
                        }
                    })
                    .collect();

                let Some(index) =
                    prompt_select_index("Select model", &display_items, default_index)?
                else {
                    return Ok(SlashCommandResult::StatusMessage(format!(
                        "Model unchanged: {}",
                        current_model
                    )));
                };
                items[index].clone()
            }
        };
        let model_name = model_name_owned.as_str();
        apply_runtime_settings(
            state,
            service,
            None,
            Some(model_name.to_string()),
            state.interaction_mode,
        )
        .await?;
        sync_runtime_metadata(state, service);
        let new_model = service.model().to_string();

        Ok(SlashCommandResult::StatusMessage(format!(
            "Switched to model: {}. Preserved {} messages and active mode.",
            new_model,
            state.messages.len()
        )))
    }

    pub(crate) fn handle_context_window(
        state: &mut InteractiveState,
        service: &mut AgentService,
        tokens: Option<&str>,
    ) -> Result<SlashCommandResult> {
        let Some(tokens) = tokens.map(str::trim).filter(|value| !value.is_empty()) else {
            let current = service
                .context_window()
                .and_then(|tokens| u64::try_from(tokens).ok())
                .unwrap_or(state.token_budget.max_tokens);
            return Ok(SlashCommandResult::StatusMessage(format!(
                "Current context window: {}\nUsage: /context-window <tokens>",
                format_tokens(current)
            )));
        };

        let context_window = tokens.parse::<usize>().map_err(|e| {
            anyhow::anyhow!(
                "invalid context window '{tokens}': expected a positive token count: {e}"
            )
        })?;
        let section_before = service.project_context_section().map(str::to_owned);
        service.set_context_window(context_window)?;
        state.context_window_override = Some(context_window);
        follow_project_section(state, section_before, service);
        sync_runtime_metadata(state, service);

        Ok(SlashCommandResult::StatusMessage(format!(
            "Context window set to {}.",
            format_tokens(u64::try_from(context_window)?)
        )))
    }
}
