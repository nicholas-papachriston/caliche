use super::helpers::*;
use super::{SlashCommandExecutor, SlashCommandResult};
use crate::core::state::InteractiveState;
use anyhow::Result;
use caliche::AgentService;
use caliche::llm::ProviderType;

#[cfg(feature = "local-inference-base")]
fn fit_status_label(status: caliche::llm::FitStatus) -> &'static str {
    match status {
        caliche::llm::FitStatus::FitsComfortably => "safe",
        caliche::llm::FitStatus::FitsTight => "tight",
        caliche::llm::FitStatus::ExceedsMemory => "exceeds memory",
        caliche::llm::FitStatus::Unknown => "unknown",
    }
}

#[cfg(feature = "local-inference-base")]
fn local_model_download_requested_help(args: &[String]) -> bool {
    args.iter()
        .skip(1)
        .any(|arg| matches!(arg.as_str(), "help" | "--help" | "-h"))
}

#[cfg(feature = "local-inference-base")]
fn format_local_model_download_status(
    repo: &str,
    revision: Option<&str>,
    model_dir: Option<&std::path::Path>,
    stored_at: String,
) -> String {
    let revision = revision
        .map(|revision| format!(" at revision {revision}"))
        .unwrap_or_default();
    let storage_note = model_dir
        .map(|dir| format!("\nRequested storage directory: {}", dir.display()))
        .unwrap_or_default();
    format!(
        "Downloaded local inference model {repo}{revision}.\nStored at: {stored_at}{storage_note}\nActive provider/model unchanged."
    )
}

#[cfg(feature = "local-inference-base")]
async fn handle_local_model_download(args: &[String]) -> Result<SlashCommandResult> {
    if local_model_download_requested_help(args) {
        return Ok(SlashCommandResult::StatusMessage(local_model_usage()));
    }

    let request = parse_local_model_download_args(args)?;
    let repo = request.repo.clone();
    let revision = request.revision.clone();
    let model_dir = request.model_dir.clone();
    let mut config = caliche::llm::LocalInferenceProvider::config_from_env(Some(repo.clone()))?;
    if let Some(backend) = request.backend {
        config = config.with_backend(backend);
    }
    config = config.with_download_target(repo.clone(), revision.clone());

    if let Some(model_dir) = request.model_dir {
        config = config.with_model_storage_dir(model_dir);
    }
    if let Some(hf_cache_dir) = request.hf_cache_dir {
        config = config.with_huggingface_cache_dir(hf_cache_dir);
    }

    let prepared = tokio::task::spawn_blocking(move || {
        caliche::llm::LocalInferenceProvider::prepare_artifacts(config)
    })
    .await??;

    let stored_at = prepared
        .source
        .local_path()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| format!("{:?}", prepared.source));

    Ok(SlashCommandResult::StatusMessage(
        format_local_model_download_status(
            &repo,
            revision.as_deref(),
            model_dir.as_deref(),
            stored_at,
        ),
    ))
}

#[cfg(feature = "local-inference-base")]
async fn handle_local_model_list(args: &[String]) -> Result<SlashCommandResult> {
    let tags = parse_local_model_list_tags(args);
    let tag_refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let specs: Vec<&_> = if tag_refs.is_empty() {
        caliche::llm::LocalInferenceProvider::catalog_models()
            .iter()
            .collect()
    } else {
        caliche::llm::LocalInferenceProvider::catalog_search(&tag_refs)
    };
    let options = caliche::llm::FitOptions::default();
    let fits = tokio::task::spawn_blocking(move || {
        caliche::llm::LocalInferenceProvider::fit_catalog_models(options)
    })
    .await??;
    let fit_by_id: std::collections::BTreeMap<_, _> = fits
        .into_iter()
        .map(|fit| (fit.model_id.clone(), fit))
        .collect();
    let mut items = Vec::new();
    let mut descriptions = Vec::new();
    for spec in specs {
        items.push(spec.id.to_string());
        let status = fit_by_id
            .get(spec.id)
            .map(|fit| fit_status_label(fit.status))
            .unwrap_or("unknown");
        descriptions.push(Some(format!(
            "{} [{}] ({status})",
            spec.display_name, spec.repo
        )));
    }
    Ok(SlashCommandResult::ShowList {
        title: "Local Inference Catalog".to_string(),
        items,
        descriptions,
    })
}

#[cfg(feature = "local-inference-base")]
fn format_local_model_fit_report(
    id: Option<String>,
    options: caliche::llm::FitOptions,
) -> Result<String> {
    if let Some(id) = id {
        let fit = caliche::llm::LocalInferenceProvider::fit_catalog_model(&id, options)?;
        Ok(format_model_fit_report(std::slice::from_ref(&fit)))
    } else {
        let fits = caliche::llm::LocalInferenceProvider::fit_catalog_models(options)?;
        Ok(format_model_fit_report(&fits))
    }
}

#[cfg(feature = "local-inference-base")]
async fn handle_local_model_fit(args: &[String]) -> Result<SlashCommandResult> {
    let options = caliche::llm::FitOptions::default();
    let id = args.get(1).cloned();
    let report =
        tokio::task::spawn_blocking(move || format_local_model_fit_report(id, options)).await??;
    Ok(SlashCommandResult::StatusMessage(report))
}

#[cfg(feature = "local-inference-base")]
fn handle_local_model_backends() -> SlashCommandResult {
    let backends = caliche::llm::LocalInferenceProvider::backend_names();
    SlashCommandResult::ShowList {
        title: "Local Inference Backends".to_string(),
        items: backends
            .iter()
            .map(|backend| (*backend).to_string())
            .collect(),
        descriptions: backends
            .iter()
            .map(|backend| {
                Some(
                    match *backend {
                        "candle" => "Candle safetensors backend",
                        "ort" => "ONNX Runtime backend",
                        "mlx" => "MLX backend for Apple Silicon",
                        "gguf" => "GGUF / llama.cpp",
                        _ => "Alias for gguf",
                    }
                    .to_string(),
                )
            })
            .collect(),
    }
}

impl SlashCommandExecutor {
    #[cfg(feature = "local-inference-base")]
    pub(crate) async fn handle_local_model(args: &[String]) -> Result<SlashCommandResult> {
        let Some(action) = args.first().map(String::as_str) else {
            return Ok(SlashCommandResult::StatusMessage(local_model_usage()));
        };

        match action {
            "download" => handle_local_model_download(args).await,
            "list" => handle_local_model_list(args).await,
            "fit" => handle_local_model_fit(args).await,
            "backends" | "backend" => Ok(handle_local_model_backends()),
            "help" | "--help" | "-h" => Ok(SlashCommandResult::StatusMessage(local_model_usage())),
            other => Ok(SlashCommandResult::Error(format!(
                "Unknown /local-model action '{other}'.\n{}",
                local_model_usage()
            ))),
        }
    }

    pub(crate) fn handle_subagent_provider(
        state: &mut InteractiveState,
        name: Option<&str>,
    ) -> Result<SlashCommandResult> {
        match name {
            None => {
                if supports_interactive_selection() {
                    let mut items: Vec<String> = crate::credentials::offered_providers()
                        .iter()
                        .map(|(provider, _, desc)| {
                            let current = if state.subagent_provider.as_deref() == Some(*provider) {
                                " [current]"
                            } else {
                                ""
                            };
                            format!("{provider} - {desc}{current}")
                        })
                        .collect();
                    items.push("clear - use main provider [current if unset]".to_string());

                    if let Some(index) = prompt_select_index("Select subagent provider", &items, 0)?
                    {
                        if index == items.len() - 1 {
                            state.subagent_provider = None;
                            return Ok(SlashCommandResult::StatusMessage(
                                "Subagent provider cleared; subagents will use main provider."
                                    .to_string(),
                            ));
                        }
                        state.subagent_provider =
                            Some(crate::credentials::offered_providers()[index].0.to_string());
                        return Ok(SlashCommandResult::StatusMessage(format!(
                            "Subagent provider set to: {} (persists across sessions)",
                            crate::credentials::offered_providers()[index].0
                        )));
                    }
                }

                Ok(SlashCommandResult::StatusMessage(format!(
                    "Subagent provider: {}\nUsage: /subagent-provider <provider> or /subagent-provider clear",
                    state
                        .subagent_provider
                        .as_deref()
                        .unwrap_or("(use main provider)")
                )))
            }
            Some("clear" | "") => {
                state.subagent_provider = None;
                Ok(SlashCommandResult::StatusMessage(
                    "Subagent provider cleared; subagents will use main provider.".to_string(),
                ))
            }
            Some(n) => {
                if let Err(err) = caliche::llm::ProviderType::parse_in_build(n) {
                    return Ok(SlashCommandResult::Error(err));
                }
                state.subagent_provider = Some(n.to_string());
                Ok(SlashCommandResult::StatusMessage(format!(
                    "Subagent provider set to: {} (persists across sessions)",
                    n
                )))
            }
        }
    }

    pub(crate) fn handle_subagent_model(
        state: &mut InteractiveState,
        name: Option<&str>,
    ) -> Result<SlashCommandResult> {
        match name {
            None => {
                if supports_interactive_selection() {
                    let provider = state
                        .subagent_provider
                        .as_deref()
                        .unwrap_or(state.token_budget.provider.as_str());
                    let recommended = crate::credentials::get_recommended_models(provider);

                    if !recommended.is_empty() {
                        let mut items: Vec<String> = recommended
                            .iter()
                            .map(|(model, desc)| {
                                let current = if state.subagent_model.as_deref() == Some(*model) {
                                    " [current]"
                                } else {
                                    ""
                                };
                                format!("{model} - {desc}{current}")
                            })
                            .collect();
                        items.push("clear - use main model [current if unset]".to_string());

                        if let Some(index) =
                            prompt_select_index("Select subagent model", &items, 0)?
                        {
                            if index == items.len() - 1 {
                                state.subagent_model = None;
                                return Ok(SlashCommandResult::StatusMessage(
                                    "Subagent model cleared; subagents will use main model."
                                        .to_string(),
                                ));
                            }
                            state.subagent_model = Some(recommended[index].0.to_string());
                            return Ok(SlashCommandResult::StatusMessage(format!(
                                "Subagent model set to: {} (persists across sessions)",
                                recommended[index].0
                            )));
                        }
                    }
                }

                Ok(SlashCommandResult::StatusMessage(format!(
                    "Subagent model: {}\nUsage: /subagent-model <model> or /subagent-model clear",
                    state
                        .subagent_model
                        .as_deref()
                        .unwrap_or("(use main model)")
                )))
            }
            Some("clear" | "") => {
                state.subagent_model = None;
                Ok(SlashCommandResult::StatusMessage(
                    "Subagent model cleared; subagents will use main model.".to_string(),
                ))
            }
            Some(n) => {
                state.subagent_model = Some(n.to_string());
                Ok(SlashCommandResult::StatusMessage(format!(
                    "Subagent model set to: {} (persists across sessions)",
                    n
                )))
            }
        }
    }

    pub(crate) async fn handle_models(
        service: &AgentService,
        provider: Option<&str>,
    ) -> Result<SlashCommandResult> {
        use caliche::llm::ProviderType;

        // Use provided provider or fall back to active provider from service
        let provider_str = if let Some(p) = provider {
            p
        } else {
            // Convert active provider to string for parsing
            let active_provider_str = service.provider().to_string();
            // We need to parse it directly since we can't return a reference to a temporary
            let provider_type: ProviderType = active_provider_str
                .parse()
                .map_err(|e| anyhow::anyhow!("Failed to parse active provider: {}", e))?;
            return Self::handle_models_for_provider(provider_type).await;
        };

        let provider_type: ProviderType = provider_str
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid provider: {}", e))?;

        Self::handle_models_for_provider(provider_type).await
    }

    pub(crate) async fn handle_models_for_provider(
        provider_type: ProviderType,
    ) -> Result<SlashCommandResult> {
        let provider_str = provider_type.to_string();
        let entry = caliche::llm::discover_provider_models(
            provider_type,
            &cli_model_catalog_options().await,
        )
        .await?;
        let items = entry
            .models
            .into_iter()
            .map(|model| model.id)
            .collect::<Vec<_>>();
        let descriptions = vec![None; items.len()];

        Ok(SlashCommandResult::ShowList {
            title: format!("Available Models ({})", provider_str),
            items,
            descriptions,
        })
    }
}
