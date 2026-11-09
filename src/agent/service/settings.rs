//! Agent settings for snapshot and apply (CLI/API sync).

use crate::agent::modes::InteractionMode;
use crate::llm::ProviderType;
use serde::{Deserialize, Serialize};

/// Lightweight settings snapshot for reading/writing agent configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSettings {
    pub provider: ProviderType,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<usize>,
    pub tools: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<InteractionMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_provider: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub show_sources: Option<bool>,
}

/// Default embedding model for a provider that supports embeddings.
#[must_use]
pub fn default_embedding_model(provider: ProviderType) -> Option<String> {
    match provider {
        ProviderType::OpenAI => Some("text-embedding-3-small".to_string()),
        ProviderType::Bedrock => Some("amazon.titan-embed-text-v2:0".to_string()),
        ProviderType::OpenRouter => Some("openai/text-embedding-3-small".to_string()),
        ProviderType::CloudflareAIGateway => {
            Some("workers-ai/@cf/baai/bge-base-en-v1.5".to_string())
        }
        ProviderType::LMStudio => Some("default".to_string()),
        ProviderType::Ollama => Some("nomic-embed-text".to_string()),
        _ => None,
    }
}

use super::credentials::resolve_model_for_provider_with_discovery_at;
use super::lifecycle::{apply_context_window_limit, resolve_context_window_override};
use super::{AgentService, ProviderFactory};
use crate::agent::memory::TokenCounterFactory;
use crate::agent::modes::readonly_mode_prompt;
use anyhow::Context;
use std::sync::Arc;

/// Plan-mode exception: `write_plan` is a deliberate side-effecting tool that
/// remains available while other write tools are filtered out.
const PLAN_MODE_ALLOWED_WRITE_TOOL: &str = "write_plan";

impl AgentService {
    /// Override the provider context window for this service.
    pub fn set_context_window(&mut self, context_window: usize) -> anyhow::Result<()> {
        apply_context_window_limit(
            &mut self.agent,
            self.provider,
            &self.model,
            Some(context_window),
        )?;
        self.context_window_override = Some(context_window);
        self.rebudget_project_context_for(Some(context_window));
        Ok(())
    }

    /// The context window the user set (flag, config, environment or
    /// `/context-window`), if any.
    #[must_use]
    pub fn context_window_override(&self) -> Option<usize> {
        resolve_context_window_override(self.context_window_override)
    }

    /// Effective full context window for the active provider/model.
    #[must_use]
    pub const fn context_window(&self) -> Option<usize> {
        self.agent.cached_context_window
    }

    /// Snapshot current agent settings for API/UI sync.
    #[must_use]
    pub fn snapshot_settings(&self) -> AgentSettings {
        let tools: Vec<String> = self
            .list_tools()
            .into_iter()
            .map(|(name, _, _)| name)
            .collect();
        AgentSettings {
            provider: self.provider,
            model: self.model.clone(),
            system_prompt: Some(self.agent.system_prompt().to_string()),
            temperature: self.agent.temperature(),
            max_tokens: self.agent.max_tokens(),
            context_window: self.agent.cached_context_window,
            tools,
            mode: Some(self.interaction_mode),
            embedding_provider: self.embedding_provider_id.clone(),
            embedding_model: self.embedding_model_id.clone(),
            show_sources: Some(self.show_sources),
        }
    }

    /// Get the current interaction mode.
    #[must_use]
    pub const fn interaction_mode(&self) -> crate::agent::modes::InteractionMode {
        self.interaction_mode
    }

    /// Snapshot of the evolving plan document (populated during Plan mode turns).
    pub async fn plan_document(&self) -> Option<crate::agent::modes::PlanDocument> {
        self.session.plan_document.read().await.clone()
    }

    /// Replace the plan document (e.g. after loading a persisted session snapshot).
    pub async fn set_plan_document(&mut self, plan: Option<crate::agent::modes::PlanDocument>) {
        *self.session.plan_document.write().await = plan;
    }

    /// Set the active interaction mode without reapplying settings. Prefer
    /// [`apply_interaction_mode`] or [`apply_settings`] when callers want the
    /// readonly tool filter and prompt overlay applied.
    pub async fn set_interaction_mode(&mut self, mode: crate::agent::modes::InteractionMode) {
        self.interaction_mode = mode;
        if matches!(mode, crate::agent::modes::InteractionMode::Plan) {
            let mut plan_guard = self.session.plan_document.write().await;
            if plan_guard.is_none() {
                *plan_guard = Some(crate::agent::modes::PlanDocument::new());
            }
        }
    }

    /// Tools permitted in a readonly interaction mode.
    ///
    /// Uses registry `is_readonly` so custom/MCP tools that declare write
    /// capabilities are filtered. Plan mode additionally keeps `write_plan`.
    #[must_use]
    pub fn readonly_allowed_tool_names(&self, mode: InteractionMode) -> Vec<String> {
        self.list_tools()
            .into_iter()
            .map(|(name, _, _)| name)
            .filter(|name| {
                self.tool_registry().is_readonly(name)
                    || (matches!(mode, InteractionMode::Plan)
                        && name == PLAN_MODE_ALLOWED_WRITE_TOOL)
            })
            .collect()
    }

    /// Apply an interaction mode with readonly tool filtering and prompt overlay.
    ///
    /// When leaving a readonly mode, restores `base_system_prompt` (if set) and
    /// clears the allowed-tool filter so the full registry is visible again.
    pub async fn apply_interaction_mode(
        &mut self,
        mode: crate::agent::modes::InteractionMode,
    ) -> anyhow::Result<()> {
        let previous_mode = self.interaction_mode;
        self.set_interaction_mode(mode).await;
        if mode.is_readonly() {
            let readonly_tools = self.readonly_allowed_tool_names(mode);
            self.agent_mut().set_allowed_tools(Some(readonly_tools));
            if let Some(mode_prompt) = readonly_mode_prompt(mode) {
                let base = self
                    .base_system_prompt
                    .clone()
                    .unwrap_or_else(|| self.agent.system_prompt().to_string());
                if self.base_system_prompt.is_none() {
                    self.base_system_prompt = Some(base.clone());
                }
                let new_prompt = format!("{mode_prompt}\n\n{base}");
                self.agent_mut().set_system_prompt(new_prompt);
            }
        } else if previous_mode.is_readonly() {
            self.agent_mut().set_allowed_tools(None);
            if let Some(base) = self.base_system_prompt.clone() {
                self.agent_mut().set_system_prompt(base);
            }
        }
        Ok(())
    }

    /// Apply settings to this service. For provider/model changes calls `change_model`.
    /// For mode, applies readonly restrictions when mode is Plan or Ask.
    pub async fn apply_settings(&mut self, settings: AgentSettings) -> anyhow::Result<()> {
        let provider_changed = settings.provider != self.provider;
        let model_changed = settings.model != self.model;
        // A snapshot carries the window and project section in effect when
        // it was taken. Only a different value is a change the caller asked
        // for; the old ones must not pin the previous model's window.
        let window_before = self.agent.cached_context_window;
        let section_before = self.project_context_section().map(str::to_owned);
        if provider_changed || model_changed {
            self.change_model(Some(settings.provider), Some(settings.model.clone()))
                .await
                .context("apply_settings: change_model")?;
        }
        if let Some(prompt) = settings.system_prompt.clone() {
            self.base_system_prompt = Some(prompt.clone());
            self.agent_mut().set_system_prompt(prompt);
            if let Some(stale) = section_before {
                self.replace_project_section(&stale);
            }
        }
        if let Some(temp) = settings.temperature {
            self.agent_mut().set_temperature(Some(temp));
        }
        if let Some(max) = settings.max_tokens {
            self.agent_mut().set_max_tokens(Some(max));
        }
        if let Some(context_window) = settings.context_window
            && Some(context_window) != window_before
        {
            self.set_context_window(context_window)?;
        }
        // In settings payloads an empty tool list means "no explicit tool
        // filter", matching chat requests where `tools: None` uses defaults.
        // Treating it as "allow zero tools" makes the hosted UI look connected
        // while silently preventing repo/file inspection.
        if settings.tools.is_empty() {
            self.agent_mut().set_allowed_tools(None);
        } else {
            self.agent_mut()
                .set_allowed_tools(Some(settings.tools.clone()));
        }
        if let Some(mode) = settings.mode {
            self.apply_interaction_mode(mode).await?;
            // `apply_interaction_mode` clears the filter when leaving readonly.
            // Re-apply the caller's explicit tool selection afterward.
            if !mode.is_readonly() {
                if settings.tools.is_empty() {
                    self.agent_mut().set_allowed_tools(None);
                } else {
                    self.agent_mut()
                        .set_allowed_tools(Some(settings.tools.clone()));
                }
            }
        }
        self.apply_embedding_settings(&settings).await?;
        if let Some(show_sources) = settings.show_sources {
            self.show_sources = show_sources;
        }
        Ok(())
    }

    async fn apply_embedding_settings(&mut self, settings: &AgentSettings) -> anyhow::Result<()> {
        let Some(ref provider_name) = settings.embedding_provider else {
            return Ok(());
        };
        let trimmed = provider_name.trim();
        if trimmed.is_empty() {
            self.set_embedding_selection(None, None, None);
            return Ok(());
        }
        let provider_type: ProviderType = trimmed
            .parse()
            .map_err(|err| anyhow::anyhow!("invalid embedding provider '{trimmed}': {err}"))?;
        let model = settings
            .embedding_model
            .as_deref()
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| default_embedding_model(provider_type))
            .ok_or_else(|| anyhow::anyhow!("provider '{trimmed}' does not support embeddings"))?;
        let embedding = ProviderFactory::create_embedding_provider(
            provider_type,
            model.clone(),
            Arc::clone(&self.credential_provider),
        )
        .await
        .context("apply_settings: create embedding provider")?;
        match embedding {
            Some(provider) => {
                self.set_embedding_selection(
                    Some(provider),
                    Some(provider_type.to_string()),
                    Some(model),
                );
                Ok(())
            }
            None => Err(anyhow::anyhow!(
                "provider '{trimmed}' does not support embeddings"
            )),
        }
    }

    /// When in Plan mode, ask the LLM for a structured plan extraction from
    /// the last user/assistant exchange and merge it into `plan_document`.
    ///
    /// This is intentionally best-effort: extraction failures are logged at
    /// `warn` level and never propagate to the caller because plan-mode
    /// capture should not break the primary chat path.
    pub(super) async fn capture_plan_turn(&self, user_message: &str, assistant_reply: &str) {
        use crate::agent::modes::{InteractionMode, PlanDocument, plan_extraction_prompt};
        use crate::messages::{CompletionRequest, Message as MessageType};
        if !matches!(self.interaction_mode, InteractionMode::Plan) {
            return;
        }
        if user_message.trim().is_empty() || assistant_reply.trim().is_empty() {
            return;
        }

        let extraction_user = format!(
            "{}\n\nEXCHANGE:\nUser: {}\nAssistant: {}",
            plan_extraction_prompt(),
            user_message,
            assistant_reply
        );
        let messages = vec![
            MessageType::system("You are a precise JSON extractor. Respond with only valid JSON."),
            MessageType::user(extraction_user),
        ];
        let request = CompletionRequest {
            messages: std::sync::Arc::new(messages),
            max_tokens: Some(512),
            temperature: Some(0.0),
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };
        let completion = match self.agent.llm_provider().complete(request).await {
            Ok(c) => c,
            Err(err) => {
                tracing::warn!(error = %err, "plan extraction LLM call failed");
                return;
            }
        };
        let extracted_text = completion.content.as_text().to_string();
        let mut plan_guard = self.session.plan_document.write().await;
        let plan = plan_guard.get_or_insert_with(PlanDocument::new);
        match plan.merge_extraction(&extracted_text) {
            Ok(true) => tracing::debug!("plan document updated from turn extraction"),
            Ok(false) => tracing::trace!("plan extraction yielded no new data"),
            Err(err) => tracing::warn!(
                error = %err,
                snippet = %extracted_text.chars().take(200).collect::<String>(),
                "failed to parse plan extraction JSON"
            ),
        }
    }

    /// Change the model (and optionally provider) mid-session.
    /// This preserves conversation history and other agent state.
    ///
    /// # Arguments
    /// * `provider` - Optional new provider type. If None, keeps current provider.
    /// * `model` - New model name. Must be compatible with the provider.
    ///
    /// # Errors
    /// Returns an error if the provider/model cannot be created or if credentials are missing.
    pub async fn change_model(
        &mut self,
        provider: Option<ProviderType>,
        model: Option<String>,
    ) -> anyhow::Result<()> {
        let new_provider_type = provider.unwrap_or(self.provider);
        // A URL set for one provider's server is never sent to another
        // provider; that one uses its own environment URL or its default.
        let base_url = self
            .configured_base_url
            .as_ref()
            .filter(|(configured, _)| *configured == new_provider_type)
            .map(|(_, url)| url.as_str());
        let new_model =
            resolve_model_for_provider_with_discovery_at(new_provider_type, model, base_url)
                .await?;

        let new_provider_arc = ProviderFactory::create(
            new_provider_type,
            new_model.clone(),
            Arc::clone(&self.credential_provider),
            base_url,
            None,
            None,
            #[cfg(feature = "reqwest")]
            None,
        )
        .await
        .context("Failed to create new provider for model change")?;

        // Update the agent's provider (llm_provider is pub(crate), so we can access it)
        self.agent.llm_provider = new_provider_arc;

        // Update model name and provider type
        self.model = new_model;
        self.provider = new_provider_type;

        // Update token counter for the new model
        self.agent
            .set_token_counter(TokenCounterFactory::with_model(Some(self.model.clone())).build());

        apply_context_window_limit(
            &mut self.agent,
            self.provider,
            &self.model,
            resolve_context_window_override(self.context_window_override),
        )?;
        self.apply_reported_context_window().await;
        self.rebudget_project_context();

        Ok(())
    }
}
