//! High-level service layer for managing agent setup and configuration.

mod config;
pub mod credentials;
mod lifecycle;
mod project_context;
mod provider_build;
mod provider_factory;
mod tool_registry_builder;

pub mod sandbox;
pub mod settings;

mod chat;
mod streaming;
#[cfg(feature = "tools-rag")]
mod workspace_rag;

use crate::agent::Agent;
use crate::agent::memory::MemoryStrategy;
use crate::agent::session::{AgentSession, SessionStore};
use crate::agent::skills::Skill;
#[cfg(feature = "tools-rag")]
use crate::agent::tools::RagTool;
use crate::agent::tools::{Tool, ToolRegistry};
use crate::artifacts::ArtifactStore;
use crate::llm::{EmbeddingProvider, ImageGenerationProvider, Provider, ProviderType};
use crate::messages::Message;
#[cfg(feature = "tools-rag")]
use crate::rag::RagHashlineOptions;
use crate::traits::credentials::CredentialProvider;
use crate::utils::user_context::UserContext;
use anyhow::Context;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

pub use config::{AgentServiceConfig, ToolSelection};
pub use credentials::{
    ResolvedRuntimeIdentity, resolve_model_for_provider, resolve_model_for_provider_with_discovery,
    resolve_provider, resolve_runtime_identity, resolve_runtime_identity_with_discovery,
};
pub use project_context::{DEFAULT_PROJECT_CONTEXT_SHARE, ProjectContext};
pub use provider_factory::{ProviderFactory, ProviderFactoryBuilder};
pub use settings::{AgentSettings, default_embedding_model};
pub use tool_registry_builder::ToolRegistryBuilder;

/// High-level service layer for managing agent setup and configuration.
///
/// `AgentService` provides a convenient API for creating and using AI agents
/// with automatic provider initialization, tool registration, and conversation management.
///
/// # Examples
///
/// ## Basic usage with default settings
///
/// ```rust,no_run
/// use caliche::AgentService;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     // Creates agent with provider and model from environment variables
///     let mut agent = AgentService::new().await?;
///     
///     let response = agent.chat("Hello!").await?;
///     println!("Agent: {}", response);
///     
///     Ok(())
/// }
/// ```
///
/// ## Specifying provider and model
///
/// ```rust,no_run
/// use caliche::{AgentService, AgentServiceConfig, ProviderType};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut agent = AgentService::from_config(
///         AgentServiceConfig::new()
///             .with_provider(Some(ProviderType::OpenAI), Some("gpt-4o".to_string()))
///             .with_all_default_tools(),
///     ).await?;
///     
///     let response = agent.chat("What is 2 + 2?").await?;
///     println!("{}", response);
///     
///     Ok(())
/// }
/// ```
///
/// ## Using specific tools only
///
/// ```rust,no_run
/// use caliche::{AgentService, AgentServiceConfig, ProviderType};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut agent = AgentService::from_config(
///         AgentServiceConfig::new()
///             .with_provider(Some(ProviderType::Bedrock), None)
///             .with_tool_names(vec!["calculator".to_string(), "fs".to_string()]),
///     ).await?;
///     
///     // This agent only has calculator and fs tools available
///     let tools = agent.list_tools();
///     for (name, desc, _) in tools {
///         println!("{}: {}", name, desc);
///     }
///     
///     Ok(())
/// }
/// ```
pub struct AgentService {
    agent: Agent,
    provider: ProviderType,
    model: String,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    embedding_provider_id: Option<String>,
    embedding_model_id: Option<String>,
    credential_provider: Arc<dyn CredentialProvider>,
    show_sources: bool,
    image_generation_provider: Option<Arc<dyn ImageGenerationProvider>>,
    artifact_store: Option<Arc<ArtifactStore>>,
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    mcp_clients: Vec<Arc<crate::mcp::McpClient>>,
    #[cfg(feature = "tools-rag")]
    rag_tool: Option<Arc<RagTool>>,
    #[cfg(not(feature = "tools-rag"))]
    rag_tool: Option<Arc<()>>,
    session: AgentSession,
    user_context: Option<UserContext>,
    /// Shared tool registry for MCP and other integrations
    tool_registry: Arc<ToolRegistry>,
    /// Response format for structured outputs
    pub(crate) response_format: Option<crate::output::ResponseFormat>,
    /// User-configured full model context window override.
    context_window_override: Option<usize>,
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_indexing_ready:
        Option<Arc<crate::agent::workspace_indexing::WorkspaceIndexingReady>>,
    #[cfg(feature = "tools-rag")]
    workspace_config: Option<crate::workspace::WorkspaceIndexerConfig>,
    #[cfg(feature = "tools-rag")]
    workspace_manifest: Option<Arc<tokio::sync::Mutex<crate::workspace::manifest::IndexManifest>>>,
    #[cfg(feature = "tools-rag")]
    file_watcher: Option<Arc<tokio::sync::Mutex<crate::workspace::WorkspaceWatcher>>>,
    retry_attempts: usize,
    session_store: Option<Arc<dyn SessionStore>>,
    /// When session-tools is enabled, the session is registered in `LiveSessionRegistry` and unregistered on drop.
    #[cfg(feature = "session-tools")]
    live_session_id: Option<String>,
    /// Live handle so slash commands can update `sessions_spawn` model choice.
    #[cfg(feature = "session-tools")]
    session_spawner: Option<Arc<dyn crate::agent::tools::session_tools::SessionSpawner>>,
    /// Skills applied from the plugin store; included in the discovery catalog for message-based detection.
    plugin_skills: Vec<Skill>,
    /// Current interaction mode; drives plan-mode extraction after each turn.
    /// The actual `PlanDocument` lives on `session.plan_document` so it survives
    /// `save_session` / `from_snapshot` without any service-level bookkeeping.
    interaction_mode: crate::agent::modes::InteractionMode,
    /// Base system prompt before readonly mode overlays are applied.
    base_system_prompt: Option<String>,
    /// Project context and the section of it now in the system prompt.
    project_context: Option<project_context::BudgetedProjectContext>,
    /// The base URL the service was built with, and the provider it is for.
    /// A model change on that provider keeps it.
    configured_base_url: Option<(ProviderType, String)>,
    /// The window the provider's server reported for the active model.
    reported_context_window: Option<usize>,
}

impl AgentService {
    /// Get a list of available tools.
    #[must_use]
    pub fn list_tools(&self) -> Vec<(String, String, serde_json::Value)> {
        self.agent.list_tools()
    }

    /// Apply the live subagent model/provider selection to `sessions_spawn`.
    #[cfg(feature = "session-tools")]
    pub fn set_subagent_selection(
        &self,
        selection: crate::agent::subagent::SubagentModelSelection,
    ) {
        if let Some(spawner) = self.session_spawner.as_ref() {
            spawner.set_selection(selection);
        }
    }

    /// Get a reference to the shared tool registry
    #[must_use]
    pub const fn tool_registry(&self) -> &Arc<ToolRegistry> {
        &self.tool_registry
    }

    /// Add a tool to the agent.
    pub fn add_tool(&mut self, tool: Arc<dyn Tool>) {
        self.agent.add_tool(tool);
    }

    /// Remove a tool from the agent.
    pub fn remove_tool(&mut self, name: &str) {
        self.agent.remove_tool(name);
    }

    /// Return the active learning-loop configuration, if the loop is enabled.
    #[cfg(feature = "learning-loop")]
    pub fn learning_loop_config(&self) -> Option<crate::agent::learning::CheckpointConfig> {
        self.agent
            .learning_loop
            .as_ref()
            .map(|learning_loop| learning_loop.config.clone())
    }

    /// Enable or update the learning loop while preserving any tracked episode history.
    #[cfg(feature = "learning-loop")]
    pub fn set_learning_loop_config(&mut self, config: crate::agent::learning::CheckpointConfig) {
        let mut learning = crate::agent::learning::LearningLoop::new(config);
        if let Some(existing) = self.agent.learning_loop.as_ref() {
            learning.episode_tracker = Arc::clone(&existing.episode_tracker);
            learning.plugin_store = existing.plugin_store.as_ref().map(Arc::clone);
        } else {
            tracing::warn!(
                "learning loop enabled at runtime without a plugin store; skill create/patch/delete will operate in-memory only"
            );
        }

        let learning_loop = Arc::new(learning);
        self.agent.learning_loop = Some(Arc::clone(&learning_loop));
        self.agent.add_tool(Arc::new(
            crate::agent::learning::skill_manage::SkillManageTool::new(learning_loop),
        ));
    }

    /// Disable the learning loop and remove its management tool.
    #[cfg(feature = "learning-loop")]
    pub fn clear_learning_loop(&mut self) {
        self.agent.learning_loop = None;
        self.agent.remove_tool("skill_manage");
    }

    /// Get the configured artifact store, if available.
    pub fn artifact_store(&self) -> Option<Arc<ArtifactStore>> {
        self.artifact_store.as_ref().map(Arc::clone)
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn mcp_clients(&self) -> Vec<Arc<crate::mcp::McpClient>> {
        self.mcp_clients.iter().map(Arc::clone).collect()
    }

    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    pub fn set_mcp_clients(&mut self, clients: Vec<Arc<crate::mcp::McpClient>>) {
        self.mcp_clients = clients;
    }

    /// Set the artifact store used by generated artifact tools.
    pub fn set_artifact_store(&mut self, artifact_store: Option<Arc<ArtifactStore>>) {
        self.artifact_store = artifact_store;
        if let Some(provider) = self.image_generation_provider.as_ref().map(Arc::clone) {
            self.register_image_generation_tool(provider);
        }
    }

    /// Configure the memory strategy after agent construction.
    pub fn set_memory_strategy(&mut self, memory: Option<Box<dyn MemoryStrategy>>) {
        self.session.memory = memory.map(|strategy| Arc::new(RwLock::new(strategy)));
    }

    /// Access the configured memory strategy, if any.
    /// Returns Arc for cheap reference counting.
    pub fn memory(&self) -> Option<Arc<RwLock<Box<dyn MemoryStrategy>>>> {
        self.session.memory.as_ref().map(Arc::clone)
    }

    /// Access the session (conversation history, memory, shared state).
    #[must_use]
    pub const fn session(&self) -> &AgentSession {
        &self.session
    }

    /// Get a reference to the underlying LLM provider (Arc clone is cheap).
    #[must_use]
    pub fn llm_provider(&self) -> Arc<dyn Provider> {
        Arc::clone(self.agent.llm_provider()) // Arc::clone is cheap
    }

    /// Get a reference to the embedding provider if available (Arc clone is cheap).
    pub fn embedding_provider(&self) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embedding_provider.as_ref().map(Arc::clone) // Arc::clone is cheap
    }

    /// Get a reference to the image generation provider if available.
    pub fn image_generation_provider(&self) -> Option<Arc<dyn ImageGenerationProvider>> {
        self.image_generation_provider.as_ref().map(Arc::clone)
    }

    /// Get a reference to the underlying agent (for advanced use cases).
    #[must_use]
    pub const fn agent(&self) -> &Agent {
        &self.agent
    }

    /// Clone the underlying agent (for flows and orchestration registration).
    #[must_use]
    pub fn clone_agent(&self) -> Agent {
        self.agent.clone()
    }

    /// Get the provider type.
    #[must_use]
    pub const fn provider(&self) -> ProviderType {
        self.provider
    }

    /// Get the model name.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Get a mutable reference to the underlying agent (for advanced use cases).
    pub const fn agent_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }

    /// Skills applied from the plugin store; included in the discovery catalog for message-based skill detection.
    #[must_use]
    pub fn plugin_skills(&self) -> &[Skill] {
        &self.plugin_skills
    }

    /// Set plugin-derived skills (used by the builder when plugins feature is enabled).
    #[cfg(feature = "plugins")]
    pub(crate) fn set_plugin_skills(&mut self, skills: Vec<Skill>) {
        self.plugin_skills = skills;
    }

    /// Set metrics tracker for tool execution observability.
    pub fn set_metrics(&mut self, metrics: Arc<crate::observability::ToolMetrics>) {
        self.agent.set_metrics(metrics);
    }

    /// Set user context for this request
    pub fn set_user_context(&mut self, context: Option<UserContext>) {
        self.user_context = context;
        // Propagate context to agent's tool registry
        if let Some(ref ctx) = self.user_context {
            self.agent.set_user_context(Some(ctx.clone()));
        } else {
            self.agent.set_user_context(None);
        }
    }

    /// Get current user context
    #[must_use]
    pub const fn user_context(&self) -> Option<&UserContext> {
        self.user_context.as_ref()
    }

    #[cfg(feature = "sandbox")]
    /// Set sandbox handle for bash/fs tool execution.
    pub fn set_sandbox(&mut self, handle: std::sync::Arc<caliche_sandbox::SandboxHandle>) {
        self.agent.set_sandbox(handle);
    }

    /// Whether assistant source citations should be shown to the user.
    #[must_use]
    pub const fn show_sources(&self) -> bool {
        self.show_sources
    }

    /// Set whether assistant source citations should be shown to the user.
    pub fn set_show_sources(&mut self, show_sources: bool) {
        self.show_sources = show_sources;
    }

    /// Configured embedding provider id, if one was selected.
    #[must_use]
    pub fn embedding_provider_id(&self) -> Option<&str> {
        self.embedding_provider_id.as_deref()
    }

    /// Configured embedding model id, if one was selected.
    #[must_use]
    pub fn embedding_model_id(&self) -> Option<&str> {
        self.embedding_model_id.as_deref()
    }

    /// Set or update the embedding provider for this service.
    /// Existing workspace RAG is invalidated because persisted embeddings are
    /// provider/model-specific and must be rebuilt with `/reindex`.
    pub fn set_embedding_provider(&mut self, provider: Option<Arc<dyn EmbeddingProvider>>) {
        let provider_id = provider.as_ref().map(|item| item.name().to_string());
        self.set_embedding_selection(provider, provider_id, None);
    }

    /// Set the embedding provider together with the ids used in settings snapshots.
    pub fn set_embedding_selection(
        &mut self,
        provider: Option<Arc<dyn EmbeddingProvider>>,
        provider_id: Option<String>,
        model_id: Option<String>,
    ) {
        if provider.is_none() {
            self.embedding_provider = None;
            self.embedding_provider_id = None;
            self.embedding_model_id = None;
        } else {
            self.embedding_provider_id =
                provider_id.or_else(|| provider.as_ref().map(|item| item.name().to_string()));
            self.embedding_model_id = model_id;
            self.embedding_provider = provider;
        }

        #[cfg(feature = "tools-rag")]
        {
            if let Some(config) = self.workspace_config.as_mut() {
                config.index_fingerprint = self.embedding_provider.as_ref().map(|provider| {
                    format!(
                        "provider={};model=manual;dimension=unknown;hashline=true;format=true;line_bm25={};chunking=AstAware",
                        provider.name(),
                        config.line_level_bm25
                    )
                });
                self.rag_tool = None;
                self.workspace_indexing_ready = None;
                self.agent_mut().set_workspace_rag(None, None);
            }
        }
    }

    /// Set or update the image generation provider for this service.
    ///
    /// When both a provider and artifact store are configured, this registers the
    /// `generate_image` tool. Clearing the provider unregisters the tool.
    pub fn set_image_generation_provider(
        &mut self,
        provider: Option<Arc<dyn ImageGenerationProvider>>,
    ) {
        self.image_generation_provider = provider;
        if let Some(provider) = self.image_generation_provider.as_ref().map(Arc::clone) {
            self.register_image_generation_tool(provider);
        } else {
            self.remove_tool(crate::agent::tools::names::TOOL_GENERATE_IMAGE);
        }
    }

    fn register_image_generation_tool(&mut self, provider: Arc<dyn ImageGenerationProvider>) {
        let Some(artifact_store) = self.artifact_store.as_ref().map(Arc::clone) else {
            self.remove_tool(crate::agent::tools::names::TOOL_GENERATE_IMAGE);
            return;
        };
        self.add_tool(Arc::new(crate::agent::tools::ImageGenerationTool::new(
            provider,
            artifact_store,
        )));
    }
}

impl Clone for AgentService {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            provider: self.provider,
            model: self.model.clone(), // String clone necessary
            embedding_provider: self.embedding_provider.as_ref().map(Arc::clone), // Arc::clone is cheap
            embedding_provider_id: self.embedding_provider_id.clone(),
            embedding_model_id: self.embedding_model_id.clone(),
            credential_provider: Arc::clone(&self.credential_provider),
            show_sources: self.show_sources,
            image_generation_provider: self.image_generation_provider.as_ref().map(Arc::clone),
            artifact_store: self.artifact_store.as_ref().map(Arc::clone),
            #[cfg(all(feature = "mcp", feature = "mcp-client"))]
            mcp_clients: self.mcp_clients.iter().map(Arc::clone).collect(),
            #[cfg(feature = "tools-rag")]
            rag_tool: self.rag_tool.as_ref().map(Arc::clone),
            #[cfg(not(feature = "tools-rag"))]
            rag_tool: self.rag_tool.as_ref().map(Arc::clone),
            session: self.session.clone(),
            user_context: self.user_context.clone(),
            tool_registry: Arc::clone(&self.tool_registry),
            response_format: self.response_format.clone(),
            context_window_override: self.context_window_override,
            #[cfg(feature = "tools-rag")]
            workspace_indexing_ready: self.workspace_indexing_ready.as_ref().map(Arc::clone),
            #[cfg(feature = "tools-rag")]
            workspace_config: self.workspace_config.clone(),
            #[cfg(feature = "tools-rag")]
            workspace_manifest: self.workspace_manifest.as_ref().map(Arc::clone),
            #[cfg(feature = "tools-rag")]
            file_watcher: self.file_watcher.as_ref().map(Arc::clone),
            retry_attempts: self.retry_attempts,
            session_store: self.session_store.as_ref().map(Arc::clone),
            #[cfg(feature = "session-tools")]
            live_session_id: None,
            #[cfg(feature = "session-tools")]
            session_spawner: self.session_spawner.as_ref().map(Arc::clone),
            plugin_skills: self.plugin_skills.clone(),
            interaction_mode: self.interaction_mode,
            base_system_prompt: self.base_system_prompt.clone(),
            project_context: self.project_context.clone(),
            configured_base_url: self.configured_base_url.clone(),
            reported_context_window: self.reported_context_window,
        }
    }
}
