//! Agent service lifecycle: construction, cloning, and teardown.

use super::{
    AgentService, AgentServiceConfig, ProviderFactory, ToolRegistryBuilder, credentials,
    resolve_model_for_provider_with_discovery, resolve_provider,
};
use crate::agent::Agent;
use crate::agent::memory::TokenCounterFactory;
#[cfg(feature = "session-tools")]
use crate::agent::session::live_registry::{
    LiveSessionHandle, LiveSessionRegistry, SessionLiveStatus,
};
use crate::agent::session::{AgentSession, SessionStore};
#[cfg(feature = "tools-rag")]
use crate::agent::tools::RagTool;
use crate::agent::tools::context_manager::get_provider_context_limits;
use crate::agent::tools::{Tool, ToolRegistry};
use crate::artifacts::ArtifactStore;
use crate::llm::cache::LlmCache;
use crate::llm::{
    CachingProvider, EmbeddingProvider, ImageGenerationProvider, Provider, ProviderType,
};
use crate::traits::credentials::{CredentialProvider, EnvironmentCredentialProvider};
use std::sync::Arc;
#[cfg(feature = "session-tools")]
use std::time::Instant;
#[cfg(feature = "session-tools")]
use tokio::sync::mpsc;
use uuid::Uuid;

/// Fields required to construct an [`AgentService`] after agent/session setup.
struct NewAgentServiceParams {
    agent: Agent,
    provider: ProviderType,
    model: String,
    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    embedding_model_id: Option<String>,
    credential_provider: Option<Arc<dyn CredentialProvider>>,
    image_generation_provider: Option<Arc<dyn ImageGenerationProvider>>,
    artifact_store: Option<Arc<ArtifactStore>>,
    #[cfg(feature = "tools-rag")]
    rag_tool: Option<Arc<RagTool>>,
    session: AgentSession,
    tool_registry: Arc<ToolRegistry>,
    context_window_override: Option<usize>,
    session_store: Option<Arc<dyn SessionStore>>,
    #[cfg(feature = "session-tools")]
    live_session_id: Option<String>,
    #[cfg(feature = "session-tools")]
    session_spawner: Option<Arc<dyn crate::agent::tools::session_tools::SessionSpawner>>,
}

fn new_agent_service(params: NewAgentServiceParams) -> AgentService {
    let embedding_provider_id = params
        .embedding_provider
        .as_ref()
        .map(|provider| provider.name().to_string());
    AgentService {
        agent: params.agent,
        provider: params.provider,
        model: params.model,
        embedding_provider: params.embedding_provider,
        embedding_provider_id,
        embedding_model_id: params.embedding_model_id,
        credential_provider: params
            .credential_provider
            .unwrap_or_else(|| Arc::new(EnvironmentCredentialProvider::new())),
        show_sources: true,
        image_generation_provider: params.image_generation_provider,
        artifact_store: params.artifact_store,
        #[cfg(all(feature = "mcp", feature = "mcp-client"))]
        mcp_clients: Vec::new(),
        #[cfg(feature = "tools-rag")]
        rag_tool: params.rag_tool,
        #[cfg(not(feature = "tools-rag"))]
        rag_tool: None,
        session: params.session,
        user_context: None,
        tool_registry: params.tool_registry,
        response_format: None,
        context_window_override: params.context_window_override,
        #[cfg(feature = "tools-rag")]
        workspace_indexing_ready: None,
        #[cfg(feature = "tools-rag")]
        workspace_config: None,
        #[cfg(feature = "tools-rag")]
        workspace_manifest: None,
        #[cfg(feature = "tools-rag")]
        file_watcher: None,
        retry_attempts: 1,
        session_store: params.session_store,
        #[cfg(feature = "session-tools")]
        live_session_id: params.live_session_id,
        #[cfg(feature = "session-tools")]
        session_spawner: params.session_spawner,
        plugin_skills: vec![],
        interaction_mode: crate::agent::modes::InteractionMode::default(),
        base_system_prompt: None,
        project_context: None,
        configured_base_url: None,
        reported_context_window: None,
    }
}

#[cfg(feature = "session-tools")]
fn register_live_session_if_enabled(session_id: &str, model: &str) -> Option<String> {
    let (message_tx, _rx) = mpsc::channel(32);
    let handle = LiveSessionHandle {
        session_id: session_id.to_string(),
        agent_id: "agent_service".to_string(),
        model: model.to_string(),
        status: SessionLiveStatus::Active,
        created_at: Instant::now(),
        message_tx,
        tags: vec![],
        depth: 0,
        history: Arc::new(parking_lot::Mutex::new(std::collections::VecDeque::new())),
    };
    LiveSessionRegistry::global().register(handle).ok()?;
    Some(session_id.to_string())
}

#[cfg(not(feature = "session-tools"))]
#[cfg_attr(
    not(feature = "session-tools"),
    expect(
        dead_code,
        reason = "without session-tools there is no live session registry"
    )
)]
fn register_live_session_if_enabled(_session_id: &str, _model: &str) -> Option<String> {
    None
}

fn context_window_from_env() -> Option<usize> {
    std::env::var("CALICHE_CONTEXT_WINDOW")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
}

pub(super) fn resolve_context_window_override(configured: Option<usize>) -> Option<usize> {
    configured.or_else(context_window_from_env)
}

pub(super) fn apply_context_window_limit(
    agent: &mut Agent,
    provider: ProviderType,
    model: &str,
    override_window: Option<usize>,
) -> anyhow::Result<()> {
    let (fallback_window, fallback_reserve) = get_provider_context_limits(provider, Some(model));
    let provider_window = agent
        .llm_provider()
        .context_window()
        .unwrap_or(fallback_window);
    let context_window = override_window.unwrap_or(provider_window);
    if context_window == 0 {
        anyhow::bail!("context window must be greater than 0 tokens");
    }
    if context_window <= fallback_reserve {
        anyhow::bail!(
            "context window ({context_window} tokens) must be larger than the reserved output budget ({fallback_reserve} tokens) for provider '{provider}' model '{model}'"
        );
    }

    agent.set_max_context_tokens(context_window.saturating_sub(fallback_reserve));
    agent.set_cached_context_window(Some(context_window));
    Ok(())
}

fn unconfigured_provider(provider: ProviderType, detail: impl Into<String>) -> Arc<dyn Provider> {
    Arc::new(crate::llm::unconfigured::UnconfiguredProvider::new(
        provider.to_string(),
        detail,
    ))
}

/// The model name `create_embedding_provider` takes for a service whose chat
/// model is `chat_model`. The Ollama branch uses that name as the embedding
/// model, and a chat model is not one, so Ollama gets its embedding model:
/// `OLLAMA_EMBEDDING_MODEL`, else the default.
fn service_embedding_model(provider: ProviderType, chat_model: &str) -> String {
    match provider {
        ProviderType::Ollama => caliche_contracts::credential_env::var("OLLAMA_EMBEDDING_MODEL")
            .ok()
            .or_else(|| super::default_embedding_model(provider))
            .unwrap_or_else(|| chat_model.to_string()),
        _ => chat_model.to_string(),
    }
}

async fn chat_and_embedding_providers(
    provider: ProviderType,
    model_name: String,
    credential_provider: Arc<dyn CredentialProvider>,
    defer_missing_credentials: bool,
    base_url: Option<&str>,
    #[cfg(feature = "local-inference-base")] local_inference_config: Option<
        crate::llm::LocalInferenceConfig,
    >,
    #[cfg(feature = "reqwest")] shared_http_client: Option<Arc<reqwest::Client>>,
) -> anyhow::Result<(Option<Arc<dyn EmbeddingProvider>>, Arc<dyn Provider>)> {
    let provider_name = provider.to_string();
    let credential_error = match credential_provider.get_credentials(&provider_name).await {
        Ok(_) => None,
        Err(err) if defer_missing_credentials => Some(err),
        Err(err) => return Err(err),
    };

    let embedding_provider = if credential_error.is_some() {
        None
    } else {
        match ProviderFactory::create_embedding_provider(
            provider,
            service_embedding_model(provider, &model_name),
            Arc::clone(&credential_provider),
        )
        .await
        {
            Ok(embedding) => embedding,
            Err(err)
                if defer_missing_credentials
                    && crate::llm::unconfigured::should_defer_provider_setup(&err) =>
            {
                None
            }
            Err(err) => return Err(err),
        }
    };

    if let Some(err) = credential_error {
        return Ok((
            embedding_provider,
            unconfigured_provider(provider, err.to_string()),
        ));
    }

    #[cfg(feature = "local-inference-base")]
    if matches!(provider, ProviderType::LocalInference)
        && let Some(local_config) = local_inference_config
    {
        return Ok((
            embedding_provider,
            Arc::new(crate::llm::LazyLocalInferenceProvider::new(local_config)),
        ));
    }
    match ProviderFactory::create(
        provider,
        model_name,
        credential_provider,
        base_url,
        None,
        None,
        #[cfg(feature = "reqwest")]
        shared_http_client,
    )
    .await
    {
        Ok(provider_arc) => Ok((embedding_provider, provider_arc)),
        Err(err)
            if defer_missing_credentials
                && crate::llm::unconfigured::should_defer_provider_setup(&err) =>
        {
            Ok((
                embedding_provider,
                unconfigured_provider(provider, format!("{err:#}")),
            ))
        }
        Err(err) => Err(err),
    }
}

impl AgentService {
    /// Create a new agent service with default tools.
    /// Uses provider and model from environment variables if not specified.
    pub async fn new() -> anyhow::Result<Self> {
        Self::from_config(AgentServiceConfig::new().with_all_default_tools()).await
    }

    pub async fn from_config(config: AgentServiceConfig) -> anyhow::Result<Self> {
        let provider = resolve_provider(config.provider);
        let model_name = credentials::resolve_model_for_provider_with_discovery_at(
            provider,
            config.model.clone(),
            config.base_url.as_deref(),
        )
        .await?;

        let configured_base_url = config.base_url.clone();
        let credential_provider = Arc::clone(&config.credential_provider);
        let (embedding_provider, mut provider_arc) = chat_and_embedding_providers(
            provider,
            model_name.clone(),
            Arc::clone(&credential_provider),
            config.defer_missing_credentials,
            config.base_url.as_deref(),
            #[cfg(feature = "local-inference-base")]
            config.local_inference_config,
            #[cfg(feature = "reqwest")]
            config.shared_http_client,
        )
        .await?;
        if let Some(cache) = config.llm_cache {
            provider_arc = Arc::new(CachingProvider::new(
                provider_arc,
                cache,
                model_name.clone(),
            )) as Arc<dyn Provider>;
        }

        let mut builder = ToolRegistryBuilder::new(provider)
            .with_tool_names(config.tool_selection.names)
            .with_built_in_tools(config.tool_selection.builtins)
            .with_mutation_style(config.mutation_style)
            .with_embedding_provider(embedding_provider.as_ref().map(Arc::clone))
            .with_image_generation(
                config.image_generation_provider.as_ref().map(Arc::clone),
                config.artifact_store.as_ref().map(Arc::clone),
            );

        if config.tool_selection.use_all_default_tools {
            builder = builder.use_all_default_tools();
        }

        #[cfg(feature = "tools-graph")]
        {
            builder = builder
                .with_graph_backend(config.graph_backend.as_ref().map(Arc::clone))
                .with_frontier_prioritizer(config.frontier_prioritizer.clone());
        }

        #[cfg(feature = "session-tools")]
        let session_spawner = config.session_spawner.clone();
        #[cfg(feature = "session-tools")]
        {
            builder = builder.session_spawner(config.session_spawner);
        }

        #[cfg(feature = "lsp")]
        {
            let lsp_manager = config.lsp_manager.or_else(|| {
                config
                    .lsp_config
                    .map(crate::lsp::LspManager::new)
                    .map(Arc::new)
            });
            builder = builder.with_lsp_manager(lsp_manager);
        }

        let tool_registry = builder.build().await?;
        let mut agent = Agent::new(provider_arc.clone(), Arc::clone(&tool_registry));
        agent.set_token_counter(TokenCounterFactory::with_model(Some(model_name.clone())).build());

        let context_window_override = resolve_context_window_override(config.context_window);
        apply_context_window_limit(&mut agent, provider, &model_name, context_window_override)?;

        let session = AgentSession::new(Uuid::now_v7().to_string());
        agent.set_session_id(Some(session.id.clone()));
        #[cfg(feature = "session-tools")]
        let live_session_id = register_live_session_if_enabled(&session.id, &model_name);

        let session_store: Option<Arc<dyn SessionStore>> = config.session_autosave_dir.map(|dir| {
            Arc::new(crate::agent::session::BinaryFileSessionStore::new(dir))
                as Arc<dyn SessionStore>
        });

        #[cfg(feature = "tools-rag")]
        let shared_rag_tool = config.shared_rag_tool;

        let embedding_model_id = embedding_provider
            .as_ref()
            .and_then(|_| super::default_embedding_model(provider));

        #[cfg_attr(not(feature = "tools-rag"), allow(unused_mut))]
        let mut service = new_agent_service(NewAgentServiceParams {
            agent,
            provider,
            model: model_name,
            embedding_provider,
            embedding_model_id,
            credential_provider: Some(credential_provider),
            image_generation_provider: config.image_generation_provider,
            artifact_store: config.artifact_store,
            #[cfg(feature = "tools-rag")]
            rag_tool: shared_rag_tool,
            session,
            tool_registry,
            context_window_override,
            session_store,
            #[cfg(feature = "session-tools")]
            live_session_id,
            #[cfg(feature = "session-tools")]
            session_spawner,
        });

        service.configured_base_url = configured_base_url.map(|url| (provider, url));
        service.apply_reported_context_window().await;

        #[cfg(feature = "tools-rag")]
        if let Some(rag_tool) = service.rag_tool.as_ref() {
            let rag_svc = rag_tool.service();
            service.agent_mut().set_workspace_rag(Some(rag_svc), None);
        }

        Ok(service)
    }

    /// Create a new agent service with custom tools.
    pub async fn with_tools(tools: Vec<Arc<dyn Tool>>) -> anyhow::Result<Self> {
        Self::with_provider_and_tools(None, None, tools, None, None).await
    }

    /// Create a new agent service from cached components.
    /// This creates a fresh `AgentService` with isolated state (memory, `SharedState`, `user_context`)
    /// while reusing expensive-to-create components (Provider, `ToolRegistry`).
    ///
    /// This is the preferred method for API handlers to ensure per-request isolation
    /// while maintaining memory efficiency through component caching.
    pub fn from_components(
        provider: Arc<dyn Provider>,
        tool_registry: Arc<ToolRegistry>,
        provider_type: ProviderType,
        model: String,
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    ) -> anyhow::Result<Self> {
        let mut agent = Agent::new(provider, Arc::clone(&tool_registry));
        apply_context_window_limit(&mut agent, provider_type, &model, None)?;

        let session = AgentSession::new(Uuid::now_v7().to_string());
        agent.set_session_id(Some(session.id.clone()));
        #[cfg(feature = "session-tools")]
        let live_session_id = register_live_session_if_enabled(&session.id, &model);

        Ok(new_agent_service(NewAgentServiceParams {
            agent,
            provider: provider_type,
            model,
            embedding_provider,
            embedding_model_id: None,
            credential_provider: None,
            image_generation_provider: None,
            artifact_store: None,
            #[cfg(feature = "tools-rag")]
            rag_tool: None,
            session,
            tool_registry,
            context_window_override: None,
            session_store: None,
            #[cfg(feature = "session-tools")]
            live_session_id,
            #[cfg(feature = "session-tools")]
            session_spawner: None,
        }))
    }

    /// Produce a per-request copy sharing expensive components (provider, tool registry,
    /// embedding provider, plugins, workspace RAG config) but with a fresh session and
    /// isolated mutable state.
    ///
    /// Prefer this over duplicating an [`AgentService`] when handlers or chains need an
    /// isolated session. Cloning via [`Clone`] is not supported because it would share
    /// session state across callers.
    pub fn clone_for_request(&self) -> Self {
        let mut agent = self.agent.clone_for_request();
        let session = AgentSession::new(Uuid::now_v7().to_string());
        agent.set_session_id(Some(session.id.clone()));
        #[cfg(feature = "session-tools")]
        let live_session_id = register_live_session_if_enabled(&session.id, &self.model);

        Self {
            agent,
            provider: self.provider,
            model: self.model.clone(),
            embedding_provider: self.embedding_provider.as_ref().map(Arc::clone),
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
            session,
            user_context: None,
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
            live_session_id,
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

    /// Create a new agent service from an existing Agent instance.
    /// Useful for testing with mock providers or custom agent configurations.
    pub fn new_with_agent(mut agent: Agent, provider_type: ProviderType) -> anyhow::Result<Self> {
        apply_context_window_limit(&mut agent, provider_type, "mock-model", None)?;
        let tool_registry = Arc::clone(&agent.tools);
        let session = AgentSession::new(Uuid::now_v7().to_string());
        agent.set_session_id(Some(session.id.clone()));
        #[cfg(feature = "session-tools")]
        let live_session_id = register_live_session_if_enabled(&session.id, "mock-model");

        Ok(new_agent_service(NewAgentServiceParams {
            agent,
            provider: provider_type,
            model: "mock-model".to_string(),
            embedding_provider: None,
            embedding_model_id: None,
            credential_provider: None,
            image_generation_provider: None,
            artifact_store: None,
            #[cfg(feature = "tools-rag")]
            rag_tool: None,
            session,
            tool_registry,
            context_window_override: None,
            session_store: None,
            #[cfg(feature = "session-tools")]
            live_session_id,
            #[cfg(feature = "session-tools")]
            session_spawner: None,
        }))
    }

    /// Create a new agent service with specified provider, model, and custom tools.
    /// If `tool_names` is provided, it filters the custom tools to only include those specified.
    /// If `tool_names` is None, all provided custom tools are registered.
    pub async fn with_provider_and_tools(
        provider: Option<ProviderType>,
        model: Option<String>,
        tools: Vec<Arc<dyn Tool>>,
        tool_names: Option<Vec<String>>,
        llm_cache: Option<Arc<LlmCache>>,
    ) -> anyhow::Result<Self> {
        let provider = resolve_provider(provider);
        let model_name = resolve_model_for_provider_with_discovery(provider, model).await?;

        // Create the LLM provider from environment
        let mut provider_arc =
            ProviderFactory::create_from_env(provider, model_name.clone(), None, None).await?;
        if let Some(cache) = llm_cache {
            provider_arc = Arc::new(CachingProvider::new(
                provider_arc,
                cache,
                model_name.clone(),
            )) as Arc<dyn Provider>;
        }

        // Build tool registry through the same canonical builder path used elsewhere.
        // This keeps alias handling and reduced default surface consistent across entry points.
        let tool_registry = ToolRegistryBuilder::new(provider)
            .with_tool_names(tool_names)
            .with_custom_tools(tools)
            .build()
            .await?;
        let mut agent = Agent::new(provider_arc.clone(), Arc::clone(&tool_registry));
        agent.set_token_counter(TokenCounterFactory::with_model(Some(model_name.clone())).build());

        apply_context_window_limit(
            &mut agent,
            provider,
            &model_name,
            resolve_context_window_override(None),
        )?;

        let session = AgentSession::new(Uuid::now_v7().to_string());
        agent.set_session_id(Some(session.id.clone()));
        #[cfg(feature = "session-tools")]
        let live_session_id = register_live_session_if_enabled(&session.id, &model_name);

        Ok(new_agent_service(NewAgentServiceParams {
            agent,
            provider,
            model: model_name,
            embedding_provider: None,
            embedding_model_id: None,
            credential_provider: None,
            image_generation_provider: None,
            artifact_store: None,
            #[cfg(feature = "tools-rag")]
            rag_tool: None,
            session,
            tool_registry,
            context_window_override: resolve_context_window_override(None),
            session_store: None,
            #[cfg(feature = "session-tools")]
            live_session_id,
            #[cfg(feature = "session-tools")]
            session_spawner: None,
        }))
    }

    /// Consume this service and return the underlying agent.
    ///
    /// Because [`AgentService`] implements [`Drop`] when `session-tools` is enabled, the agent is
    /// cloned rather than moved out.
    #[must_use]
    pub fn into_agent(self) -> Agent {
        self.clone_agent()
    }
}

#[cfg(test)]
mod credential_defer_tests {
    use super::AgentService;
    use crate::agent::service::AgentServiceConfig;
    use crate::llm::ProviderType;
    use crate::messages::{CompletionRequest, Message};
    use crate::traits::credentials::{
        CredentialProvider, Credentials, EnvironmentCredentialProvider,
    };
    use async_trait::async_trait;
    use std::sync::Arc;

    #[derive(Debug)]
    struct MissingKey;

    #[async_trait]
    impl CredentialProvider for MissingKey {
        async fn get_credentials(&self, provider: &str) -> anyhow::Result<Credentials> {
            anyhow::bail!(
                "No credentials found for provider '{provider}'. Set {provider}_API_KEY or API_KEY environment variable."
            )
        }

        fn supports_provider(&self, _provider: &str) -> bool {
            true
        }

        fn source_name(&self) -> &str {
            "missing-key-test"
        }
    }

    fn config(provider: ProviderType, defer: bool) -> AgentServiceConfig {
        AgentServiceConfig::new()
            .with_provider(Some(provider), Some("test-model".to_string()))
            .with_tool_names(Vec::new())
            .with_credential_provider(Arc::new(MissingKey))
            .with_defer_missing_credentials(defer)
    }

    #[tokio::test]
    async fn missing_key_fails_construction_unless_deferred() {
        let message = match AgentService::from_config(config(ProviderType::OpenRouter, false)).await
        {
            Err(err) => err.to_string(),
            Ok(_) => String::new(),
        };
        assert!(
            message.contains("No credentials found"),
            "construction requires a key, got: {message}"
        );
        assert!(message.contains("openrouter"), "{message}");
    }

    #[tokio::test]
    async fn deferred_missing_key_keeps_provider_and_fails_completion() {
        let service = AgentService::from_config(config(ProviderType::OpenAI, true))
            .await
            .expect("listen-style construction");
        assert_eq!(service.provider(), ProviderType::OpenAI);
        let debug = format!("{:?}", service.agent().llm_provider());
        assert!(
            debug.contains("UnconfiguredProvider"),
            "must not build a different provider: {debug}"
        );
        let request = CompletionRequest::new(Arc::new(vec![Message::user("hello")]));
        let err = service
            .agent()
            .llm_provider()
            .complete(request)
            .await
            .expect_err("turn needs a key");
        let message = err.to_string();
        assert!(message.contains("provider 'openai'"), "{message}");
        assert!(message.contains("No credentials found"), "{message}");
        assert!(
            !message.to_ascii_lowercase().contains("openrouter"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn ollama_credentials_are_not_a_missing_key() {
        let creds = EnvironmentCredentialProvider::new()
            .get_credentials("ollama")
            .await
            .expect("ollama needs no model key");
        assert!(matches!(creds, Credentials::None));
        let egress = anyhow::anyhow!(
            "egress provider url rejected: http://localhost:11434/v1: plugin http url targets a blocked host: localhost"
        );
        assert!(!crate::llm::unconfigured::is_missing_provider_credentials(
            &egress
        ));
        assert!(crate::llm::unconfigured::is_blocked_provider_url(&egress));
    }

    fn ollama_config(base_url: &str, defer: bool) -> AgentServiceConfig {
        AgentServiceConfig::new()
            .with_provider(Some(ProviderType::Ollama), Some("llama3.2".to_string()))
            .with_base_url(Some(base_url.to_string()))
            .with_tool_names(Vec::new())
            .with_credential_provider(Arc::new(EnvironmentCredentialProvider::new()))
            .with_defer_missing_credentials(defer)
    }

    fn is_egress_block(message: &str) -> bool {
        let lower = message.to_ascii_lowercase();
        lower.contains("egress provider url rejected")
            || lower.contains("plugin http url targets a blocked host")
            || lower.contains("egress denied for provider")
    }

    #[tokio::test]
    async fn local_ollama_url_builds_the_selected_provider() {
        let service = AgentService::from_config(ollama_config("http://localhost:11434/v1", false))
            .await
            .expect("local ollama is allowed");
        assert_eq!(service.provider(), ProviderType::Ollama);
        let debug = format!("{:?}", service.agent().llm_provider());
        assert!(
            !debug.contains("UnconfiguredProvider"),
            "local ollama must be a real provider: {debug}"
        );
    }

    #[tokio::test]
    async fn blocked_ollama_url_fails_construction_unless_deferred() {
        let message = match AgentService::from_config(ollama_config(
            "http://169.254.169.254/v1",
            false,
        ))
        .await
        {
            Err(err) => format!("{err:#}"),
            Ok(_) => String::new(),
        };
        assert!(
            is_egress_block(&message),
            "construction rejects the provider url, got: {message}"
        );
        assert!(
            !message.to_ascii_lowercase().contains("openrouter"),
            "{message}"
        );
    }

    #[tokio::test]
    async fn deferred_blocked_ollama_url_keeps_provider_and_fails_completion() {
        let service = AgentService::from_config(ollama_config("http://169.254.169.254/v1", true))
            .await
            .expect("listen-style construction");
        assert_eq!(service.provider(), ProviderType::Ollama);
        let debug = format!("{:?}", service.agent().llm_provider());
        assert!(
            debug.contains("UnconfiguredProvider"),
            "must keep the selected provider: {debug}"
        );
        let request = CompletionRequest::new(Arc::new(vec![Message::user("hello")]));
        let err = service
            .agent()
            .llm_provider()
            .complete(request)
            .await
            .expect_err("turn needs an allowed provider url");
        let message = err.to_string();
        assert!(message.contains("provider 'ollama'"), "{message}");
        assert!(is_egress_block(&message), "{message}");
        assert!(!message.contains("No credentials found"), "{message}");
        assert!(
            !message.to_ascii_lowercase().contains("openrouter"),
            "{message}"
        );
    }
}

#[cfg(feature = "session-tools")]
impl Drop for AgentService {
    fn drop(&mut self) {
        if let Some(ref id) = self.live_session_id {
            LiveSessionRegistry::global().unregister(id);
        }
    }
}

#[cfg(test)]
mod service_embedding_tests {
    use super::AgentService;
    use crate::agent::service::AgentServiceConfig;
    use crate::llm::{EmbeddingRequest, ProviderType};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// Serve `listener` until an embedding request arrives and return its
    /// body. Other requests, such as the service's startup model-state
    /// check, get a 404.
    async fn capture_embedding_request(listener: tokio::net::TcpListener) -> String {
        loop {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut request = Vec::new();
            let mut buf = [0_u8; 4096];
            let (head, body) = loop {
                let read = stream.read(&mut buf).await.expect("read");
                request.extend_from_slice(&buf[..read]);
                let text = String::from_utf8_lossy(&request);
                if let Some((head, body)) = text.split_once("\r\n\r\n") {
                    let length = head
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())?
                        })
                        .unwrap_or(0);
                    if body.len() >= length || read == 0 {
                        break (head.to_string(), body.to_string());
                    }
                }
            };
            if !head.starts_with("POST /v1/embeddings ") {
                let response =
                    "HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\nconnection: close\r\n\r\n";
                stream.write_all(response.as_bytes()).await.expect("write");
                continue;
            }
            let reply = r#"{"object":"list","data":[{"object":"embedding","embedding":[0.5,0.5],"index":0}],"model":"m","usage":{"prompt_tokens":1,"total_tokens":1}}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                reply.len()
            );
            stream.write_all(response.as_bytes()).await.expect("write");
            return body;
        }
    }

    #[tokio::test]
    #[serial_test::serial(env)]
    async fn an_ollama_service_embeds_with_the_embedding_model_not_the_chat_model() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let base_url = format!("http://{}/v1", listener.local_addr().expect("addr"));
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("OLLAMA_BASE_URL", &base_url);
        env.remove("OLLAMA_EMBEDDING_MODEL");
        let server = tokio::spawn(capture_embedding_request(listener));
        let service = AgentService::from_config(
            AgentServiceConfig::new()
                .with_provider(Some(ProviderType::Ollama), Some("gpt-oss:20b".to_string()))
                .with_base_url(Some(base_url.clone()))
                .with_tool_names(Vec::new()),
        )
        .await
        .expect("ollama service");
        let embedding = service.embedding_provider().expect("embedding provider");

        let _ = embedding
            .embed(EmbeddingRequest {
                texts: vec!["hello".to_string()],
                model: None,
            })
            .await;
        let body: serde_json::Value =
            serde_json::from_str(&server.await.expect("server")).expect("json body");

        assert_eq!(body["model"], "nomic-embed-text");
    }
}
