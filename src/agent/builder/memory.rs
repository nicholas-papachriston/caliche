use super::{
    AgentBuilder, AgentService, Arc, BufferMemory, BufferMemoryConfig, CompositeMemory,
    ContextBudgetAllocator, ConversationStore, ConversationStoreMemoryPersistence, EnhancedMemory,
    EntityMemory, EntityMemoryConfig, KnowledgeStore, LlmMessageCompressor, MemoryBuilder,
    MemoryConfig, MemoryPersistence, MemoryPersistenceConfig, MemoryStrategy, MemoryType,
    ProviderEntityExtractor, ProviderSummarizer, RelevanceScorer, SummaryMemory,
    SummaryMemoryConfig, TokenCounter, ToolOutputCompressor, WorkingMemoryManager, anyhow,
};
#[cfg(feature = "vector-memory")]
use super::{Backend, VectorBackendConfig, VectorMemory, VectorMemoryConfig};
use crate::agent::memory::{CompositeMemoryConfig, EnhancedMemoryConfig};
use crate::llm::EmbeddingProvider;
#[cfg(feature = "vector-memory")]
use anyhow::Context;

impl AgentBuilder {
    #[must_use]
    pub fn buffer_memory(mut self, config: BufferMemoryConfig) -> Self {
        self.memory_type = Some(MemoryType::Buffer(config));
        self
    }

    /// Configure the agent to use buffer-based memory with default settings.
    #[must_use]
    pub fn with_buffer_memory(mut self) -> Self {
        self.memory_type = Some(MemoryType::Buffer(BufferMemoryConfig::default()));
        self
    }

    /// Configure the agent to use buffer-based memory with custom max tokens.
    #[must_use]
    pub fn with_buffer_memory_tokens(mut self, max_tokens: usize) -> Self {
        self.memory_type = Some(MemoryType::Buffer(
            BufferMemoryConfig::default().with_max_tokens(max_tokens),
        ));
        self
    }

    #[must_use]
    pub fn summary_memory(mut self, config: SummaryMemoryConfig) -> Self {
        self.memory_type = Some(MemoryType::Summary(config));
        self
    }

    #[must_use]
    pub fn with_summary_memory(mut self) -> Self {
        self.memory_type = Some(MemoryType::Summary(SummaryMemoryConfig::default()));
        self
    }

    #[must_use]
    pub fn entity_memory(mut self, config: EntityMemoryConfig) -> Self {
        self.memory_type = Some(MemoryType::Entity(config));
        self
    }

    #[must_use]
    pub fn with_entity_memory(mut self) -> Self {
        self.memory_type = Some(MemoryType::Entity(EntityMemoryConfig::default()));
        self
    }

    #[cfg(feature = "vector-memory")]
    #[must_use]
    pub fn vector_memory(mut self, config: VectorMemoryConfig) -> Self {
        self.memory_type = Some(MemoryType::Vector(config));
        self
    }

    #[cfg(feature = "vector-memory")]
    #[must_use]
    pub fn with_vector_memory(mut self) -> Self {
        self.memory_type = Some(MemoryType::Vector(VectorMemoryConfig::default()));
        self
    }

    #[must_use]
    pub fn memory(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }

    #[must_use]
    pub fn with_enhanced_memory(self) -> Self {
        self.memory(
            MemoryBuilder::enhanced(MemoryType::Buffer(BufferMemoryConfig::default())).build(),
        )
    }

    #[must_use]
    pub fn with_memory_config(mut self, config: MemoryConfig) -> Self {
        if let Some(mt) = config.memory_type {
            self.memory_type = Some(mt);
        }
        if let (Some(persistence), Some(conv_id)) = (config.persistence, config.conversation_id) {
            self.memory_persistence = Some(MemoryPersistenceConfig {
                persistence,
                conversation_id: conv_id,
                autosave: config.autosave,
            });
        }
        self
    }

    pub fn persist_memory_with(
        mut self,
        persistence: Arc<dyn MemoryPersistence>,
        conversation_id: impl Into<String>,
        autosave: bool,
    ) -> Self {
        self.memory_persistence = Some(MemoryPersistenceConfig {
            persistence,
            conversation_id: conversation_id.into(),
            autosave,
        });
        self
    }

    pub fn persist_memory_in_conversation_store(
        mut self,
        store: Arc<ConversationStore>,
        agent_id: impl Into<String>,
        conversation_id: impl Into<String>,
        autosave: bool,
        ttl_seconds: Option<u64>,
    ) -> Self {
        let conversation_id = conversation_id.into();
        let persistence = Arc::new(ConversationStoreMemoryPersistence::new(
            store,
            agent_id.into(),
            ttl_seconds,
        ));
        self.memory_persistence = Some(MemoryPersistenceConfig {
            persistence,
            conversation_id,
            autosave,
        });
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn build_one_memory_strategy(
        memory_type: MemoryType,
        service: &AgentService,
        token_counter: std::sync::Arc<dyn TokenCounter>,
        agent_context_budget: usize,
        default_buffer_tokens: usize,
        default_summary_tokens: usize,
        default_entity_tokens: usize,
        #[cfg(feature = "vector-memory")] prebuilt_vector_backend: Option<
            Arc<dyn Backend + Send + Sync>,
        >,
        #[cfg(feature = "vector-memory")] vector_backend_config: Option<VectorBackendConfig>,
        persistence_cfg: Option<&MemoryPersistenceConfig>,
    ) -> anyhow::Result<Box<dyn MemoryStrategy>> {
        #[cfg(not(feature = "vector-memory"))]
        let _ = persistence_cfg;
        match memory_type {
            MemoryType::Buffer(config) => Ok(build_buffer_memory(
                config,
                token_counter,
                agent_context_budget,
            )),
            MemoryType::Summary(config) => Ok(build_summary_memory(
                config,
                service,
                token_counter,
                agent_context_budget,
            )),
            MemoryType::Entity(config) => {
                build_entity_memory(config, service, token_counter, agent_context_budget)
            }
            #[cfg(feature = "vector-memory")]
            MemoryType::Vector(config) => {
                build_vector_memory(
                    config,
                    service,
                    token_counter,
                    prebuilt_vector_backend,
                    vector_backend_config,
                    persistence_cfg,
                )
                .await
            }
            MemoryType::Composite(config) => {
                build_composite_memory(
                    config,
                    service,
                    token_counter,
                    agent_context_budget,
                    default_buffer_tokens,
                    default_summary_tokens,
                    default_entity_tokens,
                    #[cfg(feature = "vector-memory")]
                    prebuilt_vector_backend,
                    #[cfg(feature = "vector-memory")]
                    vector_backend_config,
                )
                .await
            }
            MemoryType::Enhanced(config) => {
                build_enhanced_memory(
                    config,
                    service,
                    token_counter,
                    agent_context_budget,
                    default_buffer_tokens,
                    default_summary_tokens,
                    default_entity_tokens,
                    #[cfg(feature = "vector-memory")]
                    prebuilt_vector_backend,
                    #[cfg(feature = "vector-memory")]
                    vector_backend_config,
                )
                .await
            }
        }
    }
}

fn build_buffer_memory(
    mut config: BufferMemoryConfig,
    token_counter: std::sync::Arc<dyn TokenCounter>,
    agent_context_budget: usize,
) -> Box<dyn MemoryStrategy> {
    if config.clamp_to_context_limit(agent_context_budget) {
        tracing::warn!(
            buffer_max_tokens = agent_context_budget,
            "Buffer memory max_tokens capped to model context limit"
        );
    }
    Box::new(BufferMemory::new(config, token_counter))
}

fn build_summary_memory(
    mut config: SummaryMemoryConfig,
    service: &AgentService,
    token_counter: std::sync::Arc<dyn TokenCounter>,
    agent_context_budget: usize,
) -> Box<dyn MemoryStrategy> {
    if config.clamp_to_context_limit(agent_context_budget) {
        tracing::warn!(
            summary_max_tokens = agent_context_budget,
            "Summary memory max_tokens capped to model context limit"
        );
    }
    let summary_tokens = (config.max_tokens.min(u32::MAX as usize)) as u32;
    let summarizer = Arc::new(ProviderSummarizer::new(
        service.llm_provider(),
        summary_tokens.max(128),
    ));
    Box::new(SummaryMemory::new(config, token_counter, summarizer))
}

fn build_entity_memory(
    mut config: EntityMemoryConfig,
    service: &AgentService,
    token_counter: std::sync::Arc<dyn TokenCounter>,
    agent_context_budget: usize,
) -> anyhow::Result<Box<dyn MemoryStrategy>> {
    if config.clamp_to_context_limit(agent_context_budget) {
        tracing::warn!(
            entity_max_tokens = agent_context_budget,
            "Entity memory max_tokens capped to model context limit"
        );
    }
    let embedding_provider = require_embedding_provider(service, "Entity")?;
    let entity_tokens = (config.max_tokens.min(u32::MAX as usize)) as u32;
    let extractor = Arc::new(ProviderEntityExtractor::new(
        service.llm_provider(),
        entity_tokens.max(256),
    ));
    Ok(Box::new(EntityMemory::new(
        config,
        token_counter,
        embedding_provider,
        extractor,
    )))
}

fn require_embedding_provider(
    service: &AgentService,
    kind: &str,
) -> anyhow::Result<Arc<dyn EmbeddingProvider>> {
    service.embedding_provider().ok_or_else(|| {
        anyhow!(
            "{kind} memory requires an embedding-capable provider, but {:?} does not expose embeddings",
            service.provider()
        )
    })
}

#[cfg(feature = "vector-memory")]
async fn resolve_vector_backend(
    config: &VectorMemoryConfig,
    prebuilt_vector_backend: Option<Arc<dyn Backend + Send + Sync>>,
    vector_backend_config: Option<VectorBackendConfig>,
) -> anyhow::Result<Arc<dyn Backend + Send + Sync>> {
    if let Some(backend) = prebuilt_vector_backend {
        return Ok(backend);
    }
    if let Some(ref backend_config) = vector_backend_config {
        return backend_config
            .build_backend()
            .await
            .context("Failed to build vector backend for VectorMemory");
    }
    use dripstone::backend::memory::{InMemoryConfig, MemoryBackend};
    Ok(Arc::new(MemoryBackend::with_config(InMemoryConfig {
        default_ttl_secs: Some(86_400 * 7),
        max_entries: Some(config.max_memories),
        auto_cleanup: true,
    })) as Arc<dyn Backend + Send + Sync>)
}

#[cfg(feature = "vector-memory")]
async fn build_vector_memory(
    config: VectorMemoryConfig,
    service: &AgentService,
    token_counter: std::sync::Arc<dyn TokenCounter>,
    prebuilt_vector_backend: Option<Arc<dyn Backend + Send + Sync>>,
    vector_backend_config: Option<VectorBackendConfig>,
    persistence_cfg: Option<&MemoryPersistenceConfig>,
) -> anyhow::Result<Box<dyn MemoryStrategy>> {
    let embedding_provider = require_embedding_provider(service, "Vector")?;
    let vector_backend =
        resolve_vector_backend(&config, prebuilt_vector_backend, vector_backend_config).await?;
    let conversation_id = persistence_cfg.map(|cfg| cfg.conversation_id.as_str().to_owned());
    Ok(Box::new(VectorMemory::new(
        config,
        token_counter,
        embedding_provider,
        vector_backend,
        conversation_id,
    )))
}

#[allow(clippy::too_many_arguments)]
async fn build_nested_memory(
    memory_type: MemoryType,
    service: &AgentService,
    token_counter: std::sync::Arc<dyn TokenCounter>,
    agent_context_budget: usize,
    default_buffer_tokens: usize,
    default_summary_tokens: usize,
    default_entity_tokens: usize,
    #[cfg(feature = "vector-memory")] prebuilt_vector_backend: Option<
        Arc<dyn Backend + Send + Sync>,
    >,
    #[cfg(feature = "vector-memory")] vector_backend_config: Option<VectorBackendConfig>,
) -> anyhow::Result<Box<dyn MemoryStrategy>> {
    Box::pin(AgentBuilder::build_one_memory_strategy(
        memory_type,
        service,
        token_counter,
        agent_context_budget,
        default_buffer_tokens,
        default_summary_tokens,
        default_entity_tokens,
        #[cfg(feature = "vector-memory")]
        prebuilt_vector_backend,
        #[cfg(feature = "vector-memory")]
        vector_backend_config,
        None,
    ))
    .await
}

#[allow(clippy::too_many_arguments)]
async fn build_composite_memory(
    config: CompositeMemoryConfig,
    service: &AgentService,
    token_counter: std::sync::Arc<dyn TokenCounter>,
    agent_context_budget: usize,
    default_buffer_tokens: usize,
    default_summary_tokens: usize,
    default_entity_tokens: usize,
    #[cfg(feature = "vector-memory")] prebuilt_vector_backend: Option<
        Arc<dyn Backend + Send + Sync>,
    >,
    #[cfg(feature = "vector-memory")] vector_backend_config: Option<VectorBackendConfig>,
) -> anyhow::Result<Box<dyn MemoryStrategy>> {
    let primary = build_nested_memory(
        *config.primary,
        service,
        Arc::clone(&token_counter),
        agent_context_budget,
        default_buffer_tokens,
        default_summary_tokens,
        default_entity_tokens,
        #[cfg(feature = "vector-memory")]
        prebuilt_vector_backend.clone(),
        #[cfg(feature = "vector-memory")]
        vector_backend_config.clone(),
    )
    .await?;
    let mut secondaries = Vec::with_capacity(config.secondaries.len());
    for sec in config.secondaries {
        secondaries.push(
            build_nested_memory(
                sec,
                service,
                Arc::clone(&token_counter),
                agent_context_budget,
                default_buffer_tokens,
                default_summary_tokens,
                default_entity_tokens,
                #[cfg(feature = "vector-memory")]
                prebuilt_vector_backend.clone(),
                #[cfg(feature = "vector-memory")]
                vector_backend_config.clone(),
            )
            .await?,
        );
    }
    Ok(Box::new(CompositeMemory::new(
        primary,
        secondaries,
        token_counter,
        config.max_tokens,
        config.primary_budget_pct,
    )))
}

#[allow(clippy::too_many_arguments)]
async fn build_enhanced_memory(
    config: EnhancedMemoryConfig,
    service: &AgentService,
    token_counter: std::sync::Arc<dyn TokenCounter>,
    agent_context_budget: usize,
    default_buffer_tokens: usize,
    default_summary_tokens: usize,
    default_entity_tokens: usize,
    #[cfg(feature = "vector-memory")] prebuilt_vector_backend: Option<
        Arc<dyn Backend + Send + Sync>,
    >,
    #[cfg(feature = "vector-memory")] vector_backend_config: Option<VectorBackendConfig>,
) -> anyhow::Result<Box<dyn MemoryStrategy>> {
    if matches!(config.base.as_ref(), MemoryType::Enhanced(_)) {
        return Err(anyhow!(
            "EnhancedMemory must not wrap another EnhancedMemory: circular strategy not allowed"
        ));
    }
    let base = build_nested_memory(
        *config.base.clone(),
        service,
        Arc::clone(&token_counter),
        agent_context_budget,
        default_buffer_tokens,
        default_summary_tokens,
        default_entity_tokens,
        #[cfg(feature = "vector-memory")]
        prebuilt_vector_backend.clone(),
        #[cfg(feature = "vector-memory")]
        vector_backend_config.clone(),
    )
    .await?;
    let embedding_provider = require_embedding_provider(service, "Enhanced")?;
    let enhanced = assemble_enhanced_memory(
        &config,
        base,
        Arc::clone(&embedding_provider),
        Arc::clone(&token_counter),
    );
    Ok(Box::new(apply_enhanced_options(
        enhanced,
        &config,
        service,
        embedding_provider,
        token_counter,
    )))
}

fn assemble_enhanced_memory(
    config: &EnhancedMemoryConfig,
    base: Box<dyn MemoryStrategy>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    token_counter: std::sync::Arc<dyn TokenCounter>,
) -> EnhancedMemory {
    let budget_allocator =
        ContextBudgetAllocator::with_strategy(config.max_tokens, config.budget_strategy);
    let relevance_scorer = RelevanceScorer::new(embedding_provider, Arc::clone(&token_counter));
    let working_memory_manager =
        WorkingMemoryManager::new(token_counter, config.working_memory_tokens);
    EnhancedMemory::new(
        base,
        budget_allocator,
        relevance_scorer,
        working_memory_manager,
    )
}

fn apply_enhanced_options(
    mut enhanced: EnhancedMemory,
    config: &EnhancedMemoryConfig,
    service: &AgentService,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    token_counter: std::sync::Arc<dyn TokenCounter>,
) -> EnhancedMemory {
    if config.enable_tool_compression {
        let compressor = LlmMessageCompressor::new(
            service.llm_provider(),
            Arc::clone(&token_counter),
            config.compression_strategy,
        );
        let tool_compressor = Arc::new(ToolOutputCompressor::new(
            Arc::new(compressor),
            Arc::clone(&token_counter),
            500,
        ));
        enhanced = enhanced.with_tool_compressor(tool_compressor);
    }
    if config.enable_knowledge_store {
        let knowledge = KnowledgeStore::new(
            embedding_provider,
            token_counter,
            config.knowledge_max_entries,
        );
        enhanced = enhanced.with_knowledge_store(Arc::new(tokio::sync::RwLock::new(knowledge)));
    }
    enhanced
}
