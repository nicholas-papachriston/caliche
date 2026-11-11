//! Builder pattern for constructing Agent instances with a fluent API.

#[cfg(feature = "vector-memory")]
use crate::agent::memory::{VectorMemory, VectorMemoryConfig};
#[cfg(feature = "tools-rag")]
use crate::agent::tools::names::TOOL_FILESYSTEM;
use crate::data::persistence::ConversationStore;
#[cfg(all(feature = "tools-rag", feature = "tools-sql"))]
use crate::data::persistence::{
    DatabaseIntrospector, PostgresSchemaIntrospector, SchemaEmbeddingPipeline,
};
use crate::llm::cache::LlmCache;
#[cfg(feature = "tools-rag")]
use crate::traits::cache::{CacheBackend, InMemoryCacheBackend};
use crate::{
    agent::ToolExecutionDriver,
    agent::hooks::permission::PermissionMode,
    agent::hooks::{
        AmendmentPersister, ApprovalResolver, EnvironmentOnboardingConfig,
        EnvironmentOnboardingMiddleware, HookMatcher, HookRegistry, LifecycleHook,
        LoopDetectionConfig, LoopDetectionMiddleware, PreCompletionChecklistConfig,
        PreCompletionChecklistMiddleware, ReasoningEffortSlot, ReasoningScheduleConfig,
        ReasoningScheduler, SteeringFollowUpProvider,
    },
    agent::memory::{
        BufferMemory, BufferMemoryConfig, CompositeMemory, ConversationStoreMemoryPersistence,
        EnhancedMemory, EntityMemory, EntityMemoryConfig, KnowledgeStore, MemoryBuilder,
        MemoryPersistence, MemoryStrategy, MemoryType, PersistentMemory, ProviderEntityExtractor,
        ProviderSummarizer, RelevanceScorer, SummaryMemory, SummaryMemoryConfig, TokenCounter,
        TokenCounterFactory, WorkingMemoryManager,
        budget::ContextBudgetAllocator,
        compression::{LlmMessageCompressor, ToolOutputCompressor},
    },
    agent::service::AgentService,
    agent::tools::{ResearchTool, Tool},
    llm::ProviderType,
    traits::credentials::{
        CredentialProvider, Credentials, EnvironmentCredentialProvider, StaticCredentialProvider,
    },
};

#[cfg(feature = "tools-rag")]
use crate::agent::tools::RagTool;
#[cfg(all(feature = "tools-rag", feature = "tools-sql"))]
use crate::agent::tools::{PostgresSqlExecutor, SqlExecutionTool, SqlExecutor, SqlGenerationTool};
#[cfg(feature = "tools-rag")]
use crate::agent::workspace_indexing::WorkspaceIndexingReady;
#[cfg(feature = "vector-memory")]
use crate::data::storage::vector::VectorBackendConfig;
#[cfg(feature = "tools-rag")]
use crate::discovery::embedding::discover_best_embedding_provider;
#[cfg(feature = "tools-rag")]
use crate::rag::RagHashlineOptions;
#[cfg(feature = "tools-rag")]
use crate::workspace::{PersistenceConfig, WorkspaceIndexer, WorkspaceIndexerConfig};

use anyhow::{Context, anyhow};
use beef::lean::Cow;
#[cfg(feature = "mcp")]
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "session-tools")]
use crate::agent::tools::session_tools::{SessionSpawner, SessionToolsConfig};
#[cfg(feature = "vector-memory")]
use dripstone::Backend;

mod build;
mod build_attach;
mod build_helpers;
mod durable_streams;
mod hooks;
mod memory;
mod provider;
mod tools;
mod types;
#[cfg(feature = "tools-rag")]
mod workspace;
pub use types::{AwsCredentials, MemoryConfig, ObservabilityConfig, SessionConfig};
use types::{MemoryPersistenceConfig, SqlToolsConfig};

const DEFAULT_DURABLE_STREAM_KEY_PREFIX: &str = "events:";

/// A builder for constructing `AgentService` instances with a fluent API.
///
/// # Examples
///
/// ```rust,no_run
/// use caliche::{AgentBuilder, ProviderType};
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     let mut agent = AgentBuilder::new()
///         .provider(ProviderType::OpenAI)
///         .model("gpt-4o".to_string())
///         .system_prompt("You are a helpful assistant.".to_string())
///         .enable_tool("calculator")
///         .enable_tool("fs")
///         .build()
///         .await?;
///     
///     let response = agent.chat("Hello!").await?;
///     println!("{}", response);
///     
///     Ok(())
/// }
/// ```
#[derive(Clone)]
pub struct AgentBuilder {
    provider: Option<ProviderType>,
    model: Option<String>,
    provider_instance: Option<Arc<dyn crate::llm::Provider>>,
    embedding_provider_instance: Option<Arc<dyn crate::llm::EmbeddingProvider>>,
    image_generation_provider_instance: Option<Arc<dyn crate::llm::ImageGenerationProvider>>,
    artifact_store: Option<Arc<crate::artifacts::ArtifactStore>>,
    system_prompt: Option<String>,
    tools: Vec<Cow<'static, str>>,
    built_in_tools: Vec<crate::agent::tools::BuiltInTool>,
    custom_tools: Vec<Arc<dyn Tool>>,
    #[cfg(feature = "tools-media")]
    media_providers:
        Vec<Arc<dyn crate::agent::tools::media_understanding::MediaUnderstandingProvider>>,
    #[cfg(feature = "document-loaders")]
    document_processor: Option<crate::data::loaders::DocumentProcessor>,
    use_all_default_tools: bool,
    /// File mutation style: structured `fs` write/edit vs freeform `apply_patch` (XOR).
    mutation_style: crate::agent::tools::MutationStyle,
    /// When true, register the Goose-style tool inspection chain for permission/security.
    enable_tool_inspection: bool,
    /// Optional workspace root for path-security inspection.
    inspection_workspace_root: Option<std::path::PathBuf>,
    #[cfg(feature = "tools-rag")]
    shared_rag_tool: Option<Arc<RagTool>>,
    #[cfg(feature = "tools-rag")]
    workspace_indexing_ready: Option<Arc<WorkspaceIndexingReady>>,
    #[cfg(feature = "tools-rag")]
    workspace_config: Option<crate::workspace::WorkspaceIndexerConfig>,
    #[cfg(feature = "tools-rag")]
    workspace_manifest: Option<Arc<tokio::sync::Mutex<crate::workspace::manifest::IndexManifest>>>,
    memory_type: Option<MemoryType>,
    memory_persistence: Option<MemoryPersistenceConfig>,
    // Authentication
    credential_provider: Option<Arc<dyn CredentialProvider>>,
    /// When true, missing model keys do not fail `build`. Completions still fail.
    defer_missing_credentials: bool,
    base_url: Option<String>, // Still needed for LMStudio
    // Vector store configuration
    #[cfg(feature = "vector-memory")]
    vector_backend_config: Option<VectorBackendConfig>,
    /// Enable metrics collection for vector operations (RAG, vector memory, etc.)
    vector_metrics_enabled: bool,
    sql_tools: Option<SqlToolsConfig>,
    // Tool execution driver
    tool_execution_driver: Option<ToolExecutionDriver>,
    // Max iterations for tool calls
    max_iterations: Option<usize>,
    // Tool choice preference
    tool_choice: Option<crate::messages::ToolChoice>,
    // Lifecycle hooks registry
    hooks: HookRegistry,
    /// Shared slot updated by [`ReasoningScheduler`] and read when building completion requests.
    reasoning_effort_slot: Option<Arc<ReasoningEffortSlot>>,
    // Permission mode
    permission_mode: PermissionMode,
    // Allowed tools (if None, all tools are allowed)
    allowed_tools: Option<Vec<String>>,
    // MCP servers (in-process servers for direct tool registration)
    #[cfg(feature = "mcp")]
    mcp_servers: HashMap<String, Arc<crate::mcp::InProcessMcpServer>>,
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    remote_mcp_servers: Vec<crate::mcp::McpClientConfig>,
    /// When true, `build()` scans local MCP config files (e.g. `~/.cursor/mcp.json`)
    /// and attaches discovered remote servers. Defaults to false so SDK agents stay
    /// isolated unless the caller opts in (CLI enables this).
    #[cfg(all(feature = "mcp", feature = "mcp-client"))]
    auto_discover_mcp: bool,
    /// When true, `build()` indexes on-disk skills (`SKILL.md`) for progressive
    /// disclosure in the system prompt. Defaults to false so thin SDK / benchmark
    /// agents do not inherit workspace skill catalogs (CLI enables this).
    auto_discover_skills: bool,
    agent_retry_attempts: usize,
    session_autosave_path: Option<std::path::PathBuf>,
    /// AgentFS configuration: when set, uses AgentFS for session store and tool audit trail.
    #[cfg(feature = "agentfs")]
    agentfs_id: Option<String>,
    #[cfg(feature = "agentfs")]
    agentfs_path: Option<std::path::PathBuf>,
    // Response format for structured outputs
    response_format: Option<crate::output::ResponseFormat>,
    // Skills to apply to the agent
    skills: Vec<crate::agent::skills::Skill>,
    /// Optional research runner for the research tool (sub-agent with fs).
    research_runner: Option<Arc<dyn crate::agent::tools::ResearchRunner>>,
    /// Workspace embedding: auto-detect provider, index cwd, use RAG for context.
    #[cfg(feature = "tools-rag")]
    workspace_embedding: Option<WorkspaceEmbeddingMode>,
    /// Pre-discovered embedding provider, allowing callers to skip the discovery
    /// step during `build()` (e.g. when discovery ran concurrently with other startup work).
    #[cfg(feature = "tools-rag")]
    pre_discovered_embedding: Option<crate::discovery::embedding::EmbeddingProviderDiscovery>,
    /// RAG namespace for document isolation (e.g. per-session). When set, applied to workspace RAG.
    #[cfg(feature = "tools-rag")]
    rag_namespace: Option<String>,
    /// Workspace RAG `top_k` (number of chunks). When set, overrides default 12.
    #[cfg(feature = "tools-rag")]
    workspace_rag_top_k: Option<usize>,
    /// Workspace RAG minimum similarity (0.0–1.0). When set, overrides default 0.25.
    #[cfg(feature = "tools-rag")]
    workspace_rag_min_similarity: Option<f32>,
    /// Optional short-TTL cache for workspace RAG context (key: `query+top_k+min_sim`).
    #[cfg(feature = "tools-rag")]
    workspace_rag_context_cache: Option<Arc<dyn CacheBackend>>,
    /// TTL in seconds for workspace RAG context cache. Default 60.
    #[cfg(feature = "tools-rag")]
    workspace_rag_context_cache_ttl_secs: u64,
    /// Queryable graph backend for `graph_query` and `graph_explorer` tools.
    #[cfg(feature = "tools-graph")]
    graph_backend: Option<Arc<dyn dripstone::graph::QueryableGraphBackend>>,
    /// Frontier prioritizer for `graph_explorer` (e.g. `security_default()`).
    #[cfg(feature = "tools-graph")]
    frontier_prioritizer: Option<crate::agent::tools::graph::FrontierPrioritizer>,
    #[cfg(feature = "lsp")]
    lsp_config: Option<crate::lsp::LspConfig>,
    #[cfg(feature = "lsp")]
    lsp_manager: Option<Arc<crate::lsp::LspManager>>,
    /// Max tokens for LLM completion
    max_tokens: Option<u32>,
    /// Temperature for LLM completion
    temperature: Option<f32>,
    /// When set, controls whether to include "Current date and time" and "Current working directory" in the system message when tools/workspace are present. None = default (true). Set Some(false) for minimal prompt (benchmark parity, low latency).
    include_context_in_system: Option<bool>,
    /// Optional LLM response cache for completion requests (exact-match). When set, repeated prompts return cached responses.
    llm_cache: Option<Arc<LlmCache>>,
    /// When true and an embedding provider is available, use embedding-based tool schema ordering.
    use_embedding_tool_ordering: bool,
    /// Max characters for tool output kept in conversation history. When set, overrides default 1000.
    max_tool_output_in_history: Option<usize>,
    /// Override provider-reported context window size.
    context_window: Option<usize>,
    /// `HuggingFace` repository to download for local inference.
    local_inference_download_repo: Option<String>,
    /// Optional `HuggingFace` revision for the local inference download target.
    local_inference_download_revision: Option<String>,
    /// Local inference backend override (candle, ort, mlx).
    local_inference_backend: Option<String>,
    /// Exact directory where local inference artifacts should be stored.
    local_inference_model_dir: Option<PathBuf>,
    /// `HuggingFace` cache directory used during local inference bootstrap.
    local_inference_hf_cache_dir: Option<PathBuf>,
    /// When true, add "[truncated for context]" when tool output is summarized for history.
    annotate_tool_output_truncation: Option<bool>,
    /// Shared HTTP client for connection pooling across multiple agents (OpenAI-compat and other HTTP providers).
    #[cfg(feature = "reqwest")]
    shared_http_client: Option<Arc<reqwest::Client>>,
    /// Durable stream backend for event persistence (replay, resume).
    durable_stream_backend: Option<Arc<dyn crate::streams::DurableStreamBackend>>,
    /// Key prefix for durable stream keys (e.g. "events:" -> "`events:{session_id`}").
    durable_stream_key_prefix: Cow<'static, str>,
    /// Max events per stream for in-memory backend when using `durable_stream_capacity`.
    durable_stream_capacity: Option<usize>,
    /// TTL for auto-trim (best-effort; backends may use `MaxAgeSecs`).
    durable_stream_ttl: Option<std::time::Duration>,
    /// If set, in build() we create a SQLite durable stream backend at this path.
    #[cfg(feature = "durable-streams-sqlite")]
    durable_stream_sqlite_path: Option<String>,
    /// If set, in build() we create a Postgres durable stream backend with this URL.
    #[cfg(feature = "durable-streams-postgres")]
    durable_stream_postgres_url: Option<String>,
    /// If set, in build() we create an S3 durable stream backend (bucket, prefix, region).
    #[cfg(feature = "durable-streams-s3")]
    durable_stream_s3_config: Option<(String, String, Option<String>)>,
    /// When set, use context window guard with this compaction strategy instead of simple pruning.
    #[cfg(feature = "compaction")]
    compaction_strategy: Option<Arc<dyn crate::agent::core::compaction::CompactionStrategy>>,
    /// Trigger compaction when history exceeds this fraction of context window (e.g. 0.90). Used when `compaction_strategy` is set.
    #[cfg(feature = "compaction")]
    compaction_trigger_threshold: Option<f64>,
    /// Target fraction of context window after compaction (e.g. 0.70). Used when `compaction_strategy` is set.
    #[cfg(feature = "compaction")]
    compaction_target_threshold: Option<f64>,
    /// Max compaction attempts per turn. Used when `compaction_strategy` is set.
    #[cfg(feature = "compaction")]
    compaction_max_attempts: Option<u32>,
    /// When set, applied in `build()` and overrides individual compaction_* fields (when compaction feature is enabled).
    #[cfg(feature = "compaction")]
    compaction_config: Option<crate::agent::core::compaction::CompactionConfig>,
    /// Optional Jev asker (TypeSafe System One) for Jev-assisted compaction pre-filtering.
    #[cfg(feature = "compaction")]
    jev_asker: Option<Arc<dyn crate::agent::core::compaction_jev::JevAsker>>,
    /// Optional Jev compaction options.
    #[cfg(feature = "compaction")]
    jev_options: Option<crate::agent::core::compaction_jev::JevCompactOptions>,
    /// Optional spawner for `sessions_spawn` tool (when session-tools is enabled).
    #[cfg(feature = "session-tools")]
    session_spawner: Option<Arc<dyn SessionSpawner>>,
    /// When set, applied in `build()` and sets `session_spawner` (when session-tools is enabled).
    #[cfg(feature = "session-tools")]
    session_tools_config: Option<SessionToolsConfig>,
    /// Resolver for `RequireApproval` from hooks (e.g. CLI stdio prompt, API callback).
    approval_resolver: Option<Arc<dyn ApprovalResolver>>,
    /// Persister for `AllowAndPersist` outcomes (appends allow rules to config).
    amendment_persister: Option<Arc<dyn AmendmentPersister>>,
    /// Optional steering/follow-up provider; when set, follow-up messages are checked when the agent would stop.
    steering_followup_provider: Option<Arc<dyn SteeringFollowUpProvider>>,
    /// When set, in build() we create a plugin store at this path and register the plugin tool plus any persisted tool plugins.
    #[cfg(feature = "plugins")]
    plugins_store_path: Option<std::path::PathBuf>,
    /// How to handle failures when loading plugins from the configured store.
    #[cfg(feature = "plugins")]
    plugin_load_policy: crate::plugins::PluginLoadPolicy,
    /// Optional learning-loop configuration. When set, `build()` attaches a
    /// `LearningLoop` to the agent and registers the `skill_manage` tool.
    #[cfg(feature = "learning-loop")]
    learning_config: Option<crate::agent::learning::CheckpointConfig>,
    /// Custom operation manager (local or remote). Default `None` keeps the
    /// existing tool-engine path.
    operation_manager: Option<Arc<dyn crate::agent::operations::OperationManager>>,
    /// When true (or `CALICHE_OPERATION_DISPATCH=1`), install a local manager.
    enable_operation_dispatch: bool,
}

/// Workspace embedding mode when enabled.
#[cfg(feature = "tools-rag")]
#[derive(Clone)]
pub enum WorkspaceEmbeddingMode {
    /// Auto-detect embedding provider, index current directory with default config.
    Auto,
    /// Use provided workspace indexer config.
    Custom(WorkspaceIndexerConfig),
}

impl AgentBuilder {
    /// Create a new agent builder with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            provider: None,
            model: None,
            provider_instance: None,
            embedding_provider_instance: None,
            image_generation_provider_instance: None,
            artifact_store: None,
            system_prompt: None,
            tools: Vec::new(),
            built_in_tools: Vec::new(),
            custom_tools: Vec::new(),
            #[cfg(feature = "tools-media")]
            media_providers: Vec::new(),
            #[cfg(feature = "document-loaders")]
            document_processor: None,
            use_all_default_tools: false,
            mutation_style: crate::agent::tools::MutationStyle::EditWrite,
            enable_tool_inspection: false,
            inspection_workspace_root: None,
            memory_type: None,
            memory_persistence: None,
            credential_provider: None,
            defer_missing_credentials: false,
            base_url: None,
            #[cfg(feature = "vector-memory")]
            vector_backend_config: None,
            vector_metrics_enabled: false,
            #[cfg(feature = "tools-rag")]
            shared_rag_tool: None,
            #[cfg(feature = "tools-rag")]
            workspace_indexing_ready: None,
            #[cfg(feature = "tools-rag")]
            workspace_config: None,
            #[cfg(feature = "tools-rag")]
            workspace_manifest: None,
            sql_tools: None,
            tool_execution_driver: None,
            #[cfg(feature = "plugins")]
            plugins_store_path: None,
            #[cfg(feature = "plugins")]
            plugin_load_policy: crate::plugins::PluginLoadPolicy::from_env(),
            #[cfg(feature = "learning-loop")]
            learning_config: None,
            max_iterations: None,
            tool_choice: None,
            reasoning_effort_slot: None,
            hooks: HookRegistry::new(),
            permission_mode: PermissionMode::Default,
            allowed_tools: None,
            #[cfg(feature = "mcp")]
            mcp_servers: HashMap::new(),
            #[cfg(all(feature = "mcp", feature = "mcp-client"))]
            remote_mcp_servers: Vec::new(),
            #[cfg(all(feature = "mcp", feature = "mcp-client"))]
            auto_discover_mcp: false,
            auto_discover_skills: false,
            agent_retry_attempts: 1,
            session_autosave_path: None,
            #[cfg(feature = "agentfs")]
            agentfs_id: None,
            #[cfg(feature = "agentfs")]
            agentfs_path: None,
            response_format: None,
            skills: Vec::new(),
            research_runner: None,
            #[cfg(feature = "tools-rag")]
            workspace_embedding: None,
            #[cfg(feature = "tools-rag")]
            pre_discovered_embedding: None,
            #[cfg(feature = "tools-rag")]
            rag_namespace: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_context_cache: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_context_cache_ttl_secs: 60,
            #[cfg(feature = "tools-graph")]
            graph_backend: None,
            #[cfg(feature = "tools-graph")]
            frontier_prioritizer: None,
            #[cfg(feature = "lsp")]
            lsp_config: None,
            #[cfg(feature = "lsp")]
            lsp_manager: None,
            max_tokens: None,
            temperature: None,
            include_context_in_system: None,
            llm_cache: None,
            use_embedding_tool_ordering: false,
            max_tool_output_in_history: None,
            context_window: None,
            local_inference_download_repo: None,
            local_inference_download_revision: None,
            local_inference_backend: None,
            local_inference_model_dir: None,
            local_inference_hf_cache_dir: None,
            annotate_tool_output_truncation: None,
            #[cfg(feature = "reqwest")]
            shared_http_client: None,
            durable_stream_backend: None,
            durable_stream_key_prefix: Cow::borrowed(DEFAULT_DURABLE_STREAM_KEY_PREFIX),
            durable_stream_capacity: None,
            durable_stream_ttl: None,
            #[cfg(feature = "durable-streams-sqlite")]
            durable_stream_sqlite_path: None,
            #[cfg(feature = "durable-streams-postgres")]
            durable_stream_postgres_url: None,
            #[cfg(feature = "durable-streams-s3")]
            durable_stream_s3_config: None,
            #[cfg(feature = "compaction")]
            compaction_strategy: None,
            #[cfg(feature = "compaction")]
            compaction_trigger_threshold: None,
            #[cfg(feature = "compaction")]
            compaction_target_threshold: None,
            #[cfg(feature = "compaction")]
            compaction_max_attempts: None,
            #[cfg(feature = "compaction")]
            compaction_config: None,
            #[cfg(feature = "compaction")]
            jev_asker: None,
            #[cfg(feature = "compaction")]
            jev_options: None,
            #[cfg(feature = "session-tools")]
            session_spawner: None,
            #[cfg(feature = "session-tools")]
            session_tools_config: None,
            approval_resolver: None,
            amendment_persister: None,
            steering_followup_provider: None,
            operation_manager: None,
            enable_operation_dispatch: false,
        }
    }

    /// When true (default), include "Current date and time" and "Current working directory"
    /// in the system message when tools or workspace are present. When false, omit for minimal
    /// prompt (benchmark parity, low latency).
    #[must_use]
    pub const fn include_context_in_system(mut self, value: bool) -> Self {
        self.include_context_in_system = Some(value);
        self
    }

    /// Use `manager` for translate → commit → execute instead of the default
    /// engine-only path.
    ///
    /// [`crate::agent::operations::LocalOperationManager`] is audit-only: it
    /// commits specs, then [`crate::agent::tools::execution::ToolExecutionEngine`]
    /// runs the calls (sandbox, cancel, workspace root, metrics, AgentFS, hook
    /// deny, parallel batching). [`crate::agent::operations::RemoteOperationManager`]
    /// ships specs to a worker and fails closed when a sandbox is configured
    /// unless the transport reports `isolates_in_sandbox()` and binds the
    /// agent's configured sandbox handle (`SandboxOperationTransport`).
    #[must_use]
    pub fn with_operation_manager(
        mut self,
        manager: Arc<dyn crate::agent::operations::OperationManager>,
    ) -> Self {
        self.operation_manager = Some(manager);
        self
    }

    /// Install [`crate::agent::operations::LocalOperationManager`] at build time.
    ///
    /// Equivalent to `CALICHE_OPERATION_DISPATCH=1`. Isolation stays on the
    /// engine path; this does not drop the OS sandbox.
    #[must_use]
    pub const fn enable_operation_dispatch(mut self) -> Self {
        self.enable_operation_dispatch = true;
        self
    }

    /// Enable optional LLM response cache for completion requests (exact-match).
    /// When set, repeated identical prompts return cached responses without calling the provider.
    pub fn with_llm_cache(mut self, cache: Arc<LlmCache>) -> Self {
        self.llm_cache = Some(cache);
        self
    }

    /// Use embedding-based tool schema ordering when an embedding provider is available.
    #[must_use]
    pub const fn use_embedding_tool_ordering(mut self, value: bool) -> Self {
        self.use_embedding_tool_ordering = value;
        self
    }

    /// Set max characters for tool output kept in conversation history. Default 1000.
    #[must_use]
    pub const fn max_tool_output_in_history(mut self, max_chars: usize) -> Self {
        self.max_tool_output_in_history = Some(max_chars);
        self
    }

    /// Override the provider-reported context window size.
    ///
    /// The value is the full model context window. Caliche still reserves
    /// provider-specific output tokens before assembling prompts.
    #[must_use]
    pub const fn context_window(mut self, tokens: usize) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// Set the `HuggingFace` repository/revision to download for local inference.
    pub fn local_inference_download_target(
        mut self,
        repo: impl Into<String>,
        revision: Option<String>,
    ) -> Self {
        self.local_inference_download_repo = Some(repo.into());
        self.local_inference_download_revision = revision;
        self
    }

    /// Set the local inference backend (`candle`, `ort`, or `mlx`).
    pub fn local_inference_backend(mut self, backend: impl Into<String>) -> Self {
        self.local_inference_backend = Some(backend.into());
        self
    }

    /// Set the exact directory where local inference model artifacts are stored.
    pub fn local_inference_model_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.local_inference_model_dir = Some(dir.into());
        self
    }

    /// Set the `HuggingFace` cache directory used during local inference bootstrap.
    pub fn local_inference_hf_cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.local_inference_hf_cache_dir = Some(dir.into());
        self
    }

    /// Set whether to add "[truncated for context]" when tool output is summarized for history.
    #[must_use]
    pub const fn annotate_tool_output_truncation(mut self, value: bool) -> Self {
        self.annotate_tool_output_truncation = Some(value);
        self
    }

    #[cfg(feature = "local-inference-base")]
    fn local_inference_config_for_builder(&self) -> Option<crate::llm::LocalInferenceConfig> {
        if !matches!(self.provider, Some(ProviderType::LocalInference)) {
            return None;
        }

        let has_download_options = self.local_inference_download_repo.is_some()
            || self.local_inference_backend.is_some()
            || self.local_inference_model_dir.is_some()
            || self.local_inference_hf_cache_dir.is_some();
        if !has_download_options {
            return None;
        }

        let model = self
            .local_inference_download_repo
            .as_deref()
            .or(self.model.as_deref())
            .unwrap_or("hf-internal-testing/tiny-random-LlamaForCausalLM");
        let backend = if let Some(backend) = &self.local_inference_backend {
            crate::llm::LocalInferenceProvider::parse_backend(backend)?
        } else {
            crate::llm::LocalInferenceProvider::resolve_backend_for_model(model).ok()?
        };
        let mut config = crate::llm::LocalInferenceConfig::new(
            dripstone::loader::ModelSource::from(model),
            backend,
        );

        if let Some(repo) = &self.local_inference_download_repo {
            config = config
                .with_download_target(repo.clone(), self.local_inference_download_revision.clone());
        } else if config.source.is_huggingface() {
            let source = config.source.clone();
            config = config.with_bootstrap(
                crate::llm::LocalInferenceProvider::bootstrap_for_source(&source, backend),
            );
        }
        if let Some(dir) = &self.local_inference_model_dir {
            config = config.with_model_storage_dir(dir.clone());
        }
        if let Some(dir) = &self.local_inference_hf_cache_dir {
            config = config.with_huggingface_cache_dir(dir.clone());
        }

        // Honor the same env knobs as `LocalInferenceProvider::config_from_env`.
        if let Ok(device) = crate::llm::env_var_nonempty("CALICHE_LOCAL_INFERENCE_DEVICE")
            && let Some(parsed) = crate::llm::LocalInferenceProvider::parse_device(&device)
        {
            config = config.with_device(parsed);
        }
        if let Ok(max_tokens) = crate::llm::env_var_nonempty("CALICHE_LOCAL_INFERENCE_MAX_TOKENS")
            && let Ok(parsed) = max_tokens.parse::<usize>()
        {
            config = config.with_max_tokens(parsed);
        }

        Some(config)
    }

    /// Set compaction strategy for context window (LLM summarization, sliding window, or importance scoring).
    #[cfg(feature = "compaction")]
    pub fn compaction_strategy(
        mut self,
        strategy: Arc<dyn crate::agent::core::compaction::CompactionStrategy>,
    ) -> Self {
        self.compaction_strategy = Some(strategy);
        self
    }

    /// Set fraction of context window that triggers compaction (e.g. 0.90). Used when `compaction_strategy` is set.
    #[cfg(feature = "compaction")]
    #[must_use]
    pub const fn compaction_trigger_threshold(mut self, f: f64) -> Self {
        self.compaction_trigger_threshold = Some(f);
        self
    }

    /// Set target fraction of context window after compaction (e.g. 0.70). Used when `compaction_strategy` is set.
    #[cfg(feature = "compaction")]
    #[must_use]
    pub const fn compaction_target_threshold(mut self, f: f64) -> Self {
        self.compaction_target_threshold = Some(f);
        self
    }

    /// Set max compaction attempts per chat turn. Used when `compaction_strategy` is set.
    #[cfg(feature = "compaction")]
    #[must_use]
    pub const fn compaction_max_attempts(mut self, n: u32) -> Self {
        self.compaction_max_attempts = Some(n);
        self
    }

    /// Set compaction from a single config (strategy + guard thresholds). Overrides individual compaction_* when applied in `build()`.
    #[cfg(feature = "compaction")]
    #[must_use]
    pub fn compaction_config(
        mut self,
        config: crate::agent::core::compaction::CompactionConfig,
    ) -> Self {
        self.compaction_config = Some(config);
        self
    }

    /// Set Jev asker (TypeSafe System One) for Jev-assisted compaction pre-filtering.
    #[cfg(feature = "compaction")]
    pub fn jev_asker(
        mut self,
        asker: Arc<dyn crate::agent::core::compaction_jev::JevAsker>,
    ) -> Self {
        self.jev_asker = Some(asker);
        self
    }

    /// Set options for Jev-assisted compaction pre-filtering.
    #[cfg(feature = "compaction")]
    #[must_use]
    pub fn jev_options(
        mut self,
        options: crate::agent::core::compaction_jev::JevCompactOptions,
    ) -> Self {
        self.jev_options = Some(options);
        self
    }

    /// Set the session spawner for `sessions_spawn` tool (when session-tools is enabled).
    #[cfg(feature = "session-tools")]
    pub fn session_spawner(mut self, spawner: Arc<dyn SessionSpawner>) -> Self {
        self.session_spawner = Some(spawner);
        self
    }

    /// Set session tools from a single config (e.g. spawner). Applied in `build()`; overrides `session_spawner` when set.
    #[cfg(feature = "session-tools")]
    #[must_use]
    pub fn session_tools_config(mut self, config: SessionToolsConfig) -> Self {
        self.session_tools_config = Some(config);
        self
    }

    /// Use a shared HTTP client for connection pooling across multiple agents.
    ///
    /// When building several agents that hit the same API (e.g. `OpenAI`), pass the same
    /// `Arc<reqwest::Client>` to reuse connections and reduce latency.
    #[cfg(feature = "reqwest")]
    #[must_use]
    pub fn shared_http_client(mut self, client: Arc<reqwest::Client>) -> Self {
        self.shared_http_client = Some(client);
        self
    }

    /// Set max tokens for LLM completion.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set temperature for LLM completion.
    #[must_use]
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    #[must_use]
    pub const fn tool_execution_driver(mut self, driver: ToolExecutionDriver) -> Self {
        self.tool_execution_driver = Some(driver);
        self
    }

    /// Configure top-level chat retry attempts for transient failures.
    #[must_use]
    pub fn retry_attempts(mut self, attempts: usize) -> Self {
        self.agent_retry_attempts = attempts.max(1);
        self
    }

    /// Enable session autosave to a directory.
    pub fn session_autosave_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.session_autosave_path = Some(path.into());
        self
    }

    /// Configure session persistence and storage using a [`SessionConfig`] struct.
    ///
    /// This is an ergonomic alternative to chaining `session_autosave_path`,
    /// `with_agentfs`, etc. individually.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentBuilder;
    /// use caliche::agent::builder::SessionConfig;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .with_session_config(SessionConfig {
    ///         autosave_path: Some("/var/sessions".into()),
    ///         ..Default::default()
    ///     })
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn with_session_config(mut self, config: SessionConfig) -> Self {
        if let Some(path) = config.autosave_path {
            self.session_autosave_path = Some(path);
        }
        #[cfg(feature = "agentfs")]
        {
            if let Some(id) = config.agentfs_id {
                self.agentfs_id = Some(id);
            }
            if let Some(path) = config.agentfs_path {
                self.agentfs_path = Some(path);
            }
        }
        self
    }

    /// Enable AgentFS for session storage and tool call audit trail.
    /// Creates `.agentfs/{id}.db` for persistent, queryable agent state.
    #[cfg(feature = "agentfs")]
    pub fn with_agentfs(mut self, id: impl Into<String>) -> Self {
        self.agentfs_id = Some(id.into());
        self
    }

    /// Enable AgentFS with a custom database path.
    #[cfg(feature = "agentfs")]
    pub fn with_agentfs_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.agentfs_path = Some(path.into());
        self
    }

    /// Configure observability using an [`ObservabilityConfig`] struct.
    ///
    /// Applies vector metrics settings. The optional `hook` field is reserved for a
    /// future lifecycle integration and is currently ignored.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentBuilder;
    /// use caliche::agent::builder::ObservabilityConfig;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::new()
    ///     .with_observability_config(ObservabilityConfig {
    ///         vector_metrics_enabled: true,
    ///         hook: None,
    ///     })
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn with_observability_config(mut self, config: ObservabilityConfig) -> Self {
        self.vector_metrics_enabled = config.vector_metrics_enabled;
        if config.hook.is_some() {
            tracing::warn!(
                "ObservabilityConfig.hook is reserved and currently ignored by AgentBuilder"
            );
        }
        self
    }

    /// Create an agent builder from a template.
    ///
    /// This loads a template from a file and applies its configuration to the builder.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use caliche::AgentBuilder;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let agent = AgentBuilder::from_template("templates/rag_agent.yaml")
    ///     .await?
    ///     .build()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "templates")]
    pub async fn from_template<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
        use crate::templates::TemplateLoader;
        let loader = TemplateLoader::new();
        let template = loader.load_template(path).await?;
        template.instantiate().await
    }
}
impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_sets_provider_model_and_temperature() {
        let builder = AgentBuilder::new()
            .provider(ProviderType::OpenAI)
            .model("gpt-4o".to_string())
            .temperature(0.2)
            .system_prompt("Test prompt".to_string())
            .enable_tool("calculator")
            .enable_tool("fs");

        assert_eq!(builder.provider, Some(ProviderType::OpenAI));
        assert_eq!(builder.model.as_deref(), Some("gpt-4o"));
        assert_eq!(builder.temperature, Some(0.2));
        assert_eq!(builder.system_prompt.as_deref(), Some("Test prompt"));
        assert_eq!(builder.tools.len(), 2);
    }

    #[test]
    fn typed_tool_configuration_filters_allowed_builtins() {
        let builder = AgentBuilder::new()
            .enable_builtin_tool(crate::agent::tools::BuiltInTool::Calculator)
            .enable_builtin_tools(&[
                crate::agent::tools::BuiltInTool::Fs,
                crate::agent::tools::BuiltInTool::Http,
            ])
            .allowed_builtin_tools(&[
                crate::agent::tools::BuiltInTool::Calculator,
                crate::agent::tools::BuiltInTool::Fs,
            ]);

        assert_eq!(builder.built_in_tools.len(), 3);
        assert_eq!(
            builder
                .allowed_tools
                .expect("allowed tools should be configured"),
            vec!["calculator".to_string(), "fs".to_string()]
        );
    }

    #[test]
    fn default_builder_leaves_provider_model_and_tools_unset() {
        let builder = AgentBuilder::default();
        assert!(builder.provider.is_none());
        assert!(builder.model.is_none());
        assert!(builder.system_prompt.is_none());
        assert!(builder.tools.is_empty());
        assert!(builder.built_in_tools.is_empty());
    }

    async fn mock_service(builder: AgentBuilder) -> AgentService {
        builder
            .provider_instance(Arc::new(crate::testing::MockProvider::new(vec![
                "ok".into(),
            ])))
            .model("mock".to_string())
            .build()
            .await
            .expect("build mock service")
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn default_build_leaves_operation_manager_unset() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("CALICHE_OPERATION_DISPATCH");
        let service = Box::pin(mock_service(AgentBuilder::new())).await;
        assert!(
            !service.agent().operation_dispatch_enabled(),
            "default path must keep ToolExecutionEngine"
        );
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn enable_operation_dispatch_installs_local_manager() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("CALICHE_OPERATION_DISPATCH");
        let service = Box::pin(mock_service(
            AgentBuilder::new().enable_operation_dispatch(),
        ))
        .await;
        assert!(service.agent().operation_dispatch_enabled());
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn with_operation_manager_installs_supplied_manager() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("CALICHE_OPERATION_DISPATCH");
        let manager = Arc::new(crate::agent::operations::LocalOperationManager::new());
        let service = Box::pin(mock_service(
            AgentBuilder::new().with_operation_manager(manager),
        ))
        .await;
        assert!(service.agent().operation_dispatch_enabled());
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn env_operation_dispatch_installs_local_manager() {
        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.set("CALICHE_OPERATION_DISPATCH", "1");
        let result = Box::pin(mock_service(AgentBuilder::new())).await;
        assert!(result.agent().operation_dispatch_enabled());
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    #[serial_test::serial]
    async fn install_refuses_non_isolating_remote_when_sandbox_configured() {
        use crate::agent::operations::{RecordingTransport, RemoteOperationManager};
        use caliche_sandbox::backends::noop::NoopBackend;
        use caliche_sandbox::config::{NamedPolicy, ToolPolicy, policy_to_config};
        use caliche_sandbox::handle::SandboxHandle;
        use caliche_sandbox::traits::SandboxBackend;

        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("CALICHE_OPERATION_DISPATCH");
        let mut service = Box::pin(mock_service(AgentBuilder::new())).await;
        let workspace = tempfile::tempdir().expect("workspace");
        let config = policy_to_config(
            NamedPolicy::ReadOnly,
            workspace.path(),
            None,
            caliche_sandbox::config::SandboxBackendType::Noop,
        );
        let instance = NoopBackend
            .create(&config, workspace.path())
            .await
            .expect("noop sandbox");
        service.agent_mut().set_sandbox(Arc::new(SandboxHandle::new(
            instance,
            ToolPolicy::default(),
        )));
        let manager = Arc::new(RemoteOperationManager::new(Arc::new(
            RecordingTransport::default(),
        )));
        let err = super::build_helpers::install_operation_manager(&mut service, manager)
            .expect_err("sandbox + RecordingTransport must fail closed");
        assert!(
            err.to_string().contains("SandboxOperationTransport"),
            "refuse reason: {err}"
        );
        assert!(
            !service.agent().operation_dispatch_enabled(),
            "manager must not be installed after refuse"
        );
    }

    #[cfg(feature = "sandbox")]
    async fn noop_sandbox_handle() -> Arc<caliche_sandbox::handle::SandboxHandle> {
        use caliche_sandbox::backends::noop::NoopBackend;
        use caliche_sandbox::config::{NamedPolicy, ToolPolicy, policy_to_config};
        use caliche_sandbox::handle::SandboxHandle;
        use caliche_sandbox::traits::SandboxBackend;

        let workspace = Box::leak(Box::new(tempfile::tempdir().expect("workspace")));
        let config = policy_to_config(
            NamedPolicy::ReadOnly,
            workspace.path(),
            None,
            caliche_sandbox::config::SandboxBackendType::Noop,
        );
        let instance = NoopBackend
            .create(&config, workspace.path())
            .await
            .expect("noop sandbox");
        Arc::new(SandboxHandle::new(instance, ToolPolicy::default()))
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    #[serial_test::serial]
    async fn install_refuses_isolating_remote_bound_to_other_sandbox() {
        use crate::agent::operations::{RemoteOperationManager, SandboxOperationTransport};

        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("CALICHE_OPERATION_DISPATCH");
        let mut service = Box::pin(mock_service(AgentBuilder::new())).await;
        let configured = noop_sandbox_handle().await;
        let other = noop_sandbox_handle().await;
        service.agent_mut().set_sandbox(configured);
        let manager = Arc::new(RemoteOperationManager::new(Arc::new(
            SandboxOperationTransport::new(other),
        )));
        let err = super::build_helpers::install_operation_manager(&mut service, manager)
            .expect_err("mismatched sandbox handle must fail closed");
        assert!(
            err.to_string().contains("configured sandbox handle"),
            "refuse reason: {err}"
        );
        assert!(
            !service.agent().operation_dispatch_enabled(),
            "manager must not be installed after refuse"
        );
    }

    #[cfg(feature = "sandbox")]
    #[tokio::test]
    #[serial_test::serial]
    async fn install_accepts_isolating_remote_bound_to_agent_sandbox() {
        use crate::agent::operations::{RemoteOperationManager, SandboxOperationTransport};

        #[expect(unsafe_code, reason = "takes the process env guard; see SAFETY")]
        // SAFETY: this test is serial and starts no thread that reads the environment while the guard lives.
        let mut env = unsafe { caliche_test_env::EnvGuard::acquire() };
        env.remove("CALICHE_OPERATION_DISPATCH");
        let mut service = Box::pin(mock_service(AgentBuilder::new())).await;
        let handle = noop_sandbox_handle().await;
        service.agent_mut().set_sandbox(Arc::clone(&handle));
        let manager = Arc::new(RemoteOperationManager::new(Arc::new(
            SandboxOperationTransport::new(handle),
        )));
        super::build_helpers::install_operation_manager(&mut service, manager)
            .expect("same sandbox handle must install");
        assert!(service.agent().operation_dispatch_enabled());
    }

    #[cfg(feature = "lsp")]
    #[test]
    fn enable_lsp_tools_registers_diagnostics_surface() {
        let builder = AgentBuilder::new().enable_lsp_tools();
        assert_eq!(builder.tools.len(), 3);
        assert!(
            builder
                .tools
                .iter()
                .any(|tool| tool.as_ref() == "lsp_diagnostics")
        );
    }
}
