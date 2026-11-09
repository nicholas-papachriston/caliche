//! Core agent implementation

use crate::agent::hooks::HookRegistry;
use crate::agent::hooks::SteeringFollowUpProvider;
use crate::agent::hooks::approval::{AmendmentPersister, ApprovalResolver};
use crate::agent::hooks::permission::PermissionMode;
use crate::agent::kernel::SharedState;
use crate::agent::memory::{TokenCounter, TokenCounterFactory};
use crate::agent::tools::{CallDeduplicator, ExecutionHistoryStore, ToolRegistry};
use crate::llm::{EmbeddingProvider, Provider};
use crate::utils::user_context::UserContext;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use tokio::sync::Mutex;

#[cfg(feature = "tools-rag")]
use crate::rag::RagService;
#[cfg(feature = "tools-rag")]
use crate::traits::cache::CacheBackend;

pub(crate) mod agent_config;
pub(crate) mod agent_skills;
pub(crate) mod cache;
pub mod chat;
pub(crate) mod chat_helpers;
pub(crate) mod chat_loop;
#[cfg(feature = "compaction")]
pub mod compaction;
#[cfg(feature = "compaction")]
pub mod compaction_jev;
pub(crate) mod context_budget;
#[cfg(feature = "context-engine")]
pub mod context_engine;
#[cfg(feature = "compaction")]
pub mod context_window_guard;
pub(crate) mod hook_runner;
pub mod loop_detection;
pub mod message_builder;
pub mod message_history;
pub(crate) mod runtime;
pub(crate) mod step;
pub mod stream_chat;
pub(crate) mod stream_chat_loop;
pub mod tool_extraction;
pub(crate) mod tool_history;
pub(crate) mod tool_ordering;
pub(crate) mod tool_output;

// Explicit exports from chat module
pub use chat::StepResult;
// TurnResult is exported from agent::kernel

// Explicit exports from message_builder module
pub use message_builder::{build_system_message, default_system_prompt};

// Explicit exports from tool_extraction module
pub use tool_extraction::extract_tool_calls_from_text;

/// Cached system message store:
/// `(hour_key, include_context_in_system, include_tool_descriptions) -> messages`.
pub(crate) type CachedSystemMessages =
    Arc<parking_lot::RwLock<Option<(u32, bool, bool, Vec<crate::messages::Message>)>>>;
/// Cached system messages when workspace context is present:
/// `(hour_key, include_context, include_tool_descriptions, workspace_hash) -> messages`.
#[cfg(feature = "tools-rag")]
pub(crate) type CachedSystemMessagesWithWorkspace =
    Arc<parking_lot::RwLock<Option<(u32, bool, bool, u64, Vec<crate::messages::Message>)>>>;

/// Controls how the agent drives tool availability and execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionDriver {
    /// Current behaviour (LLM decides tool calls; execute as extracted).
    #[default]
    Off,
    /// Deduplicate redundant tool calls, but do not change execution structure.
    DedupOnly,
    /// Planner-first + phased execution with LLM interleaving between phases.
    PlannerFirstPhased,
}

impl ToolExecutionDriver {
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let value = s.trim();
        if value.eq_ignore_ascii_case("off") {
            Some(Self::Off)
        } else if value.eq_ignore_ascii_case("dedup_only")
            || value.eq_ignore_ascii_case("dedup-only")
            || value.eq_ignore_ascii_case("dedup")
        {
            Some(Self::DedupOnly)
        } else if value.eq_ignore_ascii_case("planner_first_phased")
            || value.eq_ignore_ascii_case("planner-first-phased")
            || value.eq_ignore_ascii_case("planner_first")
            || value.eq_ignore_ascii_case("planner-first")
            || value.eq_ignore_ascii_case("phased")
        {
            Some(Self::PlannerFirstPhased)
        } else {
            None
        }
    }
}

#[derive(Debug)]
pub(crate) struct ToolIntelligenceState {
    pub last_planned_query: Option<String>,
    pub phases: Option<Vec<Vec<String>>>,
    pub phase_idx: usize,
    pub deduplicator: CallDeduplicator,
    pub execution_history: ExecutionHistoryStore,
}

impl ToolIntelligenceState {
    pub fn new() -> Self {
        Self {
            last_planned_query: None,
            phases: None,
            phase_idx: 0,
            deduplicator: CallDeduplicator::new(),
            execution_history: ExecutionHistoryStore::new(),
        }
    }

    pub fn reset_plan(&mut self) {
        self.phases = None;
        self.phase_idx = 0;
    }
}

impl Default for ToolIntelligenceState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct ToolCallingConfig {
    pub tool_schemas: Arc<Vec<crate::messages::ToolSchema>>,
    pub use_native: bool,
    pub include_tool_descriptions: bool,
    pub supports_text_based_tools: bool,
}

/// Default maximum context tokens before pruning (128K tokens).
const DEFAULT_MAX_CONTEXT_TOKENS: usize = 128_000;

#[derive(Clone)]
pub struct Agent {
    pub(crate) llm_provider: Arc<dyn Provider>,
    pub tools: Arc<ToolRegistry>,
    pub(crate) system_prompt: String,
    /// Progressive skill index (name/description/location); full bodies load via `fs`.
    pub(crate) skill_index: Vec<crate::agent::skills::SkillIndexEntry>,
    pub(crate) current_dir: String,
    pub shared_state: SharedState,
    /// Token counter used for consistent context budgeting.
    pub(crate) token_counter: Arc<dyn TokenCounter>,
    /// Per-request user context (not stored in `ToolRegistry` to avoid cloning).
    pub(crate) user_context: Option<UserContext>,
    /// Per-request metrics tracker (not stored in `ToolRegistry` to avoid cloning).
    pub(crate) metrics: Option<Arc<crate::observability::ToolMetrics>>,
    /// Session identifier propagated to lifecycle hooks.
    pub(crate) session_id: Option<String>,
    /// Maximum number of tool call iterations before stopping (default: 50).
    pub(crate) max_iterations: usize,
    /// Tool choice preference (default: Auto).
    pub(crate) tool_choice: Option<crate::messages::ToolChoice>,
    /// Lifecycle hooks registry.
    pub(crate) hooks: HookRegistry,
    /// Provider tool-use id to UUID v7, shared with harness tool hooks.
    pub(crate) action_ids: Arc<std::sync::Mutex<HashMap<String, String>>>,
    /// Permission mode for automatic permission handling.
    pub(crate) permission_mode: PermissionMode,
    /// Allowed tools (None = all tools allowed). `HashSet` for O(1) lookup.
    pub(crate) allowed_tools: Option<Arc<HashSet<String>>>,
    /// Count of tool calls denied by the allowed-tools / readonly filter.
    pub(crate) readonly_filter_hits: Arc<AtomicUsize>,
    /// Maximum context tokens before aggressive pruning.
    pub(crate) max_context_tokens: usize,
    /// Tool execution driver mode.
    pub(crate) tool_execution_driver: ToolExecutionDriver,
    pub(crate) tool_intelligence: Arc<Mutex<ToolIntelligenceState>>,
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag: Option<Arc<RagService>>,
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_indexing_ready:
        Option<Arc<crate::agent::workspace_indexing::WorkspaceIndexingReady>>,
    /// Workspace RAG `top_k` (number of chunks to retrieve). Default 12.
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag_top_k: usize,
    /// Workspace RAG minimum similarity threshold (0.0–1.0). Default 0.25.
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag_min_similarity: f32,
    /// Hybrid dense+sparse retrieval for injected workspace context.
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag_line_level_bm25: bool,
    /// MMR / temporal-decay options for injected workspace context.
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag_search_options: Option<crate::rag::RagSearchOptions>,
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag_context_cache: Option<Arc<dyn CacheBackend>>,
    #[cfg(feature = "tools-rag")]
    pub(crate) workspace_rag_context_cache_ttl_secs: u64,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) temperature: Option<f32>,
    /// When `true`, append "Current date and time" and "Current working directory" to
    /// the system message. Set `false` for minimal prompt (benchmark parity, low latency).
    pub(crate) include_context_in_system: bool,
    /// Cached tools description for system message.
    cached_tools_description: Arc<arc_swap::ArcSwap<Option<String>>>,
    /// Cached system messages to avoid rebuilding on every chat call.
    cached_system_messages: CachedSystemMessages,
    #[cfg(feature = "tools-rag")]
    cached_system_messages_with_workspace: CachedSystemMessagesWithWorkspace,
    /// Provider context window limit (computed at construction).
    pub(crate) cached_context_window: Option<usize>,
    #[cfg(feature = "compaction")]
    pub(crate) compaction_guard: Option<Arc<context_window_guard::ContextWindowGuard>>,
    #[cfg(feature = "context-engine")]
    pub(crate) context_engine: Option<Arc<dyn context_engine::ContextEngine>>,
    /// Optional embedding provider for tool schema ordering by query similarity.
    pub(crate) tool_ordering_embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Max characters for tool output kept in conversation history. Default `12_000`.
    pub(crate) max_tool_output_in_history: usize,
    /// When `true`, add "[truncated for context]" when tool output is summarized.
    pub(crate) annotate_tool_output_truncation: bool,
    #[cfg(feature = "agentfs")]
    pub(crate) agentfs_tool_logger: Option<Arc<crate::agent::tools::AgentFsToolLogger>>,
    #[cfg(feature = "sandbox")]
    pub(crate) sandbox: Option<Arc<caliche_sandbox::SandboxHandle>>,
    /// Resolver for `RequireApproval` from hooks; when None, `RequireApproval` is treated as Deny
    pub(crate) approval_resolver: Option<Arc<dyn ApprovalResolver>>,
    /// Persister for `AllowAndPersist` outcomes; when None, rule is not persisted
    pub(crate) amendment_persister: Option<Arc<dyn AmendmentPersister>>,
    /// Optional provider for steering/follow-up messages; when set, follow-up messages are checked when the agent would stop.
    pub(crate) steering_followup_provider: Option<Arc<dyn SteeringFollowUpProvider>>,
    #[cfg(feature = "learning-loop")]
    pub(crate) learning_loop: Option<Arc<crate::agent::learning::LearningLoop>>,
    /// Shared reasoning-effort hint written by [`crate::agent::hooks::ReasoningScheduler`].
    pub(crate) reasoning_effort_slot: Option<Arc<crate::agent::hooks::ReasoningEffortSlot>>,
    /// Optional operation manager (translate → commit → execute). Default `None`
    /// keeps the existing `ToolExecutionEngine` path.
    pub(crate) operation_manager: Option<Arc<dyn crate::agent::operations::OperationManager>>,
    /// Translators used when `operation_manager` is set.
    pub(crate) operation_translators: Option<Arc<crate::agent::operations::TranslatorRegistry>>,
    /// Last compaction report (Jev + LLM / sliding-window), interior-mutable.
    pub(crate) last_compaction_report:
        Arc<parking_lot::RwLock<Option<crate::messages::CompactionReport>>>,
}

impl Agent {
    /// Create an agent with a provider and tool registry.
    ///
    /// Prefer [`AgentBuilder`](crate::AgentBuilder) for constructing agents with full
    /// configuration support. This constructor is provided for programmatic agent
    /// composition (e.g. inside orchestration workflows) where a builder would be overly verbose.
    #[doc(hidden)]
    pub fn new(llm_provider: Arc<dyn Provider>, tools: Arc<ToolRegistry>) -> Self {
        Self::with_shared_state(llm_provider, tools, SharedState::new())
    }

    /// Create an agent with a pre-initialised shared state.
    pub fn with_shared_state(
        llm_provider: Arc<dyn Provider>,
        tools: Arc<ToolRegistry>,
        shared_state: SharedState,
    ) -> Self {
        let current_dir =
            env::current_dir().map_or_else(|_| "unknown".into(), |p| p.display().to_string());
        let system_prompt = message_builder::default_system_prompt(&current_dir).to_string();
        let token_counter = TokenCounterFactory::new().build();
        Self {
            llm_provider,
            tools,
            system_prompt,
            skill_index: Vec::new(),
            current_dir,
            shared_state,
            token_counter,
            user_context: None,
            metrics: None,
            session_id: None,
            max_iterations: 50,
            tool_choice: None,
            hooks: HookRegistry::new(),
            action_ids: Arc::new(std::sync::Mutex::new(HashMap::new())),
            permission_mode: PermissionMode::Default,
            allowed_tools: None,
            readonly_filter_hits: Arc::new(AtomicUsize::new(0)),
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
            tool_execution_driver: ToolExecutionDriver::default(),
            tool_intelligence: Arc::new(Mutex::new(ToolIntelligenceState::new())),
            #[cfg(feature = "tools-rag")]
            workspace_rag: None,
            #[cfg(feature = "tools-rag")]
            workspace_indexing_ready: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_top_k: 12,
            #[cfg(feature = "tools-rag")]
            workspace_rag_min_similarity: 0.25,
            #[cfg(feature = "tools-rag")]
            workspace_rag_line_level_bm25: false,
            #[cfg(feature = "tools-rag")]
            workspace_rag_search_options: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_context_cache: None,
            #[cfg(feature = "tools-rag")]
            workspace_rag_context_cache_ttl_secs: 60,
            max_tokens: None,
            temperature: None,
            include_context_in_system: true,
            cached_tools_description: Arc::new(arc_swap::ArcSwap::from_pointee(None)),
            cached_system_messages: Arc::new(parking_lot::RwLock::new(None)),
            #[cfg(feature = "tools-rag")]
            cached_system_messages_with_workspace: Arc::new(parking_lot::RwLock::new(None)),
            cached_context_window: None,
            #[cfg(feature = "compaction")]
            compaction_guard: None,
            #[cfg(feature = "context-engine")]
            context_engine: None,
            tool_ordering_embedding_provider: None,
            max_tool_output_in_history: 8_000,
            annotate_tool_output_truncation: false,
            #[cfg(feature = "agentfs")]
            agentfs_tool_logger: None,
            #[cfg(feature = "sandbox")]
            sandbox: None,
            approval_resolver: None,
            amendment_persister: None,
            steering_followup_provider: None,
            #[cfg(feature = "learning-loop")]
            learning_loop: None,
            reasoning_effort_slot: None,
            operation_manager: None,
            operation_translators: None,
            last_compaction_report: Arc::new(parking_lot::RwLock::new(None)),
        }
    }

    /// Add a tool to this agent's registry at runtime.
    ///
    /// Because [`ToolRegistry::register`] now takes `&self` (backed by an internal
    /// `RwLock`), this works even when multiple references to the registry exist —
    /// no registry clone is needed.
    pub fn add_tool(&mut self, tool: Arc<dyn crate::agent::tools::Tool>) {
        self.tools.register(tool);
        // Invalidate cached tools description so the new tool appears in system prompts.
        self.cached_tools_description.as_ref().store(Arc::new(None));
        *self.cached_system_messages.write() = None;
        #[cfg(feature = "tools-rag")]
        {
            *self.cached_system_messages_with_workspace.write() = None;
        }
    }

    /// Remove a tool from this agent's registry at runtime.
    pub fn remove_tool(&mut self, name: &str) {
        self.tools.unregister(name);
        self.cached_tools_description.as_ref().store(Arc::new(None));
        *self.cached_system_messages.write() = None;
        #[cfg(feature = "tools-rag")]
        {
            *self.cached_system_messages_with_workspace.write() = None;
        }
    }

    #[must_use]
    pub fn list_tools(&self) -> Vec<(String, String, serde_json::Value)> {
        self.tools.list()
    }

    #[must_use]
    pub const fn agent(&self) -> &Self {
        self
    }

    #[must_use]
    pub fn llm_provider(&self) -> &Arc<dyn Provider> {
        &self.llm_provider
    }

    pub(crate) fn tool_calling_config(&self) -> ToolCallingConfig {
        let tool_schemas = self.tools.get_tool_schemas();
        let use_native = self.llm_provider.supports_native_tools() && !tool_schemas.is_empty();

        // Prefer the provider-declared capability over a hardcoded provider-name
        // check so new text-based tool providers opt in without touching core.
        let supports_text_based_tools =
            self.llm_provider.supports_text_based_tools() && !tool_schemas.is_empty();
        let include_tool_descriptions = !use_native && supports_text_based_tools;

        ToolCallingConfig {
            tool_schemas,
            use_native,
            include_tool_descriptions,
            supports_text_based_tools,
        }
    }

    /// Produces a per-request copy with fresh conversation state but shared provider, tools, and config.
    ///
    /// Fields kept fresh per request (not inherited from the template):
    /// - `user_context`: must be re-populated by the caller for the specific user/session.
    /// - Cached tool descriptions / system messages: rebuilt lazily on first use.
    /// - `tool_intelligence`: deliberately given a fresh `Mutex<ToolIntelligenceState>` (not
    ///   `Arc::clone`d from the template). This state tracks per-conversation tool-call
    ///   deduplication, phased-planning progress, and execution history; sharing it across
    ///   concurrent per-request clones would leak one caller's tool-call history into another
    ///   caller's redundancy checks (cross-request contamination) and would serialize every
    ///   concurrent request's tool-calling loop on a single global lock.
    ///
    /// Fields shared via `Arc` across template and clones:
    /// - `token_counter`, `metrics`, `learning_loop`, registry handles, and caches.
    pub fn clone_for_request(&self) -> Self {
        let mut new_agent = Self::with_shared_state(
            Arc::clone(&self.llm_provider),
            Arc::clone(&self.tools),
            SharedState::new(),
        );
        new_agent.system_prompt = self.system_prompt.clone();
        new_agent.skill_index = self.skill_index.clone();
        new_agent.current_dir = self.current_dir.clone();
        new_agent.temperature = self.temperature;
        new_agent.max_tokens = self.max_tokens;
        new_agent.session_id = self.session_id.clone();
        new_agent.max_iterations = self.max_iterations;
        new_agent.tool_choice = self.tool_choice.clone();
        new_agent.hooks = self.hooks.clone();
        new_agent.permission_mode = self.permission_mode;
        new_agent.allowed_tools = self.allowed_tools.as_ref().map(Arc::clone);
        new_agent.readonly_filter_hits = Arc::clone(&self.readonly_filter_hits);
        new_agent.max_context_tokens = self.max_context_tokens;
        new_agent.tool_execution_driver = self.tool_execution_driver;
        new_agent.cached_context_window = self.cached_context_window;
        new_agent.include_context_in_system = self.include_context_in_system;
        new_agent.max_tool_output_in_history = self.max_tool_output_in_history;
        new_agent.annotate_tool_output_truncation = self.annotate_tool_output_truncation;
        new_agent.tool_ordering_embedding_provider = self.tool_ordering_embedding_provider.clone();
        // Share per-request-independent state across clones so per-request calls
        // benefit from the template's token counter configuration, execution
        // planner state, and aggregated metrics instead of starting fresh.
        new_agent.token_counter = Arc::clone(&self.token_counter);
        // Intentionally NOT shared: see the doc comment on this method. Each per-request
        // clone gets its own dedup/planning/history state so concurrent requests derived
        // from the same template agent cannot see or block on each other's tool-call state.
        new_agent.tool_intelligence = Arc::new(Mutex::new(ToolIntelligenceState::new()));
        new_agent.metrics = self.metrics.as_ref().map(Arc::clone);
        new_agent.cached_tools_description = Arc::clone(&self.cached_tools_description);
        new_agent.cached_system_messages = Arc::clone(&self.cached_system_messages);
        #[cfg(feature = "tools-rag")]
        {
            new_agent.cached_system_messages_with_workspace =
                Arc::clone(&self.cached_system_messages_with_workspace);
        }
        #[cfg(feature = "tools-rag")]
        {
            new_agent.workspace_rag = self.workspace_rag.clone();
            new_agent.workspace_indexing_ready = self.workspace_indexing_ready.clone();
            new_agent.workspace_rag_top_k = self.workspace_rag_top_k;
            new_agent.workspace_rag_min_similarity = self.workspace_rag_min_similarity;
            new_agent.workspace_rag_line_level_bm25 = self.workspace_rag_line_level_bm25;
            new_agent.workspace_rag_search_options = self.workspace_rag_search_options.clone();
            new_agent.workspace_rag_context_cache = self.workspace_rag_context_cache.clone();
            new_agent.workspace_rag_context_cache_ttl_secs =
                self.workspace_rag_context_cache_ttl_secs;
        }
        #[cfg(feature = "compaction")]
        {
            new_agent.compaction_guard = self.compaction_guard.clone();
        }
        #[cfg(feature = "context-engine")]
        {
            new_agent.context_engine = self.context_engine.clone();
        }
        #[cfg(feature = "agentfs")]
        {
            new_agent.agentfs_tool_logger = self.agentfs_tool_logger.clone();
        }
        #[cfg(feature = "sandbox")]
        {
            new_agent.sandbox = self.sandbox.clone();
        }
        new_agent.approval_resolver = self.approval_resolver.clone();
        new_agent.steering_followup_provider = self.steering_followup_provider.clone();
        new_agent.amendment_persister = self.amendment_persister.clone();
        #[cfg(feature = "learning-loop")]
        {
            new_agent.learning_loop = self.learning_loop.clone();
        }
        new_agent.reasoning_effort_slot = self.reasoning_effort_slot.as_ref().map(Arc::clone);
        new_agent
    }
}

#[cfg(test)]
mod tests {
    use super::ToolExecutionDriver;

    #[test]
    fn tool_execution_driver_parse_aliases_and_default_off() {
        assert_eq!(ToolExecutionDriver::default(), ToolExecutionDriver::Off);
        assert_eq!(
            ToolExecutionDriver::parse("dedup"),
            Some(ToolExecutionDriver::DedupOnly)
        );
        assert_eq!(
            ToolExecutionDriver::parse("dedup-only"),
            Some(ToolExecutionDriver::DedupOnly)
        );
        assert_eq!(
            ToolExecutionDriver::parse("planner-first"),
            Some(ToolExecutionDriver::PlannerFirstPhased)
        );
        assert_eq!(
            ToolExecutionDriver::parse("phased"),
            Some(ToolExecutionDriver::PlannerFirstPhased)
        );
        assert_eq!(
            ToolExecutionDriver::parse("off"),
            Some(ToolExecutionDriver::Off)
        );
        assert_eq!(ToolExecutionDriver::parse("nope"), None);
    }
}
