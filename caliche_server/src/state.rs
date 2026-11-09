//! Shared API server state types.

use caliche::AgentService;
use caliche::ProviderType;
use caliche::agent::session::ShareConfig;
#[cfg(feature = "tools-rag")]
use caliche::agent::tools::RagTool;
use caliche::agent::tools::ToolRegistry;
use caliche::harness::HarnessConfig;
use caliche::llm::Provider;
#[cfg(feature = "router")]
use caliche::llm::router::ProviderRouter;
use caliche::observability::{CacheMetrics, MetricsRecorder, ToolMetrics};
use caliche::orchestration::{WorkflowEngine, WorkflowState};
#[cfg(feature = "router")]
use caliche::security::SpendTracker;
use dashmap::DashMap;
use moka::future::Cache;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::handlers::endpoints::EndpointManager;
use crate::handlers::harnesses::HarnessManager;
use crate::rpc::MethodRegistrar;

/// Options passed when creating the router (e.g. from CLI so server uses same config).
#[derive(Clone, Default)]
pub struct RouterOptions {
    /// Harness config for session-based routes (provider, model, tools, workspace, etc.).
    /// When set, the API adapter uses this instead of `HarnessConfig::default()`.
    pub harness_config: Option<HarnessConfig>,
    /// Shared agent service when the CLI is hosting the API. Handlers derive per-request
    /// copies via `clone_for_request()` so API and CLI share the same agent configuration.
    pub shared_service: Option<Arc<tokio::sync::RwLock<AgentService>>>,
    /// Shared endpoint manager for router construction and shutdown coordination.
    pub endpoint_manager: Option<Arc<EndpointManager>>,
    /// Default context window override for services created by request handlers.
    pub context_window: Option<usize>,
    /// Additional JSON-RPC method registrars supplied by embedding crates such as the CLI.
    pub rpc_method_registrars: Vec<MethodRegistrar>,
    /// Session share policy, visibility defaults, and redaction patterns.
    pub share_config: Option<ShareConfig>,
    /// Optional hook invoked after settings are applied to the shared service.
    pub on_settings_applied: Option<Arc<dyn Fn(&caliche::AgentSettings) + Send + Sync>>,
    /// Override the MCP config path (and the sibling `settings.json`).
    ///
    /// `None` uses `dirs::config_dir()/caliche/mcp.json`.
    pub mcp_config_path: Option<std::path::PathBuf>,
}

/// Cache key for Provider instances (by provider type and model)
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ProviderCacheKey {
    pub provider: ProviderType,
    pub model: String,
}

/// Cache key for ToolRegistry instances (by provider type and tool names)
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ToolRegistryCacheKey {
    pub provider: ProviderType,
    pub tools: Option<Vec<String>>,
}

#[derive(Clone)]
pub struct AppState {
    /// Default provider type (used when request doesn't specify one)
    pub default_provider: ProviderType,
    /// Default model name (used when request doesn't specify one)
    pub default_model: String,
    /// Default context window override for services created by request handlers.
    pub context_window: Option<usize>,
    /// Shared agent service when the CLI is hosting the API. Handlers derive per-request
    /// copies via `clone_for_request()` so API and CLI share the same agent configuration.
    pub shared_service: Option<Arc<tokio::sync::RwLock<AgentService>>>,
    /// When set, PUT /api/settings and WS UpdateSettings broadcast here so all WS clients get SettingsChanged.
    pub settings_broadcast: Option<tokio::sync::broadcast::Sender<caliche::AgentSettings>>,
    /// Optional hook invoked after settings are applied to the shared service.
    pub on_settings_applied: Option<Arc<dyn Fn(&caliche::AgentSettings) + Send + Sync>>,
    #[cfg(feature = "tools-rag")]
    pub shared_rag_tool: Option<Arc<RagTool>>,
    /// Optional provider router for dynamic (provider, model) selection
    #[cfg(feature = "router")]
    pub provider_router: Option<Arc<dyn ProviderRouter>>,
    /// Optional spend tracker for per-tenant budget enforcement
    #[cfg(feature = "router")]
    pub spend_tracker: Option<Arc<dyn SpendTracker>>,
    /// Cache of Provider instances by (provider, model) with TTL and LRU eviction
    /// Providers are expensive to create (network clients, auth setup)
    pub provider_cache: Arc<Cache<ProviderCacheKey, Arc<dyn Provider>>>,
    /// Cache of ToolRegistry instances by (provider, tools) with TTL and LRU eviction
    /// Tool registries are expensive to build (tool initialization)
    pub tool_registry_cache: Arc<Cache<ToolRegistryCacheKey, Arc<ToolRegistry>>>,
    /// Workflow execution cache with TTL
    pub workflow_cache: Arc<Cache<String, Arc<RwLock<WorkflowExecution>>>>,
    /// Workflow engine for managing workflows and interrupts
    pub workflow_engine: Arc<WorkflowEngine>,
    /// Document collection registry
    /// Note: Using DashMap for iteration support; a background task prunes idle entries (see `create_router`).
    pub document_collections: Arc<DashMap<String, CollectionMetadata>>,
    /// Tool execution metrics
    pub tool_metrics: Arc<ToolMetrics>,
    /// Cache performance metrics
    pub cache_metrics: Arc<CacheMetrics>,
    /// Cache of provider model catalog entries with TTL.
    pub model_list_cache:
        Arc<Cache<String, Arc<caliche_contracts::models::ProviderModelsResponseDto>>>,
    /// Cache of the aggregated model catalog with TTL.
    pub all_models_cache:
        Arc<Cache<String, Arc<caliche_contracts::models::ModelsCatalogResponseDto>>>,
    pub plugin_registry: Option<Arc<caliche::plugins::registry::PluginRegistry>>,
    pub mcp_servers: Arc<RwLock<Vec<crate::handlers::mcp_config::McpServerConfigPayload>>>,
    pub mcp_config_path: std::path::PathBuf,
    /// Deployment `policy.toml`. `CALICHE_POLICY_PATH` overrides the default.
    pub policy_path: std::path::PathBuf,
    /// Managed child-process endpoint registry
    pub endpoint_manager: Arc<EndpointManager>,
    /// Registered external CLI harnesses (Claude Code, Codex, custom, …)
    pub harness_manager: Arc<HarnessManager>,
    /// Dedicated Prometheus metrics registry for HTTP-layer counters/histograms.
    /// Kept alongside the global default registry so handlers or middleware can
    /// register per-process metrics without polluting global state across tests.
    pub metrics_recorder: Arc<MetricsRecorder>,
}

/// Workflow execution state for tracking
#[derive(Debug, Clone)]
pub struct WorkflowExecution {
    pub workflow_id: String,
    pub execution_id: String,
    pub state: WorkflowState,
    pub current_node: Option<String>,
    pub completed_nodes: Vec<String>,
    pub results: Vec<serde_json::Value>,
    pub error: Option<String>,
}

/// Document collection metadata
#[derive(Debug, Clone)]
pub struct CollectionMetadata {
    pub name: String,
    pub document_count: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last time this entry was created or updated (indexing activity).
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// State for agent routes (WS, execute, stream). Holds adapter and optional shared service
/// so WS can apply settings and broadcast when CLI is hosting.
#[derive(Clone)]
pub struct AgentRouterState {
    pub adapter: Arc<crate::adapter::ApiAdapter>,
    pub shared_service: Option<Arc<tokio::sync::RwLock<AgentService>>>,
    pub settings_broadcast: Option<tokio::sync::broadcast::Sender<caliche::AgentSettings>>,
}
