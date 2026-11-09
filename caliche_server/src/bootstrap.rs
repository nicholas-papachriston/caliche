//! Server bootstrap: caches, shared services, and [`AppState`] assembly.

use caliche::ProviderType;
#[cfg(feature = "tools-rag")]
use caliche::agent::tools::RagTool;
#[cfg(feature = "router")]
use caliche::llm::router::ProviderRouter;
#[cfg(feature = "router")]
use caliche::llm::router::{CostOptimizedRouter, LatencyRouter, StaticRouter, WeightedRouter};
use caliche::observability::{CacheMetrics, MetricsRecorder, ToolMetrics};
#[cfg(feature = "router")]
use caliche::security::{InMemorySpendTracker, SpendTracker};
use dashmap::DashMap;
use moka::future::Cache;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

use crate::handlers::endpoints::EndpointManager;
use crate::handlers::harnesses::HarnessManager;
use crate::state::{AppState, RouterOptions};
use crate::workflow_executor;

pub(crate) fn default_plugin_store_path() -> std::path::PathBuf {
    crate::paths::plugin_store_path()
}

pub(crate) fn default_mcp_config_path() -> std::path::PathBuf {
    crate::paths::mcp_config_path()
}

/// Drop document collection registry entries that have been idle longer than [`MAX_IDLE`].
pub(crate) fn sweep_stale_document_collections(
    collections: &Arc<DashMap<String, crate::state::CollectionMetadata>>,
) {
    const MAX_IDLE: chrono::Duration = chrono::Duration::days(7);
    let now = chrono::Utc::now();
    let stale: Vec<String> = collections
        .iter()
        .filter(|e| now.signed_duration_since(e.value().updated_at) > MAX_IDLE)
        .map(|e| e.key().clone())
        .collect();
    for name in stale {
        collections.remove(&name);
        tracing::debug!(
            target: "caliche_server::documents",
            collection = %name,
            "removed idle document collection registry entry"
        );
    }
}

#[cfg(feature = "tools-rag")]
async fn bootstrap_shared_rag_tool() -> Option<Arc<RagTool>> {
    if let Ok(api_key) = caliche_contracts::credential_env::var("OPENAI_API_KEY") {
        use caliche::llm::EmbeddingProvider;

        let model =
            caliche::llm::provider_utils::read_model_from_env("OPENAI_MODEL", "gpt-4o", None);
        let openai_provider = match caliche::llm::OpenAIProvider::try_new(api_key, model) {
            Ok(provider) => provider,
            Err(err) => {
                tracing::warn!(
                    "Failed to initialize OpenAI embeddings for RAG tool: {}",
                    err
                );
                return None;
            }
        };
        let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(openai_provider);
        tracing::info!("Using OpenAI embeddings for RAG tool");
        return Some(Arc::new(RagTool::new(embedding_provider, None::<String>)));
    }

    if caliche_contracts::credential_env::var("BEDROCK_EMBEDDING_MODEL_ID").is_ok() {
        return bootstrap_bedrock_rag_tool().await;
    }

    tracing::info!(
        "No embedding provider available (OPENAI_API_KEY or BEDROCK_EMBEDDING_MODEL_ID), RAG tool will not be available"
    );
    None
}

#[cfg(all(feature = "tools-rag", feature = "bedrock"))]
async fn bootstrap_bedrock_rag_tool() -> Option<Arc<RagTool>> {
    match caliche::llm::BedrockProvider::from_env_for_embeddings().await {
        Ok(bedrock_provider) => {
            use caliche::llm::EmbeddingProvider;

            let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::new(bedrock_provider);
            tracing::info!("Using Bedrock embeddings for RAG tool");
            Some(Arc::new(RagTool::new(embedding_provider, None::<String>)))
        }
        Err(e) => {
            tracing::warn!(
                "BEDROCK_EMBEDDING_MODEL_ID is set but failed to initialize Bedrock embeddings: {}",
                e
            );
            None
        }
    }
}

#[cfg(all(feature = "tools-rag", not(feature = "bedrock")))]
#[expect(
    clippy::unused_async,
    reason = "same signature as the Bedrock variant so the caller does not branch on features"
)]
async fn bootstrap_bedrock_rag_tool() -> Option<Arc<RagTool>> {
    tracing::warn!(
        "BEDROCK_EMBEDDING_MODEL_ID is set, but this server was built without the `bedrock` feature; RAG tool will not be available"
    );
    None
}

fn bootstrap_caches() -> (
    Arc<Cache<crate::state::ProviderCacheKey, Arc<dyn caliche::llm::Provider>>>,
    Arc<Cache<crate::state::ToolRegistryCacheKey, Arc<caliche::agent::tools::ToolRegistry>>>,
    Arc<Cache<String, Arc<RwLock<crate::state::WorkflowExecution>>>>,
    Arc<Cache<String, Arc<caliche_contracts::models::ProviderModelsResponseDto>>>,
    Arc<Cache<String, Arc<caliche_contracts::models::ModelsCatalogResponseDto>>>,
) {
    let provider_cache = Arc::new(
        Cache::builder()
            .max_capacity(100)
            .time_to_live(Duration::from_secs(1800))
            .build(),
    );
    let tool_registry_cache = Arc::new(
        Cache::builder()
            .max_capacity(100)
            .time_to_live(Duration::from_secs(1800))
            .build(),
    );
    let workflow_cache = Arc::new(
        Cache::builder()
            .max_capacity(100)
            .time_to_live(Duration::from_secs(3600))
            .build(),
    );
    let model_list_cache = Arc::new(
        Cache::builder()
            .max_capacity(64)
            .time_to_live(Duration::from_secs(300))
            .build(),
    );
    let all_models_cache = Arc::new(
        Cache::builder()
            .max_capacity(4)
            .time_to_live(Duration::from_secs(300))
            .build(),
    );
    (
        provider_cache,
        tool_registry_cache,
        workflow_cache,
        model_list_cache,
        all_models_cache,
    )
}

#[cfg(feature = "router")]
fn bootstrap_provider_router() -> (
    Option<Arc<dyn ProviderRouter>>,
    Option<Arc<dyn SpendTracker>>,
) {
    let strategy = caliche_contracts::credential_env::var("CALICHE_ROUTER_STRATEGY")
        .unwrap_or_else(|_| "static".to_string())
        .to_lowercase();

    let provider_router: Option<Arc<dyn ProviderRouter>> = match strategy.as_str() {
        "weighted" => {
            if let Ok(weights) = caliche_contracts::credential_env::var("CALICHE_ROUTER_WEIGHTS") {
                match WeightedRouter::from_env_weights(&weights) {
                    Ok(r) => {
                        tracing::info!("Provider router: weighted");
                        Some(Arc::new(r))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse CALICHE_ROUTER_WEIGHTS: {}", e);
                        Some(Arc::new(StaticRouter::new()))
                    }
                }
            } else {
                tracing::warn!(
                    "CALICHE_ROUTER_STRATEGY=weighted but CALICHE_ROUTER_WEIGHTS not set"
                );
                Some(Arc::new(StaticRouter::new()))
            }
        }
        "latency" => {
            let env_value =
                caliche_contracts::credential_env::var("CALICHE_ROUTER_LATENCY_CANDIDATES").ok();
            let alpha: f64 = caliche_contracts::credential_env::var("CALICHE_ROUTER_LATENCY_ALPHA")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.2);
            let router_result = match env_value.as_deref() {
                Some(v) if !v.trim().is_empty() => LatencyRouter::from_env(v, alpha),
                _ => {
                    tracing::warn!(
                        "CALICHE_ROUTER_STRATEGY=latency but CALICHE_ROUTER_LATENCY_CANDIDATES \
                         is unset; provide e.g. 'openai:gpt-4o-mini,anthropic:claude-3-haiku'"
                    );
                    Err(anyhow::anyhow!("CALICHE_ROUTER_LATENCY_CANDIDATES missing"))
                }
            };
            match router_result {
                Ok(r) => {
                    tracing::info!(alpha = alpha, "Provider router: latency (EWMA)");
                    Some(Arc::new(r) as Arc<dyn ProviderRouter>)
                }
                Err(err) => {
                    tracing::warn!(error = %err, "falling back to static router");
                    Some(Arc::new(StaticRouter::new()) as Arc<dyn ProviderRouter>)
                }
            }
        }
        "cost" => {
            if let Ok(json) = caliche_contracts::credential_env::var("CALICHE_ROUTER_COST_TABLE") {
                match CostOptimizedRouter::from_json(&json) {
                    Ok(r) => {
                        tracing::info!("Provider router: cost-optimized");
                        Some(Arc::new(r))
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse CALICHE_ROUTER_COST_TABLE: {}", e);
                        Some(Arc::new(StaticRouter::new()))
                    }
                }
            } else {
                tracing::warn!(
                    "CALICHE_ROUTER_STRATEGY=cost but CALICHE_ROUTER_COST_TABLE not set"
                );
                Some(Arc::new(StaticRouter::new()))
            }
        }
        _ => {
            tracing::debug!("Provider router: static (default)");
            Some(Arc::new(StaticRouter::new()))
        }
    };

    let spend_tracker = InMemorySpendTracker::from_env();
    let spend_tracker: Option<Arc<dyn SpendTracker>> =
        spend_tracker.map(|t| Arc::new(t) as Arc<dyn SpendTracker>);

    (provider_router, spend_tracker)
}

/// Assemble [`AppState`] and related bootstrap artifacts for router construction.
pub(crate) async fn bootstrap_app_state(
    default_provider: ProviderType,
    default_model: String,
    options: &Option<RouterOptions>,
) -> anyhow::Result<AppState> {
    let discovery_snapshot = caliche::discovery::auto::discover_all(
        &caliche::discovery::auto::DiscoveryOptions::default(),
    );
    discovery_snapshot.log_summary("api.bootstrap");

    #[cfg(feature = "tools-rag")]
    let shared_rag_tool = bootstrap_shared_rag_tool().await;

    let tool_metrics = Arc::new(ToolMetrics::new());
    let (provider_cache, tool_registry_cache, workflow_cache, model_list_cache, all_models_cache) =
        bootstrap_caches();
    let document_collections = Arc::new(DashMap::new());
    let cache_metrics = Arc::new(CacheMetrics::new());

    #[cfg(feature = "router")]
    let (provider_router, spend_tracker) = bootstrap_provider_router();
    #[cfg(not(feature = "router"))]
    let (): () = ();

    let workflow_executor = workflow_executor::ApiWorkflowNodeExecutor {
        default_provider,
        default_model: default_model.clone(),
        provider_cache: Arc::clone(&provider_cache),
        tool_registry_cache: Arc::clone(&tool_registry_cache),
        tool_metrics: Arc::clone(&tool_metrics),
        cache_metrics: Arc::clone(&cache_metrics),
        #[cfg(feature = "tools-rag")]
        shared_rag_tool: shared_rag_tool.as_ref().map(Arc::clone),
    };

    let shared_service = options.as_ref().and_then(|o| o.shared_service.clone());
    let context_window = options.as_ref().and_then(|o| o.context_window);
    let on_settings_applied = options.as_ref().and_then(|o| o.on_settings_applied.clone());

    let settings_broadcast = shared_service.as_ref().map(|_| {
        let (tx, _) = tokio::sync::broadcast::channel(16);
        tx
    });

    let plugin_registry =
        match caliche::plugins::persistence::PluginStore::from_path(default_plugin_store_path())
            .await
        {
            Ok(store) => {
                let registry = Arc::new(caliche::plugins::registry::PluginRegistry::new(Arc::new(
                    store,
                )));
                if let Err(err) = registry.load_all().await {
                    tracing::warn!("Failed to load plugin registry for API: {}", err);
                }
                Some(registry)
            }
            Err(err) => {
                tracing::warn!("Failed to initialize plugin store for API: {}", err);
                None
            }
        };
    let mcp_config_path = options
        .as_ref()
        .and_then(|o| o.mcp_config_path.clone())
        .unwrap_or_else(default_mcp_config_path);
    let mcp_servers = Arc::new(RwLock::new(
        crate::handlers::mcp_config::load_mcp_configs(&mcp_config_path).await,
    ));

    let endpoint_manager = options
        .as_ref()
        .and_then(|o| o.endpoint_manager.clone())
        .unwrap_or_else(EndpointManager::new);
    endpoint_manager.load_persisted().await;

    let metrics_recorder = Arc::new(
        MetricsRecorder::new()
            .map_err(|err| anyhow::anyhow!("failed to initialise API metrics recorder: {err}"))?,
    );

    Ok(AppState {
        default_provider,
        default_model,
        context_window,
        shared_service,
        settings_broadcast,
        on_settings_applied,
        #[cfg(feature = "tools-rag")]
        shared_rag_tool,
        #[cfg(feature = "router")]
        provider_router,
        #[cfg(feature = "router")]
        spend_tracker,
        provider_cache,
        tool_registry_cache,
        workflow_cache,
        workflow_engine: Arc::new(caliche::orchestration::WorkflowEngine::with_executor(
            Arc::new(workflow_executor),
        )),
        document_collections,
        tool_metrics,
        cache_metrics,
        model_list_cache,
        all_models_cache,
        plugin_registry,
        mcp_servers,
        mcp_config_path,
        policy_path: caliche::policy::default_policy_path(),
        endpoint_manager: Arc::clone(&endpoint_manager),
        harness_manager: HarnessManager::new(),
        metrics_recorder: Arc::clone(&metrics_recorder),
    })
}
