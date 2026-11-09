//! HTTP router construction and middleware wiring.

use crate::adapter::ApiAdapter;
use crate::bootstrap::{bootstrap_app_state, sweep_stale_document_collections};
use crate::handlers;
use crate::handlers::endpoints::EndpointManager;
use crate::middleware;
use crate::security::SecurityPolicy;
use crate::state::{AgentRouterState, RouterOptions};
use axum::Router;
use axum::response::IntoResponse;
use caliche::ProviderType;
use std::sync::Arc;
use std::time::Duration;

fn json_rpc_only_mode() -> bool {
    std::env::var("CALICHE_JSON_RPC_ONLY")
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

/// Create the API router
///
/// Takes default provider and model for fallback when requests don't specify them.
/// When `options.harness_config` is set (e.g. from CLI), session-based routes use that config.
/// All handlers use `from_components()` to create fresh AgentService instances per request.
pub async fn create_router(
    default_provider: ProviderType,
    default_model: String,
    options: Option<RouterOptions>,
) -> anyhow::Result<Router> {
    let security = Arc::new(SecurityPolicy::from_env()?);
    let state = bootstrap_app_state(default_provider, default_model.clone(), &options).await?;

    let mut router = Router::new()
        // Basic chat endpoints
        .route(
            "/api/chat",
            axum::routing::post(handlers::chat::chat_handler),
        )
        .route(
            "/api/chat/multipart",
            axum::routing::post(handlers::chat::chat_multipart_handler),
        )
        .route(
            "/api/chat/stream",
            axum::routing::post(handlers::chat_stream::chat_stream_handler),
        )
        .route(
            "/api/chat/stream/multipart",
            axum::routing::post(handlers::chat_stream::chat_stream_multipart_handler),
        )
        .route(
            "/api/chat/estimate",
            axum::routing::post(handlers::estimate::estimate_chat_handler),
        )
        .route(
            "/api/settings",
            axum::routing::get(handlers::settings::get_settings_handler)
                .put(handlers::settings::put_settings_handler),
        )
        .route(
            "/api/policy/effective",
            axum::routing::get(handlers::policy::get_effective_policy),
        )
        .route(
            "/api/policy",
            axum::routing::put(handlers::policy::put_policy),
        )
        // Tools and models
        .route(
            "/api/tools",
            axum::routing::get(handlers::tools::tools_handler),
        )
        .route(
            "/api/models",
            axum::routing::get(handlers::models::all_models_handler),
        )
        .route(
            "/api/models/by-provider",
            axum::routing::get(handlers::models::models_handler),
        )
        .route(
            "/api/storage/options",
            axum::routing::get(handlers::storage::storage_options_handler),
        )
        // Metrics endpoints
        .route(
            "/api/metrics/tools",
            axum::routing::get(handlers::metrics::get_tools_metrics_handler),
        )
        .route(
            "/api/metrics/tools/{name}",
            axum::routing::get(handlers::metrics::get_tool_metrics_handler),
        )
        .route(
            "/api/metrics/cache",
            axum::routing::get(handlers::metrics::get_cache_metrics_handler),
        )
        // Health check
        .route(
            "/health",
            axum::routing::get(handlers::health::health_handler),
        )
        // File browser endpoints (must be enabled via CALICHE_FILE_BROWSING=true)
        .route(
            "/api/files/enabled",
            axum::routing::get(handlers::files::files_enabled_handler),
        )
        .route(
            "/api/files/list",
            axum::routing::get(handlers::files::list_files_handler),
        )
        .route(
            "/api/files/read",
            axum::routing::get(handlers::files::read_file_handler),
        )
        // Prometheus metrics endpoint
        .route(
            "/metrics",
            axum::routing::get(handlers::prometheus_metrics::prometheus_metrics_handler),
        );

    // Reasoning pattern endpoints
    #[cfg(feature = "reasoning-patterns")]
    {
        router = router
            .route(
                "/api/reasoning/react",
                axum::routing::post(handlers::reasoning::react_handler),
            )
            .route(
                "/api/reasoning/plan-execute",
                axum::routing::post(handlers::reasoning::plan_execute_handler),
            )
            .route(
                "/api/reasoning/self-ask",
                axum::routing::post(handlers::reasoning::self_ask_handler),
            )
            .route(
                "/api/reasoning/reflexion",
                axum::routing::post(handlers::reasoning::reflexion_handler),
            )
            .route(
                "/api/reasoning/tree-of-thoughts",
                axum::routing::post(handlers::reasoning::tree_of_thoughts_handler),
            )
            .route(
                "/api/reasoning/patterns",
                axum::routing::get(handlers::reasoning::list_patterns_handler),
            );
    }

    // Chain endpoints
    #[cfg(feature = "chains")]
    {
        router = router
            .route(
                "/api/chains/llm",
                axum::routing::post(handlers::chains::llm_chain_handler),
            )
            .route(
                "/api/chains/sequential",
                axum::routing::post(handlers::chains::sequential_chain_handler),
            )
            .route(
                "/api/chains/map-reduce",
                axum::routing::post(handlers::chains::map_reduce_handler),
            );
    }

    // Retrieval chain endpoints
    #[cfg(feature = "retrieval-chains")]
    {
        router = router
            .route(
                "/api/chains/retrieval",
                axum::routing::post(handlers::chains::retrieval_chain_handler),
            )
            .route(
                "/api/chains/hybrid-search",
                axum::routing::post(handlers::chains::hybrid_search_handler),
            )
            .route(
                "/api/chains/document-map-reduce",
                axum::routing::post(handlers::chains::document_map_reduce_handler),
            );
    }

    // Document processing endpoints
    #[cfg(any(feature = "document-loaders", feature = "tools-rag"))]
    {
        router = router.route(
            "/api/documents/split",
            axum::routing::post(handlers::documents::split_documents_handler),
        );
    }

    #[cfg(feature = "document-loaders")]
    {
        router = router.route(
            "/api/documents/load",
            axum::routing::post(handlers::documents::load_documents_handler),
        );
    }

    #[cfg(feature = "tools-rag")]
    {
        router = router
            .route(
                "/api/documents/index",
                axum::routing::post(handlers::documents::index_documents_handler),
            )
            .route(
                "/api/rag/index-batch",
                axum::routing::post(handlers::documents::batch_index_documents_handler),
            )
            .route(
                "/api/documents/collections",
                axum::routing::get(handlers::documents::list_collections_handler),
            )
            .route(
                "/api/documents/collections/{name}",
                axum::routing::delete(handlers::documents::delete_collection_handler),
            )
            .route(
                "/api/documents/search",
                axum::routing::post(handlers::documents::search_documents_handler),
            );
    }

    // Orchestration endpoints
    router = router
        .route(
            "/api/orchestration/workflow",
            axum::routing::post(handlers::orchestration::execute_workflow_handler),
        )
        .route(
            "/api/orchestration/workflow/{id}",
            axum::routing::get(handlers::orchestration::get_workflow_status_handler),
        )
        .route(
            "/api/orchestration/workflow/{id}/pause",
            axum::routing::post(handlers::orchestration::pause_workflow_handler),
        )
        .route(
            "/api/orchestration/workflow/{id}/resume",
            axum::routing::post(handlers::orchestration::resume_workflow_handler),
        )
        .route(
            "/api/orchestration/interrupts",
            axum::routing::get(handlers::orchestration::list_interrupts_handler),
        )
        .route(
            "/api/orchestration/interrupts/{id}",
            axum::routing::get(handlers::orchestration::get_interrupt_handler),
        )
        .route(
            "/api/orchestration/interrupts/{id}/resume",
            axum::routing::post(handlers::orchestration::resume_interrupt_handler),
        )
        .route(
            "/api/orchestration/multi-agent",
            axum::routing::post(handlers::orchestration::multi_agent_handler),
        )
        .route(
            "/api/orchestration/coordinator",
            axum::routing::post(handlers::orchestration::coordinator_handler),
        )
        .route(
            "/api/orchestration/harnesses",
            axum::routing::get(handlers::harnesses::list_harnesses)
                .post(handlers::harnesses::create_harness),
        )
        .route(
            "/api/orchestration/harnesses/{id}",
            axum::routing::delete(handlers::harnesses::delete_harness),
        )
        .route(
            "/api/orchestration/harnesses/{id}/dispatch",
            axum::routing::post(handlers::harnesses::dispatch_harness),
        )
        // Workflow builder endpoints
        .route(
            "/api/workflows",
            axum::routing::post(handlers::orchestration::save_workflow_handler),
        )
        .route(
            "/api/workflows/{id}",
            axum::routing::get(handlers::orchestration::load_workflow_handler),
        )
        .route(
            "/api/workflows/validate",
            axum::routing::post(handlers::orchestration::validate_workflow_handler),
        )
        .route(
            "/api/workflows/execute",
            axum::routing::post(handlers::orchestration::execute_workflow_builder_handler),
        );

    router = router
        .route(
            "/api/plugins",
            axum::routing::get(handlers::plugins::list_plugins_handler)
                .post(handlers::plugins::save_plugin_handler),
        )
        .route(
            "/api/plugins/catalog",
            axum::routing::get(handlers::plugins::plugin_catalog_handler),
        )
        .route(
            "/api/plugins/validate",
            axum::routing::post(handlers::plugins::validate_plugin_handler),
        )
        .route(
            "/api/plugins/{name}",
            axum::routing::get(handlers::plugins::get_plugin_handler)
                .delete(handlers::plugins::delete_plugin_handler),
        )
        .route(
            "/api/plugins/{name}/export",
            axum::routing::get(handlers::plugins::export_plugin_handler),
        )
        .route(
            "/api/mcp/config",
            axum::routing::get(handlers::mcp_config::get_mcp_config_handler)
                .put(handlers::mcp_config::put_mcp_config_handler),
        )
        .route(
            "/api/mcp/config/test",
            axum::routing::post(handlers::mcp_config::test_mcp_config_handler),
        );

    // Endpoint management routes
    router = router
        .route(
            "/api/endpoints",
            axum::routing::get(handlers::endpoints::list_endpoints)
                .post(handlers::endpoints::create_endpoint),
        )
        .route(
            "/api/endpoints/{id}",
            axum::routing::get(handlers::endpoints::get_endpoint)
                .delete(handlers::endpoints::delete_endpoint),
        )
        .route(
            "/api/endpoints/{id}/start",
            axum::routing::post(handlers::endpoints::start_endpoint),
        )
        .route(
            "/api/endpoints/{id}/stop",
            axum::routing::post(handlers::endpoints::stop_endpoint),
        );

    // MCP endpoints
    #[cfg(feature = "mcp")]
    {
        router = router
            .route(
                "/api/mcp/capabilities",
                axum::routing::get(handlers::mcp::capabilities_handler),
            )
            .route(
                "/api/mcp/tools",
                axum::routing::get(handlers::mcp::tools_handler),
            )
            .route(
                "/api/mcp/prompts",
                axum::routing::get(handlers::mcp::prompts_handler),
            )
            .route(
                "/api/mcp/resources",
                axum::routing::get(handlers::mcp::resources_handler),
            );
    }

    // Session-based agent endpoints (harness)
    // These use a separate nested router with their own state (ApiAdapter)
    {
        let harness_config = options
            .as_ref()
            .and_then(|o| o.harness_config.clone())
            .unwrap_or_default();
        let mut adapter = ApiAdapter::new_with_env(harness_config, state.shared_service.clone())
            .await
            .with_security(Arc::clone(&security));
        if let Some(share_config) = options.as_ref().and_then(|o| o.share_config.clone()) {
            adapter = adapter.with_share_config(share_config);
        }
        let sessions_state = std::sync::Arc::new(adapter);

        {
            let collections = state.document_collections.clone();
            let adapter = sessions_state.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(300));
                interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
                loop {
                    interval.tick().await;
                    sweep_stale_document_collections(&collections);
                    adapter.sweep_orphan_gateway_bindings().await;
                }
            });
        }

        let agent_router_state = AgentRouterState {
            adapter: sessions_state.clone(),
            shared_service: state.shared_service.clone(),
            settings_broadcast: state.settings_broadcast.clone(),
        };

        let rpc_state = crate::rpc::RpcState::new(
            state.clone(),
            agent_router_state.clone(),
            sessions_state.clone(),
            options
                .as_ref()
                .map(|o| o.rpc_method_registrars.as_slice())
                .unwrap_or(&[]),
        );

        // Session management routes
        let sessions_router = Router::new()
            .route(
                "/",
                axum::routing::get(handlers::sessions::list_sessions_handler)
                    .post(handlers::sessions::create_session_handler),
            )
            .route(
                "/{id}",
                axum::routing::get(handlers::sessions::get_session_handler)
                    .delete(handlers::sessions::delete_session_handler),
            )
            .route(
                "/{id}/cancel",
                axum::routing::post(handlers::sessions::cancel_session_handler),
            )
            .route(
                "/{id}/stop",
                axum::routing::post(handlers::sessions::cancel_session_handler),
            )
            .route(
                "/{id}/turn",
                axum::routing::post(handlers::sessions::execute_turn_handler),
            )
            .route(
                "/{id}/task",
                axum::routing::post(handlers::sessions::execute_task_handler),
            )
            .route(
                "/{id}/events",
                axum::routing::get(handlers::sessions::session_events_handler),
            )
            .route(
                "/{id}/rollback",
                axum::routing::post(handlers::sessions::rollback_handler),
            )
            .route(
                "/{id}/checkpoint",
                axum::routing::post(handlers::sessions::checkpoint_handler),
            )
            .route(
                "/{id}/undo",
                axum::routing::post(handlers::sessions::undo_handler),
            )
            .route(
                "/{id}/diff",
                axum::routing::get(handlers::sessions::diff_handler),
            )
            .route(
                "/{id}/revert",
                axum::routing::post(handlers::sessions::revert_handler),
            )
            .route(
                "/{id}/unrevert",
                axum::routing::post(handlers::sessions::unrevert_handler),
            )
            .route(
                "/{id}/clear",
                axum::routing::post(handlers::sessions::clear_history_handler),
            )
            .route(
                "/{id}/history",
                axum::routing::post(handlers::sessions::restore_history_handler),
            )
            .route(
                "/{id}/share",
                axum::routing::post(handlers::sessions::share_handler)
                    .delete(handlers::sessions::unshare_handler),
            )
            // Context management routes
            .route(
                "/{id}/context",
                axum::routing::get(handlers::context::get_context_handler),
            )
            .route(
                "/{id}/context/track",
                axum::routing::post(handlers::context::track_file_handler),
            )
            .route(
                "/{id}/context/refresh-git",
                axum::routing::post(handlers::context::refresh_git_handler),
            )
            .route(
                "/{id}/context/detect-project",
                axum::routing::post(handlers::context::detect_project_handler),
            )
            .route(
                "/{id}/context/relevant",
                axum::routing::get(handlers::context::get_relevant_context_handler),
            )
            // Mode management routes
            .route(
                "/{id}/mode",
                axum::routing::get(handlers::sessions::get_session_mode_handler)
                    .post(handlers::sessions::set_session_mode_handler),
            )
            // Plan document routes
            .route(
                "/{id}/plan",
                axum::routing::get(handlers::sessions::get_session_plan_handler)
                    .delete(handlers::sessions::clear_session_plan_handler),
            )
            // Memory endpoints
            .route(
                "/{id}/memory/stats",
                axum::routing::get(handlers::memory::memory_stats_handler),
            )
            .route(
                "/{id}/memory/inspect",
                axum::routing::get(handlers::memory::memory_inspect_handler),
            )
            .route(
                "/{id}/memory/search",
                axum::routing::get(handlers::memory::memory_search_handler),
            )
            .route(
                "/{id}/memory/clear",
                axum::routing::post(handlers::memory::memory_clear_handler),
            )
            .route(
                "/{id}/memory/export",
                axum::routing::get(handlers::memory::memory_export_handler),
            )
            .route(
                "/{id}/memory/import",
                axum::routing::post(handlers::memory::memory_import_handler),
            )
            .route_layer(axum::middleware::from_fn_with_state(
                sessions_state.clone(),
                crate::middleware::session_auth::session_route_auth_middleware,
            ))
            .with_state(sessions_state.clone());

        // Enhanced agent execution routes (harness-based)
        let agent_router = Router::new()
            .route(
                "/execute",
                axum::routing::post(handlers::agent_stream::execute_task_handler),
            )
            .route(
                "/stream",
                axum::routing::post(handlers::agent_stream::stream_task_handler),
            )
            .route(
                "/chat/stream",
                axum::routing::post(handlers::agent_stream::chat_stream_with_events_handler),
            )
            .route(
                "/enhanced",
                axum::routing::post(handlers::agent_stream::enhanced_agent_handler),
            )
            .with_state(agent_router_state);

        // Gateway control and diagnostics routes
        let gateway_router = Router::new()
            .route(
                "/adapters",
                axum::routing::get(handlers::gateway::list_adapters_handler),
            )
            .route(
                "/adapters/{id}/enable",
                axum::routing::post(handlers::gateway::enable_adapter_handler),
            )
            .route(
                "/adapters/{id}/disable",
                axum::routing::post(handlers::gateway::disable_adapter_handler),
            )
            .route(
                "/bindings/{session_id}",
                axum::routing::get(handlers::gateway::list_bindings_handler),
            )
            .route(
                "/bindings/override",
                axum::routing::post(handlers::gateway::override_binding_handler),
            )
            .route(
                "/ingress/simulate",
                axum::routing::post(handlers::gateway::simulate_ingress_handler),
            )
            .route(
                "/ingress/simulate-envelope",
                axum::routing::post(handlers::gateway::simulate_envelope_ingress_handler),
            )
            .route(
                "/ingress/telegram",
                axum::routing::post(handlers::gateway::telegram_webhook_ingress_handler),
            )
            .route(
                "/ingress/discord",
                axum::routing::post(handlers::gateway::discord_webhook_ingress_handler),
            )
            .route(
                "/ingress/twilio",
                axum::routing::post(handlers::gateway::twilio_webhook_ingress_handler),
            )
            .route(
                "/ingress/slack",
                axum::routing::post(handlers::gateway::slack_webhook_ingress_handler),
            )
            .route(
                "/ingress/teams",
                axum::routing::post(handlers::gateway::teams_webhook_ingress_handler),
            )
            .route(
                "/delivery/diagnostics",
                axum::routing::get(handlers::gateway::delivery_diagnostics_handler),
            )
            .route(
                "/cluster/register",
                axum::routing::post(handlers::gateway::register_cluster_node_handler),
            )
            .route(
                "/cluster/nodes",
                axum::routing::get(handlers::gateway::list_cluster_nodes_handler),
            )
            .route(
                "/cluster/routes",
                axum::routing::get(handlers::gateway::list_cluster_routes_handler),
            )
            .route(
                "/cluster/routes/pin",
                axum::routing::post(handlers::gateway::pin_cluster_route_handler),
            )
            .route(
                "/cluster/routes/unpin",
                axum::routing::post(handlers::gateway::unpin_cluster_route_handler),
            )
            .with_state(sessions_state.clone());

        let shares_router = Router::new()
            .route(
                "/shares/{slug}",
                axum::routing::get(handlers::sessions::get_share_handler)
                    .delete(handlers::sessions::delete_share_handler),
            )
            .with_state(sessions_state.clone());

        let public_share_router = Router::new()
            .route(
                "/s/{slug}",
                axum::routing::get(handlers::sessions::public_share_handler),
            )
            .with_state(sessions_state);

        router = router
            .nest("/api/agent/sessions", sessions_router)
            .nest("/api/agent", shares_router)
            .nest("/api/agent", agent_router)
            .nest("/api/gateway", gateway_router)
            .merge(public_share_router)
            .merge(
                Router::new()
                    .route(
                        "/ws",
                        axum::routing::get(crate::rpc::transport::ws_rpc_upgrade),
                    )
                    .route(
                        "/rpc",
                        axum::routing::post(crate::rpc::transport::http_rpc_handler),
                    )
                    .with_state(rpc_state),
            );
    }

    // Template endpoints
    #[cfg(feature = "templates")]
    {
        router = router
            .route(
                "/api/templates",
                axum::routing::get(handlers::templates::list_templates_handler)
                    .post(handlers::templates::upload_template_handler),
            )
            .route(
                "/api/templates/categories",
                axum::routing::get(handlers::templates::get_categories_handler),
            )
            .route(
                "/api/templates/search",
                axum::routing::get(handlers::templates::search_templates_handler),
            )
            .route(
                "/api/templates/fork",
                axum::routing::post(handlers::templates::fork_template_handler),
            )
            .route(
                "/api/templates/import",
                axum::routing::post(handlers::templates::import_template_handler),
            )
            .route(
                "/api/templates/{id}",
                axum::routing::get(handlers::templates::get_template_handler)
                    .delete(handlers::templates::delete_template_handler),
            )
            .route(
                "/api/templates/{id}/export",
                axum::routing::get(handlers::templates::export_template_handler),
            )
            .route(
                "/api/templates/validate",
                axum::routing::post(handlers::templates::validate_template_handler),
            )
            .route(
                "/api/templates/instantiate",
                axum::routing::post(handlers::templates::instantiate_template_handler),
            )
            .route(
                "/api/templates/instantiate-with-variables",
                axum::routing::post(handlers::templates::instantiate_with_variables_handler),
            );
    }

    // Enforce JSON-RPC as the primary API contract when explicitly opted in.
    // Keep only health/metrics and streaming chat endpoints as explicit HTTP exceptions.
    let json_rpc_only = json_rpc_only_mode();
    tracing::info!(
        target: "caliche_server::router",
        json_rpc_only,
        "router built (CALICHE_JSON_RPC_ONLY={})",
        if json_rpc_only { "on" } else { "off" }
    );
    let router = if json_rpc_only {
        router.layer(axum::middleware::from_fn(
            |request: axum::extract::Request, next: axum::middleware::Next| async move {
                let path = request.uri().path();
                let is_allowed_http = matches!(
                    path,
                    "/health"
                        | "/metrics"
                        | "/rpc"
                        | "/ws"
                        | "/api/chat/stream"
                        | "/api/chat/stream/multipart"
                );
                let is_non_api = !path.starts_with("/api/");
                if is_allowed_http || is_non_api {
                    next.run(request).await
                } else {
                    (
                        axum::http::StatusCode::GONE,
                        "This API is JSON-RPC only. Use POST /rpc or GET /ws.",
                    )
                        .into_response()
                }
            },
        ))
    } else {
        router
    };

    // Add request logging middleware
    let router = router.layer(axum::middleware::from_fn(
        middleware::request_logging::request_logging_middleware,
    ));

    let router = router.layer(axum::middleware::from_fn(
        middleware::rate_limit::rate_limit_middleware,
    ));

    // Require authentication for sensitive path prefixes. Added BEFORE the
    // auth-state injection layer so at runtime injection runs first.
    let router = router.layer(axum::middleware::from_fn(
        middleware::jwt_auth::require_jwt_middleware,
    ));

    // Inject startup SecurityPolicy and optional JWT state for middleware/extractors.
    let security_for_layer = Arc::clone(&security);
    let router = router.layer(axum::middleware::from_fn(
        move |mut request: axum::extract::Request, next: axum::middleware::Next| {
            let security = security_for_layer.clone();
            async move {
                request.extensions_mut().insert(Arc::clone(&security));
                if let Some(auth_state) = security.jwt_state.clone() {
                    request.extensions_mut().insert(auth_state);
                }
                next.run(request).await
            }
        },
    ));

    #[cfg(feature = "embedded-ui")]
    let router = router.fallback(crate::embedded_ui::serve_embedded_ui);

    Ok(router.with_state(state))
}

/// Like [`create_router`] but also returns the [`EndpointManager`] so callers
/// can call [`EndpointManager::stop_all`] on graceful server shutdown.
pub async fn create_router_with_shutdown(
    default_provider: caliche::ProviderType,
    default_model: String,
    options: Option<RouterOptions>,
) -> anyhow::Result<(Router, Arc<EndpointManager>)> {
    let endpoint_manager = EndpointManager::new();
    let mut options = options.unwrap_or_default();
    options.endpoint_manager = Some(Arc::clone(&endpoint_manager));
    let router = create_router(default_provider, default_model, Some(options)).await?;
    Ok((router, endpoint_manager))
}
