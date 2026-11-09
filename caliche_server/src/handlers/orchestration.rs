//! Orchestration API handlers for workflows and multi-agent patterns.

use crate::AppState;
use crate::WorkflowExecution;
use crate::models::*;
use async_trait::async_trait;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::agent::service::AgentService;
use caliche::agent::subagent::SubagentModelSelection;
use caliche::harness::HarnessSessionSpawner;
use caliche::orchestration::{BuiltWorkflow, *};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::Instrument;
use uuid::Uuid;

#[derive(Debug, Clone)]
struct ApiWorkflowDefinition {
    id: String,
    name: String,
    description: String,
    nodes: Vec<WorkflowNode>,
    edges: Vec<WorkflowEdge>,
    entry_point_id: String,
    entry_point_name: String,
}

#[async_trait]
impl Workflow for ApiWorkflowDefinition {
    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn nodes(&self) -> &[WorkflowNode] {
        &self.nodes
    }

    fn edges(&self) -> &[WorkflowEdge] {
        &self.edges
    }

    fn entry_point(&self) -> Option<&str> {
        Some(self.entry_point_name.as_str())
    }

    async fn execute(
        &self,
        engine: &WorkflowEngine,
        context: &mut ExecutionContext,
    ) -> anyhow::Result<serde_json::Value> {
        // Always execute as a state machine for API workflows.
        // This matches the edge-driven execution model and supports control-flow nodes (Parallel/Loop).
        let results = engine
            .execute_state_machine(&self.nodes, &self.edges, &self.entry_point_id, context)
            .await?;

        Ok(results
            .last()
            .and_then(|r| r.output.clone())
            .unwrap_or(serde_json::json!(null)))
    }
}

fn resolve_entry_point(
    request: &WorkflowRequest,
) -> Result<(String, String), (StatusCode, String)> {
    let Some(first) = request.nodes.first() else {
        return Err((
            StatusCode::BAD_REQUEST,
            "Workflow must contain at least one node".to_string(),
        ));
    };

    let requested = request
        .metadata
        .as_ref()
        .and_then(|m| m.as_object())
        .and_then(|m| m.get("entry_point"))
        .and_then(|v| v.as_str());

    if let Some(entry_point) = requested {
        if let Some(node) = request
            .nodes
            .iter()
            .find(|n| n.id == entry_point || n.name == entry_point)
        {
            return Ok((node.id.clone(), node.name.clone()));
        }
        let available_nodes: Vec<String> = request
            .nodes
            .iter()
            .map(|n| format!("{} ({})", n.name, n.id))
            .collect();
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "Invalid metadata.entry_point '{}'. Available nodes: {}",
                entry_point,
                available_nodes.join(", ")
            ),
        ));
    }

    Ok((first.id.clone(), first.name.clone()))
}

/// POST /api/orchestration/workflow - Execute a workflow
pub async fn execute_workflow_handler(
    State(state): State<AppState>,
    Json(request): Json<WorkflowRequest>,
) -> Result<Json<WorkflowResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span =
        tracing::span!(tracing::Level::INFO, "orchestration_workflow", request_id = %request_id);

    async move {
        tracing::info!("Received workflow execution request");

        let workflow_id = Uuid::now_v7().to_string();

        let (entry_point_id, entry_point_name) = resolve_entry_point(&request)?;

        let workflow = ApiWorkflowDefinition {
            id: workflow_id.clone(),
            name: request
                .metadata
                .as_ref()
                .and_then(|m| m.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("api_workflow")
                .to_string(),
            description: request
                .metadata
                .as_ref()
                .and_then(|m| m.get("description"))
                .and_then(|v| v.as_str())
                .unwrap_or("Workflow submitted via API")
                .to_string(),
            nodes: request.nodes.clone(),
            edges: request.edges.clone(),
            entry_point_id: entry_point_id.clone(),
            entry_point_name: entry_point_name,
        };

        state
            .workflow_engine
            .register_workflow(Arc::new(workflow))
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("Workflow invalid: {}", e)))?;

        // Seed context variables
        let mut ctx = ExecutionContext::new(workflow_id.clone());
        match request.input {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    ctx.set_variable(k, v);
                }
            }
            other => {
                ctx.set_variable("input".to_string(), other);
            }
        }

        if let Some(tools) = request.tools.clone() {
            ctx.set_variable("__tools".to_string(), serde_json::json!(tools));
        }

        let execution_id = state
            .workflow_engine
            .start_execution_with_context(&workflow_id, ctx)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let exec_state = WorkflowExecution {
            workflow_id: workflow_id.clone(),
            execution_id: execution_id.clone(),
            state: WorkflowState::Running,
            current_node: Some(entry_point_id),
            completed_nodes: Vec::new(),
            results: Vec::new(),
            error: None,
        };
        state
            .workflow_cache
            .insert(
                workflow_id.clone(),
                Arc::new(tokio::sync::RwLock::new(exec_state)),
            )
            .await;

        Ok(Json(WorkflowResponse {
            workflow_id,
            execution_id: Some(execution_id),
            state: WorkflowState::Running,
            results: vec![],
            error: None,
        }))
    }
    .instrument(span)
    .await
}

/// GET /api/orchestration/workflow/:id - Get workflow status
pub async fn get_workflow_status_handler(
    State(state): State<AppState>,
    Path(workflow_id): Path<String>,
) -> Result<Json<WorkflowStatusResponse>, (StatusCode, String)> {
    let span = tracing::span!(tracing::Level::INFO, "orchestration_workflow_status", workflow_id = %workflow_id);

    async move {
        tracing::info!("Received workflow status request");

        let exec_arc = state
            .workflow_cache
            .get(&workflow_id)
            .await
            .ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    format!("Workflow {} not found", workflow_id),
                )
            })?;

        let execution_id = { exec_arc.read().await.execution_id.clone() };

        let ctx_arc = state
            .workflow_engine
            .get_execution(&execution_id)
            .await
            .ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    format!("Execution {} not found", execution_id),
                )
            })?;

        let ctx = ctx_arc.read().await;
        let results: Vec<serde_json::Value> = ctx
            .node_results
            .values()
            .filter_map(|r| r.output.clone())
            .collect();

        let completed_nodes: Vec<String> = ctx.node_results.keys().cloned().collect();
        let error = ctx
            .get_variable("__error")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        Ok(Json(WorkflowStatusResponse {
            workflow_id,
            execution_id: Some(execution_id),
            state: ctx.state.clone(),
            current_node: ctx.current_node_id.clone(),
            completed_nodes,
            results,
            error,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/orchestration/workflow/:id/pause - Pause workflow
pub async fn pause_workflow_handler(
    State(_state): State<AppState>,
    Path(workflow_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let span = tracing::span!(tracing::Level::INFO, "orchestration_workflow_pause", workflow_id = %workflow_id);

    async move {
        tracing::info!("Received workflow pause request");

        // In full implementation, pause the workflow in cache

        Ok(StatusCode::OK)
    }
    .instrument(span)
    .await
}

/// POST /api/orchestration/workflow/:id/resume - Resume workflow
pub async fn resume_workflow_handler(
    State(_state): State<AppState>,
    Path(workflow_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    let span = tracing::span!(tracing::Level::INFO, "orchestration_workflow_resume", workflow_id = %workflow_id);

    async move {
        tracing::info!("Received workflow resume request");

        // In full implementation, resume the workflow from cache

        Ok(StatusCode::OK)
    }
    .instrument(span)
    .await
}

/// POST /api/orchestration/multi-agent - Execute multi-agent pattern
pub async fn multi_agent_handler(
    State(state): State<AppState>,
    Json(request): Json<MultiAgentRequest>,
) -> Result<Json<MultiAgentResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "orchestration_multi_agent", request_id = %request_id, pattern = ?request.pattern);

    #[cfg(feature = "tools-rag")]
    let shared_rag_tool = state.shared_rag_tool.clone();
    drop(state);

    async move {
        tracing::info!("Received multi-agent pattern request");

        // Create agents from configuration
        let mut agents = Vec::new();
        for agent_config in request.agents.clone() {
            let provider_type = agent_config
                .provider
                .as_ref()
                .map(|p| p.parse())
                .transpose()
                .map_err(|e: String| {
                    (StatusCode::BAD_REQUEST, format!("Invalid provider: {}", e))
                })?;

            #[cfg(feature = "tools-rag")]
            let config = {
                let mut c = caliche::AgentServiceConfig::new()
                    .with_provider(provider_type, agent_config.model.clone())
                    .with_tool_names_or_defaults(agent_config.tools.clone());
                c.shared_rag_tool = shared_rag_tool.clone();
                c
            };
            #[cfg(not(feature = "tools-rag"))]
            let config = caliche::AgentServiceConfig::new()
                .with_provider(provider_type, agent_config.model.clone())
                .with_tool_names_or_defaults(agent_config.tools.clone());
            let service = AgentService::from_config(config).await.map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Failed to create agent {}: {}", agent_config.name, e),
                )
            })?;

            agents.push((agent_config, Arc::new(tokio::sync::Mutex::new(service))));
        }

        // Execute pattern
        let pattern_result = match request.pattern {
            MultiAgentPattern::Pipeline => execute_pipeline_pattern(agents, &request.task).await?,
            MultiAgentPattern::Debate => execute_debate_pattern(agents, &request.task).await?,
            MultiAgentPattern::Hierarchical => {
                execute_hierarchical_pattern(agents, &request.task).await?
            }
            MultiAgentPattern::Supervisor => {
                execute_supervisor_pattern(agents, &request.task).await?
            }
            MultiAgentPattern::FanOutCollect => {
                execute_fanout_collect_pattern(agents, &request.task).await?
            }
        };

        Ok(Json(MultiAgentResponse {
            pattern: format!("{:?}", request.pattern),
            final_result: pattern_result.final_result,
            agent_results: pattern_result.agent_results,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/orchestration/coordinator - Route task to an agent, or run a complex-task loop.
pub async fn coordinator_handler(
    State(state): State<AppState>,
    Json(request): Json<CoordinatorRequest>,
) -> Result<Json<CoordinatorResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span =
        tracing::span!(tracing::Level::INFO, "orchestration_coordinator", request_id = %request_id);

    Box::pin(async move {
        tracing::info!("Received coordinator routing request");
        let mut service = Box::pin(coordinator_service(&state, &request)).await?;
        if request.complex {
            tracing::debug!(
                provider = request.subagent_provider.as_deref(),
                model = request.subagent_model.as_deref(),
                "coordinator running complex task"
            );
            let config = caliche::ComplexTaskConfig {
                max_rounds: request.max_rounds.unwrap_or(8),
                verify_before_finish: true,
            };
            let outcome = Box::pin(caliche::run_complex_task(
                &mut service,
                &request.task,
                &config,
            ))
            .await
            .map_err(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Complex task failed: {error}"),
                )
            })?;
            return Ok(Json(CoordinatorResponse {
                selected_agent: "complex".to_string(),
                result: outcome.text,
                capabilities_matched: request.required_capabilities,
                request_id,
            }));
        }

        let selected = Box::pin(dispatch_coordinator_task(&mut service, &request)).await?;
        Ok(Json(CoordinatorResponse {
            selected_agent: selected.0,
            result: selected.1,
            capabilities_matched: request.required_capabilities,
            request_id,
        }))
    })
    .instrument(span)
    .await
}

async fn coordinator_service(
    state: &AppState,
    request: &CoordinatorRequest,
) -> Result<AgentService, (StatusCode, String)> {
    if let Some(shared) = state.shared_service.as_ref() {
        let guard = shared.read().await;
        let service = guard.clone_for_request();
        drop(guard);
        apply_coordinator_subagent_selection(&service, request);
        return Ok(service);
    }
    #[cfg(feature = "tools-rag")]
    let mut config = {
        let mut c = caliche::AgentServiceConfig::new()
            .with_tool_names_or_defaults(coordinator_tool_names(request.tools.clone()));
        c.shared_rag_tool = state.shared_rag_tool.clone();
        c
    };
    #[cfg(not(feature = "tools-rag"))]
    let mut config = caliche::AgentServiceConfig::new()
        .with_tool_names_or_defaults(coordinator_tool_names(request.tools.clone()));
    let workspace = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let provider = config
        .provider
        .unwrap_or_else(|| caliche::agent::service::resolve_provider(None));
    let model = config
        .model
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let mut spawner = HarnessSessionSpawner::new(provider, model, workspace);
    spawner.selection.set(SubagentModelSelection::from_parts(
        request.subagent_provider.as_deref(),
        request.subagent_model.as_deref(),
    ));
    spawner.complex = request.complex;
    config = config.with_session_spawner(Some(Arc::new(spawner)));
    let service = AgentService::from_config(config).await.map_err(|error| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create agent: {error}"),
        )
    })?;
    apply_coordinator_subagent_selection(&service, request);
    Ok(service)
}

fn coordinator_tool_names(tools: Option<Vec<String>>) -> Option<Vec<String>> {
    let Some(mut names) = tools else {
        return None;
    };
    for name in [
        "sessions_list",
        "sessions_history",
        "sessions_send",
        "sessions_spawn",
    ] {
        if !names.iter().any(|existing| existing == name) {
            names.push(name.to_string());
        }
    }
    Some(names)
}

fn apply_coordinator_subagent_selection(service: &AgentService, request: &CoordinatorRequest) {
    service.set_subagent_selection(SubagentModelSelection::from_parts(
        request.subagent_provider.as_deref(),
        request.subagent_model.as_deref(),
    ));
}

async fn dispatch_coordinator_task(
    service: &mut AgentService,
    request: &CoordinatorRequest,
) -> Result<(String, String), (StatusCode, String)> {
    let coordinator = AgentCoordinator::new();
    let capabilities = request
        .required_capabilities
        .iter()
        .map(|name| capability_from_name(name))
        .collect::<Vec<_>>();
    let mut metadata = AgentMetadata::new("default", "primary", capabilities.clone());
    metadata.agent_id = "default".to_string();
    coordinator
        .register_agent(metadata, service.clone_agent())
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to register coordinator agent: {error}"),
            )
        })?;

    if let Some(capability) = capabilities.first() {
        let (agent_id, text, _) = coordinator
            .delegate_task(&request.task, Some(capability))
            .await
            .map_err(|error| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Coordinator dispatch failed: {error}"),
                )
            })?;
        return Ok((agent_id, text));
    }

    let (text, _) = coordinator
        .execute_on_agent("default", &request.task)
        .await
        .map_err(|error| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Coordinator dispatch failed: {error}"),
            )
        })?;
    Ok(("default".to_string(), text))
}

fn capability_from_name(name: &str) -> AgentCapability {
    match name.trim().to_ascii_lowercase().as_str() {
        "research" => AgentCapability::Research,
        "writing" => AgentCapability::Writing,
        "coding" => AgentCapability::Coding,
        "data" | "data_analysis" | "dataanalysis" => AgentCapability::DataAnalysis,
        "planning" => AgentCapability::Planning,
        "coordination" => AgentCapability::Coordination,
        other => AgentCapability::Custom(other.to_owned()),
    }
}

// Helper struct for pattern execution results
struct PatternResult {
    final_result: String,
    agent_results: Vec<AgentResult>,
}

// Execute pipeline pattern
async fn execute_pipeline_pattern(
    agents: Vec<(AgentConfig, Arc<tokio::sync::Mutex<AgentService>>)>,
    task: &str,
) -> Result<PatternResult, (StatusCode, String)> {
    let mut agent_results = Vec::new();
    let mut current_input = task.to_string();

    for (config, agent) in agents {
        let mut agent_lock = agent.lock().await;
        match agent_lock.chat(&current_input).await {
            Ok(chat_response) => {
                let response_text = chat_response.text().to_string();
                agent_results.push(AgentResult {
                    agent_id: config.id.clone(),
                    agent_name: config.name.clone(),
                    output: response_text.clone(),
                    error: None,
                });
                current_input = response_text;
            }
            Err(e) => {
                agent_results.push(AgentResult {
                    agent_id: config.id.clone(),
                    agent_name: config.name.clone(),
                    output: String::new(),
                    error: Some(e.to_string()),
                });
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Pipeline failed at agent {}: {}", config.name, e),
                ));
            }
        }
    }

    Ok(PatternResult {
        final_result: current_input,
        agent_results,
    })
}

// Execute debate pattern
async fn execute_debate_pattern(
    agents: Vec<(AgentConfig, Arc<tokio::sync::Mutex<AgentService>>)>,
    task: &str,
) -> Result<PatternResult, (StatusCode, String)> {
    let mut agent_results = Vec::new();
    let mut responses = Vec::new();

    // Each agent provides their perspective
    for (config, agent) in &agents {
        let mut agent_lock = agent.lock().await;
        match agent_lock.chat(task).await {
            Ok(chat_response) => {
                let response_text = chat_response.text().to_string();
                agent_results.push(AgentResult {
                    agent_id: config.id.clone(),
                    agent_name: config.name.clone(),
                    output: response_text.clone(),
                    error: None,
                });
                responses.push(response_text);
            }
            Err(e) => {
                agent_results.push(AgentResult {
                    agent_id: config.id.clone(),
                    agent_name: config.name.clone(),
                    output: String::new(),
                    error: Some(e.to_string()),
                });
            }
        }
    }

    // Synthesize final result
    let final_result = if responses.is_empty() {
        "No responses from agents".to_string()
    } else {
        format!("Debate synthesis:\n{}", responses.join("\n\n"))
    };

    Ok(PatternResult {
        final_result,
        agent_results,
    })
}

// Execute hierarchical pattern
async fn execute_hierarchical_pattern(
    agents: Vec<(AgentConfig, Arc<tokio::sync::Mutex<AgentService>>)>,
    task: &str,
) -> Result<PatternResult, (StatusCode, String)> {
    // Use first agent as supervisor, rest as workers
    if agents.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Hierarchical pattern requires at least one agent".to_string(),
        ));
    }

    let (supervisor_config, supervisor) = &agents[0];
    let workers = &agents[1..];

    let mut agent_results = Vec::new();

    // Supervisor creates plan
    let plan_prompt = format!(
        "As a supervisor, break down this task into subtasks:\n{}",
        task
    );
    let mut supervisor_lock = supervisor.lock().await;
    let plan_response = supervisor_lock.chat(&plan_prompt).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Supervisor planning failed: {}", e),
        )
    })?;
    let plan = plan_response.text().to_string();

    agent_results.push(AgentResult {
        agent_id: supervisor_config.id.clone(),
        agent_name: supervisor_config.name.clone(),
        output: plan.clone(),
        error: None,
    });

    // Workers execute subtasks
    for (config, worker) in workers {
        let mut worker_lock = worker.lock().await;
        match worker_lock
            .chat(&format!("Execute this subtask: {}", task))
            .await
        {
            Ok(chat_response) => {
                agent_results.push(AgentResult {
                    agent_id: config.id.clone(),
                    agent_name: config.name.clone(),
                    output: chat_response.text().to_string(),
                    error: None,
                });
            }
            Err(e) => {
                agent_results.push(AgentResult {
                    agent_id: config.id.clone(),
                    agent_name: config.name.clone(),
                    output: String::new(),
                    error: Some(e.to_string()),
                });
            }
        }
    }

    Ok(PatternResult {
        final_result: format!("Hierarchical execution completed with plan:\n{}", plan),
        agent_results,
    })
}

// Execute supervisor pattern
async fn execute_supervisor_pattern(
    agents: Vec<(AgentConfig, Arc<tokio::sync::Mutex<AgentService>>)>,
    task: &str,
) -> Result<PatternResult, (StatusCode, String)> {
    // Similar to hierarchical but supervisor monitors and coordinates
    execute_hierarchical_pattern(agents, task).await
}

// Execute fan-out/collect pattern
async fn execute_fanout_collect_pattern(
    agents: Vec<(AgentConfig, Arc<tokio::sync::Mutex<AgentService>>)>,
    task: &str,
) -> Result<PatternResult, (StatusCode, String)> {
    if agents.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "FanOutCollect pattern requires at least one agent".to_string(),
        ));
    }

    let mut agent_results = Vec::new();
    let mut collected_outputs = Vec::new();

    for (config, agent) in agents {
        let mut agent_lock = agent.lock().await;
        let branch_prompt = format!("Branch analysis for task (agent={}): {}", config.name, task);
        match agent_lock.chat(&branch_prompt).await {
            Ok(chat_response) => {
                let output = chat_response.text().to_string();
                collected_outputs.push(output.clone());
                agent_results.push(AgentResult {
                    agent_id: config.id.clone(),
                    agent_name: config.name.clone(),
                    output,
                    error: None,
                });
            }
            Err(e) => {
                agent_results.push(AgentResult {
                    agent_id: config.id.clone(),
                    agent_name: config.name.clone(),
                    output: String::new(),
                    error: Some(e.to_string()),
                });
            }
        }
    }

    let final_result = if collected_outputs.is_empty() {
        "No successful branch outputs to collect".to_string()
    } else {
        collected_outputs.join("\n\n---\n\n")
    };

    Ok(PatternResult {
        final_result,
        agent_results,
    })
}

/// POST /api/workflows - Save a workflow definition
pub async fn save_workflow_handler(
    State(_state): State<AppState>,
    Json(workflow): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Validate workflow structure
    let workflow_def: BuiltWorkflow = serde_json::from_value(workflow)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid workflow: {}", e)))?;

    // Validate workflow (check for cycles, etc.)
    workflow_def.validate().map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Workflow validation failed: {}", e),
        )
    })?;

    // In a real implementation, save to database or file system
    // For now, return the workflow with an ID
    Ok(Json(serde_json::json!({
        "id": workflow_def.id(),
        "name": workflow_def.name(),
        "status": "saved"
    })))
}

/// GET /api/workflows/:id - Load a workflow definition
pub async fn load_workflow_handler(
    State(_state): State<AppState>,
    Path(workflow_id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // In a real implementation, load from database or file system
    // For now, return an error indicating workflow not found
    Err((
        StatusCode::NOT_FOUND,
        format!(
            "Workflow {} not found. Use POST /api/workflows to save workflows.",
            workflow_id
        ),
    ))
}

/// POST /api/workflows/validate - Validate a workflow definition
pub async fn validate_workflow_handler(
    State(_state): State<AppState>,
    Json(workflow): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let workflow_def: BuiltWorkflow = serde_json::from_value(workflow).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid workflow format: {}", e),
        )
    })?;

    match workflow_def.validate() {
        Ok(_) => Ok(Json(serde_json::json!({
            "valid": true,
            "errors": []
        }))),
        Err(e) => Ok(Json(serde_json::json!({
            "valid": false,
            "errors": [e.to_string()]
        }))),
    }
}

/// POST /api/workflows/execute - Execute a workflow
pub async fn execute_workflow_builder_handler(
    State(_state): State<AppState>,
    Json(workflow): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Parse workflow
    let workflow_def: BuiltWorkflow = serde_json::from_value(workflow)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid workflow: {}", e)))?;

    // Validate
    workflow_def.validate().map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Workflow validation failed: {}", e),
        )
    })?;

    // Execute workflow (use existing execute_workflow_handler logic)
    let request = WorkflowRequest {
        nodes: workflow_def.nodes().to_vec(),
        edges: workflow_def.edges().to_vec(),
        input: serde_json::json!({}),
        metadata: None,
        tools: None,
    };

    let Json(response) = execute_workflow_handler(State(_state), Json(request)).await?;
    Ok(Json(serde_json::json!({
        "workflow_id": response.workflow_id,
        "execution_id": response.execution_id,
        "state": response.state,
        "results": response.results
    })))
}

/// GET /api/orchestration/interrupts - List all pending interrupts
pub async fn list_interrupts_handler(
    State(state): State<AppState>,
) -> Result<Json<InterruptsListResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(
        tracing::Level::INFO,
        "orchestration_list_interrupts",
        request_id = %request_id
    );

    async move {
        tracing::info!("Listing pending interrupts");

        let interrupts = state.workflow_engine.list_pending_interrupts().await;

        Ok(Json(InterruptsListResponse { interrupts }))
    }
    .instrument(span)
    .await
}

/// GET /api/orchestration/interrupts/{id} - Get a specific interrupt
pub async fn get_interrupt_handler(
    State(state): State<AppState>,
    Path(interrupt_id): Path<String>,
) -> Result<Json<InterruptResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(
        tracing::Level::INFO,
        "orchestration_get_interrupt",
        request_id = %request_id,
        interrupt_id = %interrupt_id
    );

    async move {
        tracing::info!("Getting interrupt: {}", interrupt_id);

        let interrupt = state
            .workflow_engine
            .get_interrupt(&interrupt_id)
            .await
            .ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    format!("Interrupt {} not found", interrupt_id),
                )
            })?;

        Ok(Json(InterruptResponse { interrupt }))
    }
    .instrument(span)
    .await
}

/// POST /api/orchestration/interrupts/{id}/resume - Resume a paused workflow
pub async fn resume_interrupt_handler(
    State(state): State<AppState>,
    Path(interrupt_id): Path<String>,
    Json(request): Json<ResumeWorkflowRequest>,
) -> Result<Json<ResumeWorkflowResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(
        tracing::Level::INFO,
        "orchestration_resume_interrupt",
        request_id = %request_id,
        interrupt_id = %interrupt_id
    );

    async move {
        tracing::info!("Resuming workflow from interrupt: {}", interrupt_id);

        // Validate interrupt_id matches request
        if request.interrupt_id != interrupt_id {
            return Err((
                StatusCode::BAD_REQUEST,
                "Interrupt ID in path does not match request body".to_string(),
            ));
        }

        state
            .workflow_engine
            .resume_execution(&interrupt_id, request.feedback)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        // Get execution ID from interrupt
        let interrupt = state
            .workflow_engine
            .get_interrupt(&interrupt_id)
            .await
            .ok_or_else(|| {
                (
                    StatusCode::NOT_FOUND,
                    format!("Interrupt {} not found", interrupt_id),
                )
            })?;

        Ok(Json(ResumeWorkflowResponse {
            success: true,
            execution_id: interrupt.execution_id,
            message: "Workflow resumed successfully".to_string(),
        }))
    }
    .instrument(span)
    .await
}
