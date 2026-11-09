//! Reasoning pattern API handlers.

use crate::AppState;
use crate::models::{
    ReasoningPatternInfo, ReasoningPatternsResponse, ReasoningRequest, ReasoningResponse,
    TreeOfThoughtsRequest, TreeOfThoughtsResponse,
};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Json;
use caliche::agent::service::AgentService;
use caliche::reasoning::{self, ReasoningPattern, StepType};
use tracing::Instrument;
use uuid::Uuid;

/// POST /api/reasoning/react - Execute ReAct reasoning pattern
pub async fn react_handler(
    State(state): State<AppState>,
    Json(request): Json<ReasoningRequest>,
) -> Result<Json<ReasoningResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "reasoning_react", request_id = %request_id);

    async move {
        tracing::info!("Received ReAct reasoning request");

        let service = create_or_get_service(&state, &request).await?;
        let max_iterations = request.max_iterations.unwrap_or(5);

        let pattern = reasoning::react::ReActPattern::new(service, max_iterations);
        let result = pattern
            .execute(&request.message)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(ReasoningResponse {
            final_answer: result.final_answer,
            steps: result.steps,
            success: result.success,
            error: result.error,
            token_usage: result.token_usage,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/reasoning/plan-execute - Execute Plan-and-Execute pattern
pub async fn plan_execute_handler(
    State(state): State<AppState>,
    Json(request): Json<ReasoningRequest>,
) -> Result<Json<ReasoningResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span =
        tracing::span!(tracing::Level::INFO, "reasoning_plan_execute", request_id = %request_id);

    async move {
        tracing::info!("Received Plan-Execute reasoning request");

        let service = create_or_get_service(&state, &request).await?;
        let max_steps = request.max_iterations.unwrap_or(10);

        let pattern = reasoning::plan_execute::PlanExecutePattern::new(service, max_steps);
        let result = pattern
            .execute(&request.message)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(ReasoningResponse {
            final_answer: result.final_answer,
            steps: result.steps,
            success: result.success,
            error: result.error,
            token_usage: result.token_usage,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/reasoning/self-ask - Execute Self-Ask pattern
pub async fn self_ask_handler(
    State(state): State<AppState>,
    Json(request): Json<ReasoningRequest>,
) -> Result<Json<ReasoningResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "reasoning_self_ask", request_id = %request_id);

    async move {
        tracing::info!("Received Self-Ask reasoning request");

        let service = create_or_get_service(&state, &request).await?;
        let max_depth = request.max_iterations.unwrap_or(5);

        let pattern = reasoning::self_ask::SelfAskPattern::new(service, max_depth);
        let result = pattern
            .execute(&request.message)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(ReasoningResponse {
            final_answer: result.final_answer,
            steps: result.steps,
            success: result.success,
            error: result.error,
            token_usage: result.token_usage,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/reasoning/reflexion - Execute Reflexion pattern
pub async fn reflexion_handler(
    State(state): State<AppState>,
    Json(request): Json<ReasoningRequest>,
) -> Result<Json<ReasoningResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span =
        tracing::span!(tracing::Level::INFO, "reasoning_reflexion", request_id = %request_id);

    async move {
        tracing::info!("Received Reflexion reasoning request");

        let service = create_or_get_service(&state, &request).await?;
        let max_reflections = request.max_iterations.unwrap_or(3);

        let pattern = reasoning::reflexion::ReflexionPattern::new(service, max_reflections);
        let result = pattern
            .execute(&request.message)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(ReasoningResponse {
            final_answer: result.final_answer,
            steps: result.steps,
            success: result.success,
            error: result.error,
            token_usage: result.token_usage,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// POST /api/reasoning/tree-of-thoughts - Execute Tree of Thoughts pattern
pub async fn tree_of_thoughts_handler(
    State(state): State<AppState>,
    Json(request): Json<TreeOfThoughtsRequest>,
) -> Result<Json<TreeOfThoughtsResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "reasoning_tot", request_id = %request_id);

    async move {
        tracing::info!("Received Tree of Thoughts reasoning request");

        // Create service from request parameters
        let provider_type = request
            .provider
            .as_ref()
            .map(|p| p.parse())
            .transpose()
            .map_err(|e: String| (StatusCode::BAD_REQUEST, format!("Invalid provider: {}", e)))?;

        let mut config = caliche::AgentServiceConfig::new()
            .with_provider(provider_type, request.model.clone())
            .with_tool_names_or_defaults(request.tools.clone());
        #[cfg(feature = "tools-rag")]
        {
            config.shared_rag_tool = state.shared_rag_tool.clone();
        }
        let service = AgentService::from_config(config).await.map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create service: {}", e),
            )
        })?;

        let num_branches = request.num_branches.unwrap_or(3);
        let depth = request.depth.unwrap_or(3);

        let pattern =
            reasoning::tree_of_thoughts::TreeOfThoughtsPattern::new(service, depth, num_branches);
        let result = pattern
            .execute(&request.problem)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Ok(Json(TreeOfThoughtsResponse {
            final_answer: result.final_answer,
            explored_paths: result
                .steps
                .iter()
                .filter(|s| s.step_type == StepType::Thought)
                .map(|s| vec![s.content.clone()])
                .collect(),
            best_path: result
                .steps
                .iter()
                .filter(|s| s.step_type == StepType::Answer)
                .map(|s| s.content.clone())
                .collect(),
            success: result.success,
            error: result.error,
            request_id,
        }))
    }
    .instrument(span)
    .await
}

/// GET /api/reasoning/patterns - List available reasoning patterns
pub async fn list_patterns_handler() -> Json<ReasoningPatternsResponse> {
    let patterns = vec![
        ReasoningPatternInfo {
            name: "react".to_string(),
            description: "Reasoning + Acting: Interleaves thinking and tool use".to_string(),
            default_max_iterations: 5,
        },
        ReasoningPatternInfo {
            name: "plan-execute".to_string(),
            description: "Separates planning from execution for complex tasks".to_string(),
            default_max_iterations: 10,
        },
        ReasoningPatternInfo {
            name: "self-ask".to_string(),
            description: "Recursively decomposes questions into sub-questions".to_string(),
            default_max_iterations: 5,
        },
        ReasoningPatternInfo {
            name: "reflexion".to_string(),
            description: "Self-critique and iterative improvement through reflection".to_string(),
            default_max_iterations: 3,
        },
        ReasoningPatternInfo {
            name: "tree-of-thoughts".to_string(),
            description: "Explores multiple reasoning paths in parallel".to_string(),
            default_max_iterations: 3,
        },
    ];

    Json(ReasoningPatternsResponse { patterns })
}

// Helper function to create or retrieve cached service
async fn create_or_get_service(
    state: &AppState,
    request: &ReasoningRequest,
) -> Result<AgentService, (StatusCode, String)> {
    let provider_type = request
        .provider
        .as_ref()
        .map(|p| p.parse())
        .transpose()
        .map_err(|e: String| (StatusCode::BAD_REQUEST, format!("Invalid provider: {}", e)))?;

    let mut config = caliche::AgentServiceConfig::new()
        .with_provider(provider_type, request.model.clone())
        .with_tool_names_or_defaults(request.tools.clone());
    #[cfg(feature = "tools-rag")]
    {
        config.shared_rag_tool = state.shared_rag_tool.clone();
    }
    let service = AgentService::from_config(config).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create service: {}", e),
        )
    })?;

    Ok(service)
}
