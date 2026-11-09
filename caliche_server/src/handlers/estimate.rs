//! Cost estimation API handlers.

use crate::models::{ChatEstimateRequest, ChatEstimateResponse};
use axum::http::StatusCode;
use axum::response::Json;
use caliche::cost::CostEstimator;
use caliche::llm::ProviderType;
use tracing::Instrument;
use uuid::Uuid;

/// POST /api/chat/estimate - Estimate cost for a chat request without executing
pub async fn estimate_chat_handler(
    Json(request): Json<ChatEstimateRequest>,
) -> Result<Json<ChatEstimateResponse>, (StatusCode, String)> {
    let request_id = Uuid::now_v7().to_string();
    let span = tracing::span!(tracing::Level::INFO, "chat_estimate", request_id = %request_id);

    async move {
        tracing::info!("Received cost estimation request");

        let provider = request
            .provider
            .as_deref()
            .and_then(|provider| provider.parse::<ProviderType>().ok())
            .unwrap_or(ProviderType::OpenAI); // Default

        let tools = request.tools.unwrap_or_default();
        let context_size = request.context_size.unwrap_or(0);

        let estimator = CostEstimator::new();
        let estimate = estimator.estimate_chat(
            &request.message,
            provider,
            request.model.as_deref(),
            context_size,
            &tools,
        );

        Ok(Json(ChatEstimateResponse {
            estimated_cost_usd: estimate.estimated_cost_usd,
            breakdown: estimate.breakdown,
            confidence: estimate.confidence,
            request_id,
        }))
    }
    .instrument(span)
    .await
}
