//! Request inspector for debugging and tracing agent interactions

use crate::messages::{CompletionRequest, CompletionResponse, StreamEvent, TokenUsage};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Detailed request trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestTrace {
    pub request_id: String,
    pub timestamp: DateTime<Utc>,
    pub provider: String,
    pub model: String,
    pub request: CompletionRequest,
    pub response: Option<CompletionResponse>,
    pub stream_events: Vec<StreamEvent>,
    pub tool_calls: Vec<ToolCallTrace>,
    pub timing: TimingInfo,
    pub usage: Option<TokenUsage>,
}

/// Tool call trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallTrace {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub result: String,
    pub duration_ms: u64,
    pub success: bool,
    pub error: Option<String>,
}

/// Timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    pub total_duration_ms: u64,
    pub llm_duration_ms: Option<u64>,
    pub tool_duration_ms: u64,
    pub network_duration_ms: Option<u64>,
    pub processing_duration_ms: Option<u64>,
}

/// Request inspector for tracking and debugging requests
pub struct RequestInspector {
    traces: Arc<RwLock<HashMap<String, RequestTrace>>>,
    max_traces: usize,
}

impl RequestInspector {
    pub fn new(max_traces: usize) -> Self {
        Self {
            traces: Arc::new(RwLock::new(HashMap::new())),
            max_traces,
        }
    }

    pub async fn record_request(&self, trace: RequestTrace) {
        let mut traces = self.traces.write().await;

        // Evict oldest if at capacity
        if traces.len() >= self.max_traces {
            let oldest_key = traces
                .iter()
                .min_by_key(|(_, t)| t.timestamp)
                .map(|(k, _)| k.clone());
            if let Some(key) = oldest_key {
                traces.remove(&key);
            }
        }

        traces.insert(trace.request_id.clone(), trace);
    }

    pub async fn get_trace(&self, request_id: &str) -> Option<RequestTrace> {
        self.traces.read().await.get(request_id).cloned()
    }

    pub async fn list_traces(&self, limit: Option<usize>) -> Vec<RequestTrace> {
        let traces = self.traces.read().await;
        let mut traces: Vec<RequestTrace> = traces.values().cloned().collect();
        traces.sort_by_key(|t| t.timestamp);
        traces.reverse(); // Most recent first

        if let Some(limit) = limit {
            traces.truncate(limit);
        }
        traces
    }

    pub async fn clear(&self) {
        self.traces.write().await.clear();
    }

    pub async fn export_traces(&self, format: &str) -> Result<String, anyhow::Error> {
        let traces: Vec<RequestTrace> = self.traces.read().await.values().cloned().collect();

        match format {
            "json" => Ok(serde_json::to_string_pretty(&traces)?),
            "yaml" => Ok(serde_norway::to_string(&traces)?),
            _ => Err(anyhow::anyhow!("Unsupported format: {}", format)),
        }
    }
}

impl Default for RequestInspector {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Helper to create a request trace
pub fn create_trace(
    request_id: String,
    provider: String,
    model: String,
    request: CompletionRequest,
) -> RequestTrace {
    RequestTrace {
        request_id,
        timestamp: Utc::now(),
        provider,
        model,
        request,
        response: None,
        stream_events: Vec::new(),
        tool_calls: Vec::new(),
        timing: TimingInfo {
            total_duration_ms: 0,
            llm_duration_ms: None,
            tool_duration_ms: 0,
            network_duration_ms: None,
            processing_duration_ms: None,
        },
        usage: None,
    }
}
