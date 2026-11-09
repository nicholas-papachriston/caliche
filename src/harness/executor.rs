//! Parallel Tool Executor with progress tracking.
//!
//! Executes tool calls in parallel with real-time event emission.

use crate::agent::tools::{Tool, ToolCall, ToolRegistry, ToolResult};
use crate::harness::events::AgentEvent;
use crate::observability::ToolMetrics;
use crate::utils::user_context::UserContext;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;

/// Parallel tool executor with event streaming.
pub struct ParallelToolExecutor {
    event_sender: broadcast::Sender<AgentEvent>,
    max_concurrent: usize,
}

impl ParallelToolExecutor {
    /// Build an executor that broadcasts harness tool events.
    pub fn new(event_sender: broadcast::Sender<AgentEvent>, max_concurrent: usize) -> Self {
        Self {
            event_sender,
            max_concurrent,
        }
    }

    /// Execute a batch of tool calls in parallel
    pub async fn execute_batch(
        &self,
        tools: &ToolRegistry,
        calls: Vec<ToolCall>,
        _user_context: Option<&UserContext>,
        _metrics: Option<&Arc<ToolMetrics>>,
    ) -> Vec<(String, ToolResult)> {
        if calls.is_empty() {
            return Vec::new();
        }

        let calls: Vec<(String, ToolCall)> = calls
            .into_iter()
            .map(|call| (crate::utils::time_id::generate_time_id().to_string(), call))
            .collect();

        for (action_id, call) in &calls {
            self.event_sender
                .send(AgentEvent::tool_queued(
                    action_id,
                    call.name.as_ref(),
                    &call.arguments,
                    action_id,
                ))
                .ok();
        }

        // Use semaphore to limit concurrency
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.max_concurrent));

        let futures: Vec<_> = calls
            .into_iter()
            .map(|(action_id, call)| {
                let permit = semaphore.clone();
                let event_sender = self.event_sender.clone();
                let tool_name = call.name.clone();
                let tool_args = call.arguments.clone();
                let tool_id = action_id;

                async move {
                    let _permit = match permit.acquire().await {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::warn!("Failed to acquire semaphore permit: {}", e);
                            return (
                                tool_name.as_ref().to_string(),
                                ToolResult {
                                    success: false,
                                    content: format!("Failed to acquire execution permit: {}", e),
                                    error: Some(e.to_string()),
                                    typed_payload: None,
                                    usage: None,
                                    warnings: None,
                                },
                            );
                        }
                    };

                    event_sender
                        .send(AgentEvent::ToolExecuting {
                            tool_id: tool_id.clone(),
                            tool_name: tool_name.as_ref().to_string(),
                            progress: Some(0.0),
                            action_id: tool_id.clone(),
                        })
                        .ok();

                    let start = Instant::now();
                    let result = tools.execute(tool_name.as_ref(), tool_args).await;
                    let duration_ms = start.elapsed().as_millis() as u64;

                    if result.success {
                        event_sender
                            .send(AgentEvent::tool_complete(
                                &tool_id,
                                tool_name.as_ref(),
                                &result,
                                duration_ms,
                                &tool_id,
                            ))
                            .ok();
                    } else {
                        event_sender
                            .send(AgentEvent::ToolError {
                                tool_id: tool_id.clone(),
                                tool_name: tool_name.as_ref().to_string(),
                                error: result
                                    .error
                                    .clone()
                                    .unwrap_or_else(|| "Unknown error".to_string()),
                                timestamp: chrono::Utc::now(),
                                action_id: tool_id.clone(),
                            })
                            .ok();
                    }

                    (tool_name.as_ref().to_string(), result)
                }
            })
            .collect();

        futures_util::future::join_all(futures).await
    }

    /// Execute a single tool call
    pub async fn execute_single(
        &self,
        tools: &ToolRegistry,
        call: ToolCall,
        user_context: Option<&UserContext>,
        metrics: Option<&Arc<ToolMetrics>>,
    ) -> ToolResult {
        let results = self
            .execute_batch(tools, vec![call], user_context, metrics)
            .await;
        results
            .into_iter()
            .next()
            .map(|(_, result)| result)
            .unwrap_or_else(|| ToolResult::error("No result returned".to_string()))
    }

    /// Execute a tool call with progress updates
    pub async fn execute_with_progress<F>(
        &self,
        tool: &Arc<dyn Tool>,
        args: serde_json::Value,
        progress_callback: F,
    ) -> ToolResult
    where
        F: Fn(f32) + Send + Sync + 'static,
    {
        let tool_id = crate::utils::time_id::generate_time_id().to_string();
        let tool_name = tool.name().to_string();

        self.event_sender
            .send(AgentEvent::tool_queued(
                &tool_id, &tool_name, &args, &tool_id,
            ))
            .ok();

        self.event_sender
            .send(AgentEvent::ToolExecuting {
                tool_id: tool_id.clone(),
                tool_name: tool_name.clone(),
                progress: Some(0.0),
                action_id: tool_id.clone(),
            })
            .ok();

        let start = Instant::now();

        // Note: Currently tools don't support progress callbacks internally
        // This is a framework for future enhancement
        progress_callback(0.0);

        let result = tool.execute(args).await;

        progress_callback(1.0);

        let duration_ms = start.elapsed().as_millis() as u64;

        // Emit completion event
        if result.success {
            self.event_sender
                .send(AgentEvent::tool_complete(
                    &tool_id,
                    &tool_name,
                    &result,
                    duration_ms,
                    &tool_id,
                ))
                .ok();
        } else {
            self.event_sender
                .send(AgentEvent::ToolError {
                    tool_id: tool_id.clone(),
                    tool_name,
                    error: result
                        .error
                        .clone()
                        .unwrap_or_else(|| "Unknown error".to_string()),
                    timestamp: chrono::Utc::now(),
                    action_id: tool_id,
                })
                .ok();
        }

        result
    }
}

impl std::fmt::Debug for ParallelToolExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelToolExecutor")
            .field("max_concurrent", &self.max_concurrent)
            .finish()
    }
}
