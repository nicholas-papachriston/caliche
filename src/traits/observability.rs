//! Observability hooks abstraction for custom observability and debugging.
//!
//! This module provides a unified interface for observing agent behavior
//! through hooks. Implement the `ObservabilityHook` trait to add custom
//! observability integrations like metrics, tracing, or logging.

use crate::messages::{CompletionRequest, CompletionResponse};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// Trait for observability hooks that can observe agent behavior
#[async_trait]
pub trait ObservabilityHook: Send + Sync + std::fmt::Debug {
    /// Called when a request starts
    ///
    /// # Arguments
    /// * `request` - The completion request
    ///
    /// # Returns
    /// * `Ok(())` - Successfully handled
    /// * `Err` - Error handling hook (should not fail the request)
    async fn on_request_start(&self, request: &CompletionRequest) -> Result<()>;

    /// Called when a request completes
    ///
    /// # Arguments
    /// * `request` - The original completion request
    /// * `response` - The completion response
    ///
    /// # Returns
    /// * `Ok(())` - Successfully handled
    /// * `Err` - Error handling hook (should not fail the request)
    async fn on_request_complete(
        &self,
        request: &CompletionRequest,
        response: &CompletionResponse,
    ) -> Result<()>;

    /// Called when a tool is called
    ///
    /// # Arguments
    /// * `tool_name` - Name of the tool being called
    /// * `args` - Tool arguments as JSON
    ///
    /// # Returns
    /// * `Ok(())` - Successfully handled
    /// * `Err` - Error handling hook (should not fail the request)
    async fn on_tool_call(&self, tool_name: &str, args: &serde_json::Value) -> Result<()>;

    /// Called when an error occurs
    ///
    /// # Arguments
    /// * `error` - The error that occurred
    ///
    /// # Returns
    /// * `Ok(())` - Successfully handled
    /// * `Err` - Error handling hook (should not fail the request)
    async fn on_error(&self, error: &anyhow::Error) -> Result<()>;

    /// Get the name of this hook (for debugging/logging)
    fn hook_name(&self) -> &str;
}

/// Observability hook that records metrics
#[derive(Debug)]
pub struct MetricsHook {
    request_count: std::sync::atomic::AtomicU64,
    tool_call_count: std::sync::atomic::AtomicU64,
    error_count: std::sync::atomic::AtomicU64,
}

impl MetricsHook {
    /// Create a new metrics hook
    pub fn new() -> Self {
        Self {
            request_count: std::sync::atomic::AtomicU64::new(0),
            tool_call_count: std::sync::atomic::AtomicU64::new(0),
            error_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get current metrics
    pub fn get_metrics(&self) -> HookMetrics {
        HookMetrics {
            request_count: self
                .request_count
                .load(std::sync::atomic::Ordering::Relaxed),
            tool_call_count: self
                .tool_call_count
                .load(std::sync::atomic::Ordering::Relaxed),
            error_count: self.error_count.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

impl Default for MetricsHook {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics collected by MetricsHook
#[derive(Debug, Clone)]
pub struct HookMetrics {
    pub request_count: u64,
    pub tool_call_count: u64,
    pub error_count: u64,
}

#[async_trait]
impl ObservabilityHook for MetricsHook {
    async fn on_request_start(&self, _request: &CompletionRequest) -> Result<()> {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn on_request_complete(
        &self,
        _request: &CompletionRequest,
        _response: &CompletionResponse,
    ) -> Result<()> {
        // Metrics already incremented in on_request_start
        Ok(())
    }

    async fn on_tool_call(&self, _tool_name: &str, _args: &serde_json::Value) -> Result<()> {
        self.tool_call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn on_error(&self, _error: &anyhow::Error) -> Result<()> {
        self.error_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn hook_name(&self) -> &str {
        "metrics"
    }
}

/// Observability hook that logs events using tracing
#[derive(Debug, Clone)]
pub struct TracingHook {
    log_level: TracingLogLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum TracingLogLevel {
    Debug,
    Info,
    Trace,
}

impl TracingHook {
    /// Create a new tracing hook
    ///
    /// # Arguments
    /// * `log_level` - Log level to use
    pub fn new(log_level: TracingLogLevel) -> Self {
        Self { log_level }
    }

    /// Create a debug-level tracing hook
    pub fn debug() -> Self {
        Self::new(TracingLogLevel::Debug)
    }
}

#[async_trait]
impl ObservabilityHook for TracingHook {
    async fn on_request_start(&self, request: &CompletionRequest) -> Result<()> {
        match self.log_level {
            TracingLogLevel::Debug => {
                tracing::debug!("Request started: {} messages", request.messages.len())
            }
            TracingLogLevel::Info => {
                tracing::info!("Request started: {} messages", request.messages.len())
            }
            TracingLogLevel::Trace => {
                tracing::trace!("Request started: {:?}", request)
            }
        }
        Ok(())
    }

    async fn on_request_complete(
        &self,
        request: &CompletionRequest,
        response: &CompletionResponse,
    ) -> Result<()> {
        match self.log_level {
            TracingLogLevel::Debug => {
                tracing::debug!(
                    "Request completed: {} messages, response length: {}",
                    request.messages.len(),
                    response.content.as_text().len()
                )
            }
            TracingLogLevel::Info => {
                tracing::info!(
                    "Request completed: {} messages, response length: {}",
                    request.messages.len(),
                    response.content.as_text().len()
                )
            }
            TracingLogLevel::Trace => {
                tracing::trace!("Request completed: {:?}", response)
            }
        }
        Ok(())
    }

    async fn on_tool_call(&self, tool_name: &str, args: &serde_json::Value) -> Result<()> {
        match self.log_level {
            TracingLogLevel::Debug => {
                tracing::debug!("Tool called: {} with args: {:?}", tool_name, args)
            }
            TracingLogLevel::Info => {
                tracing::info!("Tool called: {}", tool_name)
            }
            TracingLogLevel::Trace => {
                tracing::trace!("Tool called: {} with args: {:?}", tool_name, args)
            }
        }
        Ok(())
    }

    async fn on_error(&self, error: &anyhow::Error) -> Result<()> {
        tracing::error!("Error occurred: {}", error);
        Ok(())
    }

    fn hook_name(&self) -> &str {
        "tracing"
    }
}

/// Observability hook that logs events using structured logging
#[derive(Debug, Clone)]
pub struct LoggingHook {
    log_requests: bool,
    log_tool_calls: bool,
    log_errors: bool,
}

impl LoggingHook {
    /// Create a new logging hook
    ///
    /// # Arguments
    /// * `log_requests` - Whether to log requests
    /// * `log_tool_calls` - Whether to log tool calls
    /// * `log_errors` - Whether to log errors
    pub fn new(log_requests: bool, log_tool_calls: bool, log_errors: bool) -> Self {
        Self {
            log_requests,
            log_tool_calls,
            log_errors,
        }
    }

    /// Create a logging hook that logs everything
    pub fn all() -> Self {
        Self::new(true, true, true)
    }
}

#[async_trait]
impl ObservabilityHook for LoggingHook {
    async fn on_request_start(&self, request: &CompletionRequest) -> Result<()> {
        if self.log_requests {
            tracing::info!("Request started: {} messages", request.messages.len());
        }
        Ok(())
    }

    async fn on_request_complete(
        &self,
        request: &CompletionRequest,
        response: &CompletionResponse,
    ) -> Result<()> {
        if self.log_requests {
            tracing::info!(
                "Request completed: {} messages, response length: {}",
                request.messages.len(),
                response.content.as_text().len()
            );
        }
        Ok(())
    }

    async fn on_tool_call(&self, tool_name: &str, args: &serde_json::Value) -> Result<()> {
        if self.log_tool_calls {
            tracing::info!("Tool called: {} with args: {}", tool_name, args);
        }
        Ok(())
    }

    async fn on_error(&self, error: &anyhow::Error) -> Result<()> {
        if self.log_errors {
            tracing::error!("Error occurred: {}", error);
        }
        Ok(())
    }

    fn hook_name(&self) -> &str {
        "logging"
    }
}

/// Observability hook that combines multiple hooks
#[derive(Debug)]
pub struct CompositeHook {
    hooks: Vec<Arc<dyn ObservabilityHook>>,
}

impl CompositeHook {
    /// Create a new composite hook
    pub fn new(hooks: Vec<Arc<dyn ObservabilityHook>>) -> Self {
        Self { hooks }
    }
}

#[async_trait]
impl ObservabilityHook for CompositeHook {
    async fn on_request_start(&self, request: &CompletionRequest) -> Result<()> {
        for hook in &self.hooks {
            if let Err(e) = hook.on_request_start(request).await {
                tracing::warn!(
                    "Hook '{}' failed in on_request_start: {}",
                    hook.hook_name(),
                    e
                );
            }
        }
        Ok(())
    }

    async fn on_request_complete(
        &self,
        request: &CompletionRequest,
        response: &CompletionResponse,
    ) -> Result<()> {
        for hook in &self.hooks {
            if let Err(e) = hook.on_request_complete(request, response).await {
                tracing::warn!(
                    "Hook '{}' failed in on_request_complete: {}",
                    hook.hook_name(),
                    e
                );
            }
        }
        Ok(())
    }

    async fn on_tool_call(&self, tool_name: &str, args: &serde_json::Value) -> Result<()> {
        for hook in &self.hooks {
            if let Err(e) = hook.on_tool_call(tool_name, args).await {
                tracing::warn!("Hook '{}' failed in on_tool_call: {}", hook.hook_name(), e);
            }
        }
        Ok(())
    }

    async fn on_error(&self, error: &anyhow::Error) -> Result<()> {
        for hook in &self.hooks {
            if let Err(e) = hook.on_error(error).await {
                tracing::warn!("Hook '{}' failed in on_error: {}", hook.hook_name(), e);
            }
        }
        Ok(())
    }

    fn hook_name(&self) -> &str {
        "composite"
    }
}

/// No-op observability hook that does nothing
#[derive(Debug, Clone)]
pub struct NoOpHook;

impl NoOpHook {
    /// Create a new no-op hook
    pub fn new() -> Self {
        Self
    }
}

impl Default for NoOpHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ObservabilityHook for NoOpHook {
    async fn on_request_start(&self, _request: &CompletionRequest) -> Result<()> {
        Ok(())
    }

    async fn on_request_complete(
        &self,
        _request: &CompletionRequest,
        _response: &CompletionResponse,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_tool_call(&self, _tool_name: &str, _args: &serde_json::Value) -> Result<()> {
        Ok(())
    }

    async fn on_error(&self, _error: &anyhow::Error) -> Result<()> {
        Ok(())
    }

    fn hook_name(&self) -> &str {
        "no-op"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{CompletionContent, Message};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_metrics_hook() {
        let hook = MetricsHook::new();

        let request = CompletionRequest {
            messages: Arc::new(vec![Message::user("test")]),
            max_tokens: None,
            temperature: None,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        hook.on_request_start(&request)
            .await
            .expect("on_request_start should succeed");
        hook.on_tool_call("test_tool", &serde_json::json!({}))
            .await
            .expect("on_tool_call should succeed");
        hook.on_error(&anyhow::anyhow!("test error"))
            .await
            .expect("on_error should succeed");

        let metrics = hook.get_metrics();
        assert_eq!(metrics.request_count, 1);
        assert_eq!(metrics.tool_call_count, 1);
        assert_eq!(metrics.error_count, 1);
    }

    #[tokio::test]
    async fn test_composite_hook() {
        let metrics_hook = Arc::new(MetricsHook::new());
        let logging_hook = Arc::new(LoggingHook::all());

        let composite = CompositeHook::new(vec![metrics_hook.clone(), logging_hook]);

        let request = CompletionRequest {
            messages: Arc::new(vec![Message::user("test")]),
            max_tokens: None,
            temperature: None,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        composite
            .on_request_start(&request)
            .await
            .expect("on_request_start should succeed");

        let metrics = metrics_hook.get_metrics();
        assert_eq!(metrics.request_count, 1);
    }

    #[tokio::test]
    async fn test_no_op_hook() {
        let hook = NoOpHook::new();

        let request = CompletionRequest {
            messages: Arc::new(vec![Message::user("test")]),
            max_tokens: None,
            temperature: None,
            reasoning_effort: None,
            tools: None,
            tool_choice: None,
            response_format: None,
        };

        let response = CompletionResponse {
            content: CompletionContent::Text {
                content: "response".to_string(),
            },
            finish_reason: None,
            usage: None,
            reasoning_content: None,
        };

        hook.on_request_start(&request)
            .await
            .expect("on_request_start should succeed");
        hook.on_request_complete(&request, &response)
            .await
            .expect("on_request_complete should succeed");
        hook.on_tool_call("test", &serde_json::json!({}))
            .await
            .expect("on_tool_call should succeed");
        hook.on_error(&anyhow::anyhow!("test"))
            .await
            .expect("on_error should succeed");
    }
}
