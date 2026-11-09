//! Tool execution metrics collection for observability.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Statistics for a single tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolStats {
    pub total_calls: u64,
    pub successful_calls: u64,
    pub failed_calls: u64,
    pub total_duration_ms: u64,
    pub error_types: HashMap<String, u64>,
    pub last_executed: Option<DateTime<Utc>>,
}

impl ToolStats {
    pub fn new() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            total_duration_ms: 0,
            error_types: HashMap::new(),
            last_executed: None,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.successful_calls as f64 / self.total_calls as f64
    }

    pub fn avg_duration_ms(&self) -> f64 {
        if self.total_calls == 0 {
            return 0.0;
        }
        self.total_duration_ms as f64 / self.total_calls as f64
    }
}

impl Default for ToolStats {
    fn default() -> Self {
        Self::new()
    }
}

/// Tool metrics tracker
pub struct ToolMetrics {
    executions: Arc<RwLock<HashMap<String, ToolStats>>>,
}

impl ToolMetrics {
    pub fn new() -> Self {
        Self {
            executions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a tool execution
    pub async fn record_execution(
        &self,
        tool_name: &str,
        duration_ms: u64,
        success: bool,
        error_type: Option<&str>,
    ) {
        let mut stats = self.executions.write().await;
        let tool_stats = stats
            .entry(tool_name.to_string())
            .or_insert_with(ToolStats::new);

        tool_stats.total_calls += 1;
        tool_stats.total_duration_ms += duration_ms;
        tool_stats.last_executed = Some(Utc::now());

        if success {
            tool_stats.successful_calls += 1;
        } else {
            tool_stats.failed_calls += 1;
            if let Some(error) = error_type {
                *tool_stats.error_types.entry(error.to_string()).or_insert(0) += 1;
            }
        }
    }

    /// Get statistics for a specific tool
    pub async fn get_tool_stats(&self, tool_name: &str) -> Option<ToolStats> {
        let stats = self.executions.read().await;
        stats.get(tool_name).cloned()
    }

    /// Get statistics for all tools
    pub async fn get_all_stats(&self) -> HashMap<String, ToolStats> {
        let stats = self.executions.read().await;
        stats.clone()
    }

    /// Clear all metrics (useful for testing)
    pub async fn clear(&self) {
        let mut stats = self.executions.write().await;
        stats.clear();
    }
}

impl Default for ToolMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Error categorization helper
pub fn categorize_error(error: &str) -> String {
    let error_lower = error.to_lowercase();

    if error_lower.contains("timeout") || error_lower.contains("timed out") {
        "timeout".to_string()
    } else if error_lower.contains("validation") || error_lower.contains("invalid") {
        "validation".to_string()
    } else if error_lower.contains("not found") || error_lower.contains("missing") {
        "not_found".to_string()
    } else if error_lower.contains("permission") || error_lower.contains("unauthorized") {
        "permission".to_string()
    } else if error_lower.contains("rate limit") || error_lower.contains("quota") {
        "rate_limit".to_string()
    } else {
        "execution".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_metrics() {
        let metrics = ToolMetrics::new();

        metrics.record_execution("calculator", 50, true, None).await;
        metrics.record_execution("calculator", 60, true, None).await;
        metrics
            .record_execution("calculator", 100, false, Some("validation"))
            .await;

        let stats = metrics
            .get_tool_stats("calculator")
            .await
            .expect("Should get tool stats");
        assert_eq!(stats.total_calls, 3);
        assert_eq!(stats.successful_calls, 2);
        assert_eq!(stats.failed_calls, 1);
        assert_eq!(stats.error_types.get("validation"), Some(&1));
        assert!((stats.success_rate() - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_error_categorization() {
        assert_eq!(categorize_error("Request timed out"), "timeout");
        assert_eq!(categorize_error("Invalid parameter"), "validation");
        assert_eq!(categorize_error("File not found"), "not_found");
    }
}
