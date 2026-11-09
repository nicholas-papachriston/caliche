//! Persistent tool call audit trail via AgentFS ToolCalls interface.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

impl std::fmt::Debug for AgentFsToolLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentFsToolLogger").finish_non_exhaustive()
    }
}
use std::sync::Arc;

/// Logs tool executions to AgentFS for persistent, queryable audit trail.
#[derive(Clone)]
pub struct AgentFsToolLogger {
    agentfs: Arc<agentfs_sdk::AgentFS>,
}

pub struct ToolLogRecord<'a> {
    pub name: &'a str,
    pub started_at: DateTime<Utc>,
    pub completed_at: DateTime<Utc>,
    pub parameters: serde_json::Value,
    pub success: bool,
    pub result_content: &'a str,
    pub error: Option<&'a str>,
}

impl AgentFsToolLogger {
    /// Create a tool logger from an AgentFS instance.
    pub fn new(agentfs: Arc<agentfs_sdk::AgentFS>) -> Self {
        Self { agentfs }
    }

    /// Record a completed tool execution to the audit trail.
    pub async fn record(&self, record: ToolLogRecord<'_>) -> Result<()> {
        let started_ts = record.started_at.timestamp();
        let completed_ts = record.completed_at.timestamp();
        let params = if record.parameters.is_null()
            || record
                .parameters
                .as_object()
                .map(|o| o.is_empty())
                .unwrap_or(true)
        {
            None
        } else {
            Some(record.parameters)
        };
        let (result_val, err_val) = if record.success {
            (
                Some(serde_json::json!({ "content": record.result_content })),
                None as Option<&str>,
            )
        } else {
            (None, record.error)
        };

        let _ = self
            .agentfs
            .tools
            .record(
                record.name,
                started_ts,
                completed_ts,
                params,
                result_val,
                err_val,
            )
            .await
            .context("AgentFS tool record failed")?;
        Ok(())
    }
}
