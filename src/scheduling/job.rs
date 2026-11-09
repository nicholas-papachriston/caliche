//! Cron job definition and schedule types.

use crate::llm::ProviderType;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// A scheduled agent task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    pub id: String,
    pub name: String,
    pub schedule: JobSchedule,
    pub agent_config: JobAgentConfig,
    pub delivery: Vec<DeliveryTarget>,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    /// One-shot jobs disable themselves after first run.
    pub one_shot: bool,
    /// Max runtime before abort (default: 300s).
    pub timeout_seconds: u64,
}

/// When the job runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobSchedule {
    /// Standard cron expression (e.g. "0 9 * * MON-FRI").
    Cron(String),
    /// Interval-based (e.g. every 30 minutes).
    Interval { seconds: u64 },
}

/// Agent configuration for a scheduled job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobAgentConfig {
    pub provider: Option<ProviderType>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub message: String,
    pub tools: Option<Vec<String>>,
    pub workspace: Option<PathBuf>,
    /// Use an existing session (for continuity) or create new each time.
    pub session_id: Option<String>,
}

/// Where to deliver job output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryTarget {
    /// Write output to a file.
    File { path: PathBuf },
    /// POST output to a webhook URL.
    Webhook {
        url: String,
        headers: HashMap<String, String>,
        retry_policy: Option<RetryPolicy>,
    },
    /// Deliver output to a gateway channel destination.
    Channel {
        adapter_id: String,
        channel_id: String,
        user_id: Option<String>,
        thread_id: Option<String>,
        retry_policy: Option<RetryPolicy>,
    },
    /// Send as a system event on the event stream.
    EventStream,
    /// Callback (for programmatic use; not serialized as a concrete target).
    Callback,
}

/// Retry policy for outbound scheduling delivery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_attempts: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff_ms: 250,
            max_backoff_ms: 5_000,
            jitter: true,
        }
    }
}
