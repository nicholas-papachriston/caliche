//! Executes a CronJob in an isolated agent session.

use crate::messages::TokenUsage;
use crate::scheduling::delivery::{DeliveryResult, deliver};
use crate::scheduling::job::{CronJob, JobAgentConfig};
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// Result of a single job run.
#[derive(Debug, Clone)]
pub struct JobRunResult {
    pub job_id: String,
    pub run_id: String,
    pub started_at: DateTime<Utc>,
    pub ended_at: DateTime<Utc>,
    pub status: JobRunStatus,
    pub output: String,
    pub usage: TokenUsage,
    pub delivery_results: Vec<DeliveryResult>,
}

/// Status of a job run.
#[derive(Debug, Clone)]
pub enum JobRunStatus {
    Success,
    Failed(String),
    TimedOut,
    Aborted,
}

/// Executes agent config (build agent, run message, return output and usage).
/// Implemented by the service layer so it can use AgentBuilder/AgentService.
#[async_trait]
pub trait JobExecutor: Send + Sync {
    async fn execute(&self, config: &JobAgentConfig) -> Result<(String, TokenUsage)>;
}

/// Runs a CronJob: execute via executor, then deliver output.
pub struct JobRunner;

impl JobRunner {
    /// Run the job once: execute agent config, deliver output, return result.
    pub async fn run(job: &CronJob, executor: &dyn JobExecutor) -> Result<JobRunResult> {
        let run_id = Uuid::now_v7().to_string();
        let started_at = Utc::now();

        let (output, usage, status) = match executor.execute(&job.agent_config).await {
            Ok((output, usage)) => (output, usage, JobRunStatus::Success),
            Err(e) => (
                String::new(),
                TokenUsage::default(),
                JobRunStatus::Failed(e.to_string()),
            ),
        };

        let delivery_results = deliver(&output, &job.delivery).await;
        let ended_at = Utc::now();

        Ok(JobRunResult {
            job_id: job.id.clone(),
            run_id,
            started_at,
            ended_at,
            status,
            output,
            usage,
            delivery_results,
        })
    }
}
