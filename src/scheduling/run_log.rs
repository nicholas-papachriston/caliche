//! Persistent run log for job execution history.

use crate::scheduling::runner::JobRunResult;
use crate::scheduling::service::RunLog;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::fs;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// In-memory run log (for testing or when no persistence is configured).
#[derive(Debug, Default)]
pub struct MemoryRunLog {
    runs: dashmap::DashMap<String, Vec<JobRunResult>>,
}

impl MemoryRunLog {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl RunLog for MemoryRunLog {
    async fn record(&self, result: &JobRunResult) -> Result<()> {
        self.runs
            .entry(result.job_id.clone())
            .or_default()
            .push(result.clone());
        Ok(())
    }

    async fn query(&self, job_id: &str, limit: usize) -> Result<Vec<JobRunResult>> {
        let mut v = self.runs.get(job_id).map(|r| r.clone()).unwrap_or_default();
        v.sort_by(|a, b| b.started_at.cmp(&a.started_at));
        v.truncate(limit);
        Ok(v)
    }

    async fn last_run(&self, job_id: &str) -> Result<Option<JobRunResult>> {
        let v = self.query(job_id, 1).await?;
        Ok(v.into_iter().next())
    }
}

/// File-based run log: one JSONL file per job.
#[derive(Debug)]
pub struct FileRunLog {
    dir: PathBuf,
}

impl FileRunLog {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn path(&self, job_id: &str) -> PathBuf {
        let safe = job_id.replace(|c: char| !c.is_alphanumeric(), "_");
        self.dir.join(format!("{}.jsonl", safe))
    }
}

#[async_trait]
impl RunLog for FileRunLog {
    async fn record(&self, result: &JobRunResult) -> Result<()> {
        fs::create_dir_all(&self.dir).await?;
        let path = self.path(&result.job_id);
        let line = serde_json::to_string(&SerializableRunResult::from(result))?;
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await?;
        f.write_all(line.as_bytes()).await?;
        f.write_all(b"\n").await?;
        f.flush().await?;
        Ok(())
    }

    async fn query(&self, job_id: &str, limit: usize) -> Result<Vec<JobRunResult>> {
        let path = self.path(job_id);
        if !path.is_file() {
            return Ok(Vec::new());
        }
        let f = fs::File::open(&path).await?;
        let reader = BufReader::new(f);
        let mut lines = reader.lines();
        let mut results = Vec::new();
        while let Some(line) = lines.next_line().await? {
            if let Ok(rec) = serde_json::from_str::<SerializableRunResult>(&line) {
                results.push(rec.into());
            }
        }
        results.sort_by(|a: &JobRunResult, b: &JobRunResult| b.started_at.cmp(&a.started_at));
        results.truncate(limit);
        Ok(results)
    }

    async fn last_run(&self, job_id: &str) -> Result<Option<JobRunResult>> {
        let v = self.query(job_id, 1).await?;
        Ok(v.into_iter().next())
    }
}

#[derive(Serialize, Deserialize)]
struct SerializableRunResult {
    job_id: String,
    run_id: String,
    started_at: DateTime<Utc>,
    ended_at: DateTime<Utc>,
    status: String,
    status_detail: Option<String>,
    output: String,
    delivery_results: Vec<()>,
}

impl From<&JobRunResult> for SerializableRunResult {
    fn from(r: &JobRunResult) -> Self {
        let (status, status_detail) = match &r.status {
            crate::scheduling::runner::JobRunStatus::Success => ("success", None),
            crate::scheduling::runner::JobRunStatus::Failed(s) => ("failed", Some(s.clone())),
            crate::scheduling::runner::JobRunStatus::TimedOut => ("timed_out", None),
            crate::scheduling::runner::JobRunStatus::Aborted => ("aborted", None),
        };
        Self {
            job_id: r.job_id.clone(),
            run_id: r.run_id.clone(),
            started_at: r.started_at,
            ended_at: r.ended_at,
            status: status.to_string(),
            status_detail,
            output: r.output.clone(),
            delivery_results: Vec::new(),
        }
    }
}

impl From<SerializableRunResult> for JobRunResult {
    fn from(s: SerializableRunResult) -> Self {
        use crate::scheduling::runner::JobRunStatus;
        let status = match s.status.as_str() {
            "success" => JobRunStatus::Success,
            "failed" => JobRunStatus::Failed(s.status_detail.unwrap_or_default()),
            "timed_out" => JobRunStatus::TimedOut,
            "aborted" => JobRunStatus::Aborted,
            _ => JobRunStatus::Failed(s.status_detail.unwrap_or(s.status)),
        };
        JobRunResult {
            job_id: s.job_id,
            run_id: s.run_id,
            started_at: s.started_at,
            ended_at: s.ended_at,
            status,
            output: s.output,
            usage: crate::messages::TokenUsage::default(),
            delivery_results: Vec::new(),
        }
    }
}
