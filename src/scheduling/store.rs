//! Job persistence (save, load, list, delete).

use crate::scheduling::job::CronJob;
use anyhow::Result;
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// Persistence for cron jobs.
#[async_trait]
pub trait JobStore: Send + Sync {
    async fn save_job(&self, job: &CronJob) -> Result<()>;
    async fn get_job(&self, id: &str) -> Result<Option<CronJob>>;
    async fn list_jobs(&self) -> Result<Vec<CronJob>>;
    async fn delete_job(&self, id: &str) -> Result<()>;
    async fn update_job(&self, job: &CronJob) -> Result<()>;
}

/// File-based job store: one JSON file per job in a directory.
#[derive(Debug)]
pub struct FileJobStore {
    dir: PathBuf,
}

impl FileJobStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn path(&self, id: &str) -> PathBuf {
        let safe = id.replace(|c: char| !c.is_alphanumeric(), "_");
        self.dir.join(format!("{}.json", safe))
    }
}

#[async_trait]
impl JobStore for FileJobStore {
    async fn save_job(&self, job: &CronJob) -> Result<()> {
        fs::create_dir_all(&self.dir).await?;
        let path = self.path(&job.id);
        let json = serde_json::to_string_pretty(job)?;
        let mut f = fs::File::create(&path).await?;
        f.write_all(json.as_bytes()).await?;
        f.flush().await?;
        Ok(())
    }

    async fn get_job(&self, id: &str) -> Result<Option<CronJob>> {
        let path = self.path(id);
        if !path.is_file() {
            return Ok(None);
        }
        let data = fs::read_to_string(&path).await?;
        let job: CronJob = serde_json::from_str(&data)?;
        Ok(Some(job))
    }

    async fn list_jobs(&self) -> Result<Vec<CronJob>> {
        if !self.dir.is_dir() {
            return Ok(Vec::new());
        }
        let mut jobs = Vec::new();
        let mut read_dir = fs::read_dir(&self.dir).await?;
        while let Some(entry) = read_dir.next_entry().await? {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false)
                && let Ok(data) = fs::read_to_string(&path).await
                && let Ok(job) = serde_json::from_str::<CronJob>(&data)
            {
                jobs.push(job);
            }
        }
        Ok(jobs)
    }

    async fn delete_job(&self, id: &str) -> Result<()> {
        let path = self.path(id);
        if path.is_file() {
            fs::remove_file(&path).await?;
        }
        Ok(())
    }

    async fn update_job(&self, job: &CronJob) -> Result<()> {
        self.save_job(job).await
    }
}
