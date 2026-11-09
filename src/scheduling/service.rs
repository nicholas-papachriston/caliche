//! Main scheduling service: timers for all jobs and execution dispatch.

use crate::scheduling::job::{CronJob, JobSchedule};
use crate::scheduling::runner::{JobExecutor, JobRunResult, JobRunner};
use crate::scheduling::store::JobStore;
use anyhow::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::watch;
use tokio::time::{Duration, Instant, sleep_until};

/// Run log interface: record and query job runs.
#[async_trait]
pub trait RunLog: Send + Sync {
    async fn record(&self, result: &JobRunResult) -> Result<()>;
    async fn query(&self, job_id: &str, limit: usize) -> Result<Vec<JobRunResult>>;
    async fn last_run(&self, job_id: &str) -> Result<Option<JobRunResult>>;
}

/// Main scheduling service. Manages timers for all jobs and dispatches execution.
pub struct CronService {
    store: Arc<dyn JobStore>,
    run_log: Arc<dyn RunLog>,
    executor: Arc<dyn JobExecutor>,
    shutdown_tx: watch::Sender<()>,
    shutdown_rx: watch::Receiver<()>,
    running: Arc<DashMap<String, tokio::task::JoinHandle<()>>>,
}

impl CronService {
    pub fn new(
        store: Arc<dyn JobStore>,
        run_log: Arc<dyn RunLog>,
        executor: Arc<dyn JobExecutor>,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(());
        Self {
            store,
            run_log,
            executor,
            shutdown_tx,
            shutdown_rx,
            running: Arc::new(DashMap::new()),
        }
    }

    /// Start the service (spawns timer task for all enabled jobs).
    #[expect(
        clippy::unused_async,
        reason = "public API consumed as async by caliche_cli; changing the signature is a cross-repo break out of scope here"
    )]
    pub async fn start(&self) -> Result<()> {
        let store = Arc::clone(&self.store);
        let run_log = Arc::clone(&self.run_log);
        let executor = Arc::clone(&self.executor);
        let mut shutdown_rx = self.shutdown_rx.clone();
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            'outer: loop {
                let jobs = loop {
                    match store.list_jobs().await {
                        Ok(j) => break j,
                        Err(_) => {
                            let tick = Instant::now() + Duration::from_secs(60);
                            tokio::select! {
                                _ = shutdown_rx.changed() => break 'outer,
                                _ = sleep_until(tick) => {}
                            }
                        }
                    }
                };

                let now = Utc::now();
                let mut next_run: Option<(String, DateTime<Utc>)> = None;
                for job in &jobs {
                    if !job.enabled {
                        continue;
                    }
                    let last = run_log.last_run(&job.id).await.ok().flatten();
                    let last_at = last.as_ref().map(|r| r.started_at);
                    if let Some(at) = next_run_for_job(job, now, last_at)
                        && next_run.as_ref().map(|(_, t)| *t > at).unwrap_or(true)
                    {
                        next_run = Some((job.id.clone(), at));
                    }
                }

                let (sleep_until_instant, job_id) = match next_run {
                    Some((id, dt)) => {
                        let secs = (dt - now).num_seconds().max(0) as u64;
                        (Instant::now() + Duration::from_secs(secs), Some(id))
                    }
                    None => (Instant::now() + Duration::from_secs(60), None),
                };

                tokio::select! {
                    _ = shutdown_rx.changed() => break,
                    _ = sleep_until(sleep_until_instant) => {
                        if let Some(id) = job_id
                            && let Ok(Some(job)) = store.get_job(&id).await {
                            let store = Arc::clone(&store);
                            let run_log = Arc::clone(&run_log);
                            let executor = Arc::clone(&executor);
                            let handle = tokio::spawn(async move {
                                if let Ok(result) = JobRunner::run(&job, executor.as_ref()).await {
                                    let _ = run_log.record(&result).await;
                                }
                                if job.one_shot {
                                    let _ = store.delete_job(&job.id).await;
                                }
                            });
                            running.insert(id, handle);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Stop all running jobs and the timer loop.
    pub fn stop(&self) {
        let _ = self.shutdown_tx.send(());
        self.running.clear();
    }

    /// Add a new job (persist and it will be picked up on next timer tick).
    pub async fn add_job(&self, job: CronJob) -> Result<()> {
        self.store.save_job(&job).await
    }

    /// Remove a job (cancel any running execution and delete from store).
    pub async fn remove_job(&self, id: &str) -> Result<()> {
        if let Some((_, handle)) = self.running.remove(id) {
            handle.abort();
        }
        self.store.delete_job(id).await
    }

    /// Get execution history for a job.
    pub async fn get_run_log(&self, job_id: &str, limit: usize) -> Result<Vec<JobRunResult>> {
        self.run_log.query(job_id, limit).await
    }

    /// Run a job once immediately (for manual trigger / catchup).
    pub async fn run_now(&self, job_id: &str) -> Result<Option<JobRunResult>> {
        let job = self
            .store
            .get_job(job_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("job not found"))?;
        let result = JobRunner::run(&job, self.executor.as_ref()).await?;
        self.run_log.record(&result).await?;
        Ok(Some(result))
    }
}

fn next_run_for_job(
    job: &CronJob,
    after: DateTime<Utc>,
    last_run: Option<DateTime<Utc>>,
) -> Option<DateTime<Utc>> {
    match &job.schedule {
        JobSchedule::Cron(expr) => {
            #[cfg(feature = "scheduling")]
            {
                use cron::Schedule;
                use std::str::FromStr;
                let schedule = Schedule::from_str(expr).ok()?;
                schedule.after(&after).next().map(|t| t.with_timezone(&Utc))
            }
            #[cfg(not(feature = "scheduling"))]
            {
                let _ = (job, after, last_run);
                None
            }
        }
        JobSchedule::Interval { seconds } => {
            let next = last_run
                .map(|t| t + chrono::Duration::seconds(*seconds as i64))
                .unwrap_or(after);
            if next > after {
                Some(next)
            } else {
                Some(after + chrono::Duration::seconds(*seconds as i64))
            }
        }
    }
}
