//! Cron / scheduled agent tasks.
//!
//! Run agents on a schedule (cron expressions or intervals), persist jobs,
//! and deliver output to files, webhooks, or event streams.

mod delivery;
mod job;
mod run_log;
mod runner;
mod service;
mod store;

pub use delivery::{DeliveryResult, DeliveryTargetSummary, deliver};
pub use job::{CronJob, DeliveryTarget, JobAgentConfig, JobSchedule, RetryPolicy};
pub use run_log::{FileRunLog, MemoryRunLog};
pub use runner::{JobExecutor, JobRunResult, JobRunStatus, JobRunner};
pub use service::{CronService, RunLog};
pub use store::{FileJobStore, JobStore};
