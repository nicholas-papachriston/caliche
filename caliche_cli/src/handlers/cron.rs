//! Cron / scheduled job command handlers.

use crate::commands::CronAction;
use crate::errors::{CliError, CliResult};
use crate::traits::command::{CommandContext, CommandHandler, CommandResult};
use anyhow::Result;
use async_trait::async_trait;
use caliche::scheduling::{
    CronJob, CronService, FileJobStore, FileRunLog, JobAgentConfig, JobExecutor, JobRunner,
    JobSchedule, JobStore, RunLog,
};
use chrono::Utc;
use std::path::PathBuf;
use std::sync::Arc;

fn cron_jobs_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("caliche")
        .join("cron")
}

fn cron_logs_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("caliche")
        .join("cron_logs")
}

/// CLI implementation of JobExecutor that respects JobAgentConfig (provider, model, system_prompt, tools, workspace).
struct CliExecutor;

#[async_trait]
impl JobExecutor for CliExecutor {
    async fn execute(
        &self,
        config: &JobAgentConfig,
    ) -> Result<(String, caliche::messages::TokenUsage)> {
        if let Some(ref w) = config.workspace {
            std::env::set_current_dir(w).map_err(|e| {
                anyhow::anyhow!("failed to set workspace directory {}: {}", w.display(), e)
            })?;
        }
        let mut svc = caliche::AgentService::from_config(
            caliche::AgentServiceConfig::new()
                .with_provider(config.provider, config.model.clone())
                .with_tool_names_or_defaults(config.tools.clone()),
        )
        .await?;
        if let Some(ref prompt) = config.system_prompt {
            svc.agent_mut().set_system_prompt(prompt.clone());
        }
        let response = svc.chat(&config.message).await?;
        Ok((response.text().to_string(), response.usage().clone()))
    }
}

/// Handler for cron subcommands.
pub struct CronHandler;

#[async_trait]
impl CommandHandler for CronHandler {
    type Args = CronAction;

    async fn execute(
        &self,
        args: Self::Args,
        _ctx: &mut CommandContext<'_>,
    ) -> CliResult<CommandResult> {
        handle_cron_command(args)
            .await
            .map_err(|e| CliError::Command(e.to_string()))?;
        Ok(CommandResult::Success(None))
    }

    fn name(&self) -> &'static str {
        "cron"
    }

    fn description(&self) -> &'static str {
        "Scheduled agent tasks (list, add, remove, run, start)"
    }

    fn requires_credentials(&self) -> bool {
        false
    }
}

pub async fn handle_cron_command(action: CronAction) -> Result<()> {
    let store_dir = cron_jobs_dir();
    let log_dir = cron_logs_dir();
    let store = std::sync::Arc::new(FileJobStore::new(store_dir));
    let run_log = std::sync::Arc::new(FileRunLog::new(log_dir));

    match action {
        CronAction::List => cron_list(store.as_ref()).await,
        CronAction::Add {
            name,
            schedule,
            message,
            model,
            delivery_file,
        } => {
            cron_add(
                store.as_ref(),
                name,
                schedule,
                message,
                model,
                delivery_file,
            )
            .await
        }
        CronAction::Remove { job_id } => cron_remove(store.as_ref(), job_id).await,
        CronAction::Status { job_id } => {
            cron_status(store.as_ref(), run_log.as_ref(), job_id).await
        }
        CronAction::Logs { job_id, limit } => cron_logs(run_log.as_ref(), job_id, limit).await,
        CronAction::Run { job_id } => cron_run(store.as_ref(), run_log.as_ref(), job_id).await,
        CronAction::Start => cron_start(store, run_log).await,
    }
}

async fn cron_list(store: &FileJobStore) -> Result<()> {
    let jobs = store.list_jobs().await?;
    if jobs.is_empty() {
        println!("No scheduled jobs.");
        return Ok(());
    }
    println!("{:<40} {:<20} {}", "ID", "Schedule", "Name");
    println!("{}", "-".repeat(80));
    for j in jobs {
        let sched = match &j.schedule {
            JobSchedule::Cron(s) => s.clone(),
            JobSchedule::Interval { seconds } => format!("every {seconds}s"),
        };
        println!("{:<40} {:<20} {}", j.id, sched, j.name);
    }
    Ok(())
}

fn parse_job_schedule(schedule: String) -> JobSchedule {
    if schedule.ends_with('m') {
        let secs = schedule
            .trim_end_matches('m')
            .parse::<u64>()
            .unwrap_or(30)
            .saturating_mul(60);
        JobSchedule::Interval { seconds: secs }
    } else {
        JobSchedule::Cron(schedule)
    }
}

async fn cron_add(
    store: &FileJobStore,
    name: String,
    schedule: String,
    message: String,
    model: Option<String>,
    delivery_file: Option<String>,
) -> Result<()> {
    let id = uuid::Uuid::now_v7().to_string();
    let mut delivery = Vec::new();
    if let Some(path) = delivery_file {
        delivery.push(caliche::scheduling::DeliveryTarget::File {
            path: PathBuf::from(path),
        });
    }
    let job = CronJob {
        id: id.clone(),
        name,
        schedule: parse_job_schedule(schedule),
        agent_config: JobAgentConfig {
            provider: None,
            model,
            system_prompt: None,
            message,
            tools: None,
            workspace: None,
            session_id: None,
        },
        delivery,
        enabled: true,
        created_at: Utc::now(),
        updated_at: Utc::now(),
        one_shot: false,
        timeout_seconds: 300,
    };
    store.save_job(&job).await?;
    println!("Created job {id}");
    Ok(())
}

async fn cron_remove(store: &FileJobStore, job_id: String) -> Result<()> {
    store.delete_job(&job_id).await?;
    println!("Removed job {job_id}");
    Ok(())
}

async fn cron_status(store: &FileJobStore, run_log: &FileRunLog, job_id: String) -> Result<()> {
    let job = store
        .get_job(&job_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("job not found"))?;
    println!("Job: {} ({})", job.name, job.id);
    println!("  Schedule: {:?}", job.schedule);
    println!("  Enabled: {}", job.enabled);
    let last = run_log.last_run(&job_id).await?;
    if let Some(r) = last {
        println!("  Last run: {} - {:?}", r.started_at, r.status);
    } else {
        println!("  Last run: never");
    }
    Ok(())
}

async fn cron_logs(run_log: &FileRunLog, job_id: String, limit: usize) -> Result<()> {
    let runs = run_log.query(&job_id, limit).await?;
    if runs.is_empty() {
        println!("No runs for job {job_id}");
        return Ok(());
    }
    for r in runs {
        println!("{} {} {:?}", r.run_id, r.started_at, r.status);
        print_run_output_preview(&r.output);
    }
    Ok(())
}

fn print_run_output_preview(output: &str) {
    if output.is_empty() {
        return;
    }
    if output.len() < 200 {
        println!("  {output}");
        return;
    }
    println!(
        "  {}...",
        caliche_contracts::text::truncate_utf8(&output, 200)
    );
}

async fn cron_run(store: &FileJobStore, run_log: &FileRunLog, job_id: String) -> Result<()> {
    let job = store
        .get_job(&job_id)
        .await?
        .ok_or_else(|| anyhow::anyhow!("job not found"))?;
    let executor = Arc::new(CliExecutor);
    let result = JobRunner::run(&job, executor.as_ref()).await?;
    run_log.record(&result).await?;
    println!("Run {}: {:?}", result.run_id, result.status);
    if !result.output.is_empty() {
        println!("{}", result.output);
    }
    Ok(())
}

async fn cron_start(store: Arc<FileJobStore>, run_log: Arc<FileRunLog>) -> Result<()> {
    let executor = Arc::new(CliExecutor);
    let service = CronService::new(store, run_log, executor);
    service.start().await?;
    println!("Cron daemon started. Press Ctrl+C to stop.");
    tokio::signal::ctrl_c().await?;
    service.stop();
    Ok(())
}
