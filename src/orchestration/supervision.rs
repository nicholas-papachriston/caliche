//! OTP-style per-agent task isolation and supervision.
//!
//! Wraps agent execution in isolated tokio tasks with panic detection,
//! automatic restart, configurable strategies, and backoff.

use crate::config::{BackoffType, SupervisionConfig, SupervisionStrategy};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock, mpsc};
use tokio::task::{AbortHandle, JoinHandle};

/// Factory that produces a fresh agent task. Used for restarts.
pub trait AgentFactory: Send + Sync {
    /// Spawn a new agent task and return its JoinHandle.
    fn spawn(&self) -> JoinHandle<Result<()>>;
}

/// A supervised agent: wraps an agent task with a restart factory and restart history.
pub struct SupervisedAgent {
    pub id: String,
    /// Flipped to `true` by the detached watcher task once the currently-running handle
    /// completes (success, error, panic, or abort). Shared independently of the `agents`
    /// lock so `is_finished()` doesn't need to be async.
    finished: Arc<AtomicBool>,
    /// Lets `restart_agent` cancel a still-running task (e.g. a healthy sibling being
    /// restarted under `OneForAll`/`RestForOne`) instead of leaking it as an orphaned,
    /// unawaited task.
    abort_handle: Option<AbortHandle>,
    /// Bumped every time this slot gets a new handle (initial spawn or restart). Lets
    /// `monitor_loop` recognize and discard a stale completion event from a handle that was
    /// aborted or superseded by a subsequent restart, so it doesn't double-restart or
    /// mis-attribute an old task's outcome to the new one.
    generation: u64,
    pub restarts: VecDeque<Instant>,
    factory: Arc<dyn AgentFactory>,
}

impl std::fmt::Debug for SupervisedAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SupervisedAgent")
            .field("id", &self.id)
            .field("finished", &self.finished.load(Ordering::Relaxed))
            .field("generation", &self.generation)
            .field("restarts", &self.restarts)
            .finish_non_exhaustive()
    }
}

impl SupervisedAgent {
    /// Check if the currently-tracked task has finished.
    pub fn is_finished(&self) -> bool {
        self.finished.load(Ordering::Acquire)
    }

    /// Record a restart and trim the window to `restart_window_secs`.
    fn record_restart(&mut self, config: &SupervisionConfig) {
        let now = Instant::now();
        let cutoff = now
            .checked_sub(Duration::from_secs(config.restart_window_secs))
            .unwrap_or_else(|| self.restarts.front().copied().unwrap_or(now));
        self.restarts.push_back(now);
        while self.restarts.front().is_some_and(|t| *t < cutoff) {
            let _ = self.restarts.pop_front();
        }
    }

    /// Count restarts within the current window.
    fn restart_count_in_window(&self, config: &SupervisionConfig) -> u32 {
        let now = Instant::now();
        let cutoff = now
            .checked_sub(Duration::from_secs(config.restart_window_secs))
            .unwrap_or_else(|| self.restarts.front().copied().unwrap_or(now));
        self.restarts.iter().filter(|t| **t >= cutoff).count() as u32
    }
}

/// Restart counters for one supervised agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentRestartStat {
    /// Supervised agent identifier.
    pub agent_id: String,
    /// Restarts recorded since registration.
    pub restarts_total: u32,
    /// Restarts inside the current `restart_window_secs` window.
    pub restarts_in_window: u32,
    /// Whether the currently tracked task has already finished.
    pub finished: bool,
    /// Whether the agent used up its restart budget for the current window.
    pub restart_budget_exhausted: bool,
}

/// Aggregate supervision metrics, suitable for an operator view or a metrics exporter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupervisorMetrics {
    /// Supervised agents currently registered.
    pub supervised_agents: usize,
    /// Restarts recorded across all agents.
    pub restarts_total: u32,
    /// Restarts across all agents inside the current window.
    pub restarts_in_window: u32,
    /// Agents that used up their restart budget.
    pub agents_over_budget: usize,
    /// Per-agent detail.
    pub agents: Vec<AgentRestartStat>,
}

/// Completions waiting for [`Supervisor::monitor_loop`].
///
/// 64 covers a fleet that exits together while the loop is inside restart
/// backoff. [`mpsc::Sender::send`] waits when the queue is full, so a finished
/// agent is still reported and a restart is not skipped. `Err` from `send`
/// means the supervisor dropped the receiver.
const FINISHED_EVENT_CAPACITY: usize = 64;

/// Wait until `event` is queued.
///
/// A full queue waits here, so the watcher does not drop a completion.
/// `Err` from `send` means the supervisor dropped the receiver.
async fn enqueue_finished(tx: &mpsc::Sender<FinishedEvent>, event: FinishedEvent) {
    let _ = tx.send(event).await;
}

/// Sent by a per-task watcher when the handle it is following completes.
struct FinishedEvent {
    index: usize,
    generation: u64,
    agent_id: String,
    outcome: std::result::Result<Result<()>, tokio::task::JoinError>,
}

/// Supervisor: monitors supervised agents and restarts them on failure.
pub struct Supervisor {
    agents: Arc<RwLock<Vec<SupervisedAgent>>>,
    config: SupervisionConfig,
    finished_tx: mpsc::Sender<FinishedEvent>,
    /// Taken by `monitor_loop` on its (expected to be sole) invocation.
    finished_rx: Mutex<Option<mpsc::Receiver<FinishedEvent>>>,
}

impl Supervisor {
    /// Create a new supervisor with the given config.
    pub fn new(config: SupervisionConfig) -> Self {
        let (finished_tx, finished_rx) = mpsc::channel(FINISHED_EVENT_CAPACITY);
        Self {
            agents: Arc::new(RwLock::new(Vec::new())),
            config,
            finished_tx,
            finished_rx: Mutex::new(Some(finished_rx)),
        }
    }

    /// Spawn a detached task that waits for `handle` to complete and reports the outcome back
    /// to `monitor_loop` via the finished-event channel. This is what makes monitoring
    /// event-driven: `monitor_loop` never polls, it just awaits the channel.
    fn watch(
        &self,
        index: usize,
        generation: u64,
        agent_id: String,
        handle: JoinHandle<Result<()>>,
        finished: Arc<AtomicBool>,
    ) {
        let tx = self.finished_tx.clone();
        tokio::spawn(async move {
            let outcome = handle.await;
            finished.store(true, Ordering::Release);
            enqueue_finished(
                &tx,
                FinishedEvent {
                    index,
                    generation,
                    agent_id,
                    outcome,
                },
            )
            .await;
        });
    }

    /// Register an agent with the supervisor. The agent is spawned via the factory.
    pub async fn register(
        &self,
        id: impl Into<String>,
        factory: Arc<dyn AgentFactory>,
    ) -> Result<()> {
        let id = id.into();
        let handle = factory.spawn();
        let finished = Arc::new(AtomicBool::new(false));
        let generation = 0;
        let abort_handle = handle.abort_handle();

        let index = {
            let mut agents = self.agents.write().await;
            let index = agents.len();
            agents.push(SupervisedAgent {
                id: id.clone(),
                finished: Arc::clone(&finished),
                abort_handle: Some(abort_handle),
                generation,
                restarts: VecDeque::new(),
                factory,
            });
            index
        };

        self.watch(index, generation, id, handle, finished);
        Ok(())
    }

    /// Snapshot restart counters for every supervised agent.
    ///
    /// Read-only: takes a read lock, so it is safe to poll from an operator endpoint while
    /// the monitor loop keeps running.
    pub async fn metrics(&self) -> SupervisorMetrics {
        let agents = self.agents.read().await;
        let mut stats = Vec::with_capacity(agents.len());
        let mut restarts_total = 0u32;
        let mut restarts_in_window = 0u32;
        let mut agents_over_budget = 0usize;

        for agent in agents.iter() {
            let total = u32::try_from(agent.restarts.len()).unwrap_or(u32::MAX);
            let in_window = agent.restart_count_in_window(&self.config);
            let exhausted = in_window >= self.config.max_restarts;
            restarts_total = restarts_total.saturating_add(total);
            restarts_in_window = restarts_in_window.saturating_add(in_window);
            if exhausted {
                agents_over_budget += 1;
            }
            stats.push(AgentRestartStat {
                agent_id: agent.id.clone(),
                restarts_total: total,
                restarts_in_window: in_window,
                finished: agent.is_finished(),
                restart_budget_exhausted: exhausted,
            });
        }

        SupervisorMetrics {
            supervised_agents: stats.len(),
            restarts_total,
            restarts_in_window,
            agents_over_budget,
            agents: stats,
        }
    }

    /// Compute backoff duration for the given restart count.
    fn backoff_duration(&self, restart_count: u32) -> Duration {
        let base_ms = self.config.backoff_base_ms;
        let restart_count = restart_count as u64;
        let ms = match self.config.backoff {
            BackoffType::Immediate => 0,
            BackoffType::Linear => base_ms * restart_count.saturating_add(1),
            BackoffType::Exponential => base_ms * 2u64.saturating_pow(restart_count.min(20) as u32),
        };
        Duration::from_millis(ms)
    }

    /// Restart a single agent by index. Returns Ok(()) if restarted, Err if max_restarts exceeded.
    ///
    /// If the agent's current task is still running (e.g. a healthy sibling being restarted
    /// under `OneForAll`/`RestForOne`), it is aborted first so it isn't leaked as an orphaned,
    /// unawaited task running alongside its replacement.
    async fn restart_agent(&self, index: usize) -> Result<()> {
        let (agent_id, backoff, factory): (String, Duration, Arc<dyn AgentFactory>) = {
            let mut agents = self.agents.write().await;
            let agent = agents
                .get_mut(index)
                .ok_or_else(|| anyhow::anyhow!("Agent index out of bounds"))?;

            if agent.restart_count_in_window(&self.config) >= self.config.max_restarts {
                tracing::error!(
                    agent_id = %agent.id,
                    "Max restarts exceeded in window; not restarting"
                );
                if let Some(abort_handle) = agent.abort_handle.take() {
                    abort_handle.abort();
                }
                agent.generation = agent.generation.wrapping_add(1);
                return Err(anyhow::anyhow!(
                    "Max restarts ({}) exceeded for agent {}",
                    self.config.max_restarts,
                    agent.id
                ));
            }

            let backoff = self.backoff_duration(agent.restarts.len() as u32);
            let agent_id = agent.id.clone();
            let factory = Arc::clone(&agent.factory);

            // Cancel the previous task (if it's still running) before starting its
            // replacement, and bump the generation so the old task's eventual completion
            // event — including the `Cancelled` `JoinError` this abort will produce — is
            // recognized as stale and ignored by `monitor_loop`.
            if let Some(abort_handle) = agent.abort_handle.take() {
                abort_handle.abort();
            }
            agent.generation = agent.generation.wrapping_add(1);

            (agent_id, backoff, factory)
        };

        if backoff > Duration::ZERO {
            tracing::info!(agent_id = %agent_id, ?backoff, "Applying backoff before restart");
            tokio::time::sleep(backoff).await;
        }

        {
            let mut agents = self.agents.write().await;
            let agent = agents
                .get_mut(index)
                .ok_or_else(|| anyhow::anyhow!("Agent index out of bounds"))?;
            let handle = factory.spawn();
            agent.record_restart(&self.config);
            let finished = Arc::new(AtomicBool::new(false));
            agent.finished = Arc::clone(&finished);
            agent.abort_handle = Some(handle.abort_handle());
            self.watch(index, agent.generation, agent_id.clone(), handle, finished);
        }

        tracing::info!(agent_id = %agent_id, "Agent restarted");
        Ok(())
    }

    /// Restart multiple agents by index (for OneForAll and RestForOne).
    async fn restart_agents(&self, indices: Vec<usize>) -> Result<()> {
        for idx in indices {
            if self.restart_agent(idx).await.is_err() {
                break;
            }
        }
        Ok(())
    }

    /// Run the monitor loop. Event-driven: awaits completion notifications from per-task
    /// watchers (see `watch`) instead of polling every agent on a fixed timer, so restarts
    /// happen as soon as a task actually exits rather than up to one tick late, and agents
    /// that are still healthy are never locked/scanned at all.
    ///
    /// Must be called at most once per `Supervisor` (typically once, right after
    /// construction); a second call returns immediately since the completion-event receiver
    /// can only be held by one consumer.
    pub async fn monitor_loop(self: Arc<Self>) {
        let mut rx = {
            let mut guard = self.finished_rx.lock().await;
            match guard.take() {
                Some(rx) => rx,
                None => {
                    tracing::error!(
                        "Supervisor::monitor_loop called more than once; ignoring second call"
                    );
                    return;
                }
            }
        };

        while let Some(event) = rx.recv().await {
            // Drain any other events that are already queued so a batch of correlated task
            // exits (e.g. several siblings failing around the same time) is resolved into one
            // restart decision per index instead of one per event.
            let mut events = vec![event];
            while let Ok(event) = rx.try_recv() {
                events.push(event);
            }

            let mut failed_indices: Vec<usize> = Vec::new();
            {
                let agents = self.agents.read().await;
                for event in events {
                    let current_generation = agents.get(event.index).map(|a| a.generation);
                    if current_generation != Some(event.generation) {
                        // Stale: this index was already restarted (or removed) since this
                        // handle was spawned — most commonly the `Cancelled` `JoinError` from
                        // `restart_agent` aborting a still-running sibling.
                        continue;
                    }

                    let is_failure = match &event.outcome {
                        Ok(Ok(())) => false,
                        Ok(Err(e)) => {
                            tracing::error!(agent_id = %event.agent_id, "Agent task failed: {}", e);
                            true
                        }
                        Err(e) => {
                            tracing::error!(
                                agent_id = %event.agent_id,
                                "Agent task panicked or was cancelled: {}",
                                e
                            );
                            true
                        }
                    };
                    if is_failure {
                        failed_indices.push(event.index);
                    }
                }
            }

            if failed_indices.is_empty() {
                continue;
            }

            let agent_count = self.agents.read().await.len();
            let indices_to_restart: Vec<usize> = match self.config.strategy {
                SupervisionStrategy::OneForOne => failed_indices,
                SupervisionStrategy::OneForAll => (0..agent_count).collect(),
                SupervisionStrategy::RestForOne => {
                    let min_failed = failed_indices.iter().copied().min().unwrap_or(0);
                    (min_failed..agent_count).collect()
                }
            };

            if let Err(e) = self.restart_agents(indices_to_restart).await {
                tracing::error!(
                    "Supervisor restart_agents failed — some agents may not have recovered: {e}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backoff_duration() {
        let config = SupervisionConfig {
            backoff: BackoffType::Immediate,
            backoff_base_ms: 500,
            ..SupervisionConfig::default()
        };
        let sup = Supervisor::new(config);
        assert_eq!(sup.backoff_duration(0), Duration::ZERO);
        assert_eq!(sup.backoff_duration(1), Duration::ZERO);

        let config = SupervisionConfig {
            backoff: BackoffType::Linear,
            backoff_base_ms: 100,
            ..SupervisionConfig::default()
        };
        let sup = Supervisor::new(config);
        assert_eq!(sup.backoff_duration(0), Duration::from_millis(100));
        assert_eq!(sup.backoff_duration(1), Duration::from_millis(200));

        let config = SupervisionConfig {
            backoff: BackoffType::Exponential,
            backoff_base_ms: 100,
            ..SupervisionConfig::default()
        };
        let sup = Supervisor::new(config);
        assert_eq!(sup.backoff_duration(0), Duration::from_millis(100));
        assert_eq!(sup.backoff_duration(1), Duration::from_millis(200));
        assert_eq!(sup.backoff_duration(2), Duration::from_millis(400));
    }

    fn finished_marker(index: usize) -> FinishedEvent {
        FinishedEvent {
            index,
            generation: 0,
            agent_id: index.to_string(),
            outcome: Ok(Ok(())),
        }
    }

    #[tokio::test]
    async fn finished_channel_waits_when_full_and_delivers_every_completion() {
        let sup = Supervisor::new(SupervisionConfig::default());
        let tx = sup.finished_tx.clone();
        for index in 0..FINISHED_EVENT_CAPACITY {
            tx.try_send(finished_marker(index)).expect("fill");
        }

        let waiting = tx.clone();
        let extra = FINISHED_EVENT_CAPACITY;
        let mut pending = std::pin::pin!(super::enqueue_finished(&waiting, finished_marker(extra)));
        let blocked = tokio::time::timeout(Duration::from_millis(40), &mut pending).await;
        assert!(
            blocked.is_err(),
            "a completion must wait while the monitor is not receiving"
        );

        let mut rx = sup
            .finished_rx
            .lock()
            .await
            .take()
            .expect("monitor receiver");
        let mut got = Vec::new();
        got.push(rx.recv().await.expect("first"));
        pending.await;
        while let Ok(event) = rx.try_recv() {
            got.push(event);
        }

        assert_eq!(got.len(), FINISHED_EVENT_CAPACITY + 1);
        let indices: Vec<usize> = got.into_iter().map(|event| event.index).collect();
        let expected: Vec<usize> = (0..=FINISHED_EVENT_CAPACITY).collect();
        assert_eq!(indices, expected);
    }
}
