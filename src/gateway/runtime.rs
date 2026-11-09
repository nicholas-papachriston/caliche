use crate::gateway::router::MessageDedupeStore;
use crate::gateway::types::GatewayIngressEnvelope;
use anyhow::{Result, anyhow};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use dashmap::DashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Clone)]
pub struct GatewayRuntimeConfig {
    pub per_key_rate_limit_per_minute: u32,
    pub global_max_in_flight: usize,
}

impl Default for GatewayRuntimeConfig {
    fn default() -> Self {
        Self {
            per_key_rate_limit_per_minute: 120,
            global_max_in_flight: 256,
        }
    }
}

#[derive(Debug, Default)]
pub struct GatewayRuntimeMetrics {
    pub accepted: AtomicU64,
    pub rejected_rate_limited: AtomicU64,
    pub rejected_backpressure: AtomicU64,
    pub duplicate_dropped: AtomicU64,
}

impl GatewayRuntimeMetrics {
    pub fn snapshot(&self) -> GatewayRuntimeMetricsSnapshot {
        GatewayRuntimeMetricsSnapshot {
            accepted: self.accepted.load(Ordering::Relaxed),
            rejected_rate_limited: self.rejected_rate_limited.load(Ordering::Relaxed),
            rejected_backpressure: self.rejected_backpressure.load(Ordering::Relaxed),
            duplicate_dropped: self.duplicate_dropped.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GatewayRuntimeMetricsSnapshot {
    pub accepted: u64,
    pub rejected_rate_limited: u64,
    pub rejected_backpressure: u64,
    pub duplicate_dropped: u64,
}

#[derive(Debug)]
pub struct GatewayRateLimiter {
    requests: DashMap<String, VecDeque<DateTime<Utc>>>,
    limit_per_minute: u32,
}

impl GatewayRateLimiter {
    pub fn new(limit_per_minute: u32) -> Self {
        Self {
            requests: DashMap::new(),
            limit_per_minute: limit_per_minute.max(1),
        }
    }

    pub fn allow(&self, key: &str, now: DateTime<Utc>) -> bool {
        let mut queue = self.requests.entry(key.to_string()).or_default();
        let threshold = now - ChronoDuration::minutes(1);
        while let Some(front) = queue.front() {
            if *front < threshold {
                queue.pop_front();
            } else {
                break;
            }
        }

        if (queue.len() as u32) >= self.limit_per_minute {
            return false;
        }
        queue.push_back(now);
        true
    }
}

/// Holds the global in-flight permit until inline gateway processing completes.
#[derive(Debug)]
pub struct GatewayAdmission {
    pub envelope: GatewayIngressEnvelope,
    _permit: OwnedSemaphorePermit,
}

/// Runtime guardrails for ingress reliability and throughput control.
pub struct GatewayRuntime {
    pub config: GatewayRuntimeConfig,
    pub limiter: GatewayRateLimiter,
    pub in_flight: Arc<Semaphore>,
    pub metrics: Arc<GatewayRuntimeMetrics>,
    pub dedupe: Option<Arc<dyn MessageDedupeStore>>,
}

impl GatewayRuntime {
    pub fn new(config: GatewayRuntimeConfig, dedupe: Option<Arc<dyn MessageDedupeStore>>) -> Self {
        let limiter = GatewayRateLimiter::new(config.per_key_rate_limit_per_minute);
        let in_flight = Arc::new(Semaphore::new(config.global_max_in_flight.max(1)));
        Self {
            config,
            limiter,
            in_flight,
            metrics: Arc::new(GatewayRuntimeMetrics::default()),
            dedupe,
        }
    }

    /// Admit an ingress envelope under rate-limit and in-flight guardrails.
    ///
    /// Callers must keep the returned [`GatewayAdmission`] alive for the duration of inline
    /// processing so the global in-flight permit is released when work completes.
    pub async fn submit(
        &self,
        fairness_key: &str,
        envelope: GatewayIngressEnvelope,
    ) -> Result<GatewayAdmission> {
        let permit = match Arc::clone(&self.in_flight).try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                self.metrics
                    .rejected_backpressure
                    .fetch_add(1, Ordering::Relaxed);
                return Err(anyhow!("global in-flight limit reached"));
            }
        };

        if !self.limiter.allow(fairness_key, Utc::now()) {
            self.metrics
                .rejected_rate_limited
                .fetch_add(1, Ordering::Relaxed);
            return Err(anyhow!("rate limited for key {}", fairness_key));
        }

        if let Some(dedupe) = &self.dedupe {
            let dedupe_key = envelope
                .idempotency_key
                .as_deref()
                .unwrap_or(envelope.event_id.as_str());
            let inserted = dedupe.mark_seen(dedupe_key).await?;
            if !inserted {
                self.metrics
                    .duplicate_dropped
                    .fetch_add(1, Ordering::Relaxed);
                return Err(anyhow!("duplicate event {}", dedupe_key));
            }
        }

        self.metrics.accepted.fetch_add(1, Ordering::Relaxed);
        Ok(GatewayAdmission {
            envelope,
            _permit: permit,
        })
    }
}
