use anyhow::Result;
#[cfg(feature = "prometheus")]
use std::sync::Arc;

#[cfg(feature = "prometheus")]
use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry};

/// No-op counter used when the `prometheus` feature is disabled.
#[cfg(not(feature = "prometheus"))]
#[derive(Debug, Default, Clone)]
pub struct NoopCounter;

#[cfg(not(feature = "prometheus"))]
impl NoopCounter {
    pub fn inc(&self) {}
    pub fn inc_by(&self, _count: u64) {}
}

/// No-op histogram used when the `prometheus` feature is disabled.
#[cfg(not(feature = "prometheus"))]
#[derive(Debug, Default, Clone)]
pub struct NoopHistogram;

#[cfg(not(feature = "prometheus"))]
impl NoopHistogram {
    pub fn observe(&self, _value: f64) {}
}

/// No-op gauge used when the `prometheus` feature is disabled.
#[cfg(not(feature = "prometheus"))]
#[derive(Debug, Default, Clone)]
pub struct NoopGauge;

#[cfg(not(feature = "prometheus"))]
impl NoopGauge {
    pub fn set(&self, _value: i64) {}
}

pub struct MetricsRecorder {
    #[cfg(feature = "prometheus")]
    registry: Arc<Registry>,
    #[cfg(feature = "prometheus")]
    pub request_total: IntCounter,
    #[cfg(feature = "prometheus")]
    pub request_duration: Histogram,
    #[cfg(feature = "prometheus")]
    pub token_usage: IntCounter,
    #[cfg(feature = "prometheus")]
    pub cache_hits: IntCounter,
    #[cfg(feature = "prometheus")]
    pub cache_misses: IntCounter,
    #[cfg(feature = "prometheus")]
    pub active_workflows: IntGauge,
    #[cfg(feature = "prometheus")]
    pub agent_pool_size: IntGauge,
    #[cfg(not(feature = "prometheus"))]
    pub request_total: NoopCounter,
    #[cfg(not(feature = "prometheus"))]
    pub request_duration: NoopHistogram,
    #[cfg(not(feature = "prometheus"))]
    pub token_usage: NoopCounter,
    #[cfg(not(feature = "prometheus"))]
    pub cache_hits: NoopCounter,
    #[cfg(not(feature = "prometheus"))]
    pub cache_misses: NoopCounter,
    #[cfg(not(feature = "prometheus"))]
    pub active_workflows: NoopGauge,
    #[cfg(not(feature = "prometheus"))]
    pub agent_pool_size: NoopGauge,
}

impl MetricsRecorder {
    pub fn new() -> Result<Self> {
        #[cfg(feature = "prometheus")]
        {
            let registry = Arc::new(Registry::new());

            let request_total = IntCounter::with_opts(Opts::new(
                "agent_requests_total",
                "Total number of agent requests",
            ))?;

            let request_duration = Histogram::with_opts(HistogramOpts::new(
                "agent_request_duration_seconds",
                "Request duration in seconds",
            ))?;

            let token_usage =
                IntCounter::with_opts(Opts::new("agent_tokens_total", "Total tokens used"))?;

            let cache_hits =
                IntCounter::with_opts(Opts::new("cache_hits_total", "Total cache hits"))?;

            let cache_misses =
                IntCounter::with_opts(Opts::new("cache_misses_total", "Total cache misses"))?;

            let active_workflows =
                IntGauge::with_opts(Opts::new("active_workflows", "Number of active workflows"))?;

            let agent_pool_size =
                IntGauge::with_opts(Opts::new("agent_pool_size", "Number of agents in pool"))?;

            registry.register(Box::new(request_total.clone()))?;
            registry.register(Box::new(request_duration.clone()))?;
            registry.register(Box::new(token_usage.clone()))?;
            registry.register(Box::new(cache_hits.clone()))?;
            registry.register(Box::new(cache_misses.clone()))?;
            registry.register(Box::new(active_workflows.clone()))?;
            registry.register(Box::new(agent_pool_size.clone()))?;

            return Ok(Self {
                registry,
                request_total,
                request_duration,
                token_usage,
                cache_hits,
                cache_misses,
                active_workflows,
                agent_pool_size,
            });
        }

        #[cfg(not(feature = "prometheus"))]
        {
            Ok(Self {
                request_total: NoopCounter,
                request_duration: NoopHistogram,
                token_usage: NoopCounter,
                cache_hits: NoopCounter,
                cache_misses: NoopCounter,
                active_workflows: NoopGauge,
                agent_pool_size: NoopGauge,
            })
        }
    }

    #[cfg(feature = "prometheus")]
    pub fn registry(&self) -> Arc<Registry> {
        Arc::clone(&self.registry)
    }
}

pub fn register_metrics() -> Result<MetricsRecorder> {
    MetricsRecorder::new()
}
